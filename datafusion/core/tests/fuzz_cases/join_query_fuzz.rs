// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Fuzz test for equi-join queries executed under randomized session
//! configurations, including randomized memory limits.
//!
//! The same SQL query is executed several times with a different
//! configuration each time (hash join vs sort merge join, number of
//! partitions, batch size, bounded vs unbounded memory pool, ...) and
//! all results must be identical.
//!
//! Memory limits are randomly set to a fraction of the in-memory size of
//! the two input tables, so that the external (spilling) code paths of the
//! join operators and their input sorts are exercised.
//!
//! `HashJoinExec` can not spill today (see
//! <https://github.com/apache/datafusion/issues/17267>), so a hash join
//! that fails with a `ResourcesExhausted` error under a memory limit is
//! reported and skipped rather than treated as a failure. Any other error,
//! and any `ResourcesExhausted` error from a sort merge join, is a failure.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    ArrayRef, Date32Array, Decimal128Array, Int32Array, Int64Array, RecordBatch,
    StringArray, TimestampNanosecondArray, UInt64Array,
};
use arrow::compute::{SortColumn, cast, concat_batches, lexsort_to_indices, take};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::{DataFusionError, Result, human_readable_size, instant::Instant};
use datafusion_execution::disk_manager::DiskManagerBuilder;
use datafusion_execution::memory_pool::{FairSpillPool, MemoryPool, UnboundedMemoryPool};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_expr::display_schema;
use datafusion_physical_plan::spill::get_record_batch_memory_size;
use datafusion_physical_plan::{
    ExecutionPlan, collect, display::DisplayableExecutionPlan,
};

use rand::Rng;
use rand::prelude::IndexedRandom;
use rand::{SeedableRng, rngs::StdRng};

use crate::fuzz_cases::record_batch_generator::{
    ColumnDescr, RecordBatchGenerator, get_supported_types_columns,
};
use crate::helper::plan_metrics::{plan_spill_count, plan_spilled_bytes};

const LEFT_TABLE: &str = "join_fuzz_left";
const RIGHT_TABLE: &str = "join_fuzz_right";

/// The memory limit is never set below this many batches per spillable
/// consumer, see `JoinFuzzerTestGenerator::generate_random_config`
const MIN_BATCHES_PER_CONSUMER: usize = 2;

/// The table sizes are chosen so that an inner join is expected to produce at
/// most this many rows, see `JoinFuzzerTestGenerator::init_datasets`
const MAX_EXPECTED_JOIN_OUTPUT_ROWS: usize = 100_000;

/// Entry point for executing the join query fuzzer.
///
/// Memory limiting is disabled in this runner for now, see
/// [`join_query_fuzzer_memory_limit_runner`].
#[tokio::test(flavor = "multi_thread")]
async fn join_query_fuzzer_runner() {
    run_join_query_fuzzer(false).await;
}

/// Same as [`join_query_fuzzer_runner`] but with randomized memory limits.
///
/// Ignored for now because batches read back from spill files are accounted
/// at the size of the spill reader's 128 KB read chunk instead of their own
/// size (each decoded batch is a zero-copy slice of the chunk). Whenever a
/// `RepartitionExec` spills under memory pressure, the `SortExec` consuming it
/// then fails to reserve memory for a single small batch even though the
/// memory limit is many times larger than the data.
///
/// TODO: remove `#[ignore]` once the spill read-back accounting is fixed
#[tokio::test(flavor = "multi_thread")]
#[ignore = "batches read back from spill files are accounted at the spill reader's chunk size"]
async fn join_query_fuzzer_memory_limit_runner() {
    run_join_query_fuzzer(true).await;
}

async fn run_join_query_fuzzer(set_memory_limit: bool) {
    let random_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // The runner seed determines the table schema and all the other seeds
    println!("[JoinQueryFuzzer] runner_seed = {random_seed}");

    let test_generator = JoinFuzzerTestGenerator::new(
        20_000,
        4,
        get_supported_types_columns(random_seed),
        set_memory_limit,
        random_seed,
    );
    let mut fuzzer = JoinQueryFuzzer::new(random_seed)
        .with_max_rounds(Some(4))
        .with_queries_per_round(4)
        .with_config_variations_per_query(5)
        .with_time_limit(Duration::from_secs(20))
        .with_test_generator(test_generator);

    fuzzer.run().await.unwrap();
}

/// `JoinQueryFuzzer` holds the runner configuration for executing join query
/// fuzz tests. The fuzzing details are managed inside `JoinFuzzerTestGenerator`.
///
/// It defines:
/// - `max_rounds`: Maximum number of rounds to run (or None to run until `time_limit`).
/// - `queries_per_round`: Number of different queries to run in each round.
/// - `config_variations_per_query`: Number of different configurations to test per query.
/// - `time_limit`: Time limit for the entire fuzzer execution.
pub struct JoinQueryFuzzer {
    test_gen: JoinFuzzerTestGenerator,
    /// Random number generator for the runner, used to generate seeds for inner components.
    /// Seeds for each choice (dataset, query, config) are printed out for reproducibility.
    runner_rng: StdRng,

    /// For each round, a new pair of datasets is generated. If `None`, keep
    /// running until the time limit is reached
    max_rounds: Option<usize>,
    /// How many different queries to run in each round
    queries_per_round: usize,
    /// For each query, how many different configurations to try and make sure their
    /// results are consistent
    config_variations_per_query: usize,
    /// The time limit for the entire fuzzer execution.
    time_limit: Option<Duration>,

    /// Statistics of the runs so far, printed at the end of `run()`
    stats: RunStats,
}

/// Counters describing what the fuzzer actually exercised, so that a passing
/// run can be checked for having reached the interesting code paths.
#[derive(Debug, Default)]
struct RunStats {
    /// Number of (dataset, query, config) combinations executed
    num_runs: usize,
    /// Number of runs executed under a memory limit
    num_memory_limited: usize,
    /// Number of runs that spilled at least once
    num_spilled: usize,
    /// Number of runs skipped because the hash join ran out of memory
    num_hash_join_oom: usize,
}

impl JoinQueryFuzzer {
    pub fn new(seed: u64) -> Self {
        let test_gen = JoinFuzzerTestGenerator::new(
            20_000,
            4,
            get_supported_types_columns(seed),
            true,
            seed,
        );

        Self {
            test_gen,
            runner_rng: StdRng::seed_from_u64(seed),
            max_rounds: Some(2),
            queries_per_round: 3,
            config_variations_per_query: 5,
            time_limit: None,
            stats: RunStats::default(),
        }
    }

    pub fn with_test_generator(mut self, test_gen: JoinFuzzerTestGenerator) -> Self {
        self.test_gen = test_gen;
        self
    }

    pub fn with_max_rounds(mut self, max_rounds: Option<usize>) -> Self {
        self.max_rounds = max_rounds;
        self
    }

    pub fn with_queries_per_round(mut self, queries_per_round: usize) -> Self {
        self.queries_per_round = queries_per_round;
        self
    }

    pub fn with_config_variations_per_query(
        mut self,
        config_variations_per_query: usize,
    ) -> Self {
        self.config_variations_per_query = config_variations_per_query;
        self
    }

    pub fn with_time_limit(mut self, time_limit: Duration) -> Self {
        self.time_limit = Some(time_limit);
        self
    }

    fn should_stop_due_to_time_limit(
        &self,
        start_time: Instant,
        n_round: usize,
        n_query: usize,
    ) -> bool {
        if let Some(time_limit) = self.time_limit
            && Instant::now().duration_since(start_time) > time_limit
        {
            println!(
                "[JoinQueryFuzzer] Time limit reached: {} queries ({} random configs each) in {} rounds",
                n_round * self.queries_per_round + n_query,
                self.config_variations_per_query,
                n_round
            );
            return true;
        }
        false
    }

    pub async fn run(&mut self) -> Result<()> {
        let start_time = Instant::now();

        // Execute until either `max_rounds` or `time_limit` is reached
        let max_rounds = self.max_rounds.unwrap_or(usize::MAX);
        'outer: for round in 0..max_rounds {
            let dataset_seed = self.runner_rng.random();
            for query_i in 0..self.queries_per_round {
                let query_seed = self.runner_rng.random();
                // The first config always runs without a memory limit and its
                // result is the expected result for all the following configs
                let mut expected_results: Option<Vec<RecordBatch>> = None;
                for config_i in 0..self.config_variations_per_query {
                    if self.should_stop_due_to_time_limit(start_time, round, query_i) {
                        break 'outer;
                    }

                    let config_seed = self.runner_rng.random();
                    let allow_memory_limit = config_i != 0;

                    println!(
                        "[JoinQueryFuzzer] Round {round}, Query {query_i} (Config {config_i})"
                    );
                    println!("  Seeds:");
                    println!("    dataset_seed = {dataset_seed}");
                    println!("    query_seed   = {query_seed}");
                    println!("    config_seed  = {config_seed}");

                    let outcome = self
                        .test_gen
                        .fuzzer_run(
                            dataset_seed,
                            query_seed,
                            config_seed,
                            allow_memory_limit,
                        )
                        .await?;
                    println!(); // Separator between tested runs

                    self.stats.num_runs += 1;
                    if outcome.memory_limited {
                        self.stats.num_memory_limited += 1;
                    }
                    if outcome.spill_count > 0 {
                        self.stats.num_spilled += 1;
                    }

                    let Some(results) = outcome.results else {
                        self.stats.num_hash_join_oom += 1;
                        continue;
                    };

                    match &expected_results {
                        None => expected_results = Some(results),
                        Some(expected) => {
                            assert_results_equal(expected, &results);
                        }
                    }
                }
            }
        }

        println!("[JoinQueryFuzzer] Finished: {:?}", self.stats);
        Ok(())
    }
}

/// Panics with a descriptive message if the two results are not the same
/// multiset of rows.
///
/// Both results are sorted by all their columns and compared column by
/// column, which is much faster than comparing formatted rows for large
/// join outputs.
fn assert_results_equal(expected: &[RecordBatch], actual: &[RecordBatch]) {
    let expected_rows: usize = expected.iter().map(|b| b.num_rows()).sum();
    let actual_rows: usize = actual.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        expected_rows, actual_rows,
        "Row count mismatch: expected {expected_rows} rows, got {actual_rows} rows"
    );
    if expected_rows == 0 {
        return;
    }

    let expected = sort_rows(expected);
    let actual = sort_rows(actual);
    assert_eq!(
        expected.schema(),
        actual.schema(),
        "Schema mismatch between configurations"
    );

    for (col_idx, (e, a)) in expected.columns().iter().zip(actual.columns()).enumerate() {
        if e != a {
            // Find the first differing row for the error message
            let row = (0..e.len())
                .find(|&i| e.slice(i, 1) != a.slice(i, 1))
                .unwrap_or(0);
            let window = row.saturating_sub(2)..(row + 3).min(e.len());
            panic!(
                "Inconsistent join results between configurations: column {} ({}) differs at sorted row {row}\nexpected:\n{}\nactual:\n{}",
                col_idx,
                expected.schema().field(col_idx).name(),
                pretty_format_batches(&[
                    expected.slice(window.start, window.end - window.start)
                ])
                .unwrap(),
                pretty_format_batches(&[
                    actual.slice(window.start, window.end - window.start)
                ])
                .unwrap(),
            );
        }
    }
}

/// Concatenates `batches` and sorts the rows by all the columns.
///
/// Dictionary columns are cast to their value type first: arrays compare
/// physically, and the same logical values can have different dictionaries
/// (or nulls encoded as a null key vs a null value) in different runs.
fn sort_rows(batches: &[RecordBatch]) -> RecordBatch {
    let batch = concat_batches(&batches[0].schema(), batches).unwrap();
    let columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| match c.data_type() {
            DataType::Dictionary(_, value_type) => cast(c.as_ref(), value_type).unwrap(),
            _ => Arc::clone(c),
        })
        .collect();
    let fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .zip(&columns)
        .map(|(f, c)| Field::new(f.name(), c.data_type().clone(), true))
        .collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();

    let sort_columns: Vec<SortColumn> = batch
        .columns()
        .iter()
        .map(|c| SortColumn {
            values: Arc::clone(c),
            options: None,
        })
        .collect();
    let indices = lexsort_to_indices(&sort_columns, None).unwrap();
    let columns = batch
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &indices, None).unwrap())
        .collect();
    RecordBatch::try_new(batch.schema(), columns).unwrap()
}

/// How the values of the join key columns are distributed
#[derive(Debug, Clone, Copy)]
struct KeyDistribution {
    /// Key values are drawn from `0..num_distinct`
    num_distinct: usize,
    /// The fraction of null key values
    null_pct: f64,
    /// If true, half of the non-null values are the single key `0`, so that
    /// one key is much hotter than the others
    skewed: bool,
}

/// The types of columns that can be used as join keys.
///
/// Join keys are generated from a small fixed domain (see
/// [`JoinKeyColumn::generate`]) so that the two sides of the join actually
/// share values. The generic [`RecordBatchGenerator`] draws distinct values
/// from the whole range of the type, which would make matches between two
/// independently generated tables vanishingly unlikely.
#[derive(Debug, Clone)]
struct JoinKeyColumn {
    name: String,
    data_type: DataType,
}

impl JoinKeyColumn {
    fn candidates() -> Vec<Self> {
        vec![
            Self::new("k_i32", DataType::Int32),
            Self::new("k_i64", DataType::Int64),
            Self::new("k_u64", DataType::UInt64),
            Self::new("k_utf8", DataType::Utf8),
            Self::new("k_date32", DataType::Date32),
            Self::new("k_ts_ns", DataType::Timestamp(TimeUnit::Nanosecond, None)),
            Self::new("k_decimal128", DataType::Decimal128(10, 2)),
        ]
    }

    fn new(name: &str, data_type: DataType) -> Self {
        Self {
            name: name.to_string(),
            data_type,
        }
    }

    fn field(&self) -> Field {
        Field::new(&self.name, self.data_type.clone(), true)
    }

    /// Generate `num_rows` key values according to `distribution`
    fn generate(
        &self,
        rng: &mut StdRng,
        num_rows: usize,
        distribution: KeyDistribution,
    ) -> ArrayRef {
        let ids: Vec<Option<i64>> = (0..num_rows)
            .map(|_| {
                if rng.random_bool(distribution.null_pct) {
                    None
                } else if distribution.skewed && rng.random_bool(0.5) {
                    Some(0)
                } else {
                    Some(rng.random_range(0..distribution.num_distinct as i64))
                }
            })
            .collect();

        match &self.data_type {
            DataType::Int32 => Arc::new(Int32Array::from_iter(
                ids.iter().map(|v| v.map(|v| v as i32)),
            )),
            DataType::Int64 => Arc::new(Int64Array::from_iter(ids)),
            DataType::UInt64 => Arc::new(UInt64Array::from_iter(
                ids.iter().map(|v| v.map(|v| v as u64)),
            )),
            DataType::Utf8 => Arc::new(StringArray::from_iter(
                ids.iter().map(|v| v.map(|v| format!("key_{v}"))),
            )),
            DataType::Date32 => Arc::new(Date32Array::from_iter(
                ids.iter().map(|v| v.map(|v| v as i32)),
            )),
            DataType::Timestamp(TimeUnit::Nanosecond, None) => {
                Arc::new(TimestampNanosecondArray::from_iter(
                    ids.iter().map(|v| v.map(|v| v * 1_000_000)),
                ))
            }
            DataType::Decimal128(p, s) => Arc::new(
                Decimal128Array::from_iter(ids.iter().map(|v| v.map(|v| v as i128)))
                    .with_precision_and_scale(*p, *s)
                    .unwrap(),
            ),
            other => unreachable!("unsupported join key type {other}"),
        }
    }
}

/// Struct to generate and manage a random pair of datasets for join fuzz
/// testing. It is able to re-run the failed test cases by setting the same
/// seeds printed out.
///
/// Both tables share the same schema: a few join key columns from a small
/// domain (see [`JoinKeyColumn`]) followed by payload columns of random
/// types.
pub struct JoinFuzzerTestGenerator {
    /// The approximate number of rows of each registered table. Each table
    /// gets a random number of rows between 1 and this value.
    max_num_rows: usize,
    /// Max number of partitions for each registered table
    max_partitions: usize,
    /// The join key columns shared by both tables
    key_columns: Vec<JoinKeyColumn>,
    /// The payload columns shared by both tables
    payload_columns: Vec<ColumnDescr>,
    /// If true, will randomly generate a memory limit for the query. Otherwise
    /// the query will run under the context with unlimited memory.
    set_memory_limit: bool,

    /// States related to the randomly generated datasets. `None` if not
    /// initialized by calling `init_datasets()`
    dataset_state: Option<DatasetState>,
    /// The seed used to generate `dataset_state`, so that the (expensive)
    /// dataset generation is skipped when the same seed is requested again
    dataset_seed: Option<u64>,
}

/// One generated table
struct TableState {
    /// Outer vector is the partitions, inner vector is staggered batches
    /// within the same partition.
    partitioned_batches: Vec<Vec<RecordBatch>>,
    /// Number of rows in the table
    num_rows: usize,
    /// The memory size of the table
    mem_size: usize,
    /// The largest memory size of a single batch of the table
    max_batch_mem_size: usize,
}

/// Struct to hold states related to the randomly generated datasets
struct DatasetState {
    schema: SchemaRef,
    left: TableState,
    right: TableState,
    /// The approximate number of rows of a batch (staggered batches are
    /// generated with a random number of rows between 1 and this value)
    approx_batch_num_rows: usize,
}

/// The join types exercised by the fuzzer, with the SQL syntax used to
/// request them.
const JOIN_TYPES: &[&str] = &[
    "INNER JOIN",
    "LEFT JOIN",
    "RIGHT JOIN",
    "FULL JOIN",
    "LEFT SEMI JOIN",
    "RIGHT SEMI JOIN",
    "LEFT ANTI JOIN",
    "RIGHT ANTI JOIN",
];

impl JoinFuzzerTestGenerator {
    /// Randomly pick a subset of `candidate_columns` to be used as the
    /// payload columns of both tables.
    pub fn new(
        max_num_rows: usize,
        max_partitions: usize,
        candidate_columns: Vec<ColumnDescr>,
        set_memory_limit: bool,
        rng_seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(rng_seed);

        let num_keys = rng.random_range(1..=3);
        let key_columns = JoinKeyColumn::candidates()
            .choose_multiple(&mut rng, num_keys)
            .cloned()
            .collect();

        // View arrays are excluded: the generator allocates a view buffer
        // block (8 KB+) for every tiny batch, and `take`/`concat` keep all the
        // source buffers alive, so the accounted memory of a batch can be two
        // orders of magnitude larger than its data. That makes memory limited
        // execution fail on the very first batch regardless of the limit.
        let candidate_columns: Vec<_> = candidate_columns
            .into_iter()
            .filter(|c| {
                !matches!(c.column_type, DataType::Utf8View | DataType::BinaryView)
            })
            .collect();
        let num_payload = rng.random_range(1..=candidate_columns.len().min(4));
        let payload_columns = candidate_columns
            .choose_multiple(&mut rng, num_payload)
            .cloned()
            .collect();

        Self {
            max_num_rows,
            max_partitions,
            key_columns,
            payload_columns,
            set_memory_limit,
            dataset_state: None,
            dataset_seed: None,
        }
    }

    fn schema(&self) -> SchemaRef {
        let mut fields: Vec<Field> = self.key_columns.iter().map(|k| k.field()).collect();
        fields.extend(
            self.payload_columns
                .iter()
                .map(|c| Field::new(&c.name, c.column_type.clone(), true)),
        );
        Arc::new(Schema::new(fields))
    }

    /// Generate one table with `target_num_rows` rows split into
    /// `num_partitions` partitions of staggered batches of at most
    /// `max_batch_num_rows` rows.
    fn generate_table(
        &self,
        rng: &mut StdRng,
        schema: &SchemaRef,
        target_num_rows: usize,
        num_partitions: usize,
        max_batch_num_rows: usize,
        key_distribution: KeyDistribution,
    ) -> TableState {
        let target_partition_size = target_num_rows.div_ceil(num_partitions);

        let mut partitions = Vec::new();
        for _ in 0..num_partitions {
            let mut partition = Vec::new();
            let mut num_rows = 0;

            while num_rows < target_partition_size {
                // Let edge case (1-row batch) more common
                let (min_nrow, max_nrow) = if rng.random_bool(0.1) {
                    (1, 3)
                } else {
                    (1, max_batch_num_rows)
                };

                // Payload columns come from the generic generator
                let mut payload_generator = RecordBatchGenerator::new(
                    min_nrow,
                    max_nrow,
                    self.payload_columns.clone(),
                )
                .with_seed(rng.random());
                let payload = payload_generator.generate().unwrap();
                let batch_num_rows = payload.num_rows();

                // Join key columns come from the small fixed domain
                let mut columns: Vec<ArrayRef> = self
                    .key_columns
                    .iter()
                    .map(|k| k.generate(rng, batch_num_rows, key_distribution))
                    .collect();
                columns.extend(payload.columns().iter().cloned());

                let batch = RecordBatch::try_new(Arc::clone(schema), columns).unwrap();
                num_rows += batch.num_rows();
                partition.push(batch);
            }

            partitions.push(partition);
        }

        // Optionally make one partition empty
        if num_partitions > 1 && rng.random_bool(0.1) {
            let partition_index = rng.random_range(0..num_partitions);
            partitions[partition_index] = Vec::new();
        }

        let num_rows = partitions
            .iter()
            .flatten()
            .map(|b| b.num_rows())
            .sum::<usize>();
        let mem_size = partitions
            .iter()
            .flatten()
            .map(get_record_batch_memory_size)
            .sum::<usize>();
        let max_batch_mem_size = partitions
            .iter()
            .flatten()
            .map(get_record_batch_memory_size)
            .max()
            .unwrap_or(0);

        TableState {
            partitioned_batches: partitions,
            num_rows,
            mem_size,
            max_batch_mem_size,
        }
    }

    /// Generate the left and right tables.
    ///
    /// The two tables have independent sizes and partition counts (so the
    /// build side is sometimes larger than the probe side), but share the
    /// key domain so that joins produce matches.
    fn init_datasets(&mut self, rng_seed: u64) {
        if self.dataset_seed == Some(rng_seed) && self.dataset_state.is_some() {
            return;
        }
        self.dataset_seed = Some(rng_seed);

        let mut rng = StdRng::seed_from_u64(rng_seed);
        let schema = self.schema();

        // Batches are kept small relative to the table so that a single batch
        // never exceeds the (per partition) memory budget
        let max_batch_num_rows = (self.max_num_rows / self.max_partitions / 50).max(1);

        // The domain of the join keys. A small domain gives many matches per
        // key, a large domain gives mostly 1:1 matches.
        let key_distribution = KeyDistribution {
            num_distinct: *[1, 10, 100, 1000, self.max_num_rows]
                .choose(&mut rng)
                .unwrap(),
            null_pct: *[0.0, 0.01, 0.1, 0.5].choose(&mut rng).unwrap(),
            skewed: rng.random_bool(0.3),
        };

        println!("  Dataset:");
        println!("    Schema: {}", display_schema(&schema));
        println!("    Keys: {key_distribution:?}");

        // Pick the table sizes, then shrink the second table if needed so that
        // the expected number of rows of an inner join stays bounded. With few
        // distinct (or skewed) keys the join output is quadratic in the table
        // sizes otherwise.
        let mut sizes: Vec<usize> = (0..2)
            .map(|_| {
                // Occasionally generate a (nearly) empty table
                if rng.random_bool(0.05) {
                    rng.random_range(0..=3)
                } else {
                    rng.random_range(1..=self.max_num_rows)
                }
            })
            .collect();
        // Expected matches per left row: uniform part plus the hot key, which
        // holds half of the rows on both sides when skewed
        let matches_per_row = sizes[1] as f64 / key_distribution.num_distinct as f64
            + if key_distribution.skewed {
                sizes[1] as f64 / 4.0
            } else {
                0.0
            };
        let expected_output = sizes[0] as f64 * matches_per_row;
        if expected_output > MAX_EXPECTED_JOIN_OUTPUT_ROWS as f64 {
            let scale = MAX_EXPECTED_JOIN_OUTPUT_ROWS as f64 / expected_output;
            sizes[1] = ((sizes[1] as f64 * scale) as usize).max(1);
        }

        let mut tables = Vec::with_capacity(2);
        for (side, target_num_rows) in ["Left", "Right"].into_iter().zip(sizes) {
            let num_partitions = rng.random_range(1..=self.max_partitions);

            let table = self.generate_table(
                &mut rng,
                &schema,
                target_num_rows,
                num_partitions,
                max_batch_num_rows,
                key_distribution,
            );

            println!(
                "    {side} table: {} rows, {} partitions, {}",
                table.num_rows,
                num_partitions,
                human_readable_size(table.mem_size)
            );
            tables.push(table);
        }
        let right = tables.pop().unwrap();
        let left = tables.pop().unwrap();

        self.dataset_state = Some(DatasetState {
            schema,
            left,
            right,
            approx_batch_num_rows: max_batch_num_rows,
        });
    }

    /// Generates a random join query.
    ///
    /// The query joins the two tables on a random non-empty subset of the key
    /// columns, optionally with an additional non-equi condition on one of
    /// the payload columns, and projects a random subset of columns from
    /// each side.
    pub fn generate_random_query(&self, rng_seed: u64) -> String {
        let mut rng = StdRng::seed_from_u64(rng_seed);

        let join_type = *JOIN_TYPES.choose(&mut rng).unwrap();
        let is_semi_or_anti = join_type.contains("SEMI") || join_type.contains("ANTI");
        let right_side_only =
            join_type.starts_with("RIGHT SEMI") || join_type.starts_with("RIGHT ANTI");

        let num_on_keys = rng.random_range(1..=self.key_columns.len());
        let on_keys: Vec<_> = self
            .key_columns
            .choose_multiple(&mut rng, num_on_keys)
            .collect();

        let mut conditions: Vec<String> = on_keys
            .iter()
            .map(|k| format!("l.{0} = r.{0}", k.name))
            .collect();

        // Optionally add a non-equi join condition on a numeric payload
        // column, so that the join filter code paths are exercised
        let numeric_payload: Vec<_> = self
            .payload_columns
            .iter()
            .filter(|c| c.column_type.is_numeric())
            .collect();
        if rng.random_bool(0.3)
            && let Some(col) = numeric_payload.choose(&mut rng)
        {
            let op = ["<", "<=", ">", ">=", "<>"].choose(&mut rng).unwrap();
            conditions.push(format!("l.{0} {op} r.{0}", col.name));
        }

        // Project a random non-empty subset of columns from the available
        // sides. Columns are aliased to avoid duplicate output names.
        let all_columns: Vec<String> = self
            .key_columns
            .iter()
            .map(|k| k.name.clone())
            .chain(self.payload_columns.iter().map(|c| c.name.clone()))
            .collect();

        let mut projections = Vec::new();
        let num_projected = rng.random_range(1..=all_columns.len());
        if !is_semi_or_anti || !right_side_only {
            for col in all_columns.choose_multiple(&mut rng, num_projected) {
                projections.push(format!("l.{col} AS l_{col}"));
            }
        }
        if !is_semi_or_anti || right_side_only {
            for col in all_columns.choose_multiple(&mut rng, num_projected) {
                projections.push(format!("r.{col} AS r_{col}"));
            }
        }

        format!(
            "SELECT {} FROM {LEFT_TABLE} l {join_type} {RIGHT_TABLE} r ON {}",
            projections.join(", "),
            conditions.join(" AND ")
        )
    }

    /// Generate a random session context for running the query.
    ///
    /// Randomized:
    /// - join algorithm (`prefer_hash_join`)
    /// - hash join partition mode (`hash_join_single_partition_threshold`)
    /// - number of target partitions and batch size
    /// - memory limit (if `with_memory_limit`), between 10% and 200% of the
    ///   combined in-memory size of both tables. The low end is well below
    ///   the size of the build side so that hash joins run out of memory and
    ///   sort merge joins have to spill.
    pub fn generate_random_config(
        &self,
        rng_seed: u64,
        with_memory_limit: bool,
    ) -> Result<SessionContext> {
        let mut rng = StdRng::seed_from_u64(rng_seed);
        let init_state = self.dataset_state.as_ref().unwrap();

        let dataset_size = init_state.left.mem_size + init_state.right.mem_size;
        let num_partitions = rng.random_range(1..=self.max_partitions);
        let batch_size = rng.random_range(1..=init_state.approx_batch_num_rows);

        let prefer_hash_join = rng.random_bool(0.5);
        // `0` forces `PartitionMode::Partitioned`, `usize::MAX` forces
        // `PartitionMode::CollectLeft`
        let hash_join_single_partition_threshold =
            *[0, usize::MAX].choose(&mut rng).unwrap();

        // Pick a fraction of the dataset size from a coarse set so that both
        // very tight and comfortable limits are well represented, then jitter
        let memory_limit_fraction = *[0.1, 0.25, 0.5, 1.0, 2.0].choose(&mut rng).unwrap()
            * rng.random_range(0.8..=1.2);
        let requested_memory_limit =
            ((dataset_size as f64 * memory_limit_fraction) as usize).max(1);

        // No operator can make progress if it can not hold a couple of
        // batches, so the limit is floored to a small multiple of the
        // (accounted) batch size for every spillable consumer in the plan.
        //
        // A sort merge join plan has 4 spillable consumers per partition (a
        // repartition channel and an external sort on each side), and
        // `FairSpillPool` divides the pool evenly between them after
        // subtracting what the unspillable consumers hold. The sort reserves
        // about twice the batch memory for each batch it buffers.
        //
        // The batches reaching the sort are re-batched to `batch_size` rows by
        // the repartition, so the largest batch of the dataset is scaled
        // accordingly. Small batches are dominated by fixed per-buffer
        // overhead, which is why the measured size is used rather than the
        // average bytes per row.
        let max_batch_mem_size = init_state
            .left
            .max_batch_mem_size
            .max(init_state.right.max_batch_mem_size);
        let batch_mem_size = max_batch_mem_size
            * batch_size.div_ceil(init_state.approx_batch_num_rows).max(1);
        let reserved_per_batch = 2 * batch_mem_size;
        // The sort needs this much reserved memory for merging spilled files,
        // about 1 to 2 batches worth
        let sort_spill_reservation_bytes =
            rng.random_range(batch_mem_size..=batch_mem_size * 2);
        let spillable_consumers = 4 * num_partitions;
        let unspillable = 2 * num_partitions * sort_spill_reservation_bytes;
        let min_memory_limit =
            spillable_consumers * MIN_BATCHES_PER_CONSUMER * reserved_per_batch
                + unspillable;
        let memory_limit = requested_memory_limit.max(min_memory_limit);

        let memory_limit_str = if with_memory_limit {
            human_readable_size(memory_limit)
        } else {
            "Unbounded".to_string()
        };

        println!("  Config: ");
        println!("    Dataset size: {}", human_readable_size(dataset_size));
        println!(
            "    Memory limit: {memory_limit_str} (requested {:.0}% of dataset size = {}, floor {})",
            memory_limit_fraction * 100.0,
            human_readable_size(requested_memory_limit),
            human_readable_size(min_memory_limit),
        );
        println!("    Prefer hash join: {prefer_hash_join}");
        println!(
            "    Hash join single partition threshold: {hash_join_single_partition_threshold}"
        );
        println!("    Target partitions: {num_partitions}");
        println!("    Batch size: {batch_size}");
        println!(
            "    Sort spill reservation bytes: {}",
            human_readable_size(sort_spill_reservation_bytes)
        );

        let mut config = SessionConfig::new()
            .with_target_partitions(num_partitions)
            .with_batch_size(batch_size)
            .with_sort_spill_reservation_bytes(sort_spill_reservation_bytes)
            // Setting this too large causes external sort to fail
            .with_sort_in_place_threshold_bytes(0);
        config.options_mut().optimizer.prefer_hash_join = prefer_hash_join;
        config
            .options_mut()
            .optimizer
            .hash_join_single_partition_threshold = hash_join_single_partition_threshold;

        let memory_pool: Arc<dyn MemoryPool> = if with_memory_limit {
            Arc::new(FairSpillPool::new(memory_limit))
        } else {
            Arc::new(UnboundedMemoryPool::default())
        };

        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(memory_pool)
            .with_disk_manager_builder(DiskManagerBuilder::default())
            .build_arc()?;

        let ctx = SessionContext::new_with_config_rt(config, runtime);

        let schema = &init_state.schema;
        let left = MemTable::try_new(
            Arc::clone(schema),
            init_state.left.partitioned_batches.clone(),
        )?;
        let right = MemTable::try_new(
            Arc::clone(schema),
            init_state.right.partitioned_batches.clone(),
        )?;
        ctx.register_table(LEFT_TABLE, Arc::new(left))?;
        ctx.register_table(RIGHT_TABLE, Arc::new(right))?;

        Ok(ctx)
    }

    /// Run one (dataset, query, config) combination.
    ///
    /// If `allow_memory_limit` is false the query always runs with an
    /// unbounded memory pool, regardless of `self.set_memory_limit`.
    async fn fuzzer_run(
        &mut self,
        dataset_seed: u64,
        query_seed: u64,
        config_seed: u64,
        allow_memory_limit: bool,
    ) -> Result<RunOutcome> {
        self.init_datasets(dataset_seed);
        let query_str = self.generate_random_query(query_seed);
        println!("  Query:");
        println!("    {query_str}");

        let with_mem_limit = allow_memory_limit
            && self.set_memory_limit
            && StdRng::seed_from_u64(config_seed).random_bool(0.7);

        let ctx = self.generate_random_config(config_seed, with_mem_limit)?;
        let prefer_hash_join = ctx.state().config().options().optimizer.prefer_hash_join;

        let plan = ctx.sql(&query_str).await?.create_physical_plan().await?;
        let join_operators = join_operators(plan.as_ref());
        println!("    Join operators: {}", join_operators.join(", "));
        // Note that the planner only picks `SortMergeJoinExec` when
        // `prefer_hash_join` is false AND `target_partitions > 1`, so the
        // hash join OOM allowance below is based on the actual plan
        let uses_hash_join = join_operators.iter().any(|name| name == "HashJoinExec");

        let result = collect(Arc::clone(&plan), ctx.task_ctx()).await;

        let spill_count = plan_spill_count(plan.as_ref());
        let spilled_bytes = plan_spilled_bytes(plan.as_ref());
        println!(
            "  Spills: {spill_count} ({})",
            human_readable_size(spilled_bytes)
        );

        let results = match result {
            Ok(results) => Some(results),
            // The error may be wrapped (e.g. in `DataFusionError::Shared` when
            // it comes from the shared build side future), so look at the root
            Err(e)
                if with_mem_limit
                    && uses_hash_join
                    && matches!(
                        e.find_root(),
                        DataFusionError::ResourcesExhausted(_)
                    ) =>
            {
                // Known limitation: hash join does not support spilling yet,
                // see https://github.com/apache/datafusion/issues/17267
                println!("  Hash join ran out of memory (known limitation, skipped):");
                println!("    {e}");
                None
            }
            Err(e) => {
                // The metrics show which operators spilled and how much
                // memory/rows flowed through each of them
                println!("  Plan with metrics:");
                println!(
                    "{}",
                    DisplayableExecutionPlan::with_metrics(plan.as_ref()).indent(true)
                );
                panic!(
                    "Query failed (memory limit: {with_mem_limit}, prefer_hash_join: {prefer_hash_join}): {e}"
                )
            }
        };

        Ok(RunOutcome {
            results,
            memory_limited: with_mem_limit,
            spill_count,
        })
    }
}

/// The outcome of one `fuzzer_run`
struct RunOutcome {
    /// The query results, or `None` if the run hit a known limitation that
    /// makes the result unavailable (currently: a hash join running out of
    /// memory under a memory limit, since `HashJoinExec` can not spill).
    results: Option<Vec<RecordBatch>>,
    /// Whether the run was executed under a memory limit
    memory_limited: bool,
    /// The number of spills across all operators of the plan
    spill_count: usize,
}

/// The names of all join operators in the plan, so that the log shows which
/// join algorithm the planner actually picked.
fn join_operators(plan: &dyn ExecutionPlan) -> Vec<String> {
    let mut names = Vec::new();
    if plan.name().contains("Join") {
        names.push(plan.name().to_string());
    }
    for child in plan.children() {
        names.extend(join_operators(child.as_ref()));
    }
    names
}

#[cfg(test)]
mod test {
    use super::*;

    /// Given the same seeds, the result should be the same
    #[tokio::test]
    async fn test_join_query_fuzzer_deterministic() {
        let gen_seed = 310104;
        let mut test_generator = JoinFuzzerTestGenerator::new(
            500,
            3,
            get_supported_types_columns(gen_seed),
            false,
            gen_seed,
        );

        let res1 = test_generator.fuzzer_run(1, 2, 3, false).await.unwrap();
        let res2 = test_generator.fuzzer_run(1, 2, 3, false).await.unwrap();
        assert_results_equal(&res1.results.unwrap(), &res2.results.unwrap());
    }
}

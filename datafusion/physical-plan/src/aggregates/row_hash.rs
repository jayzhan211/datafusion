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

//! Hash aggregation

use std::sync::Arc;
use std::task::{Context, Poll};
use std::vec;

use crate::aggregates::group_values::{new_group_values, GroupValues};
use crate::aggregates::order::GroupOrderingFull;
use crate::aggregates::{
    evaluate_group_by, evaluate_many, evaluate_optional, group_schema, AggregateMode,
    PhysicalGroupBy,
};
use crate::common::SharedMemoryReservation;
use crate::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet, RecordOutput
};
use crate::repartition::distributor_channels::{channels, DistributionSender};
use crate::repartition::{
    InputPartitionsToCurrentPartitionReceiver, InputPartitionsToCurrentPartitionSender,
    MaybeBatch, RepartitionMetrics,
};
use crate::sorts::sort::sort_batch;
use crate::sorts::streaming_merge;
use crate::spill::{read_spill_as_stream, spill_record_batch_by_size};
use crate::stream::RecordBatchStreamAdapter;
use crate::{aggregates, metrics, ExecutionPlan, ExecutionPlanProperties, PhysicalExpr};
use crate::{RecordBatchStream, SendableRecordBatchStream};

use arrow::array::*;
use arrow::datatypes::SchemaRef;
use arrow_schema::SortOptions;
use datafusion_common::utils::take_arrays;
use datafusion_common::{internal_datafusion_err, DataFusionError, Result};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_execution::TaskContext;
use datafusion_expr::{EmitTo, GroupsAccumulator};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{GroupsAccumulatorAdapter, PhysicalSortExpr};

use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use futures::ready;
use futures::stream::{Stream, StreamExt};
use hashbrown::HashMap;
use log::debug;
use parking_lot::Mutex;
use types::UInt32Type;

// type LazyState = Arc<tokio::sync::OnceCell<Mutex<GroupRepartitionExecState>>>;

use super::order::GroupOrdering;
use super::AggregateExec;

#[derive(Debug, Clone)]
/// This object tracks the aggregation phase (input/output)
pub(crate) enum ExecutionState {
    ReadingInput,
    /// When producing output, the remaining rows to output are stored
    /// here and are sliced off as needed in batch_size chunks
    ProducingOutput(RecordBatch),
    ReadingInputPartitioned(RecordBatch),
    // ProducingPartitionedOutput(Vec<RecordBatch>),
    // ReadingPartitionedInput(Vec<RecordBatch>),
    /// Produce intermediate aggregate state for each input row without
    /// aggregation.
    ///
    /// See "partial aggregation" discussion on [`GroupedHashAggregateStream`]
    SkippingAggregation,
    /// All input has been consumed and all groups have been emitted
    Done,
}

/// This encapsulates the spilling state
struct SpillState {
    // ========================================================================
    // PROPERTIES:
    // These fields are initialized at the start and remain constant throughout
    // the execution.
    // ========================================================================
    /// Sorting expression for spilling batches
    spill_expr: Vec<PhysicalSortExpr>,

    /// Schema for spilling batches
    spill_schema: SchemaRef,

    /// aggregate_arguments for merging spilled data
    merging_aggregate_arguments: Vec<Vec<Arc<dyn PhysicalExpr>>>,

    /// GROUP BY expressions for merging spilled data
    merging_group_by: PhysicalGroupBy,

    // ========================================================================
    // STATES:
    // Fields changes during execution. Can be buffer, or state flags that
    // influence the execution in parent `GroupedHashAggregateStream`
    // ========================================================================
    /// If data has previously been spilled, the locations of the
    /// spill files (in Arrow IPC format)
    spills: Vec<RefCountedTempFile>,

    /// true when streaming merge is in progress
    is_stream_merging: bool,
}

/// Tracks if the aggregate should skip partial aggregations
///
/// See "partial aggregation" discussion on [`GroupedHashAggregateStream`]
struct SkipAggregationProbe {
    // ========================================================================
    // PROPERTIES:
    // These fields are initialized at the start and remain constant throughout
    // the execution.
    // ========================================================================
    /// Aggregation ratio check performed when the number of input rows exceeds
    /// this threshold (from `SessionConfig`)
    probe_rows_threshold: usize,
    /// Maximum ratio of `num_groups` to `input_rows` for continuing aggregation
    /// (from `SessionConfig`). If the ratio exceeds this value, aggregation
    /// is skipped and input rows are directly converted to output
    probe_ratio_threshold: f64,

    // ========================================================================
    // STATES:
    // Fields changes during execution. Can be buffer, or state flags that
    // influence the exeuction in parent `GroupedHashAggregateStream`
    // ========================================================================
    /// Number of processed input rows (updated during probing)
    input_rows: usize,
    /// Number of total group values for `input_rows` (updated during probing)
    num_groups: usize,

    /// Flag indicating further data aggregation may be skipped (decision made
    /// when probing complete)
    should_skip: bool,
    /// Flag indicating further updates of `SkipAggregationProbe` state won't
    /// make any effect (set either while probing or on probing completion)
    is_locked: bool,

    /// Number of rows where state was output without aggregation.
    ///
    /// * If 0, all input rows were aggregated (should_skip was always false)
    ///
    /// * if greater than zero, the number of rows which were output directly
    ///   without aggregation
    skipped_aggregation_rows: metrics::Count,
}

impl SkipAggregationProbe {
    fn new(
        probe_rows_threshold: usize,
        probe_ratio_threshold: f64,
        skipped_aggregation_rows: metrics::Count,
    ) -> Self {
        Self {
            input_rows: 0,
            num_groups: 0,
            probe_rows_threshold,
            probe_ratio_threshold,
            should_skip: false,
            is_locked: false,
            skipped_aggregation_rows,
        }
    }

    /// Updates `SkipAggregationProbe` state:
    /// - increments the number of input rows
    /// - replaces the number of groups with the new value
    /// - on `probe_rows_threshold` exceeded calculates
    ///   aggregation ratio and sets `should_skip` flag
    /// - if `should_skip` is set, locks further state updates
    fn update_state(&mut self, input_rows: usize, num_groups: usize) {
        if self.is_locked {
            return;
        }
        self.input_rows += input_rows;
        self.num_groups = num_groups;
        if self.input_rows >= self.probe_rows_threshold {
            self.should_skip = self.num_groups as f64 / self.input_rows as f64
                >= self.probe_ratio_threshold;
            self.is_locked = true;
        }
    }

    fn should_skip(&self) -> bool {
        self.should_skip
    }

    /// Record the number of rows that were output directly without aggregation
    fn record_skipped(&mut self, batch: &RecordBatch) {
        self.skipped_aggregation_rows.add(batch.num_rows());
    }
}

/// HashTable based Grouping Aggregator
///
/// # Design Goals
///
/// This structure is designed so that updating the aggregates can be
/// vectorized (done in a tight loop) without allocations. The
/// accumulator state is *not* managed by this operator (e.g in the
/// hash table) and instead is delegated to the individual
/// accumulators which have type specialized inner loops that perform
/// the aggregation.
///
/// # Architecture
///
/// ```text
///
///     Assigns a consecutive group           internally stores aggregate values
///     index for each unique set                     for all groups
///         of group values
///
///         ┌────────────┐              ┌──────────────┐       ┌──────────────┐
///         │ ┌────────┐ │              │┌────────────┐│       │┌────────────┐│
///         │ │  "A"   │ │              ││accumulator ││       ││accumulator ││
///         │ ├────────┤ │              ││     0      ││       ││     N      ││
///         │ │  "Z"   │ │              ││ ┌────────┐ ││       ││ ┌────────┐ ││
///         │ └────────┘ │              ││ │ state  │ ││       ││ │ state  │ ││
///         │            │              ││ │┌─────┐ │ ││  ...  ││ │┌─────┐ │ ││
///         │    ...     │              ││ │├─────┤ │ ││       ││ │├─────┤ │ ││
///         │            │              ││ │└─────┘ │ ││       ││ │└─────┘ │ ││
///         │            │              ││ │        │ ││       ││ │        │ ││
///         │ ┌────────┐ │              ││ │  ...   │ ││       ││ │  ...   │ ││
///         │ │  "Q"   │ │              ││ │        │ ││       ││ │        │ ││
///         │ └────────┘ │              ││ │┌─────┐ │ ││       ││ │┌─────┐ │ ││
///         │            │              ││ │└─────┘ │ ││       ││ │└─────┘ │ ││
///         └────────────┘              ││ └────────┘ ││       ││ └────────┘ ││
///                                     │└────────────┘│       │└────────────┘│
///                                     └──────────────┘       └──────────────┘
///
///         group_values                             accumulators
///
///  ```
///
/// For example, given a query like `COUNT(x), SUM(y) ... GROUP BY z`,
/// [`group_values`] will store the distinct values of `z`. There will
/// be one accumulator for `COUNT(x)`, specialized for the data type
/// of `x` and one accumulator for `SUM(y)`, specialized for the data
/// type of `y`.
///
/// # Discussion
///
/// [`group_values`] does not store any aggregate state inline. It only
/// assigns "group indices", one for each (distinct) group value. The
/// accumulators manage the in-progress aggregate state for each
/// group, with the group values themselves are stored in
/// [`group_values`] at the corresponding group index.
///
/// The accumulator state (e.g partial sums) is managed by and stored
/// by a [`GroupsAccumulator`] accumulator. There is one accumulator
/// per aggregate expression (COUNT, AVG, etc) in the
/// stream. Internally, each `GroupsAccumulator` manages the state for
/// multiple groups, and is passed `group_indexes` during update. Note
/// The accumulator state is not managed by this operator (e.g in the
/// hash table).
///
/// [`group_values`]: Self::group_values
///
/// # Partial Aggregate and multi-phase grouping
///
/// As described on [`Accumulator::state`], this operator is used in the context
/// "multi-phase" grouping when the mode is [`AggregateMode::Partial`].
///
/// An important optimization for multi-phase partial aggregation is to skip
/// partial aggregation when it is not effective enough to warrant the memory or
/// CPU cost, as is often the case for queries many distinct groups (high
/// cardinality group by). Memory is particularly important because each Partial
/// aggregator must store the intermediate state for each group.
///
/// If the ratio of the number of groups to the number of input rows exceeds a
/// threshold, and [`GroupsAccumulator::supports_convert_to_state`] is
/// supported, this operator will stop applying Partial aggregation and directly
/// pass the input rows to the next aggregation phase.
///
/// [`Accumulator::state`]: datafusion_expr::Accumulator::state
///
/// # Spilling (to disk)
///
/// The sizes of group values and accumulators can become large. Before that causes out of memory,
/// this hash aggregator outputs partial states early for partial aggregation or spills to local
/// disk using Arrow IPC format for final aggregation. For every input [`RecordBatch`], the memory
/// manager checks whether the new input size meets the memory configuration. If not, outputting or
/// spilling happens. For outputting, the final aggregation takes care of re-grouping. For spilling,
/// later stream-merge sort on reading back the spilled data does re-grouping. Note the rows cannot
/// be grouped once spilled onto disk, the read back data needs to be re-grouped again. In addition,
/// re-grouping may cause out of memory again. Thus, re-grouping has to be a sort based aggregation.
///
/// ```text
/// Partial Aggregation [batch_size = 2] (max memory = 3 rows)
///
///  INPUTS        PARTIALLY AGGREGATED (UPDATE BATCH)   OUTPUTS
/// ┌─────────┐    ┌─────────────────┐                  ┌─────────────────┐
/// │ a │ b   │    │ a │    AVG(b)   │                  │ a │    AVG(b)   │
/// │---│-----│    │   │[count]│[sum]│                  │   │[count]│[sum]│
/// │ 3 │ 3.0 │ ─▶ │---│-------│-----│                  │---│-------│-----│
/// │ 2 │ 2.0 │    │ 2 │ 1     │ 2.0 │ ─▶ early emit ─▶ │ 2 │ 1     │ 2.0 │
/// └─────────┘    │ 3 │ 2     │ 7.0 │               │  │ 3 │ 2     │ 7.0 │
/// ┌─────────┐ ─▶ │ 4 │ 1     │ 8.0 │               │  └─────────────────┘
/// │ 3 │ 4.0 │    └─────────────────┘               └▶ ┌─────────────────┐
/// │ 4 │ 8.0 │    ┌─────────────────┐                  │ 4 │ 1     │ 8.0 │
/// └─────────┘    │ a │    AVG(b)   │               ┌▶ │ 1 │ 1     │ 1.0 │
/// ┌─────────┐    │---│-------│-----│               │  └─────────────────┘
/// │ 1 │ 1.0 │ ─▶ │ 1 │ 1     │ 1.0 │ ─▶ early emit ─▶ ┌─────────────────┐
/// │ 3 │ 2.0 │    │ 3 │ 1     │ 2.0 │                  │ 3 │ 1     │ 2.0 │
/// └─────────┘    └─────────────────┘                  └─────────────────┘
///
///
/// Final Aggregation [batch_size = 2] (max memory = 3 rows)
///
/// PARTIALLY INPUTS       FINAL AGGREGATION (MERGE BATCH)       RE-GROUPED (SORTED)
/// ┌─────────────────┐    [keep using the partial schema]       [Real final aggregation
/// │ a │    AVG(b)   │    ┌─────────────────┐                    output]
/// │   │[count]│[sum]│    │ a │    AVG(b)   │                   ┌────────────┐
/// │---│-------│-----│ ─▶ │   │[count]│[sum]│                   │ a │ AVG(b) │
/// │ 3 │ 3     │ 3.0 │    │---│-------│-----│ ─▶ spill ─┐       │---│--------│
/// │ 2 │ 2     │ 1.0 │    │ 2 │ 2     │ 1.0 │           │       │ 1 │    4.0 │
/// └─────────────────┘    │ 3 │ 4     │ 8.0 │           ▼       │ 2 │    1.0 │
/// ┌─────────────────┐ ─▶ │ 4 │ 1     │ 7.0 │     Streaming  ─▶ └────────────┘
/// │ 3 │ 1     │ 5.0 │    └─────────────────┘     merge sort ─▶ ┌────────────┐
/// │ 4 │ 1     │ 7.0 │    ┌─────────────────┐            ▲      │ a │ AVG(b) │
/// └─────────────────┘    │ a │    AVG(b)   │            │      │---│--------│
/// ┌─────────────────┐    │---│-------│-----│ ─▶ memory ─┘      │ 3 │    2.0 │
/// │ 1 │ 2     │ 8.0 │ ─▶ │ 1 │ 2     │ 8.0 │                   │ 4 │    7.0 │
/// │ 2 │ 2     │ 3.0 │    │ 2 │ 2     │ 3.0 │                   └────────────┘
/// └─────────────────┘    └─────────────────┘
/// ```
pub struct GroupedHashAggregateStream {
    // ========================================================================
    // PROPERTIES:
    // These fields are initialized at the start and remain constant throughout
    // the execution.
    // ========================================================================
    schema: SchemaRef,
    input: SendableRecordBatchStream,
    mode: AggregateMode,

    /// Arguments to pass to each accumulator.
    ///
    /// The arguments in `accumulator[i]` is passed `aggregate_arguments[i]`
    ///
    /// The argument to each accumulator is itself a `Vec` because
    /// some aggregates such as `CORR` can accept more than one
    /// argument.
    aggregate_arguments: Vec<Vec<Arc<dyn PhysicalExpr>>>,

    /// Optional filter expression to evaluate, one for each for
    /// accumulator. If present, only those rows for which the filter
    /// evaluate to true should be included in the aggregate results.
    ///
    /// For example, for an aggregate like `SUM(x) FILTER (WHERE x >= 100)`,
    /// the filter expression is  `x > 100`.
    filter_expressions: Vec<Option<Arc<dyn PhysicalExpr>>>,

    /// GROUP BY expressions
    group_by: PhysicalGroupBy,

    /// max rows in output RecordBatches
    batch_size: usize,

    /// Optional soft limit on the number of `group_values` in a batch
    /// If the number of `group_values` in a single batch exceeds this value,
    /// the `GroupedHashAggregateStream` operation immediately switches to
    /// output mode and emits all groups.
    group_values_soft_limit: Option<usize>,

    // ========================================================================
    // STATE FLAGS:
    // These fields will be updated during the execution. And control the flow of
    // the execution.
    // ========================================================================
    /// Tracks if this stream is generating input or output
    exec_state: ExecutionState,

    /// Have we seen the end of the input
    input_done: bool,

    // ========================================================================
    // STATE BUFFERS:
    // These fields will accumulate intermediate results during the execution.
    // ========================================================================
    /// An interning store of group keys
    group_values: Box<dyn GroupValues>,

    /// scratch space for the current input [`RecordBatch`] being
    /// processed. Reused across batches here to avoid reallocations
    current_group_indices: Vec<usize>,

    /// Accumulators, one for each `AggregateFunctionExpr` in the query
    ///
    /// For example, if the query has aggregates, `SUM(x)`,
    /// `COUNT(y)`, there will be two accumulators, each one
    /// specialized for that particular aggregate and its input types
    accumulators: Vec<Box<dyn GroupsAccumulator>>,

    // ========================================================================
    // TASK-SPECIFIC STATES:
    // Inner states groups together properties, states for a specific task.
    // ========================================================================
    /// Optional ordering information, that might allow groups to be
    /// emitted from the hash table prior to seeing the end of the
    /// input
    group_ordering: GroupOrdering,

    /// The spill state object
    spill_state: SpillState,

    /// Optional probe for skipping data aggregation, if supported by
    /// current stream.
    skip_aggregation_probe: Option<SkipAggregationProbe>,

    // ========================================================================
    // EXECUTION RESOURCES:
    // Fields related to managing execution resources and monitoring performance.
    // ========================================================================
    /// The memory reservation for this grouping
    reservation: MemoryReservation,

    /// Execution metrics
    baseline_metrics: BaselineMetrics,

    /// The [`RuntimeEnv`] associated with the [`TaskContext`] argument
    runtime: Arc<RuntimeEnv>,

    num_partitions: Option<usize>,

    /// Channels for sending batches from input partitions to output partitions.
    /// Key is the partition number.
    channels: HashMap<
        usize,
        (
            InputPartitionsToCurrentPartitionSender,
            InputPartitionsToCurrentPartitionReceiver,
            SharedMemoryReservation,
        ),
    >,
    partition: usize,
}

impl GroupedHashAggregateStream {

    pub fn new_stream(
        aggr_group_by: &PhysicalGroupBy,
        agg_schema: SchemaRef,
        agg_input: &Arc<dyn ExecutionPlan>,
        agg_filter_expr: &[Option<Arc<dyn PhysicalExpr>>],
        aggr_expr: &[AggregateFunctionExpr],
        agg_mode: &AggregateMode,
        agg_metric: &ExecutionPlanMetricsSet,

        // agg: &AggregateExec,
        context: Arc<TaskContext>,
        partition: usize) -> Result<SendableRecordBatchStream> {
        debug!("Creating GroupedHashAggregateStream");
        // let agg_schema = Arc::clone(&agg.schema);
        // let agg_group_by = agg.group_by.clone();
        // let agg_filter_expr = agg.filter_expr.clone();

        let agg_filter_expr = agg_filter_expr.to_vec();
        // let aggr_expr = aggr_expr.to_vec();

        let num_partition = agg_input.output_partitioning().partition_count();

        println!("num_partition count: {:?}", num_partition);

        let batch_size = context.session_config().batch_size();
        // let input = agg.input.execute(partition, Arc::clone(&context))?;
        let input = agg_input.execute(partition, Arc::clone(&context))?;
        let baseline_metrics = BaselineMetrics::new(agg_metric, partition);

        let timer = baseline_metrics.elapsed_compute().timer();

        // let aggregate_exprs = agg.aggr_expr.clone();
        let aggregate_exprs = aggr_expr.to_vec();

        // arguments for each aggregate, one vec of expressions per
        // aggregate
        let aggregate_arguments = aggregates::aggregate_expressions(
            &aggregate_exprs,
            agg_mode,
            aggr_group_by.expr.len(),
        )?;
        // arguments for aggregating spilled data is the same as the one for final aggregation
        let merging_aggregate_arguments = aggregates::aggregate_expressions(
            &aggregate_exprs,
            &AggregateMode::Final,
            aggr_group_by.expr.len(),
        )?;

        let filter_expressions = match agg.mode {
            AggregateMode::Partial
            | AggregateMode::Single
            | AggregateMode::SinglePartitionedV2
            | AggregateMode::SinglePartitioned => agg_filter_expr,
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                vec![None; aggregate_exprs.len()]
            }
        };

        // Instantiate the accumulators
        let accumulators: Vec<_> = aggregate_exprs
            .iter()
            .map(create_group_accumulator)
            .collect::<Result<_>>()?;

        let group_schema = group_schema(&agg_schema, aggr_group_by.expr.len());
        let spill_expr = group_schema
            .fields
            .into_iter()
            .enumerate()
            .map(|(idx, field)| PhysicalSortExpr {
                expr: Arc::new(Column::new(field.name().as_str(), idx)) as _,
                options: SortOptions::default(),
            })
            .collect();

        let name = format!("GroupedHashAggregateStream[{partition}]");
        let reservation = MemoryConsumer::new(name.clone())
            .with_can_spill(true)
            .register(context.memory_pool());
        let (ordering, _) = agg
            .properties()
            .equivalence_properties()
            .find_longest_permutation(&aggr_group_by.output_exprs());
        let group_ordering = GroupOrdering::try_new(
            &group_schema,
            &agg.input_order_mode,
            ordering.as_slice(),
        )?;

        let num_partitions = context.session_config().target_partitions();
        let fuse_repartition = group_schema.fields().len() > 1
            && num_partitions > 1
            && context.session_config().repartition_aggregations();
        // let fuse_repartition = false;
        let num_partitions = if fuse_repartition {
            Some(num_partitions)
        } else {
            None
        };
        println!("num_partitions: {:?}", num_partitions);
        // let group_len = group_schema.fields().len();
        // println!("group_len: {:?}", group_len);

        let group_values = new_group_values(group_schema, num_partitions)?;
        timer.done();

        let exec_state = ExecutionState::ReadingInput;

        let spill_state = SpillState {
            spills: vec![],
            spill_expr,
            spill_schema: Arc::clone(&agg_schema),
            is_stream_merging: false,
            merging_aggregate_arguments,
            merging_group_by: PhysicalGroupBy::new_single(aggr_group_by.expr.clone()),
        };

        // Skip aggregation is supported if:
        // - aggregation mode is Partial
        // - input is not ordered by GROUP BY expressions,
        //   since Final mode expects unique group values as its input
        // - all accumulators support input batch to intermediate
        //   aggregate state conversion
        // - there is only one GROUP BY expressions set
        let skip_aggregation_probe = if agg.mode == AggregateMode::Partial
            && matches!(group_ordering, GroupOrdering::None)
            && accumulators
                .iter()
                .all(|acc| acc.supports_convert_to_state())
            && aggr_group_by.is_single()
        {
            let options = &context.session_config().options().execution;
            let probe_rows_threshold =
                options.skip_partial_aggregation_probe_rows_threshold;
            let probe_ratio_threshold =
                options.skip_partial_aggregation_probe_ratio_threshold;
            let skipped_aggregation_rows = MetricBuilder::new(&agg.metrics)
                .counter("skipped_aggregation_rows", partition);
            Some(SkipAggregationProbe::new(
                probe_rows_threshold,
                probe_ratio_threshold,
                skipped_aggregation_rows,
            ))
        } else {
            None
        };

        // let input = Arc::clone(&self.input);
        // let lazy_state: LazyState = Default::default();
        // let num_input_partitions = agg.input.output_partitioning().partition_count();
        // println!("num_input_partitions: {:?}", num_input_partitions);
        // let stream = futures::stream::once(async move {
        //     let num_input_partitions = agg.input.output_partitioning().partition_count();
        //     let state = lazy_state
        //         .get_or_init(|| async move {
        //             Mutex::new(GroupRepartitionExecState::new(
        //                 input_captured,
        //                 // partitioning,
        //                 metrics_captured,
        //                 // preserve_order,
        //                 name_captured,
        //                 context_captured,
        //             ))
        //         })
        //         .await;
        // });

        let num_input_partitions = 4;
        let num_output_partitions = 4;
        // create one channel per *output* partition
        // note we use a custom channel that ensures there is always data for each receiver
        // but limits the amount of buffering if required.
        let (txs, rxs) = channels(num_output_partitions);
        // Clone sender for each input partitions
        let txs = txs
            .into_iter()
            .map(|item| vec![item; num_input_partitions])
            .collect::<Vec<_>>();
        let rxs = rxs.into_iter().map(|item| vec![item]).collect::<Vec<_>>();

        let mut channels = HashMap::with_capacity(txs.len());
        for (partition, (tx, rx)) in txs.into_iter().zip(rxs).enumerate() {
            let reservation = Arc::new(Mutex::new(
                MemoryConsumer::new(format!("{}[{partition}]", name))
                    .register(context.memory_pool()),
            ));
            channels.insert(partition, (tx, rx, reservation));
        }

        let i = partition;
        let txs: HashMap<_, _> = channels
            .iter()
            .map(|(partition, (tx, _rx, reservation))| {
                (*partition, (tx[i].clone(), Arc::clone(reservation)))
            })
            .collect();

        // println!("called stream {:?} {:?}", partition, agg.mode);
        let s = GroupedHashAggregateStream {
            schema: agg_schema,
            input,
            mode: agg.mode,
            accumulators,
            aggregate_arguments,
            filter_expressions,
            group_by: aggr_group_by.clone(),
            reservation,
            group_values,
            current_group_indices: Default::default(),
            exec_state,
            baseline_metrics,
            batch_size,
            group_ordering,
            input_done: false,
            runtime: context.runtime_env(),
            spill_state,
            group_values_soft_limit: agg.limit,
            skip_aggregation_probe,
            num_partitions,
            channels,
            partition,
        };

        Ok(Box::pin(s))

    }

    /// Create a new GroupedHashAggregateStream
    pub fn new(
        agg: &AggregateExec,
        context: Arc<TaskContext>,
        partition: usize,
    ) -> Result<Self> {
        debug!("Creating GroupedHashAggregateStream");
        let agg_schema = Arc::clone(&agg.schema);
        let agg_group_by = agg.group_by.clone();
        let agg_filter_expr = agg.filter_expr.clone();

        let num_partition = agg.input.output_partitioning().partition_count();

        println!("num_partition count: {:?}", num_partition);

        let batch_size = context.session_config().batch_size();
        let input = agg.input.execute(partition, Arc::clone(&context))?;
        let baseline_metrics = BaselineMetrics::new(&agg.metrics, partition);

        let timer = baseline_metrics.elapsed_compute().timer();

        let aggregate_exprs = agg.aggr_expr.clone();

        // arguments for each aggregate, one vec of expressions per
        // aggregate
        let aggregate_arguments = aggregates::aggregate_expressions(
            &agg.aggr_expr,
            &agg.mode,
            agg_group_by.expr.len(),
        )?;
        // arguments for aggregating spilled data is the same as the one for final aggregation
        let merging_aggregate_arguments = aggregates::aggregate_expressions(
            &agg.aggr_expr,
            &AggregateMode::Final,
            agg_group_by.expr.len(),
        )?;

        let filter_expressions = match agg.mode {
            AggregateMode::Partial
            | AggregateMode::Single
            | AggregateMode::SinglePartitionedV2
            | AggregateMode::SinglePartitioned => agg_filter_expr,
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                vec![None; agg.aggr_expr.len()]
            }
        };

        // Instantiate the accumulators
        let accumulators: Vec<_> = aggregate_exprs
            .iter()
            .map(create_group_accumulator)
            .collect::<Result<_>>()?;

        let group_schema = group_schema(&agg_schema, agg_group_by.expr.len());
        let spill_expr = group_schema
            .fields
            .into_iter()
            .enumerate()
            .map(|(idx, field)| PhysicalSortExpr {
                expr: Arc::new(Column::new(field.name().as_str(), idx)) as _,
                options: SortOptions::default(),
            })
            .collect();

        let name = format!("GroupedHashAggregateStream[{partition}]");
        let reservation = MemoryConsumer::new(name.clone())
            .with_can_spill(true)
            .register(context.memory_pool());
        let (ordering, _) = agg
            .properties()
            .equivalence_properties()
            .find_longest_permutation(&agg_group_by.output_exprs());
        let group_ordering = GroupOrdering::try_new(
            &group_schema,
            &agg.input_order_mode,
            ordering.as_slice(),
        )?;

        let num_partitions = context.session_config().target_partitions();
        let fuse_repartition = group_schema.fields().len() > 1
            && num_partitions > 1
            && context.session_config().repartition_aggregations();
        // let fuse_repartition = false;
        let num_partitions = if fuse_repartition {
            Some(num_partitions)
        } else {
            None
        };
        println!("num_partitions: {:?}", num_partitions);
        // let group_len = group_schema.fields().len();
        // println!("group_len: {:?}", group_len);

        let group_values = new_group_values(group_schema, num_partitions)?;
        timer.done();

        let exec_state = ExecutionState::ReadingInput;

        let spill_state = SpillState {
            spills: vec![],
            spill_expr,
            spill_schema: Arc::clone(&agg_schema),
            is_stream_merging: false,
            merging_aggregate_arguments,
            merging_group_by: PhysicalGroupBy::new_single(agg_group_by.expr.clone()),
        };

        // Skip aggregation is supported if:
        // - aggregation mode is Partial
        // - input is not ordered by GROUP BY expressions,
        //   since Final mode expects unique group values as its input
        // - all accumulators support input batch to intermediate
        //   aggregate state conversion
        // - there is only one GROUP BY expressions set
        let skip_aggregation_probe = if agg.mode == AggregateMode::Partial
            && matches!(group_ordering, GroupOrdering::None)
            && accumulators
                .iter()
                .all(|acc| acc.supports_convert_to_state())
            && agg_group_by.is_single()
        {
            let options = &context.session_config().options().execution;
            let probe_rows_threshold =
                options.skip_partial_aggregation_probe_rows_threshold;
            let probe_ratio_threshold =
                options.skip_partial_aggregation_probe_ratio_threshold;
            let skipped_aggregation_rows = MetricBuilder::new(&agg.metrics)
                .counter("skipped_aggregation_rows", partition);
            Some(SkipAggregationProbe::new(
                probe_rows_threshold,
                probe_ratio_threshold,
                skipped_aggregation_rows,
            ))
        } else {
            None
        };

        // let input = Arc::clone(&self.input);
        // let lazy_state: LazyState = Default::default();
        // let num_input_partitions = agg.input.output_partitioning().partition_count();
        // println!("num_input_partitions: {:?}", num_input_partitions);
        // let stream = futures::stream::once(async move {
        //     let num_input_partitions = agg.input.output_partitioning().partition_count();
        //     let state = lazy_state
        //         .get_or_init(|| async move {
        //             Mutex::new(GroupRepartitionExecState::new(
        //                 input_captured,
        //                 // partitioning,
        //                 metrics_captured,
        //                 // preserve_order,
        //                 name_captured,
        //                 context_captured,
        //             ))
        //         })
        //         .await;
        // });

        let num_input_partitions = 4;
        let num_output_partitions = 4;
        // create one channel per *output* partition
        // note we use a custom channel that ensures there is always data for each receiver
        // but limits the amount of buffering if required.
        let (txs, rxs) = channels(num_output_partitions);
        // Clone sender for each input partitions
        let txs = txs
            .into_iter()
            .map(|item| vec![item; num_input_partitions])
            .collect::<Vec<_>>();
        let rxs = rxs.into_iter().map(|item| vec![item]).collect::<Vec<_>>();

        let mut channels = HashMap::with_capacity(txs.len());
        for (partition, (tx, rx)) in txs.into_iter().zip(rxs).enumerate() {
            let reservation = Arc::new(Mutex::new(
                MemoryConsumer::new(format!("{}[{partition}]", name))
                    .register(context.memory_pool()),
            ));
            channels.insert(partition, (tx, rx, reservation));
        }

        let i = partition;
        let txs: HashMap<_, _> = channels
            .iter()
            .map(|(partition, (tx, _rx, reservation))| {
                (*partition, (tx[i].clone(), Arc::clone(reservation)))
            })
            .collect();

        // println!("called stream {:?} {:?}", partition, agg.mode);
        Ok(GroupedHashAggregateStream {
            schema: agg_schema,
            input,
            mode: agg.mode,
            accumulators,
            aggregate_arguments,
            filter_expressions,
            group_by: agg_group_by,
            reservation,
            group_values,
            current_group_indices: Default::default(),
            exec_state,
            baseline_metrics,
            batch_size,
            group_ordering,
            input_done: false,
            runtime: context.runtime_env(),
            spill_state,
            group_values_soft_limit: agg.limit,
            skip_aggregation_probe,
            num_partitions,
            channels,
            partition,
        })
    }

    // async fn pull_from_input(
    //     input: Arc<dyn ExecutionPlan>,
    //     partition: usize,
    //     mut output_channels: HashMap<
    //         usize,
    //         (DistributionSender<MaybeBatch>, SharedMemoryReservation),
    //     >,
    //     // partitioning: Partitioning,
    //     // metrics: RepartitionMetrics,
    //     context: Arc<TaskContext>,
    // ) -> Result<()> {
    //     Ok(())
    // }

    // async fn wait_for_task(
    //     input_task: SpawnedTask<Result<()>>,
    //     txs: HashMap<usize, DistributionSender<MaybeBatch>>,
    // ) {
    // }
}

// #[derive(Debug)]
// struct GroupRepartitionExecState {
//     /// Channels for sending batches from input partitions to output partitions.
//     /// Key is the partition number.
//     channels: HashMap<
//         usize,
//         (
//             InputPartitionsToCurrentPartitionSender,
//             InputPartitionsToCurrentPartitionReceiver,
//             SharedMemoryReservation,
//         ),
//     >,

//     /// Helper that ensures that that background job is killed once it is no longer needed.
//     abort_helper: Arc<Vec<SpawnedTask<()>>>,
// }

// impl GroupRepartitionExecState {
//     fn new(
//         input: Arc<dyn ExecutionPlan>,
//         // partitioning: Partitioning,
//         metrics: ExecutionPlanMetricsSet,
//         // preserve_order: bool,
//         name: String,
//         context: Arc<TaskContext>,
//     ) -> Self {
//         let num_input_partitions = input.output_partitioning().partition_count();
//         let num_output_partitions = 4;

//         let (txs, rxs) = channels(num_output_partitions);
//         // Clone sender for each input partitions
//         let txs = txs
//             .into_iter()
//             .map(|item| vec![item; num_input_partitions])
//             .collect::<Vec<_>>();
//         let rxs = rxs.into_iter().map(|item| vec![item]).collect::<Vec<_>>();

//         let mut channels = HashMap::with_capacity(txs.len());
//         for (partition, (tx, rx)) in txs.into_iter().zip(rxs).enumerate() {
//             let reservation = Arc::new(Mutex::new(
//                 MemoryConsumer::new(format!("{}[{partition}]", name))
//                     .register(context.memory_pool()),
//             ));
//             channels.insert(partition, (tx, rx, reservation));
//         }

//         // launch one async task per *input* partition
//         let mut spawned_tasks = Vec::with_capacity(num_input_partitions);
//         for i in 0..num_input_partitions {
//             let txs: HashMap<_, _> = channels
//                 .iter()
//                 .map(|(partition, (tx, _rx, reservation))| {
//                     (*partition, (tx[i].clone(), Arc::clone(reservation)))
//                 })
//                 .collect();

//             let r_metrics = RepartitionMetrics::new(i, num_output_partitions, &metrics);

//             let input_task =
//                 SpawnedTask::spawn(AggregateExec::pull_from_input(
//                     Arc::clone(&input),
//                     i,
//                     txs.clone(),
//                     // partitioning.clone(),
//                     // r_metrics,
//                     Arc::clone(&context),
//                 ));

//             // In a separate task, wait for each input to be done
//             // (and pass along any errors, including panic!s)
//             let wait_for_task =
//                 SpawnedTask::spawn(GroupedHashAggregateStream::wait_for_task(
//                     input_task,
//                     txs.into_iter()
//                         .map(|(partition, (tx, _reservation))| (partition, tx))
//                         .collect(),
//                 ));
//             spawned_tasks.push(wait_for_task);
//         }

//         Self {
//             channels,
//             abort_helper: Arc::new(spawned_tasks),
//         }
//     }
// }

/// Create an accumulator for `agg_expr` -- a [`GroupsAccumulator`] if
/// that is supported by the aggregate, or a
/// [`GroupsAccumulatorAdapter`] if not.
pub(crate) fn create_group_accumulator(
    agg_expr: &AggregateFunctionExpr,
) -> Result<Box<dyn GroupsAccumulator>> {
    if agg_expr.groups_accumulator_supported() {
        agg_expr.create_groups_accumulator()
    } else {
        // Note in the log when the slow path is used
        debug!(
            "Creating GroupsAccumulatorAdapter for {}: {agg_expr:?}",
            agg_expr.name()
        );
        let agg_expr_captured = agg_expr.clone();
        let factory = move || agg_expr_captured.create_accumulator();
        Ok(Box::new(GroupsAccumulatorAdapter::new(factory)))
    }
}

/// Extracts a successful Ok(_) or returns Poll::Ready(Some(Err(e))) with errors
macro_rules! extract_ok {
    ($RES: expr) => {{
        match $RES {
            Ok(v) => v,
            Err(e) => return Poll::Ready(Some(Err(e))),
        }
    }};
}

impl Stream for GroupedHashAggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();

        loop {
            match &self.exec_state {
                ExecutionState::ReadingInputPartitioned(batch) => {
                    // slice off a part of the batch, if needed
                    let output_batch;
                    let size = self.batch_size;
                    (self.exec_state, output_batch) = if batch.num_rows() <= size {
                        (
                            if self.input_done {
                                ExecutionState::Done
                            } else if self.should_skip_aggregation() {
                                ExecutionState::SkippingAggregation
                            } else {
                                ExecutionState::ReadingInput
                            },
                            batch.clone(),
                        )
                    } else {
                        // output first batch_size rows
                        let size = self.batch_size;
                        let num_remaining = batch.num_rows() - size;
                        let remaining = batch.slice(size, num_remaining);
                        let output = batch.slice(0, size);
                        (ExecutionState::ReadingInputPartitioned(remaining), output)
                    };

                    println!("exec_state: {:?}", self.exec_state);
                    println!("output_batch: {:?}", output_batch);

                    return Poll::Ready(Some(Ok(
                        output_batch.record_output(&self.baseline_metrics)
                    )));
                }

                ExecutionState::ReadingInput => 'reading_input: {
                    match ready!(self.input.poll_next_unpin(cx)) {
                        // New batch to aggregate in partial aggregation operator
                        Some(Ok(batch)) if self.mode == AggregateMode::Partial => {
                            let timer = elapsed_compute.timer();
                            let input_rows = batch.num_rows();

                            // Do the grouping
                            extract_ok!(self.group_aggregate_batch(batch));

                            self.update_skip_aggregation_probe(input_rows);

                            // If we can begin emitting rows, do so,
                            // otherwise keep consuming input
                            assert!(!self.input_done);

                            // If the number of group values equals or exceeds the soft limit,
                            // emit all groups and switch to producing output
                            if self.hit_soft_group_limit() {
                                timer.done();
                                // println!("hit soft group");
                                extract_ok!(self.set_input_done_and_produce_output());
                                // make sure the exec_state just set is not overwritten below
                                break 'reading_input;
                            }

                            if let Some(to_emit) = self.group_ordering.emit_to() {
                                let batch = extract_ok!(self.emit(to_emit, false));
                                self.exec_state = ExecutionState::ProducingOutput(batch);
                                timer.done();
                                // make sure the exec_state just set is not overwritten below
                                break 'reading_input;
                            }

                            extract_ok!(self.emit_early_if_necessary());

                            extract_ok!(self.switch_to_skip_aggregation());

                            timer.done();
                        }

                        // New batch to aggregate in terminal aggregation operator
                        // (Final/FinalPartitioned/Single/SinglePartitioned)
                        Some(Ok(batch)) => {
                            let timer = elapsed_compute.timer();

                            // Make sure we have enough capacity for `batch`, otherwise spill
                            extract_ok!(self.spill_previous_if_necessary(&batch));

                            // Do the grouping
                            extract_ok!(self.group_aggregate_batch(batch));

                            // If we can begin emitting rows, do so,
                            // otherwise keep consuming input
                            assert!(!self.input_done);

                            // If the number of group values equals or exceeds the soft limit,
                            // emit all groups and switch to producing output
                            if self.hit_soft_group_limit() {
                                timer.done();
                                extract_ok!(self.set_input_done_and_produce_output());
                                // make sure the exec_state just set is not overwritten below
                                break 'reading_input;
                            }

                            if let Some(to_emit) = self.group_ordering.emit_to() {
                                let batch = extract_ok!(self.emit(to_emit, false));
                                self.exec_state = ExecutionState::ProducingOutput(batch);
                                timer.done();
                                // make sure the exec_state just set is not overwritten below
                                break 'reading_input;
                            }

                            timer.done();
                        }

                        // Found error from input stream
                        Some(Err(e)) => {
                            // inner had error, return to caller
                            return Poll::Ready(Some(Err(e)));
                        }

                        // Found end from input stream
                        None => {
                            // repartition
                            // coalesce
                            // final merge

                            // println!("no batch");

                            // inner is done, emit all rows and switch to producing output
                            extract_ok!(self.set_input_done_and_produce_output());
                        }
                    }
                }

                ExecutionState::SkippingAggregation => {
                    match ready!(self.input.poll_next_unpin(cx)) {
                        Some(Ok(batch)) => {
                            let _timer = elapsed_compute.timer();
                            if let Some(probe) = self.skip_aggregation_probe.as_mut() {
                                probe.record_skipped(&batch);
                            }
                            let states = self.transform_to_states(batch)?;
                            return Poll::Ready(Some(Ok(
                                states.record_output(&self.baseline_metrics)
                            )));
                        }
                        Some(Err(e)) => {
                            // inner had error, return to caller
                            return Poll::Ready(Some(Err(e)));
                        }
                        None => {
                            // inner is done, switching to `Done` state
                            self.exec_state = ExecutionState::Done;
                        }
                    }
                }

                // ExecutionState::ProducingPartitionedOutput(batches) => todo!(""),
                ExecutionState::ProducingOutput(batch) => {
                    // slice off a part of the batch, if needed
                    let output_batch;
                    let size = self.batch_size;
                    (self.exec_state, output_batch) = if batch.num_rows() <= size {
                        (
                            if self.input_done {
                                ExecutionState::Done
                            }
                            // In Partial aggregation, we also need to check
                            // if we should trigger partial skipping
                            else if self.mode == AggregateMode::Partial
                                && self.should_skip_aggregation()
                            {
                                ExecutionState::SkippingAggregation
                            } else {
                                ExecutionState::ReadingInput
                            },
                            batch.clone(),
                        )
                    } else {
                        // output first batch_size rows
                        let size = self.batch_size;
                        let num_remaining = batch.num_rows() - size;
                        let remaining = batch.slice(size, num_remaining);
                        let output = batch.slice(0, size);
                        (ExecutionState::ProducingOutput(remaining), output)
                    };

                    // println!("exec_state: {:?}", self.exec_state);
                    // println!("output_batch: {:?}", output_batch);

                    return Poll::Ready(Some(Ok(
                        output_batch.record_output(&self.baseline_metrics)
                    )));
                }

                ExecutionState::Done => {
                    // release the memory reservation since sending back output batch itself needs
                    // some memory reservation, so make some room for it.
                    self.clear_all();
                    let _ = self.update_memory_reservation();
                    return Poll::Ready(None);
                }
            }
        }
    }
}

impl RecordBatchStream for GroupedHashAggregateStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl GroupedHashAggregateStream {
    /// Perform group-by aggregation for the given [`RecordBatch`].
    fn group_aggregate_batch(&mut self, batch: RecordBatch) -> Result<()> {
        // Evaluate the grouping expressions
        let group_by_values = if self.spill_state.is_stream_merging {
            evaluate_group_by(&self.spill_state.merging_group_by, &batch)?
        } else {
            evaluate_group_by(&self.group_by, &batch)?
        };

        // Evaluate the aggregation expressions.
        let input_values = if self.spill_state.is_stream_merging {
            evaluate_many(&self.spill_state.merging_aggregate_arguments, &batch)?
        } else {
            evaluate_many(&self.aggregate_arguments, &batch)?
        };

        // Evaluate the filter expressions, if any, against the inputs
        let filter_values = if self.spill_state.is_stream_merging {
            let filter_expressions = vec![None; self.accumulators.len()];
            evaluate_optional(&filter_expressions, &batch)?
        } else {
            evaluate_optional(&self.filter_expressions, &batch)?
        };

        for group_values in &group_by_values {
            // println!("mode: {:?} group_values len: {:?}", self.mode, group_values[0].len());

            // calculate the group indices for each input row
            let starting_num_groups = self.group_values.len();
            self.group_values
                .intern(group_values, &mut self.current_group_indices)?;
            let group_indices = &self.current_group_indices;

            // Update ordering information if necessary
            let total_num_groups = self.group_values.len();
            if total_num_groups > starting_num_groups {
                self.group_ordering.new_groups(
                    group_values,
                    group_indices,
                    total_num_groups,
                )?;
            }

            // Gather the inputs to call the actual accumulator
            let t = self
                .accumulators
                .iter_mut()
                .zip(input_values.iter())
                .zip(filter_values.iter());

            for ((acc, values), opt_filter) in t {
                let opt_filter = opt_filter.as_ref().map(|filter| filter.as_boolean());

                // Call the appropriate method on each aggregator with
                // the entire input row and the relevant group indexes
                match self.mode {
                    AggregateMode::Partial
                    | AggregateMode::Single
                    | AggregateMode::SinglePartitioned
                    | AggregateMode::SinglePartitionedV2
                        if !self.spill_state.is_stream_merging =>
                    {
                        // println!("partial values: {:?}", values);
                        // println!("partial group_indices: {:?}", group_indices);
                        acc.update_batch(
                            values,
                            group_indices,
                            opt_filter,
                            total_num_groups,
                        )?;
                    }
                    _ => {
                        if values.len() > 1 {
                            println!("more values: {:?}", values);
                        }
                        // println!("final values: {:?}", values);
                        // println!("final group_indices: {:?}", group_indices);
                        // if aggregation is over intermediate states,
                        // use merge
                        acc.merge_batch(
                            values,
                            group_indices,
                            opt_filter,
                            total_num_groups,
                        )?;
                    }
                }

                // println!("values: {:?}", values);
                // let partitioned_index = self.group_values.get_partitioned_index();
                // println!("group_indices: {:?}", group_indices);
                // println!("partitioned_index: {:?}", partitioned_index);

                // let it = partitioned_index
                //     .into_iter()
                //     .enumerate()
                //     .filter_map(|(partition, indices)| {
                //         let indices: PrimitiveArray<UInt32Type> = indices.into();
                //         (!indices.is_empty()).then_some((partition, indices))
                //     })
                //     .map(|(partition, indices)| {
                //         // Tracking time required for repartitioned batches construction
                //         // let _timer = partitioner_timer.timer();

                //         // println!("indices len: {:?}", indices.len());
                //         // println!("indices: {:?}", indices);
                //         // Produce batches based on indices
                //         let columns = take_arrays(batch.columns(), &indices)?;
                //         // println!("columns: {:?}", columns);

                //         let mut options = RecordBatchOptions::new();
                //         options = options.with_row_count(Some(indices.len()));
                //         let batch = RecordBatch::try_new_with_options(
                //             batch.schema(),
                //             columns,
                //             &options,
                //         )
                //         .unwrap();
                //         // println!("{:?} repartitioned batch: {:?}", partition, batch);

                //         // println!("row2 len: {:?}", batch.num_rows());
                //         Ok::<(usize, arrow_array::RecordBatch), DataFusionError>((
                //             partition, batch,
                //         ))
                //     });

                // for res in it {
                //     // println!("partition: {:?} batch: {:?}", partition, batch);
                //     let (partition, batch) = res?;
                //     let size = batch.get_array_memory_size();

                //     println!("p:{:?} batch: {:?}", partition, batch)
                // }

                //  Ok(())
            }
        }

        match self.update_memory_reservation() {
            // Here we can ignore `insufficient_capacity_err` because we will spill later,
            // but at least one batch should fit in the memory
            Err(DataFusionError::ResourcesExhausted(_))
                if self.group_values.len() >= self.batch_size =>
            {
                Ok(())
            }
            other => other,
        }
    }

    fn update_memory_reservation(&mut self) -> Result<()> {
        let acc = self.accumulators.iter().map(|x| x.size()).sum::<usize>();
        self.reservation.try_resize(
            acc + self.group_values.size()
                + self.group_ordering.size()
                + self.current_group_indices.allocated_size(),
        )
    }

    /// Create an output RecordBatch with the group keys and
    /// accumulator states/values specified in emit_to
    fn emit(&mut self, emit_to: EmitTo, spilling: bool) -> Result<RecordBatch> {
        let schema = if spilling {
            Arc::clone(&self.spill_state.spill_schema)
        } else {
            self.schema()
        };
        if self.group_values.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }

        let mut output = self.group_values.emit(emit_to)?;
        if let EmitTo::First(n) = emit_to {
            self.group_ordering.remove_groups(n);
        }

        // Next output each aggregate value
        for acc in self.accumulators.iter_mut() {
            match self.mode {
                AggregateMode::Partial => output.extend(acc.state(emit_to)?),
                _ if spilling => {
                    // If spilling, output partial state because the spilled data will be
                    // merged and re-evaluated later.
                    output.extend(acc.state(emit_to)?)
                }
                AggregateMode::Final
                | AggregateMode::FinalPartitioned
                | AggregateMode::Single
                | AggregateMode::SinglePartitionedV2
                | AggregateMode::SinglePartitioned => output.push(acc.evaluate(emit_to)?),
            }
        }

        // emit reduces the memory usage. Ignore Err from update_memory_reservation. Even if it is
        // over the target memory size after emission, we can emit again rather than returning Err.
        let _ = self.update_memory_reservation();
        let batch = RecordBatch::try_new(schema, output)?;

        let partitioned_index = self.group_values.get_partitioned_index();

        let it = partitioned_index
            .into_iter()
            .enumerate()
            .filter_map(|(partition, indices)| {
                let indices: PrimitiveArray<UInt32Type> = indices.into();
                (!indices.is_empty()).then_some((partition, indices))
            })
            .map(|(partition, indices)| {
                // Tracking time required for repartitioned batches construction
                // let _timer = partitioner_timer.timer();

                // println!("indices len: {:?}", indices.len());
                // println!("indices: {:?}", indices);
                // Produce batches based on indices
                let columns = take_arrays(batch.columns(), &indices)?;
                // println!("columns: {:?}", columns);

                let mut options = RecordBatchOptions::new();
                options = options.with_row_count(Some(indices.len()));
                let batch =
                    RecordBatch::try_new_with_options(batch.schema(), columns, &options)
                        .unwrap();
                // println!("{:?} repartitioned batch: {:?}", partition, batch);

                // println!("row2 len: {:?}", batch.num_rows());
                Ok::<(usize, arrow_array::RecordBatch), DataFusionError>((
                    partition, batch,
                ))
            });

        let i = self.partition;

        // let mut batches_until_yield = self.num_partitions;
        // while !txs.is_empty() {
        //     for res in it {
        //         // println!("partition: {:?} batch: {:?}", partition, batch);
        //         let (partition, batch) = res?;
        //         let size = batch.get_array_memory_size();

        //         self.send_to_channel(txs, partition, batch)
        //     }
        // }
        // self.send_to_channel(txs, partition, batch)

        Ok(batch)
    }

    // async fn send_to_channel(
    //     &self,
    //     mut output_channels: HashMap<
    //         usize,
    //         (DistributionSender<MaybeBatch>, SharedMemoryReservation),
    //     >,
    //     partition: usize,
    //     batch: RecordBatch,
    // ) {
    //     let mut batches_until_yield = 4;
    //     while !output_channels.is_empty() {
    //         if let Some((tx, _reservation)) = output_channels.get_mut(&partition) {
    //             // reservation.lock().try_grow(size)?;

    //             if tx.send(Some(Ok(batch))).await.is_err() {
    //                 // If the other end has hung up, it was an early shutdown (e.g. LIMIT)
    //                 // reservation.lock().shrink(size);
    //                 output_channels.remove(&partition);
    //             }
    //         }

    //         if batches_until_yield == 0 {
    //             tokio::task::yield_now().await;
    //             batches_until_yield = 4;
    //         } else {
    //             batches_until_yield -= 1;
    //         }
    //     }
    // }

    /// Optimistically, [`Self::group_aggregate_batch`] allows to exceed the memory target slightly
    /// (~ 1 [`RecordBatch`]) for simplicity. In such cases, spill the data to disk and clear the
    /// memory. Currently only [`GroupOrdering::None`] is supported for spilling.
    fn spill_previous_if_necessary(&mut self, batch: &RecordBatch) -> Result<()> {
        // TODO: support group_ordering for spilling
        if self.group_values.len() > 0
            && batch.num_rows() > 0
            && matches!(self.group_ordering, GroupOrdering::None)
            && !self.spill_state.is_stream_merging
            && self.update_memory_reservation().is_err()
        {
            assert_ne!(self.mode, AggregateMode::Partial);
            // Use input batch (Partial mode) schema for spilling because
            // the spilled data will be merged and re-evaluated later.
            self.spill_state.spill_schema = batch.schema();
            self.spill()?;
            self.clear_shrink(batch);
        }
        Ok(())
    }

    /// Emit all rows, sort them, and store them on disk.
    fn spill(&mut self) -> Result<()> {
        let emit = self.emit(EmitTo::All, true)?;
        let sorted = sort_batch(&emit, &self.spill_state.spill_expr, None)?;
        let spillfile = self.runtime.disk_manager.create_tmp_file("HashAggSpill")?;
        // TODO: slice large `sorted` and write to multiple files in parallel
        spill_record_batch_by_size(
            &sorted,
            spillfile.path().into(),
            sorted.schema(),
            self.batch_size,
        )?;
        self.spill_state.spills.push(spillfile);
        Ok(())
    }

    /// Clear memory and shirk capacities to the size of the batch.
    fn clear_shrink(&mut self, batch: &RecordBatch) {
        self.group_values.clear_shrink(batch);
        self.current_group_indices.clear();
        self.current_group_indices.shrink_to(batch.num_rows());
    }

    /// Clear memory and shirk capacities to zero.
    fn clear_all(&mut self) {
        let s = self.schema();
        self.clear_shrink(&RecordBatch::new_empty(s));
    }

    /// Emit if the used memory exceeds the target for partial aggregation.
    /// Currently only [`GroupOrdering::None`] is supported for early emitting.
    /// TODO: support group_ordering for early emitting
    fn emit_early_if_necessary(&mut self) -> Result<()> {
        if self.group_values.len() >= self.batch_size
            && matches!(self.group_ordering, GroupOrdering::None)
            && self.update_memory_reservation().is_err()
        {
            assert_eq!(self.mode, AggregateMode::Partial);
            let n = self.group_values.len() / self.batch_size * self.batch_size;
            let batch = self.emit(EmitTo::First(n), false)?;
            self.exec_state = ExecutionState::ProducingOutput(batch);
        }
        Ok(())
    }

    /// At this point, all the inputs are read and there are some spills.
    /// Emit the remaining rows and create a batch.
    /// Conduct a streaming merge sort between the batch and spilled data. Since the stream is fully
    /// sorted, set `self.group_ordering` to Full, then later we can read with [`EmitTo::First`].
    fn update_merged_stream(&mut self) -> Result<()> {
        let batch = self.emit(EmitTo::All, true)?;
        // clear up memory for streaming_merge
        self.clear_all();
        self.update_memory_reservation()?;
        let mut streams: Vec<SendableRecordBatchStream> = vec![];
        let expr = self.spill_state.spill_expr.clone();
        let schema = batch.schema();
        streams.push(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            futures::stream::once(futures::future::lazy(move |_| {
                sort_batch(&batch, &expr, None)
            })),
        )));
        for spill in self.spill_state.spills.drain(..) {
            let stream = read_spill_as_stream(spill, Arc::clone(&schema), 2)?;
            streams.push(stream);
        }
        self.spill_state.is_stream_merging = true;
        self.input = streaming_merge(
            streams,
            schema,
            &self.spill_state.spill_expr,
            self.baseline_metrics.clone(),
            self.batch_size,
            None,
            self.reservation.new_empty(),
        )?;
        self.input_done = false;
        self.group_ordering = GroupOrdering::Full(GroupOrderingFull::new());
        Ok(())
    }

    /// returns true if there is a soft groups limit and the number of distinct
    /// groups we have seen is over that limit
    fn hit_soft_group_limit(&self) -> bool {
        let Some(group_values_soft_limit) = self.group_values_soft_limit else {
            return false;
        };
        group_values_soft_limit <= self.group_values.len()
    }

    /// common function for signalling end of processing of the input stream
    fn set_input_done_and_produce_output(&mut self) -> Result<()> {
        println!("called set_input_done_and_produce_output");
        self.input_done = true;
        self.group_ordering.input_done();
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();
        let timer = elapsed_compute.timer();
        self.exec_state = if self.spill_state.spills.is_empty() {
            let batch = self.emit(EmitTo::All, false)?;
            // println!("{:?} output batch: {:?}", self.mode, batch);

            if let Some(num_partitions) = self.num_partitions {
                ExecutionState::ReadingInputPartitioned(batch)
            } else {
                ExecutionState::ProducingOutput(batch)
            }
        } else {
            // If spill files exist, stream-merge them.
            self.update_merged_stream()?;
            ExecutionState::ReadingInput
        };
        timer.done();
        Ok(())
    }

    /// Updates skip aggregation probe state.
    ///
    /// Notice: It should only be called in Partial aggregation
    fn update_skip_aggregation_probe(&mut self, input_rows: usize) {
        if let Some(probe) = self.skip_aggregation_probe.as_mut() {
            // Skip aggregation probe is not supported if stream has any spills,
            // currently spilling is not supported for Partial aggregation
            assert!(self.spill_state.spills.is_empty());
            probe.update_state(input_rows, self.group_values.len());
        };
    }

    /// In case the probe indicates that aggregation may be
    /// skipped, forces stream to produce currently accumulated output.
    ///
    /// Notice: It should only be called in Partial aggregation
    fn switch_to_skip_aggregation(&mut self) -> Result<()> {
        if let Some(probe) = self.skip_aggregation_probe.as_mut() {
            if probe.should_skip() {
                let batch = self.emit(EmitTo::All, false)?;
                self.exec_state = ExecutionState::ProducingOutput(batch);
            }
        }

        Ok(())
    }

    /// Returns true if the aggregation probe indicates that aggregation
    /// should be skipped.
    ///
    /// Notice: It should only be called in Partial aggregation
    fn should_skip_aggregation(&self) -> bool {
        self.skip_aggregation_probe
            .as_ref()
            .is_some_and(|probe| probe.should_skip())
    }

    /// Transforms input batch to intermediate aggregate state, without grouping it
    fn transform_to_states(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let group_values = evaluate_group_by(&self.group_by, &batch)?;
        let input_values = evaluate_many(&self.aggregate_arguments, &batch)?;
        let filter_values = evaluate_optional(&self.filter_expressions, &batch)?;

        let mut output = group_values.first().cloned().ok_or_else(|| {
            internal_datafusion_err!("group_values expected to have at least one element")
        })?;

        let iter = self
            .accumulators
            .iter()
            .zip(input_values.iter())
            .zip(filter_values.iter());

        for ((acc, values), opt_filter) in iter {
            let opt_filter = opt_filter.as_ref().map(|filter| filter.as_boolean());
            output.extend(acc.convert_to_state(values, opt_filter)?);
        }

        let states_batch = RecordBatch::try_new(self.schema(), output)?;

        Ok(states_batch)
    }
}

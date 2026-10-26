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

//! Spilling for a `Partitioned` [`HashJoinExec`](super::HashJoinExec).
//!
//! A partition starts exactly like an unspilled one: it buffers the build side
//! and builds one hash table. Only when the memory pool refuses that (see
//! [`BuildOutcome::Overflow`]) does it switch to this module, which splits the
//! build rows into buckets by a hash of the join keys and writes the largest
//! buckets to disk until the rest fits. Probe rows of a bucket kept in memory
//! are joined right away; probe rows of a spilled bucket go to that bucket's
//! probe file. Afterwards every spilled pair of files is joined on its own.
//!
//! Every key's build and probe rows land in the same bucket, so each bucket
//! pair is just a smaller hash join, run by the same [`HashJoinStream`] as an
//! unspilled join: unmatched rows, marks, join filters and NULL keys need no
//! code here. A pair that is still too big is split again with the next
//! level's hash seed.
//!
//! Memory: from the first refusal on, the partition keeps everything it holds
//! in a [`MergeMemoryPool`] and gives nothing back before it finishes. Joining
//! the spilled pairs runs on that kept memory, so it does not ask the pool for
//! more at a point where it could no longer react by spilling.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use arrow::array::{BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow::compute::{
    SortOptions, concat_batches, filter_record_batch, take_record_batch,
};
use arrow::datatypes::SchemaRef;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::utils::memory::get_record_batch_memory_size;
use datafusion_common::{
    JoinType, NullEquality, Result, internal_datafusion_err, resources_err,
};
use datafusion_execution::memory_pool::{
    MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation, MergeMemoryPool,
};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_execution::{SpillFile, TaskContext};
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;
use futures::future::BoxFuture;
use futures::{FutureExt, Stream, StreamExt, TryStreamExt, ready, stream};
use parking_lot::Mutex;

use super::exec::{
    BuildOutcome, BuildOverflow, BuildParams, JoinLeftData, SpillTrigger,
    build_table_bytes, is_resources_exhausted,
};
use super::shared_bounds::SharedBuildAccumulator;
use super::stream::{
    BuildSide, BuildSideInitialState, HashJoinStream, HashJoinStreamState,
};
use crate::joins::PartitionMode;
use crate::joins::sort_merge_join::{SortMergeJoinInputs, sort_merge_join_stream};
use crate::joins::utils::{
    BuildProbeJoinMetrics, ColumnIndex, JoinFilter, OnceFut, build_join_schema,
};
use crate::metrics::ExecutionPlanMetricsSet;
use crate::sorts::sort::ExternalSorter;
use crate::spill::in_progress_spill_file::InProgressSpillFile;
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;
use crate::{EmptyRecordBatchStream, RecordBatchStream, SendableRecordBatchStream};

const BUCKET_BITS: u32 = 4;
const BUCKETS: usize = 1 << BUCKET_BITS;
/// Levels of splitting after which a pair is joined in memory regardless
const MAX_LEVEL: usize = 4;
/// Differs from the join's and `RepartitionExec`'s seeds, so the rows of one
/// partition spread over all buckets
const BUCKET_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// What creates the [`HashJoinStream`] of the whole partition or of one bucket pair
#[derive(Clone)]
pub(super) struct StreamParams {
    pub(super) partition: usize,
    pub(super) schema: SchemaRef,
    pub(super) on_right: Vec<PhysicalExprRef>,
    pub(super) filter: Option<JoinFilter>,
    pub(super) join_type: JoinType,
    pub(super) random_state: RandomState,
    pub(super) join_metrics: BuildProbeJoinMetrics,
    pub(super) column_indices: Vec<ColumnIndex>,
    pub(super) null_equality: NullEquality,
    pub(super) batch_size: usize,
}

impl StreamParams {
    fn stream(
        &self,
        left_data: JoinLeftData,
        right: SendableRecordBatchStream,
        accumulator: Option<Arc<SharedBuildAccumulator>>,
        right_side_ordered: bool,
        fetch: Option<usize>,
    ) -> HashJoinStream {
        HashJoinStream::new(
            self.partition,
            Arc::clone(&self.schema),
            self.on_right.clone(),
            self.filter.clone(),
            self.join_type,
            right,
            self.random_state.clone(),
            self.join_metrics.clone(),
            self.column_indices.clone(),
            self.null_equality,
            HashJoinStreamState::WaitBuildSide,
            BuildSide::Initial(BuildSideInitialState {
                left_fut: OnceFut::new(async move { Ok(left_data) }),
            }),
            self.batch_size,
            vec![],
            right_side_ordered,
            accumulator,
            PartitionMode::Partitioned,
            fetch,
        )
    }
}

/// Everything one spilling partition needs
pub(super) struct SpillContext {
    pub(super) stream: StreamParams,
    pub(super) build: BuildParams,
    pub(super) build_spill: SpillManager,
    pub(super) probe_spill: SpillManager,
    pub(super) pool: Arc<dyn MemoryPool>,
    pub(super) context: Arc<TaskContext>,
    pub(super) trigger: SpillTrigger,
    pub(super) fetch: Option<usize>,
    pub(super) right_side_ordered: bool,
}

/// The stream of a partition that can spill: today's [`HashJoinStream`] if
/// the build side fits, the bucketed join otherwise.
pub(super) fn spillable_stream(
    ctx: SpillContext,
    build: BoxFuture<'static, Result<BuildOutcome>>,
    right: SendableRecordBatchStream,
    accumulator: Option<Arc<SharedBuildAccumulator>>,
) -> SendableRecordBatchStream {
    let schema = Arc::clone(&ctx.stream.schema);
    let ctx = Arc::new(ctx);
    let output = stream::once(async move {
        let stream: SendableRecordBatchStream = match build.await? {
            BuildOutcome::InMemory(left_data) => Box::pin(ctx.stream.stream(
                left_data,
                right,
                accumulator,
                ctx.right_side_ordered,
                ctx.fetch,
            )),
            BuildOutcome::Overflow(overflow) => {
                // The kept buckets cannot answer for the spilled ones: let every
                // probe row of this partition through.
                if let Some(accumulator) = &accumulator {
                    accumulator.report_canceled_partition(ctx.stream.partition);
                }
                let fetch = ctx.fetch;
                limit(spill_join(ctx, overflow, right).await?, fetch)
            }
        };
        Ok::<_, datafusion_common::DataFusionError>(stream)
    })
    .try_flatten();
    Box::pin(RecordBatchStreamAdapter::new(schema, output))
}

/// Joins a partition whose build side overflowed
async fn spill_join(
    ctx: Arc<SpillContext>,
    overflow: BuildOverflow,
    right: SendableRecordBatchStream,
) -> Result<SendableRecordBatchStream> {
    let BuildOverflow {
        batches,
        mut rest,
        reservation,
    } = overflow;
    let kept = Arc::new(KeptMemory::take_over(&ctx, reservation));

    // The collected batches are covered by the reservation taken over: release
    // each one's share once its rows are copied into the buckets.
    let mut build = BucketedBuild::new(0);
    let mut collected = kept.buffers.size();
    for batch in batches {
        build.push(&ctx, &kept, &batch).await?;
        let released = get_record_batch_memory_size(&batch).min(collected);
        kept.buffers.shrink(released);
        collected -= released;
    }
    kept.buffers.shrink(collected);
    while let Some(batch) = rest.try_next().await? {
        build.push(&ctx, &kept, &batch).await?;
    }
    drop(rest);

    let (mut resident, mut spilled) = build.finish(&ctx, &kept).await?;
    let mut reservation = kept.buffers.split(kept.buffers.size());
    let left_data = match ctx.build.build(&mut resident, &mut reservation) {
        Ok(left_data) => left_data,
        // The table took more than estimated and the pool had no more: the
        // build side can still spill, so spill the kept buckets too
        Err(e) if is_resources_exhausted(&e) => {
            reservation.free();
            let mut files: Vec<Option<BucketFile>> = (0..BUCKETS).map(|_| None).collect();
            let mut sides: Vec<SpilledSide> =
                (0..BUCKETS).map(|_| SpilledSide::default()).collect();
            for batch in std::mem::take(&mut resident) {
                for (bucket, part) in split(&batch, &ctx.build.on_left, 0)? {
                    let file = match &mut files[bucket] {
                        Some(file) => file,
                        slot => slot.insert(BucketFile::try_new(
                            &ctx.build_spill,
                            ctx.stream.batch_size,
                            "HashJoin build bucket",
                            kept.buffers.new_empty(),
                        )?),
                    };
                    file.append(&part)?;
                    sides[bucket].rows += part.num_rows();
                    sides[bucket].bytes += get_record_batch_memory_size(&part);
                }
            }
            for (bucket, (file, mut side)) in files.into_iter().zip(sides).enumerate() {
                if let Some(file) = file {
                    side.file = file.finish()?;
                    spilled.push((bucket, side));
                }
            }
            ctx.build
                .build(&mut vec![], &mut kept.buffers.new_empty())
                .map_err(|e| e.context("PoC: empty resident build"))?
        }
        Err(e) => return Err(e),
    };
    // Joining a spilled pair needs at most what its largest bucket takes:
    // keep that, and let the rest go back to the pool once the kept buckets
    // are joined, so operators that start later are not starved.
    let largest = spilled
        .iter()
        .map(|(_, side)| join_bytes(side.rows, side.bytes))
        .max()
        .unwrap_or(0);
    kept.keep_for_spilled_pairs(largest);

    let mut spilled_buckets = [false; BUCKETS];
    for (bucket, _) in &spilled {
        spilled_buckets[*bucket] = true;
    }
    let probe_files = Arc::new(Mutex::new(empty_sides()));
    let router = SharedRouter(Arc::new(Mutex::new(ProbeRouter {
        input: right,
        on_right: ctx.stream.on_right.clone(),
        spill: ctx.probe_spill.clone(),
        level: 0,
        spilled: spilled_buckets,
        batch_size: ctx.stream.batch_size,
        memory: kept.buffers.new_empty(),
        writers: (0..BUCKETS).map(|_| None).collect(),
        sides: empty_sides(),
        output: Arc::clone(&probe_files),
        finished: false,
    })));
    let resident_join =
        ctx.stream
            .stream(left_data, Box::pin(router.clone()), None, false, None);

    // The join of the kept buckets stops reading probe rows once its result
    // is complete (e.g. an inner join with no rows kept), but the spilled
    // buckets still need theirs: route the rest of the probe side first. The
    // probe rows of kept buckets read here cannot change that result.
    let schema = Arc::clone(&ctx.stream.schema);
    let spilled_joins = stream::once(async move {
        router.drain().await?;
        let joins = stream::iter(spilled)
            .then(move |(bucket, build)| {
                let probe = probe_files.lock()[bucket].take();
                join_pair(
                    Arc::clone(&ctx),
                    Arc::clone(&kept),
                    build,
                    probe.unwrap_or_default(),
                    0,
                )
            })
            .try_flatten();
        Ok::<_, datafusion_common::DataFusionError>(joins)
    })
    .try_flatten();
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema,
        resident_join.chain(spilled_joins),
    )))
}

/// The probe router, shared by the join of the kept buckets and the drain
/// that runs after it
#[derive(Clone)]
struct SharedRouter(Arc<Mutex<ProbeRouter>>);

impl SharedRouter {
    async fn drain(self) -> Result<()> {
        futures::future::poll_fn(|cx| {
            let mut router = self.0.lock();
            loop {
                match ready!(router.poll_next_unpin(cx)) {
                    Some(Ok(_)) => continue,
                    Some(Err(e)) => return Poll::Ready(Err(e)),
                    None => return Poll::Ready(Ok(())),
                }
            }
        })
        .await
    }
}

impl Stream for SharedRouter {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.0.lock().poll_next_unpin(cx)
    }
}

impl RecordBatchStream for SharedRouter {
    fn schema(&self) -> SchemaRef {
        self.0.lock().input.schema()
    }
}

/// Memory a spilling partition holds from the first refusal until it finishes
struct KeptMemory {
    pool: Arc<MergeMemoryPool>,
    /// The bucket buffers while collecting the build side
    buffers: MemoryReservation,
    trigger: SpillTrigger,
    /// What joining one spilled pair may use, set once the build side is split
    for_spilled_pairs: AtomicUsize,
}

impl KeptMemory {
    /// Moves what the build side held into a pool that keeps it
    fn take_over(ctx: &SpillContext, held: MemoryReservation) -> Self {
        let partition = ctx.stream.partition;
        let pool = Arc::new(MergeMemoryPool::new(
            Arc::clone(&ctx.pool),
            MemoryConsumer::new(format!("HashJoinSpill[{partition}]"))
                .with_can_spill(true),
        ));
        let buffers = MemoryConsumer::new(format!("HashJoinSpillBuffers[{partition}]"))
            .register(&(Arc::clone(&pool) as Arc<dyn MemoryPool>));
        let bytes = held.size();
        drop(held);
        // Another consumer may take the bytes just released first; the rows are
        // then re-accounted as they are copied into buckets
        let _ = buffers.try_grow(bytes);
        pool.retain(pool.reserved());
        Self {
            pool,
            buffers,
            trigger: ctx.trigger,
            for_spilled_pairs: AtomicUsize::new(0),
        }
    }

    /// Grows `reservation`, a child of the kept pool, while the build side is
    /// split, unless the execution pool or the test knob refuses. What it gets
    /// is kept until the spilled pairs are joined.
    fn try_grow(&self, reservation: &MemoryReservation, bytes: usize) -> bool {
        if self.trigger.refuses(reservation.size(), bytes) {
            return false;
        }
        let granted = reservation.try_grow(bytes).is_ok();
        if granted {
            self.pool.retain(self.pool.reserved());
        }
        granted
    }

    /// Keeps at most `bytes` once the memory in use is released
    fn keep_for_spilled_pairs(&self, bytes: usize) {
        let bytes = bytes.min(self.pool.reserved());
        self.pool.retain(bytes);
        self.for_spilled_pairs.store(bytes, Ordering::Relaxed);
    }

    /// What joining one spilled pair may use without asking the execution pool
    fn capacity(&self) -> usize {
        self.trigger
            .cap
            .unwrap_or_else(|| self.for_spilled_pairs.load(Ordering::Relaxed))
    }
}

/// Bytes to hold a bucket's rows, a concatenated copy, and its hash table
fn join_bytes(rows: usize, bytes: usize) -> usize {
    bytes
        .saturating_mul(2)
        .saturating_add(build_table_bytes(rows))
        .saturating_add(rows.div_ceil(8))
}

/// Most bytes a bucket file buffers before writing
const BUCKET_FILE_BUFFER_BYTES: usize = 1024 * 1024;

/// One bucket's spill file. A split batch leaves each bucket only about
/// 1/16 of its rows, so rows are buffered and written in batches of
/// `batch_size` rows: fewer, larger writes, and full batches when joined.
/// The buffered rows are charged to the kept memory; when it refuses them,
/// they are written right away.
struct BucketFile {
    file: InProgressSpillFile,
    schema: SchemaRef,
    buffered: Vec<RecordBatch>,
    buffered_rows: usize,
    batch_size: usize,
    reservation: MemoryReservation,
}

impl BucketFile {
    fn try_new(
        spill: &SpillManager,
        batch_size: usize,
        request: &str,
        reservation: MemoryReservation,
    ) -> Result<Self> {
        Ok(Self {
            file: spill.create_in_progress_file(request)?,
            schema: Arc::clone(spill.schema()),
            buffered: vec![],
            buffered_rows: 0,
            batch_size,
            reservation,
        })
    }

    fn append(&mut self, batch: &RecordBatch) -> Result<()> {
        let granted = self
            .reservation
            .try_grow(get_record_batch_memory_size(batch))
            .is_ok();
        self.buffered_rows += batch.num_rows();
        self.buffered.push(batch.clone());
        if !granted
            || self.buffered_rows >= self.batch_size
            || self.reservation.size() >= BUCKET_FILE_BUFFER_BYTES
        {
            self.write_buffered()?;
        }
        Ok(())
    }

    fn write_buffered(&mut self) -> Result<()> {
        let batch = match self.buffered.len() {
            0 => return Ok(()),
            1 => self.buffered.pop().expect("one batch"),
            _ => concat_batches(&self.schema, &self.buffered)?,
        };
        self.buffered.clear();
        self.buffered_rows = 0;
        self.file.append_batch(&batch)?;
        self.reservation.free();
        Ok(())
    }

    fn finish(mut self) -> Result<Option<Arc<dyn SpillFile>>> {
        self.write_buffered()?;
        self.file.finish()
    }
}

/// One side of a spilled bucket
#[derive(Default)]
struct SpilledSide {
    file: Option<Arc<dyn SpillFile>>,
    rows: usize,
    bytes: usize,
}

fn empty_sides() -> Vec<Option<SpilledSide>> {
    (0..BUCKETS).map(|_| None).collect()
}

enum Bucket {
    Memory {
        batches: Vec<RecordBatch>,
        rows: usize,
        bytes: usize,
    },
    Disk {
        file: BucketFile,
        rows: usize,
        bytes: usize,
    },
}

impl Bucket {
    fn memory_bytes(&self) -> Option<usize> {
        match self {
            Bucket::Memory { bytes, .. } => Some(*bytes),
            Bucket::Disk { .. } => None,
        }
    }
}

/// The build side of a spilling partition, split into buckets
struct BucketedBuild {
    level: usize,
    buckets: Vec<Bucket>,
}

impl BucketedBuild {
    fn new(level: usize) -> Self {
        let buckets = (0..BUCKETS)
            .map(|_| Bucket::Memory {
                batches: vec![],
                rows: 0,
                bytes: 0,
            })
            .collect();
        Self { level, buckets }
    }

    async fn push(
        &mut self,
        ctx: &SpillContext,
        kept: &KeptMemory,
        batch: &RecordBatch,
    ) -> Result<()> {
        for (bucket, part) in split(batch, &ctx.build.on_left, self.level)? {
            let part_bytes = get_record_batch_memory_size(&part);
            loop {
                match &mut self.buckets[bucket] {
                    Bucket::Disk { file, rows, bytes } => {
                        file.append(&part)?;
                        *rows += part.num_rows();
                        *bytes += part_bytes;
                        break;
                    }
                    Bucket::Memory {
                        batches,
                        rows,
                        bytes,
                    } => {
                        if kept.try_grow(&kept.buffers, part_bytes) {
                            *rows += part.num_rows();
                            *bytes += part_bytes;
                            batches.push(part);
                            break;
                        }
                    }
                }
                // Refused: write the largest bucket in memory to disk and retry
                let victim = self.largest_in_memory().unwrap_or(bucket);
                self.destage(ctx, kept, victim).await?;
            }
        }
        Ok(())
    }

    /// The bucket in memory holding the most bytes, if any holds some
    fn largest_in_memory(&self) -> Option<usize> {
        (0..BUCKETS)
            .filter_map(|b| self.buckets[b].memory_bytes().map(|bytes| (bytes, b)))
            .filter(|(bytes, _)| *bytes > 0)
            .max()
            .map(|(_, b)| b)
    }

    async fn destage(
        &mut self,
        ctx: &SpillContext,
        kept: &KeptMemory,
        bucket: usize,
    ) -> Result<()> {
        let Bucket::Memory {
            batches,
            rows,
            bytes,
        } = std::mem::replace(
            &mut self.buckets[bucket],
            Bucket::Memory {
                batches: vec![],
                rows: 0,
                bytes: 0,
            },
        )
        else {
            unreachable!("only buckets in memory are destaged")
        };
        let mut file = BucketFile::try_new(
            &ctx.build_spill,
            ctx.stream.batch_size,
            "HashJoin build bucket",
            kept.buffers.new_empty(),
        )?;
        for batch in &batches {
            file.append(batch)?;
        }
        drop(batches);
        // Back to the kept pool, not to the execution pool
        kept.buffers.shrink(bytes);
        self.buckets[bucket] = Bucket::Disk { file, rows, bytes };
        Ok(())
    }

    /// Makes room for the hash table of the buckets kept in memory, then
    /// returns their rows and the spilled build sides
    async fn finish(
        mut self,
        ctx: &SpillContext,
        kept: &KeptMemory,
    ) -> Result<(Vec<RecordBatch>, Vec<(usize, SpilledSide)>)> {
        loop {
            let (rows, bytes) = self.buckets.iter().fold((0, 0), |acc, b| match b {
                Bucket::Memory { rows, bytes, .. } => (acc.0 + rows, acc.1 + bytes),
                Bucket::Disk { .. } => acc,
            });
            if rows == 0 {
                break;
            }
            // What the table adds on top of the rows held already
            let probe = kept.buffers.new_empty();
            if kept.try_grow(&probe, join_bytes(rows, bytes) - bytes) {
                break;
            }
            let victim = self.largest_in_memory().expect("rows are in memory");
            self.destage(ctx, kept, victim).await?;
        }

        let mut resident = vec![];
        let mut spilled = vec![];
        for (bucket, b) in self.buckets.into_iter().enumerate() {
            match b {
                Bucket::Memory { batches, .. } => resident.extend(batches),
                Bucket::Disk { file, rows, bytes } => spilled.push((
                    bucket,
                    SpilledSide {
                        file: file.finish()?,
                        rows,
                        bytes,
                    },
                )),
            }
        }
        Ok((resident, spilled))
    }
}

/// Joins one spilled bucket pair, splitting it again if it does not fit in
/// the kept memory
fn join_pair(
    ctx: Arc<SpillContext>,
    kept: Arc<KeptMemory>,
    build: SpilledSide,
    probe: SpilledSide,
    level: usize,
) -> BoxFuture<'static, Result<SendableRecordBatchStream>> {
    async move {
        let fits = join_bytes(build.rows, build.bytes) <= kept.capacity();
        if build.rows == 0 || fits || level >= MAX_LEVEL {
            match join_in_memory(&ctx, &kept, &build, &probe).await {
                Ok(stream) => return Ok(stream),
                // The estimate fell short: split further rather than fail
                Err(e)
                    if build.rows > 0
                        && level < MAX_LEVEL
                        && is_resources_exhausted(&e) => {}
                // Splitting cannot shrink this pair any further
                Err(e) if is_resources_exhausted(&e) => {
                    return join_sort_merge(&ctx, &kept, &build, &probe)
                        .await
                        .map_err(|e| {
                            e.context(format!(
                                "PoC: sort-merge joining a pair at level {level}: {} build rows",
                                build.rows
                            ))
                        });
                }
                Err(e) => {
                    return Err(e.context(format!(
                        "PoC: joining a spilled pair at level {level}: {} build rows, {} bytes, capacity {}",
                        build.rows,
                        build.bytes,
                        kept.capacity()
                    )));
                }
            }
        }

        let level = level + 1;
        let batch_size = ctx.stream.batch_size;
        let builds =
            resplit(
                &ctx.build_spill,
                batch_size,
                &kept.buffers,
                &ctx.build.on_left,
                build,
                level,
            )
                .await?;
        let probes =
            resplit(
                &ctx.probe_spill,
                batch_size,
                &kept.buffers,
                &ctx.stream.on_right,
                probe,
                level,
            )
                .await?;
        let total_rows: usize = builds.iter().map(|b| b.rows).sum();
        let schema = Arc::clone(&ctx.stream.schema);
        let pairs = builds.into_iter().zip(probes).filter_map(move |(b, p)| {
            // A child left with nearly all rows is one hot key: splitting
            // further cannot shrink it, so join it as it is
            let skewed = b.rows * 10 >= total_rows * 9 && b.rows > 0;
            (b.rows + p.rows > 0).then_some((b, p, if skewed { MAX_LEVEL } else { level }))
        });
        let joins = stream::iter(pairs)
            .then(move |(b, p, level)| {
                join_pair(Arc::clone(&ctx), Arc::clone(&kept), b, p, level)
            })
            .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, joins))
            as SendableRecordBatchStream)
    }
    .boxed()
}

async fn join_in_memory(
    ctx: &SpillContext,
    kept: &KeptMemory,
    build: &SpilledSide,
    probe: &SpilledSide,
) -> Result<SendableRecordBatchStream> {
    let mut reservation = kept.buffers.new_empty();
    let mut batches = vec![];
    if let Some(file) = &build.file {
        let mut rows = ctx
            .build_spill
            .read_spill_as_stream(Arc::clone(file), None)?;
        while let Some(batch) = rows.try_next().await? {
            // Within the kept memory unless the pair cannot be split further,
            // which then needs the pool to grant the rest
            reservation.try_grow(get_record_batch_memory_size(&batch))?;
            batches.push(batch);
        }
    }
    let left_data = ctx.build.build(&mut batches, &mut reservation)?;
    let probe_rows = match &probe.file {
        Some(file) => ctx
            .probe_spill
            .read_spill_as_stream(Arc::clone(file), None)?,
        None => Box::pin(EmptyRecordBatchStream::new(Arc::clone(
            ctx.probe_spill.schema(),
        ))),
    };
    Ok(Box::pin(
        ctx.stream.stream(left_data, probe_rows, None, false, None),
    ))
}

/// Joins a pair that splitting cannot shrink (one hot key, or the depth cap)
/// and that does not fit, as a sort-merge join: its sorts and its buffered key
/// groups spill, so it finishes in whatever memory it gets. It draws on the
/// kept memory first.
async fn join_sort_merge(
    ctx: &SpillContext,
    kept: &KeptMemory,
    build: &SpilledSide,
    probe: &SpilledSide,
) -> Result<SendableRecordBatchStream> {
    // The rung's own budget, split so the sorts spill early and leave the
    // sort-merge join room for what it holds in memory regardless (the join
    // keys of a buffered key group, a batch read back from its spill)
    let budget = kept.capacity().max(RUNG_MIN_BYTES);
    let kept_pool = Arc::clone(&kept.pool) as Arc<dyn MemoryPool>;
    let capped = |cap| -> Result<Arc<TaskContext>> {
        rung_context(
            &ctx.context,
            Arc::new(CappedPool::new(Arc::clone(&kept_pool), cap)),
        )
    };
    let partition = ctx.stream.partition;
    let sort_options = vec![SortOptions::default(); ctx.build.on_left.len()];
    let left = sort_side(
        &capped(budget / 4)?,
        partition,
        &ctx.build_spill,
        build,
        &ctx.build.on_left,
        &sort_options,
        budget / 4,
    )
    .await?;
    let right = sort_side(
        &capped(budget / 4)?,
        partition,
        &ctx.probe_spill,
        probe,
        &ctx.stream.on_right,
        &sort_options,
        budget / 4,
    )
    .await?;
    let context = capped(budget / 2)?;
    let (join_schema, columns) = build_join_schema(
        ctx.build_spill.schema(),
        ctx.probe_spill.schema(),
        &ctx.stream.join_type,
    );
    let joined = sort_merge_join_stream(
        SortMergeJoinInputs {
            schema: Arc::new(join_schema),
            sort_options,
            null_equality: ctx.stream.null_equality,
            left,
            right,
            on_left: ctx.build.on_left.clone(),
            on_right: ctx.stream.on_right.clone(),
            filter: ctx.stream.filter.clone(),
            join_type: ctx.stream.join_type,
            partition,
        },
        &ExecutionPlanMetricsSet::new(),
        &context,
    )?;
    // The hash join's output columns, in its order
    let projection = ctx
        .stream
        .column_indices
        .iter()
        .map(|column| {
            columns.iter().position(|c| c == column).ok_or_else(|| {
                internal_datafusion_err!(
                    "join output column {column:?} not in the join schema"
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let schema = Arc::clone(&ctx.stream.schema);
    let output = Arc::clone(&schema);
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema,
        joined.map(move |batch| {
            let batch = batch?;
            let columns = projection
                .iter()
                .map(|&i| Arc::clone(batch.column(i)))
                .collect();
            let options =
                RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
            Ok(RecordBatch::try_new_with_options(
                Arc::clone(&output),
                columns,
                &options,
            )?)
        }),
    )))
}

/// Least memory the sort-merge rung asks for, even if less is kept
const RUNG_MIN_BYTES: usize = 8 * 1024 * 1024;

/// A task context like `task` whose memory pool is `pool`
fn rung_context(
    task: &TaskContext,
    pool: Arc<dyn MemoryPool>,
) -> Result<Arc<TaskContext>> {
    let runtime = RuntimeEnvBuilder::from_runtime_env(&task.runtime_env())
        .with_memory_pool(pool)
        .build_arc()?;
    Ok(Arc::new(TaskContext::new(
        task.task_id(),
        task.session_id(),
        task.session_config().clone(),
        task.scalar_functions().clone(),
        task.higher_order_functions().clone(),
        task.aggregate_functions().clone(),
        task.window_functions().clone(),
        runtime,
    )))
}

/// A child of the kept memory that refuses growth past `cap`, so a sort in
/// the sort-merge rung spills within its share instead of taking all the pool
/// has. Infallible growth (the sort-merge join's) is counted but not refused.
#[derive(Debug)]
struct CappedPool {
    inner: Arc<dyn MemoryPool>,
    cap: usize,
    used: AtomicUsize,
}

impl CappedPool {
    fn new(inner: Arc<dyn MemoryPool>, cap: usize) -> Self {
        Self {
            inner,
            cap,
            used: AtomicUsize::new(0),
        }
    }
}

impl std::fmt::Display for CappedPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CappedPool({})", self.cap)
    }
}

impl MemoryPool for CappedPool {
    fn name(&self) -> &str {
        "CappedPool"
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.used.fetch_add(additional, Ordering::Relaxed);
        self.inner.grow(reservation, additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.used.fetch_sub(shrink, Ordering::Relaxed);
        self.inner.shrink(reservation, shrink);
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        let used = self.used.load(Ordering::Relaxed);
        if used + additional > self.cap {
            return resources_err!(
                "sort-merge rung budget of {} bytes: {used} used, {additional} more requested",
                self.cap
            );
        }
        self.inner.try_grow(reservation, additional)?;
        self.used.fetch_add(additional, Ordering::Relaxed);
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    fn memory_limit(&self) -> MemoryLimit {
        MemoryLimit::Finite(self.cap)
    }
}

/// One spilled side, sorted on the join keys by an external sort
async fn sort_side(
    context: &Arc<TaskContext>,
    partition: usize,
    spill: &SpillManager,
    side: &SpilledSide,
    on: &[PhysicalExprRef],
    options: &[SortOptions],
    budget: usize,
) -> Result<SendableRecordBatchStream> {
    let schema = Arc::clone(spill.schema());
    let Some(file) = &side.file else {
        return Ok(Box::pin(EmptyRecordBatchStream::new(schema)));
    };
    let ordering = LexOrdering::new(
        on.iter()
            .zip(options)
            .map(|(expr, options)| PhysicalSortExpr::new(Arc::clone(expr), *options)),
    )
    .ok_or_else(|| internal_datafusion_err!("join without keys"))?;
    let config = context.session_config();
    let execution = &config.options().execution;
    let mut sorter = ExternalSorter::new(
        partition,
        Arc::clone(&schema),
        ordering,
        config.batch_size(),
        // The merge reserve and in-place sort must fit the budget
        execution.sort_spill_reservation_bytes.min(budget / 4),
        execution.sort_in_place_threshold_bytes.min(budget / 4),
        config.spill_compression(),
        &ExecutionPlanMetricsSet::new(),
        context.runtime_env(),
    )?;
    let mut rows = spill.read_spill_as_stream(Arc::clone(file), None)?;
    while let Some(batch) = rows.try_next().await? {
        if let Err(e) = sorter.insert_batch(batch).await {
            sorter.abort_in_progress_spill().await;
            return Err(e);
        }
    }
    sorter.sort().await
}

/// Splits one spilled side into the buckets of the next level
async fn resplit(
    spill: &SpillManager,
    batch_size: usize,
    memory: &MemoryReservation,
    on: &[PhysicalExprRef],
    side: SpilledSide,
    level: usize,
) -> Result<Vec<SpilledSide>> {
    let mut writers: Vec<Option<BucketFile>> = (0..BUCKETS).map(|_| None).collect();
    let mut sides: Vec<SpilledSide> =
        (0..BUCKETS).map(|_| SpilledSide::default()).collect();
    if let Some(file) = side.file {
        let mut rows = spill.read_spill_as_stream(file, None)?;
        while let Some(batch) = rows.try_next().await? {
            for (bucket, part) in split(&batch, on, level)? {
                let writer = match &mut writers[bucket] {
                    Some(writer) => writer,
                    slot => slot.insert(BucketFile::try_new(
                        spill,
                        batch_size,
                        "HashJoin bucket",
                        memory.new_empty(),
                    )?),
                };
                writer.append(&part)?;
                sides[bucket].rows += part.num_rows();
                sides[bucket].bytes += get_record_batch_memory_size(&part);
            }
        }
    }
    for (side, writer) in sides.iter_mut().zip(writers) {
        if let Some(writer) = writer {
            side.file = writer.finish()?;
        }
    }
    Ok(sides)
}

/// A seed per level. Seeds that differ in a few low bits give nearly the same
/// high hash bits, so a split would route every row as the level before did:
/// mix the level in first (splitmix64).
fn bucket_random_state(level: usize) -> RandomState {
    let mut z =
        BUCKET_SEED.wrapping_add((level as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    RandomState::with_seed(z ^ (z >> 31))
}

/// The bucket of every row of `batch` at `level`
fn bucket_ids(
    batch: &RecordBatch,
    on: &[PhysicalExprRef],
    level: usize,
) -> Result<Vec<usize>> {
    let keys = evaluate_expressions_to_arrays(on, batch)?;
    let mut hashes = vec![0; batch.num_rows()];
    // Values SQL considers equal (e.g. +0.0 and -0.0) hash the same, so they
    // reach the same bucket on both sides
    create_hashes(&keys, &bucket_random_state(level), &mut hashes)?;
    Ok(hashes
        .into_iter()
        .map(|hash| (hash >> (u64::BITS - BUCKET_BITS)) as usize)
        .collect())
}

/// The rows of `batch` per non-empty bucket
fn split(
    batch: &RecordBatch,
    on: &[PhysicalExprRef],
    level: usize,
) -> Result<Vec<(usize, RecordBatch)>> {
    if batch.num_rows() == 0 {
        return Ok(vec![]);
    }
    let mut rows: Vec<Vec<u32>> = vec![vec![]; BUCKETS];
    for (row, bucket) in bucket_ids(batch, on, level)?.into_iter().enumerate() {
        rows[bucket].push(row as u32);
    }
    if let Some(bucket) = rows.iter().position(|r| r.len() == batch.num_rows()) {
        return Ok(vec![(bucket, batch.clone())]);
    }
    rows.into_iter()
        .enumerate()
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(bucket, rows)| {
            Ok((bucket, take_record_batch(batch, &UInt32Array::from(rows))?))
        })
        .collect()
}

/// Passes the probe rows of buckets kept in memory on to their join, and
/// writes those of spilled buckets to the bucket's probe file
struct ProbeRouter {
    input: SendableRecordBatchStream,
    on_right: Vec<PhysicalExprRef>,
    spill: SpillManager,
    level: usize,
    spilled: [bool; BUCKETS],
    batch_size: usize,
    /// Charges the buffered probe rows to the kept memory
    memory: MemoryReservation,
    writers: Vec<Option<BucketFile>>,
    sides: Vec<Option<SpilledSide>>,
    /// The probe files, handed over once the input is exhausted
    output: Arc<Mutex<Vec<Option<SpilledSide>>>>,
    finished: bool,
}

impl ProbeRouter {
    fn route(&mut self, batch: RecordBatch) -> Result<Option<RecordBatch>> {
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        let ids = bucket_ids(&batch, &self.on_right, self.level)?;
        if ids.iter().all(|b| !self.spilled[*b]) {
            return Ok(Some(batch));
        }
        let mut rows: Vec<Vec<u32>> = vec![vec![]; BUCKETS];
        for (row, bucket) in ids.iter().enumerate() {
            if self.spilled[*bucket] {
                rows[*bucket].push(row as u32);
            }
        }
        for (bucket, rows) in rows.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let part = take_record_batch(&batch, &UInt32Array::from(rows))?;
            let writer = match &mut self.writers[bucket] {
                Some(writer) => writer,
                slot => slot.insert(BucketFile::try_new(
                    &self.spill,
                    self.batch_size,
                    "HashJoin probe bucket",
                    self.memory.new_empty(),
                )?),
            };
            writer.append(&part)?;
            let side = self.sides[bucket].get_or_insert_with(SpilledSide::default);
            side.rows += part.num_rows();
            side.bytes += get_record_batch_memory_size(&part);
        }
        let resident: BooleanArray =
            ids.iter().map(|b| Some(!self.spilled[*b])).collect();
        if resident.true_count() == 0 {
            return Ok(None);
        }
        Ok(Some(filter_record_batch(&batch, &resident)?))
    }

    fn finish(&mut self) -> Result<()> {
        let mut output = self.output.lock();
        for bucket in 0..BUCKETS {
            if let Some(writer) = self.writers[bucket].take() {
                let mut side = self.sides[bucket].take().unwrap_or_default();
                side.file = writer.finish()?;
                output[bucket] = Some(side);
            }
        }
        Ok(())
    }
}

impl Stream for ProbeRouter {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if this.finished {
                return Poll::Ready(None);
            }
            match ready!(this.input.poll_next_unpin(cx)) {
                Some(Ok(batch)) => match this.route(batch) {
                    Ok(Some(batch)) => return Poll::Ready(Some(Ok(batch))),
                    Ok(None) => continue,
                    Err(e) => return Poll::Ready(Some(Err(e))),
                },
                Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                None => {
                    this.finished = true;
                    if let Err(e) = this.finish() {
                        return Poll::Ready(Some(Err(e)));
                    }
                }
            }
        }
    }
}

impl RecordBatchStream for ProbeRouter {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }
}

/// Stops after `fetch` rows
fn limit(
    stream: SendableRecordBatchStream,
    fetch: Option<usize>,
) -> SendableRecordBatchStream {
    let Some(fetch) = fetch else {
        return stream;
    };
    let schema = stream.schema();
    let limited = stream.scan(fetch, |left, item| {
        let item = match item {
            Ok(_) if *left == 0 => None,
            Ok(batch) => {
                let rows = batch.num_rows().min(*left);
                *left -= rows;
                Some(Ok(batch.slice(0, rows)))
            }
            Err(e) => Some(Err(e)),
        };
        futures::future::ready(item)
    });
    Box::pin(RecordBatchStreamAdapter::new(schema, limited))
}

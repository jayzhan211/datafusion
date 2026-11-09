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

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::task::{Context, Poll};

use arrow::array::{
    BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array, new_null_array,
};
use arrow::compute::{concat_batches, filter_record_batch, take_record_batch};
use arrow::datatypes::SchemaRef;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::utils::memory::{
    RecordBatchMemoryCounter, get_record_batch_memory_size,
};
use datafusion_common::{JoinSide, JoinType, NullEquality, Result, internal_err};
use datafusion_execution::SpillFile;
use datafusion_execution::memory_pool::{
    MemoryConsumer, MemoryPool, MemoryReservation, MergeMemoryPool,
};
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;
use futures::future::BoxFuture;
use futures::{FutureExt, Stream, StreamExt, TryStreamExt, ready, stream};
use parking_lot::Mutex;

use super::exec::{
    BuildOutcome, BuildOverflow, BuildParams, JoinLeftData, build_table_bytes,
    is_resources_exhausted,
};
use super::not_in::{NotIn, NotInScope};
use super::shared_bounds::SharedBuildAccumulator;
use super::stream::{
    BuildSide, BuildSideInitialState, HashJoinStream, HashJoinStreamState,
};
use crate::joins::PartitionMode;
use crate::joins::utils::{
    BuildProbeJoinMetrics, ColumnIndex, JoinFilter, OnceFut, need_produce_result_in_final,
};
use crate::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};
use crate::spill::in_progress_spill_file::InProgressSpillFile;
use crate::spill::spill_manager::SpillManager;
use crate::stream::RecordBatchStreamAdapter;
use crate::{RecordBatchStream, SendableRecordBatchStream};

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
    /// Whose build data the dynamic filter expects: `CollectLeft` when the
    /// only probe stream of a `CollectLeft` join spills
    pub(super) mode: PartitionMode,
}

impl StreamParams {
    fn stream(
        &self,
        left_data: OnceFut<JoinLeftData>,
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
                left_fut: left_data,
            }),
            self.batch_size,
            vec![],
            right_side_ordered,
            accumulator,
            self.mode,
            fetch,
        )
    }
}

/// Everything one spilling partition needs
pub(super) struct SpillContext {
    pub(super) stream: StreamParams,
    pub(super) build: BuildParams,
    /// The keys rows are bucketed by, per side: the join keys, or only the
    /// scope keys of a correlated `NOT IN` (see [`NotIn::first_routing_key`]).
    /// Empty when no key can route: the whole join is then one pair.
    pub(super) build_routing: Vec<PhysicalExprRef>,
    pub(super) probe_routing: Vec<PhysicalExprRef>,
    pub(super) build_spill: SpillManager,
    pub(super) probe_spill: SpillManager,
    pub(super) pool: Arc<dyn MemoryPool>,
    pub(super) pairs: SpilledPairMetrics,
    pub(super) fetch: Option<usize>,
    pub(super) right_side_ordered: bool,
}

/// How the spilled pairs of a partition were joined, beyond the spill counts
#[derive(Clone, Debug)]
pub(super) struct SpilledPairMetrics {
    /// Pairs split again because they did not fit the kept memory
    resplit: Count,
    /// Pairs that splitting cannot shrink, joined with their probe rows hashed
    reversed: Count,
    /// Pairs that splitting cannot shrink, joined a chunk at a time
    chunked: Count,
}

impl SpilledPairMetrics {
    pub(super) fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let counter = |name| MetricBuilder::new(metrics).counter(name, partition);
        Self {
            resplit: counter("spilled_pairs_resplit"),
            reversed: counter("spilled_pairs_reversed"),
            chunked: counter("spilled_pairs_chunked"),
        }
    }
}

impl SpillContext {
    /// An uncorrelated `probe.key NOT IN (build.key)`, which no probe row
    /// passes once a build key is NULL
    fn probe_rows_not_in(&self) -> bool {
        matches!(
            self.build.not_in,
            Some(NotIn::ProbeRows {
                scope: NotInScope::Uncorrelated
            })
        )
    }

    /// An uncorrelated `build.key NOT IN (probe.key)`, whose build rows are
    /// decided by whether the probe side had a row and a NULL key
    fn build_rows_not_in(&self) -> bool {
        matches!(
            self.build.not_in,
            Some(NotIn::BuildRows {
                scope: NotInScope::Uncorrelated,
                ..
            })
        )
    }

    /// A `NOT IN` deciding build rows, correlated or not: each must meet its
    /// whole share of the probe side, so it is never joined with the roles
    /// swapped
    fn judges_build_rows(&self) -> bool {
        matches!(self.build.not_in, Some(NotIn::BuildRows { .. }))
    }
}

/// What the probe rows of every stream held, for an uncorrelated
/// `build.key NOT IN (probe.key)` (see `NotInBuild::final_chunk`): a spilled
/// pair's own probe rows are only part of the probe side
#[derive(Debug, Default)]
struct ProbeFacts {
    has_row: AtomicBool,
    has_null: AtomicBool,
}

impl ProbeFacts {
    fn record(&self, has_null: bool) {
        self.has_row.store(true, Ordering::Relaxed);
        if has_null {
            self.has_null.store(true, Ordering::Relaxed);
        }
    }

    /// Records the whole probe side on the table of one spilled pair, whose
    /// final stage then decides as if it had seen all of it. Spilled pairs are
    /// joined only after every stream finished routing.
    fn record_on(&self, left_data: &JoinLeftData) {
        left_data.record_probe_batch(
            self.has_row.load(Ordering::Relaxed),
            self.has_null.load(Ordering::Relaxed),
        );
    }
}

/// Whether a row of `batch` has a NULL `NOT IN` value `key`
fn has_null_key(key: &PhysicalExprRef, batch: &RecordBatch) -> Result<bool> {
    let values = key.evaluate(batch)?.into_array(batch.num_rows())?;
    Ok(values.logical_null_count() > 0)
}

/// The stream of a `Partitioned` partition that can spill: today's
/// [`HashJoinStream`] if the build side fits, the bucketed join otherwise.
pub(super) fn spillable_stream(
    ctx: SpillContext,
    build: BoxFuture<'static, Result<BuildOutcome>>,
    right: SendableRecordBatchStream,
    accumulator: Option<Arc<SharedBuildAccumulator>>,
) -> SendableRecordBatchStream {
    let schema = Arc::clone(&ctx.stream.schema);
    let report = PendingReport {
        accumulator,
        partition: ctx.stream.partition,
    };
    let ctx = Arc::new(ctx);
    let output = stream::once(async move {
        let stream: SendableRecordBatchStream = match build.await? {
            BuildOutcome::InMemory(left_data) => Box::pin(ctx.stream.stream(
                owned(left_data),
                right,
                report.take(),
                ctx.right_side_ordered,
                ctx.fetch,
            )),
            BuildOutcome::Overflow(overflow) => {
                // The kept buckets cannot answer for the spilled ones: let every
                // probe row of this partition through.
                drop(report);
                let left = split_build(&ctx, overflow, 1).await?;
                let fetch = ctx.fetch;
                limit(probe_spilled(ctx, Arc::new(left), right)?, fetch)
            }
        };
        Ok::<_, datafusion_common::DataFusionError>(stream)
    })
    .try_flatten();
    Box::pin(RecordBatchStreamAdapter::new(schema, output))
}

/// A `Partitioned` partition's place in the dynamic filter until its
/// [`HashJoinStream`] takes it over. Today's stream exists from `execute` on
/// and reports its partition canceled when dropped before reporting; the one
/// here is created only once the build side is collected, so this reports
/// for it when the partition is dropped earlier (for example unpolled, by a
/// parent partition whose build side is empty) or spills. Otherwise the other
/// partitions wait for its report forever.
struct PendingReport {
    accumulator: Option<Arc<SharedBuildAccumulator>>,
    partition: usize,
}

impl PendingReport {
    /// Hands the accumulator to the stream that reports for this partition
    fn take(mut self) -> Option<Arc<SharedBuildAccumulator>> {
        self.accumulator.take()
    }
}

impl Drop for PendingReport {
    fn drop(&mut self) {
        if let Some(accumulator) = self.accumulator.take() {
            accumulator.report_canceled_partition(self.partition);
        }
    }
}

/// The build side of a `CollectLeft` join that can spill, collected once for
/// all its probe streams: today's build if it fits, else the kept buckets'
/// table carrying the spilled ones, see [`SpilledBuild`]
pub(super) async fn collect_left_build(
    ctx: SpillContext,
    build: BoxFuture<'static, Result<BuildOutcome>>,
    probe_streams: usize,
) -> Result<JoinLeftData> {
    match build.await? {
        BuildOutcome::InMemory(left_data) => Ok(left_data),
        BuildOutcome::Overflow(overflow) => {
            split_build(&Arc::new(ctx), overflow, probe_streams).await
        }
    }
}

/// One probe stream of a `CollectLeft` join that can spill
pub(super) fn collect_left_stream(
    ctx: SpillContext,
    mut build: OnceFut<JoinLeftData>,
    right: SendableRecordBatchStream,
    accumulator: Option<Arc<SharedBuildAccumulator>>,
) -> SendableRecordBatchStream {
    let schema = Arc::clone(&ctx.stream.schema);
    let ctx = Arc::new(ctx);
    let output = stream::once(async move {
        let left = futures::future::poll_fn(|cx| build.get_shared(cx)).await?;
        let fetch = ctx.fetch;
        let stream: SendableRecordBatchStream = if left.spilled().is_some() {
            limit(probe_spilled(ctx, left, right)?, fetch)
        } else {
            Box::pin(ctx.stream.stream(
                OnceFut::ready(left),
                right,
                accumulator,
                ctx.right_side_ordered,
                fetch,
            ))
        };
        Ok::<_, datafusion_common::DataFusionError>(stream)
    })
    .try_flatten();
    Box::pin(RecordBatchStreamAdapter::new(schema, output))
}

/// The spilled part of a build side, shared by the probe streams that join
/// its kept part: each routes its probe rows of spilled buckets to its own
/// files, and the last one to finish joins every spilled bucket with all of
/// them (as the last stream of a `CollectLeft` join emits its unmatched build
/// rows). No stream waits for another.
pub(super) struct SpilledBuild {
    /// The kept buckets' table and the kept memory: each probe stream takes
    /// them when it starts, and the last one to finish routing empties these
    /// slots, so a `CollectLeft` build side (shared until the query ends)
    /// frees its table before the spilled pairs are joined, as a
    /// `Partitioned` one does
    resident: Mutex<Option<Arc<JoinLeftData>>>,
    kept: Mutex<Option<Arc<KeptMemory>>>,
    /// Which buckets were spilled
    buckets: [bool; BUCKETS],
    /// The spilled build sides, taken by the last probe stream
    build: Mutex<Vec<(usize, SpilledSide)>>,
    /// The probe rows of each spilled bucket, from every probe stream
    probe: Mutex<Vec<SpilledSide>>,
    /// Probe streams still routing
    routing: AtomicUsize,
    /// An uncorrelated `probe.key NOT IN (build.key)` whose build side has a
    /// NULL key: no probe row's `NOT IN` is TRUE, so nothing is emitted
    no_rows: bool,
    /// An uncorrelated `probe.key NOT IN (build.key)` whose build side has a
    /// row: a probe row with a NULL key is never TRUE (it is when the build
    /// side is empty, which can overflow too)
    drop_null_probe_keys: bool,
    /// For an uncorrelated `build.key NOT IN (probe.key)`, see [`ProbeFacts`]
    probe_facts: Option<Arc<ProbeFacts>>,
}

/// Splits an overflowed build side into buckets, spills the largest until the
/// rest fits, and builds the kept buckets' table for `probe_streams` streams
async fn split_build(
    ctx: &Arc<SpillContext>,
    overflow: BuildOverflow,
    probe_streams: usize,
) -> Result<JoinLeftData> {
    let BuildOverflow {
        batches,
        mut rest,
        reservation,
    } = overflow;
    let kept = Arc::new(KeptMemory::take_over(ctx, reservation));
    // The handle shared by the probe streams holds no rows itself. Built
    // first, so that the kept buckets below are sized around it rather than it
    // asking for memory once they fill what is kept
    let handle =
        ctx.build
            .build(&mut vec![], &mut kept.buffers.new_empty(), probe_streams)?;

    // The collected batches are covered by the reservation taken over: release
    // each one's share once its rows are copied into the buckets.
    let mut build = BucketedBuild::new(0);
    let mut collected = kept.buffers.size();
    // Only an uncorrelated `probe.key NOT IN (build.key)` looks: one key
    let mut build_has_row = false;
    let mut build_has_null = false;
    let mut look_for_null = |batch: &RecordBatch| -> Result<()> {
        if ctx.probe_rows_not_in() && !build_has_null {
            build_has_row |= batch.num_rows() > 0;
            build_has_null = has_null_key(&ctx.build.on_left[0], batch)?;
        }
        Ok(())
    };
    for batch in batches {
        let _timer = ctx.build.metrics.build_time.timer();
        look_for_null(&batch)?;
        build.push(ctx, &kept, &batch)?;
        let released = get_record_batch_memory_size(&batch).min(collected);
        kept.buffers.shrink(released);
        collected -= released;
    }
    kept.buffers.shrink(collected);
    while let Some(batch) = rest.try_next().await? {
        let _timer = ctx.build.metrics.build_time.timer();
        look_for_null(&batch)?;
        build.push(ctx, &kept, &batch)?;
    }
    drop(rest);

    let (mut resident, mut spilled) = {
        let _timer = ctx.build.metrics.build_time.timer();
        build.finish(ctx, &kept)?
    };
    let mut reservation = kept.buffers.split(kept.buffers.size());
    let left_data = match ctx
        .build
        .build(&mut resident, &mut reservation, probe_streams)
    {
        Ok(left_data) => left_data,
        // The table took more than estimated and the pool had no more: the
        // build side can still spill, so spill the kept buckets too
        Err(e) if is_resources_exhausted(&e) => {
            reservation.free();
            let mut files: Vec<Option<BucketFile>> = (0..BUCKETS).map(|_| None).collect();
            let mut sides: Vec<SpilledSide> =
                (0..BUCKETS).map(|_| SpilledSide::default()).collect();
            for batch in std::mem::take(&mut resident) {
                for (bucket, part) in split(&batch, &ctx.build_routing, 0)? {
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
                    side.files.extend(file.finish()?);
                    spilled.push((bucket, side));
                }
            }
            ctx.build
                .build(&mut vec![], &mut kept.buffers.new_empty(), probe_streams)
                .map_err(|e| e.context("an empty kept build side"))?
        }
        Err(e) => return Err(e),
    };
    // Joining a spilled pair needs at most what its largest bucket takes:
    // keep that, and let the rest go back to the pool once the kept buckets
    // are joined, so operators that start later are not starved.
    kept.keep_for_spilled_pairs(
        spilled
            .iter()
            .map(|(_, side)| join_bytes(side.rows, side.bytes)),
    );

    let mut buckets = [false; BUCKETS];
    for (bucket, _) in &spilled {
        buckets[*bucket] = true;
    }
    Ok(handle.with_spilled(Arc::new(SpilledBuild {
        resident: Mutex::new(Some(Arc::new(left_data))),
        kept: Mutex::new(Some(kept)),
        buckets,
        build: Mutex::new(spilled),
        probe: Mutex::new((0..BUCKETS).map(|_| SpilledSide::default()).collect()),
        routing: AtomicUsize::new(probe_streams),
        no_rows: build_has_null,
        drop_null_probe_keys: build_has_row,
        probe_facts: ctx.build_rows_not_in().then(Default::default),
    })))
}

/// One probe stream of a spilled build side: joins the kept buckets, routes
/// the probe rows of spilled buckets to files, and, if it is the last stream
/// to finish routing, joins the spilled buckets
fn probe_spilled(
    ctx: Arc<SpillContext>,
    handle: Arc<JoinLeftData>,
    right: SendableRecordBatchStream,
) -> Result<SendableRecordBatchStream> {
    let spilled = Arc::clone(handle.spilled().expect("a spilled build side"));
    drop(handle);
    if spilled.no_rows {
        drop(right);
        if spilled.routing.fetch_sub(1, Ordering::AcqRel) == 1 {
            spilled.resident.lock().take();
            spilled.kept.lock().take();
            spilled.build.lock().clear();
        }
        let schema = Arc::clone(&ctx.stream.schema);
        return Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::empty(),
        )));
    }
    // Every stream starts before the last one finishes routing
    let (Some(left), Some(kept)) =
        (spilled.resident.lock().clone(), spilled.kept.lock().clone())
    else {
        return internal_err!("a probe stream started after the last one finished");
    };
    let routed = Arc::new(Mutex::new(empty_sides()));
    let not_in = if spilled.drop_null_probe_keys {
        RouterNotIn::DropNullKeys {
            key: Arc::clone(&ctx.stream.on_right[0]),
        }
    } else if let Some(facts) = &spilled.probe_facts {
        RouterNotIn::RecordFacts {
            key: Arc::clone(&ctx.stream.on_right[0]),
            resident: Arc::downgrade(&left),
            facts: Arc::clone(facts),
        }
    } else {
        RouterNotIn::None
    };
    let router = SharedRouter(Arc::new(Mutex::new(ProbeRouter {
        input: right,
        on_right: ctx.probe_routing.clone(),
        not_in,
        spill: ctx.probe_spill.clone(),
        level: 0,
        spilled: spilled.buckets,
        batch_size: ctx.stream.batch_size,
        memory: kept.buffers.new_empty(),
        writers: (0..BUCKETS).map(|_| None).collect(),
        sides: empty_sides(),
        output: Arc::clone(&routed),
        finished: false,
    })));
    let resident_join = ctx.stream.stream(
        OnceFut::ready(left),
        Box::pin(router.clone()),
        None,
        false,
        None,
    );

    // The join of the kept buckets stops reading probe rows once its result
    // is complete (e.g. an inner join with no rows kept), but the spilled
    // buckets still need theirs: route the rest of the probe side first. The
    // probe rows of kept buckets read here cannot change that result.
    let schema = Arc::clone(&ctx.stream.schema);
    let spilled_joins = stream::once(async move {
        router.drain().await?;
        {
            let mut probe = spilled.probe.lock();
            for (bucket, side) in routed.lock().iter_mut().enumerate() {
                if let Some(side) = side.take() {
                    probe[bucket].add(side);
                }
            }
        }
        let last = spilled.routing.fetch_sub(1, Ordering::AcqRel) == 1;
        let build = if last {
            // Every stream has joined the kept buckets: free their table
            spilled.resident.lock().take();
            spilled.kept.lock().take();
            std::mem::take(&mut *spilled.build.lock())
        } else {
            vec![]
        };
        let facts = spilled.probe_facts.clone();
        let joins = stream::iter(build)
            .then(move |(bucket, build)| {
                let probe = std::mem::take(&mut spilled.probe.lock()[bucket]);
                join_pair(
                    Arc::clone(&ctx),
                    Arc::clone(&kept),
                    build,
                    probe,
                    0,
                    facts.clone(),
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

/// The in-memory join data of one bucket pair, for its own stream
fn owned(left_data: JoinLeftData) -> OnceFut<JoinLeftData> {
    OnceFut::ready(Arc::new(left_data))
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
                    Some(Ok(_)) => {}
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
    /// What joining one spilled pair may use, set once the build side is split
    for_spilled_pairs: AtomicUsize,
}

/// Once every stream is done with it, the memory kept for the spilled pairs
/// goes back to the pool, even while a `CollectLeft` handle (shared until the
/// query ends) still holds a reservation of it
impl Drop for KeptMemory {
    fn drop(&mut self) {
        self.pool.release_unused();
    }
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
            for_spilled_pairs: AtomicUsize::new(0),
        }
    }

    /// Grows `reservation`, a child of the kept pool, while the build side is
    /// split, unless the execution pool refuses. What it gets is kept until
    /// the spilled pairs are joined.
    fn try_grow(&self, reservation: &MemoryReservation, bytes: usize) -> bool {
        let granted = reservation.try_grow(bytes).is_ok();
        if granted {
            self.pool.retain(self.pool.reserved());
        }
        granted
    }

    /// Keeps what joining the spilled pairs takes once the memory in use is
    /// released. A pair larger than the memory held is split anyway, so it
    /// only needs what its pieces take after the splits it needs: keeping
    /// more would only starve other operators.
    fn keep_for_spilled_pairs(&self, pairs: impl Iterator<Item = usize>) {
        let held = self.pool.reserved();
        let bytes = pairs
            .map(|pair| {
                let mut piece = pair;
                while piece > held && piece >= BUCKETS {
                    piece = piece.div_ceil(BUCKETS);
                }
                piece
            })
            .max()
            .unwrap_or(0);
        let bytes = bytes.min(held);
        self.pool.retain(bytes);
        self.for_spilled_pairs.store(bytes, Ordering::Relaxed);
    }

    /// Asks the pool for more kept memory, up to `want` bytes and at least
    /// `useful`, halving the request on refusal. For a pair the splits left
    /// cannot bring under the kept memory, as when the build side overflowed
    /// holding almost nothing: splitting would only write more files.
    fn grow_toward(&self, want: usize, useful: usize) {
        let have = self.capacity();
        if want <= have {
            return;
        }
        let probe = self.buffers.new_empty();
        let mut more = want - have;
        while more > 0 && have + more >= useful {
            if probe.try_grow(more).is_ok() {
                let kept = have + more;
                self.pool.retain(kept.min(self.pool.reserved()));
                self.for_spilled_pairs.store(kept, Ordering::Relaxed);
                return;
            }
            more /= 2;
        }
    }

    /// What joining one spilled pair may use without asking the execution pool
    fn capacity(&self) -> usize {
        self.for_spilled_pairs.load(Ordering::Relaxed)
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

/// One side of a spilled bucket: its rows, in one file per writer (a probe
/// side has one per probe stream that routed rows to it)
#[derive(Clone, Default)]
struct SpilledSide {
    files: Vec<Arc<dyn SpillFile>>,
    rows: usize,
    bytes: usize,
}

impl SpilledSide {
    fn add(&mut self, other: SpilledSide) {
        self.files.extend(other.files);
        self.rows += other.rows;
        self.bytes += other.bytes;
    }
}

fn empty_sides() -> Vec<Option<SpilledSide>> {
    (0..BUCKETS).map(|_| None).collect()
}

#[expect(clippy::large_enum_variant)]
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

    fn push(
        &mut self,
        ctx: &SpillContext,
        kept: &KeptMemory,
        batch: &RecordBatch,
    ) -> Result<()> {
        for (bucket, part) in split(batch, &ctx.build_routing, self.level)? {
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
                self.destage(ctx, kept, victim)?;
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

    fn destage(
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
    #[expect(clippy::type_complexity)]
    fn finish(
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
            self.destage(ctx, kept, victim)?;
        }

        let mut resident = vec![];
        let mut spilled = vec![];
        for (bucket, b) in self.buckets.into_iter().enumerate() {
            match b {
                Bucket::Memory { batches, .. } => resident.extend(batches),
                Bucket::Disk { file, rows, bytes } => spilled.push((
                    bucket,
                    SpilledSide {
                        files: file.finish()?.into_iter().collect(),
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
    facts: Option<Arc<ProbeFacts>>,
) -> BoxFuture<'static, Result<SendableRecordBatchStream>> {
    async move {
        let need = join_bytes(build.rows, build.bytes);
        // Without a key to route by, splitting cannot shrink the pair
        let can_split = !ctx.build_routing.is_empty();
        // Even split evenly at every level left, the pair's pieces would not
        // fit: ask the pool now instead of splitting into ever more files
        let pieces = if can_split {
            BUCKETS.saturating_pow((MAX_LEVEL.saturating_sub(level)) as u32)
        } else {
            1
        };
        let smallest_piece = need.div_ceil(pieces.max(1));
        if smallest_piece > kept.capacity() {
            // Only what one more split needs, like the memory kept at the
            // first split: the whole pair would starve other operators
            kept.grow_toward(need.div_ceil(BUCKETS).max(smallest_piece), smallest_piece);
        }
        let splits_cannot_fit = smallest_piece > kept.capacity();
        let fits = need <= kept.capacity();
        if build.rows == 0 || fits || level >= MAX_LEVEL || splits_cannot_fit || !can_split
        {
            match join_in_memory(&ctx, &kept, &build, &probe, facts.as_deref()).await {
                Ok(stream) => return Ok(stream),
                // The estimate fell short: split further rather than fail
                Err(e)
                    if build.rows > 0
                        && level < MAX_LEVEL
                        && !splits_cannot_fit
                        && can_split
                        && is_resources_exhausted(&e) => {}
                // Splitting cannot shrink this pair any further: hash its
                // probe rows instead if they fit, else hash it a chunk at a time
                Err(e) if is_resources_exhausted(&e) => {
                    let reversed = !ctx.judges_build_rows()
                        && join_bytes(probe.rows, probe.bytes) <= kept.capacity();
                    if reversed {
                        match join_reversed(&ctx, &kept, &build, &probe).await {
                            Ok(stream) => {
                                ctx.pairs.reversed.add(1);
                                return Ok(stream);
                            }
                            Err(e) if is_resources_exhausted(&e) => {}
                            Err(e) => return Err(e),
                        }
                    }
                    ctx.pairs.chunked.add(1);
                    return Ok(join_chunked(&ctx, &kept, &build, &probe, facts.as_ref()));
                }
                Err(e) => {
                    return Err(e.context(format!(
                        "Joining a spilled hash join bucket pair at level {level}: {} build rows, {} bytes, capacity {}",
                        build.rows,
                        build.bytes,
                        kept.capacity()
                    )));
                }
            }
        }

        ctx.pairs.resplit.add(1);
        let level = level + 1;
        let batch_size = ctx.stream.batch_size;
        let builds =
            resplit(
                &ctx.build_spill,
                batch_size,
                &kept.buffers,
                &ctx.build_routing,
                build,
                level,
            )
                .await?;
        let probes =
            resplit(
                &ctx.probe_spill,
                batch_size,
                &kept.buffers,
                &ctx.probe_routing,
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
                join_pair(
                    Arc::clone(&ctx),
                    Arc::clone(&kept),
                    b,
                    p,
                    level,
                    facts.clone(),
                )
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
    facts: Option<&ProbeFacts>,
) -> Result<SendableRecordBatchStream> {
    let left_data = hash_side(&ctx.build_spill, build, &ctx.build, kept).await?;
    if let Some(facts) = facts {
        facts.record_on(&left_data);
    }
    let probe_rows = read_side(&ctx.probe_spill, probe)?;
    Ok(Box::pin(ctx.stream.stream(
        owned(left_data),
        probe_rows,
        None,
        false,
        None,
    )))
}

/// The parameters of a pair joined with the roles swapped: the probe rows are
/// hashed and the build rows stream through them under `join_type`, the mirror
/// of the join's type. Every join type has a mirror, and the output columns
/// are the same ones read from the other side.
fn mirrored(ctx: &SpillContext, join_type: JoinType) -> (BuildParams, StreamParams) {
    let params = BuildParams {
        random_state: ctx.build.random_state.clone(),
        schema: Arc::clone(ctx.probe_spill.schema()),
        on_left: ctx.stream.on_right.clone(),
        metrics: ctx.build.metrics.clone(),
        with_visited_indices_bitmap: need_produce_result_in_final(join_type),
        config: Arc::clone(&ctx.build.config),
        null_equality: ctx.build.null_equality,
        array_map_created_count: ctx.build.array_map_created_count.clone(),
        not_in: ctx.build.not_in.and_then(NotIn::mirrored),
    };
    let stream = StreamParams {
        on_right: ctx.build.on_left.clone(),
        filter: ctx.stream.filter.as_ref().map(JoinFilter::swap),
        join_type,
        column_indices: ctx
            .stream
            .column_indices
            .iter()
            .map(|column| ColumnIndex {
                index: column.index,
                side: column.side.negate(),
            })
            .collect(),
        ..ctx.stream.clone()
    };
    (params, stream)
}

/// Joins a pair whose build side cannot shrink but whose probe side fits, with
/// the roles swapped
async fn join_reversed(
    ctx: &SpillContext,
    kept: &KeptMemory,
    build: &SpilledSide,
    probe: &SpilledSide,
) -> Result<SendableRecordBatchStream> {
    let (params, stream) = mirrored(ctx, ctx.stream.join_type.swap());
    let left_data = hash_side(&ctx.probe_spill, probe, &params, kept).await?;
    let build_rows = read_side(&ctx.build_spill, build)?;
    Ok(Box::pin(stream.stream(
        owned(left_data),
        build_rows,
        None,
        false,
        None,
    )))
}

/// Reads one spilled side into memory and hashes it
async fn hash_side(
    spill: &SpillManager,
    side: &SpilledSide,
    params: &BuildParams,
    kept: &KeptMemory,
) -> Result<JoinLeftData> {
    let mut reservation = kept.buffers.new_empty();
    let mut batches = vec![];
    let mut rows = read_side(spill, side)?;
    while let Some(batch) = rows.try_next().await? {
        // From the kept memory, or from the pool for a pair that cannot be
        // split further
        reservation.try_grow(get_record_batch_memory_size(&batch))?;
        batches.push(batch);
    }
    params.build(&mut batches, &mut reservation, 1)
}

/// The rows of one spilled side
fn read_side(
    spill: &SpillManager,
    side: &SpilledSide,
) -> Result<SendableRecordBatchStream> {
    let schema = Arc::clone(spill.schema());
    let spill = spill.clone();
    let rows = stream::iter(side.files.clone())
        .map(move |file| spill.read_spill_as_stream(file, None))
        .try_flatten();
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, rows)))
}

/// Joins a pair that splitting cannot shrink and whose sides both exceed the
/// kept memory: one side is hashed a chunk at a time, and the other side's
/// file is read once per chunk. A row's result (unmatched, semi, anti, mark)
/// is final only once it has met every row of the other side, so the chunked
/// side is the one whose rows the join type judges. A full join judges both:
/// it runs as a left join, then emits the probe rows that match nothing.
fn join_chunked(
    ctx: &Arc<SpillContext>,
    kept: &Arc<KeptMemory>,
    build: &SpilledSide,
    probe: &SpilledSide,
    facts: Option<&Arc<ProbeFacts>>,
) -> SendableRecordBatchStream {
    let join_type = ctx.stream.join_type;
    let chunk = |params, stream, swapped| {
        let (hashed, streamed) = if swapped {
            (probe, build)
        } else {
            (build, probe)
        };
        chunks(
            ctx,
            kept,
            params,
            stream,
            swapped,
            hashed.clone(),
            streamed.clone(),
            // Facts about the probe side, for build rows chunked
            if swapped { None } else { facts.cloned() },
        )
    };
    match join_type {
        JoinType::Inner
        | JoinType::Left
        | JoinType::LeftSemi
        | JoinType::LeftAnti
        | JoinType::LeftMark => chunk(ctx.build.clone(), ctx.stream.clone(), false),
        JoinType::Right
        | JoinType::RightSemi
        | JoinType::RightAnti
        | JoinType::RightMark => {
            let (params, stream) = mirrored(ctx, join_type.swap());
            chunk(params, stream, true)
        }
        JoinType::Full => {
            let left = StreamParams {
                join_type: JoinType::Left,
                ..ctx.stream.clone()
            };
            let matched = chunk(ctx.build.clone(), left, false);
            // The probe rows that match nothing, as the hashed side of a
            // mirrored anti join, then padded with nulls for the build columns
            let (params, mut anti) = mirrored(ctx, JoinType::LeftAnti);
            let probe_schema = Arc::clone(ctx.probe_spill.schema());
            anti.column_indices = (0..probe_schema.fields().len())
                .map(|index| ColumnIndex {
                    index,
                    side: JoinSide::Left,
                })
                .collect();
            anti.schema = probe_schema;
            let schema = Arc::clone(&ctx.stream.schema);
            let columns = ctx.stream.column_indices.clone();
            let output = Arc::clone(&schema);
            let unmatched = chunk(params, anti, true).map(move |batch| {
                let batch = batch?;
                let rows = batch.num_rows();
                let arrays = columns
                    .iter()
                    .zip(output.fields())
                    .map(|(column, field)| match column.side {
                        JoinSide::Right => Ok(Arc::clone(batch.column(column.index))),
                        JoinSide::Left => Ok(new_null_array(field.data_type(), rows)),
                        JoinSide::None => {
                            internal_err!("a full join has no mark column")
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let options = RecordBatchOptions::new().with_row_count(Some(rows));
                Ok(RecordBatch::try_new_with_options(
                    Arc::clone(&output),
                    arrays,
                    &options,
                )?)
            });
            Box::pin(RecordBatchStreamAdapter::new(
                schema,
                matched.chain(unmatched),
            ))
        }
    }
}

/// One hash join per chunk of `hashed` that fits the kept memory, each reading
/// all of `streamed`. With `swapped`, `hashed` is the probe side.
#[expect(clippy::too_many_arguments)]
fn chunks(
    ctx: &Arc<SpillContext>,
    kept: &Arc<KeptMemory>,
    params: BuildParams,
    stream: StreamParams,
    swapped: bool,
    hashed: SpilledSide,
    streamed: SpilledSide,
    facts: Option<Arc<ProbeFacts>>,
) -> SendableRecordBatchStream {
    let schema = Arc::clone(&stream.schema);
    let (ctx, kept) = (Arc::clone(ctx), Arc::clone(kept));
    // Rows per chunk from the side's average row, so the chunk and its hash
    // table fit the kept memory
    let per_row = join_bytes(hashed.rows, hashed.bytes)
        .div_ceil(hashed.rows.max(1))
        .max(1);
    let chunk_rows = (kept.capacity() / per_row).max(1);
    // The rows not joined yet, the rows read but not joined yet (front first),
    // and the rows a chunk takes
    let state = (
        None::<SendableRecordBatchStream>,
        VecDeque::<RecordBatch>::new(),
        chunk_rows,
    );
    let joins = stream::try_unfold(state, move |(rows, mut pending, mut chunk_rows)| {
        let (ctx, kept) = (Arc::clone(&ctx), Arc::clone(&kept));
        let (params, stream) = (params.clone(), stream.clone());
        let (hashed, streamed) = (hashed.clone(), streamed.clone());
        let facts = facts.clone();
        async move {
            let (hashed_spill, streamed_spill) = if swapped {
                (&ctx.probe_spill, &ctx.build_spill)
            } else {
                (&ctx.build_spill, &ctx.probe_spill)
            };
            let mut rows = match rows {
                Some(rows) => rows,
                None => read_side(hashed_spill, &hashed)?,
            };
            loop {
                let mut batches = vec![];
                let mut taken = 0;
                while taken < chunk_rows {
                    let (batch, sliced) = match pending.pop_front() {
                        Some(batch) => (batch, true),
                        None => match rows.try_next().await? {
                            Some(batch) => (batch, false),
                            None => break,
                        },
                    };
                    let take = batch.num_rows().min(chunk_rows - taken);
                    let piece = batch.slice(0, take);
                    // A slice holds its whole batch: copy it, so the chunk
                    // holds (and is charged for) only its own rows
                    let piece = if sliced || take < batch.num_rows() {
                        let rows = UInt32Array::from_iter_values(0..take as u32);
                        take_record_batch(&piece, &rows)?
                    } else {
                        piece
                    };
                    if take < batch.num_rows() {
                        pending.push_front(batch.slice(take, batch.num_rows() - take));
                    }
                    batches.push(piece);
                    taken += take;
                }
                if taken == 0 {
                    return Ok::<_, datafusion_common::DataFusionError>(None);
                }
                let mut reservation = kept.buffers.new_empty();
                // What the build counts as its input: a slice holds its whole batch
                let mut counter = RecordBatchMemoryCounter::new();
                let built = reservation
                    .try_grow(batches.iter().map(|b| counter.count_batch(b)).sum())
                    .and_then(|()| params.build(&mut batches, &mut reservation, 1));
                match built {
                    Ok(left_data) => {
                        if let Some(facts) = &facts {
                            facts.record_on(&left_data);
                        }
                        let streamed_rows = read_side(streamed_spill, &streamed)?;
                        let joined: SendableRecordBatchStream = Box::pin(stream.stream(
                            owned(left_data),
                            streamed_rows,
                            None,
                            false,
                            None,
                        ));
                        return Ok(Some((joined, (Some(rows), pending, chunk_rows))));
                    }
                    // A small chunk's table takes more than its rows suggest
                    // (a hash table has a smallest size): put the rows back and
                    // take half as many
                    Err(e) if taken > 1 && is_resources_exhausted(&e) => {
                        drop(reservation);
                        for batch in batches.into_iter().rev() {
                            pending.push_front(batch);
                        }
                        chunk_rows = taken / 2;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    })
    .try_flatten();
    Box::pin(RecordBatchStreamAdapter::new(schema, joins))
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
    {
        let mut rows = read_side(spill, &side)?;
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
            side.files.extend(writer.finish()?);
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
    // No key to route by: one bucket holds every row
    if on.is_empty() {
        return Ok(vec![0; batch.num_rows()]);
    }
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
    not_in: RouterNotIn,
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

/// What routing does for an uncorrelated `NOT IN`, whose answers depend on
/// whole sides rather than on one bucket pair
enum RouterNotIn {
    None,
    /// `probe.key NOT IN (build.key)` against a build side with rows and no
    /// NULL key: a probe row with a NULL key is never TRUE, so it is dropped
    /// here, before a bucket without build rows could emit it as
    /// `NULL NOT IN (empty)`
    DropNullKeys {
        key: PhysicalExprRef,
    },
    /// `build.key NOT IN (probe.key)`: records what each batch held on the
    /// kept buckets' table, whose final stage emits the kept build rows, and
    /// in the facts the spilled pairs' tables start with
    RecordFacts {
        key: PhysicalExprRef,
        resident: Weak<JoinLeftData>,
        facts: Arc<ProbeFacts>,
    },
}

impl ProbeRouter {
    fn route(&mut self, batch: RecordBatch) -> Result<Option<RecordBatch>> {
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        let batch = match &self.not_in {
            RouterNotIn::None => batch,
            RouterNotIn::DropNullKeys { key } => {
                let values = key.evaluate(&batch)?.into_array(batch.num_rows())?;
                match values
                    .logical_nulls()
                    .filter(|nulls| nulls.null_count() > 0)
                {
                    Some(nulls) => {
                        let valid = BooleanArray::new(nulls.inner().clone(), None);
                        let batch = filter_record_batch(&batch, &valid)?;
                        if batch.num_rows() == 0 {
                            return Ok(None);
                        }
                        batch
                    }
                    None => batch,
                }
            }
            RouterNotIn::RecordFacts {
                key,
                resident,
                facts,
            } => {
                let has_null = has_null_key(key, &batch)?;
                facts.record(has_null);
                if let Some(resident) = resident.upgrade() {
                    resident.record_probe_batch(true, has_null);
                }
                batch
            }
        };
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
                side.files.extend(writer.finish()?);
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
                    Ok(None) => {}
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

#[cfg(test)]
mod tests {
    //! `NOT IN` joins that spill must answer as in memory. Each case places
    //! rows so that one bucket pair alone would decide wrongly: the fact it
    //! misses sits in another bucket, or nowhere it can see.

    use super::*;

    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::util::pretty::pretty_format_batches;
    use datafusion_common::JoinType;
    use datafusion_execution::TaskContext;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_physical_expr::expressions::Column;

    use crate::ExecutionPlan;
    use crate::joins::HashJoinExecBuilder;
    use crate::test::TestMemoryExec;
    use crate::test::memory_pool::CappedConsumerPool;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::BinaryExpr;

    /// Rows the spilled bucket gets, so that it is the largest and the pool
    /// cannot hold it
    const BIG: usize = 20_000;
    const LIMIT: usize = 256 * 1024;

    fn table(columns: &[(&str, Vec<Option<i32>>)]) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|(name, _)| Field::new(*name, DataType::Int32, true))
                .collect::<Vec<_>>(),
        ));
        let arrays = columns
            .iter()
            .map(|(_, values)| Arc::new(Int32Array::from(values.clone())) as _)
            .collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
        TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    /// The level-0 bucket of a key, as rows are routed when nothing else is a key
    fn bucket_of(key: Option<i32>) -> usize {
        bucket_at(key, 0)
    }

    fn bucket_at(key: Option<i32>, level: usize) -> usize {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![key]))],
        )
        .unwrap();
        let on: Vec<PhysicalExprRef> = vec![Arc::new(Column::new("k", 0))];
        bucket_ids(&batch, &on, level).unwrap()[0]
    }

    /// `n` keys from 1 on, all in `bucket` and none in `except`
    fn keys_in(bucket: usize, n: usize, except: &[i32]) -> Vec<i32> {
        (1..i32::MAX)
            .filter(|k| bucket_of(Some(*k)) == bucket && !except.contains(k))
            .take(n)
            .collect()
    }

    /// A key outside `bucket`
    fn key_outside(bucket: usize, from: i32) -> i32 {
        (from..i32::MAX)
            .find(|k| bucket_of(Some(*k)) != bucket)
            .unwrap()
    }

    /// Runs the null-aware join in memory and spilling under a small pool, and
    /// checks it spilled and gave the same rows
    async fn assert_spills_like_in_memory(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        keys: &[(&str, &str)],
        value_keys: usize,
        join_type: JoinType,
    ) {
        let run = |spill: bool| {
            let (left, right) = (Arc::clone(&left), Arc::clone(&right));
            async move {
                let on = keys
                    .iter()
                    .map(|(l, r)| {
                        (
                            Arc::new(Column::new_with_schema(l, &left.schema()).unwrap())
                                as PhysicalExprRef,
                            Arc::new(Column::new_with_schema(r, &right.schema()).unwrap())
                                as PhysicalExprRef,
                        )
                    })
                    .collect();
                let join = HashJoinExecBuilder::new(left, right, on, join_type)
                    .with_partition_mode(PartitionMode::CollectLeft)
                    .with_null_aware(true)
                    .with_null_aware_value_keys(value_keys)
                    .build()
                    .unwrap();
                let mut config = SessionConfig::new().with_batch_size(1024);
                config.options_mut().execution.enable_hash_join_spilling = spill;
                let mut runtime = RuntimeEnvBuilder::new();
                if spill {
                    runtime = runtime.with_memory_limit(LIMIT, 1.0);
                }
                let ctx = Arc::new(
                    TaskContext::default()
                        .with_session_config(config)
                        .with_runtime(runtime.build_arc().unwrap()),
                );
                let join = Arc::new(join);
                let batches = crate::collect(Arc::clone(&join) as _, ctx).await.unwrap();
                let spilled = join.metrics().and_then(|m| m.spill_count()).unwrap_or(0);
                let text = pretty_format_batches(&batches).unwrap().to_string();
                let mut rows: Vec<String> = text
                    .lines()
                    .filter(|line| line.starts_with('|'))
                    .skip(1)
                    .map(str::to_string)
                    .collect();
                rows.sort();
                (rows, spilled)
            }
        };
        let (expected, _) = run(false).await;
        let (actual, spilled) = run(true).await;
        assert!(
            spilled > 0,
            "the join must spill for the case to mean anything"
        );
        assert_eq!(actual, expected);
    }

    fn some(keys: &[i32]) -> Vec<Option<i32>> {
        keys.iter().copied().map(Some).collect()
    }

    fn ids(n: usize) -> Vec<Option<i32>> {
        (0..n as i32).map(Some).collect()
    }

    /// `build.k NOT IN (probe.k)`: the probe side's NULL is routed to a
    /// spilled bucket, yet it makes the kept build rows' answer UNKNOWN
    #[tokio::test]
    async fn not_in_build_rows_see_a_probe_null_in_a_spilled_bucket() {
        let null_bucket = bucket_of(None);
        let big = keys_in(null_bucket, BIG, &[]);
        let kept_bucket = (null_bucket + 1) % BUCKETS;
        let kept = keys_in(kept_bucket, 3, &[]);
        let mut build_keys = some(&big);
        build_keys.extend(some(&kept));
        let left = table(&[("id", ids(build_keys.len())), ("k", build_keys)]);
        // A NULL (in the big bucket) and a key that matches one big row
        let right = table(&[("k", vec![None, Some(big[0])])]);
        for join_type in [JoinType::LeftAnti, JoinType::LeftMark] {
            assert_spills_like_in_memory(
                Arc::clone(&left),
                Arc::clone(&right),
                &[("k", "k")],
                1,
                join_type,
            )
            .await;
        }
    }

    /// `build.k NOT IN (probe.k)`: every probe row is routed to a spilled
    /// bucket, yet the kept NULL build key's answer is UNKNOWN, not TRUE as
    /// against an empty probe side
    #[tokio::test]
    async fn not_in_build_rows_see_probe_rows_in_spilled_buckets() {
        let null_bucket = bucket_of(None);
        let big_bucket = (null_bucket + 1) % BUCKETS;
        let big = keys_in(big_bucket, BIG, &[]);
        let mut build_keys = some(&big);
        build_keys.push(None);
        build_keys.push(Some(key_outside(big_bucket, 1)));
        let left = table(&[("id", ids(build_keys.len())), ("k", build_keys)]);
        let right = table(&[("k", some(&big[..2]))]);
        for join_type in [JoinType::LeftAnti, JoinType::LeftMark] {
            assert_spills_like_in_memory(
                Arc::clone(&left),
                Arc::clone(&right),
                &[("k", "k")],
                1,
                join_type,
            )
            .await;
        }
    }

    /// `build.k NOT IN (probe.k)`: a spilled pair whose own probe rows are
    /// none still decides with the whole probe side, which has a NULL
    #[tokio::test]
    async fn not_in_build_rows_spilled_pairs_see_the_whole_probe_side() {
        let null_bucket = bucket_of(None);
        let other_bucket = (null_bucket + 1) % BUCKETS;
        // Two buckets too big to keep: both spill
        let mut build_keys = some(&keys_in(other_bucket, BIG, &[]));
        build_keys.extend(some(&keys_in(null_bucket, BIG, &[])));
        let left = table(&[("id", ids(build_keys.len())), ("k", build_keys)]);
        let right = table(&[("k", vec![None])]);
        for join_type in [JoinType::LeftAnti, JoinType::LeftMark] {
            assert_spills_like_in_memory(
                Arc::clone(&left),
                Arc::clone(&right),
                &[("k", "k")],
                1,
                join_type,
            )
            .await;
        }
    }

    /// `build.k NOT IN (probe.k)` over one hot key: splitting cannot shrink
    /// the pair, which is joined a chunk of build rows at a time, each still
    /// deciding with the probe side's NULL from another bucket. (That such a
    /// pair is never joined with the roles swapped is checked by the
    /// `NOT IN` fuzz test, whose spill runs reach that fallback.)
    #[tokio::test]
    async fn not_in_build_rows_hot_key_pair_sees_the_whole_probe_side() {
        // More rows than the pool holds, all with one key
        let hot = vec![Some(1); 3 * BIG];
        let left = table(&[("id", ids(3 * BIG)), ("k", hot)]);
        // A probe key that follows the hot key into its pair (the hot child is
        // found after one more split), so that its probe side is not empty
        let follower = (2..i32::MAX)
            .find(|k| (0..2).all(|l| bucket_at(Some(*k), l) == bucket_at(Some(1), l)))
            .unwrap();
        let right = table(&[("k", vec![None, Some(follower)])]);
        for join_type in [JoinType::LeftAnti, JoinType::LeftMark] {
            assert_spills_like_in_memory(
                Arc::clone(&left),
                Arc::clone(&right),
                &[("k", "k")],
                1,
                join_type,
            )
            .await;
        }
    }

    /// `probe.k NOT IN (build.k)`: the build side's NULL is in a spilled
    /// bucket, yet no probe row of a kept bucket may be emitted
    #[tokio::test]
    async fn not_in_probe_rows_see_a_build_null_in_a_spilled_bucket() {
        let null_bucket = bucket_of(None);
        let big = keys_in(null_bucket, BIG, &[]);
        let kept_bucket = (null_bucket + 1) % BUCKETS;
        let kept = keys_in(kept_bucket, 4, &[]);
        let mut build_keys = some(&big);
        build_keys.push(None);
        build_keys.extend(some(&kept[..2]));
        let left = table(&[("k", build_keys)]);
        let mut probe_keys = some(&kept[2..]);
        probe_keys.push(Some(big[0]));
        let right = table(&[("id", ids(probe_keys.len())), ("k", probe_keys)]);
        assert_spills_like_in_memory(left, right, &[("k", "k")], 1, JoinType::RightAnti)
            .await;
    }

    /// `probe.k NOT IN (build.k)`: every build row is spilled, so the kept
    /// table is empty, yet a NULL probe key is not `NULL NOT IN (empty)`
    #[tokio::test]
    async fn not_in_probe_rows_drop_a_null_key_with_an_empty_kept_table() {
        let null_bucket = bucket_of(None);
        let big_bucket = (null_bucket + 1) % BUCKETS;
        let big = keys_in(big_bucket, BIG, &[]);
        let left = table(&[("k", some(&big))]);
        let probe_keys = vec![
            None,
            Some(big[0]),
            Some(key_outside(big_bucket, 1)),
            Some(key_outside(null_bucket, 1)),
        ];
        let right = table(&[("id", ids(probe_keys.len())), ("k", probe_keys)]);
        assert_spills_like_in_memory(left, right, &[("k", "k")], 1, JoinType::RightAnti)
            .await;
    }

    /// Correlated `NOT IN`: a NULL value makes the answer UNKNOWN only within
    /// its scope (`s`), so a scope's rows must stay in one bucket pair even
    /// though their values differ
    #[tokio::test]
    async fn not_in_correlated_keeps_a_scope_in_one_pair() {
        // The scope keys route; put the scope with the NULL value among the big rows
        let big_bucket = bucket_of(Some(0));
        let scopes = keys_in(big_bucket, BIG, &[]);
        let other_scope = key_outside(big_bucket, 1);
        // Subquery side: one row per scope, plus a NULL value in scopes[0]
        let mut sub_v: Vec<Option<i32>> = (0..BIG as i32).map(Some).collect();
        let mut sub_s = some(&scopes);
        sub_v.push(None);
        sub_s.push(Some(scopes[0]));
        sub_v.push(Some(7));
        sub_s.push(Some(other_scope));
        // Outer side: values that match nothing, in the NULL's scope and
        // others, and enough rows to spill when it is the build side
        let mut outer_v = vec![Some(-1), Some(-2), Some(-3), None, Some(0)];
        let mut outer_s = vec![
            Some(scopes[0]),
            Some(scopes[1]),
            Some(other_scope),
            Some(scopes[2]),
            Some(scopes[0]),
        ];
        outer_v.extend((0..BIG as i32).map(|i| Some(-10 - i)));
        outer_s.extend(scopes.iter().map(|s| Some(*s)));
        let subquery = || table(&[("v", sub_v.clone()), ("s", sub_s.clone())]);
        let outer = || {
            table(&[
                ("id", ids(outer_v.len())),
                ("ov", outer_v.clone()),
                ("os", outer_s.clone()),
            ])
        };
        // `outer.ov NOT IN (subquery.v WHERE subquery.s = outer.os)`, decided
        // for probe rows (RightAnti) and for build rows (LeftAnti, LeftMark)
        assert_spills_like_in_memory(
            subquery(),
            outer(),
            &[("v", "ov"), ("s", "os")],
            1,
            JoinType::RightAnti,
        )
        .await;
        for join_type in [JoinType::LeftAnti, JoinType::LeftMark] {
            assert_spills_like_in_memory(
                outer(),
                subquery(),
                &[("ov", "v"), ("os", "s")],
                1,
                join_type,
            )
            .await;
        }
    }

    /// A tuple `NOT IN` without correlation has no key to route by: the whole
    /// join is one pair, joined a chunk at a time
    #[tokio::test]
    async fn not_in_tuple_without_correlation_is_one_pair() {
        let mut sub_a: Vec<Option<i32>> = (0..BIG as i32).map(Some).collect();
        let mut sub_b: Vec<Option<i32>> = (0..BIG as i32).map(Some).collect();
        sub_a.push(None);
        sub_b.push(Some(-5));
        let mut outer_a = vec![Some(1), Some(1), Some(-7), Some(-8), None];
        let mut outer_b = vec![Some(1), None, None, Some(-5), Some(3)];
        // Enough rows to spill when the outer side is the build side
        outer_a.extend((0..BIG as i32).map(|i| Some(-10 - i)));
        outer_b.extend((0..BIG as i32).map(Some));
        let subquery = || table(&[("a", sub_a.clone()), ("b", sub_b.clone())]);
        let outer = || {
            table(&[
                ("id", ids(outer_a.len())),
                ("oa", outer_a.clone()),
                ("ob", outer_b.clone()),
            ])
        };
        assert_spills_like_in_memory(
            subquery(),
            outer(),
            &[("a", "oa"), ("b", "ob")],
            2,
            JoinType::RightAnti,
        )
        .await;
        for join_type in [JoinType::LeftAnti, JoinType::LeftMark] {
            assert_spills_like_in_memory(
                outer(),
                subquery(),
                &[("oa", "a"), ("ob", "b")],
                2,
                join_type,
            )
            .await;
        }
    }

    /// The join types a spilled pair that splitting cannot shrink must handle
    const ALL_JOIN_TYPES: [JoinType; 10] = [
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::LeftSemi,
        JoinType::RightSemi,
        JoinType::LeftAnti,
        JoinType::RightAnti,
        JoinType::LeftMark,
        JoinType::RightMark,
    ];

    /// Runs a `Partitioned` join of `k` = `k`, optionally filtered by
    /// `left.v < right.v`, in memory and under `pool`, and returns both
    /// results (sorted rows) and the spilling run's metrics
    async fn run_both(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        join_type: JoinType,
        filtered: bool,
        pool: CappedConsumerPool,
    ) -> (Vec<String>, Vec<String>, crate::metrics::MetricsSet) {
        let join = || {
            let on = vec![(
                Arc::new(Column::new_with_schema("k", &left.schema()).unwrap())
                    as PhysicalExprRef,
                Arc::new(Column::new_with_schema("k", &right.schema()).unwrap())
                    as PhysicalExprRef,
            )];
            let filter = filtered.then(|| {
                let expr = Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("v", 0)),
                    Operator::Lt,
                    Arc::new(Column::new("rv", 1)),
                )) as PhysicalExprRef;
                let schema = Schema::new(vec![
                    Field::new("v", DataType::Int32, true),
                    Field::new("rv", DataType::Int32, true),
                ]);
                JoinFilter::new(
                    expr,
                    vec![
                        ColumnIndex {
                            index: 1,
                            side: JoinSide::Left,
                        },
                        ColumnIndex {
                            index: 1,
                            side: JoinSide::Right,
                        },
                    ],
                    Arc::new(schema),
                )
            });
            Arc::new(
                HashJoinExecBuilder::new(
                    Arc::clone(left),
                    Arc::clone(right),
                    on,
                    join_type,
                )
                .with_filter(filter)
                .with_partition_mode(PartitionMode::Partitioned)
                .build()
                .unwrap(),
            )
        };
        let collect = |join: Arc<crate::joins::HashJoinExec>, ctx: Arc<TaskContext>| async move {
            let batches = crate::collect(Arc::clone(&join) as _, ctx).await.unwrap();
            let text = pretty_format_batches(&batches).unwrap().to_string();
            let mut rows: Vec<String> = text
                .lines()
                .filter(|line| line.starts_with('|'))
                .skip(1)
                .map(str::to_string)
                .collect();
            rows.sort();
            (rows, join.metrics().unwrap())
        };
        let (expected, _) = collect(join(), Arc::new(TaskContext::default())).await;
        let mut config = SessionConfig::new().with_batch_size(512);
        config.options_mut().execution.enable_hash_join_spilling = true;
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(pool))
            .build_arc()
            .unwrap();
        let ctx = TaskContext::default()
            .with_session_config(config)
            .with_runtime(runtime);
        let (actual, metrics) = collect(join(), Arc::new(ctx)).await;
        (expected, actual, metrics)
    }

    /// One hot key with `build_hot` build rows and `probe_hot` probe rows,
    /// plus keys only one side has and NULL keys on both sides
    fn hot_key_tables(
        build_hot: usize,
        probe_hot: usize,
    ) -> (Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>) {
        let side = |hot: usize, others: std::ops::Range<i32>| {
            let mut k: Vec<Option<i32>> = vec![Some(0); hot];
            k.extend(others.map(Some));
            k.push(None);
            let v = (0..k.len() as i32).map(|i| Some(i % 7)).collect();
            (k, v)
        };
        let (lk, lv) = side(build_hot, 1..200);
        let (rk, rv) = side(probe_hot, 100..300);
        (
            table(&[("k", lk), ("v", lv)]),
            table(&[("k", rk), ("rv", rv)]),
        )
    }

    fn sum(metrics: &crate::metrics::MetricsSet, name: &str) -> usize {
        metrics.sum_by_name(name).map_or(0, |m| m.as_usize())
    }

    /// Every build side overflows, and the spilled pieces get `spill` bytes
    fn spilling_pool(spill: usize) -> CappedConsumerPool {
        CappedConsumerPool::new()
            .with_cap("HashJoinInput", 1)
            .with_cap("HashJoinSpill", spill)
    }

    /// A hot key splitting cannot shrink, whose probe rows fit the kept
    /// memory: joined with the roles swapped, under the mirrored join type
    #[tokio::test]
    async fn hot_pair_with_few_probe_rows_is_reversed() {
        let (left, right) = hot_key_tables(1000, 3);
        for join_type in ALL_JOIN_TYPES {
            for filtered in [false, true] {
                let (expected, actual, metrics) = run_both(
                    &left,
                    &right,
                    join_type,
                    filtered,
                    spilling_pool(16 * 1024),
                )
                .await;
                assert_eq!(actual, expected, "{join_type} filtered={filtered}");
                assert!(
                    sum(&metrics, "spilled_pairs_reversed") > 0,
                    "{join_type} filtered={filtered}: {metrics}"
                );
            }
        }
    }

    /// A hot key splitting cannot shrink, with too many rows on both sides:
    /// joined a chunk of the judged side at a time
    #[tokio::test]
    async fn hot_pair_with_many_probe_rows_is_chunked() {
        // Small enough to keep the output small, too big for 2 KB either way
        let (left, right) = hot_key_tables(300, 300);
        for join_type in ALL_JOIN_TYPES {
            for filtered in [false, true] {
                let (expected, actual, metrics) =
                    run_both(&left, &right, join_type, filtered, spilling_pool(2 * 1024))
                        .await;
                assert_eq!(actual, expected, "{join_type} filtered={filtered}");
                assert!(
                    sum(&metrics, "spilled_pairs_chunked") > 0,
                    "{join_type} filtered={filtered}: {metrics}"
                );
            }
        }
    }
}

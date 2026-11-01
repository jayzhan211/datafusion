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

use arrow::array::{
    BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array, new_null_array,
};
use arrow::compute::{concat_batches, filter_record_batch, take_record_batch};
use arrow::datatypes::SchemaRef;
use datafusion_common::hash_utils::{RandomState, create_hashes};
use datafusion_common::utils::memory::{
    RecordBatchMemoryCounter, get_record_batch_memory_size,
};
use datafusion_common::{
    JoinSide, JoinType, NullEquality, Result, internal_err, resources_err,
};
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
    BuildOutcome, BuildOverflow, BuildParams, JoinLeftData, SpillTrigger,
    build_table_bytes, is_resources_exhausted,
};
use super::shared_bounds::SharedBuildAccumulator;
use super::stream::{
    BuildSide, BuildSideInitialState, HashJoinStream, HashJoinStreamState,
};
use crate::joins::PartitionMode;
use crate::joins::utils::{
    BuildProbeJoinMetrics, ColumnIndex, JoinFilter, OnceFut, need_produce_result_in_final,
};
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
    pub(super) build_spill: SpillManager,
    pub(super) probe_spill: SpillManager,
    pub(super) pool: Arc<dyn MemoryPool>,
    pub(super) trigger: SpillTrigger,
    pub(super) fetch: Option<usize>,
    pub(super) right_side_ordered: bool,
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
    let ctx = Arc::new(ctx);
    let output = stream::once(async move {
        let stream: SendableRecordBatchStream = match build.await? {
            BuildOutcome::InMemory(left_data) => Box::pin(ctx.stream.stream(
                owned(left_data),
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

    // The collected batches are covered by the reservation taken over: release
    // each one's share once its rows are copied into the buckets.
    let mut build = BucketedBuild::new(0);
    let mut collected = kept.buffers.size();
    for batch in batches {
        build.push(ctx, &kept, &batch)?;
        let released = get_record_batch_memory_size(&batch).min(collected);
        kept.buffers.shrink(released);
        collected -= released;
    }
    kept.buffers.shrink(collected);
    while let Some(batch) = rest.try_next().await? {
        build.push(ctx, &kept, &batch)?;
    }
    drop(rest);

    let (mut resident, mut spilled) = build.finish(ctx, &kept)?;
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
                    side.files.extend(file.finish()?);
                    spilled.push((bucket, side));
                }
            }
            ctx.build
                .build(&mut vec![], &mut kept.buffers.new_empty(), probe_streams)
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

    let mut buckets = [false; BUCKETS];
    for (bucket, _) in &spilled {
        buckets[*bucket] = true;
    }
    // The handle shared by the probe streams holds no rows itself
    let handle =
        ctx.build
            .build(&mut vec![], &mut kept.buffers.new_empty(), probe_streams)?;
    Ok(handle.with_spilled(Arc::new(SpilledBuild {
        resident: Mutex::new(Some(Arc::new(left_data))),
        kept: Mutex::new(Some(kept)),
        buckets,
        build: Mutex::new(spilled),
        probe: Mutex::new((0..BUCKETS).map(|_| SpilledSide::default()).collect()),
        routing: AtomicUsize::new(probe_streams),
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
    // Every stream starts before the last one finishes routing
    let (Some(left), Some(kept)) =
        (spilled.resident.lock().clone(), spilled.kept.lock().clone())
    else {
        return internal_err!("a probe stream started after the last one finished");
    };
    let routed = Arc::new(Mutex::new(empty_sides()));
    let router = SharedRouter(Arc::new(Mutex::new(ProbeRouter {
        input: right,
        on_right: ctx.stream.on_right.clone(),
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
        let joins = stream::iter(build)
            .then(move |(bucket, build)| {
                let probe = std::mem::take(&mut spilled.probe.lock()[bucket]);
                join_pair(Arc::clone(&ctx), Arc::clone(&kept), build, probe, 0)
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
    trigger: SpillTrigger,
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
        let granted = buffers.try_grow(bytes).is_ok();
        if std::env::var_os("DF_HJ_RUNG_LOG").is_some() {
            eprintln!(
                "HJ_KEEP take_over partition={partition} held={bytes} granted={granted}"
            );
        }
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
        if std::env::var_os("DF_HJ_RUNG_LOG").is_some() {
            eprintln!(
                "HJ_KEEP for_spilled_pairs largest={bytes} reserved={}",
                self.pool.reserved()
            );
        }
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
) -> BoxFuture<'static, Result<SendableRecordBatchStream>> {
    async move {
        let fits = join_bytes(build.rows, build.bytes) <= kept.capacity();
        if build.rows == 0 || fits || level >= MAX_LEVEL {
            let forced = if level >= MAX_LEVEL { forced_rung() } else { None };
            let joined = match forced {
                Some(_) => resources_err!("PoC: DF_HJ_RUNG skips the in-memory join"),
                None => join_in_memory(&ctx, &kept, &build, &probe).await,
            };
            match joined {
                Ok(stream) => return Ok(stream),
                // The estimate fell short: split further rather than fail
                Err(e)
                    if build.rows > 0
                        && level < MAX_LEVEL
                        && is_resources_exhausted(&e) => {}
                // Splitting cannot shrink this pair any further: hash its
                // probe rows instead if they fit, else hash it a chunk at a time
                Err(e) if is_resources_exhausted(&e) => {
                    let reversed = match forced.as_deref() {
                        Some(rung) => rung == "reversed",
                        None => join_bytes(probe.rows, probe.bytes) <= kept.capacity(),
                    };
                    if reversed {
                        rung_log("reversed", level, &build, &probe);
                        match join_reversed(&ctx, &kept, &build, &probe).await {
                            Ok(stream) => return Ok(stream),
                            Err(e) if is_resources_exhausted(&e) => {}
                            Err(e) => return Err(e),
                        }
                    }
                    rung_log("chunked", level, &build, &probe);
                    return Ok(join_chunked(&ctx, &kept, &build, &probe));
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
    let left_data = hash_side(&ctx.build_spill, build, &ctx.build, kept).await?;
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
        // Within the kept memory unless the pair cannot be split further,
        // which then needs the pool to grant the rest
        let size = get_record_batch_memory_size(&batch);
        if forced_rung().is_none() && kept.trigger.refuses(reservation.size(), size) {
            return resources_err!("PoC: test knob refused reloading a spilled side");
        }
        reservation.try_grow(size)?;
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
fn chunks(
    ctx: &Arc<SpillContext>,
    kept: &Arc<KeptMemory>,
    params: BuildParams,
    stream: StreamParams,
    swapped: bool,
    hashed: SpilledSide,
    streamed: SpilledSide,
) -> SendableRecordBatchStream {
    let schema = Arc::clone(&stream.schema);
    let (ctx, kept) = (Arc::clone(ctx), Arc::clone(kept));
    // Rows per chunk from the side's average row, so the chunk and its hash
    // table fit the kept memory
    let per_row = join_bytes(hashed.rows, hashed.bytes)
        .div_ceil(hashed.rows.max(1))
        .max(1);
    let chunk_rows = (kept.capacity() / per_row).max(1);
    let state = (None::<SendableRecordBatchStream>, None::<RecordBatch>);
    let joins = stream::try_unfold(state, move |(rows, pending)| {
        let (ctx, kept) = (Arc::clone(&ctx), Arc::clone(&kept));
        let (params, stream) = (params.clone(), stream.clone());
        let (hashed, streamed) = (hashed.clone(), streamed.clone());
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
            let mut pending = pending;
            let mut batches = vec![];
            let mut taken = 0;
            while taken < chunk_rows {
                let (batch, sliced) = match pending.take() {
                    Some(batch) => (batch, true),
                    None => match rows.try_next().await? {
                        Some(batch) => (batch, false),
                        None => break,
                    },
                };
                let take = batch.num_rows().min(chunk_rows - taken);
                let piece = batch.slice(0, take);
                // A slice holds its whole batch: copy it, so the chunk holds
                // (and is charged for) only its own rows
                let piece = if sliced || take < batch.num_rows() {
                    let rows = UInt32Array::from_iter_values(0..take as u32);
                    take_record_batch(&piece, &rows)?
                } else {
                    piece
                };
                if take < batch.num_rows() {
                    pending = Some(batch.slice(take, batch.num_rows() - take));
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
            reservation.try_grow(batches.iter().map(|b| counter.count_batch(b)).sum())?;
            let left_data = params.build(&mut batches, &mut reservation, 1)?;
            if std::env::var_os("DF_HJ_RUNG_LOG").is_some() {
                eprintln!("HJ_CHUNK rows={taken} swapped={swapped}");
            }
            let streamed_rows = read_side(streamed_spill, &streamed)?;
            let joined: SendableRecordBatchStream = Box::pin(stream.stream(
                owned(left_data),
                streamed_rows,
                None,
                false,
                None,
            ));
            Ok(Some((joined, (Some(rows), pending))))
        }
    })
    .try_flatten();
    Box::pin(RecordBatchStreamAdapter::new(schema, joins))
}

/// PoC: DF_HJ_RUNG_LOG=1 prints which rung joins an unsplittable pair
/// PoC test knob (`DF_HJ_RUNG=reversed|chunked`): the rung every pair at the
/// depth cap takes, so the fuzz covers each rung for every join type
fn forced_rung() -> Option<String> {
    std::env::var("DF_HJ_RUNG").ok()
}

fn rung_log(rung: &str, level: usize, build: &SpilledSide, probe: &SpilledSide) {
    if std::env::var_os("DF_HJ_RUNG_LOG").is_some() {
        eprintln!(
            "HJ_RUNG {rung} level={level} build_rows={} probe_rows={}",
            build.rows, probe.rows
        );
    }
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

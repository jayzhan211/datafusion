// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Memory pools for tests

use std::collections::HashMap;

use datafusion_common::{Result, resources_datafusion_err};
use datafusion_execution::memory_pool::{
    MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation, UnboundedMemoryPool,
};
use parking_lot::Mutex;

/// A pool that refuses each consumer whose name starts with a capped prefix
/// more than that cap, and grants everything else.
///
/// Spilling operators spill when the pool refuses them, so this makes one
/// operator spill at a chosen size without starving the rest of the plan.
/// A hash join, for example, spills its build side past the cap on
/// `HashJoinInput` and joins its spilled pieces within the cap on
/// `HashJoinSpill`.
#[derive(Debug, Default)]
pub struct CappedConsumerPool {
    inner: UnboundedMemoryPool,
    caps: Vec<(String, usize)>,
    /// What each capped consumer holds, by consumer id
    held: Mutex<HashMap<usize, usize>>,
}

impl CappedConsumerPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Caps each consumer whose name starts with `prefix` at `bytes`
    pub fn with_cap(mut self, prefix: impl Into<String>, bytes: usize) -> Self {
        self.caps.push((prefix.into(), bytes));
        self
    }

    fn cap(&self, consumer: &MemoryConsumer) -> Option<usize> {
        self.caps
            .iter()
            .find(|(prefix, _)| consumer.name().starts_with(prefix.as_str()))
            .map(|(_, bytes)| *bytes)
    }
}

impl std::fmt::Display for CappedConsumerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(caps: {:?})", self.name(), self.caps)
    }
}

impl MemoryPool for CappedConsumerPool {
    fn name(&self) -> &str {
        "capped_consumer"
    }

    fn register(&self, consumer: &MemoryConsumer) {
        self.inner.register(consumer)
    }

    fn unregister(&self, consumer: &MemoryConsumer) {
        self.held.lock().remove(&consumer.id());
        self.inner.unregister(consumer)
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        if self.cap(reservation.consumer()).is_some() {
            *self
                .held
                .lock()
                .entry(reservation.consumer().id())
                .or_default() += additional;
        }
        self.inner.grow(reservation, additional)
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        if self.cap(reservation.consumer()).is_some()
            && let Some(held) = self.held.lock().get_mut(&reservation.consumer().id())
        {
            *held = held.saturating_sub(shrink);
        }
        self.inner.shrink(reservation, shrink)
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        let consumer = reservation.consumer();
        if let Some(cap) = self.cap(consumer) {
            let mut held = self.held.lock();
            let held = held.entry(consumer.id()).or_default();
            if *held + additional > cap {
                return Err(resources_datafusion_err!(
                    "Failed to allocate additional {additional} bytes for {}: {held} of its {cap} bytes held",
                    consumer.name()
                ));
            }
            *held += additional;
        }
        self.inner.try_grow(reservation, additional)
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    /// Bounded, so that operators which spill only under a limit do
    fn memory_limit(&self) -> MemoryLimit {
        MemoryLimit::Finite(usize::MAX)
    }
}

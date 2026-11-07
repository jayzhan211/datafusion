// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! An optimizer rule that detects aggregate operations that could use a limited bucket count

use std::sync::Arc;

use crate::PhysicalOptimizerRule;
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateInputMode};
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use itertools::Itertools;

/// An optimizer rule that passes a `limit` hint to aggregations if the whole result is not needed
#[derive(Debug)]
pub struct TopKAggregation {}

impl TopKAggregation {
    /// Create a new `LimitAggregation`
    pub fn new() -> Self {
        Self {}
    }

    /// Push the bound into eligible aggregates to retain fewer groups; keep the
    /// sort to enforce the final ordering and row count.
    ///
    /// ```txt
    /// Before:
    /// Sort(fetch=K)
    ///   Aggregate
    ///     input
    ///
    /// After:
    /// Sort(fetch=K)
    ///   Aggregate(TopK(k))
    ///     input
    /// ```
    fn transform_sort(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        let sort = plan.downcast_ref::<SortExec>()?;

        let children = sort.children();
        let child = children.into_iter().exactly_one().ok()?;
        let order = sort.properties().output_ordering()?;
        let order = order.iter().exactly_one().ok()?;
        let sort_options = order.options;
        let order = order.expr.downcast_ref::<Column>()?;
        let mut sort_col_name = order.name().to_string();
        let mut sort_col_index = Some(order.index());
        let limit = sort.fetch()?;

        let mut cardinality_preserved = true;
        let closure = |plan: Arc<dyn ExecutionPlan>| {
            if !cardinality_preserved {
                return Ok(Transformed::no(plan));
            }
            if let Some(aggr) = plan.downcast_ref::<AggregateExec>() {
                let optimized =
                    aggr.clone()
                        .try_optimize_topk(limit, &sort_col_name, sort_options);
                // A final stage and its partial stage are one aggregate, so the limit
                // goes on both. Below the stage that reads raw input, the key is gone.
                if optimized.is_none()
                    || aggr.mode().input_mode() == AggregateInputMode::Raw
                {
                    cardinality_preserved = false;
                }
                sort_col_index = None;
                if let Some(aggr) = optimized {
                    return Ok(Transformed::yes(Arc::new(aggr.data)));
                }
            } else if let Some(proj) = plan.downcast_ref::<ProjectionExec>() {
                // Follow the sort key through the projection by position: names
                // can repeat, and an alias can reuse a name from below it.
                let src_col = sort_col_index
                    .and_then(|index| proj.expr().get(index))
                    .and_then(|proj_expr| proj_expr.expr.downcast_ref::<Column>());
                let Some(src_col) = src_col else {
                    cardinality_preserved = false;
                    return Ok(Transformed::no(plan));
                };
                sort_col_name = src_col.name().to_string();
                sort_col_index = Some(src_col.index());
            } else if !keeps_input_columns(&plan) {
                // The key's position means nothing below here. This also stops at
                // a window, which reads rows beyond the ones the limit keeps.
                cardinality_preserved = false;
            } else {
                // or we continue down through types that don't reduce cardinality
                match plan.cardinality_effect() {
                    CardinalityEffect::Equal | CardinalityEffect::GreaterEqual => {}
                    CardinalityEffect::Unknown | CardinalityEffect::LowerEqual => {
                        cardinality_preserved = false;
                    }
                }
            }
            Ok(Transformed::no(plan))
        };
        let child = Arc::clone(child).transform_down(closure).data().ok()?;
        let sort = SortExec::new(sort.expr().clone(), child)
            .with_fetch(sort.fetch())
            .with_preserve_partitioning(sort.preserve_partitioning());
        Some(Arc::new(sort))
    }
}

/// Whether `plan` has one input and outputs that input's columns, in the same positions.
fn keeps_input_columns(plan: &Arc<dyn ExecutionPlan>) -> bool {
    matches!(plan.children().as_slice(), [input] if input.schema().fields() == plan.schema().fields())
}

impl Default for TopKAggregation {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalOptimizerRule for TopKAggregation {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if config.optimizer.enable_topk_aggregation {
            plan.transform_down(|plan| {
                Ok(if let Some(plan) = TopKAggregation::transform_sort(&plan) {
                    Transformed::yes(plan)
                } else {
                    Transformed::no(plan)
                })
            })
            .data()
        } else {
            Ok(plan)
        }
    }

    fn name(&self) -> &str {
        "LimitAggregation"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

// see `aggregate.slt` for tests

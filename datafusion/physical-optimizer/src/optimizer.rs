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

//! [`PhysicalOptimizer`] for optimizing [`ExecutionPlan`]s
//!
//! [`ExecutionPlan`]: datafusion_physical_plan::ExecutionPlan

use std::fmt::Debug;
use std::sync::Arc;

use crate::combine_partial_final_agg::CombinePartialFinalAggregate;
use crate::ensure_coop::EnsureCooperative;
use crate::filter_pushdown::FilterPushdown;
use crate::limit_pushdown::LimitPushdown;
use crate::output_requirements::OutputRequirements;
use crate::projection_pushdown::ProjectionPushdown;
use crate::sanity_checker::SanityCheckPlan;
use crate::topk_aggregation::TopKAggregation;
use crate::topk_repartition::TopKRepartition;
use crate::update_aggr_exprs::OptimizeAggregateOrder;

use crate::hash_join_buffering::HashJoinBuffering;
use crate::limit_pushdown_past_window::LimitPushPastWindows;
use crate::pushdown_sort::PushdownSort;
use datafusion_common::config::ConfigOptions;
use datafusion_physical_plan::statistics::StatisticsContext;

// Re-export from this module for backwards compatibility.
pub use datafusion_session::{PhysicalOptimizerContext, PhysicalOptimizerRule};

pub use datafusion_session::with_statistics_context;

/// Simple context wrapping [`ConfigOptions`] for backward compatibility.
///
/// This struct provides a minimal implementation of [`PhysicalOptimizerContext`]
/// that supplies configuration options and a [`StatisticsContext`] without a
/// statistics registry. Used when no statistics registry is available or
/// needed.
pub struct ConfigOnlyContext<'a> {
    config: &'a ConfigOptions,
    statistics_context: StatisticsContext,
}

impl<'a> ConfigOnlyContext<'a> {
    /// Create a new context wrapping the given config options.
    pub fn new(config: &'a ConfigOptions) -> Self {
        Self {
            config,
            // No providers, matching `statistics_registry()`, which is `None`
            statistics_context: StatisticsContext::new(),
        }
    }
}

impl PhysicalOptimizerContext for ConfigOnlyContext<'_> {
    fn config_options(&self) -> &ConfigOptions {
        self.config
    }

    fn statistics_context(&self) -> Option<&StatisticsContext> {
        Some(&self.statistics_context)
    }
}

/// A rule-based physical optimizer.
///
/// Applies a *Pipeline* (an ordered list) of [`PhysicalOptimizerRule`]s in a single pass.
///
/// # Physical Optimizer Contract
///
/// [`PhysicalOptimizer::new`] defines the default sequence of
/// [`PhysicalOptimizerRule`]s. The optimizer applies them in order, in a single
/// pass. This section describes the contract between the optimizer rules, both
/// for projects that extend or reorder the default rule list and for
/// contributors adding new built-in rules.
///
/// The [`PhysicalAnalyzer`] phase runs before these rules and ends with
/// [`EnsureRequirements`], so the first rule already receives a valid plan. A
/// rule whose rewrite breaks a requirement can repair the plan with
/// [`satisfy_requirements`].
///
/// [`PhysicalAnalyzer`]: crate::analyzer::PhysicalAnalyzer
/// [`EnsureRequirements`]: crate::ensure_requirements::EnsureRequirements
/// [`satisfy_requirements`]: crate::ensure_requirements::satisfy_requirements
///
/// ## Invariants
///
/// The following properties hold for all rules (built-in and custom):
///
/// 1. **Correctness.** The returned plan must a) be valid (it can be executed
///    without internal errors, and it meets every distribution and ordering
///    requirement its operators declare, as [`SanityCheckPlan`] checks) and
///    b) produce the same results as the input plan. A rule may change the operators in the plan, such as changing
///    which operator implements a join, but it may never change the query's
///    semantics.
///
/// 2. **Reorderable.** Because each rule preserves correctness, running the
///    rules in a different order must still yield a valid plan that produces
///    the same results.
///
/// ### Exceptions
///
/// Some built-in rules do not yet fully satisfy the invariants above:
///
/// - [`OutputRequirements`] in remove mode drops the markers that record the
///   query's output requirements, so the rules after it no longer see them.
/// - [`SanityCheckPlan`] rejects plans whose requirements are not satisfied.
///   It should run last.
///
/// Custom pipelines must keep these rules in a compatible relative order. We
/// are working to remove these exceptions, so that all rules can be reordered
/// freely.
///
/// ## Optimization Coverage Is Not Guaranteed
///
/// The invariants above guarantee *correctness*, not that every optimization is
/// applied. Built-in rules often match specific plan shapes that may only occur
/// at a certain point in the default pipeline. Adding, removing, or reordering
/// rules can change plan shapes, so a built-in rule may silently miss an
/// optimization that would apply in the default order. To keep their
/// implementations simple, built-in rules typically only recognize shapes that
/// the default pipeline produces.
///
/// For example, [`CombinePartialFinalAggregate`] runs after
/// [`EnsureRequirements`] and only merges a `Partial` aggregate that is the
/// direct child of a `Final` aggregate.
///
/// ```text
/// -- Canonical shape (default pipeline):
/// -- built-in rules recognize and optimize this
/// Limit
///   Aggregate(Final)
///     Aggregate(Partial)
///
/// -- Non-canonical shape (repartitioning inserted first):
/// -- built-in rules skip the optimization
/// Limit
///   Aggregate(Final)
///     Repartition
///       Aggregate(Partial)
/// ```
///
/// ## Guidance for Custom Pipelines
///
/// We recommend keeping the default rule order and using the supported
/// configuration options to disable individual optimizations. For example, `SET
/// datafusion.optimizer.enable_distinct_aggregation_soft_limit = false`
/// disables the distinct aggregation soft-limit optimization. These variations
/// are tested and preserve both correctness and optimization coverage.
/// See <https://github.com/apache/datafusion/issues/26481> for more details.
///
/// If you do build a custom pipeline:
///
/// - **End a custom analyzer list with [`EnsureRequirements`],** or repair the
///   plan with [`satisfy_requirements`]. Debug builds check this.
/// - **Reordering must still yield a valid plan.** Downstream projects are
///   encouraged to add tests that run their pipeline end to end and check
///   results, as DataFusion's own tests only cover the default order.
/// - **Test for reduced optimization coverage,** as described above, and verify
///   the plans you care about with `EXPLAIN`.
/// - **Prefer adding a rule over extending a built-in one.** We discourage
///   extending a built-in rule to match plan shapes it will not encounter in
///   the default pipeline, because such extensions are difficult to test and
///   maintain. Instead, add a custom rule that rewrites your plan into the
///   canonical shape recognized by the built-in rules. If extending a built-in
///   rule still seems necessary, please open a DataFusion issue to discuss it.
#[derive(Clone, Debug)]
pub struct PhysicalOptimizer {
    /// All rules to apply
    pub rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>,
}

impl Default for PhysicalOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalOptimizer {
    /// Create a new optimizer using the recommended list of rules
    pub fn new() -> Self {
        // PoC: the lists come from `poc_placement`, which `DF_POC` rearranges.
        let (_, rules) = crate::poc_placement::pipeline();
        Self::with_rules(rules)
    }

    /// Create a new optimizer with the given rules
    ///
    /// See the [`PhysicalOptimizer`](Self) documentation for the contract
    /// governing non-default optimizer pipelines.
    pub fn with_rules(rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>) -> Self {
        Self { rules }
    }
}

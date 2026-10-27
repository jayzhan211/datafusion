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

//! Physical analyzer interfaces.

use std::fmt::Debug;
use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_physical_plan::ExecutionPlan;

use crate::physical_optimizer::PhysicalOptimizerContext;

/// A `PhysicalAnalyzerRule` is a correctness transformation: it makes an
/// [`ExecutionPlan`] executable, as a logical [`AnalyzerRule`] makes a logical
/// plan valid. The analyzer list ends with `EnsureRequirements`, which
/// repartitions and sorts the inputs to match
/// [`ExecutionPlan::required_input_distribution`] and
/// [`ExecutionPlan::required_input_ordering`], so every
/// [`PhysicalOptimizerRule`] after it receives a valid plan and must return
/// one.
///
/// Some performance rules still have to run before `EnsureRequirements`, such
/// as `WindowTopN` and the statistics-based choices in `join_selection`, so
/// they are analyzer rules too. Performance rules that do not have to, such as
/// `LimitPushdown`, are [`PhysicalOptimizerRule`]s (see also
/// [`OptimizerRule`]). A rule is one or the other: moving it to the other
/// phase means implementing the other trait.
///
/// An analyzer rule must:
///
/// - Keep the query's results unchanged. It may leave a distribution or
///   ordering requirement unmet; `EnsureRequirements` at the end of the phase
///   meets it.
/// - Stay correct when the plan already contains repartitions, sorts, or
///   operators it does not know, as happens when a plan is optimized again.
///   It may skip its rewrite there, and decides what is safe around an
///   operator only from what the operator declares.
///
/// # Where analyzer rules run
///
/// [`DefaultPhysicalPlanner::optimize_physical_plan`] runs every analyzer rule,
/// then every optimizer rule. A rule appended to the analyzer list runs after
/// `EnsureRequirements`, so it must leave the plan valid. At the end of the
/// analyzer phase, debug builds check that the plan is valid.
///
/// ```text
/// ExecutionPlan  (every operator declares its distribution and ordering
///                 requirements; nothing satisfies them yet)
///                                │
///                                ▼
/// ┌─ PhysicalAnalyzer phase ── make the plan executable ─────────┐
/// │ rules that must run before EnsureRequirements                │
/// │ EnsureRequirements: enforce distribution, ordering           │
/// └──────────────────────────────┬───────────────────────────────┘
///                                │  the plan is valid
///                                ▼
/// ┌─ PhysicalOptimizer phase ── make the plan faster ────────────┐
/// │ every rule: valid plan in, valid plan out                    │
/// └──────────────────────────────┬───────────────────────────────┘
///                                ▼
///                   valid, optimized ExecutionPlan
/// ```
///
/// [`DefaultPhysicalPlanner::optimize_physical_plan`]: https://docs.rs/datafusion/latest/datafusion/physical_planner/struct.DefaultPhysicalPlanner.html#method.optimize_physical_plan
/// [`AnalyzerRule`]: https://docs.rs/datafusion/latest/datafusion/optimizer/analyzer/trait.AnalyzerRule.html
/// [`OptimizerRule`]: https://docs.rs/datafusion/latest/datafusion/optimizer/trait.OptimizerRule.html
///
/// [`PhysicalOptimizerRule`]: crate::physical_optimizer::PhysicalOptimizerRule
/// [`ExecutionPlan::required_input_distribution`]: datafusion_physical_plan::ExecutionPlan::required_input_distribution
/// [`ExecutionPlan::required_input_ordering`]: datafusion_physical_plan::ExecutionPlan::required_input_ordering
pub trait PhysicalAnalyzerRule: Debug + std::any::Any {
    /// Rewrite `plan`. See the trait docs for what the result must keep.
    ///
    /// This is the primary method. Rules that need access to the statistics
    /// registry should override [`analyze_with_context`](Self::analyze_with_context)
    /// instead.
    fn analyze(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>>;

    /// Rewrite `plan` with access to extended context (statistics registry, etc.).
    ///
    /// The default implementation calls [`analyze`](Self::analyze) with the
    /// config options from the context. This mirrors
    /// [`PhysicalOptimizerRule::optimize_with_context`], so analyzer rules
    /// have the same statistics-registry access as optimizer rules.
    ///
    /// [`PhysicalOptimizerRule::optimize_with_context`]: crate::physical_optimizer::PhysicalOptimizerRule::optimize_with_context
    fn analyze_with_context(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        context: &dyn PhysicalOptimizerContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.analyze(plan, context.config_options())
    }

    /// A human readable name for this analyzer rule.
    fn name(&self) -> &str;

    /// A flag to indicate whether the physical planner should validate that
    /// the rule will not change the schema of the plan after the rewrite.
    ///
    /// This mirrors [`PhysicalOptimizerRule::schema_check`]; the default is
    /// `true`.
    ///
    /// [`PhysicalOptimizerRule::schema_check`]: crate::physical_optimizer::PhysicalOptimizerRule::schema_check
    fn schema_check(&self) -> bool {
        true
    }
}

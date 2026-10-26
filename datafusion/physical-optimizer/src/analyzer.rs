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

//! Physical analyzer

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_physical_plan::ExecutionPlan;

use crate::PhysicalOptimizerRule;
use crate::aggregate_statistics::AggregateStatistics;
use crate::eliminate_constant_join::EliminateConstantJoin;
use crate::ensure_requirements::EnsureRequirements;
use crate::filter_pushdown::FilterPushdown;
use crate::join_selection::JoinSelection;
use crate::limited_distinct_aggregation::LimitedDistinctAggregation;
use crate::optimizer::PhysicalOptimizerContext;
use crate::window_topn::WindowTopN;

// Re-export from this module for convenience.
pub use datafusion_session::PhysicalAnalyzerRule;

/// A rule-based physical analyzer.
///
/// The analyzer phase makes the plan executable and ends with
/// `EnsureRequirements`, so every rule of the
/// [`PhysicalOptimizer`](crate::optimizer::PhysicalOptimizer) phase receives a
/// valid plan. See [`PhysicalAnalyzerRule`] for details.
#[derive(Clone, Debug)]
pub struct PhysicalAnalyzer {
    /// All rules to apply
    pub rules: Vec<Arc<dyn PhysicalAnalyzerRule + Send + Sync>>,
}

impl Default for PhysicalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalAnalyzer {
    /// Create a new analyzer using the recommended list of rules
    pub fn new() -> Self {
        // `EnsureRequirements`, after the rules that must run before it today.
        // See `PhysicalAnalyzerRule`.
        //
        // Every built-in rule is still a `PhysicalOptimizerRule`, so each one
        // runs through `RunInAnalyzer`, the correctness rules included:
        // `OutputRequirements` (add) records the query's output requirement,
        // `JoinSelection` resolves `PartitionMode::Auto`, and
        // `EnsureRequirements` satisfies every requirement. The others are
        // performance rules that must run before `EnsureRequirements`.
        //
        // <https://github.com/apache/datafusion/issues/26481> plans to move the
        // correctness work out of these rules: the planner records the output
        // requirement and creates joins in an executable mode, and the
        // insert-only part of `EnsureRequirements` becomes a
        // `PhysicalAnalyzerRule`. `RunInAnalyzer` then wraps only performance
        // rules that must run before enforcement.
        let rules: Vec<Arc<dyn PhysicalAnalyzerRule + Send + Sync>> = vec![
            Arc::new(RunInAnalyzer::new(Arc::new(AggregateStatistics::new()))),
            // Runs after AggregateStatistics, which can fold a scalar subquery into a
            // row of literals, and before EnsureRequirements, so that the filter
            // replacing a join against such a row keeps the other input's ordering.
            // Also runs before FilterPushdown, which pushes that filter into the scan.
            Arc::new(RunInAnalyzer::new(Arc::new(EliminateConstantJoin::new()))),
            // Statistics-based join selection will change the Auto mode to a real join implementation,
            // like collect left, or hash join, or future sort merge join, which will influence the
            // EnsureRequirements rule as it decides whether to add additional repartitioning and
            // local sorting steps to meet distribution and ordering requirements. Therefore, it
            // should run before EnsureRequirements.
            Arc::new(RunInAnalyzer::new(Arc::new(JoinSelection::new()))),
            // The LimitedDistinctAggregation rule should be applied before EnsureRequirements,
            // as that rule may inject other operations in between the different AggregateExecs.
            // Applying the rule early means only directly-connected AggregateExecs must be examined.
            Arc::new(RunInAnalyzer::new(Arc::new(
                LimitedDistinctAggregation::new(),
            ))),
            // The FilterPushdown rule tries to push down filters as far as it can.
            // For example, it will push down filtering from a `FilterExec` to `DataSourceExec`.
            // Note that this does not push down dynamic filters (such as those created by a `SortExec` operator in TopK mode),
            // those are handled by `FilterPushdown(Post)` in the optimizer list.
            // See `FilterPushdownPhase` for more details.
            Arc::new(RunInAnalyzer::new(Arc::new(FilterPushdown::new()))),
            // WindowTopN: replaces Filter(rn<=K) → Window(ROW_NUMBER) with
            // PartitionedTopKExec(fetch=K), which emits `rn` itself; the
            // window node is kept above it only when it has other window
            // expressions to compute.
            // Must run before EnsureRequirements (so it can rewrite against the
            // window's declared ordering without pattern-matching a SortExec)
            // and before ProjectionPushdown (which embeds projections into FilterExec).
            Arc::new(RunInAnalyzer::new(Arc::new(WindowTopN::new()))),
            // Ensures each input plan satisfies the distribution and ordering
            // requirements declared by `ExecutionPlan::required_input_distribution`
            // and `ExecutionPlan::required_input_ordering`.
            //
            // If the requirements are already satisfied, this rule leaves the plan
            // unchanged. For example, it does not add sorting when the input is a
            // file scan whose existing order already satisfies the required ordering.
            // Otherwise, this rule inserts the necessary repartitioning and sorting
            // operators.
            //
            // This used to be implemented as two separate rules: `EnforceDistribution`
            // and `EnforceSorting`. It is now a single idempotent rule that decides
            // distribution and sorting together in one bottom-up pass, so the
            // `pushdown_sorts` step no longer breaks distribution invariants set
            // earlier in the pipeline. See the module-level doc on
            // [`EnsureRequirements`](crate::ensure_requirements) for the per-phase
            // breakdown, and <https://github.com/apache/datafusion/issues/21973>
            // for the original failure mode.
            Arc::new(RunInAnalyzer::new(Arc::new(EnsureRequirements::new()))),
        ];
        Self::with_rules(rules)
    }

    /// Create a new analyzer with the given rules
    pub fn with_rules(rules: Vec<Arc<dyn PhysicalAnalyzerRule + Send + Sync>>) -> Self {
        Self { rules }
    }
}

/// Runs a [`PhysicalOptimizerRule`] as a [`PhysicalAnalyzerRule`].
///
/// Use it for a performance rule that must run before `EnsureRequirements`,
/// built-in or your own. Every built-in rule runs through it today, the
/// correctness rules included, because none of them implements
/// [`PhysicalAnalyzerRule`] yet; see [`PhysicalAnalyzer::new`].
#[derive(Debug)]
pub struct RunInAnalyzer {
    rule: Arc<dyn PhysicalOptimizerRule + Send + Sync>,
}

impl RunInAnalyzer {
    /// Run `rule` in the analyzer phase.
    pub fn new(rule: Arc<dyn PhysicalOptimizerRule + Send + Sync>) -> Self {
        Self { rule }
    }
}

impl PhysicalAnalyzerRule for RunInAnalyzer {
    fn analyze(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.rule.optimize(plan, config)
    }

    fn analyze_with_context(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        context: &dyn PhysicalOptimizerContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.rule.optimize_with_context(plan, context)
    }

    fn name(&self) -> &str {
        self.rule.name()
    }

    fn schema_check(&self) -> bool {
        self.rule.schema_check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::PhysicalOptimizer;

    /// The default analyzer is the former optimizer prefix, in the same order,
    /// ending with enforcement.
    #[test]
    fn default_analyzer_ends_with_enforcement() {
        let analyzer = PhysicalAnalyzer::new();
        let names: Vec<&str> = analyzer.rules.iter().map(|r| r.name()).collect();
        assert_eq!(
            names,
            vec![
                "OutputRequirements",
                "aggregate_statistics",
                "EliminateConstantJoin",
                "join_selection",
                "LimitedDistinctAggregation",
                "FilterPushdown",
                "WindowTopN",
                "EnsureRequirements",
            ]
        );
    }

    /// The rules that moved to the analyzer are not run a second time by the
    /// default optimizer. `OutputRequirements` is the exception by design: its
    /// add mode runs in the analyzer and its remove mode in the optimizer.
    #[test]
    fn default_optimizer_does_not_repeat_analyzer_rules() {
        let analyzer_names: Vec<String> = PhysicalAnalyzer::new()
            .rules
            .iter()
            .map(|r| r.name().to_string())
            .filter(|n| n != "OutputRequirements")
            .collect();
        let optimizer_names: Vec<String> = PhysicalOptimizer::new()
            .rules
            .iter()
            .map(|r| r.name().to_string())
            .collect();
        for name in analyzer_names {
            assert!(
                !optimizer_names.contains(&name),
                "{name} runs in the analyzer and must not repeat in the optimizer, got {optimizer_names:?}"
            );
        }
    }
}

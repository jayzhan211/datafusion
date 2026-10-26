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

use crate::PhysicalOptimizerRule;
use crate::aggregate_statistics::AggregateStatistics;
use crate::eliminate_constant_join::EliminateConstantJoin;
use crate::ensure_requirements::EnsureRequirements;
use crate::filter_pushdown::FilterPushdown;
use crate::join_selection::JoinSelection;
use crate::limited_distinct_aggregation::LimitedDistinctAggregation;
use crate::output_requirements::OutputRequirements;
use crate::window_topn::WindowTopN;

/// The physical analyzer: the rules that make an `ExecutionPlan` executable,
/// as the logical analyzer makes a logical plan valid.
///
/// The list ends with [`EnsureRequirements`], which repartitions and sorts the
/// inputs to match each operator's required distribution and ordering, so
/// every rule of the [`PhysicalOptimizer`](crate::optimizer::PhysicalOptimizer)
/// list receives a valid plan and must return one. Both lists hold
/// [`PhysicalOptimizerRule`]s; the list a rule is in decides its contract.
///
/// Some performance rules still have to run before `EnsureRequirements`, such
/// as `WindowTopN` and the statistics-based choices in `JoinSelection`, so they
/// are in this list too. Performance rules that do not have to, such as
/// `LimitPushdown`, are in the optimizer list.
///
/// A rule in this list must:
///
/// - Keep the query's results unchanged. It may leave a distribution or
///   ordering requirement unmet; `EnsureRequirements` at the end of the list
///   meets it.
/// - Stay correct when the plan already contains repartitions, sorts, or
///   operators it does not know, as happens when a plan is optimized again.
///   It may skip its rewrite there, and decides what is safe around an
///   operator only from what the operator declares.
///
/// A rule appended after `EnsureRequirements` must leave the plan valid. At
/// the end of the analyzer list, debug builds check that the plan is valid.
#[derive(Clone, Debug)]
pub struct PhysicalAnalyzer {
    /// All rules to apply
    pub rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>,
}

impl Default for PhysicalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalAnalyzer {
    /// Create a new analyzer using the recommended list of rules
    pub fn new() -> Self {
        // `OutputRequirements` (add) records the query's output requirement,
        // `JoinSelection` resolves `PartitionMode::Auto`, and
        // `EnsureRequirements` satisfies every requirement. The others are
        // performance rules that must run before `EnsureRequirements`.
        let rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> = vec![
            // If there is a output requirement of the query, make sure that
            // this information is not lost across different rules during optimization.
            Arc::new(OutputRequirements::new_add_mode()),
            Arc::new(AggregateStatistics::new()),
            // Runs after AggregateStatistics, which can fold a scalar subquery into a
            // row of literals, and before EnsureRequirements, so that the filter
            // replacing a join against such a row keeps the other input's ordering.
            // Also runs before FilterPushdown, which pushes that filter into the scan.
            Arc::new(EliminateConstantJoin::new()),
            // Statistics-based join selection will change the Auto mode to a real join implementation,
            // like collect left, or hash join, or future sort merge join, which will influence the
            // EnsureRequirements rule as it decides whether to add additional repartitioning and
            // local sorting steps to meet distribution and ordering requirements. Therefore, it
            // should run before EnsureRequirements.
            Arc::new(JoinSelection::new()),
            // The LimitedDistinctAggregation rule should be applied before EnsureRequirements,
            // as that rule may inject other operations in between the different AggregateExecs.
            // Applying the rule early means only directly-connected AggregateExecs must be examined.
            Arc::new(LimitedDistinctAggregation::new()),
            // The FilterPushdown rule tries to push down filters as far as it can.
            // For example, it will push down filtering from a `FilterExec` to `DataSourceExec`.
            // Note that this does not push down dynamic filters (such as those created by a `SortExec` operator in TopK mode),
            // those are handled by `FilterPushdown(Post)` in the optimizer list.
            // See `FilterPushdownPhase` for more details.
            Arc::new(FilterPushdown::new()),
            // WindowTopN: replaces Filter(rn<=K) → Window(ROW_NUMBER) with
            // PartitionedTopKExec(fetch=K), which emits `rn` itself; the
            // window node is kept above it only when it has other window
            // expressions to compute.
            // Must run before EnsureRequirements (so it can rewrite against the
            // window's declared ordering without pattern-matching a SortExec)
            // and before ProjectionPushdown (which embeds projections into FilterExec).
            Arc::new(WindowTopN::new()),
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
            Arc::new(EnsureRequirements::new()),
        ];
        Self::with_rules(rules)
    }

    /// Create a new analyzer with the given rules
    pub fn with_rules(rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>) -> Self {
        Self { rules }
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

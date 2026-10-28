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
use crate::output_requirements::OutputRequirements;
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
        // PoC: the lists come from `poc_placement`, which `DF_POC` rearranges.
        let (rules, _) = crate::poc_placement::pipeline();
        let rules = rules
            .into_iter()
            .map(|rule| {
                Arc::new(RunInAnalyzer::new(rule))
                    as Arc<dyn PhysicalAnalyzerRule + Send + Sync>
            })
            .collect();
        Self::with_rules(rules)
    }

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

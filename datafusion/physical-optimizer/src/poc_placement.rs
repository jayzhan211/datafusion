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

//! PoC only: rearranges the default analyzer and optimizer lists from the
//! `DF_POC` environment variable, so one build can run every placement
//! experiment.
//!
//! `DF_POC` is a comma-separated list of directives:
//! - `move:KEY:after:OTHER` / `move:KEY:before:OTHER`: move rule `KEY` next to
//!   rule `OTHER`. `move:KEY:after:ER` puts it first in the optimizer list.
//! - `drop:KEY`: remove rule `KEY`.
//! - `js_split`: replace `join_selection` with its runnability pass followed
//!   by its statistics pass, as two rules.
//! - `js_planner`: the planner chooses join modes when it creates each join,
//!   and `join_selection` is removed from the lists.

use std::sync::Arc;

use crate::PhysicalOptimizerRule;
use crate::aggregate_statistics::AggregateStatistics;
use crate::combine_partial_final_agg::CombinePartialFinalAggregate;
use crate::eliminate_constant_join::EliminateConstantJoin;
use crate::ensure_coop::EnsureCooperative;
use crate::ensure_requirements::EnsureRequirements;
use crate::filter_pushdown::FilterPushdown;
use crate::hash_join_buffering::HashJoinBuffering;
use crate::join_selection::{
    JoinSelection, JoinSelectionRunnability, JoinSelectionStatistics,
};
use crate::limit_pushdown::LimitPushdown;
use crate::limit_pushdown_past_window::LimitPushPastWindows;
use crate::limited_distinct_aggregation::LimitedDistinctAggregation;
use crate::output_requirements::OutputRequirements;
use crate::projection_pushdown::ProjectionPushdown;
use crate::pushdown_sort::PushdownSort;
use crate::sanity_checker::SanityCheckPlan;
use crate::topk_aggregation::TopKAggregation;
use crate::topk_repartition::TopKRepartition;
use crate::update_aggr_exprs::OptimizeAggregateOrder;
use crate::window_topn::WindowTopN;

type Rule = Arc<dyn PhysicalOptimizerRule + Send + Sync>;

/// Whether directive `name` is set in `DF_POC`.
pub fn enabled(name: &str) -> bool {
    directives().iter().any(|d| d == name)
}

fn directives() -> Vec<String> {
    std::env::var("DF_POC")
        .unwrap_or_default()
        .split(',')
        .map(|d| d.trim().to_string())
        .filter(|d| !d.is_empty())
        .collect()
}

/// The analyzer rules (up to and including `EnsureRequirements`) and the
/// optimizer rules, after applying `DF_POC`.
pub(crate) fn pipeline() -> (Vec<Rule>, Vec<Rule>) {
    let mut rules: Vec<(String, Rule)> = vec![
        (
            "OR_add".into(),
            Arc::new(OutputRequirements::new_add_mode()),
        ),
        ("AS".into(), Arc::new(AggregateStatistics::new())),
        ("ECJ".into(), Arc::new(EliminateConstantJoin::new())),
        ("JS".into(), Arc::new(JoinSelection::new())),
        ("LDA".into(), Arc::new(LimitedDistinctAggregation::new())),
        ("FP_pre".into(), Arc::new(FilterPushdown::new())),
        ("WTN".into(), Arc::new(WindowTopN::new())),
        ("ER".into(), Arc::new(EnsureRequirements::new())),
        ("CPFA".into(), Arc::new(CombinePartialFinalAggregate::new())),
        ("OAO".into(), Arc::new(OptimizeAggregateOrder::new())),
        ("PP1".into(), Arc::new(ProjectionPushdown::new())),
        (
            "OR_remove".into(),
            Arc::new(OutputRequirements::new_remove_mode()),
        ),
        ("LA".into(), Arc::new(TopKAggregation::new())),
        ("LPPW".into(), Arc::new(LimitPushPastWindows::new())),
        ("HJB".into(), Arc::new(HashJoinBuffering::new())),
        ("LP".into(), Arc::new(LimitPushdown::new())),
        ("TKR".into(), Arc::new(TopKRepartition::new())),
        ("PP2".into(), Arc::new(ProjectionPushdown::new())),
        ("PS".into(), Arc::new(PushdownSort::new())),
        ("EC".into(), Arc::new(EnsureCooperative::new())),
        (
            "FP_post".into(),
            Arc::new(FilterPushdown::new_post_optimization()),
        ),
        ("SCP".into(), Arc::new(SanityCheckPlan::new())),
    ];
    let position = |rules: &[(String, Rule)], key: &str| {
        rules
            .iter()
            .position(|(k, _)| k == key)
            .unwrap_or_else(|| panic!("DF_POC: unknown rule {key}"))
    };
    for directive in directives() {
        let parts: Vec<&str> = directive.split(':').collect();
        match parts.as_slice() {
            ["move", key, side, other] => {
                let rule = rules.remove(position(&rules, key));
                let at = position(&rules, other);
                let at = if *side == "after" { at + 1 } else { at };
                rules.insert(at, rule);
            }
            ["drop", key] => {
                rules.remove(position(&rules, key));
            }
            ["js_split"] => {
                let at = position(&rules, "JS");
                rules.remove(at);
                rules.insert(
                    at,
                    ("JS_stats".into(), Arc::new(JoinSelectionStatistics::new())),
                );
                rules.insert(
                    at,
                    ("JS_run".into(), Arc::new(JoinSelectionRunnability::new())),
                );
            }
            ["js_planner"] => {
                rules.remove(position(&rules, "JS"));
            }
            _ => panic!("DF_POC: unknown directive {directive}"),
        }
    }
    let boundary = position(&rules, "ER") + 1;
    let optimizer = rules.split_off(boundary);
    (
        rules.into_iter().map(|(_, r)| r).collect(),
        optimizer.into_iter().map(|(_, r)| r).collect(),
    )
}

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

//! PoC only: runs the sqllogictest corpus under a rearranged physical optimizer
//! rule list, from the `DF_RULE_ORDER` environment variable.
//!
//! `DF_RULE_ORDER` is a comma-separated list of directives, applied in order:
//! - `default`: no change.
//! - `first:KEY` / `last:KEY`: move rule `KEY` to the start or end of the list.
//! - `move:KEY:before:OTHER` / `move:KEY:after:OTHER`: move `KEY` next to `OTHER`.
//! - `reverse`: reverse the list.
//! - `shuffle:SEED`: shuffle the list with a seeded generator.
//!
//! `KEY` is the rule's `name()`. A name that repeats in the default list gets
//! `#2` on its second use, such as `OutputRequirements#2` (remove mode) and
//! `ProjectionPushdown#2`.
//!
//! A test file can also pin its own order, see [`spec_for_file`].

use std::collections::HashMap;
use std::sync::{Arc, Once};

use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;

const ENV_VAR: &str = "DF_RULE_ORDER";

type Rule = Arc<dyn PhysicalOptimizerRule + Send + Sync>;

static PRINT_ORDER: Once = Once::new();

/// The partition count for a run, from `DF_TARGET_PARTITIONS`. With one partition,
/// joins keep their input order and enforcement removes more sorts.
pub(crate) fn target_partitions() -> usize {
    std::env::var("DF_TARGET_PARTITIONS")
        .ok()
        .and_then(|partitions| partitions.parse().ok())
        .unwrap_or(4)
}

/// The rule order a test file checks: the query results must not depend on it.
fn spec_for_file(file_name: &str) -> Option<&'static str> {
    match file_name {
        "rule_order_output_requirements.slt" => {
            Some("move:OutputRequirements#2:before:OutputRequirements")
        }
        "rule_order_join_selection.slt" => {
            Some("move:join_selection:after:EnsureRequirements")
        }
        "rule_order_filter_past_fetch.slt" => {
            Some("move:FilterPushdown:after:LimitPushdown")
        }
        "rule_order_window_limit.slt" => {
            Some("move:LimitPushPastWindows:before:EnsureRequirements")
        }
        "rule_order_aggregate_order.slt" => {
            Some("move:OptimizeAggregateOrder:before:join_selection")
        }
        "rule_order_sort_fetch.slt" => Some("move:EnsureRequirements:after:PushdownSort"),
        "rule_order_symmetric_join_limit.slt" => {
            Some("move:LimitPushdown:before:join_selection")
        }
        "rule_order_volatile_filter.slt" => {
            Some("move:ProjectionPushdown:before:FilterPushdown")
        }
        "rule_order_join_order.slt" => {
            Some("move:join_selection:after:OutputRequirements#2")
        }
        "rule_order_symmetric_join.slt" => {
            Some("move:ProjectionPushdown:before:join_selection")
        }
        "rule_order_filter_pushdown_post.slt" => {
            Some("move:FilterPushdown(Post):before:join_selection")
        }
        "rule_order_limit_pushdown.slt" => {
            Some("move:LimitPushdown:before:EnsureRequirements")
        }
        "rule_order_limit_statistics.slt" => {
            Some("move:LimitPushdown:before:aggregate_statistics")
        }
        "rule_order_pushdown_sort.slt" => Some(
            "move:OutputRequirements#2:before:EnsureRequirements,move:PushdownSort:before:EnsureRequirements",
        ),
        "rule_order_combine_aggregate.slt" => {
            Some("move:CombinePartialFinalAggregate:before:EnsureRequirements")
        }
        "rule_order_topk_aggregation.slt" => {
            Some("move:LimitAggregation:before:EnsureRequirements")
        }
        "rule_order_window_topn.slt" => {
            Some("move:WindowTopN:after:OutputRequirements#2")
        }
        "rule_order_projection_pushdown.slt" => {
            Some("move:ProjectionPushdown:before:EnsureRequirements")
        }
        _ => None,
    }
}

/// The rule list for a test file, if it isn't the default one.
pub(crate) fn rules_for_file(file_name: &str) -> Option<Vec<Rule>> {
    if let Some(spec) = spec_for_file(file_name) {
        return Some(rules_with(spec));
    }
    std::env::var_os(ENV_VAR).map(|_| rules())
}

/// The default rule list, rearranged by `DF_RULE_ORDER`.
pub(crate) fn rules() -> Vec<Rule> {
    let spec = std::env::var(ENV_VAR).unwrap_or_default();
    let rules = keyed_rules(&spec);
    PRINT_ORDER.call_once(|| {
        let keys: Vec<&str> = rules.iter().map(|(key, _)| key.as_str()).collect();
        eprintln!("RULE_ORDER [{spec}]: {}", keys.join(" > "));
    });
    rules.into_iter().map(|(_, rule)| rule).collect()
}

/// The default rule list, rearranged by `spec`.
fn rules_with(spec: &str) -> Vec<Rule> {
    keyed_rules(spec)
        .into_iter()
        .map(|(_, rule)| rule)
        .collect()
}

fn keyed_rules(spec: &str) -> Vec<(String, Rule)> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut rules: Vec<(String, Rule)> = PhysicalOptimizer::new()
        .rules
        .into_iter()
        .map(|rule| {
            let count = seen.entry(rule.name().to_string()).or_default();
            *count += 1;
            let key = match *count {
                1 => rule.name().to_string(),
                n => format!("{}#{n}", rule.name()),
            };
            (key, rule)
        })
        .collect();
    for directive in spec.split(',').map(str::trim).filter(|d| !d.is_empty()) {
        apply(&mut rules, directive);
    }
    rules
}

fn apply(rules: &mut Vec<(String, Rule)>, directive: &str) {
    let parts: Vec<&str> = directive.split(':').collect();
    match parts.as_slice() {
        ["default"] => {}
        ["reverse"] => rules.reverse(),
        ["first", key] => {
            let rule = rules.remove(position(rules, key));
            rules.insert(0, rule);
        }
        ["last", key] => {
            let rule = rules.remove(position(rules, key));
            rules.push(rule);
        }
        ["move", key, side, other] => {
            let rule = rules.remove(position(rules, key));
            let at = position(rules, other);
            let at = if *side == "after" { at + 1 } else { at };
            rules.insert(at, rule);
        }
        ["shuffle", seed] => shuffle(rules, seed.parse().expect("numeric seed")),
        _ => panic!("{ENV_VAR}: unknown directive {directive}"),
    }
}

fn position(rules: &[(String, Rule)], key: &str) -> usize {
    rules
        .iter()
        .position(|(k, _)| k == key)
        .unwrap_or_else(|| panic!("{ENV_VAR}: unknown rule {key}"))
}

/// Fisher-Yates with SplitMix64, so a seed gives the same order on every machine.
fn shuffle(rules: &mut [(String, Rule)], seed: u64) {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..rules.len()).rev() {
        rules.swap(i, (next() % (i as u64 + 1)) as usize);
    }
}

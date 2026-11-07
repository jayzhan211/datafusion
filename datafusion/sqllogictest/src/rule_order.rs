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

use std::collections::HashMap;
use std::sync::{Arc, Once};

use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;

const ENV_VAR: &str = "DF_RULE_ORDER";

type Rule = Arc<dyn PhysicalOptimizerRule + Send + Sync>;

static PRINT_ORDER: Once = Once::new();

/// Whether a rule order was requested.
pub(crate) fn is_set() -> bool {
    std::env::var_os(ENV_VAR).is_some()
}

/// The default rule list, rearranged by `DF_RULE_ORDER`.
pub(crate) fn rules() -> Vec<Rule> {
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

    let spec = std::env::var(ENV_VAR).unwrap_or_default();
    for directive in spec.split(',').map(str::trim).filter(|d| !d.is_empty()) {
        apply(&mut rules, directive);
    }

    PRINT_ORDER.call_once(|| {
        let keys: Vec<&str> = rules.iter().map(|(key, _)| key.as_str()).collect();
        eprintln!("RULE_ORDER [{spec}]: {}", keys.join(" > "));
    });
    rules.into_iter().map(|(_, rule)| rule).collect()
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

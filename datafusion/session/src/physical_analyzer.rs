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

//! Physical analyzer rules.

use crate::physical_optimizer::PhysicalOptimizerRule;

/// A `PhysicalAnalyzerRule` is a correctness transformation: it makes an
/// [`ExecutionPlan`] executable, as a logical [`AnalyzerRule`] makes a logical
/// plan valid. The analyzer list ends with `EnsureRequirements`, which
/// repartitions and sorts the inputs to match each operator's required
/// distribution and ordering, so every rule in the optimizer list receives a
/// valid plan and must return one.
///
/// It has no methods of its own: a rule implements [`PhysicalOptimizerRule`]
/// and declares itself a correctness rule with
/// `impl PhysicalAnalyzerRule for MyRule {}`. A performance rule that must run
/// before `EnsureRequirements`, such as `WindowTopN`, is not an analyzer rule;
/// it runs in the analyzer list through `RunInAnalyzer`.
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
/// [`AnalyzerRule`]: https://docs.rs/datafusion/latest/datafusion/optimizer/analyzer/trait.AnalyzerRule.html
/// [`ExecutionPlan`]: datafusion_physical_plan::ExecutionPlan
pub trait PhysicalAnalyzerRule: PhysicalOptimizerRule {}

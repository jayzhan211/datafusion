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

//! [`satisfy_requirements`]: an insert-only counterpart of [`EnsureRequirements`].
//!
//! [`EnsureRequirements`]: super::EnsureRequirements

use std::sync::Arc;

use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{Result, internal_err};
use datafusion_physical_expr::{Distribution, LexRequirement, Partitioning};
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::execution_plan::replace_children_if_necessary;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::tree_node::PlanContext;
use datafusion_physical_plan::union::{InterleaveExec, UnionExec, can_interleave};
use datafusion_physical_plan::windows::{
    BoundedWindowAggExec, WindowAggExec, get_best_fitting_window,
};
use datafusion_physical_plan::{
    ChildSatisfactionOptions, ExecutionPlan, ExecutionPlanProperties,
    InputDistributionRequirements,
};

use super::enforce_sorting::replace_with_partial_sort;
use crate::utils::{add_sort_above_with_distribution, is_window};

/// Makes `plan` satisfy the distribution and ordering requirements that its
/// operators declare, by inserting `RepartitionExec`, `CoalescePartitionsExec`,
/// `SortPreservingMergeExec` and `SortExec` only where a requirement is unmet.
///
/// Unlike [`EnsureRequirements`], this makes no performance choice: it does
/// not add round-robin parallelism, reorder join keys, or move existing
/// operators. A plan that already satisfies every requirement is returned
/// unchanged, so a rule can call this after its rewrite as often as it needs
/// to.
///
/// Besides inserting operators, it makes three replacements that the plan
/// needs to run at all: an `InterleaveExec` whose children are no longer
/// identically partitioned becomes a `UnionExec`; a window whose input order
/// is unmet becomes the variant that fits that order, when one exists, instead
/// of getting a sort; and a sort of an unbounded input that is already ordered
/// on a prefix of the sort keys becomes a `PartialSortExec`.
///
/// A requirement counts as satisfied exactly when [`SanityCheckPlan`] accepts
/// it, so a plan that passes [`SanityCheckPlan`] returns [`Transformed::no`].
///
/// [`EnsureRequirements`]: super::EnsureRequirements
/// [`SanityCheckPlan`]: crate::sanity_checker::SanityCheckPlan
pub fn satisfy_requirements(
    plan: Arc<dyn ExecutionPlan>,
    config: &ConfigOptions,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let target_partitions = config.execution.target_partitions;
    plan.transform_up(|node| satisfy_children(node, target_partitions))
}

/// Whether child `idx` of `plan` satisfies its distribution requirement.
///
/// A key requirement is met by partitioning on a subset of the keys, and an
/// aggregate's is also met when each group's rows are already in one input
/// partition (#25344). Co-partitioning between children is checked separately
/// (see [`InputDistributionRequirements::unsatisfied_co_partitioned_children`]).
pub(crate) fn distribution_satisfied(
    plan: &Arc<dyn ExecutionPlan>,
    requirements: &InputDistributionRequirements,
    idx: usize,
    child: &Arc<dyn ExecutionPlan>,
) -> Result<bool> {
    let satisfied = requirements
        .child_satisfaction(
            idx,
            child.as_ref(),
            ChildSatisfactionOptions::new().with_allow_subset(true),
        )?
        .is_satisfied();
    Ok(satisfied
        || plan
            .downcast_ref::<AggregateExec>()
            .is_some_and(|aggregate| {
                aggregate.input_keeps_groups_local(child.as_ref(), true)
            }))
}

/// Fixes one node whose children are already fixed: the three replacements
/// first, then distribution, then ordering.
fn satisfy_children(
    plan: Arc<dyn ExecutionPlan>,
    target_partitions: usize,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    if let Some(interleave) = plan.downcast_ref::<InterleaveExec>() {
        if can_interleave(interleave.inputs().iter()) {
            return Ok(Transformed::no(plan));
        }
        let union = UnionExec::try_new(interleave.inputs().clone())?;
        return Ok(Transformed::yes(union));
    }
    if plan.children().is_empty() {
        return Ok(Transformed::no(plan));
    }
    sort_unbounded_partially(plan)?
        .transform_data(fit_window_to_input)?
        .transform_data(|plan| satisfy_distribution(plan, target_partitions))?
        // Read the ordering requirements from the rebuilt node: some
        // operators, such as `AggregateExec`, derive them from their input.
        .transform_data(|plan| satisfy_ordering(plan, target_partitions))
}

/// A full sort of an unbounded input never emits. When the input is already
/// ordered on a prefix of the sort keys, sorting within each group of that
/// prefix (`PartialSortExec`) can run.
fn sort_unbounded_partially(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let Some(sort) = plan.downcast_ref::<SortExec>() else {
        return Ok(Transformed::no(plan));
    };
    let input = sort.input();
    if !input.boundedness().is_unbounded()
        || input
            .equivalence_properties()
            .ordering_satisfy(sort.expr().clone())?
    {
        return Ok(Transformed::no(plan));
    }
    let partial = replace_with_partial_sort(Arc::clone(&plan))?;
    Ok(if Arc::ptr_eq(&partial, &plan) {
        Transformed::no(plan)
    } else {
        Transformed::yes(partial)
    })
}

/// A window whose input order fits a reversed or partially sorted variant of
/// it runs without a sort. On an unbounded input, that variant is the only
/// one that can run at all.
fn fit_window_to_input(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    if !is_window(&plan) {
        return Ok(Transformed::no(plan));
    }
    let input = &plan.children()[0];
    let ordering_met = match plan.required_input_ordering().swap_remove(0) {
        Some(ordering) => input
            .equivalence_properties()
            .ordering_satisfy_requirement(ordering.into_single())?,
        None => true,
    };
    if ordering_met {
        return Ok(Transformed::no(plan));
    }
    let fitted = if let Some(exec) = plan.downcast_ref::<WindowAggExec>() {
        get_best_fitting_window(
            exec.window_expr(),
            exec.input(),
            &exec.partition_keys(),
            None,
        )?
    } else if let Some(exec) = plan.downcast_ref::<BoundedWindowAggExec>() {
        get_best_fitting_window(
            exec.window_expr(),
            exec.input(),
            &exec.partition_keys(),
            exec.state_observer().cloned(),
        )?
    } else {
        None
    };
    Ok(match fitted {
        Some(window) => Transformed::yes(window),
        None => Transformed::no(plan),
    })
}

/// Adds the exchange each child needs, then repartitions co-partitioned
/// children into the same number of partitions.
#[expect(
    deprecated,
    reason = "HashPartitioned is accepted during the KeyPartitioned migration"
)]
fn satisfy_distribution(
    plan: Arc<dyn ExecutionPlan>,
    target_partitions: usize,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let requirements = plan.input_distribution_requirements();
    let orderings = required_orderings(&plan);
    let mut children = plan.children().into_iter().cloned().collect::<Vec<_>>();
    let mut changed = false;

    for (idx, (child, ordering)) in children.iter_mut().zip(&orderings).enumerate() {
        if distribution_satisfied(&plan, &requirements, idx, child)? {
            continue;
        }
        let Some(distribution) = requirements.child_distribution(idx) else {
            return internal_err!("{} has no requirement for child {idx}", plan.name());
        };
        // Skip a child that needs one partition and an ordering it lacks:
        // `satisfy_ordering` sorts each partition and merges them, which meets
        // both.
        if matches!(distribution, Distribution::SinglePartition)
            && let Some(ordering) = ordering
            && !child
                .equivalence_properties()
                .ordering_satisfy_requirement(ordering.clone())?
        {
            continue;
        }
        *child = add_exchange(
            Arc::clone(child),
            distribution.clone(),
            ordering.as_ref(),
            target_partitions,
        )?;
        changed = true;
    }

    let unmet = unmet_co_partitioned_children(&plan, &requirements, &children)?;
    if !unmet.is_empty() {
        // Partition every co-partitioned child on its own keys into the same
        // number of partitions, which is always compatible.
        for idx in unmet {
            let (Some(Distribution::KeyPartitioned(exprs))
            | Some(Distribution::HashPartitioned(exprs))) =
                requirements.child_distribution(idx)
            else {
                continue;
            };
            let partitioning = Distribution::KeyPartitioned(exprs.clone())
                .create_partitioning(target_partitions);
            if *children[idx].output_partitioning() == partitioning {
                continue;
            }
            let preserve_order = orderings[idx].is_some();
            children[idx] =
                repartition(Arc::clone(&children[idx]), partitioning, preserve_order)?;
            changed = true;
        }
        if !unmet_co_partitioned_children(&plan, &requirements, &children)?.is_empty()
        {
            return internal_err!(
                "{} has co-partitioning requirements that could not be satisfied",
                plan.name()
            );
        }
    }

    if !changed {
        return Ok(Transformed::no(plan));
    }
    replace_children_if_necessary(plan, children).map(Transformed::yes)
}

/// Adds the sort each child needs, with a merge on top when the child must
/// also be one partition.
fn satisfy_ordering(
    plan: Arc<dyn ExecutionPlan>,
    target_partitions: usize,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let requirements = plan.input_distribution_requirements();
    let orderings = required_orderings(&plan);
    let mut children = plan.children().into_iter().cloned().collect::<Vec<_>>();
    let mut changed = false;

    for (idx, (child, ordering)) in children.iter_mut().zip(orderings).enumerate() {
        let distribution = requirements
            .child_distribution(idx)
            .cloned()
            .unwrap_or(Distribution::UnspecifiedDistribution);
        if let Some(ordering) = ordering
            && !child
                .equivalence_properties()
                .ordering_satisfy_requirement(ordering.clone())?
        {
            let node = PlanContext::<()>::new_default(Arc::clone(child));
            let sorted =
                add_sort_above_with_distribution(node, ordering, None, &distribution)
                    .plan;
            // The new sort is either `sorted` or the input of its merge.
            *child = match sorted.downcast_ref::<SortPreservingMergeExec>() {
                Some(merge) => {
                    let sort = sort_unbounded_partially(Arc::clone(merge.input()))?;
                    if sort.transformed {
                        replace_children_if_necessary(
                            Arc::clone(&sorted),
                            vec![sort.data],
                        )?
                    } else {
                        sorted
                    }
                }
                None => sort_unbounded_partially(sorted)?.data,
            };
            changed = true;
        }
        // `satisfy_distribution` skipped a child that needs one partition and
        // an ordering. When the rebuilt node no longer requires that ordering,
        // no merge was added above, so add the exchange here.
        if !distribution_satisfied(&plan, &requirements, idx, child)? {
            *child =
                add_exchange(Arc::clone(child), distribution, None, target_partitions)?;
            changed = true;
        }
    }

    if !changed {
        return Ok(Transformed::no(plan));
    }
    replace_children_if_necessary(plan, children).map(Transformed::yes)
}

fn required_orderings(plan: &Arc<dyn ExecutionPlan>) -> Vec<Option<LexRequirement>> {
    plan.required_input_ordering()
        .into_iter()
        .map(|ordering| ordering.map(|ordering| ordering.into_single()))
        .collect()
}

fn unmet_co_partitioned_children(
    plan: &Arc<dyn ExecutionPlan>,
    requirements: &InputDistributionRequirements,
    children: &[Arc<dyn ExecutionPlan>],
) -> Result<Vec<usize>> {
    let children = children.iter().map(|c| c.as_ref()).collect::<Vec<_>>();
    requirements.unsatisfied_co_partitioned_children(plan.name(), &children)
}

/// Adds the exchange that gives `child` the `distribution` it lacks, keeping
/// its ordering when the parent requires an ordering that `child` already has.
fn add_exchange(
    child: Arc<dyn ExecutionPlan>,
    distribution: Distribution,
    ordering: Option<&LexRequirement>,
    target_partitions: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let preserve_order = match ordering {
        Some(ordering) => child
            .equivalence_properties()
            .ordering_satisfy_requirement(ordering.clone())?,
        None => false,
    };
    match distribution {
        Distribution::SinglePartition => match child.output_ordering() {
            Some(child_ordering) if preserve_order => Ok(Arc::new(
                SortPreservingMergeExec::new(child_ordering.clone(), child),
            )),
            _ => Ok(Arc::new(CoalescePartitionsExec::new(child))),
        },
        distribution => {
            let partitioning = distribution.create_partitioning(target_partitions);
            repartition(child, partitioning, preserve_order)
        }
    }
}

fn repartition(
    child: Arc<dyn ExecutionPlan>,
    partitioning: Partitioning,
    preserve_order: bool,
) -> Result<Arc<dyn ExecutionPlan>> {
    let repartition = RepartitionExec::try_new(child, partitioning)?;
    Ok(Arc::new(if preserve_order {
        repartition.with_preserve_order()
    } else {
        repartition
    }))
}

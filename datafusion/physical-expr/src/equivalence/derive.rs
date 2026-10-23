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

//! Output facts of projection, union and join, computed from their inputs.

use std::sync::Arc;

use super::matching::Walk;
use super::properties::{EquivalenceProperties, Facts, Scope};
use crate::equivalence::ProjectionMapping;
use crate::expressions::{Column, Literal};
use crate::physical_expr::add_offset_to_expr;
use crate::{PhysicalExpr, PhysicalExprRef, PhysicalSortExpr};

use arrow::compute::SortOptions;
use arrow::datatypes::SchemaRef;
use datafusion_common::{JoinSide, JoinType, Result, internal_err};
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_expr::sort_properties::{ExprProperties, SortProperties};
use datafusion_physical_expr_common::sort_expr::{Ordering, Step};

impl EquivalenceProperties {
    /// The facts in terms of a projection's output `schema`.
    ///
    /// - Equalities: each class re-expressed on the output, plus all targets
    ///   of one source, plus targets of equal sources.
    /// - Constants: each target of a constant source, with the source's scope.
    /// - Orderings: each ordering in normal form, step by step (see
    ///   `project_ordering`).
    /// - Keys: projected onto the output columns.
    pub fn project(&self, mapping: &ProjectionMapping, schema: SchemaRef) -> Self {
        let mut facts = Facts {
            eq: self.equalities().project(mapping),
            ..Default::default()
        };
        let indices: Vec<_> = (mapping.iter().flat_map(|(_, t)| t.iter()))
            .filter_map(|(t, _)| t.downcast_ref::<Column>().map(|c| c.index()))
            .collect();
        facts.global.keys = self.constraints().project(&indices).unwrap_or_default();
        let mut out = Self::from_facts(schema, facts);
        for (source, targets) in mapping.iter() {
            let scope = match self.global_constant(source) {
                Some(value) => Scope::Global(value),
                None if self.is_constant(source) => Scope::Local,
                None => continue,
            };
            for (target, _) in targets.iter() {
                match &scope {
                    Scope::Global(value) => {
                        out.add_constant(Arc::clone(target), value.clone())
                    }
                    Scope::Local => out.add_local_constant(Arc::clone(target)),
                }
            }
        }
        // Sources in normal form, computed once for all orderings.
        let eq = self.equalities();
        let sources: Vec<_> = (mapping.iter())
            .map(|(source, targets)| (eq.normalize(Arc::clone(source)), targets))
            .collect();
        for ordering in &self.normal().orderings {
            for projected in self.project_ordering(ordering, mapping, &sources) {
                out.add_ordering(projected);
            }
        }
        // A source ordered only by several orderings together (`a > b` with
        // `[a]` and `[b DESC]`) gets a one-step ordering of its own.
        let walk = Walk::new(self);
        for (source, targets) in &sources {
            let SortProperties::Ordered(options) = walk.order_of(source).sort_properties
            else {
                continue;
            };
            for (target, _) in targets.iter() {
                let leads =
                    (out.orderings().iter()).any(|o| o[0].exprs().any(|e| e.eq(target)));
                if !leads {
                    out.add_ordering([Step::sort(Arc::clone(target), options)]);
                }
            }
        }
        out
    }

    /// Projects one ordering (in normal form, without constant steps), one
    /// step at a time, with no cartesian product:
    ///
    /// - A member that the mapping rewrites, and a source that is a strict
    ///   image of a member on its own (`CAST(a)` for `a`), become members of
    ///   the output step, and the ordering goes on.
    /// - Any other source that is ordered here, given the earlier steps
    ///   (`a / 2` for `a`), ends its own ordering: the output prefix plus
    ///   that source's targets.
    /// - The ordering stops at the first step with no output member.
    /// - A `Group` step goes on only if every expression in it has an
    ///   output (the mapping rewrites it as is); otherwise the ordering stops.
    ///
    /// Orderings are projected one at a time, so an output ordered only by
    /// two orderings together is not found here; [`Self::project`] adds a
    /// one-step ordering for such an output, but nothing longer (L5).
    fn project_ordering(
        &self,
        ordering: &Ordering,
        mapping: &ProjectionMapping,
        sources: &[(Arc<dyn PhysicalExpr>, &crate::projection::ProjectionTargets)],
    ) -> Vec<Ordering> {
        let eq = self.equalities();
        let mut walk = Walk::over(self, ordering);
        let mut prefix = Ordering::default();
        let mut result = vec![];
        for step in ordering.iter() {
            let next = match step {
                Step::Group(_) => step.try_map(|e| eq.project_expr(mapping, e)),
                Step::Sort(members) => {
                    let mut strict = vec![];
                    let mut strict_sources = vec![];
                    for m in members {
                        if let Some(expr) = eq.project_expr(mapping, &m.expr) {
                            push_new(&mut strict, PhysicalSortExpr::new(expr, m.options));
                        }
                        for (i, (source, targets)) in sources.iter().enumerate() {
                            if !is_strict_image(source, m, self.schema()) {
                                continue;
                            }
                            strict_sources.push(i);
                            for (target, _) in targets.iter() {
                                let sort =
                                    PhysicalSortExpr::new(Arc::clone(target), m.options);
                                push_new(&mut strict, sort);
                            }
                        }
                    }
                    for (i, (source, targets)) in sources.iter().enumerate() {
                        if strict_sources.contains(&i)
                            || members.iter().any(|m| m.expr.eq(source))
                        {
                            continue;
                        }
                        let SortProperties::Ordered(options) =
                            walk.order_of(source).sort_properties
                        else {
                            continue;
                        };
                        let sorts = targets
                            .iter()
                            .map(|(t, _)| PhysicalSortExpr::new(Arc::clone(t), options));
                        let mut truncated = prefix.clone();
                        truncated.push(Step::Sort(sorts.collect()));
                        result.push(truncated);
                    }
                    (!strict.is_empty()).then_some(Step::Sort(strict))
                }
            };
            let Some(next) = next else { break };
            prefix.push(next);
            step.exprs().for_each(|e| walk.pin(Arc::clone(e)));
        }
        if !prefix.is_empty() {
            result.push(prefix);
        }
        result
    }

    /// The facts of a `UnionExec` output in `schema`. A union passes each
    /// child partition through, so a fact holds iff it holds in every child.
    /// Equalities and keys are dropped (a TODO kept from before).
    pub fn union(children: Vec<Self>, schema: SchemaRef) -> Result<Self> {
        let mut children = children.into_iter();
        let Some(first) = children.next() else {
            return internal_err!(
                "Cannot calculate EquivalenceProperties for a union with no inputs"
            );
        };
        let mut acc = if first.schema().eq(&schema) {
            first
        } else {
            first.with_new_schema(schema)?
        };
        for child in children {
            let child = if child.schema().eq(acc.schema()) {
                child
            } else {
                child.with_new_schema(Arc::clone(acc.schema()))?
            };
            acc = Self::union_pair(&acc, &child);
        }
        Ok(acc)
    }

    fn union_pair(lhs: &Self, rhs: &Self) -> Self {
        let mut out = Self::new(Arc::clone(lhs.schema()));
        // A constant of both sides: global with the same value (or both
        // unknown) on both, else per partition. Constants are read on every
        // member of a constant's class, as the classes themselves are lost.
        let rhs_constants = rhs.constant_exprs();
        for expr in lhs.constant_exprs() {
            if !rhs_constants.contains(&expr) {
                continue;
            }
            let (Some(l), Some(r)) =
                (lhs.stored_constant(&expr), rhs.stored_constant(&expr))
            else {
                continue;
            };
            match (l, r) {
                // behavior kept: two unknown values count as equal (C5, U1).
                (Scope::Global(a), Scope::Global(b)) if a == b => {
                    out.add_constant(expr, a.clone())
                }
                _ => out.add_local_constant(expr),
            }
        }
        // Each normal ordering of one side, cut down to the step members the
        // other side has too (`members_in_common`), that the other side
        // satisfies, or that fills its gaps with constants of the other side
        // (`augment`), or else its longest prefix that does.
        for (side, other) in [(lhs, rhs), (rhs, lhs)] {
            let side_constants = side.constant_exprs();
            let other_constants = other.constant_exprs();
            for ordering in &side.normal().orderings {
                let (mut candidate, truncated) = other.members_in_common(ordering);
                out.add_orderings(truncated);
                while !candidate.is_empty() {
                    if other.ordering_satisfy(candidate.clone()) {
                        out.add_ordering(candidate);
                        break;
                    }
                    let augmented: Vec<_> = if side_constants.is_empty() {
                        vec![]
                    } else {
                        (other.orderings().iter())
                            .filter_map(|o| {
                                augment(&candidate, &side_constants, o, &other_constants)
                            })
                            .collect()
                    };
                    if !augmented.is_empty() {
                        for o in augmented {
                            out.add_ordering(o);
                        }
                        break;
                    }
                    candidate.pop();
                }
            }
        }
        out
    }

    /// `ordering`, a normal ordering of another union input, with each `Sort`
    /// step cut down to the members that this input orders there too. A
    /// step's members are strict images of each other on every row (see
    /// [`Step::Sort`]), so a member stays only if it and the step's first
    /// kept member fix each other on every row here as well: equal, members
    /// of one step here, or an all-partition constant (a per-partition one
    /// fixes nothing across partitions). One that this input orders but
    /// does not tie to the first ends a truncated ordering of its own
    /// (returned second). Cutting stops at the first step this input does
    /// not order; the caller's prefix popping handles what follows. An
    /// ordering of one-member steps comes back unchanged.
    fn members_in_common(&self, ordering: &Ordering) -> (Ordering, Vec<Ordering>) {
        let eq = self.equalities();
        let fixes =
            |a: &Arc<dyn PhysicalExpr>, b: &Arc<dyn PhysicalExpr>| {
                let (a, b) = (eq.normalize(Arc::clone(a)), eq.normalize(Arc::clone(b)));
                let together = (self.normal().orderings.iter().flat_map(|o| o.iter()))
                    .any(|step| match step {
                        Step::Sort(m) => {
                            m.iter().any(|m| m.expr.eq(&a))
                                && m.iter().any(|m| m.expr.eq(&b))
                        }
                        Step::Group(_) => false,
                    });
                a.eq(&b)
                    || together
                    || matches!(self.normal().constants.get(&b), Some(Scope::Global(_)))
            };
        let mut walk = Walk::new(self);
        let mut cut = Ordering::default();
        let mut truncated = vec![];
        let mut steps = ordering.iter();
        for step in steps.by_ref() {
            let kept = match step {
                Step::Sort(members) => {
                    let held: Vec<_> = members.iter().filter(|m| walk.holds(m)).collect();
                    held.split_first().map(|(first, others)| {
                        let mut kept = vec![(*first).clone()];
                        for m in others {
                            if fixes(&first.expr, &m.expr) && fixes(&m.expr, &first.expr)
                            {
                                kept.push((*m).clone());
                            } else {
                                let mut own = cut.clone();
                                own.push(Step::Sort(vec![(*m).clone()]));
                                truncated.push(own);
                            }
                        }
                        Step::Sort(kept)
                    })
                }
                Step::Group(_) => {
                    let normal = step.try_map(|e| Some(eq.normalize(Arc::clone(e))));
                    normal
                        .is_some_and(|n| walk.match_step(&n))
                        .then(|| step.clone())
                }
            };
            let Some(kept) = kept else {
                cut.push(step.clone());
                break;
            };
            kept.exprs().for_each(|e| walk.pin_satisfied(Arc::clone(e)));
            cut.push(kept);
        }
        steps.cloned().for_each(|s| cut.push(s));
        (cut, truncated)
    }

    /// Every expression known constant: each stored constant and the other
    /// members of its class.
    fn constant_exprs(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        let eq = self.equalities();
        let stored = self.global_constants().map(|(e, _)| e);
        let stored = stored.chain(self.local_constants());
        let mut result: Vec<Arc<dyn PhysicalExpr>> = vec![];
        for expr in stored {
            let class = eq.class_of(expr).into_iter().flatten();
            for e in std::iter::once(expr).chain(class) {
                if !result.contains(e) {
                    result.push(Arc::clone(e));
                }
            }
        }
        result
    }

    /// The facts of a join output in `schema`.
    ///
    /// - Equalities: both sides', plus the `on` keys for an inner join.
    /// - Constants: a side's constants, except on the right side when the
    ///   join pads it with NULLs (Left, Full). The left side keeps its
    ///   constants even when padded (behavior kept, W3 in found.md).
    /// - Orderings: those of `ordered_side`, the side whose order the
    ///   operator keeps. When that side is also `probe_side` and the join is
    ///   Inner or keeps the probe side's unmatched rows, each of its
    ///   orderings is followed by each ordering of the other side.
    /// - Keys are dropped.
    pub fn join(
        left: &Self,
        right: &Self,
        join_type: &JoinType,
        schema: SchemaRef,
        on: &[(PhysicalExprRef, PhysicalExprRef)],
        ordered_side: Option<JoinSide>,
        probe_side: Option<JoinSide>,
    ) -> Result<Self> {
        use JoinType::*;
        let width = left.schema().fields.len();
        // Where a side's columns land in the output; `None` if they don't.
        // Assumes `schema` is the join's own output layout (left columns,
        // then right ones; one side only for Semi/Anti/Mark), before any
        // projection the operator applies on top.
        let offset = |side| match (side, join_type) {
            (JoinSide::Left, RightSemi | RightAnti | RightMark) => None,
            (JoinSide::Right, LeftSemi | LeftAnti | LeftMark) => None,
            (JoinSide::Right, RightSemi | RightAnti | RightMark) => Some(0),
            (JoinSide::Right, _) => Some(width),
            _ => Some(0),
        };
        let mut out = Self::new(schema);
        for (side, facts) in [(JoinSide::Left, left), (JoinSide::Right, right)] {
            let Some(offset) = offset(side) else { continue };
            // Rows padded with NULLs break the padded side's constants.
            let padded_right =
                side == JoinSide::Right && matches!(join_type, Left | Full);
            // behavior kept: a padded left side keeps its constants (W3, found.md).
            out.add_shifted(facts, offset, !padded_right)?;
        }
        if *join_type == Inner {
            for (l, r) in on {
                let r = add_offset_to_expr(Arc::clone(r), width as _)?;
                out.add_equal(Arc::clone(l), r);
            }
        }
        let Some(side) = ordered_side else {
            return Ok(out);
        };
        let (kept, other) = match side {
            JoinSide::Left => (left, right),
            _ => (right, left),
        };
        let Some(kept_offset) = offset(side) else {
            return Ok(out);
        };
        let suffix = probe_side == Some(side)
            && matches!(
                (side, join_type),
                (JoinSide::Left, Inner | Left) | (JoinSide::Right, Inner | Right)
            );
        let kept_orderings: Vec<_> = (kept.orderings().iter())
            .filter_map(|o| shift(o, kept_offset))
            .collect();
        let other_orderings: Vec<_> = match offset(side.negate()) {
            Some(other_offset) if suffix => (other.orderings().iter())
                .filter_map(|o| shift(o, other_offset))
                .collect(),
            _ => vec![],
        };
        if other_orderings.is_empty() {
            out.add_orderings(kept_orderings);
        } else {
            for suffix in &other_orderings {
                for prefix in &kept_orderings {
                    let steps = prefix.iter().chain(suffix.iter()).cloned();
                    out.add_ordering(Ordering::new(steps));
                }
            }
        }
        Ok(out)
    }

    /// Adds `facts`' equalities and (if `with_constants`) constants, with
    /// column indices moved by `offset`.
    fn add_shifted(
        &mut self,
        facts: &Self,
        offset: usize,
        with_constants: bool,
    ) -> Result<()> {
        let shift =
            |e: &Arc<dyn PhysicalExpr>| add_offset_to_expr(Arc::clone(e), offset as _);
        let eq = facts.equalities().shifted(offset)?;
        self.facts_mut().eq.extend(eq);
        if with_constants {
            for (expr, value) in facts.global_constants() {
                self.add_constant(shift(expr)?, value.clone());
            }
            for expr in facts.local_constants() {
                self.add_local_constant(shift(expr)?);
            }
        }
        Ok(())
    }
}

impl EquivalenceProperties {
    /// Whether `transformed_expr` is ordered in the same direction as
    /// `source_expr` when `source_expr` is the only ordered input (other
    /// columns unordered, literals constant).
    pub(crate) fn is_same_direction_ordered_transform(
        &self,
        transformed_expr: &Arc<dyn PhysicalExpr>,
        source_expr: &Arc<dyn PhysicalExpr>,
    ) -> bool {
        if transformed_expr.eq(source_expr) {
            return false;
        }
        let options = SortOptions::default();
        let member = PhysicalSortExpr::new(Arc::clone(source_expr), options);
        matches!(
            image_properties(transformed_expr, &member, self.schema())
                .map(|properties| properties.sort_properties),
            Ok(SortProperties::Ordered(actual)) if actual == options
        )
    }
}

/// `ordering` with column indices moved by `offset`.
fn shift(ordering: &Ordering, offset: usize) -> Option<Ordering> {
    let shift =
        |e: &Arc<dyn PhysicalExpr>| add_offset_to_expr(Arc::clone(e), offset as _).ok();
    let steps = ordering.iter().map(|s| s.try_map(shift));
    Some(Ordering::new(steps.collect::<Option<Vec<_>>>()?))
}

/// Walks `ordering` and `existing` side by side; a step missing on one side
/// is taken if it is a constant of the side that lacks it.
fn augment(
    ordering: &Ordering,
    constants: &[Arc<dyn PhysicalExpr>],
    existing: &Ordering,
    existing_constants: &[Arc<dyn PhysicalExpr>],
) -> Option<Ordering> {
    let constant_in = |list: &[Arc<dyn PhysicalExpr>], s: &Step| {
        s.exprs().any(|e| list.iter().any(|c| c.eq(e)))
    };
    let (mut a, mut b) = (ordering.iter().peekable(), existing.iter().peekable());
    let mut out = vec![];
    while a.peek().is_some() || b.peek().is_some() {
        match (a.peek(), b.peek()) {
            (Some(x), Some(y)) if x == y => {
                out.push((*x).clone());
                a.next();
                b.next();
            }
            (Some(x), _) if constant_in(existing_constants, x) => {
                out.extend(a.next().cloned())
            }
            (_, Some(y)) if constant_in(constants, y) => out.extend(b.next().cloned()),
            _ => break,
        }
    }
    let out = Ordering::new(out);
    (!out.is_empty()).then_some(out)
}

/// Whether `source` is a strict image of `member` on its own: with only
/// `member` ordered (every other column unordered, nothing pinned), it is
/// ordered with `member`'s options and strictly order preserving, so equal
/// values of `source` mean equal values of `member`. This is the
/// single-dependency rule; judging it with earlier steps pinned would be
/// unsound for strict functions of several inputs.
fn is_strict_image(
    source: &Arc<dyn PhysicalExpr>,
    member: &PhysicalSortExpr,
    schema: &SchemaRef,
) -> bool {
    !source.eq(&member.expr)
        && refers_to(source, &member.expr)
        && image_properties(source, member, schema).is_ok_and(|p| {
            p.strictly_order_preserving
                && p.sort_properties == SortProperties::Ordered(member.options)
        })
}

fn refers_to(expr: &Arc<dyn PhysicalExpr>, target: &Arc<dyn PhysicalExpr>) -> bool {
    expr.eq(target) || expr.children().iter().any(|c| refers_to(c, target))
}

/// How `expr` is ordered when `member` is the only ordered input.
fn image_properties(
    expr: &Arc<dyn PhysicalExpr>,
    member: &PhysicalSortExpr,
    schema: &SchemaRef,
) -> Result<ExprProperties> {
    let range = || Interval::make_unbounded(&expr.data_type(schema)?);
    if expr.eq(&member.expr) {
        // Treated as an atomic ordered input: the identity of itself.
        Ok(ExprProperties {
            sort_properties: SortProperties::Ordered(member.options),
            range: range()?,
            preserves_lex_ordering: false,
            strictly_order_preserving: true,
        })
    } else if expr.is::<Column>() {
        Ok(ExprProperties {
            sort_properties: SortProperties::Unordered,
            range: range()?,
            preserves_lex_ordering: false,
            strictly_order_preserving: true,
        })
    } else if let Some(literal) = expr.downcast_ref::<Literal>() {
        Ok(ExprProperties {
            sort_properties: SortProperties::Singleton,
            range: literal.value().into(),
            preserves_lex_ordering: true,
            strictly_order_preserving: true,
        })
    } else {
        let children = (expr.children().into_iter())
            .map(|c| image_properties(c, member, schema))
            .collect::<Result<Vec<_>>>()?;
        expr.get_properties(&children)
    }
}

fn push_new(list: &mut Vec<PhysicalSortExpr>, item: PhysicalSortExpr) {
    if !list.contains(&item) {
        list.push(item);
    }
}

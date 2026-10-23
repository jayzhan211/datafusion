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

//! The ordering questions: Q1 (does a requirement hold?), Q2 (how much of it
//! holds?) and Q3 (which permutation of some expressions holds?). All three
//! walk the provided orderings with one [`Walk`].

use std::sync::Arc;

use super::equalities::Equalities;
use super::properties::{EquivalenceProperties, NormalFacts, fixed_by, without_fixed};
use crate::expressions::{Column, Literal};
use crate::{LexOrdering, PhysicalExpr, PhysicalSortExpr};

use arrow::compute::SortOptions;
use arrow::datatypes::SchemaRef;
use datafusion_common::{Constraint, HashMap, HashSet, Result};
use datafusion_expr::interval_arithmetic::Interval;
use datafusion_expr::sort_properties::{ExprProperties, SortProperties};
use datafusion_physical_expr_common::sort_expr::{Ordering, Step, options_compatible};
use indexmap::IndexSet;

impl EquivalenceProperties {
    /// Q1: whether each partition is sorted (or grouped) as `requirement`
    /// asks.
    pub fn ordering_satisfy(
        &self,
        requirement: impl IntoIterator<Item = impl Into<Step>>,
    ) -> bool {
        let normal = self.normalize_requirement(requirement);
        normal.is_empty() || self.satisfied_len(&normal) == normal.len()
    }

    /// Q2: the longest prefix of `ordering` that holds, in normal form, and
    /// whether that is all of it.
    pub fn extract_common_sort_prefix(
        &self,
        ordering: LexOrdering,
    ) -> (Vec<PhysicalSortExpr>, bool) {
        let normal = self.normalize_requirement(ordering);
        let len = self.satisfied_len(&normal);
        let prefix = normal.iter().take(len).flat_map(Step::to_sort_exprs);
        (prefix.collect(), len == normal.len())
    }

    /// Q3: the longest permutation of `exprs` that the data is sorted by,
    /// with the index of each chosen expression in `exprs`. Each round takes
    /// every expression that is ordered (or constant) now, in input order,
    /// then pins them; constants come back `ASC NULLS FIRST`. With orderings
    /// `[a, b]` and `[c]`, `exprs = [c, b, a, d]` gives `[c, a, b]` and
    /// `[0, 2, 1]`.
    pub fn longest_permutation(
        &self,
        exprs: &[Arc<dyn PhysicalExpr>],
    ) -> (Vec<PhysicalSortExpr>, Vec<usize>) {
        let mut walk = Walk::new(self);
        let mut left: IndexSet<usize> = (0..exprs.len()).collect();
        let mut result = vec![];
        while !left.is_empty() {
            let round: Vec<_> = (left.iter())
                .filter_map(|&i| {
                    let expr = Arc::clone(&exprs[i]);
                    match walk.order_of(&expr).sort_properties {
                        SortProperties::Ordered(options) => {
                            Some((PhysicalSortExpr::new(expr, options), i))
                        }
                        SortProperties::Singleton => {
                            Some((PhysicalSortExpr::new_default(expr), i))
                        }
                        SortProperties::Unordered => None,
                    }
                })
                .collect();
            if round.is_empty() {
                break;
            }
            for (_, i) in &round {
                left.shift_remove(i);
            }
            // The last round needs no pins: nothing is left to ask about.
            if !left.is_empty() {
                for (sort_expr, _) in &round {
                    walk.pin(Arc::clone(&sort_expr.expr));
                }
            }
            result.extend(round);
        }
        result.into_iter().unzip()
    }

    /// Whether requirement `given` is equal to or more specific than
    /// `reference`, after both are normalized.
    pub fn requirements_compatible(&self, given: Ordering, reference: Ordering) -> bool {
        let given = self.normalize_requirement(given);
        let reference = self.normalize_requirement(reference);
        given.is_empty()
            || reference.is_empty()
            || (reference.len() <= given.len()
                && reference
                    .iter()
                    .zip(given.iter())
                    .all(|(r, g)| g.compatible(r)))
    }

    /// How `expr` is ordered, with its range: the expression tree evaluated
    /// against the orderings and constants.
    pub fn get_expr_properties(&self, expr: &Arc<dyn PhysicalExpr>) -> ExprProperties {
        Walk::new(self).order_of(expr)
    }

    /// How many leading steps of `normal` (a requirement in normal form) hold.
    fn satisfied_len(&self, normal: &Ordering) -> usize {
        if self.satisfied_by_keys(normal) {
            return normal.len();
        }
        let mut walk = Walk::new(self);
        let Some((last, steps)) = normal.split_last() else {
            return 0;
        };
        for (i, step) in steps.iter().enumerate() {
            if !walk.match_step(step) {
                return i;
            }
        }
        // Nothing follows the last step, so it is checked without pinning.
        let last_holds = match last {
            Step::Sort(members) => members.iter().all(|m| walk.holds(m)),
            Step::Group(_) => walk.match_step(last),
        };
        if last_holds {
            normal.len()
        } else {
            steps.len()
        }
    }

    /// A PRIMARY KEY or UNIQUE whose columns all sit in one stored ordering
    /// that `normal` starts with satisfies `normal` and every extension of
    /// it: no two rows tie on the key. A nullable UNIQUE column counts only
    /// when it is neither first nor last in that ordering (UNKNOWN U5, kept
    /// as before). The keys stay outside the walk until that rule is
    /// understood.
    fn satisfied_by_keys(&self, normal: &Ordering) -> bool {
        let schema = self.schema();
        self.constraints().iter().any(|constraint| {
            let (Constraint::PrimaryKey(indices) | Constraint::Unique(indices)) =
                constraint;
            let check_null = matches!(constraint, Constraint::Unique(_));
            indices.len() <= normal.len()
                && self.orderings().iter().any(|ordering| {
                    let length = ordering.len();
                    if indices.len() > length || normal.len() < length {
                        return false;
                    }
                    let mut positions = HashMap::with_capacity(length);
                    for (pos, step) in ordering.iter().enumerate() {
                        for expr in step.exprs() {
                            if let Some(col) = expr.downcast_ref::<Column>() {
                                let nullable = col.nullable(schema).unwrap_or(true);
                                positions.insert(col.index(), (pos, nullable));
                            }
                        }
                    }
                    let key_fits = indices.iter().all(|idx| {
                        positions.get(idx).is_some_and(|&(pos, nullable)| {
                            !check_null || !nullable || (pos != 0 && pos != length - 1)
                        })
                    });
                    key_fits
                        && (normal.iter().zip(ordering.iter())).all(
                            |(given, existing)| step_satisfies(existing, given, schema),
                        )
                })
        })
    }
}

/// Whether the provided step `existing` satisfies the required step `given`
/// on its own.
fn step_satisfies(existing: &Step, given: &Step, schema: &SchemaRef) -> bool {
    match (existing, given) {
        (Step::Sort(members), Step::Sort(required)) => required.iter().all(|r| {
            members.iter().any(|m| {
                let nullable = m.expr.nullable(schema).unwrap_or(true);
                m.expr.eq(&r.expr) && options_compatible(&m.options, &r.options, nullable)
            })
        }),
        (Step::Sort(members), Step::Group(exprs)) => {
            exprs.iter().all(|e| members.iter().any(|m| m.expr.eq(e)))
        }
        (Step::Group(provided), Step::Group(exprs)) => provided == exprs,
        (Step::Group(_), Step::Sort(_)) => false,
    }
}

/// The state of one walk over the provided orderings.
///
/// An ordering constrains a step only among the rows that tie on its earlier
/// steps. So once a requirement step holds, its expressions are *pinned*:
/// treated as constant for the rest of the walk. The pinned set is shared by
/// all provided orderings: with `[a, b]` and `[a, c]`, the requirement
/// `[a, c]` holds, because pinning `a` puts `c` at the head of the second
/// ordering.
pub(super) struct Walk<'a> {
    props: &'a EquivalenceProperties,
    normal: &'a NormalFacts,
    /// The provided orderings this walk reads (in normal form).
    base: &'a [Ordering],
    /// Expressions with one value among the rows that tie on everything
    /// matched so far, in normal form.
    pinned: HashSet<Arc<dyn PhysicalExpr>>,
    /// The other members of each multi-member `Sort` step of `base`, by
    /// member: pinning one pins them.
    members: HashMap<Arc<dyn PhysicalExpr>, Vec<Arc<dyn PhysicalExpr>>>,
    /// Orderings OD reduce finds once pins have moved a head (see
    /// [`derived_orderings`]). Each holds among the rows that tie on the
    /// pins it was found with; pins only grow, so it keeps holding.
    derived: Vec<Ordering>,
    /// For each ordering (`base`, then `derived`), the index of its first
    /// step that pins and constants do not fix. Pins only grow, so a cursor
    /// only moves forward.
    cursors: Vec<usize>,
    /// The step at each cursor, without its determined parts; orderings
    /// whose steps are all fixed have none.
    heads: Vec<Step>,
}

impl<'a> Walk<'a> {
    pub(super) fn new(props: &'a EquivalenceProperties) -> Self {
        Self::with_orderings(props, &props.normal().orderings)
    }

    /// A walk that sees only `ordering` (in normal form), for projecting it.
    pub(super) fn over(props: &'a EquivalenceProperties, ordering: &'a Ordering) -> Self {
        Self::with_orderings(props, std::slice::from_ref(ordering))
    }

    fn with_orderings(props: &'a EquivalenceProperties, base: &'a [Ordering]) -> Self {
        let mut members: HashMap<_, Vec<_>> = HashMap::new();
        for step in base.iter().flat_map(|o| o.iter()) {
            if let Step::Sort(sorts) = step
                && sorts.len() > 1
            {
                for sort in sorts {
                    let others = sorts.iter().filter(|m| !m.expr.eq(&sort.expr));
                    let entry = members.entry(Arc::clone(&sort.expr)).or_default();
                    entry.extend(others.map(|m| Arc::clone(&m.expr)));
                }
            }
        }
        let mut walk = Self {
            props,
            normal: props.normal(),
            base,
            pinned: HashSet::new(),
            members,
            derived: vec![],
            cursors: vec![0; base.len()],
            heads: vec![],
        };
        walk.advance();
        walk
    }

    fn eq(&self) -> &Equalities {
        self.props.equalities()
    }

    fn ordering(&self, i: usize) -> &Ordering {
        self.base
            .get(i)
            .unwrap_or_else(|| &self.derived[i - self.base.len()])
    }

    fn orderings(&self) -> impl Iterator<Item = &Ordering> {
        self.base.iter().chain(&self.derived)
    }

    /// Whether `expr` (in normal form) has one value among the rows that tie
    /// on the pinned expressions: a literal, a constant, a pinned
    /// expression, or a function of such.
    pub(super) fn is_determined(&self, expr: &Arc<dyn PhysicalExpr>) -> bool {
        fixed_by(expr, &|e| {
            self.pinned.contains(e) || self.normal.constants.contains_key(e)
        })
    }

    /// `step` without its determined parts (see [`without_fixed`]).
    fn undetermined(&self, step: &Step) -> Option<Step> {
        without_fixed(step, |e| self.is_determined(e))
    }

    /// Brings the walk up to date with the pins: moves each cursor past the
    /// steps that are now fixed, runs OD reduce on the orderings whose head
    /// may be a function of the steps after it, and refreshes the heads.
    /// Each cursor moves forward only, so a walk over `n` steps costs `O(n)`
    /// cursor moves in total, plus one head per ordering per pin.
    fn advance(&mut self) {
        loop {
            self.cursors.resize(self.base.len() + self.derived.len(), 0);
            for i in 0..self.cursors.len() {
                let ordering = self.ordering(i);
                let mut cursor = self.cursors[i];
                while cursor < ordering.len()
                    && self.undetermined(&ordering[cursor]).is_none()
                {
                    cursor += 1;
                }
                self.cursors[i] = cursor;
            }
            if self.pinned.is_empty() {
                // The stored orderings were reduced when they were built.
                break;
            }
            let rests: Vec<Ordering> = (0..self.cursors.len())
                .filter(|&i| {
                    let ordering = self.ordering(i);
                    let cursor = self.cursors[i];
                    cursor < ordering.len() && may_reduce(self.eq(), &ordering[cursor])
                })
                .map(|i| {
                    let steps = self.ordering(i)[self.cursors[i]..].iter();
                    steps.filter_map(|s| self.undetermined(s)).collect()
                })
                .collect();
            if rests.is_empty() {
                break;
            }
            let known: Vec<&Ordering> = self.orderings().chain(&rests).collect();
            let derived =
                derived_orderings(self.eq(), self.props.schema(), &rests, &known);
            if derived.is_empty() {
                break;
            }
            self.derived.extend(derived);
        }
        self.heads = (0..self.cursors.len())
            .filter_map(|i| {
                let ordering = self.ordering(i);
                let step = ordering.get(self.cursors[i])?;
                self.undetermined(step)
            })
            .collect();
    }

    /// How `expr` is ordered among the rows that tie on the pinned
    /// expressions. Evaluates the whole tree through
    /// [`PhysicalExpr::get_properties`], so `a + b` is ordered when both `a`
    /// and `b` lead an ordering; any node that is determined or leads an
    /// ordering takes that order instead of its children's. An error
    /// anywhere makes the answer unknown.
    pub(super) fn order_of(&self, expr: &Arc<dyn PhysicalExpr>) -> ExprProperties {
        self.try_order_of(expr)
            .unwrap_or_else(|_| ExprProperties::new_unknown())
    }

    fn try_order_of(&self, expr: &Arc<dyn PhysicalExpr>) -> Result<ExprProperties> {
        let children = expr.children();
        let mut props = if !children.is_empty() {
            let children = (children.into_iter())
                .map(|c| self.try_order_of(c))
                .collect::<Result<Vec<_>>>()?;
            expr.get_properties(&children)?
        } else if expr.is::<Literal>() {
            expr.get_properties(&[])?
        } else if expr.is::<Column>() {
            // A column maps to itself, which is trivially strict.
            let range = Interval::make_unbounded(&expr.data_type(self.props.schema())?)?;
            ExprProperties {
                range,
                strictly_order_preserving: true,
                ..ExprProperties::new_unknown()
            }
        } else {
            ExprProperties::new_unknown()
        };
        let normal = self.eq().normalize(Arc::clone(expr));
        if self.is_determined(&normal) {
            props.sort_properties = SortProperties::Singleton;
        } else if let Some(order) = head_order(&self.heads, &normal) {
            props.sort_properties = order;
        }
        Ok(props)
    }

    /// Matches one requirement step (in normal form) and pins it on
    /// success. A `Sort` step holds if every member is ordered with
    /// compatible options, or determined.
    pub(super) fn match_step(&mut self, step: &Step) -> bool {
        match step {
            Step::Sort(members) => {
                if !members.iter().all(|m| self.holds(m)) {
                    return false;
                }
                for m in members {
                    self.pin_satisfied(Arc::clone(&m.expr));
                }
                true
            }
            Step::Group(exprs) => self.match_group(exprs),
        }
    }

    /// Whether the data is sorted as `sort` asks at this point of the walk:
    /// ordered with compatible options, or determined.
    pub(super) fn holds(&self, sort: &PhysicalSortExpr) -> bool {
        let nullable = sort.expr.nullable(self.props.schema()).unwrap_or(true);
        match self.order_of(&sort.expr).sort_properties {
            SortProperties::Singleton => true,
            SortProperties::Ordered(o) => options_compatible(&o, &sort.options, nullable),
            SortProperties::Unordered => false,
        }
    }

    /// A required `Group` holds once all of its expressions are pinned. Each
    /// round takes an expression that is ordered (either direction) or
    /// determined now, or else a provided `Group` head that lies inside what
    /// is left. A larger provided group is not enough: grouped on
    /// `{p1, p2}` does not mean grouped on `{p1}`.
    fn match_group(&mut self, exprs: &IndexSet<Arc<dyn PhysicalExpr>>) -> bool {
        let mut left: Vec<_> = exprs.iter().cloned().collect();
        while !left.is_empty() {
            let ordered = left.iter().position(|e| {
                !matches!(self.order_of(e).sort_properties, SortProperties::Unordered)
            });
            if let Some(i) = ordered {
                let expr = left.remove(i);
                self.pin_satisfied(expr);
                continue;
            }
            let fits = self.heads.iter().find_map(|head| match head {
                Step::Group(g) if g.iter().all(|e| left.contains(e)) => Some(g.clone()),
                _ => None,
            });
            let Some(group) = fits else { return false };
            left.retain(|e| !group.contains(e));
            group.into_iter().for_each(|e| self.pin(e));
        }
        true
    }

    /// Pins a requirement expression that just held, and the expressions it
    /// fixes: if it is strictly order preserving, equal outputs mean equal
    /// ordered inputs, so `CAST(a)` fixes `a`. Strictness is judged before
    /// the expression itself is pinned.
    pub(super) fn pin_satisfied(&mut self, expr: Arc<dyn PhysicalExpr>) {
        let mut stack = vec![expr];
        while let Some(expr) = stack.pop() {
            if self.order_of(&expr).strictly_order_preserving {
                for child in expr.children() {
                    if matches!(
                        self.order_of(child).sort_properties,
                        SortProperties::Ordered(_)
                    ) {
                        stack.push(Arc::clone(child));
                    }
                }
            }
            self.pin(expr);
        }
    }

    /// Pins `expr` and the other members of each `Sort` step it is a member
    /// of (members are strict images of each other, checked on their own
    /// when projection built the step), then moves the heads.
    pub(super) fn pin(&mut self, expr: Arc<dyn PhysicalExpr>) {
        let mut stack = vec![self.eq().normalize(expr)];
        while let Some(expr) = stack.pop() {
            if !self.pinned.insert(Arc::clone(&expr)) {
                continue;
            }
            if let Some(others) = self.members.get(&expr) {
                stack.extend(others.iter().cloned());
            }
        }
        self.advance();
    }
}

/// How `expr` is ordered as a member of a head `Sort` step: the options of
/// the first head it leads. Leading in both a direction and its full
/// reverse (`ASC NULLS LAST` and `DESC NULLS FIRST`) means one value among
/// the ties (F11).
fn head_order(heads: &[Step], expr: &Arc<dyn PhysicalExpr>) -> Option<SortProperties> {
    let options: Vec<SortOptions> = heads
        .iter()
        .filter_map(|h| match h {
            Step::Sort(members) => members.iter().find(|m| m.expr.eq(expr)),
            Step::Group(_) => None,
        })
        .map(|m| m.options)
        .collect();
    let reverse = |o: &SortOptions| SortOptions::new(!o.descending, !o.nulls_first);
    if options.iter().any(|o| options.contains(&reverse(o))) {
        return Some(SortProperties::Singleton);
    }
    options.first().map(|o| SortProperties::Ordered(*o))
}

/// OD reduce (#9812): `[c, x1..xk, ..]` also orders by `[x1..xk, ..]` if `c`
/// or a member of its class is `f(x1..xk)` whose children are exactly the
/// next k steps, where `f` preserves lexicographic order and gives exactly
/// `c`'s options. Returns the orderings derived this way from `orderings`
/// (in normal form, without determined steps) that are not in `known`,
/// repeating on what it derives.
pub(super) fn derived_orderings(
    eq: &Equalities,
    schema: &SchemaRef,
    orderings: &[Ordering],
    known: &[&Ordering],
) -> Vec<Ordering> {
    let mut result: Vec<Ordering> = vec![];
    let mut todo: Vec<Ordering> = orderings.to_vec();
    while let Some(ordering) = todo.pop() {
        if !derived_from_next(eq, schema, &ordering) {
            continue;
        }
        let rest = Ordering::new(ordering.iter().skip(1).cloned());
        let seen = known.iter().any(|o| **o == rest) || result.contains(&rest);
        if !rest.is_empty() && !seen && !orderings.contains(&rest) {
            todo.push(rest.clone());
            result.push(rest);
        }
    }
    result
}

/// A cheap first test for [`derived_from_next`]: whether `head` is a `Sort`
/// step with a member that is, or is equal to, a function of something.
fn may_reduce(eq: &Equalities, head: &Step) -> bool {
    let Step::Sort(members) = head else {
        return false;
    };
    members.iter().any(|m| match eq.class_of(&m.expr) {
        Some(class) => class.iter().any(|e| !e.children().is_empty()),
        None => !m.expr.children().is_empty(),
    })
}

/// Whether the head of `ordering` is a lexicographic function of the steps
/// that follow it (see [`derived_orderings`]).
fn derived_from_next(eq: &Equalities, schema: &SchemaRef, ordering: &Ordering) -> bool {
    let [Step::Sort(head), rest @ ..] = &ordering[..] else {
        return false;
    };
    head.iter().any(|head| {
        let class = eq.class_of(&head.expr);
        let candidates = class
            .map(|c| c.iter().collect::<Vec<_>>())
            .unwrap_or_else(|| vec![&head.expr]);
        candidates.into_iter().any(|f| {
            let children = f.children();
            if children.is_empty() || children.len() > rest.len() {
                return false;
            }
            let child_props = children.iter().zip(rest).map(|(child, step)| {
                let Step::Sort(members) = step else {
                    return None;
                };
                let member = members.iter().find(|m| m.expr.eq(*child))?;
                let range =
                    Interval::make_unbounded(&child.data_type(schema).ok()?).ok()?;
                Some(ExprProperties {
                    sort_properties: SortProperties::Ordered(member.options),
                    range,
                    preserves_lex_ordering: true,
                    strictly_order_preserving: true,
                })
            });
            let Some(child_props) = child_props.collect::<Option<Vec<_>>>() else {
                return false;
            };
            f.get_properties(&child_props).is_ok_and(|p| {
                p.preserves_lex_ordering
                    && p.sort_properties == SortProperties::Ordered(head.options)
            })
        })
    })
}

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

//! `a = b` facts. They hold on every row, so merging partitions never
//! affects them.

use std::fmt::{self, Display};
use std::sync::Arc;

use crate::PhysicalExpr;
use crate::equivalence::ProjectionMapping;
use crate::expressions::Literal;
use crate::physical_expr::add_offset_to_expr;
use crate::projection::ProjectionTargets;

use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{HashMap, Result};
use datafusion_physical_expr_common::physical_expr::format_physical_expr_list;
use indexmap::{IndexMap, IndexSet};

/// Disjoint classes of expressions that are equal on every row.
///
/// Invariants, kept by [`Self::add`], the only writer:
/// - classes are disjoint and have at least two members;
/// - no member is a `Literal` (`a = 5` is a constant, see
///   `EquivalenceProperties::add_equal`);
/// - the canonical member of a class is its first member, and merging two
///   classes appends the later class to the earlier one, so a canonical
///   member never changes because of a later fact;
/// - `index[e] == i` exactly when `e` is a member of `classes[i]`.
#[derive(Clone, Debug, Default)]
pub struct Equalities {
    classes: Vec<IndexSet<Arc<dyn PhysicalExpr>>>,
    index: HashMap<Arc<dyn PhysicalExpr>, usize>,
}

impl Equalities {
    /// Records `a = b`. Literals are not members (see the invariants);
    /// callers turn `a = 5` into a constant instead.
    pub(super) fn add(&mut self, a: Arc<dyn PhysicalExpr>, b: Arc<dyn PhysicalExpr>) {
        if a.is::<Literal>() || b.is::<Literal>() {
            return;
        }
        match (self.index.get(&a).copied(), self.index.get(&b).copied()) {
            (Some(i), Some(j)) if i == j => {}
            (Some(i), Some(j)) => {
                let (keep, gone) = (i.min(j), i.max(j));
                let moved = self.classes.remove(gone);
                // Classes after `gone` moved down by one.
                for idx in self.index.values_mut() {
                    if *idx > gone {
                        *idx -= 1;
                    }
                }
                for e in &moved {
                    self.index.insert(Arc::clone(e), keep);
                }
                self.classes[keep].extend(moved);
            }
            (Some(i), None) => {
                self.index.insert(Arc::clone(&b), i);
                self.classes[i].insert(b);
            }
            (None, Some(j)) => {
                self.index.insert(Arc::clone(&a), j);
                self.classes[j].insert(a);
            }
            (None, None) if a.eq(&b) => {}
            (None, None) => {
                let i = self.classes.len();
                self.index.insert(Arc::clone(&a), i);
                self.index.insert(Arc::clone(&b), i);
                self.classes.push(IndexSet::from([a, b]));
            }
        }
    }

    /// Adds every equality of `other`.
    pub(super) fn extend(&mut self, other: Equalities) {
        for class in other.classes {
            let mut members = class.into_iter();
            if let Some(first) = members.next() {
                members.for_each(|e| self.add(Arc::clone(&first), e));
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// The number of classes.
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// The classes, each with its canonical member first.
    pub fn iter(&self) -> impl Iterator<Item = &IndexSet<Arc<dyn PhysicalExpr>>> {
        self.classes.iter()
    }

    /// The class containing `expr`, if `expr` is equal to anything else.
    pub fn class_of(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
    ) -> Option<&IndexSet<Arc<dyn PhysicalExpr>>> {
        self.index.get(expr).map(|i| &self.classes[*i])
    }

    /// Q5: whether `a` and `b` are equal on every row.
    pub fn are_equal(
        &self,
        a: &Arc<dyn PhysicalExpr>,
        b: &Arc<dyn PhysicalExpr>,
    ) -> bool {
        self.normalize(Arc::clone(a))
            .eq(&self.normalize(Arc::clone(b)))
    }

    /// Replaces every sub-expression that has a class by the class's
    /// canonical member, children first.
    pub fn normalize(&self, expr: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        if self.classes.is_empty() {
            return expr;
        }
        expr.transform(|e| match self.class_of(&e) {
            Some(class) => Ok(Transformed::yes(Arc::clone(&class[0]))),
            None => Ok(Transformed::no(e)),
        })
        .data()
        .expect("the closure never fails")
    }

    /// The same equalities with column indices moved by `offset` (the right
    /// side of a join).
    pub(super) fn shifted(&self, offset: usize) -> Result<Self> {
        let mut result = Self::default();
        for class in &self.classes {
            let mut members = class
                .iter()
                .map(|e| add_offset_to_expr(Arc::clone(e), offset as _));
            let Some(first) = members.next().transpose()? else {
                continue;
            };
            for member in members {
                result.add(Arc::clone(&first), member?);
            }
        }
        Ok(result)
    }

    /// The same equalities with every expression rewritten by `f`.
    pub(super) fn try_map(
        &self,
        mut f: impl FnMut(&Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>>,
    ) -> Result<Self> {
        let mut result = Self::default();
        for class in &self.classes {
            let first = f(&class[0])?;
            for member in class.iter().skip(1) {
                result.add(Arc::clone(&first), f(member)?);
            }
        }
        Ok(result)
    }

    /// Rewrites `expr` in terms of a projection's output: an exact mapping
    /// source first, then child by child, then through a class member that
    /// is a mapping source. Returns `None` if `expr` cannot be expressed.
    pub fn project_expr(
        &self,
        mapping: &ProjectionMapping,
        expr: &Arc<dyn PhysicalExpr>,
    ) -> Option<Arc<dyn PhysicalExpr>> {
        if let Some(targets) = mapping.get(expr) {
            return Some(Arc::clone(&targets.first().0));
        }
        Self::project_indirect(mapping, &self.source_classes(mapping), expr)
    }

    /// [`Self::project_expr`] for many expressions, sharing the setup.
    pub fn project_expressions<'a>(
        &'a self,
        mapping: &'a ProjectionMapping,
        exprs: impl IntoIterator<Item = &'a Arc<dyn PhysicalExpr>> + 'a,
    ) -> impl Iterator<Item = Option<Arc<dyn PhysicalExpr>>> + 'a {
        let mut classes = None;
        exprs.into_iter().map(move |expr| {
            if let Some(targets) = mapping.get(expr) {
                return Some(Arc::clone(&targets.first().0));
            }
            let classes = classes.get_or_insert_with(|| self.source_classes(mapping));
            Self::project_indirect(mapping, classes, expr)
        })
    }

    /// The equalities that hold on a projection's output: the projected
    /// classes, plus all targets of one source, plus targets of sources that
    /// are equal (`a + c` and `b + c` with `a = b`).
    pub fn project(&self, mapping: &ProjectionMapping) -> Self {
        let mut result = Self::default();
        for class in &self.classes {
            let mut projected = self.project_expressions(mapping, class.iter()).flatten();
            if let Some(first) = projected.next() {
                projected.for_each(|e| result.add(Arc::clone(&first), e));
            }
        }
        let mut by_source = IndexMap::<_, Vec<Arc<dyn PhysicalExpr>>>::new();
        for (source, targets) in mapping.iter() {
            let normal = self.normalize(Arc::clone(source));
            let class = by_source.entry(normal).or_default();
            class.extend(targets.iter().map(|(target, _)| Arc::clone(target)));
        }
        for targets in by_source.into_values() {
            let mut targets = targets.into_iter();
            if let Some(first) = targets.next() {
                targets.for_each(|e| result.add(Arc::clone(&first), e));
            }
        }
        result
    }

    /// The class of each mapping source (if any), in mapping order.
    fn source_classes<'a>(
        &'a self,
        mapping: &'a ProjectionMapping,
    ) -> Vec<(&'a ProjectionTargets, &'a IndexSet<Arc<dyn PhysicalExpr>>)> {
        mapping
            .iter()
            .filter_map(|(source, targets)| Some((targets, self.class_of(source)?)))
            .collect()
    }

    /// Projects `expr` through its children, or else through a class member
    /// that is a mapping source. Trying the children first keeps the
    /// structure: projecting the class `a + 1 = b` must not collapse it to
    /// `b = b` when `a` survives the projection (#25922).
    fn project_indirect(
        mapping: &ProjectionMapping,
        source_classes: &[(&ProjectionTargets, &IndexSet<Arc<dyn PhysicalExpr>>)],
        expr: &Arc<dyn PhysicalExpr>,
    ) -> Option<Arc<dyn PhysicalExpr>> {
        if expr.is::<Literal>() {
            return Some(Arc::clone(expr));
        }
        let children = expr.children();
        if !children.is_empty() {
            let projected = children
                .into_iter()
                .map(|child| match mapping.get(child) {
                    Some(targets) => Some(Arc::clone(&targets.first().0)),
                    None => Self::project_indirect(mapping, source_classes, child),
                })
                .collect::<Option<Vec<_>>>();
            if let Some(children) = projected {
                return Some(Arc::clone(expr).with_new_children(children).unwrap());
            }
        }
        source_classes
            .iter()
            .find(|(_, class)| class.contains(expr))
            .map(|(targets, _)| Arc::clone(&targets.first().0))
    }
}

impl Display for Equalities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, class) in self.classes.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{{{}}}", format_physical_expr_list(class))?;
        }
        write!(f, "]")
    }
}

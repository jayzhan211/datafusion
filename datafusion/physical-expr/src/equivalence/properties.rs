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

//! The fact sheet one plan node carries about its output.

use std::fmt::{self, Display};
use std::sync::{Arc, OnceLock};

use super::equalities::Equalities;
use super::matching::derived_orderings;
use crate::expressions::{Literal, with_new_schema};
use crate::{LexOrdering, PhysicalExpr};

use arrow::datatypes::SchemaRef;
use datafusion_common::{Constraints, HashMap, Result, ScalarValue, plan_err};
use datafusion_physical_expr_common::sort_expr::{Ordering, Step};
use indexmap::{IndexMap, IndexSet};

/// What is known about a plan node's output, sorted by where it holds:
///
/// - equalities (`a = b`) hold on every row;
/// - global facts hold in every partition: constants with one value
///   everywhere, and keys (PRIMARY KEY / UNIQUE);
/// - local facts hold inside each partition on its own: constants whose
///   value may differ between partitions, and the orderings producers
///   declared (each partition is sorted; the partitions together are not).
///
/// Facts are stored as given. What they imply is worked out when a question
/// is asked (see `matching.rs`), so adding a fact never rewrites another.
///
/// Please see the [Using Ordering for Better Plans] blog for more details.
///
/// [Using Ordering for Better Plans]: https://datafusion.apache.org/blog/2025/03/11/ordering-analysis/
///
/// # Example
/// ```
/// # use std::sync::Arc;
/// # use arrow::datatypes::{Schema, Field, DataType, SchemaRef};
/// # use datafusion_physical_expr::EquivalenceProperties;
/// # use datafusion_physical_expr::expressions::col;
/// use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
/// # let schema: SchemaRef = Arc::new(Schema::new(vec![
/// #   Field::new("a", DataType::Int32, false),
/// #   Field::new("b", DataType::Int32, false),
/// #   Field::new("c", DataType::Int32, false),
/// # ]));
/// # let col_a = col("a", &schema).unwrap();
/// # let col_b = col("b", &schema).unwrap();
/// # let col_c = col("c", &schema).unwrap();
/// // Each partition is sorted by a ASC, c DESC, and b has one value in each
/// // partition:
/// let mut eq_properties = EquivalenceProperties::new(schema);
/// eq_properties.add_local_constant(col_b);
/// eq_properties.add_ordering([
///     PhysicalSortExpr::new_default(Arc::clone(&col_a)).asc(),
///     PhysicalSortExpr::new_default(Arc::clone(&col_c)).desc(),
/// ]);
/// assert!(eq_properties.ordering_satisfy([
///     PhysicalSortExpr::new_default(col_a).asc(),
///     PhysicalSortExpr::new_default(col_c).desc(),
/// ]));
/// assert_eq!(
///     eq_properties.to_string(),
///     "order: [[a@0 ASC, c@2 DESC]], constants: [b@1 (heterogeneous)]"
/// );
/// ```
#[derive(Clone, Debug)]
pub struct EquivalenceProperties {
    schema: SchemaRef,
    /// Read directly; written only through [`Self::facts_mut`].
    facts: Facts,
    /// The facts in normal form, computed on the first question and dropped
    /// by [`Self::facts_mut`], so every mutator invalidates it in one place.
    /// It exists for planning time: optimizer rules ask the same node many
    /// questions, and wide schemas make normalizing each time quadratic
    /// (#17261, #13748).
    normal: OnceLock<Arc<NormalFacts>>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Facts {
    /// `a = b` on every row.
    pub(super) eq: Equalities,
    /// True in every partition.
    pub(super) global: GlobalFacts,
    /// True inside each partition on its own. Only [`EquivalenceProperties::merged`]
    /// (everything) and [`EquivalenceProperties::reordered`] (orderings) remove
    /// local facts, so partition scope is handled in two places.
    pub(super) local: LocalFacts,
}

#[derive(Clone, Debug, Default)]
pub(super) struct GlobalFacts {
    /// One value in all partitions. `None`: the value is unknown. The first
    /// value recorded for an expression wins.
    pub(super) constants: IndexMap<Arc<dyn PhysicalExpr>, Option<ScalarValue>>,
    /// PRIMARY KEY / UNIQUE: no two rows agree on these columns.
    pub(super) keys: Constraints,
}

#[derive(Clone, Debug, Default)]
pub(super) struct LocalFacts {
    /// One value inside each partition; partitions may differ.
    pub(super) constants: IndexSet<Arc<dyn PhysicalExpr>>,
    /// Orderings exactly as producers declared them (scan `WITH ORDER`,
    /// sort, merge key, window, aggregate) or as projection, join and union
    /// translated them. Never one that is a prefix of another.
    pub(super) orderings: Vec<Ordering>,
}

/// Where a constant holds.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Scope {
    /// In every partition, with this value (`None`: unknown).
    Global(Option<ScalarValue>),
    /// Inside each partition.
    Local,
}

/// The facts in normal form (see [`EquivalenceProperties::normal`]).
#[derive(Debug, Default)]
pub(super) struct NormalFacts {
    /// Every stored constant, in normal form.
    pub(super) constants: HashMap<Arc<dyn PhysicalExpr>, Scope>,
    /// The stored orderings in normal form, without steps that constants
    /// fix, followed by the orderings OD reduce derives from them (see
    /// `matching::derived_orderings`).
    pub(super) orderings: Vec<Ordering>,
}

impl EquivalenceProperties {
    /// No facts.
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            facts: Facts::default(),
            normal: OnceLock::new(),
        }
    }

    /// Each partition is sorted by each of `orderings`.
    pub fn new_with_orderings(
        schema: SchemaRef,
        orderings: impl IntoIterator<Item = impl IntoIterator<Item = impl Into<Step>>>,
    ) -> Self {
        let mut result = Self::new(schema);
        result.add_orderings(orderings);
        result
    }

    pub(super) fn from_facts(schema: SchemaRef, facts: Facts) -> Self {
        Self {
            schema,
            facts,
            normal: OnceLock::new(),
        }
    }

    /// The only way to change facts. Drops the normal-form memo.
    pub(super) fn facts_mut(&mut self) -> &mut Facts {
        self.normal = OnceLock::new();
        &mut self.facts
    }

    /// The facts in normal form, computed once per set of facts.
    pub(super) fn normal(&self) -> &NormalFacts {
        self.normal
            .get_or_init(|| Arc::new(NormalFacts::new(&self.facts, &self.schema)))
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Records PRIMARY KEY / UNIQUE constraints.
    pub fn set_constraints(&mut self, constraints: Constraints) {
        self.facts_mut().global.keys = constraints;
    }

    /// Records PRIMARY KEY / UNIQUE constraints.
    pub fn with_constraints(mut self, constraints: Constraints) -> Self {
        self.set_constraints(constraints);
        self
    }

    pub fn constraints(&self) -> &Constraints {
        &self.facts.global.keys
    }

    /// Q5: the equalities.
    pub fn equalities(&self) -> &Equalities {
        &self.facts.eq
    }

    /// The orderings as stored (each partition is sorted by each of them).
    pub fn orderings(&self) -> &[Ordering] {
        &self.facts.local.orderings
    }

    /// The stored all-partition constants and their values.
    pub fn global_constants(
        &self,
    ) -> impl Iterator<Item = (&Arc<dyn PhysicalExpr>, &Option<ScalarValue>)> {
        self.facts.global.constants.iter()
    }

    /// The stored per-partition constants.
    pub fn local_constants(&self) -> impl Iterator<Item = &Arc<dyn PhysicalExpr>> {
        self.facts.local.constants.iter()
    }

    /// `a = b` on every row. `a = 5` is recorded as a constant instead, so
    /// equality classes never contain literals (the #20231 hazard).
    pub fn add_equal(&mut self, a: Arc<dyn PhysicalExpr>, b: Arc<dyn PhysicalExpr>) {
        match (a.downcast_ref::<Literal>(), b.downcast_ref::<Literal>()) {
            (Some(_), Some(_)) => {}
            (Some(lit), None) => self.add_constant(b, Some(lit.value().clone())),
            (None, Some(lit)) => self.add_constant(a, Some(lit.value().clone())),
            (None, None) => self.facts_mut().eq.add(a, b),
        }
    }

    /// `expr` has one value in all partitions (`value`, if known).
    pub fn add_constant(
        &mut self,
        expr: Arc<dyn PhysicalExpr>,
        value: Option<ScalarValue>,
    ) {
        if expr.is::<Literal>() {
            return;
        }
        // behavior kept: two expressions with the same known value are equal (X4).
        let same_value = value.as_ref().and_then(|v| {
            let unclassed = self.facts.eq.class_of(&expr).is_none();
            let mut others = self.facts.global.constants.iter();
            let other = others.find(|(e, w)| w.as_ref() == Some(v) && !e.eq(&&expr));
            unclassed
                .then_some(other)
                .flatten()
                .map(|(e, _)| Arc::clone(e))
        });
        let facts = self.facts_mut();
        if let Some(other) = same_value {
            facts.eq.add(other, Arc::clone(&expr));
        }
        facts.global.constants.entry(expr).or_insert(value);
    }

    /// `expr` has one value inside each partition (e.g. a no-group aggregate
    /// per partition, or a window's partition keys).
    pub fn add_local_constant(&mut self, expr: Arc<dyn PhysicalExpr>) {
        if expr.is::<Literal>() || self.facts.global.constants.contains_key(&expr) {
            return;
        }
        self.facts_mut().local.constants.insert(expr);
    }

    /// A producer declares that each partition is sorted by `ordering`.
    pub fn add_ordering(&mut self, ordering: impl IntoIterator<Item = impl Into<Step>>) {
        let ordering = Ordering::new(ordering.into_iter().map(Into::into));
        let orderings = &self.facts.local.orderings;
        if ordering.is_empty() || orderings.iter().any(|o| o.starts_with(&ordering)) {
            return;
        }
        let orderings = &mut self.facts_mut().local.orderings;
        orderings.retain(|o| !ordering.starts_with(o));
        orderings.push(ordering);
    }

    /// [`Self::add_ordering`] for each of `orderings`.
    pub fn add_orderings(
        &mut self,
        orderings: impl IntoIterator<Item = impl IntoIterator<Item = impl Into<Step>>>,
    ) {
        orderings.into_iter().for_each(|o| self.add_ordering(o));
    }

    /// The facts after a merge: an operator whose output partition reads
    /// more than one input partition (SortPreservingMerge,
    /// CoalescePartitions, a Repartition). Per-partition facts do not
    /// survive. If the merge keeps an order, `sorted_by` is the only ordering
    /// left.
    ///
    /// Sound over one input partition too, but it loses facts there; such
    /// callers keep their input's facts instead.
    pub fn merged(mut self, sorted_by: Option<LexOrdering>) -> Self {
        self.facts_mut().local = LocalFacts::default();
        if let Some(key) = sorted_by {
            self.add_ordering(key);
        }
        self
    }

    /// The facts after rows move within each partition: a sort by
    /// `sorted_by`, or (`None`) an operator that emits rows in no particular
    /// order. Orderings are replaced; constants and equalities stay, as rows
    /// keep their values. A sort that the data already satisfies changes
    /// nothing, so the orderings that implied it stay too (X36).
    pub fn reordered(mut self, sorted_by: Option<LexOrdering>) -> Self {
        match sorted_by {
            Some(key) if self.ordering_satisfy(key.clone()) => {}
            Some(key) => {
                self.facts_mut().local.orderings = vec![];
                self.add_ordering(key);
            }
            None => self.facts_mut().local.orderings = vec![],
        }
        self
    }

    /// Q4: whether `expr` has one value inside each partition.
    pub fn is_constant(&self, expr: &Arc<dyn PhysicalExpr>) -> bool {
        self.normal()
            .is_constant(&self.facts.eq.normalize(Arc::clone(expr)))
    }

    /// Q4 across partitions: `Some(value)` if `expr` has one value in all
    /// partitions, where `value` is `None` when it is unknown. An expression
    /// of constants counts as per-partition only (X6), as before.
    pub fn global_constant(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
    ) -> Option<Option<ScalarValue>> {
        if let Some(lit) = expr.downcast_ref::<Literal>() {
            return Some(Some(lit.value().clone()));
        }
        match self.stored_constant(expr)? {
            Scope::Global(value) => Some(value.clone()),
            Scope::Local => None,
        }
    }

    /// The stored constant fact about `expr` or a member of its class.
    pub(super) fn stored_constant(&self, expr: &Arc<dyn PhysicalExpr>) -> Option<&Scope> {
        let normal = self.facts.eq.normalize(Arc::clone(expr));
        self.normal().constants.get(&normal)
    }

    /// A requirement in normal form: canonical members, without the steps
    /// that all-partition constants fix. This is the only place that decides
    /// which constants a requirement may drop (V2): a per-partition constant
    /// still orders rows once partitions merge, so it stays.
    pub fn normalize_requirement(
        &self,
        requirement: impl IntoIterator<Item = impl Into<Step>>,
    ) -> Ordering {
        let is_global = |e: &Arc<dyn PhysicalExpr>| {
            e.is::<Literal>()
                || matches!(self.normal().constants.get(e), Some(Scope::Global(_)))
        };
        let steps = requirement.into_iter().map(Into::into);
        let normal = steps.filter_map(|step: Step| {
            let step = step.try_map(|e| Some(self.facts.eq.normalize(Arc::clone(e))))?;
            match step {
                Step::Sort(members) if members.iter().any(|m| is_global(&m.expr)) => None,
                Step::Sort(members) => Some(Step::Sort(members)),
                Step::Group(exprs) => {
                    let exprs: IndexSet<_> =
                        exprs.into_iter().filter(|e| !is_global(e)).collect();
                    (!exprs.is_empty()).then_some(Step::Group(exprs))
                }
            }
        });
        Ordering::new(normal)
    }

    /// Q6, the ordering as one list of sort expressions (merge keys): all
    /// stored orderings concatenated, as a requirement in normal form, cut
    /// at the first `Group` step.
    pub fn output_ordering(&self) -> Option<LexOrdering> {
        let concat = self
            .facts
            .local
            .orderings
            .iter()
            .flat_map(|o| o.iter().cloned());
        self.normalize_requirement(Ordering::new(concat)).to_lex()
    }

    /// The same facts over `schema`, which extends the current schema with
    /// columns appended at the end (window and ranking outputs), so every
    /// column index stays valid.
    pub fn with_extended_schema(mut self, schema: SchemaRef) -> Self {
        debug_assert!(schema.fields.len() >= self.schema.fields.len());
        self.schema = schema;
        self
    }

    /// Aligns with another schema of the same shape (a union's output
    /// schema, which may differ in nullability or names). Constant values are
    /// cast to the new types; a constant whose value cannot be cast is
    /// dropped (#23374).
    pub fn with_new_schema(self, schema: SchemaRef) -> Result<Self> {
        let aligned = self.schema.fields.len() == schema.fields.len()
            && (self.schema.fields.iter().zip(schema.fields.iter()))
                .all(|(lhs, rhs)| lhs.data_type().eq(rhs.data_type()));
        if !aligned {
            return plan_err!(
                "Schemas have to be aligned to rewrite equivalences:\n Old schema: {}\n New schema: {}",
                self.schema,
                schema
            );
        }
        let rewrite = |e: &Arc<dyn PhysicalExpr>| with_new_schema(Arc::clone(e), &schema);
        let mut facts = Facts {
            eq: self.facts.eq.try_map(rewrite)?,
            ..Default::default()
        };
        for (expr, value) in &self.facts.global.constants {
            let expr = rewrite(expr)?;
            let value = match value {
                Some(v) => match v.cast_to(&expr.data_type(&schema)?) {
                    Ok(cast) => Some(cast),
                    // Optimizer metadata: drop a stale value rather than fail.
                    Err(_) => continue,
                },
                None => None,
            };
            facts.global.constants.insert(expr, value);
        }
        facts.global.keys = self.facts.global.keys.clone();
        for expr in &self.facts.local.constants {
            facts.local.constants.insert(rewrite(expr)?);
        }
        for ordering in &self.facts.local.orderings {
            let steps = ordering.iter().map(|s| s.try_map_result(rewrite));
            facts
                .local
                .orderings
                .push(steps.collect::<Result<Ordering>>()?);
        }
        Ok(Self::from_facts(schema, facts))
    }
}

impl NormalFacts {
    fn new(facts: &Facts, schema: &SchemaRef) -> Self {
        let eq = &facts.eq;
        let norm = |e: &Arc<dyn PhysicalExpr>| eq.normalize(Arc::clone(e));
        let mut constants = HashMap::new();
        for (expr, value) in &facts.global.constants {
            constants
                .entry(norm(expr))
                .or_insert_with(|| Scope::Global(value.clone()));
        }
        for expr in &facts.local.constants {
            constants.entry(norm(expr)).or_insert(Scope::Local);
        }
        let mut normal = Self {
            constants,
            orderings: vec![],
        };
        for ordering in &facts.local.orderings {
            let steps = ordering.iter().filter_map(|s| s.try_map(|e| Some(norm(e))));
            let steps: Vec<_> = steps.filter_map(|s| normal.unfixed(s)).collect();
            let ordering = Ordering::new(steps);
            if !ordering.is_empty() && !normal.orderings.contains(&ordering) {
                normal.orderings.push(ordering);
            }
        }
        let derived = derived_orderings(eq, schema, &normal.orderings, &[]);
        normal.orderings.extend(derived);
        normal
    }

    /// Whether `expr` (in normal form) has one value inside each partition:
    /// a literal, a constant, or a function of such (X6).
    pub(super) fn is_constant(&self, expr: &Arc<dyn PhysicalExpr>) -> bool {
        if expr.is::<Literal>() || self.constants.contains_key(expr) {
            return true;
        }
        let children = expr.children();
        !children.is_empty() && children.into_iter().all(|c| self.is_constant(c))
    }

    /// `step` without what constants fix: `None` for a `Sort` step with a
    /// constant member (its members are strict images, so all are constant),
    /// and a `Group` step loses its constant expressions.
    fn unfixed(&self, step: Step) -> Option<Step> {
        match step {
            Step::Sort(m) if m.iter().any(|m| self.is_constant(&m.expr)) => None,
            Step::Sort(m) => Some(Step::Sort(m)),
            Step::Group(g) => {
                let g: IndexSet<_> =
                    g.into_iter().filter(|e| !self.is_constant(e)).collect();
                (!g.is_empty()).then_some(Step::Group(g))
            }
        }
    }
}

/// Format: `order: [[a@0 ASC]], eq: [{a@0, b@1}], constants: [c@2 (uniform: 1)]`,
/// leaving out empty parts; `No properties` when there are none.
impl Display for EquivalenceProperties {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = vec![];
        let orderings = &self.facts.local.orderings;
        if !orderings.is_empty() {
            let list: Vec<_> = orderings.iter().map(|o| format!("[{o}]")).collect();
            parts.push(format!("order: [{}]", list.join(", ")));
        }
        if !self.facts.eq.is_empty() {
            parts.push(format!("eq: {}", self.facts.eq));
        }
        let mut constants = vec![];
        for (expr, value) in &self.facts.global.constants {
            match value {
                Some(v) => constants.push(format!("{expr} (uniform: {v})")),
                None => constants.push(format!("{expr} (uniform: unknown)")),
            }
        }
        for expr in &self.facts.local.constants {
            constants.push(format!("{expr} (heterogeneous)"));
        }
        if !constants.is_empty() {
            parts.push(format!("constants: [{}]", constants.join(", ")));
        }
        if parts.is_empty() {
            write!(f, "No properties")
        } else {
            write!(f, "{}", parts.join(", "))
        }
    }
}

/// Writes a list of orderings as `[[a@0 ASC], [b@1 DESC]]`.
pub fn format_orderings(orderings: &[Ordering]) -> impl Display + '_ {
    struct Orderings<'a>(&'a [Ordering]);
    impl Display for Orderings<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "[")?;
            for (i, ordering) in self.0.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "[{ordering}]")?;
            }
            write!(f, "]")
        }
    }
    Orderings(orderings)
}

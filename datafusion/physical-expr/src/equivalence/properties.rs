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
use crate::{LexOrdering, PhysicalExpr, PhysicalSortExpr};

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
    /// It depends on `schema` too, which only `with_extended_schema`
    /// changes in place (see there).
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
    /// (everything) and [`EquivalenceProperties::reordered`] (orderings)
    /// remove local facts. Which one applies is the operator's call: an
    /// operator whose output partition reads several input partitions calls
    /// `merged`, except two kept from before, which keep the input's
    /// per-partition constants: `SortExec` without `preserve_partitioning`
    /// (calls `reordered`; UNKNOWN U7) and the collected build side of a
    /// join (CollectLeft hash join, nested loop and cross join keep the left
    /// side's constants).
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
    /// translated them. Never empty, and never one that is a prefix of
    /// another (both kept by [`EquivalenceProperties::add_ordering`]).
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

    /// Each partition is sorted by each of `orderings` (see
    /// [`Self::add_ordering`] for what a step with several members means).
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
        // behavior kept (X4): a constant that is new (in no class, not yet a
        // constant) becomes equal to the first all-partition constant with
        // the same value, two unknown values counting as the same.
        let is_new = self.facts.eq.class_of(&expr).is_none()
            && !self.facts.global.constants.contains_key(&expr)
            && !self.facts.local.constants.contains(&expr);
        let same_value = (self.facts.global.constants.iter())
            .find(|(_, other_value)| is_new && **other_value == value)
            .map(|(other, _)| Arc::clone(other));
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
    /// An empty ordering, or a prefix of a stored one, adds nothing.
    ///
    /// A `Sort` step with several members claims that they are strict
    /// images of each other on every row (equal on one means equal on all),
    /// not only sorted alike: walks pin them together, and merges keep them
    /// with the key. Projection builds such steps and checks it; any other
    /// caller must make sure of it, as it cannot be checked here (the
    /// members are usually plain output columns).
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
    /// left, with the strict images the input knows for each of its
    /// expressions (`with_members`).
    ///
    /// Sound over one input partition too, but it loses facts there; such
    /// callers keep their input's facts instead.
    pub fn merged(mut self, sorted_by: Option<LexOrdering>) -> Self {
        let sorted_by = sorted_by.map(|key| self.with_members(key));
        self.facts_mut().local = LocalFacts::default();
        if let Some(key) = sorted_by {
            self.add_ordering(key);
        }
        self
    }

    /// `key` as an ordering whose steps also hold the other members of each
    /// stored `Sort` step that a key expression is a member of (in normal
    /// form, with the same options). Members are strict images of each
    /// other on every row (see [`Step::Sort`]), so data sorted by `key` is
    /// sorted by them too: after `CAST(a) AS x`, a merge on `a` keeps `x`.
    fn with_members(&self, key: LexOrdering) -> Ordering {
        let normal = &self.normal().orderings;
        let multi: Vec<_> = (normal.iter().flat_map(|o| o.iter()))
            .filter_map(|step| match step {
                Step::Sort(members) if members.len() > 1 => Some(members),
                _ => None,
            })
            .collect();
        if multi.is_empty() {
            return key.into();
        }
        let steps = key.into_iter().map(|sort| {
            let expr = self.facts.eq.normalize(Arc::clone(&sort.expr));
            let canonical = PhysicalSortExpr::new(expr, sort.options);
            let mut members = vec![sort];
            for step in multi.iter().filter(|m| m.contains(&canonical)) {
                for member in step.iter() {
                    if *member != canonical && !members.contains(member) {
                        members.push(member.clone());
                    }
                }
            }
            Step::Sort(members)
        });
        Ordering::new(steps)
    }

    /// The facts after rows move within each partition: a sort by
    /// `sorted_by`, or (`None`) an operator that emits rows in no particular
    /// order. Orderings are replaced; constants and equalities stay, as rows
    /// keep their values. A sort that the data already satisfies changes
    /// nothing, so the orderings that implied it stay too (X36). Assumes
    /// rows stay in their partition; `SortExec` without
    /// `preserve_partitioning` calls it anyway (see `Facts::local`).
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
        // Not `fixed_by`: a function of all-partition constants is constant
        // only per partition (X6), so it stays too.
        let is_global = |e: &Arc<dyn PhysicalExpr>| {
            e.is::<Literal>()
                || matches!(self.normal().constants.get(e), Some(Scope::Global(_)))
        };
        let normalize =
            |e: &Arc<dyn PhysicalExpr>| self.facts.eq.normalize(Arc::clone(e));
        let steps = requirement.into_iter().map(Into::into);
        let normal = steps.filter_map(|step: Step| {
            let step = match step {
                // A group already in normal form keeps its set: rebuilding
                // it (hashing each expression again) costs more than this.
                Step::Group(exprs)
                    if exprs.iter().all(|e| Arc::ptr_eq(&normalize(e), e)) =>
                {
                    Step::Group(exprs)
                }
                step => step.try_map(|e| Some(normalize(e)))?,
            };
            without_fixed(step, is_global)
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
    ///
    /// The normal-form memo is kept: it reads the schema only for the types
    /// of the columns the facts mention (OD reduce), and those keep their
    /// index and type. Keeping it saves rebuilding it for every window
    /// expression.
    pub fn with_extended_schema(mut self, schema: SchemaRef) -> Self {
        debug_assert!(
            (self.schema.fields.iter().zip(schema.fields.iter()))
                .all(|(old, new)| old.data_type() == new.data_type())
                && schema.fields.len() >= self.schema.fields.len()
        );
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
            let constant = |e: &Arc<dyn PhysicalExpr>| normal.is_constant(e);
            let steps: Vec<_> =
                steps.filter_map(|s| without_fixed(s, constant)).collect();
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
        fixed_by(expr, &|e| self.constants.contains_key(e))
    }
}

/// Whether `expr` has one value wherever each expression that `known`
/// accepts has one: a literal, such an expression, or a function of such.
/// Constants (with `known` = stored constants) and the expressions a walk
/// has determined (stored constants and pins) are both this rule.
pub(super) fn fixed_by(
    expr: &Arc<dyn PhysicalExpr>,
    known: &dyn Fn(&Arc<dyn PhysicalExpr>) -> bool,
) -> bool {
    if expr.is::<Literal>() || known(expr) {
        return true;
    }
    let children = expr.children();
    !children.is_empty() && children.into_iter().all(|c| fixed_by(c, known))
}

/// `step` without what `fixed` fixes: `None` for a `Sort` step with a fixed
/// member (its members are strict images of each other, so all are fixed),
/// and a `Group` step without its fixed expressions (`None` if that is all
/// of them). One rule for requirements (`fixed` = all-partition constants,
/// V2), stored orderings (any constant) and walks (constants and pins).
pub(super) fn without_fixed(
    step: Step,
    fixed: impl Fn(&Arc<dyn PhysicalExpr>) -> bool,
) -> Option<Step> {
    match step {
        Step::Sort(_) if is_fixed(&step, &fixed) => None,
        Step::Sort(_) => Some(step),
        Step::Group(mut exprs) => {
            exprs.retain(|e| !fixed(e));
            (!exprs.is_empty()).then_some(Step::Group(exprs))
        }
    }
}

/// Whether [`without_fixed`] leaves nothing of `step`, without building
/// what it leaves.
pub(super) fn is_fixed(
    step: &Step,
    fixed: impl Fn(&Arc<dyn PhysicalExpr>) -> bool,
) -> bool {
    match step {
        Step::Sort(members) => members.iter().any(|m| fixed(&m.expr)),
        Step::Group(exprs) => exprs.iter().all(fixed),
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

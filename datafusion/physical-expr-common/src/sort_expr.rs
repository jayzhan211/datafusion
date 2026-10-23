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

//! Sort expressions

use std::cmp;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::vec::IntoIter;

use crate::physical_expr::{PhysicalExpr, fmt_sql};

use arrow::compute::kernels::sort::{SortColumn, SortOptions};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use datafusion_expr_common::columnar_value::ColumnarValue;
use indexmap::IndexSet;
/// Represents Sort operation for a column in a RecordBatch
///
/// Example:
/// ```
/// # use std::any::Any;
/// # use std::collections::HashMap;
/// # use std::fmt::{Display, Formatter};
/// # use std::hash::Hasher;
/// # use std::sync::Arc;
/// # use arrow::array::RecordBatch;
/// # use datafusion_common::Result;
/// # use arrow::compute::SortOptions;
/// # use arrow::datatypes::{DataType, Field, FieldRef, Schema};
/// # use datafusion_expr_common::columnar_value::ColumnarValue;
/// # use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
/// # use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
/// # // this crate doesn't have a physical expression implementation
/// # // so make a really simple one
/// # #[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// # struct MyPhysicalExpr;
/// # impl PhysicalExpr for MyPhysicalExpr {
/// #  fn data_type(&self, input_schema: &Schema) -> Result<DataType> {todo!()}
/// #  fn nullable(&self, input_schema: &Schema) -> Result<bool> {todo!() }
/// #  fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {todo!() }
/// #  fn return_field(&self, input_schema: &Schema) -> Result<FieldRef> { unimplemented!() }
/// #  fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {todo!()}
/// #  fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn PhysicalExpr>>) -> Result<Arc<dyn PhysicalExpr>> {todo!()}
/// # fn fmt_sql(&self, f: &mut Formatter<'_>) -> std::fmt::Result { todo!() }
/// # }
/// # impl Display for MyPhysicalExpr {
/// #    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "a") }
/// # }
/// # fn col(name: &str) -> Arc<dyn PhysicalExpr> { Arc::new(MyPhysicalExpr) }
/// // Sort by a ASC
/// let options = SortOptions::default();
/// let sort_expr = PhysicalSortExpr::new(col("a"), options);
/// assert_eq!(sort_expr.to_string(), "a ASC");
///
/// // Sort by a DESC NULLS LAST
/// let sort_expr = PhysicalSortExpr::new_default(col("a"))
///   .desc()
///   .nulls_last();
/// assert_eq!(sort_expr.to_string(), "a DESC NULLS LAST");
/// ```
#[derive(Clone, Debug, Eq)]
pub struct PhysicalSortExpr {
    /// Physical expression representing the column to sort
    pub expr: Arc<dyn PhysicalExpr>,
    /// Option to specify how the given column should be sorted
    pub options: SortOptions,
}

impl PhysicalSortExpr {
    /// Create a new PhysicalSortExpr
    pub fn new(expr: Arc<dyn PhysicalExpr>, options: SortOptions) -> Self {
        Self { expr, options }
    }

    /// Create a new PhysicalSortExpr with default [`SortOptions`]
    pub fn new_default(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self::new(expr, SortOptions::default())
    }

    /// Reverses the sort expression. For instance, `[a ASC NULLS LAST]` turns
    /// into `[a DESC NULLS FIRST]`. Such reversals are useful in planning, e.g.
    /// when constructing equivalent window expressions.
    pub fn reverse(&self) -> Self {
        let mut result = self.clone();
        result.options = !result.options;
        result
    }

    /// Set the sort options to ASC
    pub fn asc(mut self) -> Self {
        self.options.descending = false;
        self
    }

    /// Set the sort options to DESC
    pub fn desc(mut self) -> Self {
        self.options.descending = true;
        self
    }

    /// Set the sort options to NULLS FIRST
    pub fn nulls_first(mut self) -> Self {
        self.options.nulls_first = true;
        self
    }

    /// Set the sort options to NULLS LAST
    pub fn nulls_last(mut self) -> Self {
        self.options.nulls_first = false;
        self
    }

    /// Like [`PhysicalExpr::fmt_sql`] prints a [`PhysicalSortExpr`] in a SQL-like format.
    pub fn fmt_sql(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{} {}",
            fmt_sql(self.expr.as_ref()),
            to_str(&self.options)
        )
    }

    /// Evaluates the sort expression into a `SortColumn` that can be passed
    /// into the arrow sort kernel.
    pub fn evaluate_to_sort_column(&self, batch: &RecordBatch) -> Result<SortColumn> {
        let array_to_sort = match self.expr.evaluate(batch)? {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(batch.num_rows())?,
        };
        Ok(SortColumn {
            values: array_to_sort,
            options: Some(self.options),
        })
    }

    /// Checks whether this sort expression satisfies the given `sort_expr`.
    /// See [`options_compatible`] for details on how sort options compare with
    /// one another.
    pub fn satisfy_expr(&self, sort_expr: &Self, schema: &Schema) -> bool {
        self.expr.eq(&sort_expr.expr)
            && options_compatible(
                &self.options,
                &sort_expr.options,
                self.expr.nullable(schema).unwrap_or(true),
            )
    }
}

/// Protobuf conversions for [`PhysicalSortExpr`].
///
/// This is the flat [`PhysicalSortExprNode`] representation used wherever the
/// wire format stores an ordering (scan output orderings, range partitioning,
/// window frames, …). It is *not* the `PhysicalExprNode::Sort` wrapping that
/// `SortExec` uses for its own `expr` field.
///
/// [`PhysicalSortExprNode`]: datafusion_proto_models::protobuf::PhysicalSortExprNode
#[cfg(feature = "proto")]
impl PhysicalSortExpr {
    /// Serialize this sort expression, encoding its child expression through
    /// `ctx`.
    pub fn try_to_proto(
        &self,
        ctx: &crate::physical_expr::proto_encode::PhysicalExprEncodeCtx<'_>,
    ) -> Result<datafusion_proto_models::protobuf::PhysicalSortExprNode> {
        let Self { expr, options } = self;
        Ok(datafusion_proto_models::protobuf::PhysicalSortExprNode {
            expr: Some(Box::new(ctx.encode_child(expr)?)),
            asc: !options.descending,
            nulls_first: options.nulls_first,
        })
    }

    /// Reconstruct a [`PhysicalSortExpr`] from its protobuf representation.
    pub fn try_from_proto(
        node: &datafusion_proto_models::protobuf::PhysicalSortExprNode,
        ctx: &crate::physical_expr::proto_decode::PhysicalExprDecodeCtx<'_>,
    ) -> Result<Self> {
        let datafusion_proto_models::protobuf::PhysicalSortExprNode {
            expr,
            asc,
            nulls_first,
        } = node;
        let expr =
            ctx.decode_required_expression(expr.as_deref(), "PhysicalSortExpr", "expr")?;
        Ok(PhysicalSortExpr {
            expr,
            options: SortOptions {
                descending: !asc,
                nulls_first: *nulls_first,
            },
        })
    }
}

/// Serialize a sequence of sort expressions into the flat
/// [`PhysicalSortExprNode`] list the wire format uses for an ordering.
///
/// Accepts anything that yields [`PhysicalSortExpr`]s by value or by reference,
/// so a [`LexOrdering`], a `&[PhysicalSortExpr]`, or the sort of an
/// [`Ordering`] requirement ([`Ordering::to_sort`]) all work:
///
/// ```ignore
/// let nodes = sort_exprs_try_to_proto(ordering.iter(), ctx)?;
/// ```
///
/// The `PhysicalSortExprNodeCollection` message some plans use is just this
/// list in a wrapper, so those callers wrap the result themselves rather than
/// this function guessing which shape they mean.
///
/// [`PhysicalSortExprNode`]: datafusion_proto_models::protobuf::PhysicalSortExprNode
#[cfg(feature = "proto")]
pub fn sort_exprs_try_to_proto<E: std::borrow::Borrow<PhysicalSortExpr>>(
    exprs: impl IntoIterator<Item = E>,
    ctx: &crate::physical_expr::proto_encode::PhysicalExprEncodeCtx<'_>,
) -> Result<Vec<datafusion_proto_models::protobuf::PhysicalSortExprNode>> {
    exprs
        .into_iter()
        .map(|expr| expr.borrow().try_to_proto(ctx))
        .collect()
}

/// Reconstruct a sequence of sort expressions from the flat
/// [`PhysicalSortExprNode`] list, the counterpart of
/// [`sort_exprs_try_to_proto`].
///
/// Returns the expressions rather than a [`LexOrdering`] or an
/// [`Ordering`], because callers differ in what an empty list means:
/// `LexOrdering::new` returns `None` for it, which is
/// "no ordering declared" for a scan and an error for an operator that requires
/// one. Callers with the former convention can use
/// [`optional_ordering_try_from_proto`] instead.
///
/// [`PhysicalSortExprNode`]: datafusion_proto_models::protobuf::PhysicalSortExprNode
#[cfg(feature = "proto")]
pub fn sort_exprs_try_from_proto(
    nodes: &[datafusion_proto_models::protobuf::PhysicalSortExprNode],
    ctx: &crate::physical_expr::proto_decode::PhysicalExprDecodeCtx<'_>,
) -> Result<Vec<PhysicalSortExpr>> {
    nodes
        .iter()
        .map(|node| PhysicalSortExpr::try_from_proto(node, ctx))
        .collect()
}

/// Serialize an optional [`LexOrdering`], encoding `None` as an empty list.
#[cfg(feature = "proto")]
pub fn optional_ordering_try_to_proto(
    ordering: Option<&LexOrdering>,
    ctx: &crate::physical_expr::proto_encode::PhysicalExprEncodeCtx<'_>,
) -> Result<Vec<datafusion_proto_models::protobuf::PhysicalSortExprNode>> {
    sort_exprs_try_to_proto(ordering.into_iter().flatten(), ctx)
}

/// Counterpart of [`optional_ordering_try_to_proto`]: an empty list decodes
/// as `None`.
#[cfg(feature = "proto")]
pub fn optional_ordering_try_from_proto(
    nodes: &[datafusion_proto_models::protobuf::PhysicalSortExprNode],
    ctx: &crate::physical_expr::proto_decode::PhysicalExprDecodeCtx<'_>,
) -> Result<Option<LexOrdering>> {
    Ok(LexOrdering::new(sort_exprs_try_from_proto(nodes, ctx)?))
}

impl PartialEq for PhysicalSortExpr {
    fn eq(&self, other: &Self) -> bool {
        self.options == other.options && self.expr.eq(&other.expr)
    }
}

impl Hash for PhysicalSortExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.hash(state);
        self.options.hash(state);
    }
}

impl Display for PhysicalSortExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} {}", self.expr, to_str(&self.options))
    }
}

/// Returns whether the given two [`SortOptions`] are compatible. Here,
/// compatibility means that they are either exactly equal, or they differ only
/// in whether NULL values come in first/last, which is immaterial because the
/// column in question is not nullable (specified by the `nullable` parameter).
pub fn options_compatible(
    options_lhs: &SortOptions,
    options_rhs: &SortOptions,
    nullable: bool,
) -> bool {
    if nullable {
        options_lhs == options_rhs
    } else {
        // If the column is not nullable, NULLS FIRST/LAST is not important.
        options_lhs.descending == options_rhs.descending
    }
}

/// One step of an [`Ordering`]. A step holds only among the rows that tie on
/// every earlier step of its ordering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// Sorted by each member, with that member's options. Never empty.
    ///
    /// In an ordering the data provides, the members are strict images of
    /// each other (two rows equal on one member are equal on all), so one
    /// member's value fixes the others. Only projection builds more than one
    /// member, and it checks strictness on the member alone (one input
    /// ordered, nothing else pinned). In a requirement, every member must
    /// hold; requirements built in-tree have one member.
    Sort(Vec<PhysicalSortExpr>),
    /// Rows with equal values of `exprs` are contiguous; nothing is said
    /// about the order between those groups. Never empty.
    ///
    /// As a requirement with one expression this is "sorted on it in either
    /// direction" (window `PARTITION BY` asks for one such step per key). A
    /// sort that satisfies it uses `ASC NULLS LAST`, as PostgreSQL does.
    Group(IndexSet<Arc<dyn PhysicalExpr>>),
}

impl Step {
    /// A one-member [`Step::Sort`].
    pub fn sort(expr: Arc<dyn PhysicalExpr>, options: SortOptions) -> Self {
        Step::Sort(vec![PhysicalSortExpr::new(expr, options)])
    }

    /// A [`Step::Group`] over `exprs`.
    pub fn group(exprs: impl IntoIterator<Item = Arc<dyn PhysicalExpr>>) -> Self {
        Step::Group(exprs.into_iter().collect())
    }

    /// The expressions this step is about.
    pub fn exprs(&self) -> impl Iterator<Item = &Arc<dyn PhysicalExpr>> {
        let (sorts, groups) = match self {
            Step::Sort(members) => (members.as_slice(), None),
            Step::Group(exprs) => (&[][..], Some(exprs)),
        };
        sorts
            .iter()
            .map(|m| &m.expr)
            .chain(groups.into_iter().flatten())
    }

    /// The first expression of this step, which is its only one for the
    /// steps of requirements built in-tree.
    pub fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        match self {
            Step::Sort(members) => &members[0].expr,
            Step::Group(exprs) => &exprs[0],
        }
    }

    /// The sort options of the first member, or `None` for a group.
    pub fn options(&self) -> Option<SortOptions> {
        match self {
            Step::Sort(members) => Some(members[0].options),
            Step::Group(_) => None,
        }
    }

    /// Rewrites every expression; `None` if any expression has no rewrite.
    pub fn try_map(
        &self,
        mut f: impl FnMut(&Arc<dyn PhysicalExpr>) -> Option<Arc<dyn PhysicalExpr>>,
    ) -> Option<Step> {
        Some(match self {
            Step::Sort(members) => Step::Sort(
                members
                    .iter()
                    .map(|m| Some(PhysicalSortExpr::new(f(&m.expr)?, m.options)))
                    .collect::<Option<_>>()?,
            ),
            Step::Group(exprs) => {
                Step::Group(exprs.iter().map(f).collect::<Option<_>>()?)
            }
        })
    }

    /// Like [`Self::try_map`], for rewrites that can fail with an error.
    pub fn try_map_result(
        &self,
        mut f: impl FnMut(&Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>>,
    ) -> Result<Step> {
        Ok(match self {
            Step::Sort(members) => Step::Sort(
                members
                    .iter()
                    .map(|m| Ok(PhysicalSortExpr::new(f(&m.expr)?, m.options)))
                    .collect::<Result<_>>()?,
            ),
            Step::Group(exprs) => {
                Step::Group(exprs.iter().map(f).collect::<Result<_>>()?)
            }
        })
    }

    /// Whether this requirement step is equal to or more specific than
    /// `reference`: the same expression, and `reference` either asks for the
    /// same options or is a group.
    pub fn compatible(&self, reference: &Step) -> bool {
        match (self, reference) {
            (_, Step::Group(exprs)) => exprs.len() == 1 && self.exprs().eq(exprs.iter()),
            (Step::Sort(given), Step::Sort(reference)) => given == reference,
            (Step::Group(_), Step::Sort(_)) => false,
        }
    }

    /// The sort expressions that satisfy this step as a requirement: each
    /// member, or each group expression `ASC NULLS LAST`.
    pub fn to_sort_exprs(&self) -> Vec<PhysicalSortExpr> {
        match self {
            Step::Sort(members) => members.clone(),
            Step::Group(exprs) => exprs
                .iter()
                .map(|e| {
                    PhysicalSortExpr::new(Arc::clone(e), SortOptions::new(false, false))
                })
                .collect(),
        }
    }
}

impl From<PhysicalSortExpr> for Step {
    fn from(sort_expr: PhysicalSortExpr) -> Self {
        Step::Sort(vec![sort_expr])
    }
}

impl From<&PhysicalSortExpr> for Step {
    fn from(sort_expr: &PhysicalSortExpr) -> Self {
        Step::Sort(vec![sort_expr.clone()])
    }
}

impl From<&Step> for Step {
    fn from(step: &Step) -> Self {
        step.clone()
    }
}

impl Display for Step {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Step::Sort(members) => {
                let mut first = true;
                for member in members {
                    if !first {
                        write!(f, " = ")?;
                    }
                    first = false;
                    write!(f, "{member}")?;
                }
                Ok(())
            }
            Step::Group(exprs) => {
                let mut first = true;
                for expr in exprs {
                    if !first {
                        write!(f, " = ")?;
                    }
                    first = false;
                    write!(f, "{expr}")?;
                }
                write!(f, " NA")
            }
        }
    }
}

/// Returns the SQL string representation of the given [`SortOptions`] object.
#[inline]
fn to_str(options: &SortOptions) -> &str {
    match (options.descending, options.nulls_first) {
        (true, true) => "DESC",
        (true, false) => "DESC NULLS LAST",
        (false, true) => "ASC",
        (false, false) => "ASC NULLS LAST",
    }
}

/// This object represents a lexicographical ordering and contains a vector
/// of `PhysicalSortExpr` objects.
///
/// For example, a `vec![a ASC, b DESC]` represents a lexicographical ordering
/// that first sorts by column `a` in ascending order, then by column `b` in
/// descending order.
///
/// # Invariants
///
/// The following always hold true for a `LexOrdering`:
///
/// 1. It is non-degenerate, meaning it contains at least one element.
/// 2. It is duplicate-free, meaning it does not contain multiple entries for
///    the same column.
#[derive(Clone, Debug)]
pub struct LexOrdering {
    /// Vector of sort expressions representing the lexicographical ordering.
    exprs: Vec<PhysicalSortExpr>,
    /// Set of expressions in the lexicographical ordering, used to ensure
    /// that the ordering is duplicate-free. Note that the elements in this
    /// set are the same underlying physical expressions as in `exprs`.
    set: IndexSet<Arc<dyn PhysicalExpr>>,
}

impl LexOrdering {
    /// Creates a new [`LexOrdering`] from the given vector of sort expressions.
    /// If the vector is empty, returns `None`.
    pub fn new(exprs: impl IntoIterator<Item = PhysicalSortExpr>) -> Option<Self> {
        let exprs = exprs.into_iter();
        let capacity = exprs.size_hint().0;
        let mut candidate = Self {
            // not valid yet; valid publicly-returned instance must be non-empty
            exprs: Vec::with_capacity(capacity),
            set: IndexSet::with_capacity(capacity),
        };
        for expr in exprs {
            candidate.push(expr);
        }
        if candidate.exprs.is_empty() {
            None
        } else {
            Some(candidate)
        }
    }

    /// Appends an element to the back of the `LexOrdering`.
    pub fn push(&mut self, sort_expr: PhysicalSortExpr) {
        if self.set.insert(Arc::clone(&sort_expr.expr)) {
            self.exprs.push(sort_expr);
        }
    }

    /// Add all elements from `iter` to the `LexOrdering`.
    pub fn extend(&mut self, sort_exprs: impl IntoIterator<Item = PhysicalSortExpr>) {
        for sort_expr in sort_exprs {
            self.push(sort_expr);
        }
    }

    /// Returns the leading `PhysicalSortExpr` of the `LexOrdering`. Note that
    /// this function does not return an `Option`, as a `LexOrdering` is always
    /// non-degenerate (i.e. it contains at least one element).
    pub fn first(&self) -> &PhysicalSortExpr {
        // Can safely `unwrap` because `LexOrdering` is non-degenerate:
        self.exprs.first().unwrap()
    }

    /// Returns the number of elements that can be stored in the `LexOrdering`
    /// without reallocating.
    pub fn capacity(&self) -> usize {
        self.exprs.capacity()
    }

    /// Truncates the `LexOrdering`, keeping only the first `len` elements.
    /// Returns `true` if truncation made a change, `false` otherwise. Negative
    /// cases happen in two scenarios: (1) When `len` is greater than or equal
    /// to the number of expressions inside this `LexOrdering`, making truncation
    /// a no-op, or (2) when `len` is `0`, making truncation impossible.
    pub fn truncate(&mut self, len: usize) -> bool {
        if len == 0 || len >= self.exprs.len() {
            return false;
        }
        for PhysicalSortExpr { expr, .. } in self.exprs[len..].iter() {
            self.set.swap_remove(expr);
        }
        self.exprs.truncate(len);
        true
    }

    /// Check if reversing this ordering would satisfy another ordering requirement.
    ///
    /// This supports **prefix matching**: if this ordering is `[A DESC, B ASC]`
    /// and `other` is `[A ASC]`, reversing this gives `[A ASC, B DESC]`, which
    /// satisfies `other` since `[A ASC]` is a prefix.
    ///
    /// # Arguments
    /// * `other` - The ordering requirement to check against
    ///
    /// # Returns
    /// `true` if reversing this ordering would satisfy `other`
    ///
    /// # Example
    /// ```text
    /// self:  [number DESC, letter ASC]
    /// other: [number ASC]
    /// After reversing self: [number ASC, letter DESC]  ✓ Prefix match!
    /// ```
    pub fn is_reverse(&self, other: &LexOrdering) -> bool {
        let self_exprs = self.as_ref();
        let other_exprs = other.as_ref();

        if other_exprs.len() > self_exprs.len() {
            return false;
        }

        other_exprs.iter().zip(self_exprs.iter()).all(|(req, cur)| {
            req.expr.eq(&cur.expr) && is_reversed_sort_options(&req.options, &cur.options)
        })
    }

    /// Returns the sort options for the given expression if one is defined in this `LexOrdering`.
    pub fn get_sort_options(&self, expr: &dyn PhysicalExpr) -> Option<SortOptions> {
        for e in self {
            if e.expr.as_ref().dyn_eq(expr) {
                return Some(e.options);
            }
        }

        None
    }
}

/// Check if two SortOptions represent reversed orderings.
///
/// Returns `true` if both `descending` and `nulls_first` are opposite.
///
/// # Example
/// ```
/// use arrow::compute::SortOptions;
/// # use datafusion_physical_expr_common::sort_expr::is_reversed_sort_options;
///
/// let asc_nulls_last = SortOptions {
///     descending: false,
///     nulls_first: false,
/// };
/// let desc_nulls_first = SortOptions {
///     descending: true,
///     nulls_first: true,
/// };
///
/// assert!(is_reversed_sort_options(&asc_nulls_last, &desc_nulls_first));
/// assert!(is_reversed_sort_options(&desc_nulls_first, &asc_nulls_last));
/// ```
pub fn is_reversed_sort_options(lhs: &SortOptions, rhs: &SortOptions) -> bool {
    lhs.descending != rhs.descending && lhs.nulls_first != rhs.nulls_first
}

impl PartialEq for LexOrdering {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            exprs,
            set: _, // derived from `exprs`
        } = self;
        // PartialEq must be consistent with PartialOrd
        exprs == &other.exprs
    }
}
impl Eq for LexOrdering {}
impl PartialOrd for LexOrdering {
    /// There is a partial ordering among `LexOrdering` objects. For example, the
    /// ordering `[a ASC]` is coarser (less) than ordering `[a ASC, b ASC]`.
    /// If two orderings do not share a prefix, they are incomparable.
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        // PartialEq must be consistent with PartialOrd
        self.exprs
            .iter()
            .zip(other.exprs.iter())
            .all(|(lhs, rhs)| lhs == rhs)
            .then(|| self.len().cmp(&other.len()))
    }
}

impl<const N: usize> From<[PhysicalSortExpr; N]> for LexOrdering {
    fn from(value: [PhysicalSortExpr; N]) -> Self {
        // TODO: Replace this assertion with a condition on the generic parameter
        //       when Rust supports it.
        assert!(N > 0);
        Self::new(value)
            .expect("A LexOrdering from non-empty array must be non-degenerate")
    }
}

impl Deref for LexOrdering {
    type Target = [PhysicalSortExpr];

    fn deref(&self) -> &Self::Target {
        self.exprs.as_slice()
    }
}

impl Display for LexOrdering {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut first = true;
        for sort_expr in &self.exprs {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{sort_expr}")?;
        }
        Ok(())
    }
}

impl IntoIterator for LexOrdering {
    type Item = PhysicalSortExpr;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.exprs.into_iter()
    }
}

impl<'a> IntoIterator for &'a LexOrdering {
    type Item = &'a PhysicalSortExpr;
    type IntoIter = std::slice::Iter<'a, PhysicalSortExpr>;

    fn into_iter(self) -> Self::IntoIter {
        self.exprs.iter()
    }
}

impl From<LexOrdering> for Vec<PhysicalSortExpr> {
    fn from(ordering: LexOrdering) -> Self {
        ordering.exprs
    }
}

/// A list of [`Step`]s: the one ordering type, used both for what the data
/// provides and for what an operator requires. `[Group{a}, Sort(c ASC)]`
/// reads "grouped on `a`, and sorted on `c` inside each group".
///
/// Invariant: no step repeats only expressions of earlier steps (such a step
/// holds trivially, because the earlier steps fix its value), so these are
/// dropped on construction, as `LexOrdering` drops duplicate expressions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ordering {
    steps: Vec<Step>,
}

impl Ordering {
    pub fn new(steps: impl IntoIterator<Item = Step>) -> Self {
        let mut ordering = Self::default();
        steps.into_iter().for_each(|s| ordering.push(s));
        ordering
    }

    /// The sort-only ordering of `sort_exprs`, e.g. a `WITH ORDER` clause or
    /// the key of a `SortExec`.
    pub fn from_sort_exprs(
        sort_exprs: impl IntoIterator<Item = PhysicalSortExpr>,
    ) -> Self {
        Self::new(sort_exprs.into_iter().map(Step::from))
    }

    /// Appends `step` unless every expression in it appeared earlier.
    pub fn push(&mut self, step: Step) {
        let seen = |e: &Arc<dyn PhysicalExpr>| {
            self.steps.iter().any(|s| s.exprs().any(|x| x.eq(e)))
        };
        if !step.exprs().all(seen) {
            self.steps.push(step);
        }
    }

    /// Keeps the first `len` steps.
    pub fn truncate(&mut self, len: usize) {
        self.steps.truncate(len);
    }

    /// Removes and returns the last step.
    pub fn pop(&mut self) -> Option<Step> {
        self.steps.pop()
    }

    pub fn into_steps(self) -> Vec<Step> {
        self.steps
    }

    /// Whether `prefix` is a prefix of this ordering.
    pub fn starts_with(&self, prefix: &Ordering) -> bool {
        self.steps.starts_with(&prefix.steps)
    }

    /// The provided ordering as sort expressions (Q6, merge keys): the first
    /// member of each `Sort` step, stopping at the first `Group` step, since
    /// merging grouped streams does not keep groups contiguous.
    pub fn to_lex(&self) -> Option<LexOrdering> {
        let sorts = self.steps.iter().map_while(|step| match step {
            Step::Sort(members) => Some(members[0].clone()),
            Step::Group(_) => None,
        });
        LexOrdering::new(sorts)
    }

    /// A sort that satisfies this requirement (see [`Step::to_sort_exprs`]).
    pub fn to_sort(&self) -> Option<LexOrdering> {
        LexOrdering::new(self.steps.iter().flat_map(Step::to_sort_exprs))
    }
}

impl Deref for Ordering {
    type Target = [Step];

    fn deref(&self) -> &Self::Target {
        self.steps.as_slice()
    }
}

impl IntoIterator for Ordering {
    type Item = Step;
    type IntoIter = IntoIter<Step>;

    fn into_iter(self) -> Self::IntoIter {
        self.steps.into_iter()
    }
}

impl<'a> IntoIterator for &'a Ordering {
    type Item = &'a Step;
    type IntoIter = std::slice::Iter<'a, Step>;

    fn into_iter(self) -> Self::IntoIter {
        self.steps.iter()
    }
}

impl FromIterator<Step> for Ordering {
    fn from_iter<T: IntoIterator<Item = Step>>(iter: T) -> Self {
        Self::new(iter)
    }
}

impl FromIterator<PhysicalSortExpr> for Ordering {
    fn from_iter<T: IntoIterator<Item = PhysicalSortExpr>>(iter: T) -> Self {
        Self::from_sort_exprs(iter)
    }
}

impl From<LexOrdering> for Ordering {
    fn from(ordering: LexOrdering) -> Self {
        Self::from_sort_exprs(ordering)
    }
}

impl From<&LexOrdering> for Ordering {
    fn from(ordering: &LexOrdering) -> Self {
        Self::from_sort_exprs(ordering.iter().cloned())
    }
}

impl From<Vec<PhysicalSortExpr>> for Ordering {
    fn from(sort_exprs: Vec<PhysicalSortExpr>) -> Self {
        Self::from_sort_exprs(sort_exprs)
    }
}

impl From<&[PhysicalSortExpr]> for Ordering {
    fn from(sort_exprs: &[PhysicalSortExpr]) -> Self {
        Self::from_sort_exprs(sort_exprs.iter().cloned())
    }
}

impl<const N: usize> From<[PhysicalSortExpr; N]> for Ordering {
    fn from(sort_exprs: [PhysicalSortExpr; N]) -> Self {
        Self::from_sort_exprs(sort_exprs)
    }
}

impl From<Vec<Step>> for Ordering {
    fn from(steps: Vec<Step>) -> Self {
        Self::new(steps)
    }
}

impl<const N: usize> From<[Step; N]> for Ordering {
    fn from(steps: [Step; N]) -> Self {
        Self::new(steps)
    }
}

impl Display for Ordering {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut first = true;
        for step in &self.steps {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{step}")?;
        }
        Ok(())
    }
}

/// Represents a plan's input ordering requirements. Vector elements represent
/// alternative ordering requirements in the order of preference. The list of
/// alternatives can be either hard or soft, depending on whether the operator
/// can work without an input ordering.
///
/// # Invariants
///
/// It is non-degenerate: it holds at least one non-empty ordering. The
/// absence of an input ordering requirement is represented by a `None` value
/// in `ExecutionPlan` APIs, which return an `Option<OrderingRequirements>`.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderingRequirements {
    /// The operator is not able to work without one of these requirements.
    Hard(Vec<Ordering>),
    /// The operator can benefit from these input orderings when available,
    /// but can still work in the absence of any input ordering.
    Soft(Vec<Ordering>),
}

impl OrderingRequirements {
    /// Creates a new instance from the given alternatives. Empty alternatives
    /// are skipped; if none is left, returns `None`.
    pub fn new_alternatives(
        alternatives: impl IntoIterator<Item = Ordering>,
        soft: bool,
    ) -> Option<Self> {
        let alternatives = alternatives
            .into_iter()
            .filter(|o| !o.is_empty())
            .collect::<Vec<_>>();
        (!alternatives.is_empty()).then(|| {
            if soft {
                Self::Soft(alternatives)
            } else {
                Self::Hard(alternatives)
            }
        })
    }

    /// Creates a new instance with a single hard requirement, which must not
    /// be empty.
    pub fn new(requirement: Ordering) -> Self {
        debug_assert!(!requirement.is_empty());
        Self::Hard(vec![requirement])
    }

    /// Creates a new instance with a single soft requirement, which must not
    /// be empty.
    pub fn new_soft(requirement: Ordering) -> Self {
        debug_assert!(!requirement.is_empty());
        Self::Soft(vec![requirement])
    }

    /// Adds an alternative requirement to the list of alternatives.
    pub fn add_alternative(&mut self, requirement: Ordering) {
        if requirement.is_empty() {
            return;
        }
        match self {
            Self::Hard(alts) | Self::Soft(alts) => alts.push(requirement),
        }
    }

    /// Returns the first (i.e. most preferred) requirement among the
    /// alternatives.
    pub fn into_single(self) -> Ordering {
        match self {
            Self::Hard(mut alts) | Self::Soft(mut alts) => alts.swap_remove(0),
        }
    }

    /// Returns a reference to the first (i.e. most preferred) requirement
    /// among the alternatives.
    pub fn first(&self) -> &Ordering {
        match self {
            Self::Hard(alts) | Self::Soft(alts) => &alts[0],
        }
    }

    /// Returns all alternatives and whether they are soft.
    pub fn into_alternatives(self) -> (Vec<Ordering>, bool) {
        match self {
            Self::Hard(alts) => (alts, false),
            Self::Soft(alts) => (alts, true),
        }
    }
}

impl From<Ordering> for OrderingRequirements {
    fn from(requirement: Ordering) -> Self {
        Self::new(requirement)
    }
}

impl From<LexOrdering> for OrderingRequirements {
    fn from(ordering: LexOrdering) -> Self {
        Self::new(ordering.into())
    }
}

impl Deref for OrderingRequirements {
    type Target = [Ordering];

    fn deref(&self) -> &Self::Target {
        match &self {
            Self::Hard(alts) | Self::Soft(alts) => alts.as_slice(),
        }
    }
}

impl DerefMut for OrderingRequirements {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Hard(alts) | Self::Soft(alts) => alts.as_mut_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_reversed_sort_options() {
        // Test basic reversal: ASC NULLS LAST ↔ DESC NULLS FIRST
        let asc_nulls_last = SortOptions {
            descending: false,
            nulls_first: false,
        };
        let desc_nulls_first = SortOptions {
            descending: true,
            nulls_first: true,
        };
        assert!(is_reversed_sort_options(&asc_nulls_last, &desc_nulls_first));
        assert!(is_reversed_sort_options(&desc_nulls_first, &asc_nulls_last));

        // Test another reversal: ASC NULLS FIRST ↔ DESC NULLS LAST
        let asc_nulls_first = SortOptions {
            descending: false,
            nulls_first: true,
        };
        let desc_nulls_last = SortOptions {
            descending: true,
            nulls_first: false,
        };
        assert!(is_reversed_sort_options(&asc_nulls_first, &desc_nulls_last));
        assert!(is_reversed_sort_options(&desc_nulls_last, &asc_nulls_first));

        // Test non-reversal: same options
        assert!(!is_reversed_sort_options(&asc_nulls_last, &asc_nulls_last));
        assert!(!is_reversed_sort_options(
            &desc_nulls_first,
            &desc_nulls_first
        ));

        // Test non-reversal: only descending differs
        assert!(!is_reversed_sort_options(&asc_nulls_last, &desc_nulls_last));
        assert!(!is_reversed_sort_options(&desc_nulls_last, &asc_nulls_last));

        // Test non-reversal: only nulls_first differs
        assert!(!is_reversed_sort_options(&asc_nulls_last, &asc_nulls_first));
        assert!(!is_reversed_sort_options(&asc_nulls_first, &asc_nulls_last));
    }
}

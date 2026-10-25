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

//! [`DecorrelateGeneral`] turns correlated subqueries into joins by pushing the
//! outer values down into the subquery.
//!
//! A correlated subquery reads columns of the outer row, so SQL defines it as
//! "run once per outer row". For one subquery, the outer values it reads are
//! its *domain*. The rule pushes the domain down the subquery plan, and every
//! operator on the way also groups, partitions or joins by the domain columns.
//! The subquery then returns its rows for every outer value at once, tagged
//! with that value, and is joined back to the outer plan on the domain columns.
//!
//! Operators stay where they are. Nothing moves past an aggregate, a limit or a
//! join, so no operator needs a "can this filter move here" check, and the
//! empty-input value of an aggregate (the COUNT bug) is handled in one place.
//! This follows T. Neumann, "Improving Unnesting of Complex Queries" (BTW 2025).
//!
//! Computing the domain means computing the outer plan a second time, so three
//! shortcuts avoid it for the common shapes:
//! - A conjunct `inner_expr = outer_col` binds `outer_col` to `inner_expr`:
//!   below it the domain is not needed, and `inner_expr` serves as the domain
//!   column.
//! - An `EXISTS`, `IN` or lateral subquery whose correlated conjuncts sit below
//!   filters, projections and aliases only becomes one join on those conjuncts.
//! - A scalar or lateral subquery rooted at an aggregate without `GROUP BY`
//!   becomes a left join on that aggregate grouped by the domain columns; the
//!   operators above the aggregate are evaluated after the join.

use std::collections::{HashMap, HashSet};
use std::mem::size_of_val;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{DataType, Field, FieldRef, UInt64Type};

use crate::optimizer::ApplyOrder;
use crate::simplify_expressions::ExprSimplifier;
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::alias::AliasGenerator;
use datafusion_common::tree_node::{
    Transformed, TransformedResult, TreeNode, TreeNodeRecursion,
};
use datafusion_common::{
    Column, DFSchemaRef, DataFusionError, NullEquality, Result, ScalarValue,
    TableReference, exec_err, internal_err, not_impl_err,
};
use datafusion_expr::expr::{
    Exists, GroupingSet, InSubquery, WindowFunction, in_subquery_tuple_values,
};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::logical_plan::{
    Aggregate, AsOfJoin, Distinct, FetchType, Filter, Join, JoinConstraint, JoinType,
    Limit, MaterializedCte, MaterializedCtePlan, Projection, SkipType, Sort, Subquery,
    SubqueryAlias, Union, Window,
};
use datafusion_expr::registry::FunctionRegistry;
use datafusion_expr::simplify::SimplifyContext;
use datafusion_expr::utils::{can_hash, conjunction, split_conjunction_owned};
use datafusion_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, BinaryExpr, Expr, ExprFunctionExt,
    ExprSchemable, LogicalPlan, LogicalPlanBuilder, Operator, Signature, SortExpr,
    Volatility, WindowFunctionDefinition, lit, when,
};

/// Turns correlated subqueries into joins. See the [module docs](self).
#[derive(Default, Debug)]
pub struct DecorrelateGeneral {}

impl DecorrelateGeneral {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self::default()
    }
}

impl OptimizerRule for DecorrelateGeneral {
    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let all_scalars = !config
            .options()
            .optimizer
            .enable_physical_uncorrelated_scalar_subquery;
        if !plan_has_subquery_to_rewrite(&plan, all_scalars) {
            return Ok(Transformed::no(plan));
        }
        let all_scalars = !config
            .options()
            .optimizer
            .enable_physical_uncorrelated_scalar_subquery;
        let mut rewriter = Rewriter {
            alias_gen: Arc::clone(config.alias_generator()),
            all_scalars,
            registry: config.function_registry(),
            changed: false,
            names: 0,
        };
        let plan = rewriter.rewrite_plan(plan)?;
        Ok(Transformed::new_transformed(plan, rewriter.changed))
    }

    fn name(&self) -> &str {
        "decorrelate_general"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        None
    }
}

/// The error a shape the rule does not handle yet returns. The node is then
/// left as it was.
fn unsupported<T>(what: impl std::fmt::Display) -> Result<T> {
    not_impl_err!("decorrelate_general: {what}")
}

fn is_unsupported(e: &DataFusionError) -> bool {
    matches!(e.find_root(), DataFusionError::NotImplemented(msg) if msg.starts_with("decorrelate_general"))
}

fn plan_has_subquery_to_rewrite(plan: &LogicalPlan, all_scalars: bool) -> bool {
    matches!(
        plan.apply_with_subqueries(|node| {
            Ok(
                if node_has_subquery_to_rewrite(node, all_scalars)
                    || lateral_subquery(node).is_some()
                    || matches!(node, LogicalPlan::Subquery(_))
                {
                    TreeNodeRecursion::Stop
                } else {
                    TreeNodeRecursion::Continue
                },
            )
        }),
        Ok(TreeNodeRecursion::Stop)
    )
}

fn node_has_subquery_to_rewrite(node: &LogicalPlan, all_scalars: bool) -> bool {
    match node {
        LogicalPlan::Filter(_)
        | LogicalPlan::Projection(_)
        | LogicalPlan::Aggregate(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Window(_) => {
            let mut found = false;
            let _ = node.apply_expressions(|e| {
                found |= expr_has_subquery_to_rewrite(e, all_scalars);
                Ok(TreeNodeRecursion::Continue)
            });
            found
        }
        LogicalPlan::Join(join) => join
            .filter
            .as_ref()
            .is_some_and(|f| expr_has_subquery_to_rewrite(f, all_scalars)),
        _ => false,
    }
}

fn expr_has_subquery_to_rewrite(expr: &Expr, all_scalars: bool) -> bool {
    expr.exists(|e| {
        Ok(match e {
            Expr::ScalarSubquery(sq) => all_scalars || is_correlated(sq),
            Expr::Exists(_) | Expr::InSubquery(_) => true,
            _ => false,
        })
    })
    .expect("infallible")
}

fn is_correlated(sq: &Subquery) -> bool {
    !sq.outer_ref_columns.is_empty() || plan_has_outer_reference(&sq.subquery)
}

/// Whether `plan` reads a column from outside of itself.
fn plan_has_outer_reference(plan: &LogicalPlan) -> bool {
    let mut cols = vec![];
    collect_outer_columns(plan, &[], &mut cols);
    !cols.is_empty()
}

/// The right side of a lateral join reads the columns of its left side.
fn is_lateral_right(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Subquery(_) => true,
        LogicalPlan::SubqueryAlias(a) => {
            matches!(a.input.as_ref(), LogicalPlan::Subquery(_))
        }
        _ => false,
    }
}

/// The subquery of a lateral join: `Join(left, Subquery)` or
/// `Join(left, SubqueryAlias(Subquery))`, with outer references in it.
fn lateral_subquery(node: &LogicalPlan) -> Option<(&Subquery, Option<&TableReference>)> {
    let LogicalPlan::Join(join) = node else {
        return None;
    };
    let (sq, alias) = match join.right.as_ref() {
        LogicalPlan::Subquery(sq) => (sq, None),
        LogicalPlan::SubqueryAlias(SubqueryAlias { input, alias, .. }) => {
            match input.as_ref() {
                LogicalPlan::Subquery(sq) => (sq, Some(alias)),
                _ => return None,
            }
        }
        _ => return None,
    };
    plan_has_outer_reference(&sq.subquery).then_some((sq, alias))
}

/// `subquery` filtered by the ON condition of its LEFT lateral `join`: the
/// columns of the left side become outer references, those of the right side
/// the subquery's own columns.
fn filter_lateral_subquery(
    subquery: Subquery,
    filter: Expr,
    join: &Join,
) -> Result<Subquery> {
    let left = join.left.schema();
    let left_len = left.fields().len();
    let inner_schema = Arc::clone(subquery.subquery.schema());
    let mut outer_refs = subquery.outer_ref_columns.clone();
    let filter = filter
        .transform(|e| match e {
            Expr::Column(c) => {
                if let Ok((_, field)) = left.qualified_field_from_column(&c) {
                    let e = Expr::OuterReferenceColumn(Arc::clone(field), c);
                    if !outer_refs.contains(&e) {
                        outer_refs.push(e.clone());
                    }
                    return Ok(Transformed::yes(e));
                }
                let i = join.schema.index_of_column(&c)?;
                Ok(Transformed::yes(Expr::Column(Column::from(
                    inner_schema.qualified_field(i - left_len),
                ))))
            }
            other => Ok(Transformed::no(other)),
        })
        .data()?;
    let plan = Filter::try_new(filter, subquery.subquery)?;
    Ok(Subquery {
        subquery: Arc::new(LogicalPlan::Filter(plan)),
        outer_ref_columns: outer_refs,
        spans: subquery.spans,
    })
}

fn lateral_subquery_of(join: &Join) -> Option<&Subquery> {
    let sq = match join.right.as_ref() {
        LogicalPlan::Subquery(sq) => sq,
        LogicalPlan::SubqueryAlias(SubqueryAlias { input, .. }) => match input.as_ref() {
            LogicalPlan::Subquery(sq) => sq,
            _ => return None,
        },
        _ => return None,
    };
    plan_has_outer_reference(&sq.subquery).then_some(sq)
}

/// The outer columns one subquery reads, and the outer plan they come from.
struct Correlation<'a> {
    /// Each outer column the subquery reads, in the order its domain stores it.
    outer_cols: Vec<Column>,
    /// The outer plan, which the domain is computed from.
    outer: &'a LogicalPlan,
    /// Whether some part of the rewritten subquery reads the domain. If not,
    /// every domain column holds an inner value that the subquery compared
    /// with `=` to the outer value, so it is never NULL.
    used_domain: bool,
}

/// A rewritten subquery plan and the columns of it that hold the outer values.
struct Attached {
    plan: LogicalPlan,
    /// `cols[i]` holds the value of `Correlation::outer_cols[i]` for each row.
    cols: Vec<Column>,
    /// `bound[i]` if `cols[i]` holds the inner value of an `inner = outer`
    /// conjunct rather than a domain value: a NULL there matches no outer
    /// value, so `cols[i]` is compared with `=`, not `IS NOT DISTINCT FROM`.
    bound: Vec<bool>,
}

struct Rewriter<'a> {
    alias_gen: Arc<AliasGenerator>,
    /// Whether uncorrelated scalar subqueries are rewritten too, rather than
    /// run by the physical plan.
    all_scalars: bool,
    registry: Option<&'a dyn FunctionRegistry>,
    changed: bool,
    /// Counter for the names of the columns the rule adds.
    names: usize,
}

impl Rewriter<'_> {
    fn fresh_name(&mut self, prefix: &str) -> String {
        self.names += 1;
        format!("{prefix}_{}", self.names)
    }

    /// Renames the columns of `plan` that would be ambiguous next to an
    /// unqualified field of the same name in `outer` once `plan` is aliased.
    /// Keeps the column order.
    fn disambiguate(
        &mut self,
        plan: LogicalPlan,
        outer: &DFSchemaRef,
    ) -> Result<LogicalPlan> {
        let unqualified: HashSet<&String> = outer
            .iter()
            .filter(|(q, _)| q.is_none())
            .map(|(_, f)| f.name())
            .collect();
        if !plan
            .schema()
            .fields()
            .iter()
            .any(|f| unqualified.contains(f.name()))
        {
            return Ok(plan);
        }
        let exprs: Vec<Expr> = plan
            .schema()
            .iter()
            .map(|(q, f)| {
                let e = Expr::Column(Column::new(q.cloned(), f.name()));
                if unqualified.contains(f.name()) {
                    e.alias(self.fresh_name("__sq_col"))
                } else {
                    e
                }
            })
            .collect();
        LogicalPlanBuilder::from(plan).project(exprs)?.build()
    }

    /// Rewrites the subqueries of every node, inputs first.
    fn rewrite_plan(&mut self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let plan = plan
            .map_children(|c| self.rewrite_plan(c).map(Transformed::yes))?
            .data;
        // A subquery node marks a plan that reads outer columns; once it reads
        // none, for example after its own lateral joins are rewritten, it is a
        // plain plan.
        if let LogicalPlan::Subquery(sq) = &plan
            && !plan_has_outer_reference(&sq.subquery)
        {
            let LogicalPlan::Subquery(sq) = plan else {
                unreachable!()
            };
            self.changed = true;
            return Ok(Arc::unwrap_or_clone(sq.subquery));
        }
        let plan = if node_has_subquery_to_rewrite(&plan, self.all_scalars)
            || lateral_subquery(&plan).is_some()
        {
            match self.rewrite_node(plan.clone()) {
                Ok(new_plan) => {
                    self.changed = true;
                    new_plan
                }
                Err(e) if is_unsupported(&e) => {
                    log::debug!("{e}");
                    plan
                }
                Err(e) => return Err(e),
            }
        } else {
            plan
        };
        // The subqueries that are left (for example uncorrelated scalar
        // subqueries) can have correlated subqueries of their own.
        plan.map_subqueries(|sq| {
            let LogicalPlan::Subquery(sq) = sq else {
                return internal_err!("expected a subquery");
            };
            let inner = self.rewrite_plan(Arc::unwrap_or_clone(sq.subquery))?;
            Ok(Transformed::yes(LogicalPlan::Subquery(Subquery {
                subquery: Arc::new(inner),
                outer_ref_columns: sq.outer_ref_columns,
                spans: sq.spans,
            })))
        })
        .data()
    }

    fn rewrite_node(&mut self, plan: LogicalPlan) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::Filter(filter) => self.rewrite_filter(filter),
            LogicalPlan::Projection(projection) => self.rewrite_projection(projection),
            LogicalPlan::Aggregate(agg) => self.rewrite_aggregate(agg),
            LogicalPlan::Sort(sort) => self.rewrite_sort(sort),
            LogicalPlan::Window(window) => self.rewrite_window(window),
            LogicalPlan::Join(join) if lateral_subquery_of(&join).is_some() => {
                self.rewrite_lateral(join)
            }
            LogicalPlan::Join(join) => self.rewrite_join_filter(join),
            other => internal_err!("unexpected node {}", other.display()),
        }
    }

    /// `EXISTS`/`IN` conjuncts become semi and anti joins; the other
    /// subqueries become columns of joins below the filter.
    fn rewrite_filter(&mut self, filter: Filter) -> Result<LogicalPlan> {
        let output: Vec<Expr> = filter
            .input
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        let mut outer = Arc::unwrap_or_clone(filter.input);
        let mut kept = vec![];
        for conjunct in split_conjunction_owned(filter.predicate) {
            let conjunct = match conjunct {
                Expr::Not(inner) => match *inner {
                    Expr::Exists(Exists { subquery, negated }) => {
                        Expr::Exists(Exists::new(subquery, !negated))
                    }
                    Expr::InSubquery(InSubquery {
                        expr,
                        subquery,
                        negated,
                    }) => Expr::InSubquery(InSubquery::new(expr, subquery, !negated)),
                    other => Expr::Not(Box::new(other)),
                },
                other => other,
            };
            match conjunct {
                Expr::Exists(Exists { subquery, negated }) => {
                    let join_type = if negated {
                        JoinType::LeftAnti
                    } else {
                        JoinType::LeftSemi
                    };
                    outer = self
                        .predicate_join(outer, &subquery, None, join_type, false)?
                        .0;
                }
                Expr::InSubquery(InSubquery {
                    expr,
                    subquery,
                    negated,
                }) => {
                    // `NOT IN` is FALSE or NULL as soon as the subquery holds a
                    // NULL, which a null-aware anti join implements.
                    let join_type = if negated {
                        JoinType::LeftAnti
                    } else {
                        JoinType::LeftSemi
                    };
                    let expr = self.replace_value_subqueries(&mut outer, *expr)?;
                    outer = self
                        .predicate_join(outer, &subquery, Some(expr), join_type, negated)?
                        .0;
                }
                other => kept.push(other),
            }
        }
        let mut builder = match conjunction(kept) {
            Some(predicate) => {
                let predicate =
                    self.replace_predicate_subqueries(&mut outer, predicate)?;
                LogicalPlanBuilder::from(outer).filter(predicate)?
            }
            None => LogicalPlanBuilder::from(outer),
        };
        builder = builder.project(output)?;
        builder.build()
    }

    fn rewrite_projection(&mut self, projection: Projection) -> Result<LogicalPlan> {
        let mut outer = Arc::unwrap_or_clone(projection.input);
        let mut exprs = Vec::with_capacity(projection.expr.len());
        for (i, expr) in projection.expr.into_iter().enumerate() {
            let new_expr = self.replace_value_subqueries(&mut outer, expr)?;
            exprs.push(keep_name(new_expr, &projection.schema, i));
        }
        Projection::try_new(exprs, Arc::new(outer)).map(LogicalPlan::Projection)
    }

    /// Subqueries in grouping keys, aggregate arguments and `FILTER` clauses
    /// read the aggregate's input, so they are joined below the aggregate.
    fn rewrite_aggregate(&mut self, agg: Aggregate) -> Result<LogicalPlan> {
        if agg
            .group_expr
            .iter()
            .any(|e| matches!(e, Expr::GroupingSet(_)) && contains_subquery(e))
        {
            return unsupported("subquery in a grouping set");
        }
        let mut outer = Arc::unwrap_or_clone(agg.input);
        let n_group = agg.group_expr.len();
        let mut group_expr = Vec::with_capacity(n_group);
        for (i, e) in agg.group_expr.into_iter().enumerate() {
            let new_expr = self.replace_value_subqueries(&mut outer, e)?;
            group_expr.push(keep_name(new_expr, &agg.schema, i));
        }
        let mut aggr_expr = Vec::with_capacity(agg.aggr_expr.len());
        for (i, e) in agg.aggr_expr.into_iter().enumerate() {
            let new_expr = self.replace_value_subqueries(&mut outer, e)?;
            aggr_expr.push(keep_name(new_expr, &agg.schema, n_group + i));
        }
        Aggregate::try_new(Arc::new(outer), group_expr, aggr_expr)
            .map(LogicalPlan::Aggregate)
    }

    fn rewrite_sort(&mut self, sort: Sort) -> Result<LogicalPlan> {
        let output: Vec<Expr> = sort
            .input
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        let mut outer = Arc::unwrap_or_clone(sort.input);
        let mut exprs = Vec::with_capacity(sort.expr.len());
        for s in sort.expr {
            let expr = self.replace_value_subqueries(&mut outer, s.expr.clone())?;
            exprs.push(SortExpr { expr, ..s });
        }
        let sorted = LogicalPlan::Sort(Sort {
            expr: exprs,
            input: Arc::new(outer),
            fetch: sort.fetch,
        });
        LogicalPlanBuilder::from(sorted).project(output)?.build()
    }

    fn rewrite_window(&mut self, window: Window) -> Result<LogicalPlan> {
        let n_input = window.input.schema().fields().len();
        let mut output: Vec<Expr> = window
            .input
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        let mut outer = Arc::unwrap_or_clone(window.input);
        let mut exprs = Vec::with_capacity(window.window_expr.len());
        for (i, e) in window.window_expr.into_iter().enumerate() {
            let new_expr = self.replace_value_subqueries(&mut outer, e)?;
            exprs.push(keep_name(new_expr, &window.schema, n_input + i));
            output.push(Expr::Column(Column::from(
                window.schema.qualified_field(n_input + i),
            )));
        }
        let plan = Window::try_new(exprs, Arc::new(outer))?;
        LogicalPlanBuilder::from(LogicalPlan::Window(plan))
            .project(output)?
            .build()
    }

    /// An inner join's condition is a filter over the join, so its subquery
    /// conjuncts move to a filter above it. For another join, a subquery that
    /// reads one side only is computed as a column of that side.
    fn rewrite_join_filter(&mut self, join: Join) -> Result<LogicalPlan> {
        if join.join_type != JoinType::Inner {
            return self.rewrite_outer_join_filter(join);
        }
        let Some(filter) = join.filter else {
            return internal_err!("expected a join filter");
        };
        let (with_subqueries, rest): (Vec<_>, Vec<_>) = split_conjunction_owned(filter)
            .into_iter()
            .partition(contains_subquery);
        let join = Join::try_new(
            join.left,
            join.right,
            join.on,
            conjunction(rest),
            join.join_type,
            join.join_constraint,
            join.null_equality,
            join.null_aware,
        )?
        .with_null_aware_value_keys(join.null_aware_value_keys);
        let predicate = conjunction(with_subqueries).expect("a subquery conjunct");
        let filter = Filter::try_new(predicate, Arc::new(LogicalPlan::Join(join)))?;
        self.rewrite_filter(filter)
    }

    /// Like [`Self::replace_value_subqueries`] for a filter predicate. Below
    /// only `AND` and `OR` a NULL drops the row like FALSE does, so an `IN`
    /// there needs no three-valued mark.
    fn replace_predicate_subqueries(
        &mut self,
        outer: &mut LogicalPlan,
        expr: Expr,
    ) -> Result<Expr> {
        match expr {
            Expr::BinaryExpr(BinaryExpr {
                left,
                op: op @ (Operator::And | Operator::Or),
                right,
            }) => {
                let left = self.replace_predicate_subqueries(outer, *left)?;
                let right = self.replace_predicate_subqueries(outer, *right)?;
                Ok(Expr::BinaryExpr(BinaryExpr::new(
                    Box::new(left),
                    op,
                    Box::new(right),
                )))
            }
            Expr::InSubquery(InSubquery {
                expr,
                subquery,
                negated: false,
            }) => {
                // The value is compared, not used as a truth value, so its own
                // subqueries need three-valued marks.
                let expr = self.replace_value_subqueries(outer, *expr)?;
                let (joined, mark) = self.predicate_join(
                    std::mem::take(outer),
                    &subquery,
                    Some(expr),
                    JoinType::LeftMark,
                    false,
                )?;
                *outer = joined;
                Ok(mark.expect("a mark join returns its mark"))
            }
            other => self.replace_value_subqueries(outer, other),
        }
    }

    fn rewrite_outer_join_filter(&mut self, join: Join) -> Result<LogicalPlan> {
        let Some(filter) = join.filter.clone() else {
            return internal_err!("expected a join filter");
        };
        let output: Vec<Expr> = join
            .schema
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        let mut left = Arc::unwrap_or_clone(Arc::clone(&join.left));
        let mut right = Arc::unwrap_or_clone(Arc::clone(&join.right));
        let reads = |plan: &LogicalPlan, side: &LogicalPlan| {
            let mut cols = vec![];
            collect_outer_columns(plan, &[], &mut cols);
            cols.iter().all(|c| side.schema().has_column(c))
        };
        let mut sides_ok = true;
        filter.apply(|e| {
            if let Expr::ScalarSubquery(sq)
            | Expr::Exists(Exists { subquery: sq, .. })
            | Expr::InSubquery(InSubquery { subquery: sq, .. }) = e
            {
                sides_ok &= reads(&sq.subquery, &left) || reads(&sq.subquery, &right);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        if !sides_ok {
            return unsupported(
                "subquery in an outer join condition that reads both sides",
            );
        }
        // Each subquery joins the side whose columns it reads.
        let filter = filter
            .transform_down(|e| match e {
                Expr::ScalarSubquery(_) | Expr::Exists(_) | Expr::InSubquery(_) => {
                    let sq = match &e {
                        Expr::ScalarSubquery(sq)
                        | Expr::Exists(Exists { subquery: sq, .. })
                        | Expr::InSubquery(InSubquery { subquery: sq, .. }) => sq.clone(),
                        _ => unreachable!(),
                    };
                    let side = if reads(&sq.subquery, &left) {
                        &mut left
                    } else {
                        &mut right
                    };
                    let value = self.replace_value_subqueries(side, e)?;
                    Ok(Transformed::new(value, true, TreeNodeRecursion::Jump))
                }
                other => Ok(Transformed::no(other)),
            })
            .data()?;
        let plan = LogicalPlan::Join(
            Join::try_new(
                Arc::new(left),
                Arc::new(right),
                join.on,
                Some(filter),
                join.join_type,
                join.join_constraint,
                join.null_equality,
                join.null_aware,
            )?
            .with_null_aware_value_keys(join.null_aware_value_keys),
        );
        LogicalPlanBuilder::from(plan).project(output)?.build()
    }

    /// Replaces each correlated scalar subquery and `EXISTS` in `expr` with a
    /// column of a join added to `outer`.
    fn replace_value_subqueries(
        &mut self,
        outer: &mut LogicalPlan,
        expr: Expr,
    ) -> Result<Expr> {
        expr.transform_down(|e| match e {
            Expr::ScalarSubquery(sq) if self.all_scalars || is_correlated(&sq) => {
                let (joined, value) = self.scalar_join(std::mem::take(outer), &sq)?;
                *outer = joined;
                Ok(Transformed::new(value, true, TreeNodeRecursion::Jump))
            }
            Expr::Exists(Exists { subquery, negated }) => {
                let (joined, mark) = self.predicate_join(
                    std::mem::take(outer),
                    &subquery,
                    None,
                    JoinType::LeftMark,
                    false,
                )?;
                *outer = joined;
                let mark = mark.expect("a mark join returns its mark");
                let value = if negated { !mark } else { mark };
                Ok(Transformed::new(value, true, TreeNodeRecursion::Jump))
            }
            // A null-aware mark is NULL when the answer depends on a NULL, as
            // three-valued `IN` requires.
            Expr::InSubquery(InSubquery {
                expr,
                subquery,
                negated,
            }) => {
                let expr = self.replace_value_subqueries(outer, *expr)?;
                let (joined, mark) = self.predicate_join(
                    std::mem::take(outer),
                    &subquery,
                    Some(expr),
                    JoinType::LeftMark,
                    true,
                )?;
                *outer = joined;
                let mark = mark.expect("a mark join returns its mark");
                let value = if negated { !mark } else { mark };
                Ok(Transformed::new(value, true, TreeNodeRecursion::Jump))
            }
            other => Ok(Transformed::no(other)),
        })
        .data()
    }

    fn rewrite_lateral(&mut self, join: Join) -> Result<LogicalPlan> {
        let node = LogicalPlan::Join(join);
        let Some((sq, _)) = lateral_subquery(&node) else {
            return internal_err!("expected a lateral join");
        };
        let sq = sq.clone();
        let LogicalPlan::Join(join) = node else {
            unreachable!()
        };
        if !join.on.is_empty() {
            return unsupported("lateral join with equi-join keys");
        }
        let join_type = join.join_type;
        if !matches!(join_type, JoinType::Inner | JoinType::Left) {
            return unsupported(format!("{join_type} lateral join"));
        }
        // A LEFT lateral join keeps the subquery rows that meet its ON
        // condition, which is a correlated filter on top of the subquery.
        let mut sq = sq;
        let mut join_filter = join.filter.clone();
        if join_type == JoinType::Left
            && let Some(filter) = join_filter.take()
            && !matches!(filter, Expr::Literal(ScalarValue::Boolean(Some(true)), _))
        {
            sq = filter_lateral_subquery(sq, filter, &join)?;
        }
        let outer = Arc::unwrap_or_clone(Arc::clone(&join.left));
        let alias = match join.right.as_ref() {
            LogicalPlan::SubqueryAlias(a) => Some(a.alias.clone()),
            _ => None,
        };
        let (joined, values, condition) =
            self.lateral_join(outer, &sq, join_type, alias)?;

        let mut builder = LogicalPlanBuilder::from(joined);
        if join_type == JoinType::Inner
            && let Some(condition) = &condition
        {
            builder = builder.filter(condition.clone())?;
        }
        // The output of the original join: the left columns, then the lateral
        // subquery's columns under the names the join gave them.
        let left_len = join.left.schema().fields().len();
        let mut exprs: Vec<Expr> = join
            .left
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        for (i, value) in values.into_iter().enumerate() {
            let (qualifier, field) = join.schema.qualified_field(left_len + i);
            let value = match (&condition, join_type) {
                (Some(condition), JoinType::Left) => {
                    when(condition.clone(), value).end()?
                }
                _ => value,
            };
            let same = Expr::Column(Column::new(qualifier.cloned(), field.name()));
            exprs.push(if value == same {
                value
            } else {
                value.alias_qualified(qualifier.cloned(), field.name())
            });
        }
        builder = builder.project(exprs)?;
        if let Some(filter) = join_filter
            && join_type == JoinType::Inner
        {
            builder = builder.filter(filter)?;
        }
        builder.build()
    }

    /// Joins `outer` with an `EXISTS`, `IN` or set subquery. `join_type` is
    /// `LeftSemi`, `LeftAnti` or `LeftMark`; for `LeftMark` the second value
    /// is the mark column.
    fn predicate_join(
        &mut self,
        outer: LogicalPlan,
        sq: &Subquery,
        in_expr: Option<Expr>,
        join_type: JoinType,
        null_aware: bool,
    ) -> Result<(LogicalPlan, Option<Expr>)> {
        let outer_cols = free_outer_columns(&sq.subquery, &outer)?;
        // Whether any row exists does not depend on the subquery's output.
        let subquery_plan = if in_expr.is_none() {
            strip_for_exists(sq.subquery.as_ref().clone())
        } else {
            strip_for_in(sq.subquery.as_ref().clone())
        };
        let mut corr = Correlation {
            outer_cols,
            outer: &outer,
            used_domain: false,
        };

        let (inner, conjuncts) = match self.pull_up(subquery_plan.clone(), &corr)? {
            Some((inner, conjuncts)) => {
                let inner = self.rewrite_plan(inner)?;
                let inner_schema = Arc::clone(inner.schema());
                let alias = self.alias_gen.next("__correlated_sq");
                let inner = self.disambiguate(inner, outer.schema())?;
                let aliased = alias_plan(inner, alias.as_str())?;
                let conjuncts = conjuncts
                    .into_iter()
                    .map(|c| {
                        requalify(c, &inner_schema, aliased.schema())
                            .and_then(|c| strip_outer(c, &corr))
                    })
                    .collect::<Result<Vec<_>>>()?;
                (aliased, conjuncts)
            }
            None => {
                let (aliased, cols, bound) = self.attach_subquery(
                    &subquery_plan,
                    &mut corr,
                    "__correlated_sq",
                    None,
                )?;
                let conjuncts = domain_join_keys(&corr, &cols, &bound);
                (aliased, conjuncts)
            }
        };
        let mut outer = outer;
        // The `IN` equalities come first: a null-aware join reads its leading
        // equalities as the values to compare and the rest as correlation keys.
        let mut keys = vec![];
        let mut null_aware = null_aware;
        if let Some(in_expr) = in_expr {
            let values: Vec<Expr> =
                match in_subquery_tuple_values(&in_expr, &sq.subquery)? {
                    Some(values) => values.into_owned(),
                    None => vec![in_expr],
                };
            let inner_cols: Vec<Expr> = (0..values.len())
                .map(|i| Expr::Column(Column::from(inner.schema().qualified_field(i))))
                .collect();
            // Three-valued logic only matters when a NULL can occur.
            null_aware &= values
                .iter()
                .map(|v| v.nullable(outer.schema().as_ref()))
                .chain(
                    inner_cols
                        .iter()
                        .map(|c| c.nullable(inner.schema().as_ref())),
                )
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .any(|n| n);
            if null_aware {
                let hashable = values
                    .iter()
                    .map(|v| v.get_type(outer.schema().as_ref()))
                    .chain(
                        inner_cols
                            .iter()
                            .map(|c| c.get_type(inner.schema().as_ref())),
                    )
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .all(can_hash);
                // A null-aware hash join compares hashable values, and its
                // correlation keys use `=`. Otherwise compute the UNKNOWN
                // case with more joins.
                let null_safe_key = conjuncts.iter().any(|c| {
                    matches!(c, Expr::BinaryExpr(b) if b.op == Operator::IsNotDistinctFrom)
                });
                if !hashable || null_safe_key {
                    let [value] = values.as_slice() else {
                        return unsupported("three-valued multi-column IN");
                    };
                    let [col] = inner_cols.as_slice() else {
                        return internal_err!("one value, one column");
                    };
                    return self.three_valued_in(
                        outer,
                        inner,
                        value.clone(),
                        col.clone(),
                        conjuncts,
                        join_type,
                    );
                }
            }
            // When each `IN` equality is also a correlation equality, a NULL on
            // either side leaves the scope empty, so the answer is FALSE and
            // never UNKNOWN.
            let pairs: Vec<Expr> = values
                .iter()
                .cloned()
                .zip(inner_cols.iter().cloned())
                .map(|(v, c)| v.eq(c))
                .collect();
            if pairs
                .iter()
                .all(|p| conjuncts.iter().any(|c| same_equality(c, p)))
            {
                null_aware = false;
            }
            // A null-aware join key must read a column of its side, so a
            // constant value becomes a column of the outer plan.
            let mut values = values;
            let alias = match &inner {
                LogicalPlan::SubqueryAlias(a) => a.alias.to_string(),
                _ => "__correlated_sq".to_string(),
            };
            let n_values = values.len();
            if null_aware && values.iter().any(|v| !v.any_column_refs()) {
                let mut exprs: Vec<Expr> = outer
                    .schema()
                    .columns()
                    .into_iter()
                    .map(Expr::Column)
                    .collect();
                for (i, value) in values.iter_mut().enumerate() {
                    if !value.any_column_refs() {
                        let mut name = if n_values == 1 {
                            format!("{alias}_value")
                        } else {
                            format!("{alias}_value_{i}")
                        };
                        while outer.schema().has_column_with_unqualified_name(&name) {
                            name = self.fresh_name(&name);
                        }
                        exprs.push(value.clone().alias(&name));
                        *value = Expr::Column(Column::new_unqualified(name));
                    }
                }
                outer = LogicalPlanBuilder::from(outer).project(exprs)?.build()?;
            }
            for (value, col) in values.into_iter().zip(inner_cols) {
                keys.push(value.eq(col));
            }
        } else {
            null_aware = false;
        }
        let n_value_keys = keys.len();
        for conjunct in conjuncts {
            if !keys.iter().any(|k| same_equality(k, &conjunct)) {
                keys.push(conjunct);
            }
        }
        let join = Join::try_new(
            Arc::new(outer),
            Arc::new(inner),
            vec![],
            conjunction(keys),
            join_type,
            JoinConstraint::On,
            NullEquality::NullEqualsNothing,
            null_aware,
        )?
        .with_null_aware_value_keys(n_value_keys.max(1));
        let plan = LogicalPlan::Join(join);
        if join_type != JoinType::LeftMark {
            return Ok((plan, None));
        }
        // The mark column is qualified with the subquery alias.
        let n = plan.schema().fields().len();
        let mark = Expr::Column(Column::from(plan.schema().qualified_field(n - 1)));
        Ok((plan, Some(mark)))
    }

    /// `value IN (inner)` under three-valued logic with plain mark joins: one
    /// for a match, one for a NULL in the scope of the outer row, and one for
    /// any row in that scope. `keys` select the scope. For `LeftAnti` (a
    /// `NOT IN` filter) returns the filtered plan, for `LeftMark` the value.
    fn three_valued_in(
        &mut self,
        outer: LogicalPlan,
        inner: LogicalPlan,
        value: Expr,
        col: Expr,
        keys: Vec<Expr>,
        join_type: JoinType,
    ) -> Result<(LogicalPlan, Option<Expr>)> {
        let output: Vec<Expr> = outer
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        let inner_schema = Arc::clone(inner.schema());
        let mark_join = |left: LogicalPlan, right: LogicalPlan, on: Vec<Expr>| {
            let join = Join::try_new(
                Arc::new(left),
                Arc::new(right),
                vec![],
                conjunction(on),
                JoinType::LeftMark,
                JoinConstraint::On,
                NullEquality::NullEqualsNothing,
                false,
            )?;
            let plan = LogicalPlan::Join(join);
            let n = plan.schema().fields().len();
            let mark = Expr::Column(Column::from(plan.schema().qualified_field(n - 1)));
            Ok::<_, DataFusionError>((plan, mark))
        };
        let mut matched_on = keys.clone();
        matched_on.push(value.clone().eq(col.clone()));
        let (plan, matched) = mark_join(outer, inner.clone(), matched_on)?;

        let null_alias = self.alias_gen.next("__correlated_sq");
        let nulls = alias_plan(
            LogicalPlanBuilder::from(inner.clone())
                .filter(col.is_null())?
                .build()?,
            null_alias.as_str(),
        )?;
        let null_keys = keys
            .iter()
            .map(|k| requalify(k.clone(), &inner_schema, nulls.schema()))
            .collect::<Result<Vec<_>>>()?;
        let (plan, has_null) = mark_join(plan, nulls, null_keys)?;

        let any_alias = self.alias_gen.next("__correlated_sq");
        let any = alias_plan(inner, any_alias.as_str())?;
        let any_keys = keys
            .iter()
            .map(|k| requalify(k.clone(), &inner_schema, any.schema()))
            .collect::<Result<Vec<_>>>()?;
        let (plan, non_empty) = mark_join(plan, any, any_keys)?;

        let unknown = has_null.or(value.is_null().and(non_empty));
        let result = when(matched, lit(true))
            .when(unknown, lit(ScalarValue::Boolean(None)))
            .otherwise(lit(false))?;
        match join_type {
            JoinType::LeftMark => Ok((plan, Some(result))),
            JoinType::LeftAnti => {
                let plan = LogicalPlanBuilder::from(plan)
                    .filter(Expr::Not(Box::new(result)))?
                    .project(output)?
                    .build()?;
                Ok((plan, None))
            }
            other => internal_err!("three-valued IN for a {other} join"),
        }
    }

    /// Joins `outer` with a correlated scalar subquery; returns the expression
    /// that computes the subquery's value over the join.
    fn scalar_join(
        &mut self,
        outer: LogicalPlan,
        sq: &Subquery,
    ) -> Result<(LogicalPlan, Expr)> {
        let outer_cols = free_outer_columns(&sq.subquery, &outer)?;
        if let Some(chain) = peel_to_groupless_aggregate(&sq.subquery) {
            let (joined, mut values, condition) =
                self.aggregate_join(outer, outer_cols, chain, None, true)?;
            let value = values.swap_remove(0);
            let value = match condition {
                Some(condition) => when(condition, value).end()?,
                None => value,
            };
            return Ok((joined, value));
        }
        let mut corr = Correlation {
            outer_cols,
            outer: &outer,
            used_domain: false,
        };
        // Below its correlated filters the subquery returns at most one row,
        // so a left join on those filters matches at most one row.
        if let Some((inner, conjuncts)) =
            self.pull_up(sq.subquery.as_ref().clone(), &corr)?
            && inner.max_rows().is_some_and(|n| n <= 1)
        {
            let inner = self.rewrite_plan(inner)?;
            let inner_schema = Arc::clone(inner.schema());
            let alias = self.alias_gen.next("__scalar_sq");
            let inner = self.disambiguate(inner, outer.schema())?;
            let aliased = alias_plan(inner, alias.as_str())?;
            let conjuncts = conjuncts
                .into_iter()
                .map(|c| {
                    requalify(c, &inner_schema, aliased.schema())
                        .and_then(|c| strip_outer(c, &corr))
                })
                .collect::<Result<Vec<_>>>()?;
            let value = Expr::Column(Column::from(aliased.schema().qualified_field(0)));
            let join = Join::try_new(
                Arc::new(outer),
                Arc::new(aliased),
                vec![],
                conjunction(conjuncts),
                JoinType::Left,
                JoinConstraint::On,
                NullEquality::NullEqualsNothing,
                false,
            )?;
            return Ok((LogicalPlan::Join(join), value));
        }
        let (mut inner, mut cols, bound) =
            self.attach_subquery(&sq.subquery, &mut corr, "__scalar_sq", None)?;
        let mut value = Expr::Column(Column::from(inner.schema().qualified_field(0)));
        // A scalar subquery that can return several rows for one outer value
        // must fail if it does: aggregate its value per outer value with an
        // aggregate that fails on a second row.
        if !sq.subquery.max_rows().is_some_and(|n| n <= 1)
            && !one_row_per_domain(&inner, &cols)
        {
            let name = self.fresh_name("__value");
            let single = AggregateUDF::from(SingleValue::new())
                .call(vec![value])
                .alias(&name);
            let (unique, positions) = dedup_columns(&cols);
            let keys: Vec<Expr> = unique.into_iter().map(Expr::Column).collect();
            let grouped = LogicalPlanBuilder::from(inner)
                .aggregate(keys, vec![single])?
                .build()?;
            let alias = self.alias_gen.next("__scalar_sq");
            let grouped = self.disambiguate(grouped, outer.schema())?;
            inner = alias_plan(grouped, alias.as_str())?;
            cols = positions
                .into_iter()
                .map(|i| Column::from(inner.schema().qualified_field(i)))
                .collect();
            value = Expr::Column(Column::new(Some(alias.as_str()), &name));
        }
        let keys = domain_join_keys(&corr, &cols, &bound);
        let join = Join::try_new(
            Arc::new(outer),
            Arc::new(inner),
            vec![],
            conjunction(keys),
            JoinType::Left,
            JoinConstraint::On,
            NullEquality::NullEqualsNothing,
            false,
        )?;
        Ok((LogicalPlan::Join(join), value))
    }

    /// Joins `outer` with a lateral subquery. Returns the join, one expression
    /// per subquery column, and a condition the row must meet (for an `INNER`
    /// lateral join a filter, for a `LEFT` one the columns are NULL otherwise).
    fn lateral_join(
        &mut self,
        outer: LogicalPlan,
        sq: &Subquery,
        join_type: JoinType,
        alias: Option<TableReference>,
    ) -> Result<(LogicalPlan, Vec<Expr>, Option<Expr>)> {
        let outer_cols = free_outer_columns(&sq.subquery, &outer)?;
        if let Some(chain) = peel_to_groupless_aggregate(&sq.subquery) {
            return self.aggregate_join(
                outer,
                outer_cols,
                chain,
                alias,
                join_type == JoinType::Left,
            );
        }
        let subquery_plan = sq.subquery.as_ref().clone();
        let mut corr = Correlation {
            outer_cols,
            outer: &outer,
            used_domain: false,
        };
        let (inner, conjuncts) = match self.pull_up(subquery_plan.clone(), &corr)? {
            Some((inner, conjuncts)) => {
                let inner = self.rewrite_plan(inner)?;
                let inner_schema = Arc::clone(inner.schema());
                let alias =
                    alias.unwrap_or_else(|| self.alias_gen.next("__lateral_sq").into());
                let inner = self.disambiguate(inner, outer.schema())?;
                let aliased = alias_plan(inner, alias)?;
                let conjuncts = conjuncts
                    .into_iter()
                    .map(|c| {
                        requalify(c, &inner_schema, aliased.schema())
                            .and_then(|c| strip_outer(c, &corr))
                    })
                    .collect::<Result<Vec<_>>>()?;
                (aliased, conjuncts)
            }
            None => {
                let (aliased, cols, bound) = self.attach_subquery(
                    &subquery_plan,
                    &mut corr,
                    "__lateral_sq",
                    alias,
                )?;
                let keys = domain_join_keys(&corr, &cols, &bound);
                (aliased, keys)
            }
        };
        let n = sq.subquery.schema().fields().len();
        let values = (0..n)
            .map(|i| Expr::Column(Column::from(inner.schema().qualified_field(i))))
            .collect();
        let join = Join::try_new(
            Arc::new(outer),
            Arc::new(inner),
            vec![],
            conjunction(conjuncts),
            join_type,
            JoinConstraint::On,
            NullEquality::NullEqualsNothing,
            false,
        )?;
        Ok((LogicalPlan::Join(join), values, None))
    }

    /// The scalar or lateral subquery `chain` ends at an aggregate without
    /// `GROUP BY`. Groups that aggregate by the domain columns, left joins it
    /// to `outer`, and evaluates the nodes of `chain` above it over the join.
    ///
    /// Returns the join, one expression per subquery column, and the
    /// conjunction of the filters in `chain` (for example a `HAVING`).
    fn aggregate_join(
        &mut self,
        outer: LogicalPlan,
        outer_cols: Vec<Column>,
        chain: Vec<&LogicalPlan>,
        alias: Option<TableReference>,
        no_row_is_null: bool,
    ) -> Result<(LogicalPlan, Vec<Expr>, Option<Expr>)> {
        let Some((LogicalPlan::Aggregate(agg), above)) = chain.split_last() else {
            return internal_err!("expected an aggregate at the end of the chain");
        };
        let mut corr = Correlation {
            outer_cols,
            outer: &outer,
            used_domain: false,
        };
        reject_outer_only_aggregates(agg, &corr)?;
        let input = self.attach(agg.input.as_ref().clone(), &mut corr)?;
        let scope = [Arc::clone(input.plan.schema())];
        let aggr_expr = agg
            .aggr_expr
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let new_expr = replace_outer(e.clone(), &corr, &input.cols, &scope)?;
                Ok(keep_name(new_expr, &agg.schema, i))
            })
            .collect::<Result<Vec<_>>>()?;
        let (unique_keys, key_positions) = dedup_columns(&input.cols);
        let n_keys = unique_keys.len();
        let group_expr = unique_keys.into_iter().map(Expr::Column).collect();
        let grouped = Aggregate::try_new(Arc::new(input.plan), group_expr, aggr_expr)?;
        let empty_values = empty_aggregate_values(agg)?;
        // A marker column tells the outer rows without a match apart, if an
        // aggregate's value on an empty input is not NULL (`count`).
        let marker = empty_values
            .iter()
            .any(|v| !v.is_null())
            .then(|| self.fresh_name("__matched"));
        let mut grouped = LogicalPlan::Aggregate(grouped);
        // A filter right above the aggregate (a HAVING) can run before the
        // join when a dropped row means NULL anyway (`no_row_is_null`), no
        // aggregate needs its empty-input value, and the filter is not TRUE
        // for those values.
        let mut above = above;
        while no_row_is_null
            && marker.is_none()
            && let Some((LogicalPlan::Filter(f), rest)) = above.split_last()
            && !f.predicate.contains_outer()
            && !contains_subquery(&f.predicate)
            && !true_on_empty_input(&f.predicate, agg, &empty_values)?
        {
            grouped = LogicalPlanBuilder::from(grouped)
                .filter(f.predicate.clone())?
                .build()?;
            above = rest;
        }
        if let Some(marker) = &marker {
            let mut exprs: Vec<Expr> = grouped
                .schema()
                .columns()
                .into_iter()
                .map(Expr::Column)
                .collect();
            exprs.push(lit(true).alias(marker));
            grouped = LogicalPlanBuilder::from(grouped).project(exprs)?.build()?;
        }
        let grouped = self.rewrite_plan(grouped)?;
        let alias = alias.unwrap_or_else(|| self.alias_gen.next("__scalar_sq").into());
        let grouped = self.disambiguate(grouped, outer.schema())?;
        let grouped = alias_plan(grouped, alias.clone())?;
        let cols: Vec<Column> = key_positions
            .into_iter()
            .map(|i| Column::from(grouped.schema().qualified_field(i)))
            .collect();
        let keys = domain_join_keys(&corr, &cols, &input.bound);

        // The value of each aggregate: its value on an empty input when no
        // inner row matched the outer row.
        let mut values: HashMap<Column, Expr> = HashMap::new();
        for (i, empty) in empty_values.into_iter().enumerate() {
            let col =
                Expr::Column(Column::from(grouped.schema().qualified_field(n_keys + i)));
            let value = match &marker {
                Some(marker) if !empty.is_null() => {
                    let marker = Expr::Column(Column::new(Some(alias.clone()), marker));
                    when(marker.is_null(), lit(empty)).otherwise(col)?
                }
                _ => col,
            };
            values.insert(Column::from(agg.schema.qualified_field(i)), value);
        }

        let join = Join::try_new(
            Arc::new(outer.clone()),
            Arc::new(grouped),
            vec![],
            conjunction(keys),
            JoinType::Left,
            JoinConstraint::On,
            NullEquality::NullEqualsNothing,
            false,
        )?;

        // Evaluate the nodes above the aggregate over the join, innermost first.
        let mut condition = vec![];
        for node in above.iter().rev() {
            match node {
                LogicalPlan::Projection(p) => {
                    let mut next = HashMap::new();
                    for (i, e) in p.expr.iter().enumerate() {
                        let value = substitute(e.clone().unalias(), &values)?;
                        next.insert(Column::from(p.schema.qualified_field(i)), value);
                    }
                    values = next;
                }
                LogicalPlan::SubqueryAlias(a) => {
                    let input_schema = a.input.schema();
                    let mut next = HashMap::new();
                    for i in 0..a.schema.fields().len() {
                        let from = Column::from(input_schema.qualified_field(i));
                        let Some(value) = values.get(&from) else {
                            return internal_err!("missing value for {from}");
                        };
                        next.insert(
                            Column::from(a.schema.qualified_field(i)),
                            value.clone(),
                        );
                    }
                    values = next;
                }
                LogicalPlan::Filter(f) => {
                    condition.push(substitute(f.predicate.clone(), &values)?);
                }
                other => return internal_err!("unexpected node {}", other.display()),
            }
        }
        let root_schema = chain[0].schema();
        let values = root_schema
            .columns()
            .into_iter()
            .map(|c| {
                values
                    .get(&c)
                    .cloned()
                    .ok_or_else(|| DataFusionError::Internal(format!("missing {c}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let condition = conjunction(condition).map(|c| c.is_true());
        Ok((LogicalPlan::Join(join), values, condition))
    }

    /// Pushes the domain into `plan` and aliases the result. Returns the plan,
    /// with the subquery's own columns first, and its domain columns.
    fn attach_subquery(
        &mut self,
        plan: &LogicalPlan,
        corr: &mut Correlation,
        alias_prefix: &str,
        alias: Option<TableReference>,
    ) -> Result<(LogicalPlan, Vec<Column>, Vec<bool>)> {
        let own_cols = plan.schema().columns();
        let attached = self.attach(plan.clone(), corr)?;
        let mut exprs: Vec<Expr> = own_cols.iter().cloned().map(Expr::Column).collect();
        let mut positions = Vec::with_capacity(attached.cols.len());
        for col in &attached.cols {
            let e = Expr::Column(col.clone());
            match exprs.iter().position(|x| x == &e) {
                Some(i) => positions.push(i),
                None => {
                    positions.push(exprs.len());
                    exprs.push(e);
                }
            }
        }
        let projected = LogicalPlanBuilder::from(attached.plan)
            .project(exprs)?
            .build()?;
        let projected = self.rewrite_plan(projected)?;
        let alias = alias.unwrap_or_else(|| self.alias_gen.next(alias_prefix).into());
        let projected = self.disambiguate(projected, corr.outer.schema())?;
        let aliased = alias_plan(projected, alias)?;
        let bound = attached.bound;
        let cols = positions
            .into_iter()
            .map(|i| Column::from(aliased.schema().qualified_field(i)))
            .collect();
        Ok((aliased, cols, bound))
    }

    /// The domain: the distinct outer values, under fresh column names.
    fn domain(&mut self, corr: &mut Correlation) -> Result<Attached> {
        if !is_deterministic(corr.outer) {
            return unsupported("domain of a non-deterministic outer plan");
        }
        corr.used_domain = true;
        let names: Vec<String> = corr
            .outer_cols
            .iter()
            .map(|_| self.fresh_name("__d"))
            .collect();
        let exprs: Vec<Expr> = corr
            .outer_cols
            .iter()
            .zip(&names)
            .map(|(c, name)| Expr::Column(c.clone()).alias(name))
            .collect();
        let keys: Vec<Expr> = names
            .iter()
            .map(|n| Expr::Column(Column::new_unqualified(n)))
            .collect();
        let plan = LogicalPlanBuilder::from(corr.outer.clone())
            .project(exprs)?
            .aggregate(keys, Vec::<Expr>::new())?
            .build()?;
        let cols: Vec<Column> = names.into_iter().map(Column::new_unqualified).collect();
        let bound = vec![false; cols.len()];
        Ok(Attached { plan, cols, bound })
    }

    /// Rewrites `plan` so that it returns, for every outer value at once, the
    /// rows it returns for that value, with the value in [`Attached::cols`].
    ///
    /// The result may also hold rows for outer values that do not occur; the
    /// join on the domain columns drops them.
    fn attach(&mut self, plan: LogicalPlan, corr: &mut Correlation) -> Result<Attached> {
        if corr.outer_cols.is_empty() {
            return Ok(Attached {
                plan,
                cols: vec![],
                bound: vec![],
            });
        }
        if !plan_refs_outer(&plan, &corr.outer_cols, &[]) {
            let domain = self.domain(corr)?;
            let plan = LogicalPlanBuilder::from(plan)
                .cross_join(domain.plan)?
                .build()?;
            return Ok(Attached {
                plan,
                cols: domain.cols,
                bound: domain.bound,
            });
        }
        match plan {
            LogicalPlan::Filter(filter) => self.attach_filter(filter, corr),
            LogicalPlan::Projection(projection) => {
                let input = self.attach(Arc::unwrap_or_clone(projection.input), corr)?;
                let mut exprs = Vec::with_capacity(projection.expr.len());
                for (i, e) in projection.expr.into_iter().enumerate() {
                    let new_expr = replace_outer(
                        e,
                        corr,
                        &input.cols,
                        &[input.plan.schema().clone()],
                    )?;
                    exprs.push(keep_name(new_expr, &projection.schema, i));
                }
                let (exprs, cols) =
                    add_columns(exprs, &input.cols, &mut || self.fresh_name("__d"));
                let plan = Projection::try_new(exprs, Arc::new(input.plan))?;
                Ok(Attached {
                    plan: LogicalPlan::Projection(plan),
                    cols,
                    bound: input.bound,
                })
            }
            LogicalPlan::SubqueryAlias(a) => {
                let input = self.attach(Arc::unwrap_or_clone(a.input), corr)?;
                let input_schema = Arc::clone(input.plan.schema());
                let plan = SubqueryAlias::try_new(Arc::new(input.plan), a.alias)?;
                let cols = input
                    .cols
                    .iter()
                    .map(|c| {
                        let i = input_schema.index_of_column(c)?;
                        Ok(Column::from(plan.schema.qualified_field(i)))
                    })
                    .collect::<Result<_>>()?;
                Ok(Attached {
                    plan: LogicalPlan::SubqueryAlias(plan),
                    cols,
                    bound: input.bound,
                })
            }
            LogicalPlan::Aggregate(agg) => self.attach_aggregate(agg, corr),
            LogicalPlan::Distinct(Distinct::All(input)) => {
                let input = self.attach(Arc::unwrap_or_clone(input), corr)?;
                Ok(Attached {
                    plan: LogicalPlan::Distinct(Distinct::All(Arc::new(input.plan))),
                    cols: input.cols,
                    bound: input.bound,
                })
            }
            LogicalPlan::Window(window) => {
                let n_input = window.input.schema().fields().len();
                let input = self.attach(Arc::unwrap_or_clone(window.input), corr)?;
                let partition: Vec<Expr> =
                    input.cols.iter().cloned().map(Expr::Column).collect();
                let mut exprs = Vec::with_capacity(window.window_expr.len());
                for (i, e) in window.window_expr.into_iter().enumerate() {
                    let new_expr = replace_outer(
                        e,
                        corr,
                        &input.cols,
                        &[input.plan.schema().clone()],
                    )?;
                    let new_expr = add_partition_by(new_expr, &partition)?;
                    exprs.push(keep_name(new_expr, &window.schema, n_input + i));
                }
                let plan = Window::try_new(exprs, Arc::new(input.plan))?;
                Ok(Attached {
                    plan: LogicalPlan::Window(plan),
                    cols: input.cols,
                    bound: input.bound,
                })
            }
            LogicalPlan::Sort(sort) => match sort.fetch {
                Some(fetch) => self.attach_top_k(
                    Arc::unwrap_or_clone(sort.input),
                    sort.expr,
                    0,
                    Some(fetch),
                    corr,
                ),
                None => {
                    let input = self.attach(Arc::unwrap_or_clone(sort.input), corr)?;
                    let exprs = sort
                        .expr
                        .into_iter()
                        .map(|s| {
                            Ok(SortExpr {
                                expr: replace_outer(
                                    s.expr,
                                    corr,
                                    &input.cols,
                                    &[input.plan.schema().clone()],
                                )?,
                                ..s
                            })
                        })
                        .collect::<Result<_>>()?;
                    Ok(Attached {
                        plan: LogicalPlan::Sort(Sort {
                            expr: exprs,
                            input: Arc::new(input.plan),
                            fetch: None,
                        }),
                        cols: input.cols,
                        bound: input.bound,
                    })
                }
            },
            LogicalPlan::Limit(limit) => self.attach_limit(limit, corr),
            // A subquery node only marks a plan that reads outer columns.
            LogicalPlan::Subquery(sq) => {
                self.attach(Arc::unwrap_or_clone(sq.subquery), corr)
            }
            LogicalPlan::Unnest(unnest) => {
                // Unnest repeats the columns it does not unnest, so the domain
                // columns pass through.
                let input = self.attach(Arc::unwrap_or_clone(unnest.input), corr)?;
                let plan = LogicalPlanBuilder::from(input.plan)
                    .unnest_columns_with_options(unnest.exec_columns, unnest.options)?
                    .build()?;
                Ok(Attached {
                    plan,
                    cols: input.cols,
                    bound: input.bound,
                })
            }
            LogicalPlan::Join(join) => self.attach_join(join, corr),
            LogicalPlan::Union(union) => self.attach_union(union, corr),
            // An ASOF join keeps each left row with its best right match, so
            // domain columns of the left side pass through.
            LogicalPlan::AsOfJoin(asof)
                if !plan_refs_outer(&asof.right, &corr.outer_cols, &[])
                    && !asof
                        .on
                        .iter()
                        .any(|(l, r)| l.contains_outer() || r.contains_outer()) =>
            {
                let left =
                    self.attach(Arc::unwrap_or_clone(Arc::clone(&asof.left)), corr)?;
                let plan = LogicalPlan::AsOfJoin(AsOfJoin::try_new(
                    Arc::new(left.plan),
                    Arc::clone(&asof.right),
                    asof.on.clone(),
                    asof.match_condition.as_ref().clone(),
                    asof.join_constraint,
                )?);
                Ok(Attached {
                    plan,
                    cols: left.cols,
                    bound: left.bound,
                })
            }
            // A CTE whose body reads no outer column is computed once; only
            // the query that reads it gets the domain.
            LogicalPlan::MaterializedCte(MaterializedCtePlan::Cte(cte))
                if !plan_refs_outer(&cte.cte, &corr.outer_cols, &[]) =>
            {
                let continuation = self
                    .attach(Arc::unwrap_or_clone(Arc::clone(&cte.continuation)), corr)?;
                let plan = LogicalPlan::MaterializedCte(MaterializedCtePlan::Cte(
                    MaterializedCte::try_new(
                        cte.id(),
                        cte.name.clone(),
                        Arc::clone(&cte.cte),
                        Arc::new(continuation.plan),
                    )?,
                ));
                Ok(Attached {
                    plan,
                    cols: continuation.cols,
                    bound: continuation.bound,
                })
            }
            other => unsupported(format!("correlated {} in a subquery", other.display())),
        }
    }

    fn attach_filter(
        &mut self,
        filter: Filter,
        corr: &mut Correlation,
    ) -> Result<Attached> {
        let input = Arc::unwrap_or_clone(filter.input);
        if !plan_refs_outer(&input, &corr.outer_cols, &[])
            && let Some(bound) =
                bind_by_equality(&filter.predicate, corr, input.schema())?
        {
            return self.attach_bound_filter(input, bound);
        }
        let input = self.attach(input, corr)?;
        let predicate = replace_outer(
            filter.predicate,
            corr,
            &input.cols,
            &[input.plan.schema().clone()],
        )?;
        let plan = Filter::try_new(predicate, Arc::new(input.plan))?;
        Ok(Attached {
            plan: LogicalPlan::Filter(plan),
            cols: input.cols,
            bound: input.bound,
        })
    }

    /// Every outer column of the filter is bound by an `inner = outer`
    /// conjunct: the inner expression serves as the domain column.
    fn attach_bound_filter(
        &mut self,
        input: LogicalPlan,
        bound: Bound,
    ) -> Result<Attached> {
        let n = bound.values.len();
        let filtered = match conjunction(bound.rest) {
            Some(predicate) => {
                LogicalPlanBuilder::from(input).filter(predicate)?.build()?
            }
            None => input,
        };
        if bound.values.iter().all(|e| matches!(e, Expr::Column(_))) {
            let cols = bound
                .values
                .into_iter()
                .map(|e| match e {
                    Expr::Column(c) => c,
                    _ => unreachable!(),
                })
                .collect();
            return Ok(Attached {
                plan: filtered,
                cols,
                bound: vec![true; n],
            });
        }
        let mut exprs: Vec<Expr> = filtered
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        let mut cols = vec![];
        for e in bound.values {
            match e {
                Expr::Column(c) => cols.push(c),
                e => {
                    let name = self.fresh_name("__d");
                    exprs.push(e.alias(&name));
                    cols.push(Column::new_unqualified(name));
                }
            }
        }
        let plan = LogicalPlanBuilder::from(filtered).project(exprs)?.build()?;
        Ok(Attached {
            plan,
            cols,
            bound: vec![true; n],
        })
    }

    fn attach_aggregate(
        &mut self,
        agg: Aggregate,
        corr: &mut Correlation,
    ) -> Result<Attached> {
        let grouping_sets = agg
            .group_expr
            .iter()
            .any(|e| matches!(e, Expr::GroupingSet(_)));
        reject_outer_only_aggregates(&agg, corr)?;
        let input = self.attach(Arc::unwrap_or_clone(Arc::clone(&agg.input)), corr)?;
        let scope = [input.plan.schema().clone()];
        let n_group = agg.group_expr.len();
        let mut group_expr = Vec::with_capacity(n_group);
        for (i, e) in agg.group_expr.iter().enumerate() {
            let new_expr = replace_outer(e.clone(), corr, &input.cols, &scope)?;
            // A grouping set names its output columns itself.
            if matches!(new_expr, Expr::GroupingSet(_)) {
                group_expr.push(new_expr);
            } else {
                group_expr.push(keep_name(new_expr, &agg.schema, i));
            }
        }
        let mut aggr_expr = Vec::with_capacity(agg.aggr_expr.len());
        for (i, e) in agg.aggr_expr.iter().enumerate() {
            let new_expr = replace_outer(e.clone(), corr, &input.cols, &scope)?;
            aggr_expr.push(keep_name(new_expr, &agg.schema, n_group + i));
        }

        if grouping_sets {
            let [Expr::GroupingSet(set)] = group_expr.as_slice() else {
                return unsupported("grouping sets mixed with grouping keys");
            };
            let sets: Vec<Vec<Expr>> = match set {
                GroupingSet::GroupingSets(sets) => sets.clone(),
                GroupingSet::Rollup(exprs) => {
                    (0..=exprs.len()).map(|i| exprs[..i].to_vec()).collect()
                }
                GroupingSet::Cube(_) => return unsupported("correlated CUBE"),
            };
            if sets.iter().enumerate().any(|(i, x)| sets[..i].contains(x)) {
                return unsupported("correlated grouping sets with duplicates");
            }
            let has_empty_set = sets.iter().any(|set| set.is_empty());
            // The domain columns go first in every set: they are never NULL
            // there, so they add only leading zero bits to `__grouping_id`
            // and the GROUPING() values above stay the same. A domain column
            // that is also a grouping key gets a copy, as the key is NULL in
            // the sets without it.
            let mut input = input;
            let in_sets = |c: &Column| {
                sets.iter()
                    .any(|set| set.contains(&Expr::Column(c.clone())))
            };
            if input.cols.iter().any(|c| in_sets(c)) {
                let mut exprs: Vec<Expr> = input
                    .plan
                    .schema()
                    .columns()
                    .into_iter()
                    .map(Expr::Column)
                    .collect();
                let mut cols = vec![];
                for c in &input.cols {
                    if in_sets(c) {
                        let name = self.fresh_name("__d");
                        exprs.push(Expr::Column(c.clone()).alias(&name));
                        cols.push(Column::new_unqualified(name));
                    } else {
                        cols.push(c.clone());
                    }
                }
                input.plan = LogicalPlanBuilder::from(input.plan)
                    .project(exprs)?
                    .build()?;
                input.cols = cols;
            }
            let keys: Vec<Expr> = input.cols.iter().cloned().map(Expr::Column).collect();
            let sets = sets
                .into_iter()
                .map(|set| keys.iter().cloned().chain(set).collect())
                .collect();
            let group_expr = vec![Expr::GroupingSet(GroupingSet::GroupingSets(sets))];
            let plan = Aggregate::try_new(Arc::new(input.plan), group_expr, aggr_expr)?;
            let id_type = |schema: &DFSchemaRef| {
                schema
                    .field_with_unqualified_name(Aggregate::INTERNAL_GROUPING_ID)
                    .map(|f| f.data_type().clone())
                    .ok()
            };
            if id_type(&plan.schema) != id_type(&agg.schema) {
                return unsupported("correlated grouping sets that widen __grouping_id");
            }
            if !has_empty_set {
                return Ok(Attached {
                    plan: LogicalPlan::Aggregate(plan),
                    cols: input.cols,
                    bound: input.bound,
                });
            }
            return self.add_empty_set_rows(plan, input.cols, input.bound, &agg, corr);
        }
        if n_group > 0 {
            let (group_expr, cols) =
                add_columns(group_expr, &input.cols, &mut || self.fresh_name("__d"));
            let plan = Aggregate::try_new(Arc::new(input.plan), group_expr, aggr_expr)?;
            return Ok(Attached {
                plan: LogicalPlan::Aggregate(plan),
                cols,
                bound: input.bound,
            });
        }

        // Without GROUP BY the aggregate returns one row even for an outer
        // value with no input rows. Group by the domain columns, and left join
        // the domain to it to bring those values back with the aggregates'
        // values on an empty input.
        let (unique_keys, key_positions) = dedup_columns(&input.cols);
        let n_keys = unique_keys.len();
        let key_exprs = unique_keys.into_iter().map(Expr::Column).collect();
        let grouped = Aggregate::try_new(Arc::new(input.plan), key_exprs, aggr_expr)?;
        let empty_values = empty_aggregate_values(&agg)?;
        let marker = empty_values
            .iter()
            .any(|v| !v.is_null())
            .then(|| self.fresh_name("__matched"));
        let mut grouped = LogicalPlan::Aggregate(grouped);
        if let Some(marker) = &marker {
            let mut exprs: Vec<Expr> = grouped
                .schema()
                .columns()
                .into_iter()
                .map(Expr::Column)
                .collect();
            exprs.push(lit(true).alias(marker));
            grouped = LogicalPlanBuilder::from(grouped).project(exprs)?.build()?;
        }
        let grouped_keys: Vec<Column> = key_positions
            .iter()
            .map(|&i| Column::from(grouped.schema().qualified_field(i)))
            .collect();

        let domain = self.domain(corr)?;
        let keys: Vec<Expr> = domain
            .cols
            .iter()
            .zip(&grouped_keys)
            .zip(&input.bound)
            .map(|((d, g), &bound)| key_equality(d, g, bound))
            .collect();
        let join = Join::try_new(
            Arc::new(domain.plan),
            Arc::new(grouped),
            vec![],
            conjunction(keys),
            JoinType::Left,
            JoinConstraint::On,
            NullEquality::NullEqualsNothing,
            false,
        )?;
        let join = LogicalPlan::Join(join);
        let mut exprs: Vec<Expr> =
            domain.cols.iter().cloned().map(Expr::Column).collect();
        for (i, empty) in empty_values.into_iter().enumerate() {
            let (qualifier, field) = agg.schema.qualified_field(i);
            let col = Expr::Column(Column::from(
                join.schema()
                    .qualified_field(domain.cols.len() + n_keys + i),
            ));
            let value = match &marker {
                Some(marker) if !empty.is_null() => {
                    let marker = Expr::Column(Column::new_unqualified(marker));
                    when(marker.is_null(), lit(empty)).otherwise(col)?
                }
                _ => col,
            };
            exprs.push(value.alias_qualified(qualifier.cloned(), field.name()));
        }
        let plan = LogicalPlanBuilder::from(join).project(exprs)?.build()?;
        Ok(Attached {
            plan,
            cols: domain.cols,
            bound: domain.bound,
        })
    }

    /// The empty grouping set gives one row even for an empty input, so an
    /// outer value without input rows still gets its row: the grouping
    /// columns NULL and the aggregates at their empty-input values.
    /// `grouped` groups by the domain columns `cols` first in every set.
    fn add_empty_set_rows(
        &mut self,
        grouped: Aggregate,
        cols: Vec<Column>,
        bound: Vec<bool>,
        agg: &Aggregate,
        corr: &mut Correlation,
    ) -> Result<Attached> {
        let empty_values = empty_aggregate_values(agg)?;
        let grouped = LogicalPlan::Aggregate(grouped);
        let schema = Arc::clone(grouped.schema());
        // A bound domain column can hold NULLs that match nothing; the filler
        // rows below hold domain values, which can be NULL themselves.
        let not_null: Vec<Expr> = cols
            .iter()
            .zip(&bound)
            .filter(|(_, b)| **b)
            .map(|(c, _)| Expr::Column(c.clone()).is_not_null())
            .collect();
        let grouped = match conjunction(not_null) {
            Some(predicate) => LogicalPlanBuilder::from(grouped)
                .filter(predicate)?
                .build()?,
            None => grouped,
        };
        let domain = self.domain(corr)?;
        let keys: Vec<Expr> = domain
            .cols
            .iter()
            .zip(&cols)
            .map(|(d, c)| key_equality(d, c, false))
            .collect();
        let missing = LogicalPlan::Join(Join::try_new(
            Arc::new(domain.plan),
            Arc::new(grouped.clone()),
            vec![],
            conjunction(keys),
            JoinType::LeftAnti,
            JoinConstraint::On,
            NullEquality::NullEqualsNothing,
            false,
        )?);
        // Output layout of a grouping-set aggregate: the grouping columns (the
        // domain columns first), `__grouping_id`, then the aggregates.
        let n_fields = schema.fields().len();
        let n_aggr = empty_values.len();
        let id_index = n_fields - n_aggr - 1;
        let n_group = id_index;
        let n_original = n_group - cols.len();
        let mut exprs = vec![];
        for (i, d) in domain.cols.iter().enumerate() {
            let _ = i;
            exprs.push(Expr::Column(d.clone()));
        }
        for i in cols.len()..n_group {
            let (_, field) = schema.qualified_field(i);
            exprs.push(lit(ScalarValue::try_from(field.data_type())?));
        }
        let (_, id_field) = schema.qualified_field(id_index);
        // Every original grouping column is NULL in the empty set.
        let id = (1u64 << n_original) - 1;
        exprs.push(lit(
            ScalarValue::UInt64(Some(id)).cast_to(id_field.data_type())?
        ));
        for empty in empty_values {
            exprs.push(lit(empty));
        }
        let filler = LogicalPlanBuilder::from(missing).project(exprs)?.build()?;
        let union =
            Union::try_new_with_loose_types(vec![Arc::new(grouped), Arc::new(filler)])?;
        // The union drops the qualifiers; restore the names of the aggregate.
        let exprs: Vec<Expr> = (0..n_fields)
            .map(|i| {
                let (q, f) = schema.qualified_field(i);
                Expr::Column(Column::from(union.schema.qualified_field(i)))
                    .alias_qualified(q.cloned(), f.name())
            })
            .collect();
        let plan = LogicalPlanBuilder::from(LogicalPlan::Union(union))
            .project(exprs)?
            .build()?;
        let n = cols.len();
        Ok(Attached {
            plan,
            cols,
            bound: vec![false; n],
        })
    }

    fn attach_limit(&mut self, limit: Limit, corr: &mut Correlation) -> Result<Attached> {
        let SkipType::Literal(skip) = limit.get_skip_type()? else {
            return unsupported("non-literal OFFSET in a correlated subquery");
        };
        let FetchType::Literal(fetch) = limit.get_fetch_type()? else {
            return unsupported("non-literal LIMIT in a correlated subquery");
        };
        match Arc::unwrap_or_clone(limit.input) {
            // A projection computes each row on its own, so the limit can go
            // below it, next to the sort it applies to.
            LogicalPlan::Projection(projection) => {
                let below = LogicalPlan::Limit(Limit {
                    skip: limit.skip,
                    fetch: limit.fetch,
                    input: projection.input,
                });
                let plan = Projection::try_new_with_schema(
                    projection.expr,
                    Arc::new(below),
                    projection.schema,
                )?;
                self.attach(LogicalPlan::Projection(plan), corr)
            }
            LogicalPlan::Sort(sort) if sort.fetch.is_none() => self.attach_top_k(
                Arc::unwrap_or_clone(sort.input),
                sort.expr,
                skip,
                fetch,
                corr,
            ),
            input => self.attach_top_k(input, vec![], skip, fetch, corr),
        }
    }

    /// `LIMIT` per outer value: `row_number()` over the domain columns.
    fn attach_top_k(
        &mut self,
        input: LogicalPlan,
        order: Vec<SortExpr>,
        skip: usize,
        fetch: Option<usize>,
        corr: &mut Correlation,
    ) -> Result<Attached> {
        let Some(row_number) = self.registry.and_then(|r| r.udwf("row_number").ok())
        else {
            return unsupported("LIMIT without a row_number window function");
        };
        let input = self.attach(input, corr)?;
        let scope = [input.plan.schema().clone()];
        let order = order
            .into_iter()
            .map(|s| {
                Ok(SortExpr {
                    expr: replace_outer(s.expr, corr, &input.cols, &scope)?,
                    ..s
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let output: Vec<Expr> = input
            .plan
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        let rn_name = self.fresh_name("__rn");
        let rn = Expr::from(WindowFunction::new(
            WindowFunctionDefinition::WindowUDF(row_number),
            vec![],
        ))
        .partition_by(input.cols.iter().cloned().map(Expr::Column).collect())
        .order_by(order)
        .build()?
        .alias(&rn_name);
        let rn_col = Expr::Column(Column::new_unqualified(&rn_name));
        let mut keep = vec![rn_col.clone().gt(lit(skip as u64))];
        if let Some(fetch) = fetch {
            keep.push(rn_col.lt_eq(lit((skip + fetch) as u64)));
        }
        let plan = LogicalPlanBuilder::from(input.plan)
            .window(vec![rn])?
            .filter(conjunction(keep).expect("at least one conjunct"))?
            .project(output)?
            .build()?;
        Ok(Attached {
            plan,
            cols: input.cols,
            bound: input.bound,
        })
    }

    fn attach_join(&mut self, join: Join, corr: &mut Correlation) -> Result<Attached> {
        if join.null_aware {
            return unsupported("correlated null-aware join");
        }
        let on_refs = join
            .on
            .iter()
            .any(|(l, r)| l.contains_outer() || r.contains_outer());
        if on_refs {
            return unsupported("outer reference in equi-join keys");
        }
        let left_refs = plan_refs_outer(&join.left, &corr.outer_cols, &[]);
        let right_refs = plan_refs_outer(&join.right, &corr.outer_cols, &[]);
        let join_type = join.join_type;
        // Which sides get the domain columns. A side gets them when its rows
        // depend on the outer value, or when the join must keep its rows per
        // outer value: the preserved side of an outer join, and the side a
        // semi, anti or mark join returns.
        let (attach_left, attach_right) = match join_type {
            JoinType::Inner => (left_refs || !right_refs, right_refs),
            JoinType::Left
            | JoinType::LeftSemi
            | JoinType::LeftAnti
            | JoinType::LeftMark => (true, right_refs),
            JoinType::Right
            | JoinType::RightSemi
            | JoinType::RightAnti
            | JoinType::RightMark => (left_refs, true),
            JoinType::Full => (true, true),
        };
        let left = Arc::unwrap_or_clone(join.left);
        let right = Arc::unwrap_or_clone(join.right);
        let (left, left_cols, left_bound) = if attach_left {
            let a = self.attach(left, corr)?;
            (a.plan, Some(a.cols), a.bound)
        } else {
            (left, None, vec![])
        };
        let (right, right_cols, right_bound) = if attach_right {
            let a = self.attach(right, corr)?;
            (a.plan, Some(a.cols), a.bound)
        } else {
            (right, None, vec![])
        };
        let cond_cols = left_cols
            .as_ref()
            .or(right_cols.as_ref())
            .expect("one side is attached")
            .clone();
        let scope = [Arc::clone(left.schema()), Arc::clone(right.schema())];
        let mut conjuncts = vec![];
        if let Some(filter) = join.filter {
            conjuncts.push(replace_outer(filter, corr, &cond_cols, &scope)?);
        }
        // The output's domain column, and whether it is bound, per join type.
        let both = left_cols.is_some() && right_cols.is_some();
        let either_bound: Vec<bool> = (0..corr.outer_cols.len())
            .map(|i| {
                left_bound.get(i).copied().unwrap_or(false)
                    || right_bound.get(i).copied().unwrap_or(false)
            })
            .collect();
        if let (Some(l), Some(r)) = (&left_cols, &right_cols) {
            for ((l, r), &bound) in l.iter().zip(r).zip(&either_bound) {
                conjuncts.push(key_equality(l, r, bound));
            }
        }
        let plan = LogicalPlan::Join(Join::try_new(
            Arc::new(left),
            Arc::new(right),
            join.on,
            conjunction(conjuncts),
            join_type,
            join.join_constraint,
            join.null_equality,
            false,
        )?);
        match join_type {
            JoinType::Full => {
                let (l, r) =
                    (left_cols.expect("attached"), right_cols.expect("attached"));
                let mut exprs: Vec<Expr> = plan
                    .schema()
                    .columns()
                    .into_iter()
                    .map(Expr::Column)
                    .collect();
                let mut cols = vec![];
                for (l, r) in l.into_iter().zip(r) {
                    let name = self.fresh_name("__d");
                    let value =
                        when(Expr::Column(l.clone()).is_not_null(), Expr::Column(l))
                            .otherwise(Expr::Column(r))?;
                    exprs.push(value.alias(&name));
                    cols.push(Column::new_unqualified(name));
                }
                let plan = LogicalPlanBuilder::from(plan).project(exprs)?.build()?;
                let bound = left_bound
                    .iter()
                    .zip(&right_bound)
                    .map(|(l, r)| *l && *r)
                    .collect();
                Ok(Attached { plan, cols, bound })
            }
            JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark => {
                Ok(Attached {
                    plan,
                    cols: right_cols.expect("attached"),
                    bound: right_bound,
                })
            }
            // An inner join with a bound side drops the rows whose value is
            // NULL, so its output is bound too.
            JoinType::Inner if both => Ok(Attached {
                plan,
                cols: left_cols.expect("attached"),
                bound: either_bound,
            }),
            _ => match left_cols {
                Some(cols) => Ok(Attached {
                    plan,
                    cols,
                    bound: left_bound,
                }),
                None => Ok(Attached {
                    plan,
                    cols: right_cols.expect("one side is attached"),
                    bound: right_bound,
                }),
            },
        }
    }

    fn attach_union(&mut self, union: Union, corr: &mut Correlation) -> Result<Attached> {
        let names: Vec<String> = corr
            .outer_cols
            .iter()
            .map(|_| self.fresh_name("__d"))
            .collect();
        let mut branches = Vec::with_capacity(union.inputs.len());
        for input in union.inputs {
            let own_cols = input.schema().columns();
            branches.push((own_cols, self.attach(Arc::unwrap_or_clone(input), corr)?));
        }
        // A column is bound in the union only if it is bound in every branch.
        // Elsewhere a bound branch drops its NULLs, as they match nothing.
        let bound: Vec<bool> = (0..names.len())
            .map(|i| branches.iter().all(|(_, a)| a.bound[i]))
            .collect();
        let mut inputs = Vec::with_capacity(branches.len());
        for (own_cols, attached) in branches {
            let mut plan = LogicalPlanBuilder::from(attached.plan);
            let not_null: Vec<Expr> = attached
                .cols
                .iter()
                .zip(&attached.bound)
                .zip(&bound)
                .filter(|((_, b), out)| **b && !**out)
                .map(|((c, _), _)| Expr::Column(c.clone()).is_not_null())
                .collect();
            if let Some(predicate) = conjunction(not_null) {
                plan = plan.filter(predicate)?;
            }
            let mut exprs: Vec<Expr> = own_cols.into_iter().map(Expr::Column).collect();
            for (col, name) in attached.cols.iter().zip(&names) {
                exprs.push(Expr::Column(col.clone()).alias(name));
            }
            inputs.push(Arc::new(plan.project(exprs)?.build()?));
        }
        let plan = Union::try_new_with_loose_types(inputs)?;
        let n = plan.schema.fields().len();
        let first = n - names.len();
        let cols = (first..n)
            .map(|i| Column::from(plan.schema.qualified_field(i)))
            .collect();
        Ok(Attached {
            plan: LogicalPlan::Union(plan),
            cols,
            bound,
        })
    }

    /// Pulls the correlated conjuncts of the filters that sit below the root
    /// with only filters, projections and aliases on the way. Returns `None`
    /// if any other node reads an outer column.
    ///
    /// The conjuncts are in terms of the output columns of the returned plan.
    fn pull_up(
        &mut self,
        plan: LogicalPlan,
        corr: &Correlation,
    ) -> Result<Option<(LogicalPlan, Vec<Expr>)>> {
        match plan {
            LogicalPlan::Filter(filter) => {
                let Some((input, mut conjuncts)) =
                    self.pull_up(Arc::unwrap_or_clone(filter.input), corr)?
                else {
                    return Ok(None);
                };
                let mut local = vec![];
                for conjunct in split_conjunction_owned(filter.predicate) {
                    if !expr_refs_outer(
                        &conjunct,
                        &corr.outer_cols,
                        &[],
                        &[Arc::clone(input.schema())],
                    ) {
                        local.push(conjunct);
                    } else if contains_subquery(&conjunct) {
                        return Ok(None);
                    } else {
                        conjuncts.push(conjunct);
                    }
                }
                let plan = match conjunction(local) {
                    Some(predicate) => {
                        LogicalPlan::Filter(Filter::try_new(predicate, Arc::new(input))?)
                    }
                    None => input,
                };
                Ok(Some((plan, conjuncts)))
            }
            LogicalPlan::Projection(projection) => {
                let input_refs = projection.expr.iter().any(|e| {
                    expr_refs_outer(
                        e,
                        &corr.outer_cols,
                        &[],
                        &[Arc::clone(projection.input.schema())],
                    )
                });
                if input_refs {
                    return Ok(None);
                }
                let Some((input, conjuncts)) =
                    self.pull_up(Arc::unwrap_or_clone(projection.input), corr)?
                else {
                    return Ok(None);
                };
                let mut exprs = projection.expr;
                let mut names: HashSet<(Option<TableReference>, String)> =
                    exprs.iter().map(|e| e.qualified_name()).collect();
                let mut needed = vec![];
                for conjunct in &conjuncts {
                    for col in conjunct.column_refs() {
                        if !needed.contains(col) {
                            needed.push(col.clone());
                        }
                    }
                }
                for col in needed {
                    if exprs.contains(&Expr::Column(col.clone())) {
                        continue;
                    }
                    let name = (col.relation.clone(), col.name.clone());
                    if names.contains(&name) {
                        return Ok(None);
                    }
                    names.insert(name);
                    exprs.push(Expr::Column(col));
                }
                let plan = Projection::try_new(exprs, Arc::new(input))?;
                Ok(Some((LogicalPlan::Projection(plan), conjuncts)))
            }
            LogicalPlan::SubqueryAlias(a) => {
                let Some((input, conjuncts)) =
                    self.pull_up(Arc::unwrap_or_clone(a.input), corr)?
                else {
                    return Ok(None);
                };
                let input_schema = Arc::clone(input.schema());
                let plan = SubqueryAlias::try_new(Arc::new(input), a.alias)?;
                let conjuncts = conjuncts
                    .into_iter()
                    .map(|c| requalify(c, &input_schema, &plan.schema))
                    .collect::<Result<_>>()?;
                Ok(Some((LogicalPlan::SubqueryAlias(plan), conjuncts)))
            }
            other => {
                if plan_refs_outer(&other, &corr.outer_cols, &[]) {
                    Ok(None)
                } else {
                    Ok(Some((other, vec![])))
                }
            }
        }
    }
}

/// The outer columns `subquery` reads from `outer`, in order of appearance.
fn free_outer_columns(
    subquery: &LogicalPlan,
    outer: &LogicalPlan,
) -> Result<Vec<Column>> {
    let mut cols = vec![];
    collect_outer_columns(subquery, &[], &mut cols);
    let schema = outer.schema();
    for col in &cols {
        if !schema.has_column(col) {
            return unsupported(format!(
                "outer reference {col} that is not a column of the outer plan"
            ));
        }
    }
    Ok(cols)
}

/// Collects the outer references of `plan` that do not resolve to one of
/// `scopes` (the inputs of the subqueries `plan` is nested in).
fn collect_outer_columns(
    plan: &LogicalPlan,
    scopes: &[DFSchemaRef],
    cols: &mut Vec<Column>,
) {
    let inputs = input_schemas(plan);
    let _ = plan.apply_expressions(|e| {
        let _ = e.apply(|e| {
            match e {
                Expr::OuterReferenceColumn(_, c) => {
                    if !scopes.iter().any(|s| s.has_column(c)) && !cols.contains(c) {
                        cols.push(c.clone());
                    }
                }
                Expr::ScalarSubquery(sq)
                | Expr::Exists(Exists { subquery: sq, .. })
                | Expr::InSubquery(InSubquery { subquery: sq, .. }) => {
                    let mut inner = scopes.to_vec();
                    inner.extend(inputs.iter().cloned());
                    collect_outer_columns(&sq.subquery, &inner, cols);
                }
                _ => {}
            }
            Ok(TreeNodeRecursion::Continue)
        });
        Ok(TreeNodeRecursion::Continue)
    });
    match plan {
        LogicalPlan::Join(join) if is_lateral_right(&join.right) => {
            collect_outer_columns(&join.left, scopes, cols);
            let mut inner = scopes.to_vec();
            inner.push(Arc::clone(join.left.schema()));
            collect_outer_columns(&join.right, &inner, cols);
        }
        _ => {
            for input in plan.inputs() {
                collect_outer_columns(input, scopes, cols);
            }
        }
    }
}

/// Whether `outer_col` is one of `outer_cols`, and not a column of a nearer scope.
fn resolves_to_outer(c: &Column, outer_cols: &[Column], scopes: &[DFSchemaRef]) -> bool {
    !scopes.iter().any(|s| s.has_column(c)) && outer_cols.contains(c)
}

/// Whether some node of `plan`, or a subquery nested in one, reads one of
/// `outer_cols`.
fn plan_refs_outer(
    plan: &LogicalPlan,
    outer_cols: &[Column],
    scopes: &[DFSchemaRef],
) -> bool {
    let inputs = input_schemas(plan);
    let mut found = false;
    let _ = plan.apply_expressions(|e| {
        found |= expr_refs_outer(e, outer_cols, scopes, &inputs);
        Ok(if found {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    });
    if found {
        return true;
    }
    match plan {
        LogicalPlan::Join(join) if is_lateral_right(&join.right) => {
            let mut inner = scopes.to_vec();
            inner.push(Arc::clone(join.left.schema()));
            plan_refs_outer(&join.left, outer_cols, scopes)
                || plan_refs_outer(&join.right, outer_cols, &inner)
        }
        _ => plan
            .inputs()
            .iter()
            .any(|input| plan_refs_outer(input, outer_cols, scopes)),
    }
}

/// Whether `expr`, an expression of `node`, reads one of `outer_cols`.
fn expr_refs_outer(
    expr: &Expr,
    outer_cols: &[Column],
    scopes: &[DFSchemaRef],
    node_inputs: &[DFSchemaRef],
) -> bool {
    expr.exists(|e| {
        Ok(match e {
            Expr::OuterReferenceColumn(_, c) => resolves_to_outer(c, outer_cols, scopes),
            Expr::ScalarSubquery(sq)
            | Expr::Exists(Exists { subquery: sq, .. })
            | Expr::InSubquery(InSubquery { subquery: sq, .. }) => {
                let mut inner = scopes.to_vec();
                inner.extend(node_inputs.iter().cloned());
                plan_refs_outer(&sq.subquery, outer_cols, &inner)
            }
            _ => false,
        })
    })
    .expect("infallible")
}

fn input_schemas(node: &LogicalPlan) -> Vec<DFSchemaRef> {
    node.inputs()
        .iter()
        .map(|i| Arc::clone(i.schema()))
        .collect()
}

fn contains_subquery(expr: &Expr) -> bool {
    expr.exists(|e| {
        Ok(matches!(
            e,
            Expr::ScalarSubquery(_)
                | Expr::Exists(_)
                | Expr::InSubquery(_)
                | Expr::SetComparison(_)
        ))
    })
    .expect("infallible")
}

/// Replaces the outer references to `corr.outer_cols` in `expr` with the
/// domain columns `cols`. In nested subqueries they stay outer references, to
/// the domain columns.
///
/// `scopes` is empty for an expression of the node whose input holds `cols`;
/// a nested subquery adds the inputs of the node it is in.
fn replace_outer(
    expr: Expr,
    corr: &Correlation,
    cols: &[Column],
    input: &[DFSchemaRef],
) -> Result<Expr> {
    replace_outer_at(expr, corr, cols, input, &[])
}

fn replace_outer_at(
    expr: Expr,
    corr: &Correlation,
    cols: &[Column],
    input: &[DFSchemaRef],
    scopes: &[DFSchemaRef],
) -> Result<Expr> {
    expr.transform_down(|e| match e {
        Expr::OuterReferenceColumn(field, c) => {
            if !resolves_to_outer(&c, &corr.outer_cols, scopes) {
                return Ok(Transformed::no(Expr::OuterReferenceColumn(field, c)));
            }
            let i = corr
                .outer_cols
                .iter()
                .position(|o| o == &c)
                .expect("resolved");
            let col = cols[i].clone();
            if scopes.is_empty() {
                Ok(Transformed::yes(Expr::Column(col)))
            } else {
                let field = input
                    .iter()
                    .find_map(|s| s.qualified_field_from_column(&col).ok())
                    .map(|(_, f)| Arc::clone(f))
                    .unwrap_or(field);
                Ok(Transformed::yes(Expr::OuterReferenceColumn(field, col)))
            }
        }
        Expr::ScalarSubquery(sq) => Ok(Transformed::new(
            Expr::ScalarSubquery(replace_outer_in_subquery(
                sq, corr, cols, input, scopes,
            )?),
            true,
            TreeNodeRecursion::Jump,
        )),
        Expr::Exists(Exists { subquery, negated }) => Ok(Transformed::new(
            Expr::Exists(Exists {
                subquery: replace_outer_in_subquery(subquery, corr, cols, input, scopes)?,
                negated,
            }),
            true,
            TreeNodeRecursion::Jump,
        )),
        Expr::InSubquery(InSubquery {
            expr,
            subquery,
            negated,
        }) => {
            let expr = replace_outer_at(*expr, corr, cols, input, scopes)?;
            Ok(Transformed::new(
                Expr::InSubquery(InSubquery {
                    expr: Box::new(expr),
                    subquery: replace_outer_in_subquery(
                        subquery, corr, cols, input, scopes,
                    )?,
                    negated,
                }),
                true,
                TreeNodeRecursion::Jump,
            ))
        }
        other => Ok(Transformed::no(other)),
    })
    .data()
}

fn replace_outer_in_subquery(
    sq: Subquery,
    corr: &Correlation,
    cols: &[Column],
    input: &[DFSchemaRef],
    scopes: &[DFSchemaRef],
) -> Result<Subquery> {
    // Inside the subquery, a column of the node's input shadows an outer
    // column with the same name.
    let mut inner_scopes = scopes.to_vec();
    inner_scopes.extend(input.iter().cloned());
    let plan = replace_outer_in_plan(
        Arc::unwrap_or_clone(sq.subquery),
        corr,
        cols,
        input,
        &inner_scopes,
    )?;
    let outer_ref_columns = sq
        .outer_ref_columns
        .into_iter()
        .map(|e| replace_outer_at(e, corr, cols, input, &inner_scopes))
        .collect::<Result<Vec<_>>>()?;
    let mut dedup: Vec<Expr> = vec![];
    for e in outer_ref_columns {
        if !dedup.contains(&e) {
            dedup.push(e);
        }
    }
    Ok(Subquery {
        subquery: Arc::new(plan),
        outer_ref_columns: dedup,
        spans: sq.spans,
    })
}

fn replace_outer_in_plan(
    plan: LogicalPlan,
    corr: &Correlation,
    cols: &[Column],
    input: &[DFSchemaRef],
    scopes: &[DFSchemaRef],
) -> Result<LogicalPlan> {
    plan.transform_up(|node| {
        let node_inputs: Vec<DFSchemaRef> = node
            .inputs()
            .iter()
            .map(|i| Arc::clone(i.schema()))
            .collect();
        node.map_expressions(|e| {
            let e = e.transform_down(|e| match e {
                Expr::OuterReferenceColumn(..) => {
                    replace_outer_at(e, corr, cols, input, scopes).map(Transformed::yes)
                }
                Expr::ScalarSubquery(_) | Expr::Exists(_) | Expr::InSubquery(_) => {
                    let mut nested = scopes.to_vec();
                    nested.extend(node_inputs.iter().cloned());
                    let e = replace_outer_at(e, corr, cols, input, &nested)?;
                    Ok(Transformed::new(e, true, TreeNodeRecursion::Jump))
                }
                other => Ok(Transformed::no(other)),
            })?;
            Ok(e)
        })?
        .map_data(|node| node.recompute_schema())
    })
    .data()
}

/// Replaces the outer references to `corr.outer_cols` with the outer columns
/// themselves, for a join condition above the outer plan.
fn strip_outer(expr: Expr, corr: &Correlation) -> Result<Expr> {
    expr.transform_down(|e| match e {
        Expr::OuterReferenceColumn(_, c) if corr.outer_cols.contains(&c) => {
            Ok(Transformed::yes(Expr::Column(c)))
        }
        other => Ok(Transformed::no(other)),
    })
    .data()
}

/// Whether `a` and `b` are the same comparison, maybe with swapped operands.
fn same_equality(a: &Expr, b: &Expr) -> bool {
    if a == b {
        return true;
    }
    match (a, b) {
        (Expr::BinaryExpr(x), Expr::BinaryExpr(y)) => {
            x.op == Operator::Eq
                && y.op == Operator::Eq
                && x.left == y.right
                && x.right == y.left
        }
        _ => false,
    }
}

/// The distinct columns of `cols`, and the position of each column of `cols`
/// among them.
fn dedup_columns(cols: &[Column]) -> (Vec<Column>, Vec<usize>) {
    let mut unique: Vec<Column> = vec![];
    let mut positions = Vec::with_capacity(cols.len());
    for c in cols {
        match unique.iter().position(|u| u == c) {
            Some(i) => positions.push(i),
            None => {
                positions.push(unique.len());
                unique.push(c.clone());
            }
        }
    }
    (unique, positions)
}

/// `outer_col = domain_col` for each domain column, `IS NOT DISTINCT FROM`
/// for a column that holds domain values, which can be NULL.
fn domain_join_keys(corr: &Correlation, cols: &[Column], bound: &[bool]) -> Vec<Expr> {
    corr.outer_cols
        .iter()
        .zip(cols)
        .zip(bound)
        .map(|((o, d), &bound)| key_equality(o, d, bound))
        .collect()
}

/// `a = b` if one side is bound (its NULL matches nothing), otherwise
/// `a IS NOT DISTINCT FROM b`.
fn key_equality(a: &Column, b: &Column, bound: bool) -> Expr {
    let op = if bound {
        Operator::Eq
    } else {
        Operator::IsNotDistinctFrom
    };
    Expr::BinaryExpr(BinaryExpr::new(
        Box::new(Expr::Column(a.clone())),
        op,
        Box::new(Expr::Column(b.clone())),
    ))
}

/// The correlated conjuncts that bind each outer column to an inner value.
struct Bound {
    /// `values[i]` is the inner expression `outer_cols[i]` equals.
    values: Vec<Expr>,
    /// The other conjuncts, with the bound outer columns replaced.
    rest: Vec<Expr>,
}

/// If every outer column is bound by a conjunct `inner_expr = outer_col` of
/// `predicate`, returns the bindings. The inner expressions must not read
/// outer columns or contain subqueries.
fn bind_by_equality(
    predicate: &Expr,
    corr: &Correlation,
    input_schema: &DFSchemaRef,
) -> Result<Option<Bound>> {
    let conjuncts = split_conjunction_owned(predicate.clone());
    let mut values: Vec<Option<Expr>> = vec![None; corr.outer_cols.len()];
    let mut used = vec![false; conjuncts.len()];
    for (j, conjunct) in conjuncts.iter().enumerate() {
        let Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        }) = conjunct
        else {
            continue;
        };
        let (outer, inner) = match (left.as_ref(), right.as_ref()) {
            (Expr::OuterReferenceColumn(_, c), inner) => (c, inner),
            (inner, Expr::OuterReferenceColumn(_, c)) => (c, inner),
            _ => continue,
        };
        if inner.contains_outer()
            || inner.is_volatile()
            || !inner.any_column_refs()
            || contains_subquery(inner)
        {
            continue;
        }
        let Some(i) = corr.outer_cols.iter().position(|o| o == outer) else {
            continue;
        };
        if values[i].is_none() {
            values[i] = Some(inner.clone());
            used[j] = true;
        }
    }
    let Some(values) = values.into_iter().collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    let mut map = HashMap::new();
    for (o, v) in corr.outer_cols.iter().zip(&values) {
        map.insert(o.clone(), v.clone());
    }
    // A nested subquery can read a bound outer column too: it then reads the
    // inner column instead, which needs the inner value to be a column.
    let bound_cols: Option<Vec<Column>> = values
        .iter()
        .map(|v| match v {
            Expr::Column(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    let mut rest = vec![];
    for (c, used) in conjuncts.into_iter().zip(used) {
        if used {
            continue;
        }
        let nested_refs = contains_subquery(&c)
            && expr_refs_outer(&c, &corr.outer_cols, &[], &[Arc::clone(input_schema)]);
        let c = if nested_refs {
            let Some(cols) = &bound_cols else {
                return Ok(None);
            };
            replace_outer(c, corr, cols, &[Arc::clone(input_schema)])?
        } else {
            c.transform_down(|e| match e {
                Expr::OuterReferenceColumn(_, c) if map.contains_key(&c) => {
                    Ok(Transformed::yes(map[&c].clone()))
                }
                other => Ok(Transformed::no(other)),
            })
            .data()?
        };
        if c.contains_outer() {
            return Ok(None);
        }
        rest.push(c);
    }
    Ok(Some(Bound { values, rest }))
}

/// Walks from the root of `plan` through projections, aliases and filters
/// that hold no subqueries to an aggregate without `GROUP BY`. Returns the
/// nodes from the root to the aggregate.
fn peel_to_groupless_aggregate(plan: &LogicalPlan) -> Option<Vec<&LogicalPlan>> {
    let mut chain = vec![];
    let mut node = plan;
    loop {
        chain.push(node);
        node = match node {
            LogicalPlan::Projection(p) if !p.expr.iter().any(contains_subquery) => {
                &p.input
            }
            LogicalPlan::SubqueryAlias(a) => &a.input,
            LogicalPlan::Filter(f) if !contains_subquery(&f.predicate) => &f.input,
            LogicalPlan::Aggregate(a)
                if a.group_expr.is_empty()
                    && !a.aggr_expr.iter().any(contains_subquery) =>
            {
                return Some(chain);
            }
            _ => return None,
        };
    }
}

/// SQL evaluates an aggregate whose arguments read outer columns only in the
/// outer query, which DataFusion does not plan.
fn reject_outer_only_aggregates(agg: &Aggregate, corr: &Correlation) -> Result<()> {
    for e in &agg.aggr_expr {
        if let Expr::AggregateFunction(f) = e.clone().unalias() {
            let args = &f.params.args;
            let reads_outer = args
                .iter()
                .any(|a| expr_refs_outer(a, &corr.outer_cols, &[], &[]));
            let reads_inner = args.iter().any(|a| a.any_column_refs());
            if reads_outer && !reads_inner {
                return unsupported(format!("aggregate {e} of outer columns only"));
            }
        }
    }
    Ok(())
}

/// The value of each aggregate expression of `agg` on an empty input.
fn empty_aggregate_values(agg: &Aggregate) -> Result<Vec<ScalarValue>> {
    let n_group = agg.group_expr.len();
    agg.aggr_expr
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let (_, field) = agg.schema.qualified_field(n_group + i);
            let Expr::AggregateFunction(func) = e.clone().unalias() else {
                return unsupported(format!("aggregate expression {e}"));
            };
            func.func.default_value(field.data_type())
        })
        .collect()
}

/// Whether `predicate`, over the output of `agg`, is TRUE when every
/// aggregate has its value on an empty input. Unknown counts as TRUE.
fn true_on_empty_input(
    predicate: &Expr,
    agg: &Aggregate,
    empty_values: &[ScalarValue],
) -> Result<bool> {
    let n_group = agg.group_expr.len();
    let mut values = HashMap::new();
    for (i, empty) in empty_values.iter().enumerate() {
        values.insert(
            Column::from(agg.schema.qualified_field(n_group + i)),
            lit(empty.clone()),
        );
    }
    let Ok(expr) = substitute(predicate.clone(), &values) else {
        return Ok(true);
    };
    let info = SimplifyContext::builder()
        .with_schema(Arc::clone(&agg.schema))
        .build();
    match ExprSimplifier::new(info).simplify(expr)? {
        Expr::Literal(ScalarValue::Boolean(Some(false)), _)
        | Expr::Literal(ScalarValue::Boolean(None), _)
        | Expr::Literal(ScalarValue::Null, _) => Ok(false),
        _ => Ok(true),
    }
}

/// Whether `plan` returns at most one row per value of the domain columns
/// `cols`: it groups by them (and constants) only.
fn one_row_per_domain(plan: &LogicalPlan, cols: &[Column]) -> bool {
    match plan {
        LogicalPlan::SubqueryAlias(a) => {
            let Some(inner) = cols
                .iter()
                .map(|c| {
                    let i = a.schema.index_of_column(c).ok()?;
                    Some(Column::from(a.input.schema().qualified_field(i)))
                })
                .collect::<Option<Vec<_>>>()
            else {
                return false;
            };
            one_row_per_domain(&a.input, &inner)
        }
        LogicalPlan::Projection(p) => {
            let Some(inner) = cols
                .iter()
                .map(|c| {
                    let i = p.schema.index_of_column(c).ok()?;
                    match p.expr[i].clone().unalias() {
                        Expr::Column(c) => Some(c),
                        _ => None,
                    }
                })
                .collect::<Option<Vec<_>>>()
            else {
                return false;
            };
            one_row_per_domain(&p.input, &inner)
        }
        LogicalPlan::Filter(f) => one_row_per_domain(&f.input, cols),
        LogicalPlan::Sort(s) => one_row_per_domain(&s.input, cols),
        LogicalPlan::Aggregate(a) => a.group_expr.iter().all(|g| match g {
            Expr::Column(c) => cols.contains(c),
            Expr::Literal(..) => true,
            _ => false,
        }),
        _ => false,
    }
}

/// `EXISTS` only asks whether a row exists, which the nodes stripped here do
/// not change.
fn strip_for_exists(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Projection(p) if !p.expr.iter().any(contains_subquery) => {
            strip_for_exists(Arc::unwrap_or_clone(p.input))
        }
        LogicalPlan::Distinct(Distinct::All(input)) => {
            strip_for_exists(Arc::unwrap_or_clone(input))
        }
        LogicalPlan::Sort(Sort {
            input, fetch: None, ..
        }) => strip_for_exists(Arc::unwrap_or_clone(input)),
        LogicalPlan::Aggregate(a)
            if !a.group_expr.is_empty()
                && a.aggr_expr.is_empty()
                && !a.group_expr.iter().any(contains_subquery) =>
        {
            strip_for_exists(Arc::unwrap_or_clone(a.input))
        }
        LogicalPlan::Limit(limit)
            if matches!(limit.get_skip_type(), Ok(SkipType::Literal(0)))
                && matches!(limit.get_fetch_type(), Ok(FetchType::Literal(None)))
                    | matches!(limit.get_fetch_type(), Ok(FetchType::Literal(Some(n))) if n > 0) =>
        {
            strip_for_exists(Arc::unwrap_or_clone(limit.input))
        }
        other => other,
    }
}

/// `x IN (q)` does not depend on duplicates or order of `q`.
fn strip_for_in(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Distinct(Distinct::All(input)) => {
            strip_for_in(Arc::unwrap_or_clone(input))
        }
        LogicalPlan::Sort(Sort {
            input, fetch: None, ..
        }) => strip_for_in(Arc::unwrap_or_clone(input)),
        other => other,
    }
}

/// Replaces column references with their values in `values`, and outer
/// references with the outer columns.
fn substitute(expr: Expr, values: &HashMap<Column, Expr>) -> Result<Expr> {
    expr.transform_down(|e| match e {
        Expr::Column(c) => match values.get(&c) {
            Some(v) => Ok(Transformed::new(v.clone(), true, TreeNodeRecursion::Jump)),
            None => internal_err!("no value for column {c}"),
        },
        Expr::OuterReferenceColumn(_, c) => Ok(Transformed::yes(Expr::Column(c))),
        other => Ok(Transformed::no(other)),
    })
    .data()
}

/// Keeps the output name of the `i`-th field of `schema` for `expr`.
fn keep_name(expr: Expr, schema: &DFSchemaRef, i: usize) -> Expr {
    let (qualifier, field) = schema.qualified_field(i);
    let (expr_qualifier, expr_name) = expr.qualified_name();
    if expr_name == *field.name() && expr_qualifier.as_ref() == qualifier {
        expr
    } else {
        expr.unalias()
            .alias_qualified(qualifier.cloned(), field.name())
    }
}

/// Adds `cols` to the output expressions `exprs`, unless they are output
/// already. Returns the expressions and the output columns for `cols`. A
/// column whose name would be ambiguous next to an output of `exprs` (a
/// qualified and an unqualified field of the same name) gets a fresh name.
fn add_columns(
    mut exprs: Vec<Expr>,
    cols: &[Column],
    fresh: &mut dyn FnMut() -> String,
) -> (Vec<Expr>, Vec<Column>) {
    let mut out = Vec::with_capacity(cols.len());
    for col in cols {
        let e = Expr::Column(col.clone());
        if exprs.contains(&e) {
            out.push(col.clone());
            continue;
        }
        let clash = exprs.iter().any(|x| {
            let (q, name) = x.qualified_name();
            name == col.name && (q.is_none() || col.relation.is_none())
        });
        if clash {
            let name = fresh();
            exprs.push(e.alias(&name));
            out.push(Column::new_unqualified(name));
        } else {
            exprs.push(e);
            out.push(col.clone());
        }
    }
    (exprs, out)
}

fn add_partition_by(expr: Expr, partition: &[Expr]) -> Result<Expr> {
    expr.transform_down(|e| match e {
        Expr::WindowFunction(mut wf) => {
            for e in partition {
                if !wf.params.partition_by.contains(e) {
                    wf.params.partition_by.push(e.clone());
                }
            }
            Ok(Transformed::new(
                Expr::WindowFunction(wf),
                true,
                TreeNodeRecursion::Jump,
            ))
        }
        other => Ok(Transformed::no(other)),
    })
    .data()
}

fn alias_plan(
    plan: LogicalPlan,
    alias: impl Into<TableReference>,
) -> Result<LogicalPlan> {
    SubqueryAlias::try_new(Arc::new(plan), alias).map(LogicalPlan::SubqueryAlias)
}

/// Rewrites the column references of `expr` from the fields of `from` to the
/// fields of `to` at the same positions.
fn requalify(expr: Expr, from: &DFSchemaRef, to: &DFSchemaRef) -> Result<Expr> {
    let map: HashMap<Column, Column> = (0..from.fields().len())
        .map(|i| {
            (
                Column::from(from.qualified_field(i)),
                Column::from(to.qualified_field(i)),
            )
        })
        .collect();
    expr.transform_down(|e| match e {
        Expr::Column(c) => match map.get(&c) {
            Some(new) => Ok(Transformed::yes(Expr::Column(new.clone()))),
            None => Ok(Transformed::no(Expr::Column(c))),
        },
        other => Ok(Transformed::no(other)),
    })
    .data()
}

/// Whether evaluating `plan` twice gives the same rows: no volatile function
/// and no `LIMIT`.
fn is_deterministic(plan: &LogicalPlan) -> bool {
    !plan
        .exists(|node| {
            if matches!(node, LogicalPlan::Limit(_))
                || matches!(node, LogicalPlan::Sort(Sort { fetch: Some(_), .. }))
            {
                return Ok(true);
            }
            let mut volatile = false;
            node.apply_expressions(|e| {
                volatile |= e.is_volatile();
                Ok(TreeNodeRecursion::Continue)
            })?;
            Ok(volatile)
        })
        .expect("infallible")
}

/// An aggregate that returns its only input value, and fails on a second one:
/// a scalar subquery must not return more than one row.
#[derive(Debug, PartialEq, Eq, Hash)]
struct SingleValue {
    signature: Signature,
}

impl SingleValue {
    fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for SingleValue {
    fn name(&self) -> &str {
        "__single_value"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(SingleValueAccumulator {
            value: ScalarValue::try_from(acc_args.return_field.data_type())?,
            rows: 0,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Arc::new(args.return_field.as_ref().clone().with_name("value")),
            Arc::new(Field::new("rows", DataType::UInt64, true)),
        ])
    }
}

#[derive(Debug)]
struct SingleValueAccumulator {
    value: ScalarValue,
    rows: u64,
}

impl SingleValueAccumulator {
    fn add(&mut self, value: ScalarValue, rows: u64) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        if self.rows + rows > 1 {
            return exec_err!(
                "more than one row returned by a subquery used as an expression"
            );
        }
        self.value = value;
        self.rows += rows;
        Ok(())
    }
}

impl Accumulator for SingleValueAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = &values[0];
        for i in 0..array.len() {
            self.add(ScalarValue::try_from_array(array, i)?, 1)?;
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let rows = states[1].as_primitive::<UInt64Type>();
        for i in 0..rows.len() {
            let n = if rows.is_null(i) { 0 } else { rows.value(i) };
            self.add(ScalarValue::try_from_array(&states[0], i)?, n)?;
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            self.value.clone(),
            ScalarValue::UInt64(Some(self.rows)),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(self.value.clone())
    }

    fn size(&self) -> usize {
        size_of_val(self) + self.value.size() - size_of_val(&self.value)
    }
}

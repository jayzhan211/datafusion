// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::{Between, Expr, Like, predicate_bounds};
use crate::ValueOrLambda;
use crate::expr::{
    AggregateFunction, AggregateFunctionParams, Alias, BinaryExpr, Case, Cast, InList,
    InSubquery, Lambda, Placeholder, ScalarFunction, TryCast, Unnest, WindowFunction,
    WindowFunctionParams, in_subquery_tuple_values,
};
use crate::expr::{FieldMetadata, LambdaVariable};
use crate::higher_order_function::HigherOrderReturnFieldArgs;
use crate::type_coercion::functions::value_fields_with_higher_order_udf_and_lambdas;
use crate::type_coercion::functions::{UDFCoercionExt, fields_with_udf};
use crate::type_coercion::other::get_coerce_type_for_case_expression;
use crate::udf::ReturnFieldArgs;
use crate::{LogicalPlan, Projection, Subquery, WindowFunctionDefinition, utils};
use arrow::compute::can_cast_types;
use arrow::datatypes::FieldRef;
use arrow::datatypes::{DataType, Field};
use datafusion_common::datatype::FieldExt;
use datafusion_common::{
    Column, DataFusionError, ExprSchema, Result, ScalarValue, Spans, TableReference,
    not_impl_err, plan_datafusion_err, plan_err,
};
use datafusion_expr_common::field_rules;
use datafusion_expr_common::type_coercion::binary::BinaryTypeCoercer;
use datafusion_functions_window_common::field::WindowUDFFieldArgs;
use std::sync::Arc;

/// Trait to allow expr to typable with respect to a schema
pub trait ExprSchemable {
    /// Given a schema, return the type of the expr
    ///
    /// `CASE` expressions infer a common type from all `THEN` and `ELSE`
    /// branches without inserting casts. Incompatible branches retain a
    /// provisional type until analysis.
    fn get_type(&self, schema: &dyn ExprSchema) -> Result<DataType>;

    /// Given a schema, return the nullability of the expr
    fn nullable(&self, input_schema: &dyn ExprSchema) -> Result<bool>;

    /// Given a schema, return the expr's optional metadata
    fn metadata(&self, schema: &dyn ExprSchema) -> Result<FieldMetadata>;

    /// Convert to a field with respect to a schema
    fn to_field(
        &self,
        input_schema: &dyn ExprSchema,
    ) -> Result<(Option<TableReference>, Arc<Field>)>;

    /// Cast to a type with respect to a schema
    fn cast_to(self, cast_to_type: &DataType, schema: &dyn ExprSchema) -> Result<Expr>;

    /// Given a schema, return the type and nullability of the expr
    #[deprecated(
        since = "51.0.0",
        note = "Use `to_field().1.is_nullable` and `to_field().1.data_type()` directly instead"
    )]
    fn data_type_and_nullable(&self, schema: &dyn ExprSchema)
    -> Result<(DataType, bool)>;
}

fn scalar_arguments_for_fields(
    args: &[Expr],
    arg_fields: &[FieldRef],
) -> Vec<Option<ScalarValue>> {
    args.iter()
        .zip(arg_fields)
        .map(|(expr, field)| scalar_argument_for_field(expr, field))
        .collect()
}

fn scalar_argument_for_field(expr: &Expr, arg_field: &FieldRef) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(sv, _) => Some(
            sv.cast_to(arg_field.data_type())
                .unwrap_or_else(|_| sv.clone()),
        ),
        _ => None,
    }
}

impl ExprSchemable for Expr {
    /// Returns the [arrow::datatypes::DataType] of the expression
    /// based on [ExprSchema]
    ///
    /// Note: [`DFSchema`] implements [ExprSchema].
    ///
    /// [`DFSchema`]: datafusion_common::DFSchema
    ///
    /// # Examples
    ///
    /// Get the type of an expression that adds 2 columns. Adding an Int32
    /// and Float32 results in Float32 type
    ///
    /// ```
    /// # use arrow::datatypes::{DataType, Field};
    /// # use datafusion_common::DFSchema;
    /// # use datafusion_expr::{col, ExprSchemable};
    /// # use std::collections::HashMap;
    ///
    /// fn main() {
    ///     let expr = col("c1") + col("c2");
    ///     let schema = DFSchema::from_unqualified_fields(
    ///         vec![
    ///             Field::new("c1", DataType::Int32, true),
    ///             Field::new("c2", DataType::Float32, true),
    ///         ]
    ///         .into(),
    ///         HashMap::new(),
    ///     )
    ///     .unwrap();
    ///     assert_eq!("Float32", format!("{}", expr.get_type(&schema).unwrap()));
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// This function errors when it is not possible to compute its
    /// [arrow::datatypes::DataType].  This happens when e.g. the
    /// expression refers to a column that does not exist in the
    /// schema, or when the expression is incorrectly typed
    /// (e.g. `[utf8] + [bool]`).
    fn get_type(&self, schema: &dyn ExprSchema) -> Result<DataType> {
        Ok(output_field(self, schema)?.data_type().clone())
    }

    /// Returns the nullability of the expression based on [ExprSchema].
    ///
    /// Note: [`DFSchema`] implements [ExprSchema].
    ///
    /// [`DFSchema`]: datafusion_common::DFSchema
    ///
    /// # Errors
    ///
    /// This function errors when it is not possible to compute its
    /// nullability.  This happens when the expression refers to a
    /// column that does not exist in the schema.
    fn nullable(&self, input_schema: &dyn ExprSchema) -> Result<bool> {
        Ok(output_field(self, input_schema)?.is_nullable())
    }

    fn metadata(&self, schema: &dyn ExprSchema) -> Result<FieldMetadata> {
        Ok(FieldMetadata::from(output_field(self, schema)?.metadata()))
    }

    /// Returns the datatype and nullability of the expression based on [ExprSchema].
    ///
    /// Note: [`DFSchema`] implements [ExprSchema].
    ///
    /// [`DFSchema`]: datafusion_common::DFSchema
    ///
    /// # Errors
    ///
    /// This function errors when it is not possible to compute its
    /// datatype or nullability.
    fn data_type_and_nullable(
        &self,
        schema: &dyn ExprSchema,
    ) -> Result<(DataType, bool)> {
        let field = output_field(self, schema)?;

        Ok((field.data_type().clone(), field.is_nullable()))
    }

    /// Returns a [arrow::datatypes::Field] compatible with this expression.
    ///
    /// This function converts an expression into a field with appropriate metadata
    /// and nullability based on the expression type and context. It is the primary
    /// mechanism for determining field-level schemas.
    ///
    /// [`Self::get_type`], [`Self::nullable`] and [`Self::metadata`] read the same
    /// field, so they always agree with it. Expression kinds that also exist as a
    /// `PhysicalExpr` take their rule from [`field_rules`], shared with the
    /// physical layer.
    ///
    /// # Field Property Resolution
    ///
    /// For each expression, the following properties are determined:
    ///
    /// ## Data Type Resolution
    /// - **Column references**: Data type from input schema field
    /// - **Literals**: Data type inferred from literal value
    /// - **Aliases**: Data type inherited from the underlying expression (the aliased expression)
    /// - **Binary expressions**: Result type from type coercion rules
    /// - **Boolean expressions**: Always a boolean type
    /// - **Cast expressions**: Target data type from cast operation
    /// - **Function calls**: Return type based on function signature and argument types
    ///
    /// ## Nullability Determination
    /// - **Column references**: Inherit nullability from input schema field
    /// - **Literals**: Nullable only if literal value is NULL
    /// - **Aliases**: Inherit nullability from the underlying expression (the aliased expression)
    /// - **Binary expressions**: Nullable if either operand is nullable
    /// - **Boolean expressions**: Always non-nullable (IS NULL, EXISTS, etc.)
    /// - **Cast expressions**: type-only casts keep the input's nullability; casts to an
    ///   explicit field are nullable if the input or the target is; `TRY_CAST` always is
    /// - **Function calls**: Based on function nullability rules and input nullability
    ///
    /// ## Metadata Handling
    /// - **Column references**: Preserve original field metadata from input schema
    /// - **Literals**: Use explicitly provided metadata, otherwise empty
    /// - **Aliases**: Merge underlying expr metadata with alias-specific metadata, preferring the alias metadata
    /// - **List unnesting**: Preserve the element field metadata
    /// - **Binary expressions**: field metadata is empty
    /// - **Boolean expressions**: field metadata is empty
    /// - **Cast expressions**: Type-only casts pass through source metadata (stripping extension
    ///   type keys); casts with explicit target fields use target metadata exactly
    /// - **Scalar functions**: Generate metadata via function's [`return_field_from_args`] method,
    ///   with the default implementation returning empty field metadata
    /// - **Aggregate functions**: Generate metadata via function's [`return_field`] method,
    ///   with the default implementation returning empty field metadata
    /// - **Window functions**: field metadata follows the function's return field
    ///
    /// ## Table Reference Scoping
    /// - Establishes proper qualified field references when columns belong to specific tables
    /// - Maintains table context for accurate field resolution in multi-table scenarios
    ///
    /// So for example, a projected expression `col(c1) + col(c2)` is
    /// placed in an output field **named** col("c1 + c2")
    ///
    /// [`return_field_from_args`]: crate::ScalarUDF::return_field_from_args
    /// [`return_field`]: crate::AggregateUDF::return_field
    fn to_field(
        &self,
        schema: &dyn ExprSchema,
    ) -> Result<(Option<TableReference>, Arc<Field>)> {
        let (relation, schema_name) = self.qualified_name();
        Ok((relation, output_field(self, schema)?.renamed(&schema_name)))
    }

    /// Wraps this expression in a cast to a target [arrow::datatypes::DataType].
    ///
    /// # Errors
    ///
    /// This function errors when it is impossible to cast the
    /// expression to the target [arrow::datatypes::DataType].
    fn cast_to(self, cast_to_type: &DataType, schema: &dyn ExprSchema) -> Result<Expr> {
        let this_type = self.get_type(schema)?;
        if this_type == *cast_to_type {
            return Ok(self);
        }

        // TODO(kszucs): Most of the operations do not validate the type correctness
        // like all of the binary expressions below. Perhaps Expr should track the
        // type of the expression?

        // Special handling for struct-to-struct casts with name-based field matching
        let can_cast = match (&this_type, cast_to_type) {
            (DataType::Struct(_), DataType::Struct(_)) => {
                // Always allow struct-to-struct casts; field matching happens at runtime
                true
            }
            _ => can_cast_types(&this_type, cast_to_type),
        };

        if can_cast {
            match self {
                Expr::ScalarSubquery(subquery) => {
                    Ok(Expr::ScalarSubquery(cast_subquery(subquery, cast_to_type)?))
                }
                _ => Ok(Expr::Cast(Cast::new(Box::new(self), cast_to_type.clone()))),
            }
        } else {
            plan_err!("Cannot automatically convert {this_type} to {cast_to_type}")
        }
    }
}

/// The output field of `expr`: the one place that decides an expression's data
/// type, nullability and metadata.
///
/// [`ExprSchemable::to_field`] names this field; `get_type`, `nullable` and
/// `metadata` read it. Before, `get_type` and `nullable` were matches of their
/// own and could disagree with `to_field` (they did for `Placeholder`).
///
/// The returned field's name is not meaningful: `to_field` renames it.
/// Children are visited through this function too, so no child's schema name
/// is built. Function arguments are the exception: they go through `to_field`,
/// because a UDF sees its argument fields, names included.
///
/// Kinds that also exist as a `PhysicalExpr` take their rule from
/// [`field_rules`], which the physical layer calls too.
///
/// SKETCH (perf, measure with the `sql_planner` benchmark before landing):
/// `get_type` now also computes nullability (CASE reachability analysis, the
/// first IN list items) and `nullable` also runs binary type coercion; each
/// skipped that work before. In exchange no schema names are built here, which
/// `to_field` did for every child, and which `get_type`/`nullable` paid for
/// functions.
#[cfg_attr(feature = "recursive_protection", recursive::recursive)]
fn output_field(expr: &Expr, schema: &dyn ExprSchema) -> Result<FieldRef> {
    #[expect(deprecated)]
    let field = match expr {
        Expr::Alias(Alias {
            expr: inner,
            metadata,
            ..
        }) => {
            let field = output_field(inner, schema)?;
            let mut combined_metadata = FieldMetadata::from(field.metadata());
            if let Some(metadata) = metadata {
                combined_metadata.extend(metadata.clone());
            }
            field.with_field_metadata(&combined_metadata)
        }
        Expr::Column(c) => Arc::clone(schema.field_from_column(c)?),
        Expr::OuterReferenceColumn(field, _) | Expr::ScalarVariable(field, _) => {
            Arc::clone(field)
        }
        Expr::Literal(l, metadata) => Arc::new(
            Field::new("", l.data_type(), l.is_null())
                .with_field_metadata_opt(metadata.as_ref()),
        ),
        // BEHAVIOR CHANGE: `nullable()` now reads this field too. It answered
        // `true` for every placeholder before, while `to_field` used the field.
        Expr::Placeholder(Placeholder {
            field: Some(field), ..
        }) => Arc::clone(field),
        Expr::LambdaVariable(LambdaVariable {
            field: Some(field), ..
        }) => Arc::clone(field),
        // Not typed yet (an unresolved placeholder or lambda variable is
        // reported during planning), or not a value.
        Expr::Placeholder(_)
        | Expr::LambdaVariable(_)
        | Expr::Lambda(_)
        | Expr::GroupingSet(_) => Arc::new(Field::new("", DataType::Null, true)),
        Expr::Wildcard { .. } => Arc::new(Field::new("", DataType::Null, false)),

        // Kinds shared with the physical layer: one rule each, in field_rules.
        Expr::Negative(arg) => {
            Arc::new(field_rules::negative(output_field(arg, schema)?.as_ref()))
        }
        Expr::Not(arg) => Arc::new(field_rules::not(output_field(arg, schema)?.as_ref())),
        Expr::IsNull(_)
        | Expr::IsNotNull(_)
        | Expr::IsTrue(_)
        | Expr::IsFalse(_)
        | Expr::IsUnknown(_)
        | Expr::IsNotTrue(_)
        | Expr::IsNotFalse(_)
        | Expr::IsNotUnknown(_)
        | Expr::Exists { .. } => Arc::new(field_rules::boolean_test()),
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            let left_field = output_field(left, schema)?;
            let right_field = output_field(right, schema)?;
            let mut coercer = BinaryTypeCoercer::new(
                left_field.data_type(),
                op,
                right_field.data_type(),
            );
            coercer.set_lhs_spans(left.spans().cloned().unwrap_or_default());
            coercer.set_rhs_spans(right.spans().cloned().unwrap_or_default());
            Arc::new(field_rules::binary(
                &left_field,
                op,
                &right_field,
                coercer.get_result_type()?,
            ))
        }
        // BEHAVIOR CHANGE (the rule of #26752): an explicit nullable target
        // (`CAST(x AS UUID)`) over a NOT NULL source is nullable. It kept the
        // source's nullability before, while the physical CastExpr said
        // nullable, so aggregates over such a cast failed the schema check.
        Expr::Cast(Cast { expr: arg, field }) => Arc::new(field_rules::cast(
            output_field(arg, schema)?.as_ref(),
            field,
            !field_rules::is_type_only_cast(field),
            false,
        )),
        Expr::TryCast(TryCast { expr: arg, field }) => Arc::new(field_rules::try_cast(
            output_field(arg, schema)?.as_ref(),
            field,
            !field_rules::is_type_only_cast(field),
        )),
        Expr::InList(InList {
            expr: arg, list, ..
        }) => Arc::new(field_rules::in_list(
            output_field(arg, schema)?.as_ref(),
            list.iter().map(|item| output_field(item, schema)),
        )?),
        // SKETCH: LIKE / SIMILAR TO also exist as a PhysicalExpr (LikeExpr) with
        // the same rule today; it could move to field_rules.
        Expr::Like(Like {
            expr: arg, pattern, ..
        })
        | Expr::SimilarTo(Like {
            expr: arg, pattern, ..
        }) => Arc::new(Field::new(
            "",
            DataType::Boolean,
            arg.nullable(schema)? || pattern.nullable(schema)?,
        )),
        // SKETCH: CASE exists in both layers, but its nullability looks at the
        // WHEN predicates, so it is not a field rule. The physical CaseExpr
        // keeps its own copy ("keep in sync", case.rs).
        Expr::Case(case) => Arc::new(Field::new(
            "",
            case_type(case, schema)?,
            case_nullable(case, schema)?,
        )),

        // Logical-only kinds.
        Expr::Between(Between {
            expr: arg,
            low,
            high,
            ..
        }) => Arc::new(Field::new(
            "",
            DataType::Boolean,
            arg.nullable(schema)? || low.nullable(schema)? || high.nullable(schema)?,
        )),
        Expr::SetComparison(_) => Arc::new(Field::new("", DataType::Boolean, true)),
        Expr::InSubquery(InSubquery {
            expr: arg,
            subquery,
            ..
        }) => {
            // A multi-column `(a, b) IN (SELECT x, y ...)` is UNKNOWN when
            // any element or subquery column is NULL. The tuple is a
            // `struct` call, which itself is never NULL, so its elements
            // and every subquery column count.
            let nullable = if let Some(values) =
                in_subquery_tuple_values(arg, &subquery.subquery)?
            {
                let mut nullable = subquery
                    .subquery
                    .schema()
                    .fields()
                    .iter()
                    .any(|field| field.is_nullable());
                for value in values.iter() {
                    nullable |= value.nullable(schema)?;
                }
                nullable
            } else {
                let subquery_nullable = subquery.subquery.schema().fields().first().ok_or_else(|| {
                    plan_datafusion_err!("subquery must return exactly one column of data to compare against")
                })?.is_nullable();
                arg.nullable(schema)? | subquery_nullable
            };
            Arc::new(Field::new("", DataType::Boolean, nullable))
        }
        Expr::ScalarSubquery(subquery) => {
            let field = subquery.subquery.schema().field(0);
            Arc::new(
                field
                    .as_ref()
                    .clone()
                    .with_nullable(scalar_subquery_nullable(subquery)),
            )
        }
        Expr::Unnest(Unnest { expr: arg, .. }) => {
            // The output element is always nullable.
            let arg_type = output_field(arg, schema)?.data_type().clone();
            match &arg_type {
                DataType::List(field)
                | DataType::LargeList(field)
                | DataType::FixedSizeList(field, _)
                | DataType::ListView(field)
                | DataType::LargeListView(field) => {
                    Arc::new(field.as_ref().clone().with_nullable(true))
                }
                DataType::Struct(_) => Arc::new(Field::new("", arg_type, true)),
                DataType::Null => {
                    return not_impl_err!("unnest() does not support null yet");
                }
                _ => {
                    return plan_err!(
                        "unnest() can only be applied to array, struct and null"
                    );
                }
            }
        }

        // Functions: the UDF decides.
        Expr::WindowFunction(window_function) => {
            let WindowFunction {
                fun,
                params: WindowFunctionParams { args, .. },
            } = window_function.as_ref();

            let fields = args
                .iter()
                .map(|e| e.to_field(schema).map(|(_, f)| f))
                .collect::<Result<Vec<_>>>()?;
            match fun {
                WindowFunctionDefinition::AggregateUDF(udaf) => {
                    let new_fields = verify_function_arguments(udaf.as_ref(), &fields)?;
                    udaf.return_field(&new_fields)?
                }
                WindowFunctionDefinition::WindowUDF(udwf) => {
                    let new_fields = verify_function_arguments(udwf.as_ref(), &fields)?;
                    // A window UDF is given the output name.
                    let name = expr.schema_name().to_string();
                    udwf.field(WindowUDFFieldArgs::new(&new_fields, &name))?
                }
            }
        }
        Expr::AggregateFunction(AggregateFunction {
            func,
            params: AggregateFunctionParams { args, .. },
        }) => {
            let fields = args
                .iter()
                .map(|e| e.to_field(schema).map(|(_, f)| f))
                .collect::<Result<Vec<_>>>()?;
            let new_fields = verify_function_arguments(func.as_ref(), &fields)?;
            func.return_field(&new_fields)?
        }
        Expr::ScalarFunction(ScalarFunction { func, args }) => {
            let fields = args
                .iter()
                .map(|e| e.to_field(schema).map(|(_, f)| f))
                .collect::<Result<Vec<_>>>()?;
            let new_fields = verify_function_arguments(func.as_ref(), &fields)?;

            let arguments = scalar_arguments_for_fields(args, &new_fields);
            let argument_refs = arguments.iter().map(Option::as_ref).collect::<Vec<_>>();
            let args = ReturnFieldArgs {
                arg_fields: &new_fields,
                scalar_arguments: &argument_refs,
            };

            func.return_field_from_args(args)?
        }
        Expr::HigherOrderFunction(func) => {
            let arg_fields = func
                .args
                .iter()
                .map(|arg| match arg {
                    Expr::Lambda(Lambda { params: _, body }) => {
                        let body_field = output_field(body, schema)?;
                        // use the name of the lambda instead of just the body to help with debugging
                        Ok(ValueOrLambda::Lambda(Arc::new(Field::new(
                            arg.qualified_name().1,
                            body_field.data_type().clone(),
                            body_field.is_nullable(),
                        ))))
                    }
                    _ => Ok(ValueOrLambda::Value(arg.to_field(schema)?.1)),
                })
                .collect::<Result<Vec<_>>>()?;

            let new_fields = value_fields_with_higher_order_udf_and_lambdas(
                &arg_fields,
                func.func.as_ref(),
            )?;

            let arguments = func
                .args
                .iter()
                .map(|e| match e {
                    Expr::Literal(sv, _) => Some(sv),
                    _ => None,
                })
                .collect::<Vec<_>>();

            let args = HigherOrderReturnFieldArgs {
                arg_fields: &new_fields,
                scalar_arguments: &arguments,
            };

            func.func.return_field_from_args(args)?
        }
    };
    Ok(field)
}

/// The type of a CASE: the common type of the THEN and ELSE branches.
fn case_type(case: &Case, schema: &dyn ExprSchema) -> Result<DataType> {
    let then_types = case
        .when_then_expr
        .iter()
        .map(|(_, expr)| expr.get_type(schema))
        .collect::<Result<Vec<_>>>()?;
    let else_type = case
        .else_expr
        .as_ref()
        .map(|expr| expr.get_type(schema))
        .transpose()?;
    // If the branches have no common type, keep the old provisional
    // type. The analyzer reports the error.
    let common_type =
        get_coerce_type_for_case_expression(&then_types, else_type.as_ref());
    Ok(common_type.unwrap_or_else(|| {
        then_types
            .into_iter()
            .find(|t| !t.is_null())
            .or(else_type)
            .unwrap_or(DataType::Null)
    }))
}

/// Whether a CASE may be NULL: a reachable nullable THEN, or a nullable or
/// missing ELSE.
///
/// The physical `CaseExpr` keeps a copy of this analysis; keep them in sync.
fn case_nullable(case: &Case, input_schema: &dyn ExprSchema) -> Result<bool> {
    let nullable_then = case.when_then_expr.iter().find_map(|(w, t)| {
        let is_nullable = match t.nullable(input_schema) {
            Err(e) => return Some(Err(e)),
            Ok(n) => n,
        };

        // Branches with a then expression that is not nullable do not impact the
        // nullability of the case expression.
        if !is_nullable {
            return None;
        }

        // For case-with-expression assume all 'then' expressions are reachable
        if case.expr.is_some() {
            return Some(Ok(()));
        }

        // For branches with a nullable 'then' expression, try to determine
        // if the 'then' expression is ever reachable in the situation where
        // it would evaluate to null.
        let bounds = match predicate_bounds::evaluate_bounds(
            w,
            Some(unwrap_certainly_null_expr(t)),
            input_schema,
        ) {
            Err(e) => return Some(Err(e)),
            Ok(b) => b,
        };

        let can_be_true = match bounds.contains_value(ScalarValue::Boolean(Some(true))) {
            Err(e) => return Some(Err(e)),
            Ok(b) => b,
        };

        if !can_be_true {
            // If the derived 'when' expression can never evaluate to true, the
            // 'then' expression is not reachable when it would evaluate to NULL.
            // The most common pattern for this is `WHEN x IS NOT NULL THEN x`.
            None
        } else {
            // The branch might be taken
            Some(Ok(()))
        }
    });

    if let Some(nullable_then) = nullable_then {
        // There is at least one reachable nullable 'then' expression, so the case
        // expression itself is nullable.
        // Use `Result::map` to propagate the error from `nullable_then` if there is one.
        nullable_then.map(|_| true)
    } else if let Some(e) = &case.else_expr {
        // There are no reachable nullable 'then' expressions, so all we still need to
        // check is the 'else' expression's nullability.
        e.nullable(input_schema)
    } else {
        // CASE produces NULL if there is no `else` expr
        // (aka when none of the `when_then_exprs` match)
        Ok(true)
    }
}

/// Verify that function is invoked with correct number and type of arguments as
/// defined in `TypeSignature`.
fn verify_function_arguments<F: UDFCoercionExt>(
    function: &F,
    input_fields: &[FieldRef],
) -> Result<Vec<FieldRef>> {
    fields_with_udf(input_fields, function).map_err(|err| {
        let data_types = input_fields
            .iter()
            .map(|f| f.data_type())
            .cloned()
            .collect::<Vec<_>>();
        plan_datafusion_err!(
            "{}. {}",
            match err {
                DataFusionError::Plan(msg) => msg,
                err => err.to_string(),
            },
            utils::generate_signature_error_message(
                function.name(),
                function.signature(),
                &data_types
            )
        )
    })
}

/// Returns the innermost [Expr] that is provably null if `expr` is null.
fn unwrap_certainly_null_expr(expr: &Expr) -> &Expr {
    match expr {
        Expr::Not(e) => unwrap_certainly_null_expr(e),
        Expr::Negative(e) => unwrap_certainly_null_expr(e),
        Expr::Cast(e) => unwrap_certainly_null_expr(e.expr.as_ref()),
        _ => expr,
    }
}

/// Returns whether a scalar subquery may evaluate to NULL.
///
/// This is the case if the subquery's projected field is nullable, or if the
/// subquery may return no rows: a scalar subquery that produces no rows
/// evaluates to NULL regardless of the nullability of its projected field.
fn scalar_subquery_nullable(subquery: &Subquery) -> bool {
    subquery.subquery.schema().field(0).is_nullable() || subquery.subquery.min_rows() == 0
}

/// Cast subquery in InSubquery/ScalarSubquery to a given type.
///
/// 1. **Projection plan**: If the subquery is a projection (i.e. a SELECT statement with specific
///    columns), it casts the first expression in the projection to the target type and creates a
///    new projection with the casted expression.
/// 2. **Non-projection plan**: If the subquery isn't a projection, it adds a projection to the plan
///    with the casted first column.
pub fn cast_subquery(subquery: Subquery, cast_to_type: &DataType) -> Result<Subquery> {
    cast_subquery_columns(subquery, std::slice::from_ref(cast_to_type))
}

/// Cast the leading columns of a subquery to the given types, one per column,
/// like [`cast_subquery`] does for its first column. The result keeps only
/// the `cast_to_types.len()` leading columns.
///
/// Used by a multi-column `(a, b) IN (SELECT x, y ...)`, whose tuple elements
/// are coerced column by column.
pub fn cast_subquery_columns(
    subquery: Subquery,
    cast_to_types: &[DataType],
) -> Result<Subquery> {
    let schema = subquery.subquery.schema();
    if schema
        .fields()
        .iter()
        .zip(cast_to_types)
        .all(|(field, cast_to_type)| field.data_type() == cast_to_type)
    {
        return Ok(subquery);
    }

    let plan = subquery.subquery.as_ref();
    let new_plan = match plan {
        LogicalPlan::Projection(projection) => {
            let cast_exprs = projection
                .expr
                .iter()
                .zip(cast_to_types)
                .map(|(expr, cast_to_type)| {
                    expr.clone()
                        .cast_to(cast_to_type, projection.input.schema())
                })
                .collect::<Result<Vec<_>>>()?;
            LogicalPlan::Projection(Projection::try_new(
                cast_exprs,
                Arc::clone(&projection.input),
            )?)
        }
        _ => {
            let cast_exprs = cast_to_types
                .iter()
                .enumerate()
                .map(|(i, cast_to_type)| {
                    Expr::Column(Column::from(plan.schema().qualified_field(i)))
                        .cast_to(cast_to_type, subquery.subquery.schema())
                })
                .collect::<Result<Vec<_>>>()?;
            LogicalPlan::Projection(Projection::try_new(
                cast_exprs,
                Arc::clone(&subquery.subquery),
            )?)
        }
    };
    Ok(Subquery {
        subquery: Arc::new(new_plan),
        outer_ref_columns: subquery.outer_ref_columns,
        spans: Spans::new(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::logical_plan::builder::LogicalTableSource;
    use crate::test::function_stub::count;
    use crate::{
        LogicalPlanBuilder, Operator, and, col, in_subquery, lit, not, or,
        out_ref_col_with_metadata, scalar_subquery, when,
    };

    use arrow::datatypes::Schema;
    use datafusion_common::{DFSchema, assert_or_internal_err};

    macro_rules! test_is_expr_nullable {
        ($EXPR_TYPE:ident) => {{
            let expr = lit(ScalarValue::Null).$EXPR_TYPE();
            assert!(!expr.nullable(&MockExprSchema::new()).unwrap());
        }};
    }

    #[test]
    fn scalar_arguments_match_coerced_fields() {
        let int16_field: FieldRef = Field::new("arg", DataType::Int16, true).into();

        assert_eq!(
            scalar_argument_for_field(&lit(1_i64), &int16_field),
            Some(ScalarValue::Int16(Some(1)))
        );
        assert_eq!(
            scalar_argument_for_field(&lit(ScalarValue::Null), &int16_field),
            Some(ScalarValue::Int16(None))
        );

        let int32_list = ScalarValue::List(ScalarValue::new_list(
            &[ScalarValue::Int32(Some(1))],
            &DataType::Int32,
            true,
        ));
        let int64_list_type = DataType::new_list(DataType::Int64, true);
        let int64_list_field: FieldRef =
            Field::new("arg", int64_list_type.clone(), true).into();
        assert_eq!(
            scalar_argument_for_field(&lit(int32_list.clone()), &int64_list_field),
            Some(int32_list.cast_to(&int64_list_type).unwrap())
        );
    }

    #[test]
    fn scalar_arguments_exclude_expression_casts_and_preserve_invalid_values() {
        let int16_field: FieldRef = Field::new("arg", DataType::Int16, true).into();
        let int32_field: FieldRef = Field::new("arg", DataType::Int32, true).into();
        let explicit_cast = Expr::Cast(Cast::new(Box::new(lit(1_i64)), DataType::Int16));
        let explicit_try_cast =
            Expr::TryCast(TryCast::new(Box::new(lit(1_i64)), DataType::Int16));
        let out_of_i32_range = i64::from(i32::MAX) + 1;

        assert_eq!(
            scalar_argument_for_field(&explicit_cast, &int16_field),
            None
        );
        assert_eq!(
            scalar_argument_for_field(&explicit_try_cast, &int16_field),
            None
        );
        assert_eq!(
            scalar_argument_for_field(&lit("not an integer"), &int16_field),
            Some(ScalarValue::Utf8(Some("not an integer".to_string())))
        );
        assert_eq!(
            scalar_argument_for_field(&lit(out_of_i32_range), &int32_field),
            Some(ScalarValue::Int64(Some(out_of_i32_range)))
        );
    }

    #[test]
    fn expr_schema_nullability() {
        let cases = [
            (Operator::Eq, false, false),
            (Operator::Eq, true, true),
            (Operator::IsDistinctFrom, true, false),
            (Operator::IsNotDistinctFrom, true, false),
        ];

        for (op, input_nullable, expected) in cases {
            let expr = Expr::BinaryExpr(BinaryExpr::new(
                Box::new(col("foo")),
                op,
                Box::new(col("bar")),
            ));
            let schema = MockExprSchema::new()
                .with_data_type(DataType::Boolean)
                .with_nullable(input_nullable);

            assert_eq!(expr.nullable(&schema).unwrap(), expected, "{op}");
            assert_eq!(
                expr.to_field(&schema).unwrap().1.is_nullable(),
                expected,
                "{op}"
            );
        }

        test_is_expr_nullable!(is_null);
        test_is_expr_nullable!(is_not_null);
        test_is_expr_nullable!(is_true);
        test_is_expr_nullable!(is_not_true);
        test_is_expr_nullable!(is_false);
        test_is_expr_nullable!(is_not_false);
        test_is_expr_nullable!(is_unknown);
        test_is_expr_nullable!(is_not_unknown);
    }

    #[test]
    fn test_between_nullability() {
        let get_schema = |nullable| {
            MockExprSchema::new()
                .with_data_type(DataType::Int32)
                .with_nullable(nullable)
        };

        let expr = col("foo").between(lit(1), lit(2));
        assert!(!expr.nullable(&get_schema(false)).unwrap());
        assert!(expr.nullable(&get_schema(true)).unwrap());

        let null = lit(ScalarValue::Int32(None));

        let expr = col("foo").between(null.clone(), lit(2));
        assert!(expr.nullable(&get_schema(false)).unwrap());

        let expr = col("foo").between(lit(1), null.clone());
        assert!(expr.nullable(&get_schema(false)).unwrap());

        let expr = col("foo").between(null.clone(), null);
        assert!(expr.nullable(&get_schema(false)).unwrap());
    }

    fn assert_nullability(expr: &Expr, schema: &dyn ExprSchema, expected: bool) {
        assert_eq!(
            expr.nullable(schema).unwrap(),
            expected,
            "Nullability of '{expr}' should be {expected}"
        );
    }

    fn assert_not_nullable(expr: &Expr, schema: &dyn ExprSchema) {
        assert_nullability(expr, schema, false);
    }

    fn assert_nullable(expr: &Expr, schema: &dyn ExprSchema) {
        assert_nullability(expr, schema, true);
    }

    #[test]
    fn test_case_expression_nullability() -> Result<()> {
        let nullable_schema = MockExprSchema::new()
            .with_data_type(DataType::Int32)
            .with_nullable(true);

        let not_nullable_schema = MockExprSchema::new()
            .with_data_type(DataType::Int32)
            .with_nullable(false);

        // CASE WHEN x IS NOT NULL THEN x ELSE 0
        let e = when(col("x").is_not_null(), col("x")).otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN NOT x IS NULL THEN x ELSE 0
        let e = when(not(col("x").is_null()), col("x")).otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN X = 5 THEN x ELSE 0
        let e = when(col("x").eq(lit(5)), col("x")).otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x IS NOT NULL AND x = 5 THEN x ELSE 0
        let e = when(and(col("x").is_not_null(), col("x").eq(lit(5))), col("x"))
            .otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x = 5 AND x IS NOT NULL THEN x ELSE 0
        let e = when(and(col("x").eq(lit(5)), col("x").is_not_null()), col("x"))
            .otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x IS NOT NULL OR x = 5 THEN x ELSE 0
        let e = when(or(col("x").is_not_null(), col("x").eq(lit(5))), col("x"))
            .otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x = 5 OR x IS NOT NULL THEN x ELSE 0
        let e = when(or(col("x").eq(lit(5)), col("x").is_not_null()), col("x"))
            .otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN (x = 5 AND x IS NOT NULL) OR (x = bar AND x IS NOT NULL) THEN x ELSE 0
        let e = when(
            or(
                and(col("x").eq(lit(5)), col("x").is_not_null()),
                and(col("x").eq(col("bar")), col("x").is_not_null()),
            ),
            col("x"),
        )
        .otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x = 5 OR x IS NULL THEN x ELSE 0
        let e = when(or(col("x").eq(lit(5)), col("x").is_null()), col("x"))
            .otherwise(lit(0))?;
        assert_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x IS TRUE THEN x ELSE 0
        let e = when(col("x").is_true(), col("x")).otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x IS NOT TRUE THEN x ELSE 0
        let e = when(col("x").is_not_true(), col("x")).otherwise(lit(0))?;
        assert_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x IS FALSE THEN x ELSE 0
        let e = when(col("x").is_false(), col("x")).otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x IS NOT FALSE THEN x ELSE 0
        let e = when(col("x").is_not_false(), col("x")).otherwise(lit(0))?;
        assert_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x IS UNKNOWN THEN x ELSE 0
        let e = when(col("x").is_unknown(), col("x")).otherwise(lit(0))?;
        assert_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x IS NOT UNKNOWN THEN x ELSE 0
        let e = when(col("x").is_not_unknown(), col("x")).otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN x LIKE 'x' THEN x ELSE 0
        let e = when(col("x").like(lit("x")), col("x")).otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN 0 THEN x ELSE 0
        let e = when(lit(0), col("x")).otherwise(lit(0))?;
        assert_not_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        // CASE WHEN 1 THEN x ELSE 0
        let e = when(lit(1), col("x")).otherwise(lit(0))?;
        assert_nullable(&e, &nullable_schema);
        assert_not_nullable(&e, &not_nullable_schema);

        Ok(())
    }

    #[test]
    fn test_inlist_nullability() {
        let get_schema = |nullable| {
            MockExprSchema::new()
                .with_data_type(DataType::Int32)
                .with_nullable(nullable)
        };

        let expr = col("foo").in_list(vec![lit(1); 5], false);
        assert!(!expr.nullable(&get_schema(false)).unwrap());
        assert!(expr.nullable(&get_schema(true)).unwrap());
        // Testing nullable() returns an error.
        assert!(
            expr.nullable(&get_schema(false).with_error_on_nullable(true))
                .is_err()
        );

        let null = lit(ScalarValue::Int32(None));
        let expr = col("foo").in_list(vec![null, lit(1)], false);
        assert!(expr.nullable(&get_schema(false)).unwrap());

        // Testing on long list
        let expr = col("foo").in_list(vec![lit(1); 6], false);
        assert!(expr.nullable(&get_schema(false)).unwrap());
    }

    #[test]
    fn test_like_nullability() {
        let get_schema = |nullable| {
            MockExprSchema::new()
                .with_data_type(DataType::Utf8)
                .with_nullable(nullable)
        };

        let expr = col("foo").like(lit("bar"));
        assert!(!expr.nullable(&get_schema(false)).unwrap());
        assert!(expr.nullable(&get_schema(true)).unwrap());

        let expr = col("foo").like(lit(ScalarValue::Utf8(None)));
        assert!(expr.nullable(&get_schema(false)).unwrap());
    }

    #[test]
    fn expr_schema_data_type() {
        let expr = col("foo");
        assert_eq!(
            DataType::Utf8,
            expr.get_type(&MockExprSchema::new().with_data_type(DataType::Utf8))
                .unwrap()
        );
    }

    #[test]
    fn test_expr_metadata() {
        let mut meta = HashMap::new();
        meta.insert("bar".to_string(), "buzz".to_string());
        let meta = FieldMetadata::from(meta);
        let expr = col("foo");
        let schema = MockExprSchema::new()
            .with_data_type(DataType::Int32)
            .with_metadata(meta.clone());

        // col, alias, and cast should be metadata-preserving
        assert_eq!(meta, expr.metadata(&schema).unwrap());
        assert_eq!(meta, expr.clone().alias("bar").metadata(&schema).unwrap());
        assert_eq!(
            meta,
            expr.clone()
                .cast_to(&DataType::Int64, &schema)
                .unwrap()
                .metadata(&schema)
                .unwrap()
        );

        let schema = DFSchema::from_unqualified_fields(
            vec![meta.add_to_field(Field::new("foo", DataType::Int32, true))].into(),
            HashMap::new(),
        )
        .unwrap();

        // verify to_field method populates metadata
        assert_eq!(meta, expr.metadata(&schema).unwrap());

        // outer ref constructed by `out_ref_col_with_metadata` should be metadata-preserving
        let outer_ref = out_ref_col_with_metadata(
            DataType::Int32,
            meta.to_hashmap(),
            Column::from_name("foo"),
        );
        assert_eq!(meta, outer_ref.metadata(&schema).unwrap());
    }

    #[test]
    fn test_alias_metadata_is_preserved_in_field_metadata() {
        let schema = MockExprSchema::new().with_data_type(DataType::Int32);
        let alias_metadata = FieldMetadata::from(HashMap::from([(
            "some_key".to_string(),
            "some_value".to_string(),
        )]));

        let Expr::Alias(alias) = col("foo").alias("alias") else {
            unreachable!();
        };
        let expr = Expr::Alias(alias.with_metadata(Some(alias_metadata.clone())));

        let field = expr.to_field(&schema).unwrap().1;
        assert_eq!(
            field.metadata().get("some_key"),
            Some(&"some_value".to_string())
        );
        assert_eq!(expr.metadata(&schema).unwrap(), alias_metadata);
    }

    #[test]
    fn test_nested_alias_resolves_child_field_once() -> Result<()> {
        #[derive(Debug)]
        struct CountingSchema {
            field: FieldRef,
            lookups: std::cell::Cell<usize>,
        }

        impl ExprSchema for CountingSchema {
            fn nullable(&self, _col: &Column) -> Result<bool> {
                Ok(self.field.is_nullable())
            }

            fn field_from_column(&self, _col: &Column) -> Result<&FieldRef> {
                self.lookups.set(self.lookups.get() + 1);
                Ok(&self.field)
            }
        }

        let schema = CountingSchema {
            field: Arc::new(Field::new("value", DataType::Int32, false).with_metadata(
                HashMap::from([
                    ("override".to_string(), "base".to_string()),
                    ("keep".to_string(), "base".to_string()),
                ]),
            )),
            lookups: std::cell::Cell::new(0),
        };
        let mut expr = col("value");
        for index in 0..12 {
            let metadata = Some(FieldMetadata::from(HashMap::from([(
                "override".to_string(),
                index.to_string(),
            )])));
            expr = if index == 11 {
                expr.alias_qualified_with_metadata(
                    Some("outer"),
                    format!("alias_{index}"),
                    metadata,
                )
            } else {
                expr.alias_with_metadata(format!("alias_{index}"), metadata)
            };
        }

        let (relation, field) = expr.to_field(&schema)?;
        assert_eq!(schema.lookups.get(), 1);
        assert_eq!(relation, Some(TableReference::bare("outer")));
        assert_eq!(field.name(), "alias_11");
        assert_eq!(field.data_type(), &DataType::Int32);
        assert!(!field.is_nullable());
        assert_eq!(field.metadata().get("override"), Some(&"11".to_string()));
        assert_eq!(field.metadata().get("keep"), Some(&"base".to_string()));
        Ok(())
    }

    #[test]
    fn test_expr_placeholder() {
        let schema = MockExprSchema::new();

        let mut placeholder_meta = HashMap::new();
        placeholder_meta.insert("bar".to_string(), "buzz".to_string());
        let placeholder_meta = FieldMetadata::from(placeholder_meta);

        let expr = Expr::Placeholder(Placeholder::new_with_field(
            "".to_string(),
            Some(
                Field::new("", DataType::Utf8, true)
                    .with_metadata(placeholder_meta.to_hashmap())
                    .into(),
            ),
        ));

        let field = expr.to_field(&schema).unwrap().1;
        assert_eq!(
            (field.data_type(), field.is_nullable()),
            (&DataType::Utf8, true)
        );
        assert_eq!(placeholder_meta, expr.metadata(&schema).unwrap());

        let expr_alias = expr.alias("a placeholder by any other name");
        let expr_alias_field = expr_alias.to_field(&schema).unwrap().1;
        assert_eq!(
            (expr_alias_field.data_type(), expr_alias_field.is_nullable()),
            (&DataType::Utf8, true)
        );
        assert_eq!(placeholder_meta, expr_alias.metadata(&schema).unwrap());

        // Non-nullable placeholder field should remain non-nullable
        let expr = Expr::Placeholder(Placeholder::new_with_field(
            "".to_string(),
            Some(Field::new("", DataType::Utf8, false).into()),
        ));
        let expr_field = expr.to_field(&schema).unwrap().1;
        assert_eq!(
            (expr_field.data_type(), expr_field.is_nullable()),
            (&DataType::Utf8, false)
        );

        let expr_alias = expr.alias("a placeholder by any other name");
        let expr_alias_field = expr_alias.to_field(&schema).unwrap().1;
        assert_eq!(
            (expr_alias_field.data_type(), expr_alias_field.is_nullable()),
            (&DataType::Utf8, false)
        );
    }

    #[test]
    fn test_untyped_aliased_placeholder_in_empty_schema() {
        // A UNION arm computes the type of `$1 AS a` against the input of the
        // projection, which does not have a column `a`.
        let schema = DFSchema::empty();
        let expr = Expr::Placeholder(Placeholder::new_with_field("$1".to_string(), None))
            .alias("a");

        assert_eq!(expr.get_type(&schema).unwrap(), DataType::Null);
        assert_eq!(
            expr.to_field(&schema).unwrap().1.data_type(),
            &DataType::Null
        );
    }

    #[derive(Debug)]
    struct MockExprSchema {
        field: FieldRef,
        error_on_nullable: bool,
    }

    impl MockExprSchema {
        fn new() -> Self {
            Self {
                field: Arc::new(Field::new("mock_field", DataType::Null, false)),
                error_on_nullable: false,
            }
        }

        fn with_nullable(mut self, nullable: bool) -> Self {
            Arc::make_mut(&mut self.field).set_nullable(nullable);
            self
        }

        fn with_data_type(mut self, data_type: DataType) -> Self {
            Arc::make_mut(&mut self.field).set_data_type(data_type);
            self
        }

        fn with_error_on_nullable(mut self, error_on_nullable: bool) -> Self {
            self.error_on_nullable = error_on_nullable;
            self
        }

        fn with_metadata(mut self, metadata: FieldMetadata) -> Self {
            self.field =
                Arc::new(metadata.add_to_field(Arc::unwrap_or_clone(self.field)));
            self
        }
    }

    impl ExprSchema for MockExprSchema {
        fn nullable(&self, _col: &Column) -> Result<bool> {
            assert_or_internal_err!(!self.error_on_nullable, "nullable error");
            Ok(self.field.is_nullable())
        }

        fn field_from_column(&self, _col: &Column) -> Result<&FieldRef> {
            Ok(&self.field)
        }
    }

    /// A scan of `t`, whose single column `a` has the given nullability.
    fn scan_t(a_nullable: bool) -> LogicalPlanBuilder {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, a_nullable)]);
        let source = Arc::new(LogicalTableSource::new(Arc::new(schema)));
        LogicalPlanBuilder::scan("t", source, None).unwrap()
    }

    #[test]
    fn in_subquery_nullability() {
        // `x IN (SELECT a FROM t)` evaluates to NULL when `x` is NULL, and when `x`
        // matches no row while `a` contains a NULL. So it is nullable exactly when
        // either the compared expression or the subquery's output column is.
        let cases = [
            (false, false, false),
            (false, true, true),
            (true, false, true),
            (true, true, true),
        ];

        for (x_nullable, a_nullable, expected) in cases {
            let subquery = scan_t(a_nullable)
                .project(vec![col("a")])
                .unwrap()
                .build()
                .unwrap();
            let expr = in_subquery(col("x"), Arc::new(subquery));
            let schema = MockExprSchema::new().with_nullable(x_nullable);

            assert_eq!(expr.nullable(&schema).unwrap(), expected);
        }
    }

    #[test]
    fn in_subquery_nullability_uses_subquery_output_schema() {
        // `DISTINCT` carries no expressions of its own, but its output column is still
        // nullable, so the `IN` expression must be nullable too.
        let subquery = scan_t(true)
            .project(vec![col("a")])
            .unwrap()
            .distinct()
            .unwrap()
            .build()
            .unwrap();
        let expr = in_subquery(col("x"), Arc::new(subquery));
        assert!(expr.nullable(&MockExprSchema::new()).unwrap());

        // A computed projection's expressions reference `t.a`, which does not appear in
        // the subquery's output schema, so nullability must be read off that schema's
        // single column rather than by resolving the projection's expressions against it.
        let subquery = scan_t(false)
            .project(vec![col("a") + lit(1)])
            .unwrap()
            .build()
            .unwrap();
        let expr = in_subquery(col("x"), Arc::new(subquery));
        assert!(!expr.nullable(&MockExprSchema::new()).unwrap());
    }

    #[test]
    fn in_subquery_nullability_errors_for_no_subquery_columns() {
        let subquery = LogicalPlanBuilder::empty(false).build().unwrap();
        let expr = in_subquery(col("x"), Arc::new(subquery));

        let err = expr.nullable(&MockExprSchema::new()).unwrap_err();
        assert_eq!(
            err.strip_backtrace(),
            "Error during planning: subquery must return exactly one column of data to compare against"
        );
    }

    #[test]
    fn scalar_subquery_nullability_accounts_for_min_rows() {
        let possibly_empty = LogicalPlanBuilder::empty(false)
            .project(vec![lit(1)])
            .unwrap()
            .build()
            .unwrap();
        assert!(!possibly_empty.schema().field(0).is_nullable());

        let expr = scalar_subquery(Arc::new(possibly_empty));
        assert!(expr.nullable(&MockExprSchema::new()).unwrap());

        let field = expr.to_field(&MockExprSchema::new()).unwrap().1;
        assert_eq!(field.data_type(), &DataType::Int32);
        assert!(field.is_nullable());

        let always_one = LogicalPlanBuilder::empty(false)
            .aggregate(Vec::<Expr>::new(), vec![count(lit(1))])
            .unwrap()
            .build()
            .unwrap();
        assert!(!always_one.schema().field(0).is_nullable());

        let expr = scalar_subquery(Arc::new(always_one));
        assert!(!expr.nullable(&MockExprSchema::new()).unwrap());
        assert!(
            !expr
                .to_field(&MockExprSchema::new())
                .unwrap()
                .1
                .is_nullable()
        );
    }

    #[test]
    fn test_scalar_variable() {
        let mut meta = HashMap::new();
        meta.insert("bar".to_string(), "buzz".to_string());
        let meta = FieldMetadata::from(meta);

        let field = Field::new("foo", DataType::Int32, true);
        let field = meta.add_to_field(field);
        let field = Arc::new(field);

        let expr = Expr::ScalarVariable(field, vec!["foo".to_string()]);

        let schema = MockExprSchema::new();

        assert_eq!(meta, expr.metadata(&schema).unwrap());
    }

    #[test]
    fn test_cast_and_try_cast_extension_type_metadata() {
        use crate::expr::{Cast, TryCast};
        use arrow_schema::extension::{
            EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY,
        };

        // Helper to build either Cast or TryCast expression
        fn make_cast_expr(
            expr: Expr,
            target_field: FieldRef,
            use_try_cast: bool,
        ) -> Expr {
            if use_try_cast {
                Expr::TryCast(TryCast {
                    expr: Box::new(expr),
                    field: target_field,
                })
            } else {
                Expr::Cast(Cast {
                    expr: Box::new(expr),
                    field: target_field,
                })
            }
        }

        // Run the same test logic for both Cast and TryCast
        for use_try_cast in [false, true] {
            let cast_name = if use_try_cast { "TryCast" } else { "Cast" };

            // Create a schema with a field that has extension type metadata
            let mut source_meta = HashMap::new();
            source_meta.insert(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                "arrow.uuid".to_string(),
            );
            source_meta.insert("custom_key".to_string(), "custom_value".to_string());

            let source_field = Field::new("foo", DataType::FixedSizeBinary(16), false)
                .with_metadata(source_meta);

            let schema = MockExprSchema::new()
                .with_data_type(DataType::FixedSizeBinary(16))
                .with_metadata(FieldMetadata::from(source_field.metadata().clone()));

            // Test 1: Cast to a type without extension metadata strips extension metadata
            // but preserves non-extension metadata
            let cast_expr = make_cast_expr(
                col("foo"),
                Arc::new(Field::new("", DataType::Utf8, true)),
                use_try_cast,
            );

            let (_, result_field) = cast_expr.to_field(&schema).unwrap();
            assert!(
                result_field
                    .metadata()
                    .get(EXTENSION_TYPE_NAME_KEY)
                    .is_none(),
                "{cast_name}: Extension type name should be stripped when target has no extension metadata"
            );
            assert_eq!(
                result_field.metadata().get("custom_key"),
                Some(&"custom_value".to_string()),
                "{cast_name}: Non-extension metadata should be preserved"
            );
            if use_try_cast {
                assert!(
                    result_field.is_nullable(),
                    "TryCast result should be nullable"
                );
            }

            // Test 2: Cast to a field with explicit metadata uses target metadata exactly
            let mut target_meta = HashMap::new();
            target_meta.insert(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                "arrow.json".to_string(),
            );
            target_meta.insert(EXTENSION_TYPE_METADATA_KEY.to_string(), "{}".to_string());

            let target_field =
                Field::new("", DataType::Utf8, true).with_metadata(target_meta);

            let cast_expr =
                make_cast_expr(col("foo"), Arc::new(target_field), use_try_cast);

            let (_, result_field) = cast_expr.to_field(&schema).unwrap();
            assert_eq!(
                result_field.metadata().get(EXTENSION_TYPE_NAME_KEY),
                Some(&"arrow.json".to_string()),
                "{cast_name}: Extension type name should come from target field"
            );
            assert_eq!(
                result_field.metadata().get(EXTENSION_TYPE_METADATA_KEY),
                Some(&"{}".to_string()),
                "{cast_name}: Extension type metadata should come from target field"
            );
            assert!(
                result_field.metadata().get("custom_key").is_none(),
                "{cast_name}: Source metadata should NOT propagate when target has explicit metadata"
            );
            if use_try_cast {
                assert!(
                    result_field.is_nullable(),
                    "TryCast result should be nullable"
                );
            }
        }
    }

    #[test]
    fn case_infers_all_result_types_without_casting() -> Result<()> {
        use crate::expr::Case;
        let schema = DFSchema::empty();
        let nested = Expr::Case(Case::new(
            None,
            vec![(Box::new(lit(false)), Box::new(lit(1_i64)))],
            Some(Box::new(lit(2.5))),
        ));
        for (then_values, else_value, expected) in [
            (vec![lit(1_i64)], Some(lit(2.5)), DataType::Float64),
            (vec![lit(1_i64), lit(2.5)], None, DataType::Float64),
            (vec![lit(2.5), lit(1_i64)], None, DataType::Float64),
            (vec![lit(2.5)], Some(lit(1_i64)), DataType::Float64),
            (vec![lit(3_i64)], Some(nested), DataType::Float64),
            (vec![lit(ScalarValue::Null)], None, DataType::Null),
            // Incompatible raw types retain a provisional type for the analyzer.
            (vec![lit(true)], Some(lit("a")), DataType::Boolean),
        ] {
            let when_then = then_values
                .into_iter()
                .map(|value| (Box::new(lit(true)), Box::new(value)))
                .collect();
            let expr = Expr::Case(Case::new(None, when_then, else_value.map(Box::new)));
            assert_eq!(expr.get_type(&schema)?, expected);
        }
        Ok(())
    }
}

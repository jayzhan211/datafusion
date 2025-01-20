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

use super::{Between, Expr, Like};
use crate::expr::{
    AggregateFunction, Alias, BinaryExpr, Cast, InList, InSubquery, Placeholder,
    ScalarFunction, TryCast, Unnest, WindowFunction,
};
use crate::type_coercion::binary::get_result_type;
use crate::type_coercion::functions::{
    data_types_with_aggregate_udf, data_types_with_scalar_udf, data_types_with_window_udf,
};
use crate::{utils, LogicalPlan, Projection, Subquery, WindowFunctionDefinition};
use arrow::compute::can_cast_types;
use arrow::datatypes::{DataType, Field};
use datafusion_common::{
    not_impl_err, plan_datafusion_err, plan_err, Column, DataFusionError, ExprSchema,
    Result, TableReference,
};
use datafusion_functions_window_common::field::WindowUDFFieldArgs;
use std::collections::HashMap;
use std::sync::Arc;

/// Trait to allow expr to typable with respect to a schema
pub trait ExprSchemable {
    /// Given a schema, return the type of the expr
    fn get_type(&self, schema: &dyn ExprSchema) -> Result<DataType>;

    /// Given a schema, return the nullability of the expr
    fn nullable(&self, input_schema: &dyn ExprSchema) -> Result<bool>;

    /// Given a schema, return the expr's optional metadata
    fn metadata(&self, schema: &dyn ExprSchema) -> Result<HashMap<String, String>>;

    /// Convert to a field with respect to a schema
    fn to_field(
        &self,
        input_schema: &dyn ExprSchema,
    ) -> Result<(Option<TableReference>, Arc<Field>)>;

    /// Cast to a type with respect to a schema
    fn cast_to(self, cast_to_type: &DataType, schema: &dyn ExprSchema) -> Result<Expr>;

    /// Given a schema, return the type and nullability of the expr
    fn data_type_and_nullable(&self, schema: &dyn ExprSchema)
        -> Result<(DataType, bool)>;
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
    ///   let expr = col("c1") + col("c2");
    ///   let schema = DFSchema::from_unqualified_fields(
    ///     vec![
    ///       Field::new("c1", DataType::Int32, true),
    ///       Field::new("c2", DataType::Float32, true),
    ///       ].into(),
    ///       HashMap::new(),
    ///   ).unwrap();
    ///   assert_eq!("Float32", format!("{}", expr.get_type(&schema).unwrap()));
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
    #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
    fn get_type(&self, schema: &dyn ExprSchema) -> Result<DataType> {
        match self {
            Expr::Alias(Alias { expr, name, .. }) => match &**expr {
                Expr::Placeholder(Placeholder { data_type, .. }) => match &data_type {
                    None => schema.data_type(&Column::from_name(name)).cloned(),
                    Some(dt) => Ok(dt.clone()),
                },
                _ => expr.get_type(schema),
            },
            Expr::Negative(expr) => expr.get_type(schema),
            Expr::Column(c) => Ok(schema.data_type(c)?.clone()),
            Expr::OuterReferenceColumn(ty, _) => Ok(ty.clone()),
            Expr::ScalarVariable(ty, _) => Ok(ty.clone()),
            Expr::Literal(l) => Ok(l.data_type().clone()),
            Expr::Case(case) => {
                for (_, then_expr) in &case.when_then_expr {
                    let then_type = then_expr.get_type(schema)?;
                    if !then_type.is_null() {
                        return Ok(then_type);
                    }
                }
                case.else_expr
                    .as_ref()
                    .map_or(Ok(DataType::Null), |e| e.get_type(schema))
            }
            Expr::Cast(Cast { data_type, .. })
            | Expr::TryCast(TryCast { data_type, .. }) => Ok(data_type.clone()),
            Expr::Unnest(Unnest { expr }) => {
                let arg_data_type = expr.get_type(schema)?;
                // Unnest's output type is the inner type of the list
                match arg_data_type {
                    DataType::List(field)
                    | DataType::LargeList(field)
                    | DataType::FixedSizeList(field, _) => Ok(field.data_type().clone()),
                    DataType::Struct(_) => Ok(arg_data_type),
                    DataType::Null => {
                        not_impl_err!("unnest() does not support null yet")
                    }
                    _ => {
                        plan_err!(
                            "unnest() can only be applied to array, struct and null"
                        )
                    }
                }
            }
            Expr::ScalarFunction(ScalarFunction { func, args }) => {
                let arg_data_types = args
                    .iter()
                    .map(|e| e.get_type(schema))
                    .collect::<Result<Vec<_>>>()?;

                // Verify that function is invoked with correct number and type of arguments as defined in `TypeSignature`
                let new_data_types = data_types_with_scalar_udf(&arg_data_types, func)
                    .map_err(|err| {
                        plan_datafusion_err!(
                            "{} {}",
                            match err {
                                DataFusionError::Plan(msg) => msg,
                                err => err.to_string(),
                            },
                            utils::generate_signature_error_msg(
                                func.name(),
                                func.signature().clone(),
                                &arg_data_types,
                            )
                        )
                    })?;

                // Perform additional function arguments validation (due to limited
                // expressiveness of `TypeSignature`), then infer return type
                Ok(func.return_type_from_exprs(args, schema, &new_data_types)?)
            }
            Expr::WindowFunction(window_function) => self
                .data_type_and_nullable_with_window_function(schema, window_function)
                .map(|(return_type, _)| return_type),
            Expr::AggregateFunction(AggregateFunction { func, args, .. }) => {
                let data_types = args
                    .iter()
                    .map(|e| e.get_type(schema))
                    .collect::<Result<Vec<_>>>()?;
                let new_types = data_types_with_aggregate_udf(&data_types, func)
                    .map_err(|err| {
                        plan_datafusion_err!(
                            "{} {}",
                            match err {
                                DataFusionError::Plan(msg) => msg,
                                err => err.to_string(),
                            },
                            utils::generate_signature_error_msg(
                                func.name(),
                                func.signature().clone(),
                                &data_types
                            )
                        )
                    })?;
                Ok(func.return_type(&new_types)?)
            }
            Expr::Not(_)
            | Expr::IsNull(_)
            | Expr::Exists { .. }
            | Expr::InSubquery(_)
            | Expr::Between { .. }
            | Expr::InList { .. }
            | Expr::IsNotNull(_)
            | Expr::IsTrue(_)
            | Expr::IsFalse(_)
            | Expr::IsUnknown(_)
            | Expr::IsNotTrue(_)
            | Expr::IsNotFalse(_)
            | Expr::IsNotUnknown(_) => Ok(DataType::Boolean),
            Expr::ScalarSubquery(subquery) => {
                Ok(subquery.subquery.schema().field(0).data_type().clone())
            }
            Expr::BinaryExpr(BinaryExpr {
                ref left,
                ref right,
                ref op,
            }) => get_result_type(&left.get_type(schema)?, op, &right.get_type(schema)?),
            Expr::Like { .. } | Expr::SimilarTo { .. } => Ok(DataType::Boolean),
            Expr::Placeholder(Placeholder { data_type, .. }) => {
                if let Some(dtype) = data_type {
                    Ok(dtype.clone())
                } else {
                    // If the placeholder's type hasn't been specified, treat it as
                    // null (unspecified placeholders generate an error during planning)
                    Ok(DataType::Null)
                }
            }
            Expr::Wildcard { .. } => Ok(DataType::Null),
            Expr::GroupingSet(_) => {
                // Grouping sets do not really have a type and do not appear in projections
                Ok(DataType::Null)
            }
        }
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
        match self {
            Expr::Alias(Alias { expr, .. }) | Expr::Not(expr) | Expr::Negative(expr) => {
                expr.nullable(input_schema)
            }

            Expr::InList(InList { expr, list, .. }) => {
                // Avoid inspecting too many expressions.
                const MAX_INSPECT_LIMIT: usize = 6;
                // Stop if a nullable expression is found or an error occurs.
                let has_nullable = std::iter::once(expr.as_ref())
                    .chain(list)
                    .take(MAX_INSPECT_LIMIT)
                    .find_map(|e| {
                        e.nullable(input_schema)
                            .map(|nullable| if nullable { Some(()) } else { None })
                            .transpose()
                    })
                    .transpose()?;
                Ok(match has_nullable {
                    // If a nullable subexpression is found, the result may also be nullable.
                    Some(_) => true,
                    // If the list is too long, we assume it is nullable.
                    None if list.len() + 1 > MAX_INSPECT_LIMIT => true,
                    // All the subexpressions are non-nullable, so the result must be non-nullable.
                    _ => false,
                })
            }

            Expr::Between(Between {
                expr, low, high, ..
            }) => Ok(expr.nullable(input_schema)?
                || low.nullable(input_schema)?
                || high.nullable(input_schema)?),

            Expr::Column(c) => input_schema.nullable(c),
            Expr::OuterReferenceColumn(_, _) => Ok(true),
            Expr::Literal(value) => Ok(value.value().is_null()),
            Expr::Case(case) => {
                // This expression is nullable if any of the input expressions are nullable
                let then_nullable = case
                    .when_then_expr
                    .iter()
                    .map(|(_, t)| t.nullable(input_schema))
                    .collect::<Result<Vec<_>>>()?;
                if then_nullable.contains(&true) {
                    Ok(true)
                } else if let Some(e) = &case.else_expr {
                    e.nullable(input_schema)
                } else {
                    // CASE produces NULL if there is no `else` expr
                    // (aka when none of the `when_then_exprs` match)
                    Ok(true)
                }
            }
            Expr::Cast(Cast { expr, .. }) => expr.nullable(input_schema),
            Expr::ScalarFunction(ScalarFunction { func, args }) => {
                Ok(func.is_nullable(args, input_schema))
            }
            Expr::AggregateFunction(AggregateFunction { func, .. }) => {
                Ok(func.is_nullable())
            }
            Expr::WindowFunction(window_function) => self
                .data_type_and_nullable_with_window_function(
                    input_schema,
                    window_function,
                )
                .map(|(_, nullable)| nullable),
            Expr::ScalarVariable(_, _)
            | Expr::TryCast { .. }
            | Expr::Unnest(_)
            | Expr::Placeholder(_) => Ok(true),
            Expr::IsNull(_)
            | Expr::IsNotNull(_)
            | Expr::IsTrue(_)
            | Expr::IsFalse(_)
            | Expr::IsUnknown(_)
            | Expr::IsNotTrue(_)
            | Expr::IsNotFalse(_)
            | Expr::IsNotUnknown(_)
            | Expr::Exists { .. } => Ok(false),
            Expr::InSubquery(InSubquery { expr, .. }) => expr.nullable(input_schema),
            Expr::ScalarSubquery(subquery) => {
                Ok(subquery.subquery.schema().field(0).is_nullable())
            }
            Expr::BinaryExpr(BinaryExpr {
                ref left,
                ref right,
                ..
            }) => Ok(left.nullable(input_schema)? || right.nullable(input_schema)?),
            Expr::Like(Like { expr, pattern, .. })
            | Expr::SimilarTo(Like { expr, pattern, .. }) => {
                Ok(expr.nullable(input_schema)? || pattern.nullable(input_schema)?)
            }
            Expr::Wildcard { .. } => Ok(false),
            Expr::GroupingSet(_) => {
                // Grouping sets do not really have the concept of nullable and do not appear
                // in projections
                Ok(true)
            }
        }
    }

    fn metadata(&self, schema: &dyn ExprSchema) -> Result<HashMap<String, String>> {
        match self {
            Expr::Column(c) => Ok(schema.metadata(c)?.clone()),
            Expr::Alias(Alias { expr, .. }) => expr.metadata(schema),
            Expr::Cast(Cast { expr, .. }) => expr.metadata(schema),
            _ => Ok(HashMap::new()),
        }
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
        match self {
            Expr::Alias(Alias { expr, name, .. }) => match &**expr {
                Expr::Placeholder(Placeholder { data_type, .. }) => match &data_type {
                    None => schema
                        .data_type_and_nullable(&Column::from_name(name))
                        .map(|(d, n)| (d.clone(), n)),
                    Some(dt) => Ok((dt.clone(), expr.nullable(schema)?)),
                },
                _ => expr.data_type_and_nullable(schema),
            },
            Expr::Negative(expr) => expr.data_type_and_nullable(schema),
            Expr::Column(c) => schema
                .data_type_and_nullable(c)
                .map(|(d, n)| (d.clone(), n)),
            Expr::OuterReferenceColumn(ty, _) => Ok((ty.clone(), true)),
            Expr::ScalarVariable(ty, _) => Ok((ty.clone(), true)),
            Expr::Literal(l) => Ok((l.data_type().clone(), l.value().is_null())),
            Expr::IsNull(_)
            | Expr::IsNotNull(_)
            | Expr::IsTrue(_)
            | Expr::IsFalse(_)
            | Expr::IsUnknown(_)
            | Expr::IsNotTrue(_)
            | Expr::IsNotFalse(_)
            | Expr::IsNotUnknown(_)
            | Expr::Exists { .. } => Ok((DataType::Boolean, false)),
            Expr::ScalarSubquery(subquery) => Ok((
                subquery.subquery.schema().field(0).data_type().clone(),
                subquery.subquery.schema().field(0).is_nullable(),
            )),
            Expr::BinaryExpr(BinaryExpr {
                ref left,
                ref right,
                ref op,
            }) => {
                let left = left.data_type_and_nullable(schema)?;
                let right = right.data_type_and_nullable(schema)?;
                Ok((get_result_type(&left.0, op, &right.0)?, left.1 || right.1))
            }
            Expr::WindowFunction(window_function) => {
                self.data_type_and_nullable_with_window_function(schema, window_function)
            }
            _ => Ok((self.get_type(schema)?, self.nullable(schema)?)),
        }
    }

    /// Returns a [arrow::datatypes::Field] compatible with this expression.
    ///
    /// So for example, a projected expression `col(c1) + col(c2)` is
    /// placed in an output field **named** col("c1 + c2")
    fn to_field(
        &self,
        input_schema: &dyn ExprSchema,
    ) -> Result<(Option<TableReference>, Arc<Field>)> {
        let (relation, schema_name) = self.qualified_name();
        let (data_type, nullable) = self.data_type_and_nullable(input_schema)?;
        let field = Field::new(schema_name, data_type, nullable)
            .with_metadata(self.metadata(input_schema)?)
            .into();
        Ok((relation, field))
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

        if can_cast_types(&this_type, cast_to_type) {
            match self {
                Expr::ScalarSubquery(subquery) => {
                    Ok(Expr::ScalarSubquery(cast_subquery(subquery, cast_to_type)?))
                }
                _ => Ok(Expr::Cast(Cast::new(Box::new(self), cast_to_type.clone()))),
            }
        } else {
            plan_err!("Cannot automatically convert {this_type:?} to {cast_to_type:?}")
        }
    }
}

impl Expr {
    /// Common method for window functions that applies type coercion
    /// to all arguments of the window function to check if it matches
    /// its signature.
    ///
    /// If successful, this method returns the data type and
    /// nullability of the window function's result.
    ///
    /// Otherwise, returns an error if there's a type mismatch between
    /// the window function's signature and the provided arguments.
    fn data_type_and_nullable_with_window_function(
        &self,
        schema: &dyn ExprSchema,
        window_function: &WindowFunction,
    ) -> Result<(DataType, bool)> {
        let WindowFunction { fun, args, .. } = window_function;

        let data_types = args
            .iter()
            .map(|e| e.get_type(schema))
            .collect::<Result<Vec<_>>>()?;
        match fun {
            WindowFunctionDefinition::AggregateUDF(udaf) => {
                let new_types = data_types_with_aggregate_udf(&data_types, udaf)
                    .map_err(|err| {
                        plan_datafusion_err!(
                            "{} {}",
                            match err {
                                DataFusionError::Plan(msg) => msg,
                                err => err.to_string(),
                            },
                            utils::generate_signature_error_msg(
                                fun.name(),
                                fun.signature(),
                                &data_types
                            )
                        )
                    })?;

                let return_type = udaf.return_type(&new_types)?;
                let nullable = udaf.is_nullable();

                Ok((return_type, nullable))
            }
            WindowFunctionDefinition::WindowUDF(udwf) => {
                let new_types =
                    data_types_with_window_udf(&data_types, udwf).map_err(|err| {
                        plan_datafusion_err!(
                            "{} {}",
                            match err {
                                DataFusionError::Plan(msg) => msg,
                                err => err.to_string(),
                            },
                            utils::generate_signature_error_msg(
                                fun.name(),
                                fun.signature(),
                                &data_types
                            )
                        )
                    })?;
                let (_, function_name) = self.qualified_name();
                let field_args = WindowUDFFieldArgs::new(&new_types, &function_name);

                udwf.field(field_args)
                    .map(|field| (field.data_type().clone(), field.is_nullable()))
            }
        }
    }
}

/// Cast subquery in InSubquery/ScalarSubquery to a given type.
///
/// 1. **Projection plan**: If the subquery is a projection (i.e. a SELECT statement with specific
///    columns), it casts the first expression in the projection to the target type and creates a
///    new projection with the casted expression.
/// 2. **Non-projection plan**: If the subquery isn't a projection, it adds a projection to the plan
///    with the casted first column.
///
pub fn cast_subquery(subquery: Subquery, cast_to_type: &DataType) -> Result<Subquery> {
    if subquery.subquery.schema().field(0).data_type() == cast_to_type {
        return Ok(subquery);
    }

    let plan = subquery.subquery.as_ref();
    let new_plan = match plan {
        LogicalPlan::Projection(projection) => {
            let cast_expr = projection.expr[0]
                .clone()
                .cast_to(cast_to_type, projection.input.schema())?;
            LogicalPlan::Projection(Projection::try_new(
                vec![cast_expr],
                Arc::clone(&projection.input),
            )?)
        }
        _ => {
            let cast_expr = Expr::Column(Column::from(plan.schema().qualified_field(0)))
                .cast_to(cast_to_type, subquery.subquery.schema())?;
            LogicalPlan::Projection(Projection::try_new(
                vec![cast_expr],
                subquery.subquery,
            )?)
        }
    };
    Ok(Subquery {
        subquery: Arc::new(new_plan),
        outer_ref_columns: subquery.outer_ref_columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{col, lit};

    use datafusion_common::{internal_err, DFSchema, ScalarValue};

    macro_rules! test_is_expr_nullable {
        ($EXPR_TYPE:ident) => {{
            let expr = lit(ScalarValue::Null).$EXPR_TYPE();
            assert!(!expr.nullable(&MockExprSchema::new()).unwrap());
        }};
    }

    #[test]
    fn expr_schema_nullability() {
        let expr = col("foo").eq(lit(1));
        assert!(!expr.nullable(&MockExprSchema::new()).unwrap());
        assert!(expr
            .nullable(&MockExprSchema::new().with_nullable(true))
            .unwrap());

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
        assert!(expr
            .nullable(&get_schema(false).with_error_on_nullable(true))
            .is_err());

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
            vec![Field::new("foo", DataType::Int32, true).with_metadata(meta.clone())]
                .into(),
            HashMap::new(),
        )
        .unwrap();

        // verify to_field method populates metadata
        assert_eq!(&meta, expr.to_field(&schema).unwrap().1.metadata());
    }

    #[derive(Debug)]
    struct MockExprSchema {
        nullable: bool,
        data_type: DataType,
        error_on_nullable: bool,
        metadata: HashMap<String, String>,
    }

    impl MockExprSchema {
        fn new() -> Self {
            Self {
                nullable: false,
                data_type: DataType::Null,
                error_on_nullable: false,
                metadata: HashMap::new(),
            }
        }

        fn with_nullable(mut self, nullable: bool) -> Self {
            self.nullable = nullable;
            self
        }

        fn with_data_type(mut self, data_type: DataType) -> Self {
            self.data_type = data_type;
            self
        }

        fn with_error_on_nullable(mut self, error_on_nullable: bool) -> Self {
            self.error_on_nullable = error_on_nullable;
            self
        }

        fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
            self.metadata = metadata;
            self
        }
    }

    impl ExprSchema for MockExprSchema {
        fn nullable(&self, _col: &Column) -> Result<bool> {
            if self.error_on_nullable {
                internal_err!("nullable error")
            } else {
                Ok(self.nullable)
            }
        }

        fn data_type(&self, _col: &Column) -> Result<&DataType> {
            Ok(&self.data_type)
        }

        fn metadata(&self, _col: &Column) -> Result<&HashMap<String, String>> {
            Ok(&self.metadata)
        }

        fn data_type_and_nullable(&self, col: &Column) -> Result<(&DataType, bool)> {
            Ok((self.data_type(col)?, self.nullable(col)?))
        }
    }
}

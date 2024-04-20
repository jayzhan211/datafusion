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

use std::{any::Any, sync::Arc};

use arrow::{
    compute::SortOptions,
    datatypes::{DataType, Field, Schema},
};
use datafusion_common::DFSchema;
use datafusion_common::Result;
use datafusion_expr::{execution_props::ExecutionProps, expr::AggregateFunction};

use crate::sort_expr::PhysicalSortExpr;

use super::AggregateExpr;

/// Downcast a `Box<dyn AggregateExpr>` or `Arc<dyn AggregateExpr>`
/// and return the inner trait object as [`Any`] so
/// that it can be downcast to a specific implementation.
///
/// This method is used when implementing the `PartialEq<dyn Any>`
/// for [`AggregateExpr`] aggregation expressions and allows comparing the equality
/// between the trait objects.
pub fn down_cast_any_ref(any: &dyn Any) -> &dyn Any {
    if let Some(obj) = any.downcast_ref::<Arc<dyn AggregateExpr>>() {
        obj.as_any()
    } else if let Some(obj) = any.downcast_ref::<Box<dyn AggregateExpr>>() {
        obj.as_any()
    } else {
        any
    }
}

/// Construct corresponding fields for lexicographical ordering requirement expression
pub fn ordering_fields(
    ordering_req: &[PhysicalSortExpr],
    // Data type of each expression in the ordering requirement
    data_types: &[DataType],
) -> Vec<Field> {
    ordering_req
        .iter()
        .zip(data_types.iter())
        .map(|(sort_expr, dtype)| {
            Field::new(
                sort_expr.expr.to_string().as_str(),
                dtype.clone(),
                // Multi partitions may be empty hence field should be nullable.
                true,
            )
        })
        .collect()
}

/// Selects the sort option attribute from all the given `PhysicalSortExpr`s.
pub fn get_sort_options(ordering_req: &[PhysicalSortExpr]) -> Vec<SortOptions> {
    ordering_req.iter().map(|item| item.options).collect()
}

/// Create physical aggregate expressions from
/// logical expressions. Similar to [create_aggregate_expr_with_name_and_maybe_filter]
/// but we dont return filter clause and ordering requirements.
pub fn create_aggregate_expr(
    e: AggregateFunction,
    name: impl Into<String>,
    logical_input_schema: &DFSchema,
    physical_input_schema: &Schema,
    execution_props: &ExecutionProps,
) -> Result<Arc<dyn AggregateExpr>> {
    let AggregateFunction {
        func_def,
        distinct,
        args,
        filter: _,
        order_by,
        null_treatment,
    } = e;

    let args = create_physical_exprs(&args, logical_input_schema, execution_props)?;

    let ignore_nulls = null_treatment
        .unwrap_or(sqlparser::ast::NullTreatment::RespectNulls)
        == NullTreatment::IgnoreNulls;

    match func_def {
        AggregateFunctionDefinition::BuiltIn(fun) => {
            let physical_sort_exprs = match order_by {
                Some(exprs) => Some(create_physical_sort_exprs(
                    &exprs,
                    logical_input_schema,
                    execution_props,
                )?),
                None => None,
            };
            let ordering_reqs: Vec<PhysicalSortExpr> =
                physical_sort_exprs.clone().unwrap_or(vec![]);
            aggregates::create_aggregate_expr(
                &fun,
                distinct,
                &args,
                &ordering_reqs,
                physical_input_schema,
                name,
                ignore_nulls,
            )
        }
        AggregateFunctionDefinition::UDF(fun) => {
            let sort_exprs = order_by.clone().unwrap_or(vec![]);
            let physical_sort_exprs = match order_by {
                Some(exprs) => Some(create_physical_sort_exprs(
                    &exprs,
                    logical_input_schema,
                    execution_props,
                )?),
                None => None,
            };
            let ordering_reqs: Vec<PhysicalSortExpr> =
                physical_sort_exprs.clone().unwrap_or(vec![]);
            udaf::create_aggregate_expr(
                &fun,
                &args,
                &sort_exprs,
                &ordering_reqs,
                physical_input_schema,
                name,
                ignore_nulls,
            )
        }
        AggregateFunctionDefinition::Name(_) => {
            internal_err!("Aggregate function name should have been resolved")
        }
    }
}
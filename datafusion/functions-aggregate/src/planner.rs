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

use datafusion_common::Result;
use datafusion_expr::{
    expr::{AggregateFunction, AggregateFunctionDefinition},
    planner::{ExprPlanner, PlannerResult, RawAggregateUDF},
    utils::COUNT_STAR_EXPANSION,
    Expr,
};

pub struct AggregateUDFPlanner;

impl ExprPlanner for AggregateUDFPlanner {
    fn plan_aggregate_udf(
        &self,
        aggregate_function: RawAggregateUDF,
    ) -> Result<PlannerResult<RawAggregateUDF>> {
        let RawAggregateUDF {
            udf,
            args,
            distinct,
            filter,
            order_by,
            null_treatment,
        } = aggregate_function;

        if udf.name() == "count" && args.len() == 1 && is_wildcard(&args[0]) {
            Ok(PlannerResult::Planned(Expr::AggregateFunction(
                AggregateFunction {
                    func_def: AggregateFunctionDefinition::UDF(udf),
                    args: vec![Expr::Literal(COUNT_STAR_EXPANSION)],
                    distinct,
                    filter,
                    order_by,
                    null_treatment,
                },
            )))
        } else {
            Ok(PlannerResult::Original(RawAggregateUDF {
                udf,
                args,
                distinct,
                filter,
                order_by,
                null_treatment,
            }))
        }
    }
}

fn is_wildcard(expr: &Expr) -> bool {
    if let Expr::Wildcard { qualifier: _ } = expr {
        return true;
    }
    false
}

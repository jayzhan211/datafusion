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

use std::sync::Arc;

use async_trait::async_trait;

use datafusion_common::Result;
use datafusion_expr::{AggregateUDF, CreateFunction, LogicalPlan, ScalarUDF, WindowUDF};
use datafusion_physical_plan::ExecutionPlan;

use crate::datasource::function::TableFunctionImpl;

use super::session_state::SessionState;

/// A pluggable interface to handle `CREATE FUNCTION` statements
/// and interact with [SessionState] to registers new udf, udaf or udwf.

#[async_trait]
pub trait FunctionFactory: Sync + Send {
    /// Handles creation of user defined function specified in [CreateFunction] statement
    async fn create(
        &self,
        state: &SessionState,
        statement: CreateFunction,
    ) -> Result<RegisterFunction>;
}

/// Type of function to create
pub enum RegisterFunction {
    /// Scalar user defined function
    Scalar(Arc<ScalarUDF>),
    /// Aggregate user defined function
    Aggregate(Arc<AggregateUDF>),
    /// Window user defined function
    Window(Arc<WindowUDF>),
    /// Table user defined function
    Table(String, Arc<dyn TableFunctionImpl>),
}

/// A planner used to add extensions to DataFusion logical and physical plans.
#[async_trait]
pub trait QueryPlanner {
    /// Given a `LogicalPlan`, create an [`ExecutionPlan`] suitable for execution
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>>;
}
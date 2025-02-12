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

//! Expression utilities

use crate::{scalar::LogicalScalar, ScalarValue};

///  The value to which `COUNT(*)` is expanded to in
///  `COUNT(<constant>)` expressions
pub const COUNT_STAR_EXPANSION: ScalarValue = ScalarValue::Int64(Some(1));
pub const LOGICAL_COUNT_STAR_EXPANSION: LogicalScalar = LogicalScalar::Int64(Some(1));

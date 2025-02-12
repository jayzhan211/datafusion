
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

use std::fmt;

use arrow::temporal_conversions::{date32_to_datetime, date64_to_datetime};
use chrono::{NaiveDate, NaiveDateTime};

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Hash)]
pub struct LogicalDate32(i32);

impl LogicalDate32 {
    // Mirror ArrowTemporalType Array function
    pub fn value_as_datetime(&self) -> Option<NaiveDateTime> {
        date32_to_datetime(self.0)
    }

    // Mirror ArrowTemporalType Array function
    pub fn value_as_date(&self) -> Option<NaiveDate> {
        self.value_as_datetime().map(|datetime| datetime.date())
    }
}

impl fmt::Display for LogicalDate32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i32> for LogicalDate32 {
    fn from(value: i32) -> Self {
        LogicalDate32(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Hash)]
pub struct LogicalDate64(i64);

impl LogicalDate64 {
    // Mirror ArrowTemporalType Array function
    pub fn value_as_datetime(&self) -> Option<NaiveDateTime> {
        date64_to_datetime(self.0)
    }

    // Mirror ArrowTemporalType Array function
    pub fn value_as_date(&self) -> Option<NaiveDate> {
        self.value_as_datetime().map(|datetime| datetime.date())
    }
}

impl fmt::Display for LogicalDate64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i64> for LogicalDate64 {
    fn from(value: i64) -> Self {
        LogicalDate64(value)
    }
}
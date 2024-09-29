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
use datafusion_common::Result;

use arrow::{array::{ArrayRef, AsArray}, datatypes::{DataType, Schema}};
use datafusion_common::internal_err;

// Minimum set as Datafusion Native type
#[derive(Clone, PartialEq, Eq)]
pub enum DatafusionNativeType {
    Int32,
    UInt64,
    String,
    Float32,
    Float64,
    FixedSizeList(Box<DatafusionNativeType>, usize),
}

pub trait LogicalDataType {
    fn name(&self) -> &str;
    fn native_type(&self) -> DatafusionNativeType;
}

pub enum DatafusionType {
    Builtin(DatafusionNativeType),
    Extension(Arc<dyn LogicalDataType>)
}

// This allows us to treat `DatafusionNativeType` as Trait `LogicalDataType`, 
// there might be better design but the idea is to get `DatafusionNativeType`
// for both UserDefinedType and BuiltinType
impl LogicalDataType for DatafusionNativeType {
    fn native_type(&self) -> DatafusionNativeType {
        match self {
            DatafusionNativeType::Int32 => DatafusionNativeType::Int32,
            _ => self.clone()
        }
    }

    fn name(&self) -> &str {
        match self {
            DatafusionNativeType::Int32 => "i32",
            DatafusionNativeType::Float32 => "f32",
            _ => todo!("")
        }
    }
}

fn is_numeric(logical_data_type: &Arc<dyn LogicalDataType>) -> bool {
    matches!(logical_data_type.native_type(), DatafusionNativeType::Int32 | DatafusionNativeType::UInt64) // and more
}

// function where we only care about the Logical type
fn logical_func(logical_type: Arc<dyn LogicalDataType>) {

    if is_numeric(&logical_type) {
        // process it as numeric
    }

    // For user-defined type, maybe there is another way to differentiate the type instead by name
    match logical_type.name() {
        "json" => { 
            // process json 
        },
        "geo" => {
            // process geo
        },
        _ => todo!("")
    }
}


// function where we care about the internal physical type itself
fn physical_func(logical_type: Arc<dyn LogicalDataType>, array: ArrayRef, schema: Schema) -> Result<()>{
    let data_type_in_schema = schema.field(0).data_type();
    let actual_native_type = data_type_in_schema.logical_type();

    if logical_type.native_type() != actual_native_type {
        return internal_err!("logical type mismatches with the actual data type in schema & array")
    }

    // For Json type, we know the internal physical type is String, so we need to ensure the
    // Array is able to cast to StringArray variant, we can check the schema.
    match logical_type.native_type() {
        DatafusionNativeType::String => {
            match data_type_in_schema {
                DataType::Utf8 => {
                    let string_arr = array.as_string::<i32>();
                    Ok(())
                }
                DataType::Utf8View => {
                    let string_view_arr = array.as_string_view();
                    Ok(())
                }
                _ => todo!("")
            }
        }
        _ => todo!("")
    }
}

pub struct JsonType {}

impl LogicalDataType for JsonType {
    fn native_type(&self) -> DatafusionNativeType {
        DatafusionNativeType::String
    }
    fn name(&self) -> &str {
        "json"
    }
}

pub struct GeoType {
    n_dim: usize
}

impl LogicalDataType for GeoType {
    fn native_type(&self) -> DatafusionNativeType {
        DatafusionNativeType::FixedSizeList(Box::new(DatafusionNativeType::Float64), self.n_dim)
    }
    fn name(&self) -> &str {
        "geo"
    }
}

pub trait PhysicalType {
    fn logical_type(&self) -> DatafusionNativeType;
}

impl PhysicalType for DataType {
    fn logical_type(&self) -> DatafusionNativeType {
        match self {
            DataType::Int32 => DatafusionNativeType::Int32,
            DataType::FixedSizeList(f, n) => {
                DatafusionNativeType::FixedSizeList(Box::new(f.data_type().logical_type()), *n as usize)
            }
            _ => todo!("")
        }
    }
}
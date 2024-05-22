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

//! # Median

use crate::aggregate::utils::{down_cast_any_ref, Hashable};
use crate::expressions::format_state_name;
use crate::{AggregateExpr, PhysicalExpr};
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::{DataType, Field};
use arrow_array::cast::AsArray;
use arrow_array::{downcast_integer, ArrowNativeTypeOp, ArrowNumericType};
use arrow_buffer::ArrowNativeType;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::Accumulator;
use std::any::Any;
use std::collections::HashSet;
use std::fmt::Formatter;
use std::sync::Arc;

/// MEDIAN aggregate expression. If using the non-distinct variation, then this uses a
/// lot of memory because all values need to be stored in memory before a result can be
/// computed. If an approximation is sufficient then APPROX_MEDIAN provides a much more
/// efficient solution.
///
/// If using the distinct variation, the memory usage will be similarly high if the
/// cardinality is high as it stores all distinct values in memory before computing the
/// result, but if cardinality is low then memory usage will also be lower.
#[derive(Debug)]
pub struct Median {
    name: String,
    expr: Arc<dyn PhysicalExpr>,
    data_type: DataType,
    distinct: bool,
}

impl Median {
    /// Create a new MEDIAN aggregate function
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        name: impl Into<String>,
        data_type: DataType,
        distinct: bool,
    ) -> Self {
        Self {
            name: name.into(),
            expr,
            data_type,
            distinct,
        }
    }
}

impl AggregateExpr for Median {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn field(&self) -> Result<Field> {
        Ok(Field::new(&self.name, self.data_type.clone(), true))
    }

    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        use arrow_array::types::*;
        macro_rules! helper {
            ($t:ty, $dt:expr) => {
                if self.distinct {
                    Ok(Box::new(DistinctMedianAccumulator::<$t> {
                        data_type: $dt.clone(),
                        distinct_values: HashSet::new(),
                    }))
                } else {
                    Ok(Box::new(MedianAccumulator::<$t> {
                        data_type: $dt.clone(),
                        all_values: vec![],
                    }))
                }
            };
        }
        let dt = &self.data_type;
        downcast_integer! {
            dt => (helper, dt),
            DataType::Float16 => helper!(Float16Type, dt),
            DataType::Float32 => helper!(Float32Type, dt),
            DataType::Float64 => helper!(Float64Type, dt),
            DataType::Decimal128(_, _) => helper!(Decimal128Type, dt),
            DataType::Decimal256(_, _) => helper!(Decimal256Type, dt),
            _ => Err(DataFusionError::NotImplemented(format!(
                "MedianAccumulator not supported for {} with {}",
                self.name(),
                self.data_type
            ))),
        }
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        //Intermediate state is a list of the elements we have collected so far
        let field = Field::new("item", self.data_type.clone(), true);
        let data_type = DataType::List(Arc::new(field));
        let state_name = if self.distinct {
            "distinct_median"
        } else {
            "median"
        };

        Ok(vec![Field::new(
            format_state_name(&self.name, state_name),
            data_type,
            true,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl PartialEq<dyn Any> for Median {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| {
                self.name == x.name
                    && self.data_type == x.data_type
                    && self.expr.eq(&x.expr)
                    && self.distinct == x.distinct
            })
            .unwrap_or(false)
    }
}

/// The median accumulator accumulates the raw input values
/// as `ScalarValue`s
///
/// The intermediate state is represented as a List of scalar values updated by
/// `merge_batch` and a `Vec` of `ArrayRef` that are converted to scalar values
/// in the final evaluation step so that we avoid expensive conversions and
/// allocations during `update_batch`.
struct MedianAccumulator<T: ArrowNumericType> {
    data_type: DataType,
    all_values: Vec<T::Native>,
}

impl<T: ArrowNumericType> std::fmt::Debug for MedianAccumulator<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MedianAccumulator({})", self.data_type)
    }
}

impl<T: ArrowNumericType> Accumulator for MedianAccumulator<T> {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let all_values = self
            .all_values
            .iter()
            .map(|x| ScalarValue::new_primitive::<T>(Some(*x), &self.data_type))
            .collect::<Result<Vec<_>>>()?;

        let arr = ScalarValue::new_list(&all_values, &self.data_type);
        Ok(vec![ScalarValue::List(arr)])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = values[0].as_primitive::<T>();
        self.all_values.reserve(values.len() - values.null_count());
        self.all_values.extend(values.iter().flatten());
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let array = states[0].as_list::<i32>();
        for v in array.iter().flatten() {
            self.update_batch(&[v])?
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let d = std::mem::take(&mut self.all_values);
        let median = calculate_median::<T>(d);
        ScalarValue::new_primitive::<T>(median, &self.data_type)
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self.all_values.capacity() * std::mem::size_of::<T::Native>()
    }
}

/// The distinct median accumulator accumulates the raw input values
/// as `ScalarValue`s
///
/// The intermediate state is represented as a List of scalar values updated by
/// `merge_batch` and a `Vec` of `ArrayRef` that are converted to scalar values
/// in the final evaluation step so that we avoid expensive conversions and
/// allocations during `update_batch`.
struct DistinctMedianAccumulator<T: ArrowNumericType> {
    data_type: DataType,
    distinct_values: HashSet<Hashable<T::Native>>,
}

impl<T: ArrowNumericType> std::fmt::Debug for DistinctMedianAccumulator<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "DistinctMedianAccumulator({})", self.data_type)
    }
}

impl<T: ArrowNumericType> Accumulator for DistinctMedianAccumulator<T> {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let all_values = self
            .distinct_values
            .iter()
            .map(|x| ScalarValue::new_primitive::<T>(Some(x.0), &self.data_type))
            .collect::<Result<Vec<_>>>()?;

        let arr = ScalarValue::new_list(&all_values, &self.data_type);
        Ok(vec![ScalarValue::List(arr)])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let array = values[0].as_primitive::<T>();
        match array.nulls().filter(|x| x.null_count() > 0) {
            Some(n) => {
                for idx in n.valid_indices() {
                    self.distinct_values.insert(Hashable(array.value(idx)));
                }
            }
            None => array.values().iter().for_each(|x| {
                self.distinct_values.insert(Hashable(*x));
            }),
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let array = states[0].as_list::<i32>();
        for v in array.iter().flatten() {
            self.update_batch(&[v])?
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let d = std::mem::take(&mut self.distinct_values)
            .into_iter()
            .map(|v| v.0)
            .collect::<Vec<_>>();
        let median = calculate_median::<T>(d);
        ScalarValue::new_primitive::<T>(median, &self.data_type)
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self.distinct_values.capacity() * std::mem::size_of::<T::Native>()
    }
}

fn calculate_median<T: ArrowNumericType>(
    mut values: Vec<T::Native>,
) -> Option<T::Native> {
    let cmp = |x: &T::Native, y: &T::Native| x.compare(*y);

    let len = values.len();
    if len == 0 {
        None
    } else if len % 2 == 0 {
        let (low, high, _) = values.select_nth_unstable_by(len / 2, cmp);
        let (_, low, _) = low.select_nth_unstable_by(low.len() - 1, cmp);
        let median = low.add_wrapping(*high).div_wrapping(T::Native::usize_as(2));
        Some(median)
    } else {
        let (_, median, _) = values.select_nth_unstable_by(len / 2, cmp);
        Some(*median)
    }
}

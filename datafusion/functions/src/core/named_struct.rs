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

use arrow::array::StructArray;
use arrow::datatypes::{DataType, Field, Fields};
use datafusion_common::{exec_err, internal_err, HashSet, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Documentation, ReturnInfo, ReturnTypeArgs};
use datafusion_expr::{ScalarUDFImpl, Signature, Volatility};
use datafusion_macros::user_doc;
use std::any::Any;
use std::sync::Arc;

/// Put values in a struct array.
fn named_struct_expr(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    // Do not accept 0 arguments.
    if args.is_empty() {
        return exec_err!(
            "named_struct requires at least one pair of arguments, got 0 instead"
        );
    }

    if args.len() % 2 != 0 {
        return exec_err!(
            "named_struct requires an even number of arguments, got {} instead",
            args.len()
        );
    }

    let (names, values): (Vec<_>, Vec<_>) = args
        .chunks_exact(2)
        .enumerate()
        .map(|(i, chunk)| {
            let name_column = &chunk[0];
            let name = match name_column {
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(name_scalar))) => {
                    name_scalar
                }
                // TODO: Implement Display for ColumnarValue
                _ => {
                    return exec_err!(
                    "named_struct even arguments must be string literals at position {}",
                    i * 2
                )
                }
            };

            Ok((name, chunk[1].clone()))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();

    {
        // Check to enforce the uniqueness of struct field name
        let mut unique_field_names = HashSet::new();
        for name in names.iter() {
            if unique_field_names.contains(name) {
                return exec_err!(
                    "named_struct requires unique field names. Field {name} is used more than once."
                );
            }
            unique_field_names.insert(name);
        }
    }

    let fields: Fields = names
        .into_iter()
        .zip(&values)
        .map(|(name, value)| Arc::new(Field::new(name, value.data_type().clone(), true)))
        .collect::<Vec<_>>()
        .into();

    let arrays = ColumnarValue::values_to_arrays(&values)?;

    let struct_array = StructArray::new(fields, arrays, None);
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

#[user_doc(
    doc_section(label = "Struct Functions"),
    description = "Returns an Arrow struct using the specified name and input expressions pairs.",
    syntax_example = "named_struct(expression1_name, expression1_input[, ..., expression_n_name, expression_n_input])",
    sql_example = r#"
For example, this query converts two columns `a` and `b` to a single column with
a struct type of fields `field_a` and `field_b`:
```sql
> select * from t;
+---+---+
| a | b |
+---+---+
| 1 | 2 |
| 3 | 4 |
+---+---+
> select named_struct('field_a', a, 'field_b', b) from t;
+-------------------------------------------------------+
| named_struct(Utf8("field_a"),t.a,Utf8("field_b"),t.b) |
+-------------------------------------------------------+
| {field_a: 1, field_b: 2}                              |
| {field_a: 3, field_b: 4}                              |
+-------------------------------------------------------+
```"#,
    argument(
        name = "expression_n_name",
        description = "Name of the column field. Must be a constant string."
    ),
    argument(
        name = "expression_n_input",
        description = "Expression to include in the output struct. Can be a constant, column, or function, and any combination of arithmetic or string operators."
    )
)]
#[derive(Debug)]
pub struct NamedStructFunc {
    signature: Signature,
}

impl Default for NamedStructFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl NamedStructFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for NamedStructFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "named_struct"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("named_struct: return_type called instead of return_type_from_args")
    }

    fn return_type_from_args(&self, args: ReturnTypeArgs) -> Result<ReturnInfo> {
        // do not accept 0 arguments.
        if args.arguments.is_empty() {
            return exec_err!(
                "named_struct requires at least one pair of arguments, got 0 instead"
            );
        }

        if args.arguments.len() % 2 != 0 {
            return exec_err!(
                "named_struct requires an even number of arguments, got {} instead",
                args.arguments.len()
            );
        }

        let names = args
            .arguments
            .iter()
            .enumerate()
            .step_by(2)
            .map(|(i, x)| match x {
                Some(ScalarValue::Utf8(Some(name))) if !name.is_empty() => Ok(name),
                Some(ScalarValue::Utf8(Some(_))) => {
                    exec_err!(
                        "{} requires {i}-th (0-indexed) field name as non-empty string",
                        self.name()
                    )
                }
                _ => {
                    exec_err!(
                        "{} requires {i}-th (0-indexed) field name as constant string",
                        self.name()
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let types = args.arg_types.iter().skip(1).step_by(2).collect::<Vec<_>>();

        let return_fields = names
            .into_iter()
            .zip(types.into_iter())
            .map(|(name, data_type)| Ok(Field::new(name, data_type.to_owned(), true)))
            .collect::<Result<Vec<Field>>>()?;

        Ok(ReturnInfo::new_nullable(DataType::Struct(Fields::from(
            return_fields,
        ))))
    }

    fn invoke_batch(
        &self,
        args: &[ColumnarValue],
        _number_rows: usize,
    ) -> Result<ColumnarValue> {
        named_struct_expr(args)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

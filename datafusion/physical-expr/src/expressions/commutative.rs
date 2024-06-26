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

use std::hash::{Hash, Hasher};
use std::{any::Any, sync::Arc};

use crate::expressions::datum::{apply, apply_cmp};
use crate::intervals::cp_solver::{propagate_arithmetic, propagate_comparison};
use crate::physical_expr::down_cast_any_ref;
use crate::PhysicalExpr;

use arrow::array::*;
use arrow::compute::kernels::boolean::{and_kleene, not, or_kleene};
use arrow::compute::kernels::cmp::*;
use arrow::compute::kernels::comparison::{
    regexp_is_match_utf8, regexp_is_match_utf8_scalar,
};
use arrow::compute::kernels::concat_elements::concat_elements_utf8;
use arrow::compute::{cast, ilike, like, nilike, nlike};
use arrow::datatypes::*;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{internal_err, Result, ScalarValue};
use datafusion_expr::interval_arithmetic::{apply_operator, Interval};
use datafusion_expr::sort_properties::ExprProperties;
use datafusion_expr::type_coercion::binary::get_result_type_from_commutative;
use datafusion_expr::{ColumnarValue, Operator};

use itertools::Itertools;
use kernels::{
    bitwise_and_dyn, bitwise_and_dyn_scalar, bitwise_or_dyn, bitwise_or_dyn_scalar,
    bitwise_shift_left_dyn, bitwise_shift_left_dyn_scalar, bitwise_shift_right_dyn,
    bitwise_shift_right_dyn_scalar, bitwise_xor_dyn, bitwise_xor_dyn_scalar,
};

use super::binary::kernels;

/// Create physical commutative expression
pub fn create_commutative_expr(
    exprs: Vec<Arc<dyn PhysicalExpr>>,
    op: Operator,
) -> Result<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(CommutativeExpr::new(exprs, op)))
}

/// Commutative expression
#[derive(Debug, Hash, Clone)]
pub struct CommutativeExpr {
    exprs: Vec<Arc<dyn PhysicalExpr>>,
    op: Operator,
}

impl CommutativeExpr {
    /// Create new binary expression
    pub fn new(exprs: Vec<Arc<dyn PhysicalExpr>>, op: Operator) -> Self {
        Self { exprs, op }
    }

    /// Get the operator for this binary expression
    pub fn op(&self) -> &Operator {
        &self.op
    }
}

impl std::fmt::Display for CommutativeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Put parentheses around child binary expressions so that we can see the difference
        // between `(a OR b) AND c` and `a OR (b AND c)`. We only insert parentheses when needed,
        // based on operator precedence. For example, `(a AND b) OR c` and `a AND b OR c` are
        // equivalent and the parentheses are not necessary.

        fn write_child(
            f: &mut std::fmt::Formatter,
            expr: &dyn PhysicalExpr,
            precedence: u8,
        ) -> std::fmt::Result {
            if let Some(child) = expr.as_any().downcast_ref::<CommutativeExpr>() {
                let p = child.op.precedence();
                if p == 0 || p < precedence {
                    write!(f, "({child})")?;
                } else {
                    write!(f, "{child}")?;
                }
            } else {
                write!(f, "{expr}")?;
            }

            Ok(())
        }

        let precedence = self.op.precedence();

        for (idx, e) in self.exprs.iter().enumerate() {
            if idx > 0 {
                write!(f, " {} ", self.op)?;
            }

            write_child(f, e.as_ref(), precedence)?;
        }

        Ok(())
    }
}

/// Invoke a compute kernel on a pair of binary data arrays
macro_rules! compute_utf8_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $DT:ident) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast left side array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast right side array");
        Ok(Arc::new(paste::expr! {[<$OP _utf8>]}(&ll, &rr)?))
    }};
}

macro_rules! binary_string_array_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        match $LEFT.data_type() {
            DataType::Utf8 => compute_utf8_op!($LEFT, $RIGHT, $OP, StringArray),
            DataType::LargeUtf8 => compute_utf8_op!($LEFT, $RIGHT, $OP, LargeStringArray),
            other => internal_err!(
                "Data type {:?} not supported for binary operation '{}' on string arrays",
                other, stringify!($OP)
            ),
        }
    }};
}

/// Invoke a boolean kernel on a pair of arrays
macro_rules! boolean_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        let ll = as_boolean_array($LEFT).expect("boolean_op failed to downcast array");
        let rr = as_boolean_array($RIGHT).expect("boolean_op failed to downcast array");
        Ok(Arc::new($OP(&ll, &rr)?))
    }};
}

macro_rules! binary_string_array_flag_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $NOT:expr, $FLAG:expr) => {{
        match $LEFT.data_type() {
            DataType::Utf8 => {
                compute_utf8_flag_op!($LEFT, $RIGHT, $OP, StringArray, $NOT, $FLAG)
            }
            DataType::LargeUtf8 => {
                compute_utf8_flag_op!($LEFT, $RIGHT, $OP, LargeStringArray, $NOT, $FLAG)
            }
            other => internal_err!(
                "Data type {:?} not supported for binary_string_array_flag_op operation '{}' on string array",
                other, stringify!($OP)
            ),
        }
    }};
}

/// Invoke a compute kernel on a pair of binary data arrays with flags
macro_rules! compute_utf8_flag_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $ARRAYTYPE:ident, $NOT:expr, $FLAG:expr) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$ARRAYTYPE>()
            .expect("compute_utf8_flag_op failed to downcast array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<$ARRAYTYPE>()
            .expect("compute_utf8_flag_op failed to downcast array");

        let flag = if $FLAG {
            Some($ARRAYTYPE::from(vec!["i"; ll.len()]))
        } else {
            None
        };
        let mut array = paste::expr! {[<$OP _utf8>]}(&ll, &rr, flag.as_ref())?;
        if $NOT {
            array = not(&array).unwrap();
        }
        Ok(Arc::new(array))
    }};
}

macro_rules! binary_string_array_flag_op_scalar {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $NOT:expr, $FLAG:expr) => {{
        let result: Result<Arc<dyn Array>> = match $LEFT.data_type() {
            DataType::Utf8 => {
                compute_utf8_flag_op_scalar!($LEFT, $RIGHT, $OP, StringArray, $NOT, $FLAG)
            }
            DataType::LargeUtf8 => {
                compute_utf8_flag_op_scalar!($LEFT, $RIGHT, $OP, LargeStringArray, $NOT, $FLAG)
            }
            other => internal_err!(
                "Data type {:?} not supported for binary_string_array_flag_op_scalar operation '{}' on string array",
                other, stringify!($OP)
            ),
        };
        Some(result)
    }};
}

/// Invoke a compute kernel on a data array and a scalar value with flag
macro_rules! compute_utf8_flag_op_scalar {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $ARRAYTYPE:ident, $NOT:expr, $FLAG:expr) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$ARRAYTYPE>()
            .expect("compute_utf8_flag_op_scalar failed to downcast array");

        if let ScalarValue::Utf8(Some(string_value))|ScalarValue::LargeUtf8(Some(string_value)) = $RIGHT {
            let flag = if $FLAG { Some("i") } else { None };
            let mut array =
                paste::expr! {[<$OP _utf8_scalar>]}(&ll, &string_value, flag)?;
            if $NOT {
                array = not(&array).unwrap();
            }
            Ok(Arc::new(array))
        } else {
            internal_err!(
                "compute_utf8_flag_op_scalar failed to cast literal value {} for operation '{}'",
                $RIGHT, stringify!($OP)
            )
        }
    }};
}

impl PhysicalExpr for CommutativeExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        let data_types: Result<Vec<DataType>> = self
            .exprs
            .iter()
            .map(|e| e.data_type(input_schema))
            .collect();

        get_result_type_from_commutative(data_types?, &self.op)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.exprs.iter().try_fold(false, |acc, e| {
            if acc {
                // If any previous expression is nullable, short-circuit and return true
                Ok(true)
            } else {
                // Otherwise, check the current expression
                e.nullable(input_schema)
            }
        })
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let result_type = self.data_type(&batch.schema())?;
        let res = self
            .exprs
            .iter()
            .try_fold(self.exprs[0].evaluate(batch)?, |acc, e| {
                evaluate_inner(acc, e, batch, self.op, &result_type)
            });
        res
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.exprs.iter().collect_vec()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(CommutativeExpr::new(children, self.op)))
    }

    fn evaluate_bounds(&self, children: &[&Interval]) -> Result<Interval> {
        // Get children intervals:
        let left_interval = children[0];
        let right_interval = children[1];
        // Calculate current node's interval:
        apply_operator(&self.op, left_interval, right_interval)
    }

    fn propagate_constraints(
        &self,
        interval: &Interval,
        children: &[&Interval],
    ) -> Result<Option<Vec<Interval>>> {
        // Get children intervals.
        let left_interval = children[0];
        let right_interval = children[1];

        if self.op.eq(&Operator::And) {
            if interval.eq(&Interval::CERTAINLY_TRUE) {
                // A certainly true logical conjunction can only derive from possibly
                // true operands. Otherwise, we prove infeasability.
                Ok((!left_interval.eq(&Interval::CERTAINLY_FALSE)
                    && !right_interval.eq(&Interval::CERTAINLY_FALSE))
                .then(|| vec![Interval::CERTAINLY_TRUE, Interval::CERTAINLY_TRUE]))
            } else if interval.eq(&Interval::CERTAINLY_FALSE) {
                // If the logical conjunction is certainly false, one of the
                // operands must be false. However, it's not always possible to
                // determine which operand is false, leading to different scenarios.

                // If one operand is certainly true and the other one is uncertain,
                // then the latter must be certainly false.
                if left_interval.eq(&Interval::CERTAINLY_TRUE)
                    && right_interval.eq(&Interval::UNCERTAIN)
                {
                    Ok(Some(vec![
                        Interval::CERTAINLY_TRUE,
                        Interval::CERTAINLY_FALSE,
                    ]))
                } else if right_interval.eq(&Interval::CERTAINLY_TRUE)
                    && left_interval.eq(&Interval::UNCERTAIN)
                {
                    Ok(Some(vec![
                        Interval::CERTAINLY_FALSE,
                        Interval::CERTAINLY_TRUE,
                    ]))
                }
                // If both children are uncertain, or if one is certainly false,
                // we cannot conclusively refine their intervals. In this case,
                // propagation does not result in any interval changes.
                else {
                    Ok(Some(vec![]))
                }
            } else {
                // An uncertain logical conjunction result can not shrink the
                // end-points of its children.
                Ok(Some(vec![]))
            }
        } else if self.op.eq(&Operator::Or) {
            if interval.eq(&Interval::CERTAINLY_FALSE) {
                // A certainly false logical conjunction can only derive from certainly
                // false operands. Otherwise, we prove infeasability.
                Ok((!left_interval.eq(&Interval::CERTAINLY_TRUE)
                    && !right_interval.eq(&Interval::CERTAINLY_TRUE))
                .then(|| vec![Interval::CERTAINLY_FALSE, Interval::CERTAINLY_FALSE]))
            } else if interval.eq(&Interval::CERTAINLY_TRUE) {
                // If the logical disjunction is certainly true, one of the
                // operands must be true. However, it's not always possible to
                // determine which operand is true, leading to different scenarios.

                // If one operand is certainly false and the other one is uncertain,
                // then the latter must be certainly true.
                if left_interval.eq(&Interval::CERTAINLY_FALSE)
                    && right_interval.eq(&Interval::UNCERTAIN)
                {
                    Ok(Some(vec![
                        Interval::CERTAINLY_FALSE,
                        Interval::CERTAINLY_TRUE,
                    ]))
                } else if right_interval.eq(&Interval::CERTAINLY_FALSE)
                    && left_interval.eq(&Interval::UNCERTAIN)
                {
                    Ok(Some(vec![
                        Interval::CERTAINLY_TRUE,
                        Interval::CERTAINLY_FALSE,
                    ]))
                }
                // If both children are uncertain, or if one is certainly true,
                // we cannot conclusively refine their intervals. In this case,
                // propagation does not result in any interval changes.
                else {
                    Ok(Some(vec![]))
                }
            } else {
                // An uncertain logical disjunction result can not shrink the
                // end-points of its children.
                Ok(Some(vec![]))
            }
        } else if self.op.is_comparison_operator() {
            Ok(
                propagate_comparison(&self.op, interval, left_interval, right_interval)?
                    .map(|(left, right)| vec![left, right]),
            )
        } else {
            Ok(
                propagate_arithmetic(&self.op, interval, left_interval, right_interval)?
                    .map(|(left, right)| vec![left, right]),
            )
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        let mut s = state;
        self.hash(&mut s);
    }

    /// For each operator, [`CommutativeExpr`] has distinct rules.
    /// TODO: There may be rules specific to some data types and expression ranges.
    fn get_properties(&self, children: &[ExprProperties]) -> Result<ExprProperties> {
        let (l_order, l_range) = (children[0].sort_properties, &children[0].range);
        let (r_order, r_range) = (children[1].sort_properties, &children[1].range);
        match self.op() {
            Operator::Plus => Ok(ExprProperties {
                sort_properties: l_order.add(&r_order),
                range: l_range.add(r_range)?,
            }),
            Operator::Minus => Ok(ExprProperties {
                sort_properties: l_order.sub(&r_order),
                range: l_range.sub(r_range)?,
            }),
            Operator::Gt => Ok(ExprProperties {
                sort_properties: l_order.gt_or_gteq(&r_order),
                range: l_range.gt(r_range)?,
            }),
            Operator::GtEq => Ok(ExprProperties {
                sort_properties: l_order.gt_or_gteq(&r_order),
                range: l_range.gt_eq(r_range)?,
            }),
            Operator::Lt => Ok(ExprProperties {
                sort_properties: r_order.gt_or_gteq(&l_order),
                range: l_range.lt(r_range)?,
            }),
            Operator::LtEq => Ok(ExprProperties {
                sort_properties: r_order.gt_or_gteq(&l_order),
                range: l_range.lt_eq(r_range)?,
            }),
            Operator::And => Ok(ExprProperties {
                sort_properties: r_order.and_or(&l_order),
                range: l_range.and(r_range)?,
            }),
            Operator::Or => Ok(ExprProperties {
                sort_properties: r_order.and_or(&l_order),
                range: l_range.or(r_range)?,
            }),
            _ => Ok(ExprProperties::new_unknown()),
        }
    }
}

impl PartialEq<dyn Any> for CommutativeExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| {
                self.exprs
                    .iter()
                    .zip(x.exprs.iter())
                    .all(|(left, right)| left.eq(right))
                    && self.op == x.op
            })
            .unwrap_or(false)
    }
}

/// Casts dictionary array to result type for binary numerical operators. Such operators
/// between array and scalar produce a dictionary array other than primitive array of the
/// same operators between array and array. This leads to inconsistent result types causing
/// errors in the following query execution. For such operators between array and scalar,
/// we cast the dictionary array to primitive array.
fn to_result_type_array(
    op: &Operator,
    array: ArrayRef,
    result_type: &DataType,
) -> Result<ArrayRef> {
    if array.data_type() == result_type {
        Ok(array)
    } else if op.is_numerical_operators() {
        match array.data_type() {
            DataType::Dictionary(_, value_type) => {
                if value_type.as_ref() == result_type {
                    Ok(cast(&array, result_type)?)
                } else {
                    internal_err!(
                            "Incompatible Dictionary value type {value_type:?} with result type {result_type:?} of Binary operator {op:?}"
                        )
                }
            }
            _ => Ok(array),
        }
    } else {
        Ok(array)
    }
}

/// Evaluate the expression of the left input is an array and
/// right is literal - use scalar operations
fn evaluate_array_scalar(
    array: &dyn Array,
    scalar: ScalarValue,
    op: Operator,
) -> Result<Option<Result<ArrayRef>>> {
    use Operator::*;
    let scalar_result = match &op {
        RegexMatch => binary_string_array_flag_op_scalar!(
            array,
            scalar,
            regexp_is_match,
            false,
            false
        ),
        RegexIMatch => binary_string_array_flag_op_scalar!(
            array,
            scalar,
            regexp_is_match,
            false,
            true
        ),
        RegexNotMatch => binary_string_array_flag_op_scalar!(
            array,
            scalar,
            regexp_is_match,
            true,
            false
        ),
        RegexNotIMatch => binary_string_array_flag_op_scalar!(
            array,
            scalar,
            regexp_is_match,
            true,
            true
        ),
        BitwiseAnd => bitwise_and_dyn_scalar(array, scalar),
        BitwiseOr => bitwise_or_dyn_scalar(array, scalar),
        BitwiseXor => bitwise_xor_dyn_scalar(array, scalar),
        BitwiseShiftRight => bitwise_shift_right_dyn_scalar(array, scalar),
        BitwiseShiftLeft => bitwise_shift_left_dyn_scalar(array, scalar),
        // if scalar operation is not supported - fallback to array implementation
        _ => None,
    };

    Ok(scalar_result)
}

fn evaluate_with_resolved_args(
    left: Arc<dyn Array>,
    left_data_type: &DataType,
    right: Arc<dyn Array>,
    right_data_type: &DataType,
    op: Operator,
) -> Result<ArrayRef> {
    use Operator::*;
    match &op {
        IsDistinctFrom | IsNotDistinctFrom | Lt | LtEq | Gt | GtEq | Eq | NotEq
        | Plus | Minus | Multiply | Divide | Modulo | LikeMatch | ILikeMatch
        | NotLikeMatch | NotILikeMatch => unreachable!(),
        And => {
            if left_data_type == &DataType::Boolean {
                boolean_op!(&left, &right, and_kleene)
            } else {
                internal_err!(
                    "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                    op,
                    left.data_type(),
                    right.data_type()
                )
            }
        }
        Or => {
            if left_data_type == &DataType::Boolean {
                boolean_op!(&left, &right, or_kleene)
            } else {
                internal_err!(
                    "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                    op,
                    left_data_type,
                    right_data_type
                )
            }
        }
        RegexMatch => {
            binary_string_array_flag_op!(left, right, regexp_is_match, false, false)
        }
        RegexIMatch => {
            binary_string_array_flag_op!(left, right, regexp_is_match, false, true)
        }
        RegexNotMatch => {
            binary_string_array_flag_op!(left, right, regexp_is_match, true, false)
        }
        RegexNotIMatch => {
            binary_string_array_flag_op!(left, right, regexp_is_match, true, true)
        }
        BitwiseAnd => bitwise_and_dyn(left, right),
        BitwiseOr => bitwise_or_dyn(left, right),
        BitwiseXor => bitwise_xor_dyn(left, right),
        BitwiseShiftRight => bitwise_shift_right_dyn(left, right),
        BitwiseShiftLeft => bitwise_shift_left_dyn(left, right),
        StringConcat => binary_string_array_op!(left, right, concat_elements),
        AtArrow | ArrowAt => {
            unreachable!("ArrowAt and AtArrow should be rewritten to function")
        }
    }
}

fn evaluate_inner(
    lhs: ColumnarValue,
    right: &Arc<dyn PhysicalExpr>,
    batch: &RecordBatch,
    op: Operator,
    result_type: &DataType,
) -> Result<ColumnarValue> {
    use arrow::compute::kernels::numeric::*;

    let rhs = right.evaluate(batch)?;
    let left_data_type = lhs.data_type();
    let right_data_type = rhs.data_type();

    match op {
        Operator::Plus => return apply(&lhs, &rhs, add_wrapping),
        Operator::Minus => return apply(&lhs, &rhs, sub_wrapping),
        Operator::Multiply => return apply(&lhs, &rhs, mul_wrapping),
        Operator::Divide => return apply(&lhs, &rhs, div),
        Operator::Modulo => return apply(&lhs, &rhs, rem),
        Operator::Eq => return apply_cmp(&lhs, &rhs, eq),
        Operator::NotEq => return apply_cmp(&lhs, &rhs, neq),
        Operator::Lt => return apply_cmp(&lhs, &rhs, lt),
        Operator::Gt => return apply_cmp(&lhs, &rhs, gt),
        Operator::LtEq => return apply_cmp(&lhs, &rhs, lt_eq),
        Operator::GtEq => return apply_cmp(&lhs, &rhs, gt_eq),
        Operator::IsDistinctFrom => return apply_cmp(&lhs, &rhs, distinct),
        Operator::IsNotDistinctFrom => return apply_cmp(&lhs, &rhs, not_distinct),
        Operator::LikeMatch => return apply_cmp(&lhs, &rhs, like),
        Operator::ILikeMatch => return apply_cmp(&lhs, &rhs, ilike),
        Operator::NotLikeMatch => return apply_cmp(&lhs, &rhs, nlike),
        Operator::NotILikeMatch => return apply_cmp(&lhs, &rhs, nilike),
        _ => {}
    }

    // Attempt to use special kernels if one input is scalar and the other is an array
    let scalar_result = match (&lhs, &rhs) {
        (ColumnarValue::Array(array), ColumnarValue::Scalar(scalar)) => {
            // if left is array and right is literal - use scalar operations
            evaluate_array_scalar(array, scalar.clone(), op)?
                .map(|r| r.and_then(|a| to_result_type_array(&op, a, result_type)))
        }
        (_, _) => None, // default to array implementation
    };

    if let Some(result) = scalar_result {
        return result.map(ColumnarValue::Array);
    }

    // if both arrays or both literals - extract arrays and continue execution
    let (left, right) = (
        lhs.into_array(batch.num_rows())?,
        rhs.into_array(batch.num_rows())?,
    );
    evaluate_with_resolved_args(left, &left_data_type, right, &right_data_type, op)
        .map(ColumnarValue::Array)
}

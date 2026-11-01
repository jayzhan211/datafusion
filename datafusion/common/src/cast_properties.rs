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

//! What a cast from one type to another keeps, for the optimizations that
//! look through a `CAST`.
//!
//! Each function here is the rule one optimization uses today:
//!
//! | Function | Used by |
//! |---|---|
//! | [`is_lossless`] | statistics through projections, strict sort order, predicate pre-selection |
//! | [`preserves_order`] | sort order and equivalence through casts |
//! | [`supports_min_max_pruning`] | Parquet and file pruning |
//! | [`can_propagate_bounds_back`] | interval analysis (filter selectivity, join pruning) |
//! | [`is_exact_for_unwrap`] | moving a cast from a column to a literal in comparisons |
//! | [`is_injective`] | Map key evolution, decorrelating correlated equality filters |
//!
//! They answer overlapping questions and do not always agree: for example
//! [`supports_min_max_pruning`] accepts `Int32 -> Boolean`, which
//! [`preserves_order`] rejects. The tests in this module list every type pair
//! where they disagree with each other, and every answer the Arrow cast kernel
//! contradicts, so changes to these rules show up in one place.

use arrow::datatypes::{DataType, TimeUnit};

/// Casting `from` to `to` is lossless and strictly order-preserving for all
/// source values, preserving nulls, and cannot fail. This includes widening
/// casts (e.g. `Int8` or `UInt8` to `Int16`) and representation conversions
/// such as `Int32` to `Date32`, which interprets the same integer as days since
/// the epoch, or `Int64` to `Date64`, which interprets the same integer as
/// milliseconds since the epoch.
pub fn is_lossless(from: &DataType, to: &DataType) -> bool {
    use DataType::*;
    if from == to {
        return true;
    }
    matches!(
        (from, to),
        (Int8, Int16 | Int32 | Int64)
            | (Int16, Int32 | Int64)
            | (Int32, Int64)
            | (Int32, Date32)
            | (Date32, Int32)
            | (Int64, Date64)
            | (Date64, Int64)
            | (UInt8, UInt16 | UInt32 | UInt64 | Int16 | Int32 | Int64)
            | (UInt16, UInt32 | UInt64 | Int32 | Int64)
            | (UInt32, UInt64 | Int64)
            | (Int8 | Int16 | UInt8 | UInt16, Float32)
            | (Int8 | Int16 | Int32 | UInt8 | UInt16 | UInt32, Float64)
            | (Utf8, LargeUtf8 | Utf8View)
            | (Binary, LargeBinary | BinaryView)
    )
}

/// UTC and fixed offsets have no timezone transitions.
fn is_fixed_offset(tz: &str) -> bool {
    tz == "UTC" || tz.starts_with(['+', '-'])
}

/// Successful casts from `from` to `to` preserve order when conversion
/// failures return errors. Unlike [`is_lossless`], this allows precision loss
/// and is not sufficient for propagating distinct counts or the ordering of
/// subsequent sort keys.
pub fn preserves_order(from: &DataType, to: &DataType) -> bool {
    use DataType::*;
    use TimeUnit::*;
    if from == to || (from.is_numeric() || *from == Boolean) && to.is_numeric() {
        return true;
    }
    // Temporal casts are not generally monotonic: extracting time-of-day wraps
    // at midnight, and timezone transitions can reverse the local date.
    match (from, to) {
        (Date32 | Date64, Date32 | Date64)
        | (Date32 | Date64, Timestamp(_, None))
        | (Timestamp(_, None), Date32) => true,
        (Timestamp(_, Some(tz)), Date32) => is_fixed_offset(tz),
        // Arrow converts timestamps to Date64 by scaling the epoch value,
        // whereas Date32 extracts the date in the timestamp's timezone.
        (Timestamp(_, _), Date64) => true,
        // Only admit scaling that cannot wrap on overflow. Time64 widening
        // and narrowing to Time32 do not currently check overflow in Arrow.
        (Time32(Second | Millisecond), Time32(Second | Millisecond))
        | (Time32(Second | Millisecond), Time64(Microsecond | Nanosecond))
        | (Time64(Nanosecond), Time64(Microsecond))
        | (Duration(_), Duration(_)) => true,
        (Timestamp(_, from_tz), Timestamp(_, to_tz)) => {
            // Adding a timezone to naive timestamps interprets local times;
            // other timezone changes only change metadata on the epoch value.
            from_tz.is_some() || to_tz.as_deref().is_none_or(is_fixed_offset)
        }
        _ => false,
    }
}

/// Pruning can compare `CAST(column AS to)` with a literal by casting the
/// column's min/max statistics. Casts between strings and other types are
/// rejected, because `"13"` is less than `"3"` in string order. Dictionaries
/// are looked through.
pub fn supports_min_max_pruning(from: &DataType, to: &DataType) -> bool {
    match (from, to) {
        (DataType::Dictionary(_, from), to) => supports_min_max_pruning(from, to),
        (from, DataType::Dictionary(_, to)) => supports_min_max_pruning(from, to),
        // Both strings or both not strings (numbers, timestamps, etc.)
        (from, to) => from.is_string() == to.is_string(),
    }
}

/// Bounds on `CAST(x AS to)` can be cast back to `from` without excluding valid
/// inputs, for reverse constraint propagation through the cast. Many-to-one
/// casts, such as `Float64` to `Int32`, cannot generally be inverted this way:
/// an output of 0 does not imply an input of 0.0. Returns false for
/// unrecognized conversions so the input range remains unchanged.
pub fn can_propagate_bounds_back(from: &DataType, to: &DataType) -> bool {
    is_lossless(from, to)
        || (from.is_integer() && to.is_integer())
        // NaN bounds are unbounded; finite Float32 values widen exactly.
        || (*from == DataType::Float32 && *to == DataType::Float64)
        || is_decimal_widening(from, to)
}

/// Whether a decimal-to-decimal cast keeps every value exactly, i.e. the target
/// has at least as many fractional and integer digits as the source. Casting
/// bounds back to the source then rounds to a neighboring representable value,
/// which never excludes a valid input.
fn is_decimal_widening(from: &DataType, to: &DataType) -> bool {
    fn precision_scale(data_type: &DataType) -> Option<(i16, i16)> {
        use DataType::*;
        match data_type {
            Decimal32(p, s) | Decimal64(p, s) | Decimal128(p, s) | Decimal256(p, s) => {
                Some((*p as i16, *s as i16))
            }
            _ => None,
        }
    }
    match (precision_scale(from), precision_scale(to)) {
        (Some((from_p, from_s)), Some((to_p, to_s))) => {
            to_s >= from_s && to_p - to_s >= from_p - from_s
        }
        _ => false,
    }
}

/// Casting `from` to `to` is exact and order-preserving over the source values
/// the cast succeeds on, so a comparison `CAST(column AS to) <op> literal` can
/// compare the column with the literal converted back to `from`.
///
/// Integer or decimal sources to a decimal target are admitted when the source
/// scale is nonnegative and the target scale is at least the source scale: a
/// successful cast only multiplies the unscaled value by an integral power of
/// ten. Precision or width overflow is the explicit exception: `CAST` errors
/// and `TRY_CAST` yields NULL for such source values, while the rewritten
/// predicate compares the source value and returns TRUE/FALSE. This rule is
/// type-based and does not distinguish implicit coercions from explicit casts.
///
/// Timestamp-to-timestamp casts are decided by their own unit arithmetic, not
/// by this rule. One level of dictionary is looked through.
pub fn is_exact_for_unwrap(from: &DataType, to: &DataType) -> bool {
    use DataType::*;
    let (from, to) = match (from, to) {
        (Dictionary(_, from), Dictionary(_, to)) => (from.as_ref(), to.as_ref()),
        (Dictionary(_, from), to) => (from.as_ref(), to),
        (from, Dictionary(_, to)) => (from, to.as_ref()),
        _ => (from, to),
    };
    // The literal conversion (`is_supported_type`) does not support Decimal256.
    let decimal_scale = |data_type: &DataType| match data_type {
        Decimal32(_, scale) | Decimal64(_, scale) | Decimal128(_, scale) => Some(*scale),
        _ => None,
    };

    // Integer or decimal sources to a decimal target are accepted when the
    // source scale is nonnegative and the target only adds fractional digits.
    // This precedes identity so negative-scale decimals are rejected by the
    // scale check instead of reaching the scale arithmetic and panicking.
    if let Some(to_scale) = decimal_scale(to) {
        let from_scale = match decimal_scale(from) {
            Some(scale) => Some(scale),
            None if from.is_integer() => Some(0),
            None => None,
        };
        if let Some(from_scale) = from_scale {
            return from_scale >= 0 && to_scale >= from_scale;
        }
    }

    if from == to {
        return true;
    }

    let is_string =
        |data_type: &DataType| matches!(data_type, Utf8 | LargeUtf8 | Utf8View);
    match (from, to) {
        // Integer cast that is value-preserving over the full source domain.
        (Int8, Int16 | Int32 | Int64)
        | (Int16, Int32 | Int64)
        | (Int32, Int64)
        | (UInt8, UInt16 | UInt32 | UInt64 | Int16 | Int32 | Int64)
        | (UInt16, UInt32 | UInt64 | Int32 | Int64)
        | (UInt32, UInt64 | Int64) => true,
        // Date32 <-> Int32 and Date64 <-> Int64 share the same internal
        // representation.
        (Date32, Int32) | (Int32, Date32) | (Date64, Int64) | (Int64, Date64) => true,
        // Date32 -> Date64 is injective: every source day maps to its midnight
        // millisecond value. The literal conversion enforces the exact
        // day-boundary requirement in the reverse direction.
        (Date32, Date64) => true,
        // String width casts among Utf8 / LargeUtf8 / Utf8View.
        (from, to) if is_string(from) && is_string(to) => true,
        // FixedSizeBinary(n) -> Binary is always widening.
        (FixedSizeBinary(_), Binary) => true,
        (FixedSizeBinary(n1), FixedSizeBinary(n2)) => n1 == n2,
        _ => false,
    }
}

/// Casting `from` to `to` preserves value identity, so distinct source values
/// stay distinct.
///
/// Keep this list conservative. Map keys and decorrelated equality keys depend
/// on identity, so only representation widenings that cannot merge distinct
/// source values belong here. Other Arrow-supported casts, including
/// decimal/timestamp widening and dictionaries, should stay rejected until
/// their identity semantics are deliberately reviewed.
pub fn is_injective(from: &DataType, to: &DataType) -> bool {
    use DataType::*;
    matches!(
        (from, to),
        (Int8, Int16 | Int32 | Int64)
            | (Int16, Int32 | Int64)
            | (Int32, Int64)
            | (UInt8, Int16 | Int32 | Int64 | UInt16 | UInt32 | UInt64)
            | (UInt16, Int32 | Int64 | UInt32 | UInt64)
            | (UInt32, Int64 | UInt64)
            | (Utf8, LargeUtf8 | Utf8View)
            | (Binary, LargeBinary | BinaryView)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Array, ArrayRef, BooleanArray, FixedSizeBinaryArray, Float64Array, Int64Array,
        StringArray, UInt32Array, make_comparator,
    };
    use arrow::compute::kernels::cast::{CastOptions, cast_with_options};
    use arrow::compute::{
        SortOptions, can_cast_types, concat, filter, is_not_null, sort_to_indices, take,
    };
    use std::cmp::Ordering;
    use std::fmt::Write;
    use std::sync::Arc;

    type Rule = fn(&DataType, &DataType) -> bool;

    const RULES: [(&str, Rule); 6] = [
        ("is_lossless", is_lossless),
        ("preserves_order", preserves_order),
        ("supports_min_max_pruning", supports_min_max_pruning),
        ("can_propagate_bounds_back", can_propagate_bounds_back),
        ("is_exact_for_unwrap", is_exact_for_unwrap),
        ("is_injective", is_injective),
    ];

    fn types() -> Vec<DataType> {
        use DataType::*;
        use TimeUnit::*;
        let tz = |tz: &str| Timestamp(Nanosecond, Some(tz.into()));
        vec![
            Boolean,
            Int8,
            Int16,
            Int32,
            Int64,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Float32,
            Float64,
            Decimal32(9, 1),
            Decimal64(18, 2),
            Decimal128(5, 0),
            Decimal128(10, 2),
            Decimal128(20, 2),
            Decimal128(10, 3),
            Decimal128(3, -1),
            Decimal256(40, 2),
            Utf8,
            LargeUtf8,
            Utf8View,
            Binary,
            LargeBinary,
            BinaryView,
            FixedSizeBinary(4),
            Date32,
            Date64,
            Timestamp(Second, None),
            Timestamp(Nanosecond, None),
            tz("UTC"),
            tz("+05:00"),
            tz("America/New_York"),
            Time32(Second),
            Time64(Nanosecond),
            Duration(Second),
            Duration(Nanosecond),
            Dictionary(Box::new(Int32), Box::new(Utf8)),
            Dictionary(Box::new(Int32), Box::new(Int64)),
        ]
    }

    /// Distinct pairs Arrow can cast between.
    fn castable_pairs() -> Vec<(DataType, DataType)> {
        let types = types();
        let mut pairs = vec![];
        for from in &types {
            for to in &types {
                if from != to && can_cast_types(from, to) {
                    pairs.push((from.clone(), to.clone()));
                }
            }
        }
        pairs
    }

    fn name(from: &DataType, to: &DataType) -> String {
        format!("{from} -> {to}")
    }

    /// For each fact, the rules that imply it and the rule that decides it,
    /// listing every pair where an implying rule says yes and the deciding rule
    /// says no. The rules were written separately, so each entry is knowledge
    /// one rule has and another lacks, or a wrong answer in one of them.
    #[test]
    fn rules_disagree() {
        let pairs = castable_pairs();
        let mut report = String::new();
        let mut section = |title: &str, rule: Rule, implied_by: &[(&str, Rule)]| {
            writeln!(report, "## {title}").unwrap();
            for (implying, implies) in implied_by {
                let rejected: Vec<String> = pairs
                    .iter()
                    .filter(|(from, to)| implies(from, to) && !rule(from, to))
                    .map(|(from, to)| name(from, to))
                    .collect();
                writeln!(
                    report,
                    "{implying} yes ({}): {}",
                    rejected.len(),
                    rejected.join(", ")
                )
                .unwrap();
            }
            writeln!(report).unwrap();
        };
        section(
            "order: preserves_order says no",
            preserves_order,
            &[
                ("supports_min_max_pruning", supports_min_max_pruning),
                ("is_lossless", is_lossless),
                ("can_propagate_bounds_back", can_propagate_bounds_back),
                ("is_exact_for_unwrap", is_exact_for_unwrap),
            ],
        );
        section(
            "order: supports_min_max_pruning says no",
            supports_min_max_pruning,
            &[("preserves_order", preserves_order)],
        );
        section(
            "identity: is_injective says no",
            is_injective,
            &[
                ("is_lossless", is_lossless),
                ("can_propagate_bounds_back", can_propagate_bounds_back),
                ("is_exact_for_unwrap", is_exact_for_unwrap),
            ],
        );
        insta::assert_snapshot!(report);
    }

    /// Distinct non-null sample values of `data_type`, sorted. Temporal values
    /// stay within +/-2^31 so that unit scaling cannot overflow, where Arrow
    /// panics in debug builds and wraps in release builds.
    fn samples(data_type: &DataType) -> Option<ArrayRef> {
        use DataType::*;
        let small = 1_i64 << 31;
        let ints: Vec<i64> = vec![
            i64::MIN,
            -(1 << 40),
            -small - 1,
            -small,
            -32_769,
            -129,
            -128,
            -1,
            0,
            1,
            2,
            127,
            128,
            255,
            256,
            32_767,
            65_535,
            86_399,
            small - 1,
            small,
            4_294_967_296,
            (1 << 53) + 1,
            i64::MAX,
        ];
        let seeds: Vec<ArrayRef> = match data_type {
            Date32 | Date64 | Timestamp(_, _) | Time32(_) | Time64(_) | Duration(_) => {
                let ints = ints.into_iter().filter(|v| (-small..=small).contains(v));
                vec![Arc::new(Int64Array::from_iter_values(ints))]
            }
            Utf8 | LargeUtf8 | Utf8View | Binary | LargeBinary | BinaryView => {
                vec![Arc::new(StringArray::from(vec![
                    "", "-1", "0", "1", "10", "9", "a", "b", "abcd", "abce",
                ]))]
            }
            FixedSizeBinary(_) => vec![Arc::new(
                FixedSizeBinaryArray::try_from_iter(
                    ["abcd", "abce", "zzzz"].into_iter().map(str::as_bytes),
                )
                .unwrap(),
            )],
            _ => vec![
                Arc::new(Int64Array::from(ints)),
                Arc::new(Float64Array::from(vec![
                    f64::NEG_INFINITY,
                    -1e300,
                    -1.5,
                    -0.5,
                    -0.0,
                    0.0,
                    0.5,
                    1.25,
                    1.7,
                    16_777_217.0,
                    1e300,
                    f64::INFINITY,
                    f64::NAN,
                ])),
                Arc::new(BooleanArray::from(vec![false, true])),
            ],
        };
        let safe = CastOptions {
            safe: true,
            ..Default::default()
        };
        let parts: Vec<ArrayRef> = seeds
            .iter()
            .filter_map(|seed| cast_with_options(seed, data_type, &safe).ok())
            .map(|array| filter(&array, &is_not_null(&array).unwrap()).unwrap())
            .filter(|array| !array.is_empty())
            .collect();
        if parts.is_empty() {
            return None;
        }
        let parts: Vec<&dyn Array> = parts.iter().map(|array| array.as_ref()).collect();
        let all = concat(&parts).unwrap();
        let sorted =
            take(&all, &sort_to_indices(&all, None, None).unwrap(), None).unwrap();
        let cmp = make_comparator(&sorted, &sorted, SortOptions::default()).unwrap();
        let distinct: Vec<u32> = (0..sorted.len())
            .filter(|&i| i == 0 || cmp(i - 1, i) != Ordering::Equal)
            .map(|i| i as u32)
            .collect();
        Some(take(&sorted, &UInt32Array::from(distinct), None).unwrap())
    }

    /// What a rule claims about the values a cast converts:
    /// (keeps order, keeps distinct values distinct, never fails).
    fn claims(rule: &str) -> (bool, bool, bool) {
        match rule {
            "is_lossless" => (true, true, true),
            "preserves_order" | "supports_min_max_pruning" => (true, false, false),
            "can_propagate_bounds_back" | "is_exact_for_unwrap" => (true, true, false),
            "is_injective" => (false, true, false),
            _ => unreachable!(),
        }
    }

    /// Every answer of a rule that the Arrow cast kernel contradicts on sample
    /// values. Dictionaries are skipped: the rules look through them.
    #[test]
    fn kernel_contradictions() {
        let mut report = String::new();
        for (rule_name, rule) in RULES {
            let (order, injective, infallible) = claims(rule_name);
            let mut wrong = vec![];
            for (from, to) in castable_pairs() {
                if !rule(&from, &to)
                    || matches!(from, DataType::Dictionary(_, _))
                    || matches!(to, DataType::Dictionary(_, _))
                {
                    continue;
                }
                let Some(source) = samples(&from) else {
                    continue;
                };
                let pair = name(&from, &to);
                let strict = CastOptions {
                    safe: false,
                    ..Default::default()
                };
                if infallible && cast_with_options(&source, &to, &strict).is_err() {
                    wrong.push(format!("{pair} (fails)"));
                    continue;
                }
                let safe = CastOptions {
                    safe: true,
                    ..Default::default()
                };
                let Ok(output) = cast_with_options(&source, &to, &safe) else {
                    continue;
                };
                // `source` is sorted and distinct; keep the converted values.
                let converted: Vec<usize> =
                    (0..output.len()).filter(|&i| output.is_valid(i)).collect();
                let cmp =
                    make_comparator(&output, &output, SortOptions::default()).unwrap();
                if order
                    && converted
                        .windows(2)
                        .any(|w| cmp(w[0], w[1]) == Ordering::Greater)
                {
                    wrong.push(format!("{pair} (order)"));
                } else if injective
                    && converted.iter().enumerate().any(|(n, &i)| {
                        converted[n + 1..]
                            .iter()
                            .any(|&j| cmp(i, j) == Ordering::Equal)
                    })
                {
                    wrong.push(format!("{pair} (merges values)"));
                }
            }
            writeln!(
                report,
                "{rule_name} ({}): {}",
                wrong.len(),
                wrong.join(", ")
            )
            .unwrap();
        }
        insta::assert_snapshot!(report);
    }
}

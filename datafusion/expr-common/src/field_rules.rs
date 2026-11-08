// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! One output-field rule per expression kind, used by both layers.
//!
//! An expression kind that exists both as a logical `Expr` and as a
//! `PhysicalExpr` gets one function here that computes its output [`Field`]
//! (data type, nullability, metadata) from its input fields. The logical
//! `ExprSchemable::to_field` arm and the physical `PhysicalExpr::return_field`
//! call the same function, so the two layers cannot answer differently.
//!
//! Names are not part of a rule. A rule that starts from an input field keeps
//! that field's name; a rule that makes a new value returns an empty name.
//! The logical layer renames every field to the expression's schema name; the
//! physical layer keeps the rule's name (operators name their own columns).
//!
//! Not here, on purpose:
//! - CASE: its nullability looks at the WHEN predicates, not only at fields.
//! - Functions: their field comes from the UDF (`return_field_from_args`).
//! - Kinds that exist in one layer only (EXISTS, BETWEEN, subqueries, ...).

use arrow::datatypes::{DataType, Field, FieldRef};
use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
use datafusion_common::Result;
use datafusion_common::metadata::merge_metadata_for_union;

use crate::operator::Operator;

/// `CAST(x AS target)`, and `TRY_CAST` through [`try_cast`].
///
/// `explicit_target` says whether `target` is a whole field chosen by the
/// caller (`CAST(x AS UUID)`, a schema adapter aligning a file column to the
/// table) or a data type alone (`CAST(x AS INT)`, see [`is_type_only_cast`]).
///
/// - type: the target's.
/// - metadata: explicit target: the target's, exactly. Type only: the
///   source's minus the extension type keys, because a cast to a plain
///   storage type ends the extension type (`arrow.json` -> `Utf8`).
/// - nullable: type only: the source's. Explicit target: the source's OR the
///   target's. `null_on_failure` (TRY_CAST, or CAST with `safe` options, where
///   a failed cast gives NULL): always.
/// - name: the source's.
pub fn cast(
    source: &Field,
    target: &Field,
    explicit_target: bool,
    null_on_failure: bool,
) -> Field {
    let (metadata, nullable) = if explicit_target {
        (
            target.metadata().clone(),
            source.is_nullable() || target.is_nullable(),
        )
    } else {
        let mut metadata = source.metadata().clone();
        metadata.remove(EXTENSION_TYPE_NAME_KEY);
        metadata.remove(EXTENSION_TYPE_METADATA_KEY);
        (metadata, source.is_nullable())
    };
    source
        .clone()
        .with_data_type(target.data_type().clone())
        .with_metadata(metadata)
        .with_nullable(nullable || null_on_failure)
}

/// `TRY_CAST(x AS target)`: [`cast`] whose failed values become NULL, so the
/// output is always nullable.
pub fn try_cast(source: &Field, target: &Field, explicit_target: bool) -> Field {
    cast(source, target, explicit_target, true)
}

/// Whether a cast target names a data type alone: the field that
/// `DataType::into_nullable_field` makes (empty name, nullable, no metadata).
///
/// The logical `Expr::Cast` stores a field either way and tells the two apart
/// by this shape; `CastExpr` stores the answer as `explicit_target` when it is
/// lowered.
pub fn is_type_only_cast(target: &Field) -> bool {
    target.name().is_empty() && target.is_nullable() && target.metadata().is_empty()
}

/// `NOT x`: Boolean, nullable if `x` is. No metadata and no name: NOT makes a
/// new value, it does not pass `x` through.
pub fn not(input: &Field) -> Field {
    Field::new("", DataType::Boolean, input.is_nullable())
}

/// `-x`: `x`'s whole field (type, nullability, metadata, name). The value is
/// still `x`'s kind of value.
pub fn negative(input: &Field) -> Field {
    input.clone()
}

/// A test that answers TRUE or FALSE and never NULL: `x IS [NOT] NULL`,
/// `x IS [NOT] TRUE | FALSE | UNKNOWN`, `EXISTS (...)`. Boolean, not nullable,
/// no metadata. The input's field does not matter.
pub fn boolean_test() -> Field {
    Field::new("", DataType::Boolean, false)
}

/// `left op right`, for every binary operator.
///
/// - type: `result_type`, which both layers get from `BinaryTypeCoercer`
///   (the logical layer attaches spans to its errors, so it runs the coercer
///   itself).
/// - nullable: if either side is, except `IS [NOT] DISTINCT FROM`, which
///   compares NULLs as values and is never NULL. (Physical `x IS TRUE` and
///   friends are lowered to `IS NOT DISTINCT FROM`, so they get the
///   [`boolean_test`] answer from here.)
/// - metadata: none.
pub fn binary(
    left: &Field,
    op: &Operator,
    right: &Field,
    result_type: DataType,
) -> Field {
    let nullable = match op {
        Operator::IsDistinctFrom | Operator::IsNotDistinctFrom => false,
        _ => left.is_nullable() || right.is_nullable(),
    };
    Field::new("", result_type, nullable)
}

/// How many inputs of `x IN (list)` (`x` plus list items) [`in_list`] looks
/// at before it assumes the result is nullable.
///
/// From the logical layer (#6799): `nullable()` is called many times during
/// optimization, so it must stay cheap on long lists. The author called the
/// number itself a guess; the review accepted it as conservative (it can only
/// say "nullable" too often, never too rarely).
pub const IN_LIST_INSPECT_LIMIT: usize = 6;

/// `x [NOT] IN (list)`: Boolean, no metadata.
///
/// Nullable if `x` or any list item is nullable. A list with more inputs than
/// [`IN_LIST_INSPECT_LIMIT`] is nullable without looking further. `list` is
/// lazy so that the fields of a long list are never computed; the inputs
/// before the cap are still computed, so their errors are reported.
pub fn in_list(
    expr: &Field,
    list: impl ExactSizeIterator<Item = Result<FieldRef>>,
) -> Result<Field> {
    let too_long = list.len() + 1 > IN_LIST_INSPECT_LIMIT;
    let mut nullable = expr.is_nullable();
    for item in list.take(IN_LIST_INSPECT_LIMIT - 1) {
        if nullable {
            break;
        }
        nullable = item?.is_nullable();
    }
    Ok(Field::new("", DataType::Boolean, nullable || too_long))
}

/// One output column of a UNION, from the matching column of every input.
///
/// - nullable: if any input is. (A column that UNION BY NAME is missing from
///   an input counts as a NULL input.)
/// - metadata: the inputs' metadata merged in input order, the last input
///   wins on a conflict ([`merge_metadata_for_union`]).
/// - type: `data_type`, chosen by the caller.
///
/// SKETCH: the type is not shared because the four UNION sites choose it
/// differently today: the logical plan coerces (`loose_types`) or requires
/// equal types, the analyzer always coerces, and `UnionExec` takes the type of
/// the first nullable input. They agree once the analyzer has cast every
/// input to one type, which is the only case `UnionExec` sees in practice.
/// The name is not shared either (logical dedups input 0's names, the others
/// take input 0's or the base schema's).
pub fn union_field<'a>(
    data_type: DataType,
    inputs: impl IntoIterator<Item = &'a Field>,
) -> Field {
    let mut nullable = false;
    let metadata = merge_metadata_for_union(
        inputs
            .into_iter()
            .inspect(|field| nullable |= field.is_nullable())
            .map(|field| field.metadata()),
    );
    Field::new("", data_type, nullable).with_metadata(metadata)
}

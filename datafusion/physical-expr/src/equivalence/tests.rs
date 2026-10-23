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

use super::*;
use crate::expressions::{BinaryExpr, CastExpr, Column, Literal, binary, col, lit};
use crate::physical_expr::add_offset_to_expr;
use crate::projection::tests::output_schema;
use crate::utils::tests::TestScalarUDF;
use crate::{PhysicalExprRef, ScalarFunctionExpr};

use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{
    Constraint, Constraints, JoinSide, JoinType, Result, ScalarValue,
};
use datafusion_expr::sort_properties::SortProperties;
use datafusion_expr::{Operator, ScalarUDF};
use datafusion_functions::core::{get_field, named_struct};
use datafusion_functions::string::concat;
use itertools::Itertools;

/// Converts a string to a physical sort expression
///
/// # Example
/// * `"a"` -> (`"a"`, `SortOptions::default()`)
/// * `"a ASC"` -> (`"a"`, `SortOptions { descending: false, nulls_first: false }`)
pub fn parse_sort_expr(name: &str, schema: &SchemaRef) -> PhysicalSortExpr {
    let mut parts = name.split_whitespace();
    let name = parts.next().expect("empty sort expression");
    let mut sort_expr = PhysicalSortExpr::new(
        col(name, schema).expect("invalid column name"),
        SortOptions::default(),
    );

    if let Some(options) = parts.next() {
        sort_expr = match options {
            "ASC" => sort_expr.asc(),
            "DESC" => sort_expr.desc(),
            _ => panic!("unknown sort options. Expected 'ASC' or 'DESC', got {options}"),
        }
    }

    assert!(
        parts.next().is_none(),
        "unexpected tokens in column name. Expected 'name' / 'name ASC' / 'name DESC' but got  '{name}'"
    );

    sort_expr
}

// Generate a schema which consists of 8 columns (a, b, c, d, e, f, g, h)
pub fn create_test_schema() -> Result<SchemaRef> {
    let fields = ["a", "b", "c", "d", "e", "f", "g", "h"]
        .into_iter()
        .map(|name| Field::new(name, DataType::Int32, true));
    Ok(Arc::new(Schema::new(fields.collect::<Vec<_>>())))
}

/// Construct a schema with following properties
/// Schema satisfies following orderings:
/// [a ASC], [d ASC, b ASC], [e DESC, f ASC, g ASC]
/// and
/// Column [a=c] (e.g they are aliases).
pub fn create_test_params() -> Result<(SchemaRef, EquivalenceProperties)> {
    let test_schema = create_test_schema()?;
    let col_a = col("a", &test_schema)?;
    let col_b = col("b", &test_schema)?;
    let col_c = col("c", &test_schema)?;
    let col_d = col("d", &test_schema)?;
    let col_e = col("e", &test_schema)?;
    let col_f = col("f", &test_schema)?;
    let col_g = col("g", &test_schema)?;
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&test_schema));
    eq_properties.add_equal(Arc::clone(&col_a), Arc::clone(&col_c));

    let option_asc = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let option_desc = SortOptions {
        descending: true,
        nulls_first: true,
    };
    let orderings = vec![
        // [a ASC]
        vec![(col_a, option_asc)],
        // [d ASC, b ASC]
        vec![(col_d, option_asc), (col_b, option_asc)],
        // [e DESC, f ASC, g ASC]
        vec![
            (col_e, option_desc),
            (col_f, option_asc),
            (col_g, option_asc),
        ],
    ];
    let orderings = convert_to_orderings(&orderings);
    eq_properties.add_orderings(orderings);
    Ok((test_schema, eq_properties))
}

/// A requirement from `(expr, options)` pairs; `None` options make a
/// one-expression `Group` step (sorted in either direction).
pub fn convert_to_sort_reqs(
    args: &[(&Arc<dyn PhysicalExpr>, Option<SortOptions>)],
) -> Ordering {
    Ordering::new(args.iter().map(|(expr, options)| match options {
        Some(options) => Step::sort(Arc::clone(*expr), *options),
        None => Step::group([Arc::clone(*expr)]),
    }))
}

/// Every way to read `ordering` as plain sort expressions: one member per
/// `Sort` step (members are strict images of each other).
pub fn member_choices(ordering: &Ordering) -> Vec<LexOrdering> {
    ordering
        .iter()
        .map(|step| match step {
            Step::Sort(members) => members.clone(),
            Step::Group(_) => panic!("no provided Group steps in these tests"),
        })
        .multi_cartesian_product()
        .filter_map(LexOrdering::new)
        .collect()
}

/// Asserts that `actual` provides exactly the orderings `expected` provides:
/// every expected ordering holds in `actual`, and every stored ordering of
/// `actual` (each choice of members) holds given only the expected orderings
/// and `actual`'s equalities and constants. Comparing answers instead of
/// lists keeps tests independent of how orderings are stored.
pub fn assert_orderings_equivalent(
    actual: &EquivalenceProperties,
    expected: &[LexOrdering],
    context: &str,
) {
    for ordering in expected {
        assert!(
            actual.ordering_satisfy(ordering.clone()),
            "{context}: expected ordering [{ordering}] does not hold in {actual}"
        );
    }
    let mut reference = actual.clone().reordered(None);
    reference.add_orderings(expected.iter().cloned());
    for ordering in actual.orderings() {
        for choice in member_choices(ordering) {
            assert!(
                reference.ordering_satisfy(choice.clone()),
                "{context}: ordering [{choice}] of {actual} is not implied by the expected orderings"
            );
        }
    }
}

/// Asserts that `actual` stores exactly the given all-partition constants
/// (with values) and per-partition constants, compared as sets.
fn assert_constants(
    actual: &EquivalenceProperties,
    global: &[(Arc<dyn PhysicalExpr>, Option<ScalarValue>)],
    local: &[Arc<dyn PhysicalExpr>],
    context: &str,
) {
    let actual_global: Vec<_> = actual
        .global_constants()
        .map(|(e, v)| (Arc::clone(e), v.clone()))
        .collect();
    let actual_local: Vec<_> = actual.local_constants().cloned().collect();
    let msg = format!(
        "{context}: expected global {global:?}, local {local:?}; actual {actual}"
    );
    assert_eq!(actual_global.len(), global.len(), "{msg}");
    assert!(global.iter().all(|c| actual_global.contains(c)), "{msg}");
    assert_eq!(actual_local.len(), local.len(), "{msg}");
    assert!(local.iter().all(|c| actual_local.contains(c)), "{msg}");
}

/// `name` as a column of a schema whose field names do not matter.
fn column(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new(name, index))
}

// ---------------------------------------------------------------------------
// Equalities
// ---------------------------------------------------------------------------

/// `add_equal` merges classes: a repeated equality is a no-op, an equality
/// with a new member widens a class, and an equality across two classes
/// bridges them into one.
#[test]
fn add_equal_conditions_test() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, true),
        Field::new("c", DataType::Int64, true),
        Field::new("x", DataType::Int64, true),
        Field::new("y", DataType::Int64, true),
    ]));

    let mut eq_properties = EquivalenceProperties::new(schema);
    let col_a = column("a", 0);
    let col_b = column("b", 1);
    let col_c = column("c", 2);
    let col_x = column("x", 3);
    let col_y = column("y", 4);

    // a and b are aliases
    eq_properties.add_equal(Arc::clone(&col_a), Arc::clone(&col_b));
    assert_eq!(eq_properties.equalities().len(), 1);

    // This new entry is redundant, size shouldn't increase
    eq_properties.add_equal(Arc::clone(&col_b), Arc::clone(&col_a));
    assert_eq!(eq_properties.equalities().len(), 1);
    let class = eq_properties.equalities().iter().next().unwrap();
    assert_eq!(class.len(), 2);
    assert!(class.contains(&col_a));
    assert!(class.contains(&col_b));

    // b and c are aliases. Existing equivalence class should expand,
    // however there shouldn't be any new equivalence class
    eq_properties.add_equal(Arc::clone(&col_b), Arc::clone(&col_c));
    assert_eq!(eq_properties.equalities().len(), 1);
    let class = eq_properties.equalities().iter().next().unwrap();
    assert_eq!(class.len(), 3);
    assert!(class.contains(&col_a));
    assert!(class.contains(&col_b));
    assert!(class.contains(&col_c));

    // This is a new set of equality. Hence equivalent class count should be 2.
    eq_properties.add_equal(Arc::clone(&col_x), Arc::clone(&col_y));
    assert_eq!(eq_properties.equalities().len(), 2);

    // This equality bridges distinct equality sets.
    // Hence equivalent class count should decrease from 2 to 1.
    eq_properties.add_equal(Arc::clone(&col_x), Arc::clone(&col_a));
    assert_eq!(eq_properties.equalities().len(), 1);
    let class = eq_properties.equalities().iter().next().unwrap();
    assert_eq!(class.len(), 5);
    for col in [&col_a, &col_b, &col_c, &col_x, &col_y] {
        assert!(class.contains(col));
    }

    Ok(())
}

/// Overlapping equalities end up in one class (compared as sets of sets).
#[test]
fn test_bridge_groups() -> Result<()> {
    // First entry in the tuple is argument, second entry is the bridged result
    let test_cases = vec![
        (
            vec![vec![1, 2, 3], vec![2, 4, 5], vec![11, 12, 9], vec![7, 6, 5]],
            vec![vec![1, 2, 3, 4, 5, 6, 7], vec![9, 11, 12]],
        ),
        (
            vec![vec![1, 2, 3], vec![3, 4, 5], vec![9, 8, 7], vec![7, 6, 5]],
            vec![vec![1, 2, 3, 4, 5, 6, 7, 8, 9]],
        ),
    ];
    let c = |idx: usize| column(&format!("col_{idx}"), idx);
    for (entries, expected) in test_cases {
        let mut eq_properties = EquivalenceProperties::new(Arc::new(Schema::empty()));
        for entry in &entries {
            for &idx in &entry[1..] {
                eq_properties.add_equal(c(entry[0]), c(idx));
            }
        }
        let actual: Vec<Vec<usize>> = eq_properties
            .equalities()
            .iter()
            .map(|class| {
                class
                    .iter()
                    .map(|e| e.downcast_ref::<Column>().unwrap().index())
                    .sorted()
                    .collect()
            })
            .sorted()
            .collect();
        let expected: Vec<Vec<usize>> = expected.into_iter().sorted().collect();
        assert_eq!(actual, expected, "entries: {entries:?}");
    }
    Ok(())
}

/// Normalizing replaces a class member by the class's first member.
#[test]
fn test_schema_normalize_expr_with_equivalence() -> Result<()> {
    let col_a = column("a", 0);
    let col_b = column("b", 1);
    let col_c = column("c", 2);
    // Assume that column a and c are aliases.
    let (_, eq_properties) = create_test_params()?;
    let expressions = vec![
        (Arc::clone(&col_a), Arc::clone(&col_a)),
        (col_c, Arc::clone(&col_a)),
        // Cannot normalize column b
        (Arc::clone(&col_b), col_b),
    ];
    let eq = eq_properties.equalities();
    for (expr, expected) in expressions {
        assert!(expected.eq(&eq.normalize(expr)));
    }
    Ok(())
}

/// `are_equal` compares expressions after replacing class members,
/// including inside larger expressions.
#[test]
fn test_exprs_equal() -> Result<()> {
    let col_a = column("a", 0);
    let col_b = column("b", 1);
    let col_x = column("x", 2);
    let col_y = column("y", 3);
    let lit_1 = lit(1);
    let lit_2 = lit(2);

    // Classes (a = x) and (b = y)
    let mut eq_properties = EquivalenceProperties::new(Arc::new(Schema::empty()));
    eq_properties.add_equal(Arc::clone(&col_a), Arc::clone(&col_x));
    eq_properties.add_equal(Arc::clone(&col_b), Arc::clone(&col_y));
    let eq = eq_properties.equalities();

    let plus = |l: &Arc<dyn PhysicalExpr>, r: &Arc<dyn PhysicalExpr>| {
        Arc::new(BinaryExpr::new(
            Arc::clone(l),
            Operator::Plus,
            Arc::clone(r),
        )) as Arc<dyn PhysicalExpr>
    };
    let times = |l: &Arc<dyn PhysicalExpr>, r: &Arc<dyn PhysicalExpr>| {
        Arc::new(BinaryExpr::new(
            Arc::clone(l),
            Operator::Multiply,
            Arc::clone(r),
        )) as Arc<dyn PhysicalExpr>
    };
    let test_cases = vec![
        (&col_a, &col_a, true, "Same column should be equal"),
        (
            &col_a,
            &col_x,
            true,
            "Columns in same class should be equal",
        ),
        (
            &col_b,
            &col_y,
            true,
            "Columns in same class should be equal",
        ),
        (
            &col_a,
            &col_b,
            false,
            "Columns in different classes are not equal",
        ),
        (&lit_1, &lit_1, true, "Same literal should be equal"),
        (
            &lit_1,
            &lit_2,
            false,
            "Different literals should not be equal",
        ),
    ];
    for (left, right, expected, description) in test_cases {
        assert_eq!(eq.are_equal(left, right), expected, "{description}");
    }
    let compound = vec![
        (
            plus(&col_a, &col_b),
            plus(&col_x, &col_y),
            true,
            "Binary expressions with equivalent operands should be equal",
        ),
        (
            plus(&col_a, &col_b),
            plus(&col_x, &col_a),
            false,
            "Binary expressions with non-equivalent operands should not be equal",
        ),
        (
            plus(&col_a, &lit_1),
            plus(&col_x, &lit_1),
            true,
            "Equivalent column and same literal should be equal",
        ),
        (
            times(&plus(&col_a, &col_b), &lit_1),
            times(&plus(&col_x, &col_y), &lit_1),
            true,
            "Nested binary expressions with equivalent operands should be equal",
        ),
    ];
    for (left, right, expected, description) in compound {
        assert_eq!(eq.are_equal(&left, &right), expected, "{description}");
    }
    Ok(())
}

/// With `a = b`, the projected `a + c` and `b + c` are equal.
#[test]
fn test_project_classes() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("c", DataType::Int32, false),
    ]));
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
    eq_properties.add_equal(col("a", &schema)?, col("b", &schema)?);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("a+c", DataType::Int32, false),
        Field::new("b+c", DataType::Int32, false),
    ]));
    let a_plus_c = binary(
        col("a", &schema)?,
        Operator::Plus,
        col("c", &schema)?,
        &schema,
    )?;
    let b_plus_c = binary(
        col("b", &schema)?,
        Operator::Plus,
        col("c", &schema)?,
        &schema,
    )?;
    let mapping = [
        (a_plus_c, vec![(col("a+c", &projected_schema)?, 0)].into()),
        (b_plus_c, vec![(col("b+c", &projected_schema)?, 1)].into()),
    ]
    .into_iter()
    .collect::<ProjectionMapping>();

    let projected = eq_properties.project(&mapping, Arc::clone(&projected_schema));
    let eq = projected.equalities();
    assert!(!eq.is_empty());
    assert!(eq.are_equal(
        &col("a+c", &projected_schema)?,
        &col("b+c", &projected_schema)?
    ));
    Ok(())
}

// ---------------------------------------------------------------------------
// Ordering satisfaction
// ---------------------------------------------------------------------------

/// A finer ordering satisfies a coarser one, not the other way around.
#[test]
fn test_ordering_satisfy() -> Result<()> {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, true),
    ]));
    let crude = vec![PhysicalSortExpr::new_default(column("a", 0))];
    let finer = vec![
        PhysicalSortExpr::new_default(column("a", 0)),
        PhysicalSortExpr::new_default(column("b", 1)),
    ];
    let eq_properties_finer = EquivalenceProperties::new_with_orderings(
        Arc::clone(&input_schema),
        [finer.clone()],
    );
    assert!(eq_properties_finer.ordering_satisfy(crude.clone()));

    let eq_properties_crude =
        EquivalenceProperties::new_with_orderings(Arc::clone(&input_schema), [crude]);
    assert!(!eq_properties_crude.ordering_satisfy(finer));
    Ok(())
}

/// A strictly order-preserving image (widening cast) of a leading key keeps
/// the following keys ordered; a merely monotonic one (narrowing cast) does
/// not.
#[test]
fn test_ordering_satisfy_strictly_order_preserving() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int64, true),
    ]));
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    let asc = SortOptions::default();
    let sort_a = PhysicalSortExpr::new(Arc::clone(&col_a), asc);
    let sort_b = PhysicalSortExpr::new(Arc::clone(&col_b), asc);
    let eq_properties = EquivalenceProperties::new_with_orderings(
        Arc::clone(&schema),
        [vec![sort_a.clone(), sort_b.clone()]],
    );

    assert!(eq_properties.ordering_satisfy(vec![sort_a.clone(), sort_b.clone()]));
    assert!(eq_properties.ordering_satisfy(vec![sort_a.clone()]));

    let widening = Arc::new(CastExpr::new(Arc::clone(&col_a), DataType::Int64, None))
        as PhysicalExprRef;
    let sort_widening = PhysicalSortExpr::new(widening, asc);
    assert!(eq_properties.ordering_satisfy(vec![sort_widening, sort_b.clone()]));

    let narrowing = Arc::new(CastExpr::new(Arc::clone(&col_a), DataType::Int16, None))
        as PhysicalExprRef;
    let sort_narrowing = PhysicalSortExpr::new(narrowing, asc);
    assert!(eq_properties.ordering_satisfy(vec![sort_narrowing.clone()]));
    assert!(!eq_properties.ordering_satisfy(vec![sort_narrowing, sort_b]));

    Ok(())
}

/// Requirements over monotonic functions (`floor`-like UDF, `a + b`), with
/// an equality `a = f` and a constant `e`.
#[test]
fn test_ordering_satisfy_with_equivalence2() -> Result<()> {
    let test_schema = create_test_schema()?;
    let col_a = &col("a", &test_schema)?;
    let col_b = &col("b", &test_schema)?;
    let col_c = &col("c", &test_schema)?;
    let col_d = &col("d", &test_schema)?;
    let col_e = &col("e", &test_schema)?;
    let col_f = &col("f", &test_schema)?;
    let test_fun = Arc::new(ScalarUDF::new_from_impl(TestScalarUDF::new()));
    let udf = |arg: &Arc<dyn PhysicalExpr>| -> Result<PhysicalExprRef> {
        Ok(Arc::new(ScalarFunctionExpr::try_new(
            Arc::clone(&test_fun),
            vec![Arc::clone(arg)],
            &test_schema,
            Arc::new(ConfigOptions::default()),
        )?) as _)
    };
    let floor_a = udf(col_a)?;
    let floor_f = udf(col_f)?;
    let exp_a = udf(col_a)?;
    let a_plus_b = Arc::new(BinaryExpr::new(
        Arc::clone(col_a),
        Operator::Plus,
        Arc::clone(col_b),
    )) as Arc<dyn PhysicalExpr>;
    let options = SortOptions {
        descending: false,
        nulls_first: false,
    };

    // (orderings, requirement, expected); every case has a = f and constant e.
    let test_cases = vec![
        // 1: [a, b] is not satisfied by [a, d, b], [c]
        (
            vec![
                vec![(col_a, options), (col_d, options), (col_b, options)],
                vec![(col_c, options)],
            ],
            vec![(col_a, options), (col_b, options)],
            false,
        ),
        // 2: [floor(a)] is satisfied
        (
            vec![
                vec![(col_a, options), (col_c, options), (col_b, options)],
                vec![(col_d, options)],
            ],
            vec![(&floor_a, options)],
            true,
        ),
        // 2.1: [floor(f)] is satisfied (a = f)
        (
            vec![
                vec![(col_a, options), (col_c, options), (col_b, options)],
                vec![(col_d, options)],
            ],
            vec![(&floor_f, options)],
            true,
        ),
        // 3: [a, c, a + b] is not satisfied because addition can wrap
        (
            vec![
                vec![(col_a, options), (col_c, options), (col_b, options)],
                vec![(col_d, options)],
            ],
            vec![(col_a, options), (col_c, options), (&a_plus_b, options)],
            false,
        ),
        // 4: [floor(a), a + b]
        (
            vec![vec![
                (col_a, options),
                (col_b, options),
                (col_c, options),
                (col_d, options),
            ]],
            vec![(&floor_a, options), (&a_plus_b, options)],
            false,
        ),
        // 5: [exp(a), a + b]
        (
            vec![vec![
                (col_a, options),
                (col_b, options),
                (col_c, options),
                (col_d, options),
            ]],
            vec![(&exp_a, options), (&a_plus_b, options)],
            false,
        ),
        // 6: [a, d, floor(a)] is satisfied
        (
            vec![
                vec![(col_a, options), (col_d, options), (col_b, options)],
                vec![(col_c, options)],
            ],
            vec![(col_a, options), (col_d, options), (&floor_a, options)],
            true,
        ),
        // 7: [a, floor(a), a + b] is not satisfied
        (
            vec![
                vec![(col_a, options), (col_c, options), (col_b, options)],
                vec![(col_d, options)],
            ],
            vec![(col_a, options), (&floor_a, options), (&a_plus_b, options)],
            false,
        ),
        // 8: [a, c, floor(a), a + b] is not satisfied
        (
            vec![
                vec![(col_a, options), (col_b, options), (col_c, options)],
                vec![(col_d, options)],
            ],
            vec![
                (col_a, options),
                (col_c, options),
                (&floor_a, options),
                (&a_plus_b, options),
            ],
            false,
        ),
        // 9: [a, b, c, floor(a)] is satisfied
        (
            vec![vec![
                (col_a, options),
                (col_b, options),
                (col_c, options),
                (col_d, options),
            ]],
            vec![
                (col_a, options),
                (col_b, options),
                (col_c, options),
                (&floor_a, options),
            ],
            true,
        ),
        // 10: [c, d, a + b] is not satisfied because addition can wrap
        (
            vec![
                vec![(col_d, options), (col_b, options)],
                vec![(col_c, options), (col_a, options)],
            ],
            vec![(col_c, options), (col_d, options), (&a_plus_b, options)],
            false,
        ),
    ];

    for (orderings, reqs, expected) in test_cases {
        let err_msg = format!(
            "error in test orderings: {orderings:?}, reqs: {reqs:?}, expected: {expected:?}"
        );
        let mut eq_properties = EquivalenceProperties::new(Arc::clone(&test_schema));
        eq_properties.add_orderings(convert_to_orderings(&orderings));
        eq_properties.add_equal(Arc::clone(col_a), Arc::clone(col_f));
        eq_properties.add_constant(Arc::clone(col_e), None);
        let reqs = convert_to_sort_exprs(&reqs);
        assert_eq!(eq_properties.ordering_satisfy(reqs), expected, "{err_msg}");
    }

    Ok(())
}

/// Requirements longer or shorter than the provided orderings, with `a = c`.
#[test]
fn test_ordering_satisfy_different_lengths() -> Result<()> {
    let test_schema = create_test_schema()?;
    let col_a = &col("a", &test_schema)?;
    let col_b = &col("b", &test_schema)?;
    let col_c = &col("c", &test_schema)?;
    let col_d = &col("d", &test_schema)?;
    let col_e = &col("e", &test_schema)?;
    let col_f = &col("f", &test_schema)?;
    let options = SortOptions {
        descending: false,
        nulls_first: false,
    };
    // a=c (e.g they are aliases).
    let mut eq_properties = EquivalenceProperties::new(test_schema);
    eq_properties.add_equal(Arc::clone(col_a), Arc::clone(col_c));

    // [a ASC], [e ASC], [d ASC, f ASC]
    let orderings = vec![
        vec![(col_a, options)],
        vec![(col_e, options)],
        vec![(col_d, options), (col_f, options)],
    ];
    eq_properties.add_orderings(convert_to_orderings(&orderings));

    let test_cases = vec![
        (
            vec![(col_c, options), (col_a, options), (col_e, options)],
            true,
        ),
        (vec![(col_c, options), (col_b, options)], false),
        (vec![(col_c, options), (col_d, options)], true),
        (
            vec![(col_d, options), (col_f, options), (col_b, options)],
            false,
        ),
        (vec![(col_d, options), (col_f, options)], true),
    ];

    for (reqs, expected) in test_cases {
        let err_msg = format!("error in test reqs: {reqs:?}, expected: {expected:?}");
        let reqs = convert_to_sort_exprs(&reqs);
        assert_eq!(eq_properties.ordering_satisfy(reqs), expected, "{err_msg}");
    }

    Ok(())
}

/// Redundant orderings give the same answers as their succinct form (main
/// trimmed them on insert; here only the answers are compared).
#[test]
fn test_remove_redundant_entries_oeq_class() -> Result<()> {
    let schema = create_test_schema()?;
    let col_a = &col("a", &schema)?;
    let col_b = &col("b", &schema)?;
    let col_c = &col("c", &schema)?;
    let col_d = &col("d", &schema)?;
    let col_e = &col("e", &schema)?;

    let asc = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let desc = SortOptions {
        descending: true,
        nulls_first: true,
    };

    // (given orderings, succinct equivalent orderings)
    let test_cases = vec![
        // 1
        (
            vec![vec![(col_a, asc), (col_b, asc)]],
            vec![vec![(col_a, asc), (col_b, asc)]],
        ),
        // 2
        (
            vec![
                vec![(col_a, asc), (col_b, asc)],
                vec![(col_a, asc), (col_b, asc), (col_c, asc)],
            ],
            vec![vec![(col_a, asc), (col_b, asc), (col_c, asc)]],
        ),
        // 3
        (
            vec![
                vec![(col_a, asc), (col_b, desc)],
                vec![(col_a, asc)],
                vec![(col_a, asc), (col_c, asc)],
            ],
            vec![
                vec![(col_a, asc), (col_b, desc)],
                vec![(col_a, asc), (col_c, asc)],
            ],
        ),
        // 4
        (
            vec![
                vec![(col_a, asc), (col_b, asc)],
                vec![(col_a, asc), (col_b, asc), (col_c, asc)],
                vec![(col_a, asc)],
            ],
            vec![vec![(col_a, asc), (col_b, asc), (col_c, asc)]],
        ),
        // 5: empty ordering is ignored
        (vec![], vec![]),
        // 6
        (
            vec![vec![(col_a, asc), (col_b, asc)], vec![(col_b, asc)]],
            vec![vec![(col_a, asc)], vec![(col_b, asc)]],
        ),
        // 7
        (
            vec![
                vec![(col_b, asc), (col_a, asc)],
                vec![(col_c, asc), (col_a, asc)],
                vec![(col_d, asc), (col_b, asc), (col_c, asc)],
            ],
            vec![
                vec![(col_b, asc), (col_a, asc)],
                vec![(col_c, asc), (col_a, asc)],
                vec![(col_d, asc)],
            ],
        ),
        // 8
        (
            vec![
                vec![(col_b, asc), (col_e, asc)],
                vec![(col_c, asc), (col_a, asc)],
                vec![
                    (col_d, asc),
                    (col_b, asc),
                    (col_e, asc),
                    (col_c, asc),
                    (col_a, asc),
                ],
            ],
            vec![
                vec![(col_b, asc), (col_e, asc)],
                vec![(col_c, asc), (col_a, asc)],
                vec![(col_d, asc)],
            ],
        ),
        // 9
        (
            vec![
                vec![(col_b, asc)],
                vec![(col_a, asc), (col_b, asc), (col_c, asc)],
                vec![(col_d, asc), (col_a, asc), (col_b, asc)],
            ],
            vec![
                vec![(col_b, asc)],
                vec![(col_a, asc), (col_b, asc), (col_c, asc)],
                vec![(col_d, asc)],
            ],
        ),
    ];
    for (orderings, expected) in test_cases {
        let orderings = convert_to_orderings(&orderings);
        let expected = convert_to_orderings(&expected);
        let actual =
            EquivalenceProperties::new_with_orderings(Arc::clone(&schema), orderings);
        assert_orderings_equivalent(&actual, &expected, "redundant orderings");
    }

    Ok(())
}

/// PRIMARY KEY / UNIQUE columns leading an ordering satisfy any extension of
/// it; a nullable UNIQUE key does not.
#[test]
fn test_ordering_satisfaction_with_key_constraints() -> Result<()> {
    let pk_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Int32, true),
        Field::new("d", DataType::Int32, true),
    ]));

    let unique_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("c", DataType::Int32, true),
        Field::new("d", DataType::Int32, true),
    ]));

    // (name, schema, constraint, base_ordering, satisfied_orderings, unsatisfied_orderings)
    let test_cases = vec![
        (
            "single column primary key",
            &pk_schema,
            vec![Constraint::PrimaryKey(vec![0])],
            vec!["a"],
            vec![vec!["a", "b"], vec!["a", "c", "d"]],
            vec![vec!["b", "a"], vec!["c", "a"]],
        ),
        (
            "single column unique",
            &unique_schema,
            vec![Constraint::Unique(vec![0])],
            vec!["a"],
            vec![vec!["a", "b"], vec!["a", "c", "d"]],
            vec![vec!["b", "a"], vec!["c", "a"]],
        ),
        (
            "multi-column primary key",
            &pk_schema,
            vec![Constraint::PrimaryKey(vec![0, 1])],
            vec!["a", "b"],
            vec![vec!["a", "b", "c"], vec!["a", "b", "d"]],
            vec![vec!["b", "a"], vec!["a", "c", "b"]],
        ),
        (
            "multi-column unique",
            &unique_schema,
            vec![Constraint::Unique(vec![0, 1])],
            vec!["a", "b"],
            vec![vec!["a", "b", "c"], vec!["a", "b", "d"]],
            vec![vec!["b", "a"], vec!["c", "a", "b"]],
        ),
        (
            "nullable unique",
            &unique_schema,
            vec![Constraint::Unique(vec![2, 3])],
            vec!["c", "d"],
            vec![],
            vec![vec!["c", "d", "a"]],
        ),
        (
            "ordering with arbitrary column unique",
            &unique_schema,
            vec![Constraint::Unique(vec![0, 1])],
            vec!["a", "c", "b"],
            vec![vec!["a", "c", "b", "d"]],
            vec![vec!["a", "b", "d"]],
        ),
        (
            "ordering with arbitrary column pk",
            &pk_schema,
            vec![Constraint::PrimaryKey(vec![0, 1])],
            vec!["a", "c", "b"],
            vec![vec!["a", "c", "b", "d"]],
            vec![vec!["a", "b", "d"]],
        ),
        (
            "ordering with arbitrary column pk complex",
            &pk_schema,
            vec![Constraint::PrimaryKey(vec![3, 1])],
            vec!["b", "a", "d"],
            vec![vec!["b", "a", "d", "c"]],
            vec![vec!["b", "c", "d", "a"], vec!["b", "a", "c", "d"]],
        ),
    ];

    for (name, schema, constraints, base_order, satisfied, unsatisfied) in test_cases {
        let to_ordering = |cols: &Vec<&str>| {
            cols.iter()
                .map(|c| PhysicalSortExpr::new_default(col(c, schema).unwrap()))
                .collect::<Vec<_>>()
        };
        let mut eq_properties = EquivalenceProperties::new(Arc::clone(schema));
        for ordering in satisfied.iter().map(to_ordering) {
            assert!(
                !eq_properties.ordering_satisfy(ordering.clone()),
                "{name}: ordering {ordering:?} should not be satisfied before adding constraints"
            );
        }
        eq_properties.add_ordering(to_ordering(&base_order));
        eq_properties =
            eq_properties.with_constraints(Constraints::new_unverified(constraints));
        for ordering in satisfied.iter().map(to_ordering) {
            assert!(
                eq_properties.ordering_satisfy(ordering.clone()),
                "{name}: ordering {ordering:?} should be satisfied after adding constraints"
            );
        }
        for ordering in unsatisfied.iter().map(to_ordering) {
            assert!(
                !eq_properties.ordering_satisfy(ordering.clone()),
                "{name}: ordering {ordering:?} should not be satisfied after adding constraints"
            );
        }
    }

    Ok(())
}

/// With `[a, b, c]`, constant `b` and `a = CAST(c AS DATE)`, `[c]` holds
/// (OD reduce), whatever order the facts are added in; without the
/// constant it does not.
#[test]
fn test_eliminate_redundant_monotonic_sorts() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Date32, true),
        Field::new("b", DataType::Utf8, true),
        Field::new("c", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
    ]));
    let base_properties =
        EquivalenceProperties::new(Arc::clone(&schema)).reordered(LexOrdering::new(
            ["a", "b", "c"]
                .into_iter()
                .map(|c| PhysicalSortExpr::new_default(col(c, schema.as_ref()).unwrap())),
        ));

    let col_a = col("a", schema.as_ref())?;
    let col_b = col("b", schema.as_ref())?;
    let col_c = col("c", schema.as_ref())?;
    let cast_c = Arc::new(CastExpr::new_with_target_field(
        col_c,
        Arc::new(Field::new("c", DataType::Date32, true)),
        None,
    )) as Arc<dyn PhysicalExpr>;
    let required_sort = vec![PhysicalSortExpr::new_default(col("c", &schema)?)];

    // (name, constants, equal condition, should satisfy)
    let cases = vec![
        (
            "cast_c = a",
            vec![Arc::clone(&col_b)],
            [Arc::clone(&cast_c), Arc::clone(&col_a)],
            true,
        ),
        (
            "a = cast_c",
            vec![col_b],
            [Arc::clone(&col_a), Arc::clone(&cast_c)],
            true,
        ),
        (
            "not ordered because (b) is not constant",
            vec![],
            [Arc::clone(&cast_c), Arc::clone(&col_a)],
            false,
        ),
    ];

    for (name, constants, [left, right], expected) in cases {
        // Equal conditions before constants
        let mut eq_first = base_properties.clone();
        eq_first.add_equal(Arc::clone(&left), Arc::clone(&right));
        for c in &constants {
            eq_first.add_local_constant(Arc::clone(c));
        }
        // Constants before equal conditions
        let mut constants_first = base_properties.clone();
        for c in &constants {
            constants_first.add_local_constant(Arc::clone(c));
        }
        constants_first.add_equal(left, right);
        for properties in [eq_first, constants_first] {
            assert_eq!(
                properties.ordering_satisfy(required_sort.clone()),
                expected,
                "failed test '{name}'"
            );
        }
    }

    Ok(())
}

/// `c = concat(a, b)` is not lexicographic, so `[c, a, b]` gives no
/// ordering on `[a, b]`.
#[test]
fn test_ordering_equivalence_with_non_lex_monotonic_concat() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Utf8, false),
        Field::new("c", DataType::Utf8, false),
    ]));
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    let col_c = col("c", &schema)?;
    let a_concat_b: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
        "concat",
        concat(),
        vec![Arc::clone(&col_a), Arc::clone(&col_b)],
        Field::new("f", DataType::Utf8, true).into(),
        Arc::new(ConfigOptions::default()),
    ));

    let initial_ordering: LexOrdering = [
        PhysicalSortExpr::new_default(Arc::clone(&col_c)).asc(),
        PhysicalSortExpr::new_default(Arc::clone(&col_a)).asc(),
        PhysicalSortExpr::new_default(Arc::clone(&col_b)).asc(),
    ]
    .into();
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
    eq_properties.add_ordering(initial_ordering.clone());
    eq_properties.add_equal(Arc::clone(&col_c), a_concat_b);

    // The ordering should remain unchanged since concat is not lex-monotonic
    assert_orderings_equivalent(&eq_properties, &[initial_ordering], "concat");
    assert!(!eq_properties.ordering_satisfy([
        PhysicalSortExpr::new_default(col_a).asc(),
        PhysicalSortExpr::new_default(col_b).asc(),
    ]));

    Ok(())
}

/// `c = a * b` is not lexicographic, so `[c, a, b]` gives no ordering on
/// `[a, b]`.
#[test]
fn test_ordering_equivalence_with_non_lex_monotonic_multiply() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("c", DataType::Int32, false),
    ]));
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    let col_c = col("c", &schema)?;
    let a_times_b = Arc::new(BinaryExpr::new(
        Arc::clone(&col_a),
        Operator::Multiply,
        Arc::clone(&col_b),
    )) as _;

    let initial_ordering: LexOrdering = [
        PhysicalSortExpr::new_default(Arc::clone(&col_c)).asc(),
        PhysicalSortExpr::new_default(Arc::clone(&col_a)).asc(),
        PhysicalSortExpr::new_default(Arc::clone(&col_b)).asc(),
    ]
    .into();
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
    eq_properties.add_ordering(initial_ordering.clone());
    eq_properties.add_equal(col_c, a_times_b);

    // The ordering should remain unchanged since multiplication is not lex-monotonic
    assert_orderings_equivalent(&eq_properties, &[initial_ordering], "multiply");
    assert!(!eq_properties.ordering_satisfy([
        PhysicalSortExpr::new_default(col_a).asc(),
        PhysicalSortExpr::new_default(col_b).asc(),
    ]));

    Ok(())
}

/// `named_struct('a', a, ..) AS s` ordered by `a` is ordered by `s['a']`.
#[test]
fn test_ordering_propagation_through_named_struct() -> Result<()> {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]));
    let col_a = col("a", &input_schema)?;
    let col_b = col("b", &input_schema)?;
    let config = Arc::new(ConfigOptions::new());
    let utf8 = |s: &str| {
        Arc::new(Literal::new(ScalarValue::Utf8(Some(s.to_string()))))
            as Arc<dyn PhysicalExpr>
    };

    // named_struct('a', col_a, 'b', col_b) AS s
    let named_struct_expr = Arc::new(ScalarFunctionExpr::new(
        "named_struct",
        named_struct(),
        vec![utf8("a"), Arc::clone(&col_a), utf8("b"), Arc::clone(&col_b)],
        Arc::new(Field::new(
            "named_struct",
            DataType::Struct(
                vec![
                    Field::new("a", DataType::Int32, true),
                    Field::new("b", DataType::Int32, true),
                ]
                .into(),
            ),
            true,
        )),
        Arc::clone(&config),
    )) as Arc<dyn PhysicalExpr>;
    let proj_exprs = vec![(named_struct_expr, "s".to_string())];
    let projection_mapping = ProjectionMapping::try_new(proj_exprs, &input_schema)?;

    let asc = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let mut input_properties = EquivalenceProperties::new(Arc::clone(&input_schema));
    input_properties.add_ordering([PhysicalSortExpr::new(Arc::clone(&col_a), asc)]);

    let out_schema = output_schema(&projection_mapping, &input_schema)?;
    let out_properties = input_properties.project(&projection_mapping, out_schema);

    // get_field(col("s"), "a"), what ORDER BY s.a plans to
    let get_field_expr = Arc::new(ScalarFunctionExpr::new(
        "get_field",
        get_field(),
        vec![column("s", 0), utf8("a")],
        Arc::new(Field::new("a", DataType::Int32, true)),
        Arc::clone(&config),
    )) as Arc<dyn PhysicalExpr>;
    assert!(
        out_properties.ordering_satisfy([PhysicalSortExpr::new(get_field_expr, asc)]),
        "Output should be ordered by get_field(s, 'a') since input is ordered by col_a"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Expression properties, longest permutation, normalization
// ---------------------------------------------------------------------------

/// How expressions are ordered given `[b]`, `[d]` and `b = a`.
#[test]
fn test_update_properties() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Int32, true),
        Field::new("d", DataType::Int32, true),
    ]);
    let mut eq_properties = EquivalenceProperties::new(Arc::new(schema.clone()));
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    let col_c = col("c", &schema)?;
    let col_d = col("d", &schema)?;
    let option_asc = SortOptions {
        descending: false,
        nulls_first: false,
    };
    eq_properties.add_equal(Arc::clone(&col_b), Arc::clone(&col_a));
    eq_properties.add_orderings([
        vec![PhysicalSortExpr::new(Arc::clone(&col_b), option_asc)],
        vec![PhysicalSortExpr::new(Arc::clone(&col_d), option_asc)],
    ]);

    let test_cases = vec![
        // d + b can wrap
        (
            Arc::new(BinaryExpr::new(col_d, Operator::Plus, Arc::clone(&col_b))) as _,
            SortProperties::Unordered,
        ),
        (col_b, SortProperties::Ordered(option_asc)),
        (Arc::clone(&col_a), SortProperties::Ordered(option_asc)),
        // a + c
        (
            Arc::new(BinaryExpr::new(col_a, Operator::Plus, col_c)),
            SortProperties::Unordered,
        ),
    ];
    for (expr, expected) in test_cases {
        let actual = eq_properties.get_expr_properties(&expr);
        assert_eq!(
            actual.sort_properties, expected,
            "expr: {expr:?}, properties: {eq_properties}"
        );
    }

    Ok(())
}

/// The longest permutation of `[b, a]` given `[b DESC, a ASC]`, also next to
/// an unrelated ordering, and when `c` sits between them.
#[test]
fn test_get_indices_of_matching_sort_exprs_with_order_eq() -> Result<()> {
    let sort_options = SortOptions::default();
    let sort_options_not = !SortOptions::default();

    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]);
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    let required_columns = [Arc::clone(&col_b), Arc::clone(&col_a)];
    let mut eq_properties = EquivalenceProperties::new(Arc::new(schema));
    eq_properties.add_ordering([
        PhysicalSortExpr::new(column("b", 1), sort_options_not),
        PhysicalSortExpr::new(column("a", 0), sort_options),
    ]);
    let (result, idxs) = eq_properties.longest_permutation(&required_columns);
    assert_eq!(idxs, vec![0, 1]);
    assert_eq!(
        result,
        vec![
            PhysicalSortExpr::new(Arc::clone(&col_b), sort_options_not),
            PhysicalSortExpr::new(Arc::clone(&col_a), sort_options),
        ]
    );

    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Int32, true),
    ]);
    let mut eq_properties = EquivalenceProperties::new(Arc::new(schema.clone()));
    eq_properties.add_orderings([
        vec![PhysicalSortExpr::new(column("c", 2), sort_options)],
        vec![
            PhysicalSortExpr::new(column("b", 1), sort_options_not),
            PhysicalSortExpr::new(column("a", 0), sort_options),
        ],
    ]);
    let (result, idxs) = eq_properties.longest_permutation(&required_columns);
    assert_eq!(idxs, vec![0, 1]);
    assert_eq!(
        result,
        vec![
            PhysicalSortExpr::new(col_b, sort_options_not),
            PhysicalSortExpr::new(col_a, sort_options),
        ]
    );

    let required_columns = [column("b", 1), column("a", 0)];
    let mut eq_properties = EquivalenceProperties::new(Arc::new(schema));
    // not satisfied orders
    eq_properties.add_ordering([
        PhysicalSortExpr::new(column("b", 1), sort_options_not),
        PhysicalSortExpr::new(column("c", 2), sort_options),
        PhysicalSortExpr::new(column("a", 0), sort_options),
    ]);
    let (_, idxs) = eq_properties.longest_permutation(&required_columns);
    assert_eq!(idxs, vec![0]);

    Ok(())
}

/// Longest permutations over the test params plus `[d ASC, h DESC]`.
#[test]
fn test_find_longest_permutation() -> Result<()> {
    // [a ASC], [d ASC, b ASC], [e DESC, f ASC, g ASC], a = c
    let (test_schema, mut eq_properties) = create_test_params()?;
    let col_a = &col("a", &test_schema)?;
    let col_b = &col("b", &test_schema)?;
    let col_c = &col("c", &test_schema)?;
    let col_d = &col("d", &test_schema)?;
    let col_e = &col("e", &test_schema)?;
    let col_f = &col("f", &test_schema)?;
    let col_h = &col("h", &test_schema)?;
    let a_plus_d = Arc::new(BinaryExpr::new(
        Arc::clone(col_a),
        Operator::Plus,
        Arc::clone(col_d),
    )) as _;

    let option_asc = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let option_desc = SortOptions {
        descending: true,
        nulls_first: true,
    };
    eq_properties.add_ordering([
        PhysicalSortExpr::new(Arc::clone(col_d), option_asc),
        PhysicalSortExpr::new(Arc::clone(col_h), option_desc),
    ]);
    let test_cases = vec![
        (vec![col_a], vec![(col_a, option_asc)]),
        (vec![col_c], vec![(col_c, option_asc)]),
        (
            vec![col_d, col_e, col_b],
            vec![
                (col_d, option_asc),
                (col_e, option_desc),
                (col_b, option_asc),
            ],
        ),
        (vec![col_b], vec![]),
        (vec![col_d], vec![(col_d, option_asc)]),
        // a + d is not ordered because addition can wrap.
        (vec![&a_plus_d], vec![]),
        (
            vec![col_b, col_d],
            vec![(col_d, option_asc), (col_b, option_asc)],
        ),
        (
            vec![col_c, col_e],
            vec![(col_c, option_asc), (col_e, option_desc)],
        ),
        (
            vec![col_d, col_h, col_e, col_f, col_b],
            vec![
                (col_d, option_asc),
                (col_e, option_desc),
                (col_h, option_desc),
                (col_f, option_asc),
                (col_b, option_asc),
            ],
        ),
        (
            vec![col_e, col_d, col_h, col_f, col_b],
            vec![
                (col_e, option_desc),
                (col_d, option_asc),
                (col_h, option_desc),
                (col_f, option_asc),
                (col_b, option_asc),
            ],
        ),
        (
            vec![col_e, col_d, col_b, col_h, col_f],
            vec![
                (col_e, option_desc),
                (col_d, option_asc),
                (col_b, option_asc),
                (col_h, option_desc),
                (col_f, option_asc),
            ],
        ),
    ];
    for (exprs, expected) in test_cases {
        let exprs = exprs.into_iter().cloned().collect::<Vec<_>>();
        let expected = convert_to_sort_exprs(&expected);
        let (actual, _) = eq_properties.longest_permutation(&exprs);
        assert_eq!(actual, expected, "exprs: {exprs:?}");
    }

    Ok(())
}

/// A constant comes back from the longest permutation with default options.
#[test]
fn test_find_longest_permutation2() -> Result<()> {
    let (test_schema, mut eq_properties) = create_test_params()?;
    let col_h = &col("h", &test_schema)?;
    eq_properties.add_local_constant(Arc::clone(col_h));

    let (actual, _) = eq_properties.longest_permutation(&[Arc::clone(col_h)]);
    assert_eq!(
        actual,
        convert_to_sort_exprs(&[(col_h, SortOptions::default())])
    );

    Ok(())
}

/// Normalizing requirements replaces class members and keeps `Group` steps.
#[test]
fn test_normalize_sort_reqs() -> Result<()> {
    // a = c; [a ASC], [d ASC, b ASC], [e DESC, f ASC, g ASC]
    let (test_schema, eq_properties) = create_test_params()?;
    let col_a = &col("a", &test_schema)?;
    let col_b = &col("b", &test_schema)?;
    let col_c = &col("c", &test_schema)?;
    let col_d = &col("d", &test_schema)?;
    let col_e = &col("e", &test_schema)?;
    let col_f = &col("f", &test_schema)?;
    let option_asc = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let option_desc = SortOptions {
        descending: true,
        nulls_first: true,
    };
    // (requirement, expected normal form)
    let requirements = vec![
        (
            vec![(col_a, Some(option_asc))],
            vec![(col_a, Some(option_asc))],
        ),
        (
            vec![(col_a, Some(option_desc))],
            vec![(col_a, Some(option_desc))],
        ),
        (vec![(col_a, None)], vec![(col_a, None)]),
        (
            vec![(col_c, Some(option_asc))],
            vec![(col_a, Some(option_asc))],
        ),
        (vec![(col_c, None)], vec![(col_a, None)]),
        (
            vec![(col_d, Some(option_asc)), (col_b, Some(option_asc))],
            vec![(col_d, Some(option_asc)), (col_b, Some(option_asc))],
        ),
        (
            vec![(col_d, None), (col_b, None)],
            vec![(col_d, None), (col_b, None)],
        ),
        (
            vec![(col_e, Some(option_desc)), (col_f, Some(option_asc))],
            vec![(col_e, Some(option_desc)), (col_f, Some(option_asc))],
        ),
        (
            vec![(col_e, Some(option_desc)), (col_f, None)],
            vec![(col_e, Some(option_desc)), (col_f, None)],
        ),
        (
            vec![(col_e, None), (col_f, None)],
            vec![(col_e, None), (col_f, None)],
        ),
    ];

    for (reqs, expected) in requirements {
        let req = convert_to_sort_reqs(&reqs);
        let expected = convert_to_sort_reqs(&expected);
        assert_eq!(eq_properties.normalize_requirement(req), expected);
    }

    Ok(())
}

/// Normalizing a requirement replaces `c` by `a` (a = c).
#[test]
fn test_schema_normalize_sort_requirement_with_equivalence() -> Result<()> {
    let option1 = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let (test_schema, eq_properties) = create_test_params()?;
    let col_a = &col("a", &test_schema)?;
    let col_c = &col("c", &test_schema)?;
    let col_d = &col("d", &test_schema)?;

    let test_cases = vec![
        (vec![(col_a, Some(option1))], vec![(col_a, Some(option1))]),
        (vec![(col_c, Some(option1))], vec![(col_a, Some(option1))]),
        (vec![(col_c, None)], vec![(col_a, None)]),
        (vec![(col_d, Some(option1))], vec![(col_d, Some(option1))]),
    ];
    for (reqs, expected) in test_cases {
        let reqs = convert_to_sort_reqs(&reqs);
        let expected = convert_to_sort_reqs(&expected);
        let normalized = eq_properties.normalize_requirement(reqs.clone());
        assert_eq!(
            expected, normalized,
            "error in test: reqs: {reqs:?}, expected: {expected:?}, normalized: {normalized:?}"
        );
    }

    Ok(())
}

/// A requirement is compatible with a reference it equals or extends.
#[test]
fn test_requirements_compatible() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Int32, true),
    ]));
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    let col_c = col("c", &schema)?;

    let eq_properties = EquivalenceProperties::new(schema);
    let lex_a = convert_to_sort_reqs(&[(&col_a, None)]);
    let lex_a_b = convert_to_sort_reqs(&[(&col_a, None), (&col_b, None)]);
    let lex_c = convert_to_sort_reqs(&[(&col_c, None)]);

    assert!(eq_properties.requirements_compatible(lex_a.clone(), lex_a.clone()));
    assert!(!eq_properties.requirements_compatible(lex_a.clone(), lex_a_b.clone()));
    assert!(eq_properties.requirements_compatible(lex_a_b, lex_a.clone()));
    assert!(!eq_properties.requirements_compatible(lex_c, lex_a));

    Ok(())
}

// ---------------------------------------------------------------------------
// Reorder (a sort within each partition)
// ---------------------------------------------------------------------------

/// Sorting by `[a, b]` with `a` constant replaces the (empty) orderings.
#[test]
fn test_with_reorder_constant_filtering() -> Result<()> {
    let schema = create_test_schema()?;
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    eq_properties.add_local_constant(Arc::clone(&col_a));

    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new_default(Arc::clone(&col_a)),
        PhysicalSortExpr::new_default(Arc::clone(&col_b)),
    ]);
    let expected = sort_exprs.clone().unwrap();
    let eq_properties = eq_properties.reordered(sort_exprs);
    assert_orderings_equivalent(&eq_properties, &[expected], "reorder [a, b]");

    Ok(())
}

/// Sorting by a prefix of the existing ordering keeps the whole ordering.
#[test]
fn test_with_reorder_preserve_suffix() -> Result<()> {
    let schema = create_test_schema()?;
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    let col_c = col("c", &schema)?;
    let asc = SortOptions::default();
    let desc = SortOptions {
        descending: true,
        nulls_first: true,
    };

    // Initial ordering: [a ASC, b DESC, c ASC]
    let initial: LexOrdering = [
        PhysicalSortExpr::new(Arc::clone(&col_a), asc),
        PhysicalSortExpr::new(Arc::clone(&col_b), desc),
        PhysicalSortExpr::new(Arc::clone(&col_c), asc),
    ]
    .into();
    eq_properties.add_ordering(initial.clone());

    // New ordering: [a ASC]
    let new_order = LexOrdering::new([PhysicalSortExpr::new(Arc::clone(&col_a), asc)]);
    let eq_properties = eq_properties.reordered(new_order);
    assert_orderings_equivalent(&eq_properties, &[initial], "reorder [a]");

    Ok(())
}

/// Sorting by `[b]` with `a = b` and `[a, c]` keeps `[a, c]`.
#[test]
fn test_with_reorder_equivalent_expressions() -> Result<()> {
    let schema = create_test_schema()?;
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    let col_c = col("c", &schema)?;

    eq_properties.add_equal(Arc::clone(&col_a), Arc::clone(&col_b));
    let initial: LexOrdering = [
        PhysicalSortExpr::new_default(Arc::clone(&col_a)),
        PhysicalSortExpr::new_default(Arc::clone(&col_c)),
    ]
    .into();
    eq_properties.add_ordering(initial.clone());

    let new_order = LexOrdering::new([PhysicalSortExpr::new_default(col_b)]);
    let eq_properties = eq_properties.reordered(new_order);
    assert_orderings_equivalent(&eq_properties, &[initial], "reorder [b]");

    Ok(())
}

/// Sorting by `[a DESC]` over `[a ASC, b DESC]` leaves only `[a DESC]`.
#[test]
fn test_with_reorder_incompatible_prefix() -> Result<()> {
    let schema = create_test_schema()?;
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    let asc = SortOptions::default();
    let desc = SortOptions {
        descending: true,
        nulls_first: true,
    };

    eq_properties.add_ordering([
        PhysicalSortExpr::new(Arc::clone(&col_a), asc),
        PhysicalSortExpr::new(Arc::clone(&col_b), desc),
    ]);
    let new_order: LexOrdering = [PhysicalSortExpr::new(Arc::clone(&col_a), desc)].into();
    let eq_properties = eq_properties.reordered(Some(new_order.clone()));
    assert_orderings_equivalent(&eq_properties, &[new_order], "reorder [a DESC]");

    Ok(())
}

/// Sorting by `[b, c]` that constants, equalities and orderings already
/// satisfy keeps the original orderings.
#[test]
fn test_with_reorder_comprehensive() -> Result<()> {
    let schema = create_test_schema()?;
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
    let col_a = col("a", &schema)?;
    let col_b = col("b", &schema)?;
    let col_c = col("c", &schema)?;
    let col_d = col("d", &schema)?;
    let col_e = col("e", &schema)?;

    eq_properties.add_local_constant(Arc::clone(&col_c));
    eq_properties.add_equal(Arc::clone(&col_b), Arc::clone(&col_d));
    let orderings = vec![
        LexOrdering::from([
            PhysicalSortExpr::new_default(Arc::clone(&col_d)),
            PhysicalSortExpr::new_default(Arc::clone(&col_a)),
        ]),
        LexOrdering::from([PhysicalSortExpr::new_default(Arc::clone(&col_e))]),
    ];
    eq_properties.add_orderings(orderings.clone());

    let new_order = LexOrdering::new([
        PhysicalSortExpr::new_default(Arc::clone(&col_b)),
        PhysicalSortExpr::new_default(Arc::clone(&col_c)),
    ]);
    let eq_properties = eq_properties.reordered(new_order);
    assert_orderings_equivalent(&eq_properties, &orderings, "reorder [b, c]");

    Ok(())
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

/// Four aliases of `a` are equal at the output.
#[test]
fn project_equivalence_properties_test() -> Result<()> {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, true),
        Field::new("c", DataType::Int64, true),
    ]));
    let input_properties = EquivalenceProperties::new(Arc::clone(&input_schema));
    let col_a = col("a", &input_schema)?;

    // a as a1, a as a2, a as a3, a as a4
    let proj_exprs = ["a1", "a2", "a3", "a4"]
        .into_iter()
        .map(|name| (Arc::clone(&col_a), name.to_string()))
        .collect::<Vec<_>>();
    let projection_mapping = ProjectionMapping::try_new(proj_exprs, &input_schema)?;
    let out_schema = output_schema(&projection_mapping, &input_schema)?;
    let out_properties =
        input_properties.project(&projection_mapping, Arc::clone(&out_schema));

    assert_eq!(out_properties.equalities().len(), 1);
    let class = out_properties.equalities().iter().next().unwrap();
    assert_eq!(class.len(), 4);
    for name in ["a1", "a2", "a3", "a4"] {
        assert!(class.contains(&col(name, &out_schema)?));
    }

    Ok(())
}

/// An identity projection keeps both `[a, b, c, d]` and `[a, c, b, d]`.
#[test]
fn project_equivalence_properties_test_multi() -> Result<()> {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, true),
        Field::new("c", DataType::Int64, true),
        Field::new("d", DataType::Int64, true),
    ]));
    let ordering = |names: [&str; 4]| -> LexOrdering {
        names
            .map(|name| parse_sort_expr(name, &input_schema))
            .into()
    };
    let orderings = vec![
        ordering(["a", "b", "c", "d"]),
        ordering(["a", "c", "b", "d"]),
    ];
    let input_properties = EquivalenceProperties::new_with_orderings(
        Arc::clone(&input_schema),
        orderings.clone(),
    );

    let proj_exprs = ["a", "b", "c", "d"]
        .into_iter()
        .map(|name| Ok((col(name, &input_schema)?, name.to_string())))
        .collect::<Result<Vec<_>>>()?;
    let projection_mapping = ProjectionMapping::try_new(proj_exprs, &input_schema)?;
    let out_properties = input_properties.project(&projection_mapping, input_schema);

    assert_orderings_equivalent(&out_properties, &orderings, "identity projection");

    Ok(())
}

/// With `a = c`, orderings `[b]` and `[c]` answer like `[b]` and `[c]`
/// without the equality.
#[test]
fn test_normalize_ordering_equivalence_classes() -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Int32, true),
    ]);
    let col_a_expr = col("a", &schema)?;
    let col_b_expr = col("b", &schema)?;
    let col_c_expr = col("c", &schema)?;
    let mut eq_properties = EquivalenceProperties::new(Arc::new(schema));

    eq_properties.add_equal(col_a_expr, Arc::clone(&col_c_expr));
    let orderings = vec![
        LexOrdering::from([PhysicalSortExpr::new_default(col_b_expr)]),
        LexOrdering::from([PhysicalSortExpr::new_default(col_c_expr)]),
    ];
    eq_properties.add_orderings(orderings.clone());

    assert_orderings_equivalent(&eq_properties, &orderings, "a = c");
    Ok(())
}

// ---------------------------------------------------------------------------
// Join
// ---------------------------------------------------------------------------

/// An inner join probed on the left: each left ordering followed by each
/// right ordering.
#[test]
fn test_join_equivalence_properties() -> Result<()> {
    let schema = create_test_schema()?;
    let col_a = &col("a", &schema)?;
    let col_b = &col("b", &schema)?;
    let col_c = &col("c", &schema)?;
    let offset = schema.fields.len() as _;
    let col_a2 = &add_offset_to_expr(Arc::clone(col_a), offset)?;
    let col_b2 = &add_offset_to_expr(Arc::clone(col_b), offset)?;
    let join_schema = Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .chain(schema.fields().iter())
            .cloned()
            .collect::<Vec<_>>(),
    ));
    let asc = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let test_cases = vec![
        (
            // left: [a], [b]
            vec![vec![(col_a, asc)], vec![(col_b, asc)]],
            // right: [a], [b]
            vec![vec![(col_a, asc)], vec![(col_b, asc)]],
            vec![
                vec![(col_a, asc), (col_a2, asc)],
                vec![(col_a, asc), (col_b2, asc)],
                vec![(col_b, asc), (col_a2, asc)],
                vec![(col_b, asc), (col_b2, asc)],
            ],
        ),
        (
            // left: [a], [b], [c]
            vec![vec![(col_a, asc)], vec![(col_b, asc)], vec![(col_c, asc)]],
            // right: [a], [b]
            vec![vec![(col_a, asc)], vec![(col_b, asc)]],
            vec![
                vec![(col_a, asc), (col_a2, asc)],
                vec![(col_a, asc), (col_b2, asc)],
                vec![(col_b, asc), (col_a2, asc)],
                vec![(col_b, asc), (col_b2, asc)],
                vec![(col_c, asc), (col_a2, asc)],
                vec![(col_c, asc), (col_b2, asc)],
            ],
        ),
    ];
    for (left_orderings, right_orderings, expected) in test_cases {
        let left = EquivalenceProperties::new_with_orderings(
            Arc::clone(&schema),
            convert_to_orderings(&left_orderings),
        );
        let right = EquivalenceProperties::new_with_orderings(
            Arc::clone(&schema),
            convert_to_orderings(&right_orderings),
        );
        let join_eq = EquivalenceProperties::join(
            &left,
            &right,
            &JoinType::Inner,
            Arc::clone(&join_schema),
            &[],
            Some(JoinSide::Left),
            Some(JoinSide::Left),
        )?;
        assert_orderings_equivalent(
            &join_eq,
            &convert_to_orderings(&expected),
            "inner join",
        );
    }
    Ok(())
}

/// The right side's orderings, shifted past the left columns, when the join
/// keeps the right side's order.
#[test]
fn test_get_updated_right_ordering_equivalence_properties() -> Result<()> {
    let left_fields: Fields = ["a", "b", "c", "d"]
        .into_iter()
        .map(|name| Field::new(name, DataType::Int32, true))
        .collect();
    let left_schema = Arc::new(Schema::new(left_fields));
    let child_fields: Fields = ["x", "y", "z", "w"]
        .into_iter()
        .map(|name| Field::new(name, DataType::Int32, true))
        .collect();
    let child_schema = Arc::new(Schema::new(child_fields));
    let option_asc = SortOptions {
        descending: false,
        nulls_first: false,
    };
    // Right child: [x ASC, y ASC], [z ASC, w ASC]
    let child_col = |name| col(name, &child_schema);
    let orderings = vec![
        vec![(child_col("x")?, option_asc), (child_col("y")?, option_asc)],
        vec![(child_col("z")?, option_asc), (child_col("w")?, option_asc)],
    ];
    let right = EquivalenceProperties::new_with_orderings(
        Arc::clone(&child_schema),
        convert_to_orderings(&orderings),
    );
    let left = EquivalenceProperties::new(Arc::clone(&left_schema));

    let fields: Fields = ["a", "b", "c", "d", "x", "y", "z", "w"]
        .into_iter()
        .map(|name| Field::new(name, DataType::Int32, true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    // a = x and d = w
    let on = [
        (col("a", &left_schema)?, child_col("x")?),
        (col("d", &left_schema)?, child_col("w")?),
    ];
    let join_eq = EquivalenceProperties::join(
        &left,
        &right,
        &JoinType::Inner,
        Arc::clone(&schema),
        &on,
        Some(JoinSide::Right),
        None,
    )?;

    let join_col = |name| col(name, &schema);
    let expected = vec![
        vec![(join_col("x")?, option_asc), (join_col("y")?, option_asc)],
        vec![(join_col("z")?, option_asc), (join_col("w")?, option_asc)],
    ];
    assert_orderings_equivalent(
        &join_eq,
        &convert_to_orderings(&expected),
        "right orderings",
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Union
// ---------------------------------------------------------------------------

/// Union of three children: the longest common prefix `[a, b]`.
#[test]
fn test_union_equivalence_properties_multi_children_1() -> Result<()> {
    let schema = create_test_schema()?;
    let schema2 = append_fields(&schema, "1");
    let schema3 = append_fields(&schema, "2");
    UnionEquivalenceTest::new(&schema)
        .with_child_sort(vec![vec!["a", "b", "c"]], &schema)
        .with_child_sort(vec![vec!["a1", "b1", "c1"]], &schema2)
        .with_child_sort(vec![vec!["a2", "b2"]], &schema3)
        .with_expected_sort(vec![vec!["a", "b"]])
        .run()
}

/// Union of three children with the same ordering keeps it.
#[test]
fn test_union_equivalence_properties_multi_children_2() -> Result<()> {
    let schema = create_test_schema()?;
    let schema2 = append_fields(&schema, "1");
    let schema3 = append_fields(&schema, "2");
    UnionEquivalenceTest::new(&schema)
        .with_child_sort(vec![vec!["a", "b", "c"]], &schema)
        .with_child_sort(vec![vec!["a1", "b1", "c1"]], &schema2)
        .with_child_sort(vec![vec!["a2", "b2", "c2"]], &schema3)
        .with_expected_sort(vec![vec!["a", "b", "c"]])
        .run()
}

/// Union where the first child has the shortest ordering.
#[test]
fn test_union_equivalence_properties_multi_children_3() -> Result<()> {
    let schema = create_test_schema()?;
    let schema2 = append_fields(&schema, "1");
    let schema3 = append_fields(&schema, "2");
    UnionEquivalenceTest::new(&schema)
        .with_child_sort(vec![vec!["a", "b"]], &schema)
        .with_child_sort(vec![vec!["a1", "b1", "c1"]], &schema2)
        .with_child_sort(vec![vec!["a2", "b2", "c2"]], &schema3)
        .with_expected_sort(vec![vec!["a", "b"]])
        .run()
}

/// Union with no common prefix has no ordering.
#[test]
fn test_union_equivalence_properties_multi_children_4() -> Result<()> {
    let schema = create_test_schema()?;
    let schema2 = append_fields(&schema, "1");
    let schema3 = append_fields(&schema, "2");
    UnionEquivalenceTest::new(&schema)
        .with_child_sort(vec![vec!["a", "b"]], &schema)
        .with_child_sort(vec![vec!["a1", "b1"]], &schema2)
        .with_child_sort(vec![vec!["b2", "c2"]], &schema3)
        .with_expected_sort(vec![])
        .run()
}

/// Union keeps each of several common orderings.
#[test]
fn test_union_equivalence_properties_multi_children_5() -> Result<()> {
    let schema = create_test_schema()?;
    let schema2 = append_fields(&schema, "1");
    UnionEquivalenceTest::new(&schema)
        .with_child_sort(vec![vec!["a", "b"], vec!["c"]], &schema)
        .with_child_sort(vec![vec!["a1", "b1"], vec!["c1"]], &schema2)
        .with_expected_sort(vec![vec!["a", "b"], vec!["c"]])
        .run()
}

/// A constant of both children stays; a constant of one side lets the other
/// side's ordering through.
#[test]
fn test_union_equivalence_properties_constants_common_constants() -> Result<()> {
    let schema = create_test_schema()?;
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(vec![vec!["a"]], vec!["b", "c"], &schema)
        .with_child_sort_and_const_exprs(vec![vec!["b"]], vec!["a", "c"], &schema)
        .with_expected_sort_and_const_exprs(vec![vec!["a"], vec!["b"]], vec!["c"])
        .run()
}

/// Union of `[a]` and `[a, b]` is `[a]`.
#[test]
fn test_union_equivalence_properties_constants_prefix() -> Result<()> {
    let schema = create_test_schema()?;
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(vec![vec!["a"]], vec![], &schema)
        .with_child_sort_and_const_exprs(vec![vec!["a", "b"]], vec![], &schema)
        .with_expected_sort_and_const_exprs(vec![vec!["a"]], vec![])
        .run()
}

/// Union of `[a ASC]` and `[a DESC]` has no ordering.
#[test]
fn test_union_equivalence_properties_constants_asc_desc_mismatch() -> Result<()> {
    let schema = create_test_schema()?;
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(vec![vec!["a"]], vec![], &schema)
        .with_child_sort_and_const_exprs(vec![vec!["a DESC"]], vec![], &schema)
        .with_expected_sort_and_const_exprs(vec![], vec![])
        .run()
}

/// Children with differently named fields line up by position.
#[test]
fn test_union_equivalence_properties_constants_different_schemas() -> Result<()> {
    let schema = create_test_schema()?;
    let schema2 = append_fields(&schema, "1");
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(vec![vec!["a"]], vec![], &schema)
        .with_child_sort_and_const_exprs(vec![vec!["a1", "b1"]], vec![], &schema2)
        .with_expected_sort_and_const_exprs(vec![vec!["a"]], vec![])
        .run()
}

/// Constants of each side fill the gaps of the other side's ordering.
#[test]
fn test_union_equivalence_properties_constants_fill_gaps() -> Result<()> {
    let schema = create_test_schema()?;
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(vec![vec!["a", "c"]], vec!["b"], &schema)
        .with_child_sort_and_const_exprs(vec![vec!["b", "c"]], vec!["a"], &schema)
        .with_expected_sort_and_const_exprs(
            vec![vec!["a", "b", "c"], vec!["b", "a", "c"]],
            vec![],
        )
        .run()
}

/// An unrelated constant fills no gap.
#[test]
fn test_union_equivalence_properties_constants_no_fill_gaps() -> Result<()> {
    let schema = create_test_schema()?;
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(vec![vec!["a", "c"]], vec!["d"], &schema)
        .with_child_sort_and_const_exprs(vec![vec!["b", "c"]], vec!["a"], &schema)
        .with_expected_sort_and_const_exprs(vec![vec!["a"]], vec![])
        .run()
}

/// Constants fill the leading gaps of the other side's ordering.
#[test]
fn test_union_equivalence_properties_constants_fill_some_gaps() -> Result<()> {
    let schema = create_test_schema()?;
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(vec![vec!["c"]], vec!["a", "b"], &schema)
        .with_child_sort_and_const_exprs(vec![vec!["a DESC", "b"]], vec![], &schema)
        .with_expected_sort_and_const_exprs(vec![vec!["a DESC", "b"]], vec![])
        .run()
}

/// Gap filling keeps each side's sort options.
#[test]
fn test_union_equivalence_properties_constants_fill_gaps_non_symmetric() -> Result<()> {
    let schema = create_test_schema()?;
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(vec![vec!["a", "c"]], vec!["b"], &schema)
        .with_child_sort_and_const_exprs(vec![vec!["b DESC", "c"]], vec!["a"], &schema)
        .with_expected_sort_and_const_exprs(
            vec![vec!["a", "b DESC", "c"], vec!["b DESC", "a", "c"]],
            vec![],
        )
        .run()
}

/// Gap filling in the middle of the orderings, from both sides.
#[test]
fn test_union_equivalence_properties_constants_gap_fill_symmetric() -> Result<()> {
    let schema = create_test_schema()?;
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(vec![vec!["a", "b", "d"]], vec!["c"], &schema)
        .with_child_sort_and_const_exprs(vec![vec!["a", "c", "d"]], vec!["b"], &schema)
        .with_expected_sort_and_const_exprs(
            vec![vec!["a", "c", "b", "d"], vec!["a", "b", "c", "d"]],
            vec![],
        )
        .run()
}

/// Gap filling together with a constant common to both sides.
#[test]
fn test_union_equivalence_properties_constants_gap_fill_and_common() -> Result<()> {
    let schema = create_test_schema()?;
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(
            vec![vec!["a DESC", "d"]],
            vec!["b", "c"],
            &schema,
        )
        .with_child_sort_and_const_exprs(
            vec![vec!["a DESC", "c", "d"]],
            vec!["b"],
            &schema,
        )
        .with_expected_sort_and_const_exprs(vec![vec!["a DESC", "c", "d"]], vec!["b"])
        .run()
}

/// Gap filling around a `DESC` key in the middle.
#[test]
fn test_union_equivalence_properties_constants_middle_desc() -> Result<()> {
    let schema = create_test_schema()?;
    UnionEquivalenceTest::new(&schema)
        .with_child_sort_and_const_exprs(
            vec![vec!["a", "b DESC", "d"]],
            vec!["c"],
            &schema,
        )
        .with_child_sort_and_const_exprs(vec![vec!["a", "c", "d"]], vec!["b"], &schema)
        .with_expected_sort_and_const_exprs(
            vec![vec!["a", "c", "b DESC", "d"], vec!["a", "b DESC", "c", "d"]],
            vec![],
        )
        .run()
}

/// Builds union children and checks the union's orderings and
/// (per-partition) constants for every order of the children.
struct UnionEquivalenceTest {
    /// The schema of the output of the Union
    output_schema: SchemaRef,
    /// The equivalence properties of each child to the union
    child_properties: Vec<EquivalenceProperties>,
    /// The expected output orderings
    expected_orderings: Vec<LexOrdering>,
    /// The expected output per-partition constants
    expected_constants: Vec<Arc<dyn PhysicalExpr>>,
}

impl UnionEquivalenceTest {
    fn new(output_schema: &SchemaRef) -> Self {
        Self {
            output_schema: Arc::clone(output_schema),
            child_properties: vec![],
            expected_orderings: vec![],
            expected_constants: vec![],
        }
    }

    /// Adds a union input with the given orderings (see [`parse_sort_expr`]).
    fn with_child_sort(self, orderings: Vec<Vec<&str>>, schema: &SchemaRef) -> Self {
        self.with_child_sort_and_const_exprs(orderings, vec![], schema)
    }

    /// Adds a union input with the given orderings and per-partition
    /// constants.
    fn with_child_sort_and_const_exprs(
        mut self,
        orderings: Vec<Vec<&str>>,
        constants: Vec<&str>,
        schema: &SchemaRef,
    ) -> Self {
        let orderings = make_orderings(&orderings, schema);
        let mut props =
            EquivalenceProperties::new_with_orderings(Arc::clone(schema), orderings);
        for name in constants {
            props.add_local_constant(col(name, schema).unwrap());
        }
        self.child_properties.push(props);
        self
    }

    fn with_expected_sort(self, orderings: Vec<Vec<&str>>) -> Self {
        self.with_expected_sort_and_const_exprs(orderings, vec![])
    }

    fn with_expected_sort_and_const_exprs(
        mut self,
        orderings: Vec<Vec<&str>>,
        constants: Vec<&str>,
    ) -> Self {
        let orderings = make_orderings(&orderings, &self.output_schema);
        let constants = constants
            .into_iter()
            .map(|name| col(name, &self.output_schema).unwrap())
            .collect();
        self.expected_orderings = orderings;
        self.expected_constants = constants;
        self
    }

    /// Computes the union for every permutation of the children (the code
    /// treats lhs and rhs differently) and compares it with the expectation.
    fn run(self) -> Result<()> {
        let Self {
            output_schema,
            child_properties,
            expected_orderings,
            expected_constants,
        } = self;
        let n = child_properties.len();
        for children in child_properties.into_iter().permutations(n) {
            let context = format!(
                "children: [{}]",
                children.iter().map(|c| c.to_string()).join("; ")
            );
            let actual =
                EquivalenceProperties::union(children, Arc::clone(&output_schema))?;
            assert_constants(&actual, &[], &expected_constants, &context);
            assert_orderings_equivalent(&actual, &expected_orderings, &context);
        }
        Ok(())
    }
}

fn make_orderings(orderings: &[Vec<&str>], schema: &SchemaRef) -> Vec<LexOrdering> {
    orderings
        .iter()
        .filter_map(|names| {
            LexOrdering::new(names.iter().map(|n| parse_sort_expr(n, schema)))
        })
        .collect()
}

/// Returns a new schema with the same types and `text` appended to each
/// field name ("a", "b" becomes "a1", "b1" for `text = "1"`).
fn append_fields(schema: &SchemaRef, text: &str) -> SchemaRef {
    Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .map(|field| {
                Field::new(
                    format!("{}{}", field.name(), text),
                    field.data_type().clone(),
                    field.is_nullable(),
                )
            })
            .collect::<Vec<_>>(),
    ))
}

/// A constant with the same value in both children keeps the value.
#[test]
fn test_union_constant_value_preservation() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]));
    let col_a = col("a", &schema)?;
    let literal_10 = ScalarValue::Int32(Some(10));

    let mut input1 = EquivalenceProperties::new(Arc::clone(&schema));
    input1.add_constant(Arc::clone(&col_a), Some(literal_10.clone()));
    let mut input2 = EquivalenceProperties::new(Arc::clone(&schema));
    input2.add_constant(Arc::clone(&col_a), Some(literal_10.clone()));

    let union_props = EquivalenceProperties::union(vec![input1, input2], schema)?;
    assert_constants(&union_props, &[(col_a, Some(literal_10))], &[], "a = 10");

    Ok(())
}

/// A constant value that does not cast to the union's field type is dropped.
#[test]
fn test_union_drops_unrepresentable_constant_value_after_schema_rewrite() -> Result<()> {
    let input_schema = Arc::new(Schema::new(vec![Field::new(
        "ticker",
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        true,
    )]));
    let output_schema = Arc::new(Schema::new(vec![Field::new(
        "timestamp",
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        true,
    )]));

    let ticker = col("ticker", &input_schema)?;
    let stale_value = ScalarValue::Utf8(Some("ESU6".to_owned()));
    let mut input = EquivalenceProperties::new(input_schema);
    input.add_constant(ticker, Some(stale_value));

    let union_props = EquivalenceProperties::union(vec![input], output_schema)?;
    assert_constants(&union_props, &[], &[], "stale value");

    Ok(())
}

/// Two constants sharing a value in one child do not make each other
/// uniform: `const_1` is `foo` in both children, `const_2` differs.
#[test]
fn test_constants_share_values() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("const_1", DataType::Utf8, false),
        Field::new("const_2", DataType::Utf8, false),
    ]));
    let col_const_1 = col("const_1", &schema)?;
    let col_const_2 = col("const_2", &schema)?;
    let literal_foo = ScalarValue::Utf8(Some("foo".to_owned()));
    let literal_bar = ScalarValue::Utf8(Some("bar".to_owned()));

    // | Input | Const_1 | Const_2 |
    // | ----- | ------- | ------- |
    // |     1 | foo     | foo     |
    // |     2 | foo     | bar     |
    let mut input1 = EquivalenceProperties::new(Arc::clone(&schema));
    input1.add_constant(Arc::clone(&col_const_1), Some(literal_foo.clone()));
    input1.add_constant(Arc::clone(&col_const_2), Some(literal_foo.clone()));
    let mut input2 = EquivalenceProperties::new(Arc::clone(&schema));
    input2.add_constant(Arc::clone(&col_const_1), Some(literal_foo.clone()));
    input2.add_constant(Arc::clone(&col_const_2), Some(literal_bar));

    let union_props = EquivalenceProperties::union(vec![input1, input2], schema)?;
    // const_1 = Uniform("foo"), const_2 = Heterogeneous
    assert_constants(
        &union_props,
        &[(col_const_1, Some(literal_foo))],
        &[col_const_2],
        "shared values",
    );

    Ok(())
}

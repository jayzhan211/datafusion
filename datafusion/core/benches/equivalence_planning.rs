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

//! Planning time of shapes that stress equivalence properties: many
//! declared orderings asked many ordering questions, one long ordering
//! asked about all of its columns, and deep chains of aliasing
//! projections. SQL only, so the same file runs against any
//! version of `EquivalenceProperties`.

use std::hint::black_box;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion_expr::col;
use tokio::runtime::Runtime;

/// A table with `n_cols` columns and `n_orderings` declared orderings of
/// `len` columns each (ordering `k` is `c{k*len}..c{k*len+len-1}`).
fn register_ordered_table(ctx: &SessionContext, n_orderings: usize, len: usize) {
    let n_cols = n_orderings * len + 4;
    let fields: Vec<_> = (0..n_cols)
        .map(|i| Field::new(format!("c{i}"), DataType::Int64, true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let orderings = (0..n_orderings)
        .map(|k| {
            (0..len)
                .map(|j| col(format!("c{}", k * len + j)).sort(true, false))
                .collect()
        })
        .collect();
    let table = MemTable::try_new(schema, vec![vec![]])
        .unwrap()
        .with_sort_order(orderings);
    ctx.register_table("t", Arc::new(table)).unwrap();
}

/// One query per ordering: ORDER BY its key, a window partitioned by its
/// first column, and a GROUP BY over its columns in reverse.
fn ordering_queries(n_orderings: usize, len: usize) -> Vec<String> {
    (0..n_orderings)
        .map(|k| {
            let cols: Vec<_> = (0..len).map(|j| format!("c{}", k * len + j)).collect();
            let rev: Vec<_> = cols.iter().rev().cloned().collect();
            format!(
                "SELECT {all}, count(*), max(row_number) FROM (\
                   SELECT {all}, ROW_NUMBER() OVER (PARTITION BY {first} ORDER BY {rest}) \
                   AS row_number FROM t WHERE {first} = 1) \
                 GROUP BY {rev} ORDER BY {all}",
                all = cols.join(", "),
                first = cols[0],
                rest = cols[1..].join(", "),
                rev = rev.join(", "),
            )
        })
        .collect()
}

/// One table ordered by all of its `len` columns, asked about all of them:
/// ORDER BY the whole ordering, and GROUP BY it in reverse (the sorted
/// permutation is found one round per column).
fn long_ordering_queries(len: usize) -> Vec<String> {
    let cols: Vec<_> = (0..len).map(|j| format!("c{j}")).collect();
    let rev: Vec<_> = cols.iter().rev().cloned().collect();
    vec![
        format!("SELECT {all} FROM t ORDER BY {all}", all = cols.join(", ")),
        format!(
            "SELECT {all}, count(*) FROM t GROUP BY {rev}",
            all = cols.join(", "),
            rev = rev.join(", "),
        ),
    ]
}

/// `depth` nested projections that alias every column (and add a cast of the
/// first), with an ORDER BY on the outermost names.
fn alias_chain_query(n_cols: usize, depth: usize) -> String {
    let mut sql = "SELECT * FROM t".to_string();
    for level in 0..depth {
        let prev = |i: usize| {
            if level == 0 {
                format!("c{i}")
            } else {
                format!("l{}_{i}", level - 1)
            }
        };
        let cols: Vec<_> = (0..n_cols)
            .map(|i| format!("{} AS l{level}_{i}", prev(i)))
            .collect();
        sql = format!(
            "SELECT {}, CAST({} AS DOUBLE) AS cast{level} FROM ({sql})",
            cols.join(", "),
            prev(0)
        );
    }
    let last = depth - 1;
    format!("{sql} ORDER BY l{last}_0, l{last}_1, l{last}_2")
}

fn physical_plan(ctx: &SessionContext, rt: &Runtime, sql: &str) {
    black_box(rt.block_on(async {
        ctx.sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    }));
}

fn criterion_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("equivalence_planning");

    for (n_orderings, len) in [(4, 4), (16, 4), (32, 8)] {
        let ctx = SessionContext::new();
        register_ordered_table(&ctx, n_orderings, len);
        let queries = ordering_queries(n_orderings, len);
        group.bench_function(
            BenchmarkId::new("many_orderings", format!("{n_orderings}x{len}")),
            |b| b.iter(|| queries.iter().for_each(|q| physical_plan(&ctx, &rt, q))),
        );
    }

    for len in [50, 200] {
        let ctx = SessionContext::new();
        register_ordered_table(&ctx, 1, len);
        let queries = long_ordering_queries(len);
        group.bench_function(BenchmarkId::new("long_ordering", len), |b| {
            b.iter(|| queries.iter().for_each(|q| physical_plan(&ctx, &rt, q)))
        });
    }

    for (n_cols, depth) in [(16, 8), (64, 16), (200, 8)] {
        let ctx = SessionContext::new();
        register_ordered_table(&ctx, n_cols / 4, 4);
        let query = alias_chain_query(n_cols, depth);
        group.bench_function(
            BenchmarkId::new("alias_chain", format!("{n_cols}cols_x{depth}")),
            |b| b.iter(|| physical_plan(&ctx, &rt, &query)),
        );
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

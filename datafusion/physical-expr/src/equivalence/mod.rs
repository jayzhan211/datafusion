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

//! What a plan node knows about its output: equalities, constants, keys and
//! orderings ([`EquivalenceProperties`]), and the questions optimizer rules
//! ask about them.
//!
//! Reading order: `properties.rs` (the facts and where they hold),
//! `matching.rs` (one walk answers the ordering questions), `derive.rs`
//! (projection, union, join), `equalities.rs`. The ordering type,
//! [`Ordering`] of [`Step`]s, lives in `datafusion-physical-expr-common`
//! next to [`PhysicalSortExpr`].

use std::borrow::Borrow;
use std::sync::Arc;

use crate::PhysicalExpr;

use arrow::compute::SortOptions;
use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};

mod derive;
mod equalities;
mod matching;
mod properties;

pub use datafusion_physical_expr_common::sort_expr::{Ordering, Step};
pub use equalities::Equalities;
pub use properties::{EquivalenceProperties, format_orderings};
// Re-export for backwards compatibility, we recommend importing from
// datafusion_physical_expr::projection instead
pub use crate::projection::{ProjectionMapping, project_ordering, project_orderings};

// Convert each tuple to a `PhysicalSortExpr` and construct a vector.
pub fn convert_to_sort_exprs<T: Borrow<Arc<dyn PhysicalExpr>>>(
    args: &[(T, SortOptions)],
) -> Vec<PhysicalSortExpr> {
    args.iter()
        .map(|(expr, options)| PhysicalSortExpr::new(Arc::clone(expr.borrow()), *options))
        .collect()
}

// Convert each vector of tuples to a `LexOrdering`.
pub fn convert_to_orderings<T: Borrow<Arc<dyn PhysicalExpr>>>(
    args: &[Vec<(T, SortOptions)>],
) -> Vec<LexOrdering> {
    args.iter()
        .filter_map(|sort_exprs| LexOrdering::new(convert_to_sort_exprs(sort_exprs)))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests;

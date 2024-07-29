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

use crate::aggregates::group_values::GroupValues;
use crate::aggregates::AggregateMode;
use ahash::RandomState;
use arrow::array::AsArray;
use arrow::compute::cast;
use arrow::datatypes::UInt64Type;
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, Rows, SortField};
use arrow_array::{Array, ArrayRef, UInt64Array};
use arrow_schema::{DataType, SchemaRef};
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::memory_pool::proxy::{RawTableAllocExt, VecAllocExt};
use datafusion_expr::EmitTo;
use hashbrown::raw::RawTable;

/// A [`GroupValues`] making use of [`Rows`]
pub struct GroupValuesRows {
    /// The output schema
    schema: SchemaRef,

    /// Converter for the group values
    row_converter: RowConverter,

    /// Logically maps group values to a group_index in
    /// [`Self::group_values`] and in each accumulator
    ///
    /// Uses the raw API of hashbrown to avoid actually storing the
    /// keys (group values) in the table
    ///
    /// keys: u64 hashes of the GroupValue
    /// values: (hash, group_index)
    map: RawTable<(u64, usize)>,

    /// The size of `map` in bytes
    map_size: usize,

    /// The actual group by values, stored in arrow [`Row`] format.
    /// `group_values[i]` holds the group value for group_index `i`.
    ///
    /// The row format is used to compare group keys quickly and store
    /// them efficiently in memory. Quick comparison is especially
    /// important for multi-column group keys.
    ///
    /// [`Row`]: arrow::row::Row
    group_values: Option<Rows>,
    group_hashes: Option<Vec<u64>>,

    /// reused buffer to store hashes
    hashes_buffer: Vec<u64>,

    /// reused buffer to store rows
    rows_buffer: Rows,

    /// Random state for creating hashes
    random_state: RandomState,
}

impl GroupValuesRows {
    pub fn try_new(schema: SchemaRef) -> Result<Self> {
        let row_converter = RowConverter::new(
            schema
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
        )?;

        let map = RawTable::with_capacity(0);

        let starting_rows_capacity = 1000;
        let starting_data_capacity = 64 * starting_rows_capacity;
        let rows_buffer =
            row_converter.empty_rows(starting_rows_capacity, starting_data_capacity);
        Ok(Self {
            schema,
            row_converter,
            map,
            map_size: 0,
            group_values: None,
            group_hashes: Default::default(),
            hashes_buffer: Default::default(),
            rows_buffer,
            random_state: Default::default(),
        })
    }
}

impl GroupValues for GroupValuesRows {
    fn intern(
        &mut self,
        cols: &[ArrayRef],
        groups: &mut Vec<usize>,
        hash_values: Option<&ArrayRef>,
    ) -> Result<()> {
        // Convert the group keys into the row format
        let group_rows = &mut self.rows_buffer;
        group_rows.clear();
        self.row_converter.append(group_rows, cols)?;
        let n_rows = group_rows.num_rows();

        let mut group_values = match self.group_values.take() {
            Some(group_values) => group_values,
            None => self.row_converter.empty_rows(0, 0),
        };

        let mut store_gp_hashes = match self.group_hashes.take() {
            Some(group_hashes) => group_hashes,
            None => vec![],
        };

        // tracks to which group each of the input rows belongs
        groups.clear();

        if let Some(hash_values) = hash_values {
            let hash_array = hash_values.as_primitive::<UInt64Type>();
            for (row, hash) in hash_array.iter().enumerate() {
                let hash = hash.unwrap();

                // TODO: I guess at final aggregate mode, the group values we have are all unique
                // If it is true, we could avoid map

                let entry = self.map.get_mut(hash, |(_hash, group_idx)| {
                    // verify that a group that we are inserting with hash is
                    // actually the same key value as the group in
                    // existing_idx  (aka group_values @ row)
                    group_rows.row(row) == group_values.row(*group_idx)
                });

                let group_idx = match entry {
                    // Existing group_index for this group value
                    Some((_hash, group_idx)) => *group_idx,
                    //  1.2 Need to create new entry for the group
                    None => {
                        // Add new entry to aggr_state and save newly created index
                        let group_idx = group_values.num_rows();
                        group_values.push(group_rows.row(row));
                        store_gp_hashes.push(hash);

                        // for hasher function, use precomputed hash value
                        self.map.insert_accounted(
                            (hash, group_idx),
                            |(hash, _group_index)| *hash,
                            &mut self.map_size,
                        );
                        group_idx
                    }
                };
                groups.push(group_idx);
            }
        } else {
            // 1.1 Calculate the group keys for the group values
            let batch_hashes = &mut self.hashes_buffer;
            batch_hashes.clear();
            batch_hashes.resize(n_rows, 0);
            create_hashes(cols, &self.random_state, batch_hashes)?;
            for (row, &hash) in batch_hashes.iter().enumerate() {
                let entry = self.map.get_mut(hash, |(_hash, group_idx)| {
                    // verify that a group that we are inserting with hash is
                    // actually the same key value as the group in
                    // existing_idx  (aka group_values @ row)
                    group_rows.row(row) == group_values.row(*group_idx)
                });

                let group_idx = match entry {
                    // Existing group_index for this group value
                    Some((_hash, group_idx)) => *group_idx,
                    //  1.2 Need to create new entry for the group
                    None => {
                        // Add new entry to aggr_state and save newly created index
                        let group_idx = group_values.num_rows();
                        group_values.push(group_rows.row(row));
                        store_gp_hashes.push(hash);

                        // for hasher function, use precomputed hash value
                        self.map.insert_accounted(
                            (hash, group_idx),
                            |(hash, _group_index)| *hash,
                            &mut self.map_size,
                        );
                        group_idx
                    }
                };
                groups.push(group_idx);
            }
        }

        self.group_values = Some(group_values);
        if !store_gp_hashes.is_empty() {
            self.group_hashes = Some(store_gp_hashes);
        }

        Ok(())
    }

    fn size(&self) -> usize {
        let group_values_size = self.group_values.as_ref().map(|v| v.size()).unwrap_or(0);
        self.row_converter.size()
            + group_values_size
            + self.map_size
            + self.rows_buffer.size()
            + self.hashes_buffer.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.group_values
            .as_ref()
            .map(|group_values| group_values.num_rows())
            .unwrap_or(0)
    }

    fn emit(&mut self, emit_to: EmitTo, mode: AggregateMode) -> Result<Vec<ArrayRef>> {
        let mut group_values = self
            .group_values
            .take()
            .expect("Can not emit from empty rows");

        let group_hashes = self
            .group_hashes
            .take()
            .expect("Can not emit from empty rows");

        // println!("emit_to: {:?}", emit_to);
        let mut output = match emit_to {
            EmitTo::All => {
                let mut output = self.row_converter.convert_rows(&group_values)?;
                if mode == AggregateMode::Partial {
                    let arr = Arc::new(UInt64Array::from(group_hashes)) as ArrayRef;
                    output.push(arr);
                }
                group_values.clear();
                output
            }
            EmitTo::First(n) => {
                let groups_rows = group_values.iter().take(n);
                let output = self.row_converter.convert_rows(groups_rows)?;
                // Clear out first n group keys by copying them to a new Rows.
                // TODO file some ticket in arrow-rs to make this more efficient?
                let mut new_group_values = self.row_converter.empty_rows(0, 0);
                for row in group_values.iter().skip(n) {
                    new_group_values.push(row);
                }
                std::mem::swap(&mut new_group_values, &mut group_values);

                // SAFETY: self.map outlives iterator and is not modified concurrently
                unsafe {
                    for bucket in self.map.iter() {
                        // Decrement group index by n
                        match bucket.as_ref().1.checked_sub(n) {
                            // Group index was >= n, shift value down
                            Some(sub) => bucket.as_mut().1 = sub,
                            // Group index was < n, so remove from table
                            None => self.map.erase(bucket),
                        }
                    }
                }
                output
            }
        };

        // TODO: Materialize dictionaries in group keys (#7647)
        for (field, array) in self.schema.fields.iter().zip(&mut output) {
            let expected = field.data_type();
            if let DataType::Dictionary(_, v) = expected {
                let actual = array.data_type();
                if v.as_ref() != actual {
                    return Err(DataFusionError::Internal(format!(
                        "Converted group rows expected dictionary of {v} got {actual}"
                    )));
                }
                *array = cast(array.as_ref(), expected)?;
            }
        }

        self.group_values = Some(group_values);
        self.group_hashes = None;
        Ok(output)
    }

    fn clear_shrink(&mut self, batch: &RecordBatch) {
        let count = batch.num_rows();
        self.group_values = self.group_values.take().map(|mut rows| {
            rows.clear();
            rows
        });
        self.map.clear();
        self.map.shrink_to(count, |_| 0); // hasher does not matter since the map is cleared
        self.map_size = self.map.capacity() * std::mem::size_of::<(u64, usize)>();
        self.hashes_buffer.clear();
        self.hashes_buffer.shrink_to(count);
    }
}

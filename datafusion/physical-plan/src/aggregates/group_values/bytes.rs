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

use crate::aggregates::{group_values::GroupValues, AggregateMode};
use ahash::RandomState;
use arrow::{array::AsArray, datatypes::UInt64Type};
use arrow_array::{Array, ArrayRef, OffsetSizeTrait, RecordBatch, UInt64Array};
use datafusion_common::hash_utils::create_hashes;
use datafusion_expr::EmitTo;
use datafusion_physical_expr_common::binary_map::{ArrowBytesMap, OutputType};

/// A [`GroupValues`] storing single column of Utf8/LargeUtf8/Binary/LargeBinary values
///
/// This specialization is significantly faster than using the more general
/// purpose `Row`s format
pub struct GroupValuesByes<O: OffsetSizeTrait> {
    /// Map string/binary values to group index
    map: ArrowBytesMap<O, usize>,
    /// The total number of groups so far (used to assign group_index)
    num_groups: usize,
    /// random state used to generate hashes
    random_state: RandomState,
    /// buffer that stores hash values (reused across batches to save allocations)
    hashes_buffer: Vec<u64>,
    group_hashes: Option<Vec<u64>>,
}

impl<O: OffsetSizeTrait> GroupValuesByes<O> {
    pub fn new(output_type: OutputType) -> Self {
        Self {
            map: ArrowBytesMap::new(output_type),
            num_groups: 0,
            random_state: RandomState::with_seeds(0, 0, 0, 0),
            hashes_buffer: Default::default(),
            group_hashes: Default::default(),
        }
    }
}

impl<O: OffsetSizeTrait> GroupValues for GroupValuesByes<O> {
    fn intern(
        &mut self,
        cols: &[ArrayRef],
        groups: &mut Vec<usize>,
        hash_values: Option<&ArrayRef>,
    ) -> datafusion_common::Result<()> {
        assert_eq!(cols.len(), 1);

        // look up / add entries in the table
        let arr = &cols[0];

        groups.clear();

        let mut store_gp_hashes = match self.group_hashes.take() {
            Some(group_hashes) => group_hashes,
            None => vec![],
        };

        let batch_hashes = if let Some(hash_values) = hash_values {
            let hash_array = hash_values.as_primitive::<UInt64Type>();
            hash_array.values().as_ref()
        } else {
            // step 1: compute hashes
            let batch_hashes = &mut self.hashes_buffer;
            batch_hashes.clear();
            batch_hashes.resize(arr.len(), 0);
            create_hashes(&[arr.clone()], &self.random_state, batch_hashes)
                // hash is supported for all types and create_hashes only
                // returns errors for unsupported types
                .unwrap();
            batch_hashes
        };

        self.map.insert_if_new(
            batch_hashes,
            arr,
            // called for each new group
            |_value, hash| {
                // assign new group index on each insert
                let group_idx = self.num_groups;
                self.num_groups += 1;
                store_gp_hashes.push(hash);
                group_idx
            },
            // called for each group
            |group_idx| {
                groups.push(group_idx);
            },
        );

        self.group_hashes = Some(store_gp_hashes);

        // ensure we assigned a group to for each row
        assert_eq!(groups.len(), arr.len());
        Ok(())
    }

    fn size(&self) -> usize {
        self.map.size() + std::mem::size_of::<Self>()
    }

    fn is_empty(&self) -> bool {
        self.num_groups == 0
    }

    fn len(&self) -> usize {
        self.num_groups
    }

    fn emit(
        &mut self,
        emit_to: EmitTo,
        mode: AggregateMode,
    ) -> datafusion_common::Result<Vec<ArrayRef>> {
        // Reset the map to default, and convert it into a single array
        let map_contents = self.map.take().into_state();

        let mut group_hashes = self
            .group_hashes
            .take()
            .expect("Can not emit from empty rows for hashes");

        let group_values = match emit_to {
            EmitTo::All => {
                self.num_groups -= map_contents.len();
                map_contents
            }
            EmitTo::First(n) if n == self.len() => {
                self.num_groups -= map_contents.len();
                map_contents
            }
            EmitTo::First(n) => {
                // if we only wanted to take the first n, insert the rest back
                // into the map we could potentially avoid this reallocation, at
                // the expense of much more complex code.
                // see https://github.com/apache/datafusion/issues/9195
                let emit_group_values = map_contents.slice(0, n);
                let remaining_group_values =
                    map_contents.slice(n, map_contents.len() - n);

                let remaining_group_hashes = group_hashes.split_off(n);
                let hash_array =
                    Arc::new(UInt64Array::from(remaining_group_hashes)) as ArrayRef;

                self.num_groups = 0;
                let mut group_indexes = vec![];
                // TODO: reuse hash value
                self.intern(
                    &[remaining_group_values],
                    &mut group_indexes,
                    Some(&hash_array),
                )?;

                // Verify that the group indexes were assigned in the correct order
                assert_eq!(0, group_indexes[0]);

                emit_group_values
            }
        };

        let mut output = vec![group_values];
        if mode == AggregateMode::Partial {
            let arr = Arc::new(UInt64Array::from(group_hashes)) as ArrayRef;
            output.push(arr);
        }

        Ok(output)
    }

    fn clear_shrink(&mut self, _batch: &RecordBatch) {
        // in theory we could potentially avoid this reallocation and clear the
        // contents of the maps, but for now we just reset the map from the beginning
        self.map.take();
    }
}

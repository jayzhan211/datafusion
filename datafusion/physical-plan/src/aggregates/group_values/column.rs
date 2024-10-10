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

use crate::aggregates::group_values::group_column::{
    ByteGroupValueBuilder, GroupColumn, PrimitiveGroupValueBuilder,
};
use ahash::RandomState;
use arrow::compute::cast;
use arrow::datatypes::{
    Date32Type, Date64Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
    Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow::record_batch::RecordBatch;
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Schema, SchemaRef};
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::instant::Instant;
use datafusion_common::utils::proxy::RawTableAllocExt;
use datafusion_common::{not_impl_err, DataFusionError, Result};
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::EmitTo;
use datafusion_physical_expr::binary_map::OutputType;

use hashbrown::raw::RawTable;

use super::GroupValues;

const INITIAL_CAPACITY: usize = 8192;

/// A [`GroupValues`] that stores multiple columns of group values.
///
///
pub struct GroupValuesColumn {
    /// The output schema
    schema: SchemaRef,

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

    /// The actual group by values, stored column-wise. Compare from
    /// the left to right, each column is stored as [`GroupColumn`].
    ///
    /// Performance tests showed that this design is faster than using the
    /// more general purpose [`GroupValuesRows`]. See the ticket for details:
    /// <https://github.com/apache/datafusion/pull/12269>
    ///
    /// [`GroupValuesRows`]: crate::aggregates::group_values::row::GroupValuesRows
    group_values: Vec<Box<dyn GroupColumn>>,
    group_values_v2: Vec<Box<dyn GroupColumn>>,

    /// reused buffer to store hashes
    hashes_buffer: Vec<u64>,

    /// Random state for creating hashes
    random_state: RandomState,

    current_hashes: Vec<u64>,
    current_offsets: Vec<usize>,
    new_entries: Vec<usize>,
    need_equality_check: Vec<usize>,
    no_match: Vec<usize>,
    capacity: usize,
    hash_table: Vec<usize>,
    hashes: Vec<u64>,
}

impl GroupValuesColumn {
    /// Create a new instance of GroupValuesColumn if supported for the specified schema
    pub fn try_new(schema: SchemaRef) -> Result<Self> {
        let map = RawTable::with_capacity(0);
        Ok(Self {
            schema,
            map,
            map_size: 0,
            group_values: vec![],
            group_values_v2: vec![],
            hashes_buffer: Default::default(),
            random_state: Default::default(),
            current_hashes: Default::default(),
            current_offsets: Default::default(),
            new_entries: Default::default(),
            need_equality_check: Default::default(),
            no_match: Default::default(),
            capacity: INITIAL_CAPACITY,
            hash_table: vec![0; INITIAL_CAPACITY],
            hashes: Vec::with_capacity(INITIAL_CAPACITY),
        })
    }

    /// Returns true if [`GroupValuesColumn`] supported for the specified schema
    pub fn supported_schema(schema: &Schema) -> bool {
        schema
            .fields()
            .iter()
            .map(|f| f.data_type())
            .all(Self::supported_type)
    }

    /// Returns true if the specified data type is supported by [`GroupValuesColumn`]
    ///
    /// In order to be supported, there must be a specialized implementation of
    /// [`GroupColumn`] for the data type, instantiated in [`Self::intern`]
    fn supported_type(data_type: &DataType) -> bool {
        matches!(
            *data_type,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::Date32
                | DataType::Date64
        )
    }
}

/// instantiates a [`PrimitiveGroupValueBuilder`] and pushes it into $v
///
/// Arguments:
/// `$v`: the vector to push the new builder into
/// `$nullable`: whether the input can contains nulls
/// `$t`: the primitive type of the builder
///
macro_rules! instantiate_primitive {
    ($v:expr, $v2:expr, $nullable:expr, $t:ty) => {
        if $nullable {
            let b = PrimitiveGroupValueBuilder::<$t, true>::new();
            $v.push(Box::new(b) as _);
            let b = PrimitiveGroupValueBuilder::<$t, true>::new();
            $v2.push(Box::new(b) as _)
        } else {
            let b = PrimitiveGroupValueBuilder::<$t, false>::new();
            $v.push(Box::new(b) as _);
            let b = PrimitiveGroupValueBuilder::<$t, false>::new();
            $v2.push(Box::new(b) as _)
        }
    };
}

impl GroupValues for GroupValuesColumn {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        let n_rows = cols[0].len();

        if self.group_values.is_empty() {
            let mut v = Vec::with_capacity(cols.len());
            let mut v2: Vec<Box<dyn GroupColumn>> = Vec::with_capacity(cols.len());

            for f in self.schema.fields().iter() {
                let nullable = f.is_nullable();
                match f.data_type() {
                    &DataType::Int8 => {
                        instantiate_primitive!(v, v2, nullable, Int8Type);
                    }
                    &DataType::Int16 => instantiate_primitive!(v, v2, nullable, Int16Type),
                    &DataType::Int32 => instantiate_primitive!(v, v2, nullable, Int32Type),
                    &DataType::Int64 => instantiate_primitive!(v, v2, nullable, Int64Type),
                    &DataType::UInt8 => instantiate_primitive!(v, v2, nullable, UInt8Type),
                    &DataType::UInt16 => instantiate_primitive!(v, v2, nullable, UInt16Type),
                    &DataType::UInt32 => instantiate_primitive!(v, v2, nullable, UInt32Type),
                    &DataType::UInt64 => instantiate_primitive!(v, v2, nullable, UInt64Type),
                    &DataType::Float32 => {
                        instantiate_primitive!(v, v2, nullable, Float32Type)
                    }
                    &DataType::Float64 => {
                        instantiate_primitive!(v, v2, nullable, Float64Type)
                    }
                    &DataType::Date32 => instantiate_primitive!(v, v2, nullable, Date32Type),
                    &DataType::Date64 => instantiate_primitive!(v, v2, nullable, Date64Type),
                    &DataType::Utf8 => {
                        let b = ByteGroupValueBuilder::<i32>::new(OutputType::Utf8);
                        v.push(Box::new(b) as _);
                        let b = ByteGroupValueBuilder::<i32>::new(OutputType::Utf8);
                        v2.push(Box::new(b) as _)
                    }
                    &DataType::LargeUtf8 => {
                        let b = ByteGroupValueBuilder::<i64>::new(OutputType::Utf8);
                        v.push(Box::new(b) as _);
                        let b = ByteGroupValueBuilder::<i64>::new(OutputType::Utf8);
                        v2.push(Box::new(b) as _)
                    }
                    &DataType::Binary => {
                        let b = ByteGroupValueBuilder::<i32>::new(OutputType::Binary);
                        v.push(Box::new(b) as _);
                        let b = ByteGroupValueBuilder::<i32>::new(OutputType::Binary);
                        v2.push(Box::new(b) as _)
                    }
                    &DataType::LargeBinary => {
                        let b = ByteGroupValueBuilder::<i64>::new(OutputType::Binary);
                        v.push(Box::new(b) as _);
                        let b = ByteGroupValueBuilder::<i64>::new(OutputType::Binary);
                        v2.push(Box::new(b) as _)
                    }
                    dt => {
                        return not_impl_err!("{dt} not supported in GroupValuesColumn")
                    }
                }
            }
            
            self.group_values = v;
            self.group_values_v2 = v2;
        }

        // tracks to which group each of the input rows belongs
        groups.clear();

        // 1.1 Calculate the group keys for the group values
        let batch_hashes = &mut self.current_hashes;
        batch_hashes.clear();
        batch_hashes.resize(n_rows, 0);
        create_hashes(cols, &self.random_state, batch_hashes)?;

        let mut start = Instant::now();
        for (row, &target_hash) in batch_hashes.iter().enumerate() {
            let entry = self.map.get_mut(target_hash, |(exist_hash, group_idx)| {
                // Somewhat surprisingly, this closure can be called even if the
                // hash doesn't match, so check the hash first with an integer
                // comparison first avoid the more expensive comparison with
                // group value. https://github.com/apache/datafusion/pull/11718
                if target_hash != *exist_hash {
                    return false;
                }

                fn check_row_equal(
                    array_row: &dyn GroupColumn,
                    lhs_row: usize,
                    array: &ArrayRef,
                    rhs_row: usize,
                ) -> bool {
                    array_row.equal_to(lhs_row, array, rhs_row)
                }

                for (i, group_val) in self.group_values_v2.iter().enumerate() {
                    if !check_row_equal(group_val.as_ref(), *group_idx, &cols[i], row) {
                        return false;
                    }
                }

                true
            });

            let group_idx = match entry {
                // Existing group_index for this group value
                Some((_hash, group_idx)) => *group_idx,
                //  1.2 Need to create new entry for the group
                None => {
                    // Add new entry to aggr_state and save newly created index
                    // let group_idx = group_values.num_rows();
                    // group_values.push(group_rows.row(row));

                    let mut checklen = 0;
                    let group_idx = self.group_values_v2[0].len();
                    for (i, group_value) in self.group_values_v2.iter_mut().enumerate() {
                        group_value.append_val(&cols[i], row);
                        let len = group_value.len();
                        if i == 0 {
                            checklen = len;
                        } else {
                            debug_assert_eq!(checklen, len);
                        }
                    }

                    // for hasher function, use precomputed hash value
                    self.map.insert_accounted(
                        (target_hash, group_idx),
                        |(hash, _group_index)| *hash,
                        &mut self.map_size,
                    );
                    group_idx
                }
            };
            groups.push(group_idx);
        }

        // let duration = start.elapsed();
        // println!("duration1: {:?}", duration);

        groups.clear();
        self.map.clear();

        // start = Instant::now();

        // rehash if necessary
        let current_n_rows = self.hashes.len();
        if current_n_rows + n_rows > (self.hash_table.capacity() as f64 / 1.5) as usize {
            let new_capacity = current_n_rows + n_rows;
            let new_capacity = std::cmp::max(new_capacity, 2 * self.capacity);
            let new_capacity = new_capacity.next_power_of_two();
            let mut new_table = vec![0; new_capacity];
            let new_bit_mask = new_capacity - 1;

            let table_ptr = self.hash_table.as_ptr();
            let hashes_ptr = self.hashes.as_ptr();
            let new_table_ptr = new_table.as_mut_ptr();

            unsafe {
                for i in 0..self.capacity {
                    let offset = *table_ptr.add(i);
                    if offset != 0 {
                        let hash = *hashes_ptr.add(offset as usize - 1);

                        let mut new_idx = hash as usize & new_bit_mask;
                        let mut num_iter = 0;
                        while *new_table_ptr.add(new_idx) != 0 {
                            num_iter += 1;
                            new_idx += num_iter * num_iter;
                            new_idx &= new_bit_mask;
                        }
                        *new_table_ptr.add(new_idx) = offset;
                    }
                }
            }

            self.hash_table = new_table;
            self.capacity = new_capacity;
        }


        let bit_mask = self.capacity - 1;
        self.current_offsets.resize(n_rows, 0);
        for row_idx in 0..n_rows {
            let hash = self.current_hashes[row_idx];
            let hash_table_idx = (hash as usize) & bit_mask;
            self.current_offsets[row_idx] = hash_table_idx;
        }

        // initially, `selection_vector[i]` = row at `i`
        let mut selection_vector: Vec<usize> = (0..n_rows).collect();
        let mut remaining_entries = n_rows;
        self.new_entries.resize(n_rows, 0);
        self.need_equality_check.resize(n_rows, 0);
        self.no_match.resize(n_rows, 0);
        let mut num_iter = 1;

        
        while remaining_entries > 0 {
            assert!(self.hashes.len() + remaining_entries <= self.capacity);

            let mut n_new_entries = 0;
            let mut n_need_equality_check = 0;
            let mut n_no_match = 0;
            // start = Instant::now();
            selection_vector
                .iter()
                .take(remaining_entries)
                .for_each(|&row_idx| {
                    let hash = self.current_hashes[row_idx];
                    let ht_offset = self.current_offsets[row_idx];
                    let offset = self.hash_table[ht_offset];

                    let is_empty_slot = offset == 0;
                    let is_hash_match = !is_empty_slot && self.hashes[offset - 1] == hash;

                    if is_empty_slot {
                        // the slot is empty, so we can create a new entry here
                        self.new_entries[n_new_entries] = row_idx;
                        n_new_entries += 1;

                        // we increment the slot entry offset by 1 to reserve the special value
                        // 0 for the scenario when the slot in the
                        // hash table is unoccupied.
                        self.hash_table[ht_offset] = self.hashes.len() + 1;
                        // also update hash for this slot so it can be used later
                        self.hashes.push(hash);
                    }

                    if is_hash_match {
                        // slot is not empty, and hash value match, now need to do equality
                        // check
                        self.need_equality_check[n_need_equality_check] = row_idx;
                        n_need_equality_check += 1;
                    }

                    if !is_empty_slot && self.hashes[offset - 1] != hash {
                        // slot is not empty, and hash value doesn't match, we have a hash
                        // collision and need to do probing
                        self.no_match[n_no_match] = row_idx;
                        n_no_match += 1;
                    }
                });

            // let duration = start.elapsed();
            // println!("iter: {:?} selection: {:?}", num_iter, duration);
            // start = Instant::now();

            self.new_entries
                .iter()
                .take(n_new_entries)
                .for_each(|row_idx| {
                    for (i, group_value) in self.group_values.iter_mut().enumerate() {
                        group_value.append_val(&cols[i], *row_idx)
                    }
                });
            assert_eq!(self.hashes.len(), self.group_values[0].len());
            // let duration = start.elapsed();
            // println!("iter: {:?}, append value: {:?}", num_iter, duration);
            // start = Instant::now();

            self.need_equality_check
                .iter()
                .take(n_need_equality_check)
                .for_each(|row_idx| {
                    let row_idx = *row_idx;
                    let ht_offset = self.current_offsets[row_idx];
                    let offset = self.hash_table[ht_offset];

                    // #[inline]
                    // fn check_row_equal(
                    //     array_row: &dyn GroupColumn,
                    //     lhs_row: usize,
                    //     array: &ArrayRef,
                    //     rhs_row: usize,
                    // ) -> bool {
                    //     array_row.equal_to(lhs_row, array, rhs_row)
                    // }

                    let is_equal =
                        self.group_values.iter().enumerate().all(|(i, group_val)| {
                            group_val.equal_to(offset-1, &cols[i], row_idx)
                            // check_row_equal(
                            //     group_val.as_ref(),
                            //     offset - 1,
                            //     &cols[i],
                            //     row_idx,
                            // )
                        });

                    if !is_equal {
                        self.no_match[n_no_match] = row_idx;
                        n_no_match += 1;
                    }
                });
            // let duration = start.elapsed();
            // println!("iter: {:?}, eq check: {:?}", num_iter, duration);
            // start = Instant::now();

            // now we need to probing for those rows in `no_match`
            let delta = num_iter * num_iter;
            let bit_mask = self.capacity - 1;

            self.no_match.iter().take(n_no_match).for_each(|&row_idx| {
                let slot_idx = self.current_offsets[row_idx] + delta;
                self.current_offsets[row_idx] = slot_idx & bit_mask;
            });
            // let duration = start.elapsed();
            // println!("iter: {:?}, prepare for next: {:?}", num_iter, duration);
            // start = Instant::now();

            std::mem::swap(&mut self.no_match, &mut selection_vector);
            remaining_entries = n_no_match;
            num_iter += 1;
        }
        // let duration = start.elapsed();
        // println!("looping: {:?}", duration);
        // start = Instant::now();

        groups.extend(
            self.current_offsets
                .iter()
                .take(n_rows)
                .map(|&hash_table_offset| self.hash_table[hash_table_offset] - 1),
        );

        // let duration = start.elapsed();
        // println!("duration2: {:?}", duration);

        // self.current_offsets
        //     .iter()
        //     .take(n_rows)
        //     .for_each(|&hash_table_offset| {
        //         groups.push(self.hash_table[hash_table_offset] - 1);
        //     });

        // self.group_values = Some(group_values);
        // self.group_values_v2 = Some(group_values_v2);

        Ok(())

        // for (row, &target_hash) in batch_hashes.iter().enumerate() {
        //     let entry = self.map.get_mut(target_hash, |(exist_hash, group_idx)| {
        //         // Somewhat surprisingly, this closure can be called even if the
        //         // hash doesn't match, so check the hash first with an integer
        //         // comparison first avoid the more expensive comparison with
        //         // group value. https://github.com/apache/datafusion/pull/11718
        //         if target_hash != *exist_hash {
        //             return false;
        //         }

        //         fn check_row_equal(
        //             array_row: &dyn GroupColumn,
        //             lhs_row: usize,
        //             array: &ArrayRef,
        //             rhs_row: usize,
        //         ) -> bool {
        //             array_row.equal_to(lhs_row, array, rhs_row)
        //         }

        //         for (i, group_val) in self.group_values.iter().enumerate() {
        //             if !check_row_equal(group_val.as_ref(), *group_idx, &cols[i], row) {
        //                 return false;
        //             }
        //         }

        //         true
        //     });

        //     let group_idx = match entry {
        //         // Existing group_index for this group value
        //         Some((_hash, group_idx)) => *group_idx,
        //         //  1.2 Need to create new entry for the group
        //         None => {
        //             // Add new entry to aggr_state and save newly created index
        //             // let group_idx = group_values.num_rows();
        //             // group_values.push(group_rows.row(row));

        //             let mut checklen = 0;
        //             let group_idx = self.group_values[0].len();
        //             for (i, group_value) in self.group_values.iter_mut().enumerate() {
        //                 group_value.append_val(&cols[i], row);
        //                 let len = group_value.len();
        //                 if i == 0 {
        //                     checklen = len;
        //                 } else {
        //                     debug_assert_eq!(checklen, len);
        //                 }
        //             }

        //             // for hasher function, use precomputed hash value
        //             self.map.insert_accounted(
        //                 (target_hash, group_idx),
        //                 |(hash, _group_index)| *hash,
        //                 &mut self.map_size,
        //             );
        //             group_idx
        //         }
        //     };
        //     groups.push(group_idx);
        // }

        // Ok(())
    }

    fn size(&self) -> usize {
        let group_values_size: usize = self.group_values.iter().map(|v| v.size()).sum();
        group_values_size + self.map_size + self.hashes_buffer.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        if self.group_values.is_empty() {
            return 0;
        }

        self.group_values[0].len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let mut output = match emit_to {
            EmitTo::All => {
                let group_values = std::mem::take(&mut self.group_values);
                debug_assert!(self.group_values.is_empty());

                group_values
                    .into_iter()
                    .map(|v| v.build())
                    .collect::<Vec<_>>()
            }
            EmitTo::First(n) => {
                let output = self
                    .group_values
                    .iter_mut()
                    .map(|v| v.take_n(n))
                    .collect::<Vec<_>>();

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

                self.hashes.drain(0..n);

                let hash_table_ptr = self.hash_table.as_mut_ptr();
                unsafe {
                    for i in 0..self.capacity {
                        let offset = *hash_table_ptr.add(i);
                        if offset != 0 {
                            match offset.checked_sub(n + 1) {
                                Some(sub) => *hash_table_ptr.add(i) = sub + 1,
                                None => *hash_table_ptr.add(i) = 0,
                            }
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

        Ok(output)
    }

    fn clear_shrink(&mut self, batch: &RecordBatch) {
        let count = batch.num_rows();
        self.group_values.clear();
        self.group_values_v2.clear();
        self.map.clear();
        self.map.shrink_to(count, |_| 0); // hasher does not matter since the map is cleared
        self.map_size = self.map.capacity() * std::mem::size_of::<(u64, usize)>();
        self.hashes_buffer.clear();
        self.hashes_buffer.shrink_to(count);
    }
}

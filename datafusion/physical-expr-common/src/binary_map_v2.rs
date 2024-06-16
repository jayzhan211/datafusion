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

//! [`ArrowBytesMap`] and [`ArrowBytesSet`] for storing maps/sets of values from
//! StringArray / LargeStringArray / BinaryArray / LargeBinaryArray.

use ahash::RandomState;
use arrow::array::cast::AsArray;
use arrow::array::types::{GenericBinaryType, GenericStringType};
use arrow::array::{
    Array, ArrayRef, BooleanBufferBuilder, BufferBuilder, GenericBinaryArray, GenericStringArray, OffsetSizeTrait, StructArray
};
use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::SchemaRef;
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::utils::proxy::{RawTableAllocExt, VecAllocExt};
use std::any::type_name;
use std::fmt::Debug;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use crate::binary_map::OutputType;

/// Optimized map for storing Arrow "bytes" types (`String`, `LargeString`,
/// `Binary`, and `LargeBinary`) values that can produce the set of keys on
/// output as `GenericBinaryArray` without copies.
///
/// Equivalent to `HashSet<String, V>` but with better performance for arrow
/// data.
///
/// # Generic Arguments
///
/// * `O`: OffsetSize (String/LargeString)
/// * `V`: payload type
///
/// # Description
///
/// This is a specialized HashMap with the following properties:
///
/// 1. Optimized for storing and emitting Arrow byte types  (e.g.
/// `StringArray` / `BinaryArray`) very efficiently by minimizing copying of
/// the string values themselves, both when inserting and when emitting the
/// final array.
///
///
/// 2. Retains the insertion order of entries in the final array. The values are
/// in the same order as they were inserted.
///
/// Note this structure can be used as a `HashSet` by specifying the value type
/// as `()`, as is done by [`ArrowBytesSet`].
///
/// This map is used by the special `COUNT DISTINCT` aggregate function to
/// store the distinct values, and by the `GROUP BY` operator to store
/// group values when they are a single string array.
///
/// # Example
///
/// The following diagram shows how the map would store the four strings
/// "Foo", NULL, "Bar", "TheQuickBrownFox":
///
/// * `hashtable` stores entries for each distinct string that has been
/// inserted. The entries contain the payload as well as information about the
/// value (either an offset or the actual bytes, see `Entry` docs for more
/// details)
///
/// * `offsets` stores offsets into `buffer` for each distinct string value,
/// following the same convention as the offsets in a `StringArray` or
/// `LargeStringArray`.
///
/// * `buffer` stores the actual byte data
///
/// * `null`: stores the index and payload of the null value, in this case the
/// second value (index 1)
///
/// ```text
/// ┌───────────────────────────────────┐    ┌─────┐    ┌────┐
/// │                ...                │    │  0  │    │FooB│
/// │ ┌──────────────────────────────┐  │    │  0  │    │arTh│
/// │ │      <Entry for "Bar">       │  │    │  3  │    │eQui│
/// │ │            len: 3            │  │    │  3  │    │ckBr│
/// │ │   offset_or_inline: "Bar"    │  │    │  6  │    │ownF│
/// │ │         payload:...          │  │    │     │    │ox  │
/// │ └──────────────────────────────┘  │    │     │    │    │
/// │                ...                │    └─────┘    └────┘
/// │ ┌──────────────────────────────┐  │
/// │ │<Entry for "TheQuickBrownFox">│  │    offsets    buffer
/// │ │           len: 16            │  │
/// │ │     offset_or_inline: 6      │  │    ┌───────────────┐
/// │ │         payload: ...         │  │    │    Some(1)    │
/// │ └──────────────────────────────┘  │    │ payload: ...  │
/// │                ...                │    └───────────────┘
/// └───────────────────────────────────┘
///                                              null
///               HashTable
/// ```
///
/// # Entry Format
///
/// Entries stored in a [`ArrowBytesMap`] represents a value that is either
/// stored inline or in the buffer
///
/// This helps the case where there are many short (less than 8 bytes) strings
/// that are the same (e.g. "MA", "CA", "NY", "TX", etc)
///
/// ```text
///                                                                ┌──────────────────┐
///                                                  ─ ─ ─ ─ ─ ─ ─▶│...               │
///                                                 │              │TheQuickBrownFox  │
///                                                                │...               │
///                                                 │              │                  │
///                                                                └──────────────────┘
///                                                 │               buffer of u8
///
///                                                 │
///                        ┌────────────────┬───────────────┬───────────────┐
///  Storing               │                │ starting byte │  length, in   │
///  "TheQuickBrownFox"    │   hash value   │   offset in   │  bytes (not   │
///  (long string)         │                │    buffer     │  characters)  │
///                        └────────────────┴───────────────┴───────────────┘
///                              8 bytes          8 bytes       4 or 8
///
///
///                         ┌───────────────┬─┬─┬─┬─┬─┬─┬─┬─┬───────────────┐
/// Storing "foobar"        │               │ │ │ │ │ │ │ │ │  length, in   │
/// (short string)          │  hash value   │?│?│f│o│o│b│a│r│  bytes (not   │
///                         │               │ │ │ │ │ │ │ │ │  characters)  │
///                         └───────────────┴─┴─┴─┴─┴─┴─┴─┴─┴───────────────┘
///                              8 bytes         8 bytes        4 or 8
/// ```
pub struct ArrowBytesMapV2<O, V>
where
    O: OffsetSizeTrait,
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    /// Should the output be String or Binary?
    output_type: Vec<OutputType>,
    /// Underlying hash set for each distinct value
    map: hashbrown::raw::RawTable<Entry<O, V>>,
    /// Total size of the map in bytes
    map_size: usize,
    /// In progress arrow `Buffer` containing all values
    buffer: BufferBuilder<u8>,
    /// Offsets into `buffer` for each distinct  value. These offsets as used
    /// directly to create the final `GenericBinaryArray`. The `i`th string is
    /// stored in the range `offsets[i]..offsets[i+1]` in `buffer`. Null values
    /// are stored as a zero length string.
    offsets: Vec<O>,
    /// random state used to generate hashes
    random_state: RandomState,
    /// buffer that stores hash values (reused across batches to save allocations)
    hashes_buffer: Vec<u64>,
    /// `(payload, null_index)` for the 'null' value, if any
    /// NOTE null_index is the logical index in the final array, not the index
    /// in the buffer
    null: Option<(V, usize)>,
    schema: SchemaRef,
    /// Storing group values for computing `state`
    group_values: Vec<Entry<O,V>>
}

/// The size, in number of entries, of the initial hash table
const INITIAL_MAP_CAPACITY: usize = 128;
/// The initial size, in bytes, of the string data
const INITIAL_BUFFER_CAPACITY: usize = 8 * 1024;
impl<O: OffsetSizeTrait, V> ArrowBytesMapV2<O, V>
where
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    pub fn new(output_type: Vec<OutputType>, schema: SchemaRef) -> Self {
        Self {
            output_type,
            map: hashbrown::raw::RawTable::with_capacity(INITIAL_MAP_CAPACITY),
            map_size: 0,
            buffer: BufferBuilder::new(INITIAL_BUFFER_CAPACITY),
            offsets: vec![O::default()], // first offset is always 0
            random_state: RandomState::new(),
            hashes_buffer: vec![],
            null: None,
            schema
        }
    }

    /// Return the contents of this map and replace it with a new empty map with
    /// the same output type
    pub fn take(&mut self) -> Self {
        let mut new_self = Self::new(self.output_type.clone(), self.schema.clone());
        std::mem::swap(self, &mut new_self);
        new_self
    }

    /// Inserts each value from `values` into the map, invoking `payload_fn` for
    /// each value if *not* already present, deferring the allocation of the
    /// payload until it is needed.
    ///
    /// Note that this is different than a normal map that would replace the
    /// existing entry
    ///
    /// # Arguments:
    ///
    /// `values`: array whose values are inserted
    ///
    /// `make_payload_fn`:  invoked for each value that is not already present
    /// to create the payload, in order of the values in `values`
    ///
    /// `observe_payload_fn`: invoked once, for each value in `values`, that was
    /// already present in the map, with corresponding payload value.
    ///
    /// # Returns
    ///
    /// The payload value for the entry, either the existing value or
    /// the newly inserted value
    ///
    /// # Safety:
    ///
    /// Note that `make_payload_fn` and `observe_payload_fn` are only invoked
    /// with valid values from `values`, not for the `NULL` value.
    pub fn insert_if_new<MP, OP>(
        &mut self,
        values: &[ArrayRef],
        make_payload_fn: MP,
        observe_payload_fn: OP,
    ) where
        MP: FnMut() -> V,
        OP: FnMut(V),
    {
        self.insert_if_new_inner_v2::<MP, OP>(
            values,
            make_payload_fn,
            observe_payload_fn,
        )
    }

    /// Converts this set into a `StringArray`, `LargeStringArray`,
    /// `BinaryArray`, or `LargeBinaryArray` containing each distinct value
    /// that was inserted. This is done without copying the values.
    ///
    /// The values are guaranteed to be returned in the same order in which
    /// they were first seen.
    pub fn into_state(self) -> ArrayRef {
        let Self {
            output_type,
            map: _,
            map_size: _,
            offsets,
            mut buffer,
            random_state: _,
            hashes_buffer: _,
            null,
            schema
        } = self;

        // Only make a `NullBuffer` if there was a null value
        let nulls = null.map(|(_payload, null_index)| {
            let num_values = offsets.len() - 1;
            single_null_buffer(num_values, null_index)
        });
        // SAFETY: the offsets were constructed correctly in `insert_if_new` --
        // monotonically increasing, overflows were checked.
        let offsets = unsafe { OffsetBuffer::new_unchecked(ScalarBuffer::from(offsets)) };
        let values = buffer.finish();

        let mut arrays: Vec<ArrayRef>;

        for out_type in output_type {
            let array = match out_type {
                OutputType::Binary => {
                    // SAFETY: the offsets were constructed correctly
                    Arc::new(unsafe {
                        GenericBinaryArray::new_unchecked(offsets, values, nulls)
                    }) as ArrayRef
                }
                OutputType::Utf8 => {
                    // SAFETY:
                    // 1. the offsets were constructed safely
                    //
                    // 2. we asserted the input arrays were all the correct type and
                    // thus since all the values that went in were valid (e.g. utf8)
                    // so are all the values that come out
                    Arc::new(unsafe {
                        GenericStringArray::new_unchecked(offsets, values, nulls)
                    }) as ArrayRef
                }
            };
            arrays.push(array)
        }

        let fields = schema.fields().clone();

        Arc::new(StructArray::new(fields, arrays, nulls))
    }

    /// Total number of entries (including null, if present)
    pub fn len(&self) -> usize {
        self.non_null_len() + self.null.map(|_| 1).unwrap_or(0)
    }

    /// Is the set empty?
    pub fn is_empty(&self) -> bool {
        self.map.is_empty() && self.null.is_none()
    }

    /// Number of non null entries
    pub fn non_null_len(&self) -> usize {
        self.map.len()
    }

    /// Return the total size, in bytes, of memory used to store the data in
    /// this set, not including `self`
    pub fn size(&self) -> usize {
        self.map_size
            + self.buffer.capacity() * std::mem::size_of::<u8>()
            + self.offsets.allocated_size()
            + self.hashes_buffer.allocated_size()
    }

    /// Generic version of [`Self::insert_if_new`] that handles `ByteArrayType`
    /// (both String and Binary)
    ///
    /// Note this is the only function that is generic on [`ByteArrayType`], which
    /// avoids having to template the entire structure,  making the code
    /// simpler and understand and reducing code bloat due to duplication.
    ///
    /// See comments on `insert_if_new` for more details
    ///
    /// Multi column version
    fn insert_if_new_inner_v2<MP, OP>(
        &mut self,
        values: &[ArrayRef],
        mut make_payload_fn: MP,
        mut observe_payload_fn: OP,
    ) where
        MP: FnMut() -> V,
        OP: FnMut(V),
    {

        // step 1: compute hashes
        let batch_hashes = &mut self.hashes_buffer;
        batch_hashes.clear();
        batch_hashes.resize(values.len(), 0);
        create_hashes(values, &self.random_state, batch_hashes)
            // hash is supported for all types and create_hashes only
            // returns errors for unsupported types
            .unwrap();

        // step 2: insert each value into the set, if not already present
        // let data_types: Vec<&DataType> = values.iter().map(|v|v.data_type()).collect();
        // let values_vec: Vec<&GenericByteArray<B>> = values.iter().map(|f| f.as_bytes::<B>()).collect();
        let column_values = values;

        let single_value_len = column_values[0].len();
        // Ensure lengths are equivalent
        assert_eq!(single_value_len, batch_hashes.len());

        for (row, &hash) in batch_hashes.iter().enumerate() {
            // iterate by row, get the concated values
                let entry = self.map.get_mut(hash, |header| {
                    // compare value if hashes match
                    for (col_id, values) in column_values.iter().enumerate() {

                        let value: Option<&[u8]> = match self.output_type[col_id] {
                            OutputType::Binary => {
                                let values = values.as_bytes::<GenericBinaryType<O>>();
                                if values.is_null(row) {
                                    None
                                } else {
                                    let value = values.value(row);
                                    Some(value)
                                }
                            }
                            OutputType::Utf8 => {
                                let values = values.as_bytes::<GenericStringType<O>>();
                                if values.is_null(row) {
                                    None
                                } else {
                                    let value = values.value(row).as_ref();
                                    Some(value)
                                }
                            }
                        };

                        if let Some(value) = value {
                            let value_len = O::usize_as(value.len());
                            if header.len[col_id].is_none() || header.len[col_id].unwrap() != value_len {
                                return false;
                            }
                            
                            if value.len() <= SHORT_VALUE_LEN {
                                let inline = value.iter().fold(0usize, |acc, &x| acc << 8 | x as usize);
                                if inline != header.offset_or_inline[col_id] {
                                    return false;
                                }
                            } else {
                                // Need to compare the bytes in the buffer
                                // SAFETY: buffer is only appended to, and we correctly inserted values and offsets
                                let existing_value =
                                    unsafe { self.buffer.as_slice().get_unchecked(header.range(col_id)) };
                                if value != existing_value {
                                    return false;
                                }
                            }
                        } else {
                            if header.len[col_id].is_some() {
                                return false;
                            }
                        }
                    }
                    true
                });

                // handle all null case?

                let payload = if let Some(entry) = entry {
                    entry.payload
                }
                // if no existing entry, make a new one
                else {

                    let mut len = vec![None;column_values.len()];
                    let mut offset_or_inline = vec![0usize;column_values.len()];
                    
                    for (col_id, values) in column_values.iter().enumerate() {
                        let value: Option<&[u8]> = match self.output_type[col_id] {
                            OutputType::Binary => {
                                let values = values.as_bytes::<GenericBinaryType<O>>();
                                if values.is_null(row) {
                                    None
                                } else {
                                    let value = values.value(row);
                                    Some(value)
                                }
                            }
                            OutputType::Utf8 => {
                                let values = values.as_bytes::<GenericStringType<O>>();
                                if values.is_null(row) {
                                    None
                                } else {
                                    let value = values.value(row).as_ref();
                                    Some(value)
                                }
                            }
                        };

                        if let Some(value) = value {
                            let value_len = O::usize_as(value.len());

                            if value.len() <= SHORT_VALUE_LEN {
                                let inline = value.iter().fold(0usize, |acc, &x| acc << 8 | x as usize);
                                offset_or_inline[col_id] = inline;
                            } else {
                                let offset = self.buffer.len(); // offset of start of data
                                offset_or_inline[col_id] = offset;
                            }
                            // Put the small values into buffer and offsets so it appears
                            // the output array, but store the actual bytes inline for
                            // comparison
                            self.buffer.append_slice(value);
                            self.offsets.push(O::usize_as(self.buffer.len()));

                            len[col_id] = Some(value_len);
                        }
                    }

                    let payload = make_payload_fn();

                    let new_header = Entry {
                        hash,
                        len,
                        offset_or_inline,
                        payload
                    };

                    self.map.insert_accounted(
                        new_header,
                        |header| header.hash,
                        &mut self.map_size,
                    );
                    payload
                };
                observe_payload_fn(payload);
        }

        // Check for overflow in offsets (if more data was sent than can be represented)
        if O::from_usize(self.buffer.len()).is_none() {
            panic!(
                "Put {} bytes in buffer, more than can be represented by a {}",
                self.buffer.len(),
                type_name::<O>()
            );
        }
    }
}

/// Returns a `NullBuffer` with a single null value at the given index
fn single_null_buffer(num_values: usize, null_index: usize) -> NullBuffer {
    let mut bool_builder = BooleanBufferBuilder::new(num_values);
    bool_builder.append_n(num_values, true);
    bool_builder.set_bit(null_index, false);
    NullBuffer::from(bool_builder.finish())
}

impl<O: OffsetSizeTrait, V> Debug for ArrowBytesMapV2<O, V>
where
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowBytesMap")
            .field("map", &"<map>")
            .field("map_size", &self.map_size)
            .field("buffer", &self.buffer)
            .field("random_state", &self.random_state)
            .field("hashes_buffer", &self.hashes_buffer)
            .finish()
    }
}

/// Maximum size of a value that can be inlined in the hash table
const SHORT_VALUE_LEN: usize = mem::size_of::<usize>();

/// Entry in the hash table -- see [`ArrowBytesMap`] for more details
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct Entry<O, V>
where
    O: OffsetSizeTrait,
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    /// hash of the value (stored to avoid recomputing it in hash table check)
    hash: u64,
    /// if len =< [`SHORT_VALUE_LEN`]: the data inlined
    /// if len > [`SHORT_VALUE_LEN`], the offset of where the data starts
    offset_or_inline: Vec<usize>,
    /// length of the value, in bytes (use O here so we use only i32 for
    /// strings, rather 64 bit usize)
    len: Vec<Option<O>>,
    /// value stored by the entry
    payload: V,
}

impl<O, V> Entry<O, V>
where
    O: OffsetSizeTrait,
    V: Debug + PartialEq + Eq + Clone + Copy + Default,
{
    /// returns self.offset..self.offset + self.len
    #[inline(always)]
    fn range(&self, col_id: usize) -> Range<usize> {
        self.offset_or_inline[col_id]..self.offset_or_inline[col_id] + self.len[col_id].unwrap().as_usize()
    }
}
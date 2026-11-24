/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/script.h>

namespace kv_mem {

/// How a DRAM KV table stores the rows written into it, whoever writes them:
/// an enrichment fetch, the trainer's row writes, or another writer.
struct RowStorage {
  /// Leading columns of a written row; the storage formats encode these, and
  /// the columns past them stay fp16.
  int64_t dim{0};
  /// Rowwise bit rate the table is published at (0 = off). When set, rows are
  /// stored as the values serving decodes at that rate.
  int64_t serving_qdq_bit_rate{0};
  /// 0 for the rowwise publish (fp16 scale and bias); else the group size of a
  /// group-wise INT4 publish with float8_e4m3fn scales (requires
  /// serving_qdq_bit_rate 4).
  int64_t serving_qdq_group_size{0};
  /// Store rows in DRAM in that group-wise INT4 encoding instead of fp16
  /// (requires serving_qdq_group_size); reads still return fp16.
  bool packed_storage{false};
  /// Store rows in DRAM as row-wise INT8 (codes, then fp32 scale and bias)
  /// instead of fp16; reads and writes stay fp16. Excludes packed_storage and
  /// serving_qdq_bit_rate.
  bool int8_storage{false};
};

/// RowStorage passed from Python as a TorchScript custom class, so that a table
/// stores its rows in a publish format without an enrichment source.
struct RowStorageConfig : public torch::jit::CustomClassHolder {
  explicit RowStorageConfig(
      int64_t dim,
      int64_t serving_qdq_bit_rate = 0,
      int64_t serving_qdq_group_size = 0,
      bool packed_storage = false,
      bool int8_storage = false)
      : storage_{
            dim,
            serving_qdq_bit_rate,
            serving_qdq_group_size,
            packed_storage,
            int8_storage} {}

  RowStorage storage_;
};

} // namespace kv_mem

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/Tensor.h> // @manual=//caffe2:ATen-core
#include <torch/custom_class.h>

namespace ssd {

class EmbeddingRocksDB;
class EmbeddingRocksDBWrapper;
class SnapshotHandle;

// @lint-ignore CLANGTIDY cppcoreguidelines-special-member-functions
struct EmbeddingSnapshotHandleWrapper : public torch::jit::CustomClassHolder {
  explicit EmbeddingSnapshotHandleWrapper(
      const SnapshotHandle* handle,
      std::shared_ptr<EmbeddingRocksDB> db);

  ~EmbeddingSnapshotHandleWrapper();

  const SnapshotHandle* handle;
  std::shared_ptr<EmbeddingRocksDB> db;
};

class KVTensorWrapper : public torch::jit::CustomClassHolder {
 public:
  explicit KVTensorWrapper(
      c10::intrusive_ptr<EmbeddingRocksDBWrapper> db,
      std::vector<int64_t> shape,
      int64_t dtype,
      int64_t row_offset,
      std::optional<c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper>>
          snapshot_handle = std::nullopt,
      std::optional<std::vector<int64_t>> materialized_shape = std::nullopt);

  at::Tensor narrow(int64_t dim, int64_t start, int64_t length);

  void set_range(
      int64_t dim,
      const int64_t start,
      const int64_t length,
      const at::Tensor& weights);

  void set_weights_and_ids(const at::Tensor& weights, const at::Tensor& ids);

  at::Tensor get_weights_by_ids(const at::Tensor& ids);

  c10::IntArrayRef sizes();

  c10::IntArrayRef strides();

  c10::ScalarType dtype();

  std::string_view dtype_str();

  c10::Device device();

  std::string device_str();

  std::string layout_str();

 private:
  std::shared_ptr<EmbeddingRocksDB> db_;
  c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper> snapshot_handle_;
  at::TensorOptions options_;
  std::vector<int64_t> shape_;
  std::optional<std::vector<int64_t>> materialized_shape_;
  std::vector<int64_t> strides_;
  int64_t row_offset_;
};

} // namespace ssd

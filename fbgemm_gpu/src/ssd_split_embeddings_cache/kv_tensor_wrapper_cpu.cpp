/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/library.h>

#include "./kv_tensor_wrapper.h"
#include "common/base/Exception.h"

using namespace at;
using namespace ssd;

namespace ssd {
class EmbeddingRocksDB {};

// @lint-ignore CLANGTIDY facebook-hte-ShadowingClass
class EmbeddingRocksDBWrapper : public torch::jit::CustomClassHolder {
 private:
  friend class KVTensorWrapper;
  std::shared_ptr<EmbeddingRocksDB> impl_;
};

class SnapshotHandle {};

KVTensorWrapper::KVTensorWrapper(
    std::vector<int64_t> shape,
    int64_t dtype [[maybe_unused]],
    int64_t row_offset,
    const std::optional<c10::intrusive_ptr<EmbeddingSnapshotHandleWrapper>>
        snapshot_handle [[maybe_unused]],
    const std::optional<at::Tensor> sorted_indices [[maybe_unused]],
    int64_t width_offset [[maybe_unused]],
    [[maybe_unused]] const std::optional<
        c10::intrusive_ptr<RocksdbCheckpointHandleWrapper>>,
    bool read_only [[maybe_unused]],
    bool only_load_weight [[maybe_unused]])
    // @lint-ignore CLANGTIDY clang-diagnostic-missing-noreturn
    : shape_(std::move(shape)), row_offset_(row_offset) {
  FBEXCEPTION("Not implemented");
}

void KVTensorWrapper::set_embedding_rocks_dp_wrapper(
    c10::intrusive_ptr<EmbeddingRocksDBWrapper> db) {
  FBEXCEPTION("Not implemented");
}

void KVTensorWrapper::set_dram_db_wrapper(
    c10::intrusive_ptr<kv_mem::DramKVEmbeddingCacheWrapper> db) {
  FBEXCEPTION("Not implemented");
}

at::Tensor KVTensorWrapper::narrow(
    int64_t dim [[maybe_unused]],
    int64_t start [[maybe_unused]],
    int64_t length [[maybe_unused]]) {
  FBEXCEPTION("Not implemented");
  return {};
}

void KVTensorWrapper::set_range(
    int64_t dim [[maybe_unused]],
    const int64_t start [[maybe_unused]],
    const int64_t length [[maybe_unused]],
    // @lint-ignore CLANGTIDY clang-diagnostic-missing-noreturn
    at::Tensor& weights [[maybe_unused]]) {
  FBEXCEPTION("Not implemented");
}

c10::IntArrayRef KVTensorWrapper::sizes() {
  FBEXCEPTION("Not implemented");
  return shape_;
}

c10::IntArrayRef KVTensorWrapper::strides() {
  FBEXCEPTION("Not implemented");
  return shape_; // make linter happy.
}

c10::ScalarType KVTensorWrapper::dtype() {
  FBEXCEPTION("Not implemented");
  return options_.dtype().toScalarType();
}

std::string_view KVTensorWrapper::dtype_str() {
  FBEXCEPTION("Not implemented");
  return scalarTypeToTypeMeta(dtype()).name();
}

c10::Device KVTensorWrapper::device() {
  FBEXCEPTION("Not implemented");
  return options_.device();
}

std::string KVTensorWrapper::device_str() {
  FBEXCEPTION("Not implemented");
  return device().str();
}

std::string KVTensorWrapper::layout_str() {
  FBEXCEPTION("Not implemented");
  std::ostringstream oss;
  oss << options_.layout();
  return oss.str();
}

std::vector<std::string> KVTensorWrapper::get_kvtensor_serializable_metadata()
    const {
  FBEXCEPTION("Not implemented");
  return {};
  return std::vector<std::string>{};
}

void KVTensorWrapper::delete_rocksdb_checkpoint_dir() const {
  FBEXCEPTION("Not implemented");
}
} // namespace ssd

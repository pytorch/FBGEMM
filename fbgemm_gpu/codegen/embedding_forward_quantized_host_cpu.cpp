/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <ostream>
#ifdef FBCODE_CAFFE2
#include <folly/container/Enumerate.h>
#include <folly/container/F14Map.h>
#endif
#include <torch/serialize/input-archive.h>
#include <torch/serialize/output-archive.h>

using namespace at;

Tensor int4_split_embedding_codegen_forward_unweighted_cpu(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t unused);

Tensor int4_split_embedding_codegen_forward_weighted_cpu(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int64_t unused);

Tensor int4_split_embedding_codegen_lookup_function_cpu(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights) {
  if (!indice_weights) {
    return int4_split_embedding_codegen_forward_unweighted_cpu(
        dev_weights,
        weights_offsets,
        D_offsets,
        total_D,
        max_D,
        indices,
        offsets,
        pooling_mode,
        0);
  }
  return int4_split_embedding_codegen_forward_weighted_cpu(
      dev_weights,
      weights_offsets,
      D_offsets,
      total_D,
      max_D,
      indices,
      offsets,
      pooling_mode,
      *indice_weights,
      0);
}

void pruned_hashmap_insert_unweighted_cpu(
    Tensor indices,
    Tensor dense_indices,
    Tensor offsets,
    Tensor hash_table,
    int64_t T);

Tensor pruned_hashmap_lookup_unweighted_cpu(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    int64_t T);

TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.impl(
      "int4_split_embedding_codegen_lookup_function",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(int4_split_embedding_codegen_lookup_function_cpu)));

  // GPU version of pruned_hashmap needs to use CPU version of
  // pruned_hashmap_insert
  m.def(
      "pruned_hashmap_insert(Tensor indices, Tensor dense_indices, Tensor offsets, Tensor hash_table, int T) -> ()");
  m.impl(
      "pruned_hashmap_insert",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(pruned_hashmap_insert_unweighted_cpu)));

  // CPU version of Lookup isn't used. For CPUs, we should use PrunedMapCPU
  // below.
  m.impl(
      "pruned_hashmap_lookup",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(pruned_hashmap_lookup_unweighted_cpu)));
}

// TODO: 1) switch from doing a single flat table keyed on (table_idx, idx) ->
// value (i.e. 3x32bits per entry) and instead have T separate tables with
// idx->value (i.e. 2x32bits per entry).
// 2) Possibly reuse the concurrent hash table.
class PrunedMapCPU : public torch::jit::CustomClassHolder {
 public:
  PrunedMapCPU() {}
  explicit PrunedMapCPU(std::string serialized) {
    torch::serialize::InputArchive archive;
    archive.load_from(serialized.data(), serialized.size());
    Tensor values;
    archive.read(std::string("table"), values);
    auto values_acc = values.accessor<int32_t, 2>();
    map_.reserve(values.size(0));
    for (auto i = 0; i < values.size(0); ++i) {
      auto index = values_acc[i][0];
      auto table = values_acc[i][1];
      auto value = values_acc[i][2];
      std::pair<int32_t, int32_t> key = {index, table};
      map_.emplace(key, value);
    }
  }
  std::string serialize() const {
    torch::serialize::OutputArchive archive(
        std::make_shared<torch::jit::CompilationUnit>());
    int64_t N = map_.size();
    auto values =
        at::empty({N, 3}, at::TensorOptions(at::kCPU).dtype(at::kInt));
    auto values_acc = values.accessor<int32_t, 2>();
#ifdef FBCODE_CAFFE2
    for (const auto& [index, kv] : folly::enumerate(map_)) {
#else
    int index = 0;
    for (auto& kv : map_) {
#endif
      values_acc[index][0] = kv.first.first;
      values_acc[index][1] = kv.first.second;
      values_acc[index][2] = kv.second;
#ifndef FBCODE_CAFFE2
      index++;
#endif
    }
    std::ostringstream oss;
    archive.write(std::string("table"), values);
    archive.save_to(oss);
    return oss.str();
  }

  void insert(Tensor indices, Tensor dense_indices, Tensor offsets, int64_t T) {
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();
    const auto* offsets_acc = offsets.data_ptr<int32_t>();
    for (int32_t t = 0; t < T; ++t) {
      for (int32_t b = 0; b < B; ++b) {
        int32_t indices_start = offsets_acc[t * B + b];
        int32_t indices_end = offsets_acc[t * B + b + 1];
        int32_t L = indices_end - indices_start;
        for (int32_t l = 0; l < L; ++l) {
          int32_t idx = indices_acc[indices_start + l];
          int32_t value = dense_indices_acc[indices_start + l];
          std::pair<int32_t, int32_t> key = {idx, t};
          map_.emplace(key, value);
        }
      }
    }
  }

  Tensor lookup(Tensor indices, Tensor offsets, int64_t T) const {
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    auto dense_indices = empty_like(indices);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();
    const auto* offsets_acc = offsets.data_ptr<int32_t>();
    for (int32_t t = 0; t < T; ++t) {
      for (int32_t b = 0; b < B; ++b) {
        int32_t indices_start = offsets_acc[t * B + b];
        int32_t indices_end = offsets_acc[t * B + b + 1];
        int32_t L = indices_end - indices_start;
        for (int32_t l = 0; l < L; ++l) {
          int32_t idx = indices_acc[indices_start + l];
          auto it = map_.find({idx, t});
          dense_indices_acc[indices_start + l] =
              it != map_.end() ? it->second : -1;
        }
      }
    }
    return dense_indices;
  }

 private:
#ifdef FBCODE_CAFFE2
  folly::F14FastMap<std::pair<int32_t, int32_t>, int32_t> map_;
#else
  struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
      return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
  };
  std::unordered_map<std::pair<int32_t, int32_t>, int32_t, pair_hash> map_;
#endif
};

static auto PrunedMapCPURegistry =
    torch::class_<PrunedMapCPU>("fb", "PrunedMapCPU")
        .def(torch::init<>())
        .def("insert", &PrunedMapCPU::insert)
        .def("lookup", &PrunedMapCPU::lookup)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<PrunedMapCPU>& self) -> std::string {
              return self->serialize();
            },
            // __setstate__
            [](std::string data) -> c10::intrusive_ptr<PrunedMapCPU> {
              return c10::make_intrusive<PrunedMapCPU>(data);
            });

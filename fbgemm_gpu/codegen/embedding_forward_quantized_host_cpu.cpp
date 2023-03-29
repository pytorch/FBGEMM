/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

///@defgroup embedding-cpu Embedding CPU Operators
///

Tensor int_nbit_split_embedding_codegen_forward_unweighted_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t row_alignment,
    int64_t output_dtype,
    int64_t fp8_exponent_bits,
    int64_t fp8_exponent_bias);

Tensor int_nbit_split_embedding_codegen_forward_weighted_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t row_alignment,
    Tensor indice_weights,
    int64_t output_dtype,
    int64_t fp8_exponent_bits,
    int64_t fp8_exponent_bias);

Tensor int_nbit_split_embedding_nobag_codegen_forward_unweighted_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    int64_t D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t row_alignment,
    int64_t output_dtype,
    int64_t fp8_exponent_bits,
    int64_t fp8_exponent_bias);

///@ingroup embedding-cpu
Tensor int_nbit_split_embedding_codegen_lookup_function_cpu(
    Tensor dev_weights,
    Tensor uvm_weights, // to match the interface of CUDA op using UVM
    Tensor weights_placements, // to match the interface of CUDA op using UVM
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    int64_t output_dtype,
    c10::optional<Tensor>
        lxu_cache_weights, // Not used, to match cache interface for CUDA op
    c10::optional<Tensor>
        lxu_cache_locations, // Not used, to match cache interface for CUDA op
    c10::optional<int64_t> row_alignment,
    c10::optional<int64_t> max_float8_D,
    c10::optional<int64_t> fp8_exponent_bits,
    c10::optional<int64_t> fp8_exponent_bias) {
  if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
    std::vector<int64_t> max_D_list{
        max_int2_D,
        max_int4_D,
        max_int8_D,
        max_float8_D ? *max_float8_D : 0,
        max_float16_D,
        max_float32_D};
    int64_t max_D = *std::max_element(max_D_list.begin(), max_D_list.end());
    return int_nbit_split_embedding_nobag_codegen_forward_unweighted_cpu(
        dev_weights,
        uvm_weights,
        weights_placements,
        weights_offsets,
        weights_tys,
        max_D,
        indices,
        offsets,
        pooling_mode,
        row_alignment ? *row_alignment : 1,
        output_dtype,
        fp8_exponent_bits ? *fp8_exponent_bits : -1,
        fp8_exponent_bias ? *fp8_exponent_bias : -1);
  }
  if (!indice_weights || indice_weights->numel() == 0) {
    return int_nbit_split_embedding_codegen_forward_unweighted_cpu(
        dev_weights,
        uvm_weights,
        weights_placements,
        weights_offsets,
        weights_tys,
        D_offsets,
        total_D,
        indices,
        offsets,
        pooling_mode,
        row_alignment ? *row_alignment : 1,
        output_dtype,
        fp8_exponent_bits ? *fp8_exponent_bits : -1,
        fp8_exponent_bias ? *fp8_exponent_bias : -1);
  }
  return int_nbit_split_embedding_codegen_forward_weighted_cpu(
      dev_weights,
      uvm_weights,
      weights_placements,
      weights_offsets,
      weights_tys,
      D_offsets,
      total_D,
      indices,
      offsets,
      pooling_mode,
      row_alignment ? *row_alignment : 1,
      *indice_weights,
      output_dtype,
      fp8_exponent_bits ? *fp8_exponent_bits : -1,
      fp8_exponent_bias ? *fp8_exponent_bias : -1);
}

///@ingroup embedding-cpu
Tensor int_nbit_split_embedding_uvm_caching_codegen_lookup_function_cpu(
    Tensor dev_weights,
    Tensor uvm_weights, // to match the interface of CUDA op using UVM
    Tensor weights_placements, // to match the interface of CUDA op using UVM
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    int64_t output_dtype,
    c10::optional<Tensor> lxu_cache_weights,
    c10::optional<Tensor> lxu_cache_locations,
    c10::optional<int64_t> row_alignment,
    c10::optional<int64_t> max_float8_D,
    c10::optional<int64_t> fp8_exponent_bits,
    c10::optional<int64_t> fp8_exponent_bias,
    // Additinal args for uvm_caching version.
    c10::optional<Tensor> cache_hash_size_cumsum [[maybe_unused]],
    c10::optional<int64_t> total_cache_hash_size [[maybe_unused]],
    c10::optional<Tensor> cache_index_table_map [[maybe_unused]],
    c10::optional<Tensor> lxu_cache_state [[maybe_unused]],
    c10::optional<Tensor> lxu_state [[maybe_unused]]) {
  LOG(WARNING)
      << "int_nbit_split_embedding_uvm_caching_codegen_lookup_function shouldn't be called for CPU; it is only for GPU.";
  return int_nbit_split_embedding_codegen_lookup_function_cpu(
      dev_weights,
      uvm_weights,
      weights_placements,
      weights_offsets,
      weights_tys,
      D_offsets,
      total_D,
      max_int2_D,
      max_int4_D,
      max_int8_D,
      max_float16_D,
      max_float32_D,
      indices,
      offsets,
      pooling_mode,
      indice_weights,
      output_dtype,
      lxu_cache_weights,
      lxu_cache_locations,
      row_alignment,
      max_float8_D,
      fp8_exponent_bits,
      fp8_exponent_bias);
}

///@ingroup embedding-cpu
void pruned_hashmap_insert_unweighted_cpu(
    Tensor indices,
    Tensor dense_indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets);

///@ingroup embedding-cpu
Tensor pruned_hashmap_lookup_unweighted_cpu(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets);

///@ingroup embedding-cpu
Tensor pruned_array_lookup_cpu(
    Tensor indices,
    Tensor offsets,
    Tensor index_remappings,
    Tensor index_remappings_offsets);

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "int_nbit_split_embedding_codegen_lookup_function(Tensor dev_weights, Tensor uvm_weights, Tensor weights_placements, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, int total_D, int max_int2_D, int max_int4_D, int max_int8_D, int max_float16_D, int max_float32_D, Tensor indices, Tensor offsets, int pooling_mode, Tensor? indice_weights, int output_dtype=1, Tensor? lxu_cache_weights=None, Tensor? lxu_cache_locations=None, int? row_alignment = None, int? max_float8_D=0, int? fp8_exponent_bits=-1, int? fp8_exponent_bias=-1) -> Tensor");
  DISPATCH_TO_CPU(
      "int_nbit_split_embedding_codegen_lookup_function",
      int_nbit_split_embedding_codegen_lookup_function_cpu);

  m.def(
      "int_nbit_split_embedding_uvm_caching_codegen_lookup_function(Tensor dev_weights, Tensor uvm_weights, Tensor weights_placements, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, int total_D, int max_int2_D, int max_int4_D, int max_int8_D, int max_float16_D, int max_float32_D, Tensor indices, Tensor offsets, int pooling_mode, Tensor? indice_weights=None, int output_dtype=1, Tensor? lxu_cache_weights=None, Tensor? lxu_cache_locations=None, int? row_alignment=-1, int? max_float8_D=0, int? fp8_exponent_bits=-1, int? fp8_exponent_bias=-1, Tensor? cache_hash_size_cumsum=None, int? total_cache_hash_size=-1, Tensor? cache_index_table_map=None, Tensor? lxu_cache_state=None, Tensor? lxu_state=None) -> Tensor");
  DISPATCH_TO_CPU(
      "int_nbit_split_embedding_uvm_caching_codegen_lookup_function",
      int_nbit_split_embedding_uvm_caching_codegen_lookup_function_cpu);

  // GPU version of pruned_hashmap needs to use CPU version of
  // pruned_hashmap_insert
  m.def(
      "pruned_hashmap_insert(Tensor indices, Tensor dense_indices, Tensor offsets, Tensor(a!) hash_table, Tensor hash_table_offsets) -> ()");
  DISPATCH_TO_CPU(
      "pruned_hashmap_insert", pruned_hashmap_insert_unweighted_cpu);

  // CPU version of hashmap Lookup isn't used. For CPUs, we should use
  // PrunedMapCPU below.
  m.def(
      "pruned_hashmap_lookup(Tensor indices, Tensor offsets, Tensor hash_table, Tensor hash_table_offsets) -> Tensor");
  DISPATCH_TO_CPU(
      "pruned_hashmap_lookup", pruned_hashmap_lookup_unweighted_cpu);

  // CPU version of array lookup.
  m.def(
      "pruned_array_lookup(Tensor indices, Tensor offsets, Tensor index_remappings, Tensor index_remappings_offsets) -> Tensor");
  DISPATCH_TO_CPU("pruned_array_lookup", pruned_array_lookup_cpu);
}

class PrunedMapCPU : public torch::jit::CustomClassHolder {
 public:
  PrunedMapCPU() {}
  explicit PrunedMapCPU(std::string serialized) {
    torch::serialize::InputArchive archive;
    archive.load_from(serialized.data(), serialized.size());
    Tensor values;
    archive.read(std::string("values"), values);
    Tensor table_offsets;
    archive.read(std::string("table_offsets"), table_offsets);

    auto T = table_offsets.numel() - 1;

    auto values_acc = values.accessor<int32_t, 2>();
    auto table_offsets_acc = table_offsets.accessor<int64_t, 1>();

    maps_.resize(T);
    for (auto t = 0; t < T; ++t) {
      auto& map = maps_[t];
      const auto table_start = table_offsets_acc[t];
      for (auto i = 0; i < values.size(0); ++i) {
        auto slot_sparse_index = values_acc[table_start + i][0];
        auto slot_dense_index = values_acc[table_start + i][1];
        map.emplace(slot_sparse_index, slot_dense_index);
      }
    }
  }
  std::string serialize() const {
    torch::serialize::OutputArchive archive(
        std::make_shared<torch::jit::CompilationUnit>());
    int64_t T = maps_.size();
    auto table_offsets =
        at::empty({T + 1}, at::TensorOptions(at::kCPU).dtype(at::kLong));
    auto table_offsets_acc = table_offsets.accessor<int64_t, 1>();
    table_offsets_acc[0] = 0;
    int64_t N = 0;
    for (auto t = 0; t < T; ++t) {
      N += maps_[t].size();
      table_offsets_acc[t + 1] = N;
    }
    auto values =
        at::empty({N, 2}, at::TensorOptions(at::kCPU).dtype(at::kInt));
    auto values_acc = values.accessor<int32_t, 2>();
    for (auto t = 0; t < maps_.size(); ++t) {
      const auto& map = maps_[t];
      const auto table_start = table_offsets_acc[t];
      TORCH_CHECK(
          map.size() == (table_offsets_acc[t + 1] - table_offsets_acc[t]));
      int index = 0;
      for (const auto& kv : map) {
        values_acc[table_start + index][0] = kv.first;
        values_acc[table_start + index][1] = kv.second;
        index++;
      }
    }
    std::ostringstream oss;
    archive.write(std::string("values"), values);
    archive.write(std::string("table_offsets"), table_offsets);
    archive.save_to(oss);
    return oss.str();
  }

  void insert(Tensor indices, Tensor dense_indices, Tensor offsets, int64_t T) {
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();
    const auto* offsets_acc = offsets.data_ptr<int32_t>();
    maps_.resize(T);
    for (int32_t t = 0; t < T; ++t) {
      auto& map = maps_[t];
      for (int32_t b = 0; b < B; ++b) {
        int32_t indices_start = offsets_acc[t * B + b];
        int32_t indices_end = offsets_acc[t * B + b + 1];
        int32_t L = indices_end - indices_start;
        for (int32_t l = 0; l < L; ++l) {
          int32_t slot_sparse_index = indices_acc[indices_start + l];
          int32_t slot_dense_index = dense_indices_acc[indices_start + l];
          if (slot_dense_index == -1) {
            // -1 means this row has been pruned, do not insert it.
            continue;
          }
          map.emplace(slot_sparse_index, slot_dense_index);
        }
      }
    }
  }

  Tensor lookup(Tensor indices, Tensor offsets) const {
    int32_t T = maps_.size();
    TORCH_CHECK(T > 0);
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    TORCH_CHECK(maps_.size() == T);
    auto dense_indices = empty_like(indices);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();
    const auto* offsets_acc = offsets.data_ptr<int32_t>();
    for (int32_t t = 0; t < T; ++t) {
      auto& map = maps_[t];
      for (int32_t b = 0; b < B; ++b) {
        int32_t indices_start = offsets_acc[t * B + b];
        int32_t indices_end = offsets_acc[t * B + b + 1];
        int32_t L = indices_end - indices_start;
        for (int32_t l = 0; l < L; ++l) {
          int32_t slot_sparse_index = indices_acc[indices_start + l];
          auto it = map.find(slot_sparse_index);
          dense_indices_acc[indices_start + l] =
              it != map.end() ? it->second : -1;
        }
      }
    }
    return dense_indices;
  }

 private:
#ifdef FBCODE_CAFFE2
  std::vector<folly::F14FastMap<int32_t, int32_t>> maps_;
#else
  std::vector<std::unordered_map<int32_t, int32_t>> maps_;
#endif
};

static auto PrunedMapCPURegistry =
    torch::class_<PrunedMapCPU>("fbgemm", "PrunedMapCPU")
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

class AtomicCounter : public torch::jit::CustomClassHolder {
 public:
  AtomicCounter() {
    counter_ = 0;
  }
  explicit AtomicCounter(std::string serialized) {
    std::stringstream ss(serialized);
    int64_t val;
    ss >> val;
    counter_ = val;
  }
  int64_t increment() {
    return counter_++;
  }
  int64_t decrement() {
    return counter_--;
  }
  void reset() {
    counter_ = 0;
  }
  int64_t get() {
    return counter_;
  }
  void set(int64_t val) {
    counter_ = val;
  }

  std::string serialize() const {
    std::ostringstream oss;
    oss << counter_;
    return oss.str();
  }

 private:
  std::atomic<int64_t> counter_{0};
};

static auto AtomicCounterRegistry =
    torch::class_<AtomicCounter>("fbgemm", "AtomicCounter")
        .def(torch::init<>())
        .def("increment", &AtomicCounter::increment)
        .def("decrement", &AtomicCounter::decrement)
        .def("reset", &AtomicCounter::reset)
        .def("get", &AtomicCounter::get)
        .def("set", &AtomicCounter::set)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<AtomicCounter>& self) -> std::string {
              return self->serialize();
            },
            // __setstate__
            [](std::string data) -> c10::intrusive_ptr<AtomicCounter> {
              return c10::make_intrusive<AtomicCounter>(data);
            });

// Thread-safe Tensor Queue
struct TensorQueue : torch::CustomClassHolder {
  explicit TensorQueue(Tensor t) : init_tensor_(t) {}

  explicit TensorQueue(c10::Dict<std::string, at::Tensor> dict) {
    init_tensor_ = dict.at(std::string("init_tensor"));
    const std::string key = "queue";
    Tensor size_tensor;
    size_tensor = dict.at(std::string(key + "/size")).cpu();
    const auto* size_tensor_acc = size_tensor.data_ptr<int64_t>();
    int64_t queue_size = size_tensor_acc[0];

    for (const auto index : c10::irange(queue_size)) {
      Tensor val;
      queue_[index] = dict.at(key + "/" + c10::to_string(index));
      queue_.push_back(val);
    }
  }

  c10::Dict<std::string, at::Tensor> serialize() const {
    c10::Dict<std::string, at::Tensor> dict;
    dict.insert(std::string("init_tensor"), init_tensor_);
    const std::string key = "queue";
    dict.insert(
        key + "/size", torch::tensor(static_cast<int64_t>(queue_.size())));
    for (const auto index : c10::irange(queue_.size())) {
      dict.insert(key + "/" + c10::to_string(index), queue_[index]);
    }
    return dict;
  }
  // Push the element to the rear of queue.
  // Lock is added for thread safe.
  void push(Tensor x) {
    std::lock_guard<std::mutex> guard(mutex_);
    queue_.push_back(x);
  }
  // Pop the front element of queue and return it.
  // If empty, return init_tensor_.
  // Lock is added for thread safe.
  Tensor pop() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!queue_.empty()) {
      auto val = queue_.front();
      queue_.pop_front();
      return val;
    } else {
      return init_tensor_;
    }
  }
  // Return front element of queue, read-only.
  // We might further optimize with read-write lock.
  Tensor top() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!queue_.empty()) {
      auto val = queue_.front();
      return val;
    } else {
      return init_tensor_;
    }
  }
  int64_t size() {
    return queue_.size();
  }

 private:
  std::deque<Tensor> queue_;
  std::mutex mutex_;
  Tensor init_tensor_;
};

static auto TensorQueueRegistry =
    torch::class_<TensorQueue>("fbgemm", "TensorQueue")
        .def(torch::init<Tensor>())
        .def("push", &TensorQueue::push)
        .def("pop", &TensorQueue::pop)
        .def("top", &TensorQueue::top)
        .def("size", &TensorQueue::size)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<TensorQueue>& self)
                -> c10::Dict<std::string, at::Tensor> {
              return self->serialize();
            },
            // __setstate__
            [](c10::Dict<std::string, at::Tensor> data)
                -> c10::intrusive_ptr<TensorQueue> {
              return c10::make_intrusive<TensorQueue>(std::move(data));
            });

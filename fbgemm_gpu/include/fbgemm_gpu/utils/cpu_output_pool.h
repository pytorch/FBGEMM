/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/util/Logging.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "fbgemm_gpu/utils/embedding_cpu_threading.h"

namespace fbgemm_gpu {

// Reuses large CPU output buffers across TBE inference calls. A fresh
// at::empty of a large output is backed by memory the allocator may already
// have returned to the OS, so writing it takes a minor page fault per 4 KiB
// page; under load bursts that dominates the op's latency. Pooled blocks stay
// allocated (and therefore resident) while cached, so reuse is fault-free.
//
// Sizes are rounded up to 4 buckets per power of two (<= 25% overhead) so that
// a varying total_L still hits the cache. Released blocks are kept LIFO per
// bucket until the cached total would exceed max_cached_bytes.
//
// The pool is itself the c10::Allocator: storages keep a pointer to their
// allocator, so it must live as long as any tensor it produced.
class CpuOutputPool final : public c10::Allocator {
 public:
  CpuOutputPool(size_t min_bytes, size_t max_cached_bytes)
      : min_bytes_(min_bytes), max_cached_bytes_(max_cached_bytes) {}

  CpuOutputPool(const CpuOutputPool&) = delete;
  CpuOutputPool& operator=(const CpuOutputPool&) = delete;

  ~CpuOutputPool() override {
    for (auto& [bucket, blocks] : free_) {
      for (void* p : blocks) {
        c10::free_cpu(p);
      }
    }
  }

  static size_t bucket_bytes(size_t n) {
    if (n <= 4) {
      return n;
    }
    // A quarter of n's leading power of two; at least 1 because n > 4.
    const size_t step = size_t{1} << (std::bit_width(n) - 3);
    return (n + step - 1) & ~(step - 1);
  }

  size_t min_bytes() const {
    return min_bytes_;
  }

  size_t cached_bytes() const {
    std::scoped_lock lock(mu_);
    return cached_bytes_;
  }

  c10::DataPtr allocate(size_t n) override {
    const size_t bucket = bucket_bytes(n);
    void* data = acquire(bucket);
    auto* block = new Block{this, data, bucket};
    return {data, block, &release_block, c10::Device(c10::DeviceType::CPU)};
  }

  void copy_data(void* dest, const void* src, std::size_t count)
      const override {
    default_copy_data(dest, src, count);
  }

 private:
  struct Block {
    CpuOutputPool* pool;
    void* data;
    size_t bucket;
  };

  static void release_block(void* ctx) {
    auto* block = static_cast<Block*>(ctx);
    block->pool->release(block->data, block->bucket);
    delete block;
  }

  void* acquire(size_t bucket) {
    {
      std::scoped_lock lock(mu_);
      auto it = free_.find(bucket);
      if (it != free_.end() && !it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        cached_bytes_ -= bucket;
        return p;
      }
    }
    return c10::alloc_cpu(bucket);
  }

  void release(void* p, size_t bucket) {
    {
      std::scoped_lock lock(mu_);
      if (cached_bytes_ + bucket <= max_cached_bytes_) {
        free_[bucket].push_back(p);
        cached_bytes_ += bucket;
        return;
      }
    }
    c10::free_cpu(p);
  }

  const size_t min_bytes_;
  const size_t max_cached_bytes_;
  mutable std::mutex mu_;
  std::unordered_map<size_t, std::vector<void*>> free_;
  size_t cached_bytes_ = 0;
};

inline size_t tbe_cpu_output_pool_min_bytes() {
  static const size_t bytes = static_cast<size_t>(get_env_int(
                                  "FBGEMM_TBE_CPU_OUTPUT_POOL_MIN_KB", 1024))
      << 10;
  return bytes;
}

inline size_t tbe_cpu_output_pool_max_cached_bytes() {
  static const size_t bytes = static_cast<size_t>(get_env_int(
                                  "FBGEMM_TBE_CPU_OUTPUT_POOL_MAX_MB", 16384))
      << 20;
  return bytes;
}

// Plain env var rather than a feature gate: gates resolve through JustKnobs in
// fbcode and fail hard when the knob does not exist. Read (and logged) once
// per process.
inline bool tbe_cpu_output_pool_enabled() {
  static const bool enabled = [] {
    const bool on =
        get_env_bool("FBGEMM_TBE_CPU_OUTPUT_POOL", /*default_val=*/false);
    if (on) {
      LOG(INFO)
          << "FBGEMM_TBE_CPU_OUTPUT_POOL enabled: CPU NOBAG TBE outputs >= "
          << (tbe_cpu_output_pool_min_bytes() >> 10)
          << " KiB are served from the output pool (max cached "
          << (tbe_cpu_output_pool_max_cached_bytes() >> 20) << " MiB)";
    } else {
      LOG(INFO) << "FBGEMM_TBE_CPU_OUTPUT_POOL disabled: CPU NOBAG TBE outputs "
                   "use at::empty";
    }
    return on;
  }();
  return enabled;
}

// Process-wide pool for CPU TBE outputs. Intentionally leaked: tensors that
// outlive static destruction may still return blocks to it.
inline CpuOutputPool& tbe_cpu_output_pool() {
  static auto* pool = new CpuOutputPool(
      tbe_cpu_output_pool_min_bytes(), tbe_cpu_output_pool_max_cached_bytes());
  return *pool;
}

// Contiguous CPU tensor whose storage comes from `pool`, constructed exactly as
// at::empty would on CPU: a plain CPU TensorImpl for regular dtypes, and for
// quantized dtypes (e.g. kQUInt4x2 used by INT4 NOBAG outputs) a QTensorImpl
// with an unknown quantizer and sub-byte packed storage, mirroring
// at::new_qtensor. Outputs smaller than the pool's min_bytes use at::empty.
inline at::Tensor pooled_empty_cpu(
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    CpuOutputPool& pool) {
  const bool quantized = c10::isQIntType(dtype);
  const int64_t itemsize = static_cast<int64_t>(c10::elementSize(dtype));
  int64_t nbytes = c10::multiply_integers(sizes) * itemsize;
  if (quantized && !sizes.empty()) {
    const int64_t per_byte = dtype == c10::ScalarType::QUInt4x2 ? 2
        : dtype == c10::ScalarType::QUInt2x4                    ? 4
                                                                : 1;
    nbytes = c10::multiply_integers(sizes.begin(), sizes.end() - 1) *
        ((sizes.back() * itemsize + per_byte - 1) / per_byte);
  }
  if (static_cast<size_t>(nbytes) < pool.min_bytes()) {
    return at::empty(sizes, at::TensorOptions().dtype(dtype));
  }
  if (!quantized) {
    return at::Tensor(
        at::detail::empty_generic(
            sizes,
            &pool,
            c10::DispatchKeySet(c10::DispatchKey::CPU),
            dtype,
            std::nullopt));
  }
  auto storage = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
      pool.allocate(nbytes),
      &pool,
      /*resizable=*/true);
  auto tensor = at::detail::make_tensor<at::QTensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::QuantizedCPU),
      caffe2::TypeMeta::fromScalarType(dtype),
      at::make_unknown_quantizer(dtype));
  at::get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  at::get_qtensorimpl(tensor)->empty_tensor_restride(
      c10::MemoryFormat::Contiguous);
  return tensor;
}

} // namespace fbgemm_gpu

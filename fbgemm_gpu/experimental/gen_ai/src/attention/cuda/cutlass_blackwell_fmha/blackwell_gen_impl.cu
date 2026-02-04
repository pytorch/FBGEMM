// @nolint
/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#ifndef USE_ROCM
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#endif

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include "blackwell_gen_interface.hpp"

#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <random>
#include <regex>
#include <vector>

#include "cute/tensor.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "reference/fmha_fwd_gen_reference.hpp"
#include "reference/reference_abs_error.hpp"

#include "collective/fmha_fusion.hpp"

#include "collective/sm100_fmha_gen_mainloop_warpspecialized.hpp"
#include "kernel/sm100_fmha_gen_kernel_warpspecialized.hpp"
#include "collective/sm100_fmha_gen_epilogue_warpspecialized.hpp"
#include "collective/sm100_fmha_gen_mainloop_warpspecialized.hpp"
#include "device/fmha.hpp"
#include "kernel/fmha_tile_scheduler.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

struct InputShape {
  int b;
  int h;
  int d;
  int sk;
  int h_k;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename Element,
    typename ElementOut,
    KernelType kKernelType,
    class TileShape>
struct GenRunner {
  using ElementAcc = float;

  using ProblemShape =
      Shape<_1, int, int, Shape<Shape<int, int>, int>>; // (Sq, Sk, D, ((H, Hr), B))

  using StrideQ =
      Stride<_0, _1, Stride<Stride<int, int>, int>>; // Q D ((H, Hr), B)
  using StrideNewK = Stride<_0, _1, Stride<Stride<_0, int64_t>, int64_t>>;
  using StrideCacheK =
      Stride<int, _1, Stride<Stride<_0, int64_t>, int64_t>>; // K D ((H, Hr), B)
  using StrideNewV = StrideNewK;
  using StrideCacheV = StrideCacheK;
  using StrideO = StrideQ;
  using StrideLSE = Stride<int, int, _1>;

  using Mainloop =
      cutlass::fmha::collective::Sm100FmhaGenMainloopWarpspecialized<
          Element,
          ElementAcc,
          ElementAcc,
          ElementOut,
          TileShape,
          StrideQ,
          StrideNewK,
          StrideNewV,
          StrideCacheK,
          StrideCacheV,
          StrideO>;

  using Epilogue =
      cutlass::fmha::collective::Sm100FmhaGenEpilogueWarpspecialized<
          ElementOut,
          StrideO,
          ElementAcc,
          StrideLSE>;

  using TileScheduler = std::conditional_t<
      kKernelType == KernelType::UMMA_P,
      cutlass::fmha::kernel::PersistentTileScheduler,
      cutlass::fmha::kernel::IndividualTileSchedulerSplitK>;

  using Kernel = cutlass::fmha::kernel::Sm100FmhaGenKernelWarpspecialized<
      ProblemShape,
      Mainloop,
      Epilogue,
      TileScheduler>;

  using Operation = cutlass::fmha::device::FMHA<Kernel>;

  StrideQ stride_q;
  StrideNewK stride_new_k;
  StrideNewV stride_new_v;
  StrideCacheK stride_cache_k;
  StrideCacheV stride_cache_v;
  StrideO stride_o;
  StrideLSE stride_lse;

  at::Tensor block_o;
  at::Tensor block_lse;
  at::Tensor q, k, v, seqlen_kv;
  std::optional<at::Tensor> batch_idx;
  int64_t split_k_size;
  int64_t window_size;

  std::tuple<at::Tensor, at::Tensor> fmha_fwd(
      const at::Tensor& q_input,
      const at::Tensor& k_input,
      const at::Tensor& v_input,
      const at::Tensor& seqlen_kv_input,
      const std::optional<at::Tensor>& batch_idx_input,
      int64_t split_k_size,
      int64_t window_size) {

    this->q = q_input;
    this->k = k_input;
    this->v = v_input;
    this->seqlen_kv = seqlen_kv_input;
    this->batch_idx = batch_idx_input;
    this->split_k_size = split_k_size;
    this->window_size = window_size;
    auto q_sizes = q.sizes();
    auto k_sizes = k.sizes();
    int b = q_sizes[0];
    int sq = q_sizes[1];
    int h = q_sizes[2];
    int d = q_sizes[3];
    assert(sq == 1);
    int sk = k_sizes[1];
    int h_k = k_sizes[2];

    InputShape options = {b, h, d, sk, h_k};

    const auto device = q.device();
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device.index();
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    run(options, hw_info);

    return std::make_tuple(block_o, block_lse);
  }

  ProblemShape _initialize(const InputShape& options) {
    int h_r = options.h / options.h_k;
    assert(options.h % options.h_k == 0);

    // Calculate split_kv = num_splits based on split_k_size parameter
    // split_k_size <= 0 means no split-K (num_splits = 1)
    // For sliding window attention, use window_size as the effective seqlen
    // This ensures output tensors are allocated with the correct number of splits
    int effective_seqlen = options.sk;
    if (window_size > 0 && window_size < options.sk) {
      effective_seqlen = window_size;
    }

    int split_kv = 1;
    if (split_k_size > 0) {
      split_kv = (effective_seqlen + split_k_size - 1) / split_k_size;
    }

    ProblemShape result = make_shape(
        _1{},
        options.sk,
        options.d,
        make_shape(make_shape(h_r, options.h_k), options.b));

    stride_q = make_stride(
        _0{},
        _1{},
        make_stride(
            make_stride(options.d, options.d * h_r),
            options.d * h_r * options.h_k));
    stride_new_k = make_stride(
        _0{},
        _1{},
        make_stride(make_stride(_0{}, static_cast<int64_t>(options.d)), static_cast<int64_t>(options.d * options.h_k)));
    stride_cache_k = make_stride(
        options.d * options.h_k,
        _1{},
        make_stride(
            make_stride(_0{}, static_cast<int64_t>(options.d)),
            static_cast<int64_t>(options.d * options.h_k * options.sk)));

    stride_new_v = stride_new_k;
    stride_cache_v = stride_cache_k;

    // Output layout: [B, H, num_splits, D]
    // Stride: (D*num_splits, 1, num_splits*H_K*D) for (H_K, D, B)
    stride_o = make_stride(
        _0{},
        _1{},
        make_stride(
            make_stride(options.d * split_kv, options.d * h_r * split_kv),
            options.d * h_r * options.h_k * split_kv));

    // LSE layout: [B, num_splits, H_R * H_K]
    stride_lse = make_stride(options.h_k * h_r * split_kv, h_r, _1{});
    // Output layout: [B, H_K, num_splits, D]
    // Shape for allocation: (B, H, num_splits, D) where H = h_r * h_k
    block_o = at::empty(
        {options.b, options.h, split_kv, options.d},
        at::TensorOptions()
            .dtype(to_torch_type<ElementOut>())
            .device(at::Device(at::kCUDA, at::cuda::current_device())));

    // LSE layout: [B, num_splits, H_R * H_K]
    // Shape for allocation: (B, num_splits, H) where H = h_r * h_k
    block_lse = at::empty(
        {options.b, split_kv, options.h},
        at::TensorOptions()
            .dtype(at::kFloat)
            .device(at::Device(at::kCUDA, at::cuda::current_device())));

    return result;
  }

  void run(
      const InputShape& options,
      const cutlass::KernelHardwareInfo& hw_info) {
    auto problem_shape = _initialize(options);

    typename Operation::Arguments arguments{
        problem_shape,
        static_cast<const int*>(seqlen_kv.const_data_ptr()),
        static_cast<const int*>(batch_idx ? batch_idx.value().const_data_ptr() : nullptr),
        static_cast<int>(split_k_size),
        static_cast<int>(window_size),
        static_cast<const Element*>(q.const_data_ptr()),
        stride_q,
        static_cast<const Element*>(nullptr),  // ptr_new_k
        stride_new_k,
        static_cast<const Element*>(nullptr),  // ptr_new_v
        stride_new_v,
        static_cast<Element*>(k.data_ptr()),
        stride_cache_k,
        static_cast<Element*>(v.data_ptr()),
        stride_cache_v,
        static_cast<ElementOut*>(block_o.data_ptr()),
        stride_o,
        static_cast<ElementAcc*>(block_lse.data_ptr()),
        stride_lse,
        hw_info,
        0.0f  // scale_softmax
    };

    Operation op;
    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize CUTLASS kernel." << std::endl;
      return;
    }


    status = op.run(
        at::cuda::getCurrentCUDAStream()
    );
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch CUTLASS kernel." << std::endl;
      return;
    }
  }
};

// Dispatch macros for different element types
#define DISPATCH_ELEMENT_TYPE(DTYPE, ELEMENT_TYPE, ...)                       \
  [&] {                                                                       \
    if (DTYPE == at::kFloat8_e4m3fn) {                                 \
      using ELEMENT_TYPE = cutlass::float_e4m3_t;                             \
      return __VA_ARGS__();                                                   \
    } else if (DTYPE == at::kBFloat16) {                                    \
      using ELEMENT_TYPE = cutlass::bfloat16_t;                             \
      return __VA_ARGS__();                                                 \
    } else {                                                                  \
      throw std::runtime_error("Unsupported dtype: " + std::to_string(static_cast<int>(DTYPE))); \
    }                                                                         \
  }()

// Dispatch macro for different kernel types
#define DISPATCH_KERNEL_TYPE(KTYPE, KERNEL_TYPE, ...)                         \
  [&] {                                                                       \
    if (KTYPE == static_cast<int>(KernelType::UMMA_P)) {                      \
      constexpr auto KERNEL_TYPE = KernelType::UMMA_P;                        \
      return __VA_ARGS__();                                                   \
    } else if (KTYPE == static_cast<int>(KernelType::UMMA_I)) {               \
      constexpr auto KERNEL_TYPE = KernelType::UMMA_I;                        \
      return __VA_ARGS__();                                                   \
    } else {                                                                  \
      throw std::runtime_error("Unsupported kernel type: " + std::to_string(KTYPE)); \
    }                                                                         \
  }()

// Dispatch macro for head dimension
#define DISPATCH_HEAD_DIM(HEAD_DIM, HEAD_DIM_VALUE, ...)        \
  [&] {                                                         \
    if (HEAD_DIM == 128) {                                      \
      constexpr int HEAD_DIM_VALUE = 128;                       \
      return __VA_ARGS__();                                     \
    } else if (HEAD_DIM == 64) {                                \
      constexpr int HEAD_DIM_VALUE = 64;                        \
      return __VA_ARGS__();                                     \
    } else {                                                    \
      throw std::runtime_error(                                 \
          "Unsupported head dim: " + std::to_string(HEAD_DIM)); \
    }                                                           \
  }()

template <typename Element, typename ElementOut, KernelType KType, int HeadDim>
std::tuple<at::Tensor, at::Tensor> run_gen_runner_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& seqlen_kv,
    const std::optional<at::Tensor>& batch_idx,
    int64_t split_k_size,
    int64_t window_size) {
  if constexpr (HeadDim == 128) {
    GenRunner<Element, ElementOut, KType, Shape<_64, _256, _128>> runner;
    return runner.fmha_fwd(q, k, v, seqlen_kv, batch_idx, split_k_size, window_size);
  } else if constexpr (HeadDim == 64) {
    GenRunner<Element, ElementOut, KType, Shape<_64, _256, _64>> runner;
    return runner.fmha_fwd(q, k, v, seqlen_kv, batch_idx, split_k_size, window_size);
  }
}

std::tuple<at::Tensor, at::Tensor> dispatch_fmha_gen_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& seqlen_kv,
    const std::optional<at::Tensor>& batch_idx,
    int64_t kernel_type,
    int64_t window_left,
    int64_t window_right,
    int64_t split_k_size
  ) {
  const auto device = q.device();
  at::cuda::CUDAGuard device_guard(device);
  const int head_dim = q.size(q.dim() - 1);
  const bool is_split_k = split_k_size > 0;

  // Convert window_left to window_size for the kernel
  // For decode, window_left is the relevant parameter (how many tokens to look back)
  // window_right is typically 0 for decode (causal)
  int64_t window_size = window_left + 1;
  assert(window_right == 0 || window_right == -1);

  // Decode kernel only supports window_right values of 0 or -1
  assert(window_right == 0 || window_right == -1);

  return DISPATCH_ELEMENT_TYPE(q.scalar_type(), Element, [&] {
    return DISPATCH_KERNEL_TYPE(static_cast<int>(kernel_type), KType, [&] {
      return DISPATCH_HEAD_DIM(head_dim, HeadDim, [&] {
        if (is_split_k) {
          // Split-K: output in float (accumulator precision for later reduction)
          return run_gen_runner_fwd<Element, float, KType, HeadDim>(
              q, k, v, seqlen_kv, batch_idx, split_k_size, window_size);
        } else {
          // Non-split: output in bfloat16
          return run_gen_runner_fwd<Element, cutlass::bfloat16_t, KType, HeadDim>(
              q, k, v, seqlen_kv, batch_idx, split_k_size, window_size);
        }
      });
    });
  });
}

std::tuple<at::Tensor, at::Tensor> dispatch_fmha_gen_fwd_meta(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& seqlen_kv,
    const std::optional<at::Tensor>& batch_idx,
    int64_t kernel_type,
    int64_t window_left,
    int64_t window_right,
    int64_t split_k_size
  ) {
  auto b = q.sym_size(0);
  auto sq = q.sym_size(1);
  auto h = q.sym_size(2);
  auto d = q.sym_size(3);
  assert(sq == 1);
  auto sk = k.sym_size(1);
  auto h_k = k.sym_size(2);

  c10::SymInt split_kv = 1;
  if (split_k_size > 0) {
    split_kv = (sk + split_k_size - 1) / split_k_size;
  }

  auto out_dtype = q.scalar_type();
  if(q.scalar_type() == at::kFloat8_e4m3fn) {
    // Output is BF16 when input is FP8
    out_dtype = at::kBFloat16;
  }

  auto output = at::empty_symint({b, h, split_kv, d}, q.options().dtype(out_dtype));
  auto lse = at::empty_symint({b, split_kv, h}, q.options().dtype(at::kFloat));
  return std::make_tuple(output, lse);
}

// -------------------------------------------------------------------------------------------------
// Op registration
// -------------------------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("fmha_gen_fwd("
      "    Tensor query, "
      "    Tensor key, "
      "    Tensor value, "
      "    Tensor seqlen_kv, "
      "    Tensor? batch_idx = None,"
      "    int kernel_type = 0,"
      "    int window_left = -1,"
      "    int window_right = -1,"
      "    int split_k_size = 1024"
      ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("fmha_gen_fwd", dispatch_fmha_gen_fwd);
}
TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("fmha_gen_fwd", dispatch_fmha_gen_fwd_meta);
}
#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

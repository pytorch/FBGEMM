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
    KernelType kKernelType,
    class TileShape,
    class ThreadShape>
struct GenRunner {
  using ElementAcc = float;
  using ElementOut = cutlass::bfloat16_t;

  using ProblemShape =
      Shape<_1, int, int, Shape<Shape<int, int>, int>>; // (Sq, Sk, D, ((H, Hr), B))

  using StrideQ =
      Stride<_0, _1, Stride<Stride<int, int>, int>>; // Q D ((H, Hr), B)
  using StrideNewK = Stride<_0, _1, Stride<Stride<_0, int>, int>>;
  using StrideCacheK =
      Stride<int, _1, Stride<Stride<_0, int>, int>>; // K D ((H, Hr), B)
  using StrideNewV = StrideNewK;
  using StrideCacheV = StrideCacheK;
  using StrideO = StrideQ;

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
          StrideO>;

  using TileScheduler = std::conditional_t<
      kKernelType == KernelType::UMMA_P,
      cutlass::fmha::kernel::PersistentTileScheduler,
      cutlass::fmha::kernel::IndividualTileScheduler>;

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

  at::Tensor block_o;
  at::Tensor q, k, v, seqlen_kv, batch_idx;

  at::Tensor fmha_fwd(
      const at::Tensor& q_input,
      const at::Tensor& k_input,
      const at::Tensor& v_input,
      const at::Tensor& seqlen_kv_input,
      const at::Tensor& batch_idx_input) {

    this->q = q_input;
    this->k = k_input;
    this->v = v_input;
    this->seqlen_kv = seqlen_kv_input;
    this->batch_idx = batch_idx_input;

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

    return block_o;
  }

  ProblemShape _initialize(const InputShape& options) {
    int h_r = options.h / options.h_k;
    assert(options.h % options.h_k == 0);

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
        make_stride(make_stride(_0{}, options.d), options.d * options.h_k));
    stride_cache_k = make_stride(
        options.d * options.h_k,
        _1{},
        make_stride(
            make_stride(_0{}, options.d),
            options.d * options.h_k * options.sk));

    stride_new_v = stride_new_k;
    stride_cache_v = stride_cache_k;
    stride_o = stride_q;

    block_o = at::empty(
        q.sizes(),
        at::TensorOptions()
            .dtype(to_torch_type<ElementOut>())
            .device(at::Device(at::kCUDA, at::cuda::current_device())));

    return result;
  }

  void run(
      const InputShape& options,
      const cutlass::KernelHardwareInfo& hw_info) {
    auto problem_shape = _initialize(options);

    typename Operation::Arguments arguments{
        problem_shape,
        static_cast<int*>(seqlen_kv.data_ptr()),
        static_cast<int*>(batch_idx.data_ptr()),
        static_cast<Element*>(q.data_ptr()),
        stride_q,
        nullptr,
        stride_new_k,
        nullptr,
        stride_new_v,
        static_cast<Element*>(k.data_ptr()),
        stride_cache_k,
        static_cast<Element*>(v.data_ptr()),
        stride_cache_v,
        static_cast<ElementOut*>(block_o.data_ptr()),
        stride_o,
        hw_info};

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
// TODO(henrylhtsang / ayaoibrahim1123): Add support for other data types.
#define DISPATCH_ELEMENT_TYPE(DTYPE, ELEMENT_TYPE, ...)                       \
  [&] {                                                                       \
    if (DTYPE == at::kFloat8_e4m3fn) {                                 \
      using ELEMENT_TYPE = cutlass::float_e4m3_t;                             \
      return __VA_ARGS__();                                                   \
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

at::Tensor dispatch_fmha_gen_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& seqlen_kv,
    const at::Tensor& batch_idx,
    int64_t kernel_type) {
  const auto device = q.device();
  at::cuda::CUDAGuard device_guard(device);

  return DISPATCH_ELEMENT_TYPE(q.scalar_type(), Element, [&] {
    return DISPATCH_KERNEL_TYPE(static_cast<int>(kernel_type), KType, [&] {
      GenRunner<Element, KType, Shape<_128, _128, _128>, Shape<_1, _1, _1>>
          runner;
      return runner.fmha_fwd(q, k, v, seqlen_kv, batch_idx);
    });
  });
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
        "    Tensor batch_idx, "
        "    int kernel_type = 0"
        ") -> Tensor"
  );
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("fmha_gen_fwd", dispatch_fmha_gen_fwd);
}
#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

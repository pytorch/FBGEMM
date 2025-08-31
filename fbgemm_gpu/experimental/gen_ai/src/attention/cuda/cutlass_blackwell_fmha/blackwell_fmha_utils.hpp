/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef USE_ROCM
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#endif

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <iostream>
#include "reference/reference_abs_error.hpp"

#include <ATen/core/dispatch/Dispatcher.h>
#include <signal.h>
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "device/fmha.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

#include "blackwell_gen_interface.hpp"
#include "collective/fmha_fusion.hpp"
#include "device/fmha_device_bwd.hpp"

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;

template <typename T>
struct TensorWrapper {
  at::Tensor tensor_;
  size_t offset_ = 0;
  size_t size_ = 0;

  TensorWrapper(TensorWrapper const&) = delete;
  TensorWrapper& operator=(TensorWrapper const&) = delete;

  TensorWrapper() = default;
  TensorWrapper(size_t size) {
    reset(size);
  }

  void reset(size_t size, size_t offset = 0) {
    tensor_ = at::empty(
        {static_cast<int64_t>(size + offset)},
        at::TensorOptions()
            .dtype(to_torch_type<T>())
            .device(at::Device(at::kCUDA, at::cuda::current_device())));
    size_ = size;
    offset_ = offset;
  }

  T* ptr() {
    return static_cast<T*>(tensor_.data_ptr());
  }

  T* get() {
    return ptr() + offset_;
  }

  const T* get() const {
    return ptr() + offset_;
  }

  size_t size() const {
    return size_;
  }

  at::Tensor get_data_tensor(const c10::IntArrayRef& shape) {
    if (offset_ == 0) {
      return tensor_.view(shape);
    }
    auto t = tensor_.narrow(0, offset_, size_).view(shape);
    TORCH_CHECK(
        t.is_contiguous(),
        "The underlying tensor in TensorWrapper must be contiguous");
    return t;
  }

  void copy_from_device(const at::Tensor& tensor) {
    // Use memcpyAsync to avoid H-D sync
    auto ret = cudaMemcpyAsync(
        get(),
        tensor.data_ptr(),
        tensor.numel() * sizeof(T),
        cudaMemcpyDefault,
        at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(ret == cudaSuccess);
  }
};

template <typename Operation>
static void launch_fmha_op(const typename Operation::Arguments& arguments) {
  size_t workspace_size = 0;
  workspace_size = Operation::get_workspace_size(arguments);

  auto workspace = at::empty(
      {static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte).device(
          at::Device(at::kCUDA, at::cuda::current_device())));

  Operation op;

  cutlass::Status status = cutlass::Status::kSuccess;

  status = op.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "This kernel is not supported. Last CUDA error is: ",
      cudaGetErrorString(cudaGetLastError()));

  status = op.initialize(arguments, workspace.mutable_data_ptr());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to initialize the CUTLASS kernel. Last CUDA error is: ",
      cudaGetErrorString(cudaGetLastError()));

  // Run
  status = op.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to launch the CUTLASS kernel. Last CUDA error is: ",
      cudaGetErrorString(cudaGetLastError()));
}
#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

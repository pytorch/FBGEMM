/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include "fbgemm_gpu/utils/device_properties.cuh"
#include "fbgemm_gpu/utils/source_context.h"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

#include <type_traits>

namespace fbgemm_gpu::utils {

#define U64(x) static_cast<uint64_t>(x)

////////////////////////////////////////////////////////////////////////////////
// Helpers to detect TensorAccessorBuilder type (regardless of template params)
////////////////////////////////////////////////////////////////////////////////

template <typename>
struct is_tensor_accessor_builder : std::false_type {};

template <
    typename T,
    size_t N,
    size_t INB,
    bool P,
    template <typename>
    class PT>
struct is_tensor_accessor_builder<TensorAccessorBuilder<T, N, INB, P, PT>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_tensor_accessor_builder_v =
    is_tensor_accessor_builder<T>::value;

////////////////////////////////////////////////////////////////////////////////
// Transform Kernel Argument
//
// Transform certain arguments before passing them to the kernel invocation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
decltype(auto) transform_kernel_arg(const SourceContext& context, T&& arg) {
  if constexpr (is_tensor_accessor_builder_v<std::decay_t<T>>) {
    // If the arg is a TensorAccessorBuilder, build it out to a tensor accessor.
    // This is the mechanism that allows us to log kernel function names on
    // failed checks and assertions when comopiled with FBGEMM_GPU_MEMCHECK
    // turned ON.
    return arg.build(context.description());
  } else {
    // Otherwise, perfect-forward the argument as is
    return std::forward<T>(arg);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Verify Kernel Argument
//
// Verify certain arguments before and after kernel invocation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
decltype(auto) check_kernel_arg(const SourceContext& context, T&& arg) {
  if constexpr (is_tensor_accessor_builder_v<std::decay_t<T>>) {
    // If the arg is a TensorAccessorBuilder, run verifications on the tensor it
    // is ref-wrapping, e.g. NaN value checks.
    return arg.checkValues(context.description());
  } else {
    // Otherwise, perfect-forward the argument as is
    return std::forward<T>(arg);
  }
}

////////////////////////////////////////////////////////////////////////////////
// GPU Kernel Launcher
//
// This class encapsulates the common ceremonial pre- and post-execution
// routines when launching GPU kernels.
////////////////////////////////////////////////////////////////////////////////

template <
    bool EnableDSA = false,
    bool EnableBarrierIsolation = false,
    bool EnableNaNChecks = false>
struct KernelLauncher {
  const SourceContext context;

  constexpr inline KernelLauncher(
      const source_location& location,
      const std::string_view& summary,
      const std::string_view& secondaryLocation) noexcept
      : context(SourceContext(location, summary, secondaryLocation)) {}

  constexpr inline void checkGridSizesInRange(
      const cudaDeviceProp& properties,
      const dim3& grid) const {
    const auto grid_limits = properties.maxGridSize;

    TORCH_CHECK(
        grid.x > 0 && grid.x <= grid_limits[0],
        context.description(),
        ": grid.x value ",
        grid.x,
        " is not within the range (0, ",
        grid_limits[0],
        ")");

    TORCH_CHECK(
        grid.y > 0 && grid.y <= grid_limits[1],
        context.description(),
        ": grid.y value ",
        grid.y,
        " is not within the range (0, ",
        grid_limits[1],
        ")");

    TORCH_CHECK(
        grid.z > 0 && grid.z <= grid_limits[2],
        context.description(),
        ": grid.z value ",
        grid.z,
        " is not within the range (0, ",
        grid_limits[2],
        ")");
  }

  constexpr inline void checkBlockSizesInRange(
      const cudaDeviceProp& properties,
      const dim3& block) const {
    const auto block_limits = properties.maxThreadsDim;

    TORCH_CHECK(
        block.x > 0 && block.x <= block_limits[0],
        context.description(),
        ": block.x value ",
        block.x,
        " is not within the range (0, ",
        block_limits[0],
        ")");

    TORCH_CHECK(
        block.y > 0 && block.y <= block_limits[1],
        context.description(),
        ": block.y value ",
        block.y,
        " is not within the range (0, ",
        block_limits[1],
        ")");

    TORCH_CHECK(
        block.z > 0 && block.z <= block_limits[2],
        context.description(),
        ": block.z value ",
        block.z,
        " is not within the range (0, ",
        block_limits[2],
        ")");
  }

  constexpr inline void checkThreadCountNotExceeded(
      const cudaDeviceProp& properties,
      const dim3& grid,
      const dim3& block) const {
    const uint64_t threads_per_block =
        U64(block.x) * U64(block.y) * U64(block.z);

    TORCH_CHECK(
        threads_per_block <= properties.maxThreadsPerBlock,
        context.description(),
        ": Threads per block ",
        threads_per_block,
        " is greater than the limit of ",
        properties.maxThreadsPerBlock);

#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
    // ROCm has a limit of 2^32 elements per kernel launch, but doesn't
    // automatically work around problem like CUDA does (V100 or newer
    // architectures), see:
    //    https://github.com/ROCm/hip/issues/2253
    const uint64_t total_threads = U64(grid.x) * U64(grid.y) * U64(grid.z) *
        U64(block.x) * U64(block.y) * U64(block.z);

    TORCH_CHECK(
        total_threads < U64(std::numeric_limits<uint32_t>::max()),
        context.description(),
        ": Total number of threads ",
        total_threads,
        " is greater than the limit of 2^32");
#endif
  }

  constexpr inline void checkSharedMemoryPerBlockNotExceeded(
      const cudaDeviceProp& properties,
      const size_t shared_mem_per_block) const {
    // NOTE: sharedMemPerBlockOptin is the maximum possible shared memory that
    // can be used per block by explicit special opt-in, and is generally larger
    // than sharedMemPerBlock.
    //
    // However, this feature does not exist in HIP at the moment, and while more
    // recent versions of ROCm (6.4+?) set the value of sharedMemPerBlockOptin
    // to be sharedMemPerBlock, older versions of ROCm set the value to zero.
    //
    // See:
    //  https://github.com/ROCm/HIP/issues/3516
#ifdef __HIP_PLATFORM_AMD__
    const auto smem_limits = properties.sharedMemPerBlock;
#else
    const auto smem_limits = properties.sharedMemPerBlockOptin;
#endif

    TORCH_CHECK(
        shared_mem_per_block <= smem_limits,
        context.description(),
        ": Requested shared memory per block (",
        shared_mem_per_block,
        " bytes) is not within the range [0, ",
        smem_limits,
        "]");
  }

  inline void kernelLaunchCheck() const {
    // This is a replacement for C10_CUDA_KERNEL_LAUNCH_CHECK() that adds more
    // context information to the error message.  See:
    //  https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAException.cpp

    const auto cuda_error = cudaGetLastError();

    const auto cuda_kernel_failure =
        c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().has_failed();

    if (C10_LIKELY(cuda_error == cudaSuccess && !cuda_kernel_failure)) {
      return;
    }

    // Inject the context information into the error message on CUDA failures
    TORCH_CHECK(
        false,
        context.description(),
        " CUDA Error: ",
        cudaGetErrorString(cuda_error),
#ifdef __HIPCC__
        // c10::cuda::get_cuda_check_suffix has only been recently added to
        // Torch HIPify mappings, so wrap with __HIPCC__ until the mapping land
        // in PyTorch OSS.
        //
        // TODO: Remove when HIPify mappings are updated in PyTorch OSS
        c10::hip::get_hip_check_suffix(),
#else
        c10::cuda::get_cuda_check_suffix(),
#endif
        "\n",
        c10::cuda::c10_retrieve_device_side_assertion_info());
  }

  template <typename KernelFunc, typename... Args>
  inline void launch_kernel(
      const KernelFunc& kernel,
      const dim3 grid,
      const dim3 block,
      const size_t shared_mem_per_block,
      const c10::cuda::CUDAStream stream,
      Args&&... args) const {
    // Fetch device properties from the stream information
    const auto device = stream.device_index();
    const auto properties = *at::cuda::getDeviceProperties(device);
    const auto streamId = stream.id();

    // Check that the grid sizes are within the range per the device associated
    // with the compute stream
    checkGridSizesInRange(properties, grid);

    // Check that the grid sizes are within the range per the device associated
    // with the compute stream
    checkBlockSizesInRange(properties, block);

    // Check that the thread count (per block and global) is not exceeded
    checkThreadCountNotExceeded(properties, grid, block);

    // Check that the shared memory allocation is within the range per the
    // device associated with the compute stream
    checkSharedMemoryPerBlockNotExceeded(properties, shared_mem_per_block);

    // If NaN checks are enabled, run verifications on all kernel arguments that
    // are tensors
    if constexpr (EnableNaNChecks) {
      const auto summary = std::string(context.summary) + " (pre-execution)";
      (check_kernel_arg(context.withSummary(summary), std::forward<Args>(args)),
       ...);
    }

    // If barrier isolation is enabled, synchronize the stream first before
    // launching the kernel.  This has roughly the same effect as setting
    // `CUDA_LAUNCH_BLOCKING=1` as an environment variable.
    if constexpr (EnableBarrierIsolation) {
      cudaDeviceSynchronize();
    }

    if constexpr (EnableDSA) {
      // This launch code here is essentially the same as the contents of
      // TORCH_USE_CUDA_DSA macro, but with the addition of kernel argument
      // transformation.

      auto& launch_registry =
          c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref();

      // Launch the kernel
      kernel<<<grid, block, shared_mem_per_block, stream>>>(
          // Transform arguments to the kernel before forwarding them.
          transform_kernel_arg(context, std::forward<Args>(args))...,
          launch_registry.get_uvm_assertions_ptr_for_current_device(),
          launch_registry.insert(
              context.location.file_name(),
              context.location.function_name(),
              context.location.line(),
              context.summary.data(),
              streamId));

    } else {
      // Launch the kernel
      kernel<<<grid, block, shared_mem_per_block, stream>>>(
          // Transform arguments to the kernel before forwarding them.
          transform_kernel_arg(context, std::forward<Args>(args))...);
    }

    // If barrier isolation is enabled, synchronize the stream again to wait for
    // kernel execution to complete
    if constexpr (EnableBarrierIsolation) {
      cudaDeviceSynchronize();
    }

    // Check for CUDA errors.  This is a replacement for
    // C10_CUDA_KERNEL_LAUNCH_CHECK() that adds more context information to the
    // error message.
    kernelLaunchCheck();

    // If NaN checks are enabled, run post-kernel verifications on all kernel
    // arguments that are tensors
    if constexpr (EnableNaNChecks) {
      const auto summary = std::string(context.summary) + " (post-execution)";
      (check_kernel_arg(context.withSummary(summary), std::forward<Args>(args)),
       ...);
    }
  }
};

#undef U64

} // namespace fbgemm_gpu::utils

////////////////////////////////////////////////////////////////////////////////
// General Kernel Launch Macros for FBGEMM GPU Kernels
//
// This macro is used to launch GPU kernels in FBGEMM GPU codebase. It runs a
// set of constraint checks on kernel parameters and and tensor arguments, and
// throws descriptive errors on constraint failures.
//
// NOTES:
//
//  - Since the code is wrapped inside an immediately-invoked lambda,
//  source_location::current() will point to the function where the macro is
//  called.
//
//  - The constexpr decltype(KERNEL) declaration is added to enable for better
//  compilation error messages upon template argument and function overload
//  mismatches.
//
//  - The macro expression is wrapped inside a parenthesis to avoid commas from
//  interfering with preoprocessing when this macro is invoked inside another
//  macro.
////////////////////////////////////////////////////////////////////////////////

#ifdef __TEMPLATE_SOURCE_FILE__
#define _FKL_TFILE_ __TEMPLATE_SOURCE_FILE__
#else
#define _FKL_TFILE_ ""
#endif

#ifdef FBGEMM_GPU_ISOLATE_KERNEL_LAUNCH
#define _FKL_BLOCKING_ true
#else
#define _FKL_BLOCKING_ false
#endif

#ifdef FBGEMM_GPU_TENSORCHECK
#define _FKL_TENSORCHECK_ true
#else
#define _FKL_TENSORCHECK_ false
#endif

#define FBGEMM_LAUNCH_KERNEL(KERNEL, GRID, BLOCK, SMEM, STREAM, ...)        \
  ([&] {                                                                    \
    using source_location = fbgemm_gpu::utils::source_location;             \
    constexpr auto location = source_location::current();                   \
    decltype(KERNEL)& kernel = KERNEL;                                      \
                                                                            \
    return fbgemm_gpu::utils::                                              \
        KernelLauncher<false, _FKL_BLOCKING_, _FKL_TENSORCHECK_>(           \
               location, #KERNEL, _FKL_TFILE_)                              \
            .launch_kernel(kernel, GRID, BLOCK, SMEM, STREAM, __VA_ARGS__); \
  }())

#define FBGEMM_LAUNCH_DSA_KERNEL(KERNEL, GRID, BLOCK, SMEM, STREAM, ...)    \
  ([&] {                                                                    \
    using source_location = fbgemm_gpu::utils::source_location;             \
    constexpr auto location = source_location::current();                   \
    decltype(KERNEL)& kernel = KERNEL;                                      \
                                                                            \
    return fbgemm_gpu::utils::                                              \
        KernelLauncher<true, _FKL_BLOCKING_, _FKL_TENSORCHECK_>(            \
               location, #KERNEL, _FKL_TFILE_)                              \
            .launch_kernel(kernel, GRID, BLOCK, SMEM, STREAM, __VA_ARGS__); \
  }())

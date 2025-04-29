/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
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
// GPU Kernel Launcher
//
// This class encapsulates the common ceremonial pre- and post-execution
// routines when launching GPU kernels.
////////////////////////////////////////////////////////////////////////////////

template <bool EnableDSA = false, bool EnableBarrierIsolation = false>
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
    const auto smem_limits = properties.sharedMemPerBlockOptin;

    TORCH_CHECK(
        shared_mem_per_block <= smem_limits,
        context.description(),
        ": Requested shared memory per block (",
        shared_mem_per_block,
        " bytes) is not within the range [0, ",
        smem_limits,
        "]");
  }

  template <typename KernelFunc, typename... Args>
  inline void launch_kernel(
      const KernelFunc& kernel,
      const dim3 grid,
      const dim3 block,
      const size_t shared_mem_per_block,
      const cudaStream_t stream,
      Args&&... args) const {
    // Fetch device properties from the stream information
    const auto device = get_device_for_stream(stream);
    const auto properties = get_device_properties(device);
    const auto streamId = get_stream_id(stream);

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

    if constexpr (EnableDSA) {
      // This launch code here is essentially the same as the contents of
      // TORCH_USE_CUDA_DSA macro, but with the addition of kernel argument
      // transformation.

      auto& launch_registry =
#ifdef __HIPCC__
          // CUDAKernelLaunchRegistry has only been recently added to Torch
          // HIPify mappings, so wrap this with USE_ROCM until the mappings land
          // in PyTorch OSS.
          //
          // TODO: Remove when CUDAKernelLaunchRegistry lands in the nightlies
          c10::hip::HIPKernelLaunchRegistry::get_singleton_ref();
#else
          c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref();
#endif

      // If barrier isolation is enabled, synchronize the stream first before
      // launching the kernel.  This has roughly the same effect as setting
      // `CUDA_LAUNCH_BLOCKING=1` as an environment variable.
      if constexpr (EnableBarrierIsolation) {
        cudaDeviceSynchronize();
      }

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

    // Check for CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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

#ifdef FBGEMM_GPU_KERNEL_DEBUG
#define _FKL_KDEBUG_ true
#else
#define _FKL_KDEBUG_ false
#endif

#define FBGEMM_LAUNCH_KERNEL(KERNEL, GRID, BLOCK, SMEM, STREAM, ...)    \
  ([&] {                                                                \
    using source_location = fbgemm_gpu::utils::source_location;         \
    constexpr auto location = source_location::current();               \
    decltype(KERNEL)& kernel = KERNEL;                                  \
                                                                        \
    return fbgemm_gpu::utils::KernelLauncher<false, _FKL_KDEBUG_>(      \
               location, #KERNEL, _FKL_TFILE_)                          \
        .launch_kernel(kernel, GRID, BLOCK, SMEM, STREAM, __VA_ARGS__); \
  }())

#define FBGEMM_LAUNCH_DSA_KERNEL(KERNEL, GRID, BLOCK, SMEM, STREAM, ...) \
  ([&] {                                                                 \
    using source_location = fbgemm_gpu::utils::source_location;          \
    constexpr auto location = source_location::current();                \
    decltype(KERNEL)& kernel = KERNEL;                                   \
                                                                         \
    return fbgemm_gpu::utils::KernelLauncher<true, _FKL_KDEBUG_>(        \
               location, #KERNEL, _FKL_TFILE_)                           \
        .launch_kernel(kernel, GRID, BLOCK, SMEM, STREAM, __VA_ARGS__);  \
  }())

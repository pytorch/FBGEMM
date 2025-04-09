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

#include "fbgemm_gpu/utils/source_context.h"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

#include <iostream>
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
    // Otherwise, forward the argument as is
    return std::forward<T>(arg);
  }
}

////////////////////////////////////////////////////////////////////////////////
// GPU Kernel Launcher
//
// This class encapsulates the common ceremonial pre- and post-execution
// routines when launching GPU kernels.
////////////////////////////////////////////////////////////////////////////////

struct KernelLauncher {
  const SourceContext context;

  constexpr inline KernelLauncher(
      const source_location& location,
      const std::string_view& summary) noexcept
      : context(SourceContext(location, summary)) {}

  template <typename KernelFunc, typename... Args>
  constexpr inline void launch_kernel(
      const KernelFunc& kernel,
      const dim3 grid,
      const dim3 block,
      const size_t Ns,
      cudaStream_t stream,
      Args&&... args) {
    // TODO: Check smem size based on actual hardware

#ifdef USE_ROCM
    // ROCm has a limit of 2^32 elements per kernel launch, but doens't
    // automatically work around problem like CUDA does, see:
    //  https://github.com/ROCm/hip/issues/2253
    const uint64_t grid_size = U64(grid.x) * U64(grid.y) * U64(grid.z) *
        U64(block.x) * U64(block.y) * U64(block.z);
    TORCH_CHECK(
        grid_size < U64(std::numeric_limits<uint32_t>::max()),
        context.description(),
        ": Kernel launch grid size ",
        grid_size,
        " is greater than the ROCm limit of 2^32");
#endif

    // Launch the kernel
    kernel<<<grid, block, Ns, stream>>>(
        // Transform arguments to the kernel before forwarding them.
        transform_kernel_arg(context, std::forward<Args>(args))...);

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
////////////////////////////////////////////////////////////////////////////////

#define FBGEMM_LAUNCH_KERNEL(KERNEL, GRID, BLOCK, SMEM, STREAM, ...)    \
  [&] {                                                                 \
    using source_location = fbgemm_gpu::utils::source_location;         \
    constexpr auto location = source_location::current();               \
    constexpr decltype(KERNEL)& kernel = KERNEL;                        \
                                                                        \
    return fbgemm_gpu::utils::KernelLauncher(location, #KERNEL)         \
        .launch_kernel(kernel, GRID, BLOCK, SMEM, STREAM, __VA_ARGS__); \
  }()

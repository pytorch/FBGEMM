/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdlib>
#include <string>
#include <tuple>

#include <ATen/core/Tensor.h>
#include <c10/hip/HIPStream.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp"
#include "kernels/fp8_rowwise_grouped_kernel_manifest.h"
#include "kernels/fp8_rowwise_grouped_heuristic.hpp"
#include "fbgemm_gpu/quantize/tuning_cache.hpp"
#include "fbgemm_gpu/quantize/utils.h"

namespace fbgemm_gpu {
namespace {

enum GroupedGemmInputType {
  // MNK fixed (batched gemm)
  _3D3D,
  // K dynamic
  _2D2D,
  // N dynamic
  _3D2D,
  // M dynamic (MoE style)
  _2D3D
};

// Define useful types that are needed for various kernels.
using KernelArguments =
    ck::tensor_operation::device::GroupedGemmKernelArgument<2>;
using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using D0DataType = float;
using D1DataType = float;
using DsDataType = ck::Tuple<D0DataType, D1DataType>;
using EDataType = ck::bhalf_t;

TuningCache& getTuningCache() {
  // This kernel has multiple APIs templated based on InputType, so we use this
  // to have a single cache instance across APIs.
  static TuningCache cache("fp8_rowwise_grouped_gemm");
  return cache;
}

RowwiseGroupedKernel<at::Tensor, at::Tensor>
get_kernel_via_tuning(int64_t M, int64_t N, int64_t K, at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor kernel_args, at::Tensor out) {
  auto& cache = getTuningCache();

  // Reducing amount of auto tuning by rounding up M, N, K to next power of 2.
  M = nextPowerOf2(M);
  N = nextPowerOf2(N);
  K = nextPowerOf2(K);

  // Use (M, N, K) shape as the key.
  const std::string shape_key = std::to_string(M) + "_" +
      std::to_string(N) + "_" + std::to_string(K);
  const auto& kernels = get_fp8_rowwise_grouped_kernels<at::Tensor, at::Tensor>();
  auto kernel = cache.findBestKernelMaybeAutotune(
      shape_key, kernels, XQ, WQ, x_scale, w_scale, kernel_args, out);

  return kernel;
}

__global__ void set_kernel_args_kernel(
    KernelArguments* kernel_args,
    ADataType* XQ,
    BDataType* WQ,
    D0DataType* w_scale,
    D1DataType* x_scale,
    EDataType* output,
    int64_t M,
    int64_t N,
    int64_t K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each kernel annoyingly can only set the kernel args for one group.
  // This could only be avoided with complicated memory management.
  if (idx == 0) {
    // Write kernel arguments directly to memory.
    KernelArguments kernel_group_args = {
        XQ, WQ, {w_scale, x_scale}, output, int(M), int(N), int(K), int(K), int(K), {0, 0}, int(N)};
    kernel_args[0] = kernel_group_args;
  }
}

template <typename OutputType>
void set_static_kernel_args(
    at::Tensor kernel_args,
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    OutputType output) {
  // Get current cuda stream.
  auto stream = at::cuda::getCurrentHIPStream().stream();
  int64_t group_count = XQ.size();
  // Declare variables for loop.
  EDataType* output_ptr;
  int64_t output_offset = 0;
  // When group count is large, we can more efficiently initialize
  // by doing host setup and a memcpy. This is only viable if cuda
  // graphs aren't being used.
  // Iterate over inputs and get group information.
  for (int i = 0; i < group_count; i++) {
    int64_t M = XQ[i].size(0);
    int64_t K = XQ[i].size(1);
    int64_t N = WQ[i].size(0);

    // Compute proper output pointer.
    if constexpr (std::is_same_v<OutputType, std::vector<at::Tensor>>) {
      // Output is a list of tensors and we can access each individually.
      output_ptr = reinterpret_cast<EDataType*>(output[i].data_ptr());
    } else {
      // Output is a single contiguous tensor and must be accessed via offset.
      output_ptr =
          reinterpret_cast<EDataType*>(output.data_ptr()) + output_offset;
      output_offset += M * N;
    }

    // We use the smallest reasonable block size since we effectively need only
    // 1 thread.
    // Launch a kernel for each group to set kernel memory on device.
    // Using multiple kernels this way allows us to support arbitrary M,N,K.
    // For some reason, this approach is faster than using hipmemcpy.
    // Launch kernel to set kernel arguments.
    set_kernel_args_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<KernelArguments*>(
            reinterpret_cast<char*>(kernel_args.data_ptr()) +
            (i * sizeof(KernelArguments))),
        reinterpret_cast<ADataType*>(XQ[i].data_ptr()),
        reinterpret_cast<BDataType*>(WQ[i].data_ptr()),
        reinterpret_cast<D0DataType*>(w_scale[i].data_ptr()),
        reinterpret_cast<D1DataType*>(x_scale[i].data_ptr()),
        output_ptr,
        M,
        N,
        K);
  }
}

// Supports using either M_sizes or offsets.
__global__ void set_kernel_args(
    KernelArguments* kernel_args,
    ADataType* XQ,
    BDataType* WQ,
    D0DataType* w_scale,
    D1DataType* x_scale,
    EDataType* output,
    int64_t* M_sizes,
    int32_t* offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_count,
    std::optional<GroupedGemmInputType> input_type = std::nullopt) {
  // The "message" part is not working on AMD currently :(
  CUDA_KERNEL_ASSERT_MSG((M_sizes == nullptr && offsets == nullptr) || (M_sizes == nullptr ^ offsets == nullptr), "Cannot set both M_sizes and offsets");
  CUDA_KERNEL_ASSERT_MSG(input_type.has_value() || M_sizes != nullptr, "M_sizes should not be used with input_type");

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread is responsible for setting up the arguments for one group.
  if (thread_idx < group_count) {
    // In cases where M < G, we want to only set M groups since the rest are empty.
    // To do this, we use a counter into the group argument tensor.
    __shared__ int non_zero_counter;
    // Initialize the counter in the first thread.
    if (thread_idx == 0) {
      non_zero_counter = 0;
    }
    // We need to set a default argument for all M groups.
    KernelArguments default_group_args = {
      XQ,
      WQ,
      {w_scale, x_scale},
      output,
      0,
      0,
      0,
      0,
      0,
      {0, 0},
      0};
    kernel_args[thread_idx] = default_group_args;
    // Sync threads to get consistent state.
    __syncthreads();

    // Offset information for this group.
    int64_t XQ_offset;
    int64_t WQ_offset;
    int64_t x_scale_offset;
    int64_t w_scale_offset;
    int64_t output_offset;
    // Strides for this group
    int64_t A_stride = K;
    int64_t B_stride = K;
    int64_t output_stride = N;
    // Problem size for this group. Dynamic dimension for the group would updated below.
    int64_t M_group = M;
    int64_t N_group = N;
    int64_t K_group = K;

    // M_sizes API implies 2D-3D inputs
    if (M_sizes != nullptr) {
      M_group = M_sizes[thread_idx];
      if (M_group > 0) {
        // Offset is computed by finding the sum of previous group Ms.
        int64_t offset_M = 0;
        for (int i = 0; i < thread_idx; i++) {
          offset_M += M_sizes[i];
        }

        XQ_offset = offset_M * K;
        WQ_offset = thread_idx * N * K;
        x_scale_offset = offset_M;
        w_scale_offset = thread_idx * N;
        output_offset = offset_M * N;
      }
    } else {
      if (input_type == GroupedGemmInputType::_2D3D) {
        const int32_t offset_M = thread_idx == 0 ? 0 : offsets[thread_idx - 1];
        M_group = offsets[thread_idx] - offset_M;

        XQ_offset = offset_M * K;
        WQ_offset = thread_idx * N * K;
        x_scale_offset = offset_M;
        w_scale_offset = thread_idx * N;
        output_offset = offset_M * N;
      } else if (input_type == GroupedGemmInputType::_3D2D) {
        const int32_t offset_N = thread_idx == 0 ? 0 : offsets[thread_idx - 1];
        N_group = offsets[thread_idx] - offset_N;

        XQ_offset = thread_idx * M * K;
        WQ_offset = offset_N * K;
        x_scale_offset = thread_idx * M;
        w_scale_offset = offset_N;
        // Offset of offset_N as the N dimension of the output is across the input gemm problems.
        output_offset = offset_N;
      } else if (input_type == GroupedGemmInputType::_2D2D) {
        const int32_t offset_K = thread_idx == 0 ? 0 : offsets[thread_idx - 1];
        K_group = offsets[thread_idx] - offset_K;

        XQ_offset = offset_K;
        WQ_offset = offset_K;
        x_scale_offset = thread_idx * M;
        w_scale_offset = thread_idx * N;
        output_offset = thread_idx * M * N;
      } else {
        XQ_offset = thread_idx * M * K;
        WQ_offset = thread_idx * N * K;
        x_scale_offset = thread_idx * M;
        w_scale_offset = thread_idx * N;
        output_offset = thread_idx * M * N;
      }
    }

    CUDA_KERNEL_ASSERT_MSG(N_group % 8 == 0, "N must be divisible by 8");

    // Only write actual group information if this group is nonzero.
    if (M_group > 0 && N_group > 0 && K_group > 0) {
      // Get index automatically for this group.
      const int non_zero_idx = atomicAdd(&non_zero_counter, 1);
      KernelArguments kernel_group_args = {
          XQ + XQ_offset, // A
          WQ + WQ_offset, // B
          {w_scale + w_scale_offset, x_scale + x_scale_offset}, // Ds
          output + output_offset, // E
          int(M_group), // M
          int(N_group), // N
          int(K_group), // K
          int(A_stride), // StrideA
          int(B_stride), // StrideB
          {0, 0}, // StrideDs
          int(output_stride) // StrideE
      };
      // Write kernel args to memory.
      kernel_args[non_zero_idx] = kernel_group_args;
    }
  }
}

__global__ void set_kernel_args_fixed_nk_kernel(
    KernelArguments* kernel_args,
    ADataType* XQ,
    BDataType* WQ,
    D0DataType* w_scale,
    D1DataType* x_scale,
    EDataType* output,
    int64_t* prepad_M,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_count) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread is responsible for setting up the arguments for one group.
  if (thread_idx < group_count) {
    // Compute offsets for this group.
    int64_t group_M = prepad_M[thread_idx];
    KernelArguments kernel_group_args = {
        XQ + (thread_idx * M * K),
        WQ + (thread_idx * N * K),
        {w_scale + (thread_idx * N), x_scale + (thread_idx * M)},
        output + (thread_idx * M * N),
        int(group_M),
        int(N),
        int(K),
        int(K),
        int(K),
        {0, 0},
        int(N)};
    // Write kernel args to memory.
    kernel_args[thread_idx] = kernel_group_args;
  }
}

__global__ void set_kernel_args_fixed_nk_kernel_zeroing(
    KernelArguments* kernel_args,
    ADataType* XQ,
    BDataType* WQ,
    D0DataType* w_scale,
    D1DataType* x_scale,
    EDataType* output,
    int64_t* prepad_M,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_count) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread is responsible for setting up the arguments for one group.
  if (thread_idx < group_count) {
    // Compute offsets for this group.
    int64_t group_M = prepad_M[thread_idx];
    KernelArguments kernel_group_args = {
        XQ + (thread_idx * M * K),
        WQ + (thread_idx * N * K),
        {w_scale + (thread_idx * N), x_scale + (thread_idx * M)},
        output + (thread_idx * M * N),
        int(group_M),
        int(N),
        int(K),
        int(K),
        int(K),
        {0, 0},
        int(N)};
    // Write kernel args to memory.
    kernel_args[thread_idx] = kernel_group_args;
  }

  // Figure out where in memory we are.
  // Each thread sets one float 4 which corresponds to 8 bf16 values.
  int output_offset = (thread_idx * 8);
  int current_group = output_offset / (M * N);
  // Skip if outside of valid groups.
  if (current_group < group_count) {
    int nonzeros = prepad_M[current_group];
    int current_M = (output_offset % (M * N)) / N;
    // Only write zeros if we're currently in a sparse row.
    if (current_M >= nonzeros) {
      // Write out a block of 8 output values via vectorized float4.
      float4* output_block = reinterpret_cast<float4*>(output + output_offset);
      *output_block = {0, 0, 0, 0};
    }
  }
}

void set_dynamic_kernel_args(
    at::Tensor kernel_args,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    bool zeroing_output_tensor) {
  // Get current cuda stream.
  auto stream = at::cuda::getCurrentHIPStream().stream();
  int64_t group_count;
  // Check provided tensors are valid.
  TORCH_CHECK(
      zero_start_index_M.has_value() != M_sizes.has_value(),
      "One of zero_start_index_M or M_sizes must be provided.");
  if (zero_start_index_M.has_value()) {
    group_count = zero_start_index_M.value().size(0);
    TORCH_CHECK(
        XQ.device() == zero_start_index_M.value().device(),
        "zero_start_index_M and inputs must be on the same device.");
    TORCH_CHECK(
        zero_start_index_M.value().dtype() == at::kLong,
        "zero_start_index_M must be int64.");
  }
  if (M_sizes.has_value()) {
    group_count = M_sizes.value().size(0);
    TORCH_CHECK(
        XQ.device() == M_sizes.value().device(),
        "M_sizes and inputs must be on the same device.");
    TORCH_CHECK(M_sizes.value().dtype() == at::kLong, "M_sizes must be int64.");
  }

  // When m_sizes is used XQ is shape [tota_M, K]. When zero_start_index_M is
  // used it is shape [G, M, K].
  int64_t M = XQ.size(XQ.dim() - 2);
  int64_t K = WQ.size(2);
  int64_t N = WQ.size(1);

  // Depending on the mode, use appropriate setup kernel.
  if (M_sizes.has_value()) {
    set_kernel_args<<<1, group_count, 0, stream>>>(
        reinterpret_cast<KernelArguments*>(kernel_args.data_ptr()),
        reinterpret_cast<ADataType*>(XQ.data_ptr()),
        reinterpret_cast<BDataType*>(WQ.data_ptr()),
        reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
        reinterpret_cast<D1DataType*>(x_scale.data_ptr()),
        reinterpret_cast<EDataType*>(output.data_ptr()),
        reinterpret_cast<int64_t*>(M_sizes.value().data_ptr()),
        nullptr,
        M,
        N,
        K,
        group_count);
  } else if (zeroing_output_tensor) {
    // Launch a kernel that sets kernel argument memory and zeros the output.
    // Each thread sets one float4 which corresponds to 8 bf16 values.
    const int64_t BLOCK_SIZE = 8;
    TORCH_CHECK(
        N % BLOCK_SIZE == 0, "N must be divisible 8 for dynamic grouped gemm.");
    int block_factor =
        std::max(group_count, (group_count * M * N) / BLOCK_SIZE);
    int blockSize = std::min(512, block_factor);
    int numBlocks = (block_factor + blockSize - 1) / blockSize;
    set_kernel_args_fixed_nk_kernel_zeroing<<<
        numBlocks,
        blockSize,
        0,
        stream>>>(
        reinterpret_cast<KernelArguments*>(kernel_args.data_ptr()),
        reinterpret_cast<ADataType*>(XQ.data_ptr()),
        reinterpret_cast<BDataType*>(WQ.data_ptr()),
        reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
        reinterpret_cast<D1DataType*>(x_scale.data_ptr()),
        reinterpret_cast<EDataType*>(output.data_ptr()),
        reinterpret_cast<int64_t*>(zero_start_index_M.value().data_ptr()),
        M,
        N,
        K,
        group_count);
  } else {
    set_kernel_args_fixed_nk_kernel<<<1, group_count, 0, stream>>>(
        reinterpret_cast<KernelArguments*>(kernel_args.data_ptr()),
        reinterpret_cast<ADataType*>(XQ.data_ptr()),
        reinterpret_cast<BDataType*>(WQ.data_ptr()),
        reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
        reinterpret_cast<D1DataType*>(x_scale.data_ptr()),
        reinterpret_cast<EDataType*>(output.data_ptr()),
        reinterpret_cast<int64_t*>(zero_start_index_M.value().data_ptr()),
        M,
        N,
        K,
        group_count);
  }
}

static std::pair<at::ScalarType, std::string> get_float8_e4m3_dtype() {
  if (at::detail::getCUDAHooks().isGPUArch({"gfx942"})) {
    return std::make_pair(at::kFloat8_e4m3fnuz, "float8_e4m3fnuz");
  } else if (at::detail::getCUDAHooks().isGPUArch({"gfx950"})) {
    return std::make_pair(at::kFloat8_e4m3fn, "float8_e4m3fn");
  } else {
    std::string gcn_arch_name = at::cuda::getCurrentDeviceProperties()->gcnArchName;
    TORCH_CHECK(false, "Unsupported GPU architecture for FP8: ", gcn_arch_name);
  }
}

template <typename OutputType>
OutputType _f8f8bf16_rowwise_grouped(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale) {
  // Check that input datatypes are valid.
  // First confirm that there are the same number of groups in all inputs.
  static auto float8_dtype = get_float8_e4m3_dtype();
  TORCH_CHECK(
      XQ.size() == WQ.size() && XQ.size() == x_scale.size() &&
          XQ.size() == w_scale.size(),
      "All inputs must have the same number of groups.");
  int group_count = XQ.size();
  // Iterate over inputs and check they are valid.
  for (at::Tensor x : XQ) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(
        x.dtype() == float8_dtype.first,
        "Inputs must be type ", float8_dtype.second);
  }
  for (at::Tensor w : WQ) {
    TORCH_CHECK(w.is_cuda() && w.is_contiguous());
    TORCH_CHECK(w.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(
        w.dtype() == float8_dtype.first,
        "Inputs must be type ", float8_dtype.second);
    TORCH_CHECK(
        w.size(0) >= 512 && w.size(1) >= 512,
        "N and K must be at least 512 for grouped gemm. For smaller inputs, consider unrolling.");
  }
  for (at::Tensor xs : x_scale) {
    TORCH_CHECK(xs.dtype() == at::kFloat, "Scales must be float32.");
  }
  for (at::Tensor ws : w_scale) {
    TORCH_CHECK(ws.dtype() == at::kFloat, "Scales must be float32.");
  }

  // Compute the total number of elements in the output.
  int64_t total_output_size = 0;
  int64_t total_M = 0;
  std::vector<int64_t> output_sizes;
  for (int i = 0; i < group_count; i++) {
    int64_t M = XQ[i].size(0);
    int64_t N = WQ[i].size(0);
    total_M += M;
    const int64_t output_size = M * N;
    total_output_size += output_size;
    output_sizes.push_back(output_size);
  }
  at::Tensor Y =
      at::empty({total_output_size}, XQ[0].options().dtype(at::kBFloat16));

  // Prepare kernel arguments by copying them to the proper device location.
  at::Tensor kernel_args = at::empty(
      {static_cast<long>(group_count * sizeof(KernelArguments))},
      XQ[0].options().dtype(at::kByte));
  set_static_kernel_args<at::Tensor>(kernel_args, XQ, WQ, x_scale, w_scale, Y);

  // We use the largest of each shape for heuristics.
  int64_t MaxN = 0;
  int64_t MaxK = 0;
  for (int i = 0; i < group_count; i++) {
    MaxN = max(MaxN, WQ[i].size(0));
    MaxK = max(MaxK, XQ[i].size(1));
  }
  RowwiseGroupedKernel<at::TensorList, at::Tensor> selected_kernel =
      rowwise_grouped_heuristic_dispatch<at::TensorList, at::Tensor>(
          group_count, total_M, MaxN, MaxK);
  at::Tensor g_out = selected_kernel(XQ, WQ, x_scale, w_scale, kernel_args, Y);
  // Get output in appropriate format.
  if constexpr (std::is_same_v<OutputType, at::Tensor>) {
    int64_t N = WQ[0].size(0);
    return g_out.view({total_M, N});
  } else {
    std::vector<at::Tensor> output_groups = g_out.split(output_sizes);
    for (int i = 0; i < group_count; i++) {
      output_groups[i] = output_groups[i].view({XQ[i].size(0), WQ[i].size(0)});
    }
    return output_groups;
  }
}

void validate_inputs_common(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale
) {
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(x_scale.is_cuda());
  TORCH_CHECK(w_scale.is_cuda());

  static auto float8_dtype = get_float8_e4m3_dtype();
  TORCH_CHECK(
      XQ.dtype() == float8_dtype.first,
      "Input XQ must be type ", float8_dtype.second);
  TORCH_CHECK(
      WQ.dtype() == float8_dtype.first,
      "Input WQ must be type ", float8_dtype.second);

  TORCH_CHECK(x_scale.dtype() == at::kFloat, "Scales must be float32.");
  TORCH_CHECK(w_scale.dtype() == at::kFloat, "Scales must be float32.");
}

} // namespace


std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale) {
  return _f8f8bf16_rowwise_grouped<std::vector<at::Tensor>>(
      XQ, WQ, x_scale, w_scale);
}

at::Tensor f8f8bf16_rowwise_grouped_cat(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale) {
  return _f8f8bf16_rowwise_grouped<at::Tensor>(XQ, WQ, x_scale, w_scale);
}

// Wrapper function for list input single tensor output.
at::Tensor f8f8bf16_rowwise_grouped_stacked(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes) {
  // Check that input datatypes are valid.
  // First confirm that there are the same number of groups in all inputs.
  int64_t group_count = M_sizes.size(0);
  // XQ is expected to be shape [total_M, K].
  int64_t total_M = XQ.size(0);
  // WQ is expected to be shape [G, N, K].
  int64_t N = WQ.size(1);
  int64_t K = XQ.size(1);

  validate_inputs_common(XQ, WQ, x_scale, w_scale);

  TORCH_CHECK(
      WQ.size(0) == group_count && x_scale.numel() == total_M &&
          w_scale.numel() / group_count == N,
      "All inputs must have the same number of groups.");
  // Iterate over inputs and check they are valid.
  TORCH_CHECK(XQ.dim() == 2, "Input XQ must be 2D (total_M,K).");
  TORCH_CHECK(WQ.dim() == 3, "Input WQ must be 3D (G,N,K).");
  TORCH_CHECK(
      WQ.size(1) >= 512 && WQ.size(2) >= 512,
      "N and K must be at least 512 for grouped gemm. For smaller inputs, consider unrolling.");

  // Allocate an empty output array. We will set its values to zero as part
  // of kernel setup.
  at::Tensor Y = at::empty({total_M, N}, XQ.options().dtype(at::kBFloat16));

  // Early exit for empty input.
  if (total_M == 0) {
    return Y;
  }

  // Prepare kernel arguments by copying them to the proper device location.
  at::Tensor kernel_args = at::empty(
      {static_cast<long>(group_count * sizeof(KernelArguments))},
      XQ.options().dtype(at::kByte));
  set_dynamic_kernel_args(
      kernel_args, XQ, WQ, x_scale, w_scale, Y, std::nullopt, M_sizes, false);

  RowwiseGroupedKernel<at::Tensor, at::Tensor> selected_kernel =
      rowwise_grouped_heuristic_dispatch<at::Tensor, at::Tensor>(
          group_count, total_M, N, K);
  return selected_kernel(XQ, WQ, x_scale, w_scale, kernel_args, Y);
}

at::Tensor f8f8bf16_rowwise_grouped_dynamic(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor zero_start_index_M,
    bool zeroing_output_tensor = true) {
  // Check that input datatypes are valid.
  // First confirm that there are the same number of groups in all inputs.
  int64_t group_count = XQ.size(0);
  int64_t M = XQ.size(1);
  int64_t N = WQ.size(1);
  int64_t K = WQ.size(2);
  TORCH_CHECK(
      WQ.size(0) == group_count && x_scale.numel() / group_count == M &&
          w_scale.numel() / group_count == N,
      "All inputs must have the same number of groups.");

  validate_inputs_common(XQ, WQ, x_scale, w_scale);
  // Iterate over inputs and check they are valid.
  TORCH_CHECK(XQ.dim() == 3, "Input XQ must be 3D (G,M,K).");
  TORCH_CHECK(WQ.dim() == 3, "Input WQ must be 3D (G,N,K).");
  TORCH_CHECK(
      WQ.size(1) >= 512 && WQ.size(2) >= 512,
      "N and K must be at least 512 for grouped gemm. For smaller inputs, consider unrolling.");

  // Allocate an empty output array. We will set its values to zero as part
  // of kernel setup.
  at::Tensor Y =
      at::empty({group_count, M, N}, XQ.options().dtype(at::kBFloat16));

  // Early exit for empty input.
  if (Y.numel() == 0) {
    return Y;
  }

  // Prepare kernel arguments by copying them to the proper device location.
  at::Tensor kernel_args = at::empty(
      {static_cast<long>(group_count * sizeof(KernelArguments))},
      XQ.options().dtype(at::kByte));
  set_dynamic_kernel_args(
      kernel_args,
      XQ,
      WQ,
      x_scale,
      w_scale,
      Y,
      zero_start_index_M,
      std::nullopt,
      zeroing_output_tensor);

  RowwiseGroupedKernel<at::Tensor, at::Tensor> selected_kernel =
      rowwise_grouped_heuristic_dispatch<at::Tensor, at::Tensor>(group_count, M, N, K);
  return selected_kernel(XQ, WQ, x_scale, w_scale, kernel_args, Y);
}

/**
 * PyTorch compliant grouped GEMM API.
 * Supports 2D-2D (K dynamic), 2D-3D (M dynamic), 3D-2D (N dynamic), and 3D-3D (BMM).
 */
at::Tensor f8f8bf16_rowwise_grouped_mm(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale, // FP32
    at::Tensor w_scale, // FP32
    std::optional<at::Tensor> offsets, // int32
    at::Tensor& out) {
  validate_inputs_common(XQ, WQ, x_scale, w_scale);

  // M, N, K could be the "total" dimension in the case of 2D inputs.
  int64_t G;
  int64_t M;
  int64_t N;
  int64_t K;
  std::optional<GroupedGemmInputType> inputType;

  if (XQ.dim() == 2 && WQ.dim() == 3) {
    TORCH_CHECK(offsets.has_value(), "Must pass offsets for 2D input XQ.");
    TORCH_CHECK(offsets->dtype() == at::kInt, "offsets must be int32.");

    G = offsets->size(0);
    M = XQ.size(0);
    N = WQ.size(1);
    K = WQ.size(2);
    inputType = GroupedGemmInputType::_2D3D;

    TORCH_CHECK(XQ.size(1) == K && WQ.size(0) == G, "XQ shape must be (total_M, K) and WQ shape must be (G, N, K).");
    TORCH_CHECK(x_scale.size(0) == M, "x_scale shape must be (total_M).");
    TORCH_CHECK(w_scale.size(0) == G && w_scale.size(1) == N, "w_scale shape must be (G, N).");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == N, "out shape must be (total_M, N).");
  } else if (XQ.dim() == 3 && WQ.dim() == 2) {
    TORCH_CHECK(offsets.has_value(), "Must pass offsets for 2D input WQ.");
    TORCH_CHECK(offsets->dtype() == at::kInt, "offsets must be int32.");

    G = offsets->size(0);
    M = XQ.size(1);
    N = WQ.size(0);
    K = WQ.size(1);
    inputType = GroupedGemmInputType::_3D2D;

    TORCH_CHECK(XQ.size(0) == G && XQ.size(2) == K, "XQ shape must be (G, M, K) and WQ shape must be (total_N, K).");
    TORCH_CHECK(x_scale.size(0) == G && x_scale.size(1) == M, "x_scale shape must be (G, M).");
    TORCH_CHECK(w_scale.size(0) == N, "w_scale shape must be (total_N).");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == N, "out shape must be (M, total_N).");
  } else if (XQ.dim() == 3 && WQ.dim() == 3) {
    TORCH_CHECK(!offsets.has_value(), "Offsets should not be passed for 3D-3D inputs.");

    G = XQ.size(0);
    M = XQ.size(1);
    N = WQ.size(1);
    K = XQ.size(2);
    inputType = GroupedGemmInputType::_3D3D;

    TORCH_CHECK(WQ.size(0) == G && WQ.size(2) == K, "XQ shape must be (G, M, K) and WQ shape must be (G, N, K).");
    TORCH_CHECK(x_scale.size(0) == G && x_scale.size(1) == M, "x_scale shape must be (G, M).");
    TORCH_CHECK(w_scale.size(0) == G && w_scale.size(1) == N, "w_scale shape must be (G, N).");
    TORCH_CHECK(out.dim() == 3 && out.size(0) == G && out.size(1) == M && out.size(2) == N, "out shape must be (G, M, N).");
  } else if (XQ.dim() == 2 && WQ.dim() == 2) {
    TORCH_CHECK(offsets.has_value(), "Must pass offsets for 2D inputs XQ and WQ.");
    TORCH_CHECK(offsets->dtype() == at::kInt, "offsets must be int32.");

    G = offsets->size(0);
    M = XQ.size(0);
    N = WQ.size(0);
    K = XQ.size(1);
    inputType = GroupedGemmInputType::_2D2D;

    TORCH_CHECK(XQ.dim() == 2 && WQ.dim() == 2 && WQ.size(1) == K, "XQ shape must be (M, total_K) and WQ shape must be (N, total_K).");
    TORCH_CHECK(x_scale.size(0) == G * M, "x_scale shape must be (G * M).");
    TORCH_CHECK(w_scale.size(0) == G * N, "w_scale shape must be (G * N).");
    TORCH_CHECK(out.dim() == 3 && out.size(0) == G && out.size(1) == M && out.size(2) == N, "out shape must be (G, M, N).");
  } else {
    TORCH_CHECK(false, "Invalid input shapes. Must be one of 2D-2D, 3D-3D, 2D-3D, 3D-2D.");
  }

  // Enforce CK CShuffleBlockTransferScalarPerVector_NPerBlock divisibility
  // Technically, 8 is more strict than required, but otherwise this validation becomes more complex.
  if (inputType != GroupedGemmInputType::_3D2D) {
    TORCH_CHECK(N % 8 == 0, "N must be divisible by 8.");
  }

  // Early exit for empty input.
  if (out.numel() == 0) {
    return out;
  }

  at::Tensor kernel_args = at::empty(
      {static_cast<long>(G * sizeof(KernelArguments))},
      XQ.options().dtype(at::kByte));

  auto stream = at::cuda::getCurrentHIPStream().stream();
  set_kernel_args<<<1, G, 0, stream>>>(
      reinterpret_cast<KernelArguments*>(kernel_args.data_ptr()),
      reinterpret_cast<ADataType*>(XQ.data_ptr()),
      reinterpret_cast<BDataType*>(WQ.data_ptr()),
      reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
      reinterpret_cast<D1DataType*>(x_scale.data_ptr()),
      reinterpret_cast<EDataType*>(out.data_ptr()),
      nullptr,
      offsets.has_value() ? reinterpret_cast<int32_t*>(offsets.value().data_ptr()) : nullptr,
      M,
      N,
      K,
      G,
      inputType);

  // For heuristics normalize the dynamic dimension by G so we can approximate the problem shape per group.
  if (inputType == GroupedGemmInputType::_2D3D ) {
    M /= G;
  } else if (inputType == GroupedGemmInputType::_3D2D) {
    N /= G;
  } else if (inputType == GroupedGemmInputType::_2D2D) {
    K /= G;
  }

  // Select kernel to run via heuristics or tuning.
  auto kernel = [&]() {
    if (std::getenv("FBGEMM_AUTOTUNE_ENABLE")) {
      return get_kernel_via_tuning(M, N, K, XQ, WQ, x_scale, w_scale, kernel_args, out);
    } else {
      return get_kernel_via_heuristic<at::Tensor, at::Tensor>(M, N, K);
    }
  }();
  // Invoke kernel
  return kernel(XQ, WQ, x_scale, w_scale, kernel_args, out);
}

} // namespace fbgemm_gpu

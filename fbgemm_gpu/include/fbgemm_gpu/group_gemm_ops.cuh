/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "group_gemm_ops_gpu.h"

namespace fbgemm_gpu {

namespace {
template <typename T, typename ArchTag>
struct cutlass_traits;

template <typename ArchTag>
struct cutlass_traits<float, ArchTag> {
  using Element = float;
  using ElementAcc = float;
  using OpClass = cutlass::arch::OpClassSimt;

 private:
  using GemmConfiguration_ =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OpClass,
          ArchTag,
          Element,
          Element,
          Element,
          ElementAcc>;

 public:
  using ThreadblockShape = typename GemmConfiguration_::ThreadblockShape;
  using WarpShape = typename GemmConfiguration_::WarpShape;
  using InstructionShape = typename GemmConfiguration_::InstructionShape;
};

template <typename ArchTag>
struct cutlass_traits<double, ArchTag> {
  using Element = double;
  using ElementAcc = double;
  using OpClass = cutlass::arch::OpClassSimt;

 private:
  using GemmConfiguration_ =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OpClass,
          ArchTag,
          Element,
          Element,
          Element,
          ElementAcc>;

 public:
  using ThreadblockShape = typename GemmConfiguration_::ThreadblockShape;
  using WarpShape = typename GemmConfiguration_::WarpShape;
  using InstructionShape = typename GemmConfiguration_::InstructionShape;
};

template <typename ArchTag>
struct cutlass_traits<at::Half, ArchTag> {
  using Element = cutlass::half_t;
  using ElementAcc = float;
  using OpClass = cutlass::arch::OpClassWmmaTensorOp;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 16, 16>;
};

// TODO: add bfloat16

// From
// https://github.com/NVIDIA/cutlass/blob/master/examples/24_gemm_grouped/gemm_grouped.cu
/// Returns the number of threadblocks to launch if the kernel can run on the
/// target device. Otherwise, returns zero.
template <typename Gemm>
int sufficient() {
  //
  // Determine SMEM requirements and waive if not satisfied
  //

  const int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));

  const cudaDeviceProp* deviceProp =
      at::cuda::getDeviceProperties(c10::cuda::current_device());

  const int occupancy =
      std::min(2, int(deviceProp->sharedMemPerMultiprocessor / smem_size));

  return deviceProp->multiProcessorCount * occupancy;
}

} // namespace

template <typename scalar_t, typename LayoutB, typename ArchTag>
std::vector<at::Tensor> gemm_grouped_cuda(
    const std::vector<at::Tensor>& a_group,
    const std::vector<at::Tensor>& b_group,
    const c10::optional<std::vector<at::Tensor>>& c_group) {
  const int problem_count = a_group.size();
  TORCH_CHECK(b_group.size() == problem_count)

  using cutlass_traits_ = cutlass_traits<scalar_t, ArchTag>;
  using Element = typename cutlass_traits_::Element;
  using ElementAcc = typename cutlass_traits_::ElementAcc;
  using OpClass = typename cutlass_traits_::OpClass;

  using GemmConfiguration =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OpClass,
          ArchTag,
          Element,
          Element,
          Element,
          ElementAcc>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element,
      cutlass::layout::RowMajor,
      cutlass::ComplexTransform::kNone,
      GemmConfiguration::kAlignmentA,
      Element,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      GemmConfiguration::kAlignmentB,
      Element,
      cutlass::layout::RowMajor,
      ElementAcc,
      OpClass,
      ArchTag,
      typename cutlass_traits_::ThreadblockShape,
      typename cutlass_traits_::WarpShape,
      typename cutlass_traits_::InstructionShape,
      typename GemmConfiguration::EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      GemmConfiguration::kStages>::GemmKernel;

  using GemmGrouped = typename cutlass::gemm::device::GemmGrouped<GemmKernel>;

  const int threadblock_count = sufficient<GemmGrouped>();

  using EpilogueOutputOp = typename GemmGrouped::GemmKernel::Epilogue::OutputOp;
  const ElementAcc alpha = 1;
  // beta = 1 if there is c_group is passed, otherwise beta = 0
  // Note: gmm performs alpha * a_group * b_group + beta * c_group
  const ElementAcc beta = c_group.has_value();
  typename EpilogueOutputOp::Params epilogue_op(alpha, beta);

  auto device = a_group[0].device();

  // Compute the total output tensor size for all GEMMs. We allocate a single
  // output tensor for all GEMMs to avoid memory fragmentation. We later split
  // the output tensor into a vector of tensors before returning it.
  int64_t total_output_size = 0;
  std::vector<int64_t> output_sizes;
  output_sizes.reserve(problem_count);
  for (int i = 0; i < problem_count; ++i) {
    int64_t output_size = a_group[i].size(0) * b_group[i].size(1);
    total_output_size += output_size;
    output_sizes.push_back(output_size);
  }

  at::Tensor output_tensor =
      at::empty({total_output_size}, a_group[0].options());
  scalar_t* output_data = output_tensor.data_ptr<scalar_t>();

  // Allocate a tensor for GemmGrouped::Arguments's arguments
  // (https://github.com/NVIDIA/cutlass/blob/v2.9.0/include/cutlass/gemm/kernel/gemm_grouped.h#L323).
  // This tensor contains up to 8 arguments: *lda, *ldc, *ldd, *ptr_A, *ptr_B,
  // *ptr_C, *ptr_D and *problem_sizes. (*ldc and *ptr_C are excluded if
  // c_group is not passed.) We allocate a single tensor for all arguments
  // because this tensor has to be transferred to GPU before passing to
  // GemmGrouped::Arguments. Having a single tensor coalesces data transfer
  // into one.
  const int64_t gemm_coord_size =
      problem_count * ((int64_t)sizeof(cutlass::gemm::GemmCoord));
  // Number of gmm args not including *problem_sizes
  const int num_gmm_args = c_group.has_value() ? 7 : 5;
  at::Tensor gmm_args = at::empty(
      {problem_count * num_gmm_args + gemm_coord_size},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));

  // Obtain pointers for each argument (on host)
  int64_t* lda_data = gmm_args.data_ptr<int64_t>(); // Base pointer
  int64_t* ldd_data = lda_data + problem_count;
  int64_t* ptr_a_data = lda_data + 2 * problem_count;
  int64_t* ptr_b_data = lda_data + 3 * problem_count;
  int64_t* ptr_d_data = lda_data + 4 * problem_count;
  uint8_t* problem_sizes_buf =
      reinterpret_cast<uint8_t*>(lda_data + 5 * problem_count);
  cutlass::gemm::GemmCoord* problem_sizes_data =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(problem_sizes_buf);
  // ldc and ptr_c are not used if c_group is not passed
  int64_t* ldc_data = c_group.has_value()
      ? lda_data + 5 * problem_count + gemm_coord_size
      : nullptr;
  int64_t* ptr_c_data = c_group.has_value()
      ? lda_data + 6 * problem_count + gemm_coord_size
      : nullptr;

  // Set arguments
  int64_t output_offset = 0;
  for (int i = 0; i < problem_count; ++i) {
    const int m = a_group[i].size(0);
    const int n = b_group[i].size(1);
    const int k = a_group[i].size(1);
    TORCH_CHECK(b_group[i].size(0) == k);
    problem_sizes_data[i] = cutlass::gemm::GemmCoord(m, n, k);

    lda_data[i] = k;
    ldd_data[i] = n;
    ptr_a_data[i] = reinterpret_cast<int64_t>(a_group[i].data_ptr<scalar_t>());
    ptr_b_data[i] = reinterpret_cast<int64_t>(b_group[i].data_ptr<scalar_t>());
    ptr_d_data[i] = reinterpret_cast<int64_t>(output_data + output_offset);

    // Compute ptr_c and ldc if c_group exists
    if (c_group.has_value()) {
      auto& c_group_ = c_group.value()[i];
      ptr_c_data[i] = reinterpret_cast<int64_t>(c_group_.data_ptr<scalar_t>());
      if (c_group_.dim() == 1) {
        TORCH_CHECK(c_group_.numel() == n)
        // Set Tensor C's stride to zero so that it is treated as a vector
        ldc_data[i] = 0;
      } else {
        TORCH_CHECK(
            c_group_.dim() == 2, "Tensors in c_group must be either 1D or 2D")
        TORCH_CHECK(c_group_.size(0) == m || c_group_.size(1) == n)
        ldc_data[i] = n;
      }
    }

    output_offset += output_sizes[i];
  }

  // Transfer arguments to GPU
  gmm_args = gmm_args.to(device, /*non_blocking=*/true);

  // Obtain pointers for each of arguments (on GPU)
  lda_data = gmm_args.data_ptr<int64_t>(); // Base pointer
  ldd_data = lda_data + problem_count;
  ptr_a_data = lda_data + 2 * problem_count;
  ptr_b_data = lda_data + 3 * problem_count;
  ptr_d_data = lda_data + 4 * problem_count;
  problem_sizes_buf = reinterpret_cast<uint8_t*>(lda_data + 5 * problem_count);
  problem_sizes_data =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(problem_sizes_buf);
  // ldc and ptr_c are not used if c_group is not passed, but we still need to
  // pass valid device pointers to CUTLASS. So, we set ldc to ldd and ptr_c to
  // ptr_d.
  ldc_data = c_group.has_value()
      ? lda_data + 5 * problem_count + gemm_coord_size
      : ldd_data;
  ptr_c_data = c_group.has_value()
      ? lda_data + 6 * problem_count + gemm_coord_size
      : ptr_d_data;

  // Create GemmGrouped::Arguments using the arguments prepared above
  typename GemmGrouped::Arguments args(
      problem_sizes_data,
      problem_count,
      threadblock_count,
      epilogue_op,
      reinterpret_cast<Element**>(ptr_a_data),
      reinterpret_cast<Element**>(ptr_b_data),
      reinterpret_cast<Element**>(ptr_c_data),
      reinterpret_cast<Element**>(ptr_d_data),
      lda_data,
      (std::is_same<LayoutB, cutlass::layout::RowMajor>::value ? ldd_data
                                                               : lda_data),
      ldc_data,
      ldd_data);

  GemmGrouped gemm;
  cutlass::Status status = gemm.initialize(
      args, /*workspace=*/nullptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status != cutlass::Status::kErrorWorkspaceNull,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to workspace.");
  TORCH_CHECK(
      status != cutlass::Status::kErrorInternal,
      "Failed to initialize CUTLASS Grouped GEMM kernel due to internal error.");
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to initialize CUTLASS Grouped GEMM kernel.");

  // Run CUTLASS group GEMM
  status = gemm.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Failed to run CUTLASS Grouped GEMM kernel.");

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Perform output split after kernel is launched to avoid blocking kernel
  // launch
  std::vector<at::Tensor> output_group = output_tensor.split(output_sizes);
  for (int i = 0; i < problem_count; ++i) {
    output_group[i] =
        output_group[i].view({a_group[i].size(0), b_group[i].size(1)});
  }
  return output_group;
}

} // namespace fbgemm_gpu

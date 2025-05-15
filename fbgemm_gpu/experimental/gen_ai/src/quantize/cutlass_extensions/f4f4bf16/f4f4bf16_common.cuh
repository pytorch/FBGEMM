/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
  // The fixed ordering of the headers is required for CUTLASS 3.2+
  #include <cute/tensor.hpp>
  #include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
  #include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
  #include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

template <
    typename InputType,
    int TB_M,
    int TB_N,
    int TBS_M,
    int TBS_N,
    int TBS_K>
at::Tensor _f4f4bf16(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> global_scale) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1) * 2; // Since K is packed

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(x_scale.is_cuda() && x_scale.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  constexpr int TileShapeK = 128 * 8 / cutlass::sizeof_bits<InputType>::value;

  using ElementA = InputType;
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutATag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutATag>::type;
  constexpr int AlignmentA = 32;

  using ElementB = InputType;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutBTag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutBTag>::type;
  constexpr int AlignmentB = 32;

  // TODO: Verify if bfloat16 is enough
  using ElementScale = float;
  using ElementCompute = float;
  using ElementAccumulator = float;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutputTag = cutlass::layout::RowMajor;
  using LayoutOutputTag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutOutputTag>::type;
  constexpr int AlignmentOutput =
      128 /
      cutlass::sizeof_bits<
          ElementOutput>::value; // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

  using ArchTag = cutlass::arch::Sm100; // Tag indicating the minimum SM that
                                        // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using MmaTileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TileShapeK>>; // Threadblock-level MMA
                              // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          MmaTileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          ElementOutput,
          LayoutOutputTag_Transpose,
          AlignmentOutput,
          ElementOutput,
          LayoutOutputTag_Transpose,
          AlignmentOutput,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementB,
          LayoutBTag_Transpose,
          AlignmentB,
          ElementA,
          LayoutATag_Transpose,
          AlignmentA,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto // Kernel schedule
                                                        // policy. Auto or using
                                                        // targeted scheduling
                                                        // policy
          >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA =
      decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::
      LayoutSFA; // Scale Factor tensors have an interleaved layout. Bring
                 // Layout instead of stride.
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB =
      decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::
      LayoutSFB; // Scale Factor tensors have an interleaved layout. Bring
                 // Layout instead of stride.
  using StrideOutput = typename Gemm::GemmKernel::StrideC;
  using LayoutOutput =
      decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideOutput{}));

  // For SFA and SFB tensors layouts
  using Sm100BlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm100BlkScaledConfig;
  // For SFD tensor layout
  using Sm100BlockScaledOutputConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm100BlkScaledConfig;

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(N, M, 1));

  LayoutA layout_A = make_layout(cute::make_shape(M, K, 1), stride_A);
  LayoutB layout_B = make_layout(cute::make_shape(N, K, 1), stride_B);
  LayoutOutput layout_output =
      make_layout(cute::make_shape(N, M, 1), stride_output);
  LayoutSFA layout_SFA = Sm100BlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  LayoutSFB layout_SFB = Sm100BlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  using DataTypeA = typename ElementA::DataType;
  using DataTypeB = typename ElementB::DataType;
  using SFTypeA = typename ElementA::ScaleFactorType;
  using SFTypeB = typename ElementB::ScaleFactorType;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K, 1},
      {// Mainloop arguments
       reinterpret_cast<DataTypeB*>(WQ.data_ptr()),
       stride_B,
       reinterpret_cast<DataTypeA*>(XQ.data_ptr()),
       stride_A,
       reinterpret_cast<SFTypeB*>(w_scale.data_ptr()),
       layout_SFB,
       reinterpret_cast<SFTypeA*>(x_scale.data_ptr()),
       layout_SFA},
      {// Epilogue arguments
       {1, 0},
       reinterpret_cast<ElementOutput*>(Y.data_ptr()),
       stride_output,
       reinterpret_cast<ElementOutput*>(Y.data_ptr()),
       stride_output}};

  if constexpr (std::is_same_v<
                    InputType,
                    cutlass::nv_float4_t<cutlass::float_e2m1_t>>) {
    TORCH_CHECK(global_scale.has_value(), "global_scale is required in nvfp4.");
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr =
        static_cast<ElementCompute const*>(global_scale.value().data_ptr());
  }

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm(at::cuda::getCurrentCUDAStream());

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return Y;
}

#endif

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/Atomic.cuh>
#if !(                                                  \
    defined(USE_ROCM) ||                                \
    ((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#elif (defined(USE_ROCM))
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#endif
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include <cutlass/util/host_tensor.h>
#include "cublas_utils.h"

#if CUDART_VERSION >= 12000
#include <cuda_fp8.h>
#endif

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
#include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "fp8_blockwise_cutlass_helpers.h"

// Each block handles a single batch and head

// Each warp handles separate D dimension.

// Load Q into registers in all warps.
// Split T across warps in a block
// Compute S[MAX_T] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
// Use shared reduction to compute max and compute softmax on shared memory.

// Split T across warps in a block

// each warp compute sum(t_subset) P[t] * V[t_subset, d]
// outputs are of size float[D]

namespace cutlass::epilogue::threadblock::detail {

/// Partial specialization for bfloat16 <= int32_t x 4
template <
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename ThreadMap>
struct DefaultIteratorsTensorOp<
    cutlass::bfloat16_t,
    int32_t,
    8,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    ThreadMap> {
  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
      WarpShape,
      InstructionShape,
      int32_t,
      layout::RowMajor>;

  using SharedLoadIterator =
      cutlass::epilogue::threadblock::SharedLoadIterator<ThreadMap, int32_t>;

  static int const kFragmentsPerIteration = 1;
};

} // namespace cutlass::epilogue::threadblock::detail

// Wrapper to allow passing alpha/beta scaling params
// as device pointers.
namespace cutlass::epilogue::thread {

template <
    typename ElementOutput_, ///< Data type used to load and store tensors
    int Count, ///< Number of elements computed per operation.
               ///< Usually it is 128/sizeof_bits<ElementOutput_>,
               ///< but we use 64 or 32 sometimes when there are not enough data
               ///< to store
    typename ElementAccumulator_ = ElementOutput_, ///< Accumulator data type
    typename ElementCompute_ =
        ElementOutput_, ///< Data type used to compute linear combination
    ScaleType::Kind Scale =
        ScaleType::Default, ///< Control Alpha and Beta scaling
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class LinearCombinationOnDevice {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  static const ScaleType::Kind kScale = Scale;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  using ParamsBase = LinearCombinationParams;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params : ParamsBase {
    ElementCompute alpha; ///< scales accumulators
    ElementCompute beta; ///< scales source tensor
    ElementCompute const* alpha_ptr; ///< pointer to accumulator scalar - if not
                                     ///< null, loads it from memory
    ElementCompute const* beta_ptr; ///< pointer to source scalar - if not null,
                                    ///< loads it from memory

    CUTLASS_HOST_DEVICE
    Params()
        : ParamsBase(ElementCompute(1), ElementCompute(0)),
          alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta)
        : ParamsBase(alpha, beta),
          alpha(alpha),
          beta(beta),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha)
        : ParamsBase(alpha, ElementCompute(0)),
          alpha(alpha),
          beta(0),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr, ElementCompute const* beta_ptr)
        : ParamsBase(*alpha_ptr, *beta_ptr),
          alpha(0),
          beta(0),
          alpha_ptr(alpha_ptr),
          beta_ptr(beta_ptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr)
        : ParamsBase(ElementCompute(1), ElementCompute(0)),
          alpha(0),
          beta(0),
          alpha_ptr(alpha_ptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ParamsBase const& base)
        : ParamsBase(base), alpha_ptr(nullptr), beta_ptr(nullptr) {
#if defined(__CUDA_ARCH__)
      alpha = reinterpret_cast<ElementCompute const&>(base.alpha_data);
      beta = reinterpret_cast<ElementCompute const&>(base.beta_data);
#else
      memcpy(alpha, base.alpha_data, sizeof(ElementCompute));
      memcpy(beta, base.alpha_data, sizeof(ElementCompute));
#endif
    }
  };

 private:
  //
  // Data members
  //

  const ElementCompute* alpha_ptr_;
  ElementCompute beta_;

 public:
  /// Constructs the function object, possibly loading from pointers in host
  /// memory
  CUTLASS_HOST_DEVICE
  LinearCombinationOnDevice(Params const& params) {
    alpha_ptr_ = params.alpha_ptr;
    beta_ = ElementCompute(0);
    // beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling)
      return true;

    if (Scale == ScaleType::OnlyAlphaScaling)
      return false;

    if (Scale == ScaleType::Nothing)
      return false;

    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& accumulator,
      FragmentOutput const& source) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
        source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    if (Scale == ScaleType::Nothing)
      return destination_converter(converted_accumulator);

    // Perform binary operations
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;

    if (Scale == ScaleType::NoBetaScaling)
      intermediate = converted_source;
    else
      intermediate =
          mul_add_source(beta_, converted_source); // X =  beta * C + uniform

    intermediate = mul_add_accumulator(
        *alpha_ptr_,
        converted_accumulator,
        intermediate); // D = alpha * Accum + X

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    if (Scale == ScaleType::Nothing)
      return destination_converter(converted_accumulator);

    // Perform binary operations
    ComputeFragment intermediate;
    multiplies<ComputeFragment> mul_accumulator;

    intermediate = mul_accumulator(
        *alpha_ptr_, converted_accumulator); // D = alpha * Accum

    return destination_converter(intermediate);
  }
};

} // namespace cutlass::epilogue::thread

namespace {

int64_t ceil_div(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

} // namespace

namespace fbgemm_gpu {

template <int TB_M, int TB_N, int TB_K, int W_M, int W_N, int W_K>
at::Tensor i8i8bf16_impl(
    at::Tensor XQ, // INT8
    at::Tensor WQ, // INT8
    double scale,
    int64_t split_k) {
  auto M = XQ.size(0);
  auto N = WQ.size(0);
  auto K = XQ.size(1);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,
      cutlass::layout::RowMajor,
      int8_t,
      cutlass::layout::ColumnMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>, // ThreadBlockShape
      cutlass::gemm::GemmShape<W_M, W_N, W_K>, // WarpShape
      cutlass::gemm::GemmShape<16, 8, 32>, // InstructionShape
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput,
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      3,
      16,
      16,
      true>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      XQ.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      WQ.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      (ElementOutput*)Y.data_ptr<at::BFloat16>(),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size,
      input_ref,
      weight_ref,
      out_ref,
      out_ref,
      {float(scale), 0.0},
      int(split_k)};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  auto workspace =
      at::empty({int64_t(workspace_size)}, Y.options().dtype(at::kChar));

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(
      arguments, workspace.data_ptr(), at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return Y;
}

template <int TB_M, int TB_N, int TB_K, int TBS_M, int TBS_N, int TBS_K>
at::Tensor i8i8bf16sm90a_impl(
    at::Tensor XQ, // INT8
    at::Tensor WQ, // INT8
    double scale) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ElementInputA = int8_t;
  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA =
      128 /
      cutlass::sizeof_bits<
          ElementInputA>::value; // Memory access granularity/alignment of A
                                 // matrix in units of elements (up to 16 bytes)

  using ElementInputB = int8_t;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB =
      128 /
      cutlass::sizeof_bits<
          ElementInputB>::value; // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  constexpr int AlignmentOutput =
      128 /
      cutlass::sizeof_bits<
          ElementOutput>::value; // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TB_K>>; // Threadblock-level
                        // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto; // Stage count maximized based
                                                 // on the tile size
  using KernelSchedule = cutlass::gemm::collective::
      KernelScheduleAuto; // Kernel to launch based on the default setting in
                          // the Collective Builder

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementInputA,
          LayoutInputA,
          AlignmentInputA,
          ElementInputB,
          LayoutInputB,
          AlignmentInputB,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAuto,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementComputeEpilogue,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, cute::Int<1>{}));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, cute::Int<1>{}));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(N, M, cute::Int<1>{}));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K},
      {WQ.data_ptr<ElementInputB>(),
       stride_b,
       XQ.data_ptr<ElementInputA>(),
       stride_a},
      {{float(scale), 0},
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};
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

#if CUDART_VERSION >= 12000
enum class KernelMode { Small, Large, Default };

KernelMode get_kernel_mode(at::Tensor XQ, at::Tensor WQ) {
  auto M = XQ.size(0);
  auto K = XQ.size(1);
  auto N = WQ.size(0);
  // Use a large kernel if at least two shapes are large....
  bool use_large_kernel =
      ((M >= 2048 && K >= 2048) || (M >= 2048 && N >= 2048) ||
       (K >= 2048 && N >= 2048));
  if (M <= 128 || N <= 128) {
    return KernelMode::Small;
  } else if (use_large_kernel) {
    return KernelMode::Large;
  } else {
    return KernelMode::Default;
  }
}

// Cutlass tensorwise kernel
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool FAST_ACCUM>
at::Tensor f8f8bf16_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor scale) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ElementInputA = cutlass::float_e4m3_t;
  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA =
      128 /
      cutlass::sizeof_bits<
          ElementInputA>::value; // Memory access granularity/alignment of A
                                 // matrix in units of elements (up to 16 bytes)

  using ElementInputB = cutlass::float_e4m3_t;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB =
      128 /
      cutlass::sizeof_bits<
          ElementInputB>::value; // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  constexpr int AlignmentOutput =
      128 /
      cutlass::sizeof_bits<
          ElementOutput>::value; // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TB_K>>; // Threadblock-level
                        // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto; // Stage count maximized
                                                 // based on the tile size
  using KernelSchedule = cutlass::gemm::collective::
      KernelScheduleAuto; // Kernel to launch based on the default setting in
                          // the Collective Builder

  using MainLoopSchedule = cute::conditional_t<
      FAST_ACCUM,
      cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum,
      cutlass::gemm::KernelTmaWarpSpecialized>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementInputA,
          LayoutInputA,
          AlignmentInputA,
          ElementInputB,
          LayoutInputB,
          AlignmentInputB,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAuto,
          MainLoopSchedule>::CollectiveOp;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementComputeEpilogue,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, cute::Int<1>{}));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, cute::Int<1>{}));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(N, M, cute::Int<1>{}));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K},
      {reinterpret_cast<ElementInputB*>(WQ.data_ptr()),
       stride_b,
       reinterpret_cast<ElementInputA*>(XQ.data_ptr()),
       stride_a},
      {{scale.data_ptr<float>(), 0},
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};
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

at::Tensor f8f8bf16(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor scale,
    bool use_fast_accum) {
  auto M = XQ.size(0);
  // auto K = XQ.size(1);
  // auto N = WQ.size(0);
  if (use_fast_accum) {
    if (M <= 128) {
      return f8f8bf16_impl<64, 128, 128, 2, 1, 1, true>(XQ, WQ, scale);
    } else {
      return f8f8bf16_impl<128, 128, 128, 1, 2, 1, true>(XQ, WQ, scale);
    }
  } else {
    if (M <= 128) {
      return f8f8bf16_impl<64, 128, 128, 2, 1, 1, false>(XQ, WQ, scale);
    } else {
      return f8f8bf16_impl<128, 128, 128, 1, 2, 1, false>(XQ, WQ, scale);
    }
  }
}

template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    bool FAST_ACCUM>
at::Tensor f8f8bf16_tensorwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    double scale) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ElementInputA = cutlass::float_e4m3_t;
  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 /
      sizeof(ElementInputA); // Memory access granularity/alignment of A
                             // matrix in units of elements (up to 16 bytes)

  using ElementInputB = cutlass::float_e4m3_t;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 /
      sizeof(ElementInputB); // Memory access granularity/alignment of B
                             // matrix in units of elements (up to 16 bytes)

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 /
      sizeof(ElementOutput); // Memory access granularity/alignment of C
                             // matrix in units of elements (up to 16 bytes)

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TB_K>>; // Threadblock-level
                        // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto; // Stage count maximized
                                                 // based on the tile size
  using KernelSchedule = cutlass::gemm::collective::
      KernelScheduleAuto; // Kernel to launch based on the default setting in
                          // the Collective Builder

  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using FastDefaultSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using FastPongSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using SlowAccum = cute::conditional_t<PONG, PongSchedule, DefaultSchedule>;
  using FastAccum =
      cute::conditional_t<PONG, FastPongSchedule, FastDefaultSchedule>;
  using MainLoopSchedule =
      cute::conditional_t<FAST_ACCUM, FastAccum, SlowAccum>;

  using Scale_ =
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementComputeEpilogue>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementOutput, // First stage output type.
      ElementComputeEpilogue, // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EpilogueEVT =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, Scale_, Accum>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementComputeEpilogue,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          cutlass::epilogue::TmaWarpSpecialized,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementInputA,
          LayoutInputA,
          AlignmentInputA,
          ElementInputB,
          LayoutInputB,
          AlignmentInputB,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainLoopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, cute::Int<1>{}));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, cute::Int<1>{}));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(M, N, cute::Int<1>{}));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {reinterpret_cast<ElementInputA*>(XQ.data_ptr()),
       stride_a,
       reinterpret_cast<ElementInputB*>(WQ.data_ptr()),
       stride_b},
      {{},
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};

  arguments.epilogue.thread = {
      {float(scale)}, // scale
      {}, // Accumulator
      {}, // Multiplies
  };

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

at::Tensor f8f8bf16_tensorwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    double scale,
    bool use_fast_accum) {
  KernelMode kernel = get_kernel_mode(XQ, WQ);
  if (kernel == KernelMode::Small) {
    return f8f8bf16_tensorwise_impl<64, 128, 128, 2, 1, 1, true, true>(
        XQ, WQ, scale);
  } else if (kernel == KernelMode::Large) {
    return f8f8bf16_tensorwise_impl<128, 128, 128, 2, 1, 1, true, true>(
        XQ, WQ, scale);
  } else {
    return f8f8bf16_tensorwise_impl<128, 128, 128, 1, 2, 1, false, true>(
        XQ, WQ, scale);
  }
}

// Cutlass rowwise kernel
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    bool FAST_ACCUM,
    bool USE_BIAS,
    typename INPUT_DTYPE,
    typename BIAS_DTYPE>
at::Tensor f8f8bf16_rowwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> output) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    // Make sure the provided output has the proper shape and dtype.
    TORCH_CHECK(Y.size(0) == M && Y.size(1) == N);
    TORCH_CHECK(Y.dtype() == at::kBFloat16);
  } else {
    Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));
  }

  using ElementInputA = INPUT_DTYPE;
  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(ElementInputA);

  using ElementInputB = cutlass::float_e4m3_t;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(ElementInputB);

  using ElementBias = BIAS_DTYPE;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 / sizeof(ElementOutput);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TB_K>>; // Threadblock-level
                        // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto; // Stage count maximized
                                                 // based on the tile size
  using KernelSchedule = cutlass::gemm::collective::
      KernelScheduleAuto; // Kernel to launch based on the default setting in
                          // the Collective Builder

  // Implement rowwise scaling epilogue.
  using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      PONG ? 2 : 1,
      TileShape,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      PONG ? 2 : 1,
      TileShape,
      ElementBias,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementComputeEpilogue, // First stage output type.
      ElementComputeEpilogue, // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      cute::conditional_t< // Second stage output type.
          USE_BIAS,
          ElementBias,
          ElementOutput>,
      ElementComputeEpilogue, // Second stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  using ComputeBias = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::plus,
      ElementOutput, // Final (optional) stage output type.
      ElementBias, // Final stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeBias =
      cutlass::epilogue::fusion::Sm90EVT<ComputeBias, Bias, EVTCompute1>;

  using EpilogueEVT =
      cute::conditional_t<USE_BIAS, EVTComputeBias, EVTCompute1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementComputeEpilogue,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          cutlass::epilogue::TmaWarpSpecialized,
          EpilogueEVT>::CollectiveOp;

  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using FastDefaultSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using FastPongSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using SlowAccum = cute::conditional_t<PONG, PongSchedule, DefaultSchedule>;
  using FastAccum =
      cute::conditional_t<PONG, FastPongSchedule, FastDefaultSchedule>;
  using MainLoopSchedule =
      cute::conditional_t<FAST_ACCUM, FastAccum, SlowAccum>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementInputA,
          LayoutInputA,
          AlignmentInputA,
          ElementInputB,
          LayoutInputB,
          AlignmentInputB,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainLoopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, cute::Int<1>{}));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, cute::Int<1>{}));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(M, N, cute::Int<1>{}));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      {reinterpret_cast<ElementInputA*>(XQ.data_ptr()),
       stride_a,
       reinterpret_cast<ElementInputB*>(WQ.data_ptr()),
       stride_b},
      {{}, // Epilogue thread we populate below.
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};

  if constexpr (USE_BIAS) {
    arguments.epilogue.thread = {
        {reinterpret_cast<ElementBias*>(bias.value().data_ptr())}, // bias
        // compute_1
        {
            {reinterpret_cast<ElementComputeEpilogue*>(
                x_scale.data_ptr())}, // x_scale
            // compute_0
            {
                {reinterpret_cast<ElementComputeEpilogue*>(
                    w_scale.data_ptr())}, // w_scale
                {}, // Accumulator
                {} // Multiplies
            },
            {}, // Multiplies
        },
        {}, // Plus
    };
  } else {
    arguments.epilogue.thread = {
        {reinterpret_cast<ElementComputeEpilogue*>(
            x_scale.data_ptr())}, // x_scale
        // compute_0
        {
            {reinterpret_cast<ElementComputeEpilogue*>(
                w_scale.data_ptr())}, // w_scale
            {}, // Accumulator
            {} // Multiplies
        },
        {}, // Multiplies
    };
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

// FP8 Rowwise Cutlass kernel dispatch.
template <typename InputDType, bool FastAccum, bool UseBias, typename BiasDType>
at::Tensor dispatch_fp8_rowwise_kernel(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    std::optional<at::Tensor> output) {
  KernelMode kernel = get_kernel_mode(XQ, WQ);
  if (kernel == KernelMode::Small) {
    return f8f8bf16_rowwise_impl<
        64,
        128,
        128,
        2,
        1,
        1,
        false,
        FastAccum,
        UseBias,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  } else if (kernel == KernelMode::Large) {
    return f8f8bf16_rowwise_impl<
        128,
        128,
        128,
        2,
        1,
        1,
        true,
        FastAccum,
        UseBias,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  } else {
    return f8f8bf16_rowwise_impl<
        128,
        128,
        128,
        1,
        2,
        1,
        false,
        FastAccum,
        UseBias,
        InputDType,
        BiasDType>(XQ, WQ, x_scale, w_scale, bias, output);
  }
}

at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale, // FP32
    at::Tensor w_scale, // FP32
    std::optional<at::Tensor> bias = c10::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = c10::nullopt) {
  // Check datatypes.
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat && w_scale.dtype() == at::kFloat,
      "Scale tensors must be float32.");
  if (bias.has_value()) {
    TORCH_CHECK(
        bias.value().dtype() == at::kFloat ||
            bias.value().dtype() == at::kBFloat16,
        "Bias type must be bfloat16 or float32 if provided.");
  }
  bool use_bias = bias.has_value();
  bool bf16_bias = use_bias && bias.value().dtype() == at::kBFloat16;

  // Templatize based on input dtype.
  bool use_e5m2 = XQ.dtype() == at::kFloat8_e5m2;

  if (use_bias) {
    if (bf16_bias) {
      if (use_fast_accum) {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e5m2_t,
              true,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e4m3_t,
              true,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      } else {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e5m2_t,
              false,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e4m3_t,
              false,
              true,
              cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      }
    } else {
      if (use_fast_accum) {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e5m2_t,
              true,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e4m3_t,
              true,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      } else {
        if (use_e5m2) {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e5m2_t,
              false,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        } else {
          return dispatch_fp8_rowwise_kernel<
              cutlass::float_e4m3_t,
              false,
              true,
              float>(XQ, WQ, x_scale, w_scale, bias, output);
        }
      }
    }
  } else {
    if (use_fast_accum) {
      if (use_e5m2) {
        return dispatch_fp8_rowwise_kernel<
            cutlass::float_e5m2_t,
            true,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return dispatch_fp8_rowwise_kernel<
            cutlass::float_e4m3_t,
            true,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      }
    } else {
      if (use_e5m2) {
        return dispatch_fp8_rowwise_kernel<
            cutlass::float_e5m2_t,
            false,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      } else {
        return dispatch_fp8_rowwise_kernel<
            cutlass::float_e4m3_t,
            false,
            false,
            float>(XQ, WQ, x_scale, w_scale, bias, output);
      }
    }
  }
}

// Cutlass blockwise kernel
template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K>
at::Tensor f8f8bf16_blockwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t block_m,
    int64_t block_n,
    int64_t block_k) {
  TORCH_CHECK(XQ.dim() == 2);
  TORCH_CHECK(WQ.dim() == 2);
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);
  TORCH_CHECK(WQ.size(1) == K);
  TORCH_CHECK(XQ.stride(0) == K);
  TORCH_CHECK(XQ.stride(1) == 1);
  TORCH_CHECK(WQ.stride(0) == K);
  TORCH_CHECK(WQ.stride(1) == 1);

  TORCH_CHECK(block_m % TB_N == 0);
  TORCH_CHECK(block_n % TB_M == 0);
  TORCH_CHECK(block_k % TB_K == 0);

  TORCH_CHECK(x_scale.dim() == 2);
  TORCH_CHECK(w_scale.dim() == 2);
  TORCH_CHECK(x_scale.size(0) == ceil_div(M, block_m));
  TORCH_CHECK(x_scale.size(1) == ceil_div(K, block_k));
  TORCH_CHECK(w_scale.size(0) == ceil_div(N, block_n));
  TORCH_CHECK(w_scale.size(1) == ceil_div(K, block_k));
  TORCH_CHECK(x_scale.stride(0) == ceil_div(K, block_k));
  TORCH_CHECK(x_scale.stride(1) == 1);
  TORCH_CHECK(w_scale.stride(0) == ceil_div(K, block_k));
  TORCH_CHECK(w_scale.stride(1) == 1);

  TORCH_CHECK(XQ.dtype() == at::kFloat8_e4m3fn);
  TORCH_CHECK(WQ.dtype() == at::kFloat8_e4m3fn);
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(WQ.is_cuda());
  TORCH_CHECK(XQ.device().index() == WQ.device().index());
  TORCH_CHECK(x_scale.dtype() == at::kFloat);
  TORCH_CHECK(w_scale.dtype() == at::kFloat);
  TORCH_CHECK(x_scale.is_cuda());
  TORCH_CHECK(w_scale.is_cuda());
  TORCH_CHECK(x_scale.device().index() == XQ.device().index());
  TORCH_CHECK(w_scale.device().index() == XQ.device().index());

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ElementInputA = cutlass::float_e4m3_t;
  using LayoutInputA = cutlass::layout::RowMajor;
  constexpr int AlignmentInputA = 16 / sizeof(ElementInputA);

  using ElementInputB = cutlass::float_e4m3_t;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputB = 16 / sizeof(ElementInputB);

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  constexpr int AlignmentOutput = 16 / sizeof(ElementOutput);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TB_K>>; // Threadblock-level
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
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementComputeEpilogue,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

  using MainLoopSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaling;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementInputA,
          LayoutInputA,
          AlignmentInputA,
          ElementInputB,
          LayoutInputB,
          AlignmentInputB,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainLoopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideD;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(N, K, cute::Int<1>{}));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(M, K, cute::Int<1>{}));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(N, M, cute::Int<1>{}));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K},
      {reinterpret_cast<cutlass::float_e4m3_t*>(WQ.data_ptr()),
       stride_a,
       reinterpret_cast<cutlass::float_e4m3_t*>(XQ.data_ptr()),
       stride_b,
       w_scale.data_ptr<float>(),
       x_scale.data_ptr<float>(),
       static_cast<uint8_t>(block_n / TB_M),
       static_cast<uint8_t>(block_m / TB_N),
       static_cast<uint8_t>(block_k / TB_K)},
      {{},
       (cutlass::bfloat16_t*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (cutlass::bfloat16_t*)Y.data_ptr<at::BFloat16>(),
       stride_output},
  };

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

// FP8 blockwise Cutlass kernel dispatch.
at::Tensor dispatch_fp8_blockwise_kernel(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t block_m,
    int64_t block_n,
    int64_t block_k) {
  KernelMode kernel = get_kernel_mode(XQ, WQ);
  if (kernel == KernelMode::Small) {
    return f8f8bf16_blockwise_impl<128, 128, 128, 2, 1, 1>(
        XQ, WQ, x_scale, w_scale, block_m, block_n, block_k);
  } else if (kernel == KernelMode::Large) {
    return f8f8bf16_blockwise_impl<128, 128, 128, 2, 1, 1>(
        XQ, WQ, x_scale, w_scale, block_m, block_n, block_k);
  } else {
    return f8f8bf16_blockwise_impl<128, 128, 128, 1, 2, 1>(
        XQ, WQ, x_scale, w_scale, block_m, block_n, block_k);
  }
}

at::Tensor f8f8bf16_blockwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale, // FP32
    at::Tensor w_scale, // FP32
    int64_t block_m = 256,
    int64_t block_n = 256,
    int64_t block_k = 256) {
  // Check datatypes.
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat && w_scale.dtype() == at::kFloat,
      "Scale tensors must be float32.");

  return dispatch_fp8_blockwise_kernel(
      XQ, WQ, x_scale, w_scale, block_m, block_n, block_k);
}

template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    typename WEIGHT_SCALE_DTYPE>
at::Tensor bf16i4bf16_rowwise_impl(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  int M = X.size(0);
  int N = WQ.size(0);
  int K = X.size(1);
  int num_groups = w_scale.size(0);

  TORCH_CHECK(X.is_cuda() && X.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());
  TORCH_CHECK(w_zp.is_cuda() && w_zp.is_contiguous());
  TORCH_CHECK(K >= num_groups && K % num_groups == 0);

  int group_size = K / num_groups;

  auto Y = at::empty({M, N}, X.options().dtype(at::kBFloat16));

  using ElementInputA = cutlass::bfloat16_t;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputA =
      128 /
      cutlass::sizeof_bits<
          ElementInputA>::value; // Memory access granularity/alignment of A
                                 // matrix in units of elements (up to 16 bytes)

  using ElementInputB = cutlass::int4b_t;
  using LayoutInputB = cutlass::layout::RowMajor;
  constexpr int AlignmentInputB =
      128 /
      cutlass::sizeof_bits<
          ElementInputB>::value; // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)

  using ElementScale = WEIGHT_SCALE_DTYPE;
  using ElementZeroPoint = WEIGHT_SCALE_DTYPE;
  using ElementComputeEpilogue = float;
  using ElementAccumulator = float;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  constexpr int AlignmentOutput =
      128 /
      cutlass::sizeof_bits<
          ElementOutput>::value; // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TB_K>>; // Threadblock-level
                        // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster
  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecializedMixedInput;
  using PongSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using MainLoopSchedule =
      cute::conditional_t<PONG, PongSchedule, DefaultSchedule>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          EpilogueTileType,
          ElementAccumulator,
          ElementAccumulator,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          cute::tuple<ElementInputB, ElementScale, ElementZeroPoint>,
          LayoutInputB,
          AlignmentInputB,
          ElementInputA,
          LayoutInputA,
          AlignmentInputA,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainLoopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;
  using StrideS = typename CollectiveMainloop::StrideScale;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, cute::Int<1>{}));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, cute::Int<1>{}));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(N, M, cute::Int<1>{}));
  StrideS stride_S = cutlass::make_cute_packed_stride(
      StrideS{}, cute::make_shape(N, num_groups, cute::Int<1>{}));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K},
      {reinterpret_cast<ElementInputB*>(WQ.data_ptr()),
       stride_b,
       reinterpret_cast<ElementInputA*>(X.data_ptr()),
       stride_a,
       reinterpret_cast<ElementScale*>(w_scale.data_ptr()),
       stride_S,
       group_size,
       reinterpret_cast<ElementZeroPoint*>(w_zp.data_ptr())},
      {{1.0, 0.0},
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};

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

template <typename WEIGHT_SCALE_DTYPE>
at::Tensor dispatch_bf16i4bf16_rowwise_kernel(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  KernelMode kernel = get_kernel_mode(X, WQ);
  if (kernel == KernelMode::Small) {
    return bf16i4bf16_rowwise_impl<
        64,
        128,
        128,
        1,
        1,
        1,
        false,
        WEIGHT_SCALE_DTYPE>(X, WQ, w_scale, w_zp);
  } else if (kernel == KernelMode::Large) {
    return bf16i4bf16_rowwise_impl<
        64,
        256,
        128,
        1,
        1,
        1,
        true,
        WEIGHT_SCALE_DTYPE>(X, WQ, w_scale, w_zp);
  } else {
    return bf16i4bf16_rowwise_impl<
        64,
        256,
        128,
        1,
        1,
        1,
        false,
        WEIGHT_SCALE_DTYPE>(X, WQ, w_scale, w_zp);
  }
}

at::Tensor bf16i4bf16_rowwise(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  // Check datatypes.
  TORCH_CHECK(
      (w_scale.dtype() == at::kFloat && w_zp.dtype() == at::kFloat) ||
          (w_scale.dtype() == at::kHalf && w_zp.dtype() == at::kHalf) ||
          (w_scale.dtype() == at::kBFloat16 && w_zp.dtype() == at::kBFloat16),
      "Weight scale and zero point tensors must be float32, bfloat16, or float16, and dtype of weight scale and zero point tensors must be the same .");

  if (w_scale.dtype() == at::kFloat) {
    return dispatch_bf16i4bf16_rowwise_kernel<float>(X, WQ, w_scale, w_zp);
  } else if (w_scale.dtype() == at::kHalf) {
    return dispatch_bf16i4bf16_rowwise_kernel<cutlass::half_t>(
        X, WQ, w_scale, w_zp);
  } else if (w_scale.dtype() == at::kBFloat16) {
    return dispatch_bf16i4bf16_rowwise_kernel<cutlass::bfloat16_t>(
        X, WQ, w_scale, w_zp);
  } else {
    throw std::runtime_error(
        "Weight scale and zero point data type not supported in bf16i4bf16_rowwise");
  }
}

template <
    int TB_M,
    int TB_N,
    int TB_K,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool PONG,
    typename INPUT_DTYPE,
    typename WEIGHT_SCALE_DTYPE>
at::Tensor f8i4bf16_rowwise_impl(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // INT4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp) {
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);
  int num_groups = w_scale.size(0);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(x_scale.is_cuda() && x_scale.is_contiguous());
  TORCH_CHECK(w_scale.is_cuda() && w_scale.is_contiguous());
  TORCH_CHECK(w_zp.is_cuda() && w_zp.is_contiguous());
  TORCH_CHECK(K >= num_groups && K % num_groups == 0);

  int group_size = K / num_groups;

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ElementInputA = INPUT_DTYPE;
  using LayoutInputA = cutlass::layout::ColumnMajor;
  constexpr int AlignmentInputA =
      128 /
      cutlass::sizeof_bits<
          ElementInputA>::value; // Memory access granularity/alignment of A
                                 // matrix in units of elements (up to 16 bytes)

  using ElementInputB = cutlass::int4b_t;
  using LayoutInputB = cutlass::layout::RowMajor;
  constexpr int AlignmentInputB =
      128 /
      cutlass::sizeof_bits<
          ElementInputB>::value; // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)

  using ElementScale = WEIGHT_SCALE_DTYPE;
  using ElementZeroPoint = WEIGHT_SCALE_DTYPE;
  using ElementComputeEpilogue = float;
  using ElementAccumulator = float;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::ColumnMajor;
  constexpr int AlignmentOutput =
      128 /
      cutlass::sizeof_bits<
          ElementOutput>::value; // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

  using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                       // supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TB_K>>; // Threadblock-level
                        // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster
  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecializedMixedInput;
  using PongSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using MainLoopSchedule =
      cute::conditional_t<PONG, PongSchedule, DefaultSchedule>;

  // Implement rowwise scaling epilogue for x
  using XScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      PONG ? 2 : 1,
      TileShape,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementOutput, // First stage output type.
      ElementComputeEpilogue, // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EpilogueEVT =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, XScale, Accum>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          EpilogueTileType,
          ElementAccumulator,
          ElementAccumulator,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          ElementOutput,
          LayoutOutput,
          AlignmentOutput,
          EpilogueSchedule,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          cute::tuple<ElementInputB, ElementScale, ElementZeroPoint>,
          LayoutInputB,
          AlignmentInputB,
          ElementInputA,
          LayoutInputA,
          AlignmentInputA,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainLoopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideInputA = typename Gemm::GemmKernel::StrideA;
  using StrideInputB = typename Gemm::GemmKernel::StrideB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;
  using StrideS = typename CollectiveMainloop::StrideScale;

  StrideInputA stride_a = cutlass::make_cute_packed_stride(
      StrideInputA{}, cute::make_shape(M, K, cute::Int<1>{}));
  StrideInputB stride_b = cutlass::make_cute_packed_stride(
      StrideInputB{}, cute::make_shape(N, K, cute::Int<1>{}));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(N, M, cute::Int<1>{}));
  StrideS stride_S = cutlass::make_cute_packed_stride(
      StrideS{}, cute::make_shape(N, num_groups, cute::Int<1>{}));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K},
      {reinterpret_cast<ElementInputB*>(WQ.data_ptr()),
       stride_b,
       reinterpret_cast<ElementInputA*>(XQ.data_ptr()),
       stride_a,
       reinterpret_cast<ElementScale*>(w_scale.data_ptr()),
       stride_S,
       group_size,
       reinterpret_cast<ElementZeroPoint*>(w_zp.data_ptr())},
      {{},
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output,
       (ElementOutput*)Y.data_ptr<at::BFloat16>(),
       stride_output}};

  arguments.epilogue.thread = {
      {reinterpret_cast<ElementComputeEpilogue*>(
          x_scale.data_ptr())}, // x_scale
      {}, // Accumulator
      {}, // Multiplies
  };

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

template <typename InputDType, typename WEIGHT_SCALE_DTYPE>
at::Tensor dispatch_f8i4bf16_rowwise_kernel(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp) {
  KernelMode kernel = get_kernel_mode(XQ, WQ);
  if (kernel == KernelMode::Small) {
    return f8i4bf16_rowwise_impl<
        64,
        128,
        128,
        1,
        1,
        1,
        false,
        InputDType,
        WEIGHT_SCALE_DTYPE>(XQ, WQ, x_scale, w_scale, w_zp);
  } else if (kernel == KernelMode::Large) {
    return f8i4bf16_rowwise_impl<
        64,
        256,
        128,
        1,
        1,
        1,
        true,
        InputDType,
        WEIGHT_SCALE_DTYPE>(XQ, WQ, x_scale, w_scale, w_zp);
  } else {
    return f8i4bf16_rowwise_impl<
        64,
        256,
        128,
        1,
        1,
        1,
        false,
        InputDType,
        WEIGHT_SCALE_DTYPE>(XQ, WQ, x_scale, w_scale, w_zp);
  }
}

at::Tensor f8i4bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // INT4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp) {
  // Check datatypes.
  TORCH_CHECK(
      x_scale.dtype() == at::kFloat, "Input scale tensor must be float32.");
  TORCH_CHECK(
      (w_scale.dtype() == at::kFloat && w_zp.dtype() == at::kFloat) ||
          (w_scale.dtype() == at::kHalf && w_zp.dtype() == at::kHalf) ||
          (w_scale.dtype() == at::kBFloat16 && w_zp.dtype() == at::kBFloat16),
      "Weight scale and zero point tensors must be float32, bfloat16, or float16, and dtype of weight scale and zero point tensors must be the same .");

  // Templatize based on input and weight scale/zero point dtype.
  bool use_e5m2 = XQ.dtype() == at::kFloat8_e5m2;

  if (w_scale.dtype() == at::kFloat) {
    if (use_e5m2) {
      return dispatch_f8i4bf16_rowwise_kernel<cutlass::float_e5m2_t, float>(
          XQ, WQ, x_scale, w_scale, w_zp);
    } else {
      return dispatch_f8i4bf16_rowwise_kernel<cutlass::float_e4m3_t, float>(
          XQ, WQ, x_scale, w_scale, w_zp);
    }
  } else if (w_scale.dtype() == at::kHalf) {
    if (use_e5m2) {
      return dispatch_f8i4bf16_rowwise_kernel<
          cutlass::float_e5m2_t,
          cutlass::half_t>(XQ, WQ, x_scale, w_scale, w_zp);
    } else {
      return dispatch_f8i4bf16_rowwise_kernel<
          cutlass::float_e4m3_t,
          cutlass::half_t>(XQ, WQ, x_scale, w_scale, w_zp);
    }
  } else if (w_scale.dtype() == at::kBFloat16) {
    if (use_e5m2) {
      return dispatch_f8i4bf16_rowwise_kernel<
          cutlass::float_e5m2_t,
          cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, w_zp);
    } else {
      return dispatch_f8i4bf16_rowwise_kernel<
          cutlass::float_e4m3_t,
          cutlass::bfloat16_t>(XQ, WQ, x_scale, w_scale, w_zp);
    }
  } else {
    throw std::runtime_error(
        "Weight scale and zero point data type not supported in f8i4bf16_rowwise");
  }
}

at::Tensor f8f8bf16_cublas(
    at::Tensor A, // FP8
    at::Tensor B, // FP8
    std::optional<at::Tensor> Ainvs = c10::nullopt,
    std::optional<at::Tensor> Binvs = c10::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = c10::nullopt) {
  auto m = A.size(0);
  auto n = B.size(0);
  auto k = A.size(1);
  size_t workspaceSize = CUBLAS_WORKSPACE_SIZE;
  const int8_t fastAccuMode = use_fast_accum ? 1 : 0;

  TORCH_CHECK(A.is_cuda() && A.is_contiguous());
  TORCH_CHECK(B.is_cuda() && B.is_contiguous());

  cublasLtHandle_t ltHandle;
  checkCublasStatus(cublasLtCreate(&ltHandle));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspace = allocator.allocate(workspaceSize);
  if (output.has_value()) {
    auto output_tensor = output.value();
    TORCH_CHECK(output_tensor.is_cuda());
    TORCH_CHECK(output_tensor.is_contiguous());
    TORCH_CHECK(
        output_tensor.numel() == m * n,
        "output_tensor.numel=",
        output_tensor.numel(),
        ", m=",
        m,
        ", n=",
        n);
    TORCH_CHECK(output_tensor.options().dtype() == at::kBFloat16);
  }

  const cudaDataType_t A_type = CUDA_R_8F_E4M3;
  const cudaDataType_t B_type = CUDA_R_8F_E4M3;
  const cudaDataType_t D_type = CUDA_R_16BF;

  float one = 1.0;
  float zero = 0.0;

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  // Create matrix descriptors. Not setting any extra attributes.

  auto lda = k;
  auto ldb = k;
  auto ldd = n;
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, A_type, k, m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, B_type, k, n, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, D_type, n, m, ldd));

  checkCublasStatus(
      cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_FAST_ACCUM,
      &fastAccuMode,
      sizeof(fastAccuMode)));

  if (Ainvs.has_value()) {
    const float* Ainvs_pt = Ainvs.value().data_ptr<float>();
    checkCublasStatus(cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &Ainvs_pt,
        sizeof(Ainvs_pt)));
  }

  if (Binvs.has_value()) {
    const float* Binvs_pt = Binvs.value().data_ptr<float>();
    checkCublasStatus(cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &Binvs_pt,
        sizeof(Binvs_pt)));
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue,
      sizeof(epilogue)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));

  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize,
      sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      operationDesc,
      Bdesc,
      Adesc,
      Ddesc,
      Ddesc,
      preference,
      1,
      &heuristicResult,
      &returnedResults));

  if (returnedResults == 0)
    throw std::runtime_error("Unable to find any suitable algorithms");

  // D = alpha * (A * B) + beta * C
  // Warmup
  auto Y = output.value_or(at::empty({m, n}, A.options().dtype(at::kBFloat16)));
  checkCublasStatus(cublasLtMatmul(
      ltHandle,
      operationDesc,
      static_cast<const void*>(&one), /* alpha */
      B.data_ptr(), /* B */
      Bdesc,
      A.data_ptr(), /* A */
      Adesc,
      static_cast<const void*>(&zero), /* beta */
      nullptr, /* C */
      Ddesc,
      Y.data_ptr(), /* D */
      Ddesc,
      &heuristicResult.algo, /* algo */
      workspace.mutable_get(), /* workspace */
      workspaceSize,
      at::cuda::getCurrentCUDAStream())); /* stream */
  return Y;
}
#else
at::Tensor f8f8bf16_cublas(
    at::Tensor A, // FP8
    at::Tensor B, // FP8
    std::optional<at::Tensor> Ainvs = c10::nullopt,
    std::optional<at::Tensor> Binvs = c10::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = c10::nullopt) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
at::Tensor f8f8bf16(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor scale,
    bool use_fast_accum) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
at::Tensor f8f8bf16_tensorwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    double scale,
    bool use_fast_accum) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
at::Tensor f8i4bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // INT4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor w_zp) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
at::Tensor bf16i4bf16_rowwise(
    at::Tensor X, // BF16
    at::Tensor WQ, // INT4
    at::Tensor w_scale,
    at::Tensor w_zp) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias = c10::nullopt,
    bool use_fast_accum = true,
    std::optional<at::Tensor> output = c10::nullopt) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
at::Tensor f8f8bf16_blockwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t block_m = 256,
    int64_t block_n = 256,
    int64_t block_k = 256) {
  throw std::runtime_error(
      "CUDA version is older than 12.0"); // requires CUDA>=12
}
#endif

at::Tensor i8i8bf16(
    at::Tensor XQ, // INT8
    at::Tensor WQ, // INT8
    double scale,
    int64_t split_k) {
  auto M = XQ.size(0);
  auto N = WQ.size(0);
  auto K = XQ.size(1);
#ifdef SMOOTHQUANT_SM90A
  if (M <= 128) {
    return i8i8bf16sm90a_impl<64, 128, 128, 2, 1, 1>(XQ, WQ, scale);
  } else {
    return i8i8bf16sm90a_impl<128, 128, 128, 1, 2, 1>(XQ, WQ, scale);
  }
#else
  if (M <= 128 && N >= K) {
    return i8i8bf16_impl<64, 128, 64, 32, 64, 64>(XQ, WQ, scale, split_k);
  } else if (M <= 128 && N < K) {
    return i8i8bf16_impl<64, 64, 128, 32, 32, 128>(XQ, WQ, scale, split_k);
  } else {
    return i8i8bf16_impl<256, 128, 64, 64, 64, 64>(XQ, WQ, scale, split_k);
  }
#endif
}

template <int TB_M, int TB_N, int TB_K, int W_M, int W_N, int W_K>
at::Tensor i8i8bf16_dynamic_impl(
    at::Tensor XQ, // INT8
    at::Tensor WQ, // INT8
    at::Tensor scale,
    int64_t split_k) {
  auto M = XQ.size(0);
  auto N = WQ.size(0);
  auto K = XQ.size(1);

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,
      cutlass::layout::RowMajor,
      int8_t,
      cutlass::layout::ColumnMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>, // ThreadBlockShape
      cutlass::gemm::GemmShape<W_M, W_N, W_K>, // WarpShape
      cutlass::gemm::GemmShape<16, 8, 32>, // InstructionShape
      cutlass::epilogue::thread::LinearCombinationOnDevice<
          ElementOutput,
          128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      3,
      16,
      16,
      true>;

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      XQ.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      WQ.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      (ElementOutput*)Y.data_ptr<at::BFloat16>(),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size,
      input_ref,
      weight_ref,
      out_ref,
      out_ref,
      {scale.data_ptr<float>()},
      int(split_k)};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  auto workspace =
      at::empty({int64_t(workspace_size)}, Y.options().dtype(at::kChar));

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(
      arguments, workspace.data_ptr(), at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return Y;
}

at::Tensor i8i8bf16_dynamic(
    at::Tensor XQ, // INT8
    at::Tensor WQ, // INT8
    at::Tensor scale,
    int64_t split_k) {
  auto M = XQ.size(0);
  auto N = WQ.size(0);
  auto K = XQ.size(1);
  if (M <= 128 && N >= K) {
    return i8i8bf16_dynamic_impl<64, 128, 64, 32, 64, 64>(
        XQ, WQ, scale, split_k);
  } else if (M <= 128 && N < K) {
    return i8i8bf16_dynamic_impl<64, 64, 128, 32, 32, 128>(
        XQ, WQ, scale, split_k);
  } else {
    return i8i8bf16_dynamic_impl<256, 128, 64, 64, 64, 64>(
        XQ, WQ, scale, split_k);
  }
}

} // namespace fbgemm_gpu

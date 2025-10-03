// @nolint
#pragma once
#include "blackwell_fmha_utils.hpp"
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <
    typename Element,
    typename ActiveMask,
    int HeadDim,
    bool kIsVarlen,
    bool kIsDeterministic,
    class... KernelOptions>
std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd(
    const at::Tensor& dO,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& o,
    const at::Tensor& softmax_lse,
    const std::optional<const at::Tensor>& cu_seqlens_q,
    const std::optional<const at::Tensor>& cu_seqlens_k,
    std::optional<int> max_seq_len_q,
    std::optional<int> max_seq_len_k,
    const std::optional<double> softmax_scale,
    const int window_size_left,
    const int window_size_right
) {
  const auto device = q.device();
  at::cuda::CUDAGuard device_guard(device);

  using ElementAccumulator = float;

  // Q K D D_VO ((H_R, H_K) B)
  using ProblemShapeType = std::conditional_t<
    kIsVarlen,
    cute::tuple<VariableLength, VariableLength, int, int, cute::tuple<cute::tuple<int, int>, int>>,
    cute::tuple<int, int, int, int, cute::tuple<cute::tuple<int, int>, int>>
  >;
  using D_H = cute::Int<HeadDim>;
  using TileShape = Shape<_128, _128, D_H>;

  using Operation = cutlass::fmha::device::
      Sm100FmhaBwd<ProblemShapeType, Element, ElementAccumulator, TileShape, /*kIsMla=*/false, ActiveMask, kIsDeterministic>;

  using StrideQ = Stride<int, _1, Stride<Stride<int, int>, int>>; // Q D    ((H_R, H_K), B)
  using StrideK = Stride<int, _1, Stride<Stride<_0, int>, int>>;  // K D    ((H_R, H_K), B)
  using StrideV = StrideK;                                        // K D_VO ((H_R, H_K), B)
  using StrideO = StrideQ;                                        // Q D_VO ((H_R, H_K), B)
  using StrideLSE = Stride<_1, Stride<Stride<int, int>, int>>;    // Q      ((H_R, H_K), B)

  // Backwards specific
  using StrideDQ = StrideQ;
  using StrideDK = StrideK;
  using StrideDV = StrideV;
  using StrideDO = StrideO;

  if (kIsVarlen) {
    TORCH_CHECK(
        q.dim() == 3,
        "Expect Q shape to be (total_Q_seqlen, num_Q_heads, head_dim) ",
        "Found shape ", q.sizes());
    TORCH_CHECK(
        k.dim() == 3,
        "Expect K shape to be (total_KV_seqlen, num_KV_heads, head_dim) ",
        "Found shape ", k.sizes());
    TORCH_CHECK(
        v.dim() == 3,
        "Expect V shape to be (total_KV_seqlen, num_KV_heads, head_dim) ",
        "Found shape ", v.sizes());
  }
  else {
    TORCH_CHECK(
        q.dim() == 4,
        "Expect Q shape to be (batch_size, Q_seqlen, num_Q_heads, head_dim). ",
        "Found shape ", q.sizes());
    TORCH_CHECK(
        k.dim() == 4,
        "Expect K shape to be (batch_size, KV_seqlen, num_KV_heads, head_dim) ",
        "Found shape ", k.sizes());
    TORCH_CHECK(
        v.dim() == 4,
        "Expect V shape to be (batch_size, KV_seqlen, num_KV_heads, head_dim) ",
        "Found shape ", v.sizes());
  }

  if constexpr (kIsVarlen) {
    TORCH_CHECK(cu_seqlens_q.has_value(), "cu_seqlens_q should be set");
    TORCH_CHECK(cu_seqlens_k.has_value(), "cu_seqlens_k should be set");
    TORCH_CHECK(max_seq_len_q.has_value(), "max_seq_len_q should be set");
    TORCH_CHECK(max_seq_len_k.has_value(), "max_seq_len_k should be set");
  }

  int B = kIsVarlen ? cu_seqlens_q->size(0) - 1 : q.size(0);
  // Q represents SumB(Q) for varlen (jagged len)
  int Q = kIsVarlen ? q.size(0) : q.size(1);
  int K = kIsVarlen ? k.size(0) : k.size(1);
  int H_Q = kIsVarlen ? q.size(1) : q.size(2);
  int H_K = kIsVarlen ? k.size(1) : k.size(2);
  int D = q.size(q.dim() - 1); // Head dimension (D)

  TORCH_CHECK(H_Q % H_K == 0, "Q heads must be a multiple of KV heads");
  int H_R = H_Q / H_K;

  ProblemShapeType problem_shape;
  if constexpr (kIsVarlen) {
    problem_shape = cute::make_tuple(
        VariableLength{
            *max_seq_len_q, static_cast<int*>(cu_seqlens_q->data_ptr()), int(q.size(0))},
        VariableLength{
            *max_seq_len_k, static_cast<int*>(cu_seqlens_k->data_ptr()), int(k.size(0))},
        D,
        D,
        make_shape(make_shape(H_R, H_K), B));
  }
  else {
    problem_shape = cute::make_tuple(
        Q, K, D, D, make_shape(make_shape(H_R, H_K), B));
  }
  TORCH_CHECK(D == HeadDim);
  TORCH_CHECK(D % 8 == 0); // Alignment
  if constexpr (!kIsVarlen) {
    // TODO: support Q < 8
    TORCH_CHECK(Q >= 8);
  }

  // Reshape to get strides
  auto B_ = kIsVarlen ? 1 : B;
  auto q_ = q.reshape({B_, Q, H_K, H_R, D});
  auto o_ = o.reshape({B_, Q, H_K, H_R, D});
  auto dO_ = dO.reshape({B_, Q, H_K, H_R, D});
  auto k_ = k.reshape({B_, K, H_K, 1, D}).expand({B_, K, H_K, H_R, D});
  auto lse_ = softmax_lse.reshape({B_, H_K, H_R, Q});
  auto ndim = q_.dim();

  TORCH_CHECK(q_.stride(ndim - 1) == 1, "The head dim in Q must be contiguous");
  TORCH_CHECK(k_.stride(ndim - 1) == 1, "The head dim in KV must be contiguous");
  TORCH_CHECK(o_.stride(ndim - 1) == 1, "The head dim in O must be contiguous");
  TORCH_CHECK(dO_.stride(ndim - 1) == 1, "The head dim in dO must be contiguous");
  TORCH_CHECK(lse_.stride(lse_.dim() - 1) == 1, "The seqlen dim in LSE must be contiguous");
  if (H_R != 1) {
    TORCH_CHECK(k_.stride(3) == 0, "The shared KV head stride must be zero");
  }

  // Note: We use a different layout from 77_blackwell_fmha_bwd.cu.
  // Q shape = (B, Q, H_K, H_R, D)
  StrideQ stride_Q = make_stride(
      static_cast<int>(q_.stride(1)), _1{},
      make_stride(
        make_stride(
          static_cast<int>(q_.stride(3)),
          static_cast<int>(q_.stride(2))),
        static_cast<int>(q_.stride(0))));

  // K shape = (B, K, H_K, 1, D)
  StrideK stride_K = make_stride(
      static_cast<int>(k_.stride(1)), _1{},
      make_stride(
        make_stride(_0{}, static_cast<int>(k_.stride(2))),
        static_cast<int>(k_.stride(0))));

  StrideV stride_V = stride_K;

  // LSE shape = (B, H_K, H_R, Q)
  StrideLSE stride_LSE = make_stride(
      _1{},
      make_stride(
        make_stride(
          static_cast<int>(lse_.stride(2)),
          static_cast<int>(lse_.stride(1))),
        static_cast<int>(lse_.stride(0))));

  // O shape = (B, Q, H_K, H_R, D)
  StrideO stride_O = make_stride(
      static_cast<int>(o_.stride(1)), _1{},
      make_stride(
        make_stride(
          static_cast<int>(o_.stride(3)),
          static_cast<int>(o_.stride(2))),
        static_cast<int>(o_.stride(0))));

  // dO shape = (B, Q, H_K, H_R, D)
  StrideDO stride_dO = make_stride(
      static_cast<int>(dO_.stride(1)), _1{},
      make_stride(
        make_stride(
          static_cast<int>(dO_.stride(3)),
          static_cast<int>(dO_.stride(2))),
        static_cast<int>(dO_.stride(0))));

  // Outputs are always contiguous
  StrideDQ stride_dQ = make_stride(
      H_Q * D, _1{},
      make_stride(make_stride(D, H_R * D), D * Q * H_Q));
  StrideDK stride_dK = make_stride(
      H_K * D, _1{},
      make_stride(make_stride(_0{}, D), D * K * H_K));
  StrideDV stride_dV = stride_dK;

  if constexpr (kIsVarlen) {
    get<2, 1>(stride_Q) = 0;
    get<2, 1>(stride_K) = 0;
    get<2, 1>(stride_V) = 0;
    get<2, 1>(stride_O) = 0;
    get<2, 1>(stride_dO) = 0;

    get<1, 1>(stride_LSE) = 0;

    get<2, 1>(stride_dQ) = 0;
    get<2, 1>(stride_dK) = 0;
    get<2, 1>(stride_dV) = 0;
  }

  ElementAccumulator softmax_scale_value = softmax_scale.has_value() ? softmax_scale.value() : (1.0f / sqrtf(D));

  at::Tensor dQ = torch::empty_like(q);
  at::Tensor dK = torch::empty_like(k);
  at::Tensor dV = torch::empty_like(v);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device.index();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  auto seqlen_q = kIsVarlen ? max_seq_len_q.value() : q.size(1);

  int* dq_semaphore_ptr = nullptr;
  at::Tensor dq_semaphore;
  if (kIsDeterministic) {
    auto kBlockM = cute::get<0>(TileShape{});
    auto opts = q.options();
    dq_semaphore = torch::zeros(
        {(seqlen_q + kBlockM - 1) / kBlockM, B, H_Q},
        opts.dtype(torch::kInt32));
    dq_semaphore_ptr = static_cast<int*>(dq_semaphore.data_ptr());
  }

  typename Operation::Arguments arguments{
    problem_shape,
    static_cast<Element*>(q.data_ptr()),
    stride_Q,
    static_cast<Element*>(k.data_ptr()),
    stride_K,
    static_cast<Element*>(v.data_ptr()),
    stride_V,
    static_cast<Element*>(o.data_ptr()),
    stride_O,
    static_cast<ElementAccumulator*>(softmax_lse.data_ptr()),
    stride_LSE,
    static_cast<Element*>(dO.data_ptr()),
    stride_dO,
    static_cast<Element*>(dQ.data_ptr()),
    stride_dQ,
    static_cast<Element*>(dK.data_ptr()),
    stride_dK,
    static_cast<Element*>(dV.data_ptr()),
    stride_dV,
    softmax_scale_value,
    dq_semaphore_ptr,
    window_size_left,
    window_size_right,
    hw_info};
  launch_fmha_op<Operation>(arguments);

  return std::make_tuple(dQ, dK, dV);
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

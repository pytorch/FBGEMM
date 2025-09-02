// @nolint
#include "blackwell_fmha_utils.hpp"
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <
    typename Element,
    typename ActiveMask,
    bool kIsVarlen,
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

  using TileShape = Shape<_128, _128, _128>;

  using Operation = cutlass::fmha::device::
      Sm100FmhaBwd<ProblemShapeType, Element, ElementAccumulator, TileShape, /*kIsMla=*/false, ActiveMask>;

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

  // TODO: pass in softmax_scale?
  ElementAccumulator softmax_scale = 1.0f / sqrtf(D);

  at::Tensor dQ = torch::empty_like(q);
  at::Tensor dK = torch::empty_like(k);
  at::Tensor dV = torch::empty_like(v);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device.index();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

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
    softmax_scale,
    window_size_left,
    window_size_right,
    hw_info};
  launch_fmha_op<Operation>(arguments);

  return std::make_tuple(dQ, dK, dV);
}

struct KernelCoop {};

std::tuple<at::Tensor, at::Tensor, at::Tensor> dispatch_fmha_bwd(
    const at::Tensor& dOutput,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& output,
    const at::Tensor& softmax_lse,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const std::optional<at::Tensor>& cu_seqlens_k,
    std::optional<int64_t> max_seq_len_q,
    std::optional<int64_t> max_seq_len_k,
    bool causal,
    int64_t window_size_left,
    int64_t window_size_right,
    bool bottom_right
) {
  // This workaround initializes the CUDA context to prevent the 201 error
  // (invalid context).  When this function is invoked through PyTorch
  // autograd, it runs on a new thread that hasn't been associated with a CUDA
  // context. To bind this thread to a CUDA context, we call a CUDA runtime API
  // (e.g., cudaFree), which will automatically initialize the context.  This
  // ensures that subsequent calls to driver APIs, which assume an initialized
  // CUDA context, do not result in an invalid context error.
  // TODO: initialize context properly
  cudaFree(0);

  // Handle local attention parameters
  bool local = (window_size_left >= 0 || window_size_right >= 0);
  if (local) {
    // If causal is enabled, override window_size_right to 0 for causal+local behavior
    if (causal) {
      window_size_right = 0;
      causal = false;  // Use local attention instead of causal
    }
    // Expand -1 window sizes to full sequence length if available
    if (window_size_left < 0 && max_seq_len_k.has_value()) {
      window_size_left = max_seq_len_k.value();
    }
    if (window_size_right < 0 && max_seq_len_k.has_value()) {
      window_size_right = max_seq_len_k.value();
    }
  }

  auto dispatch_fmha =
    [&](auto element, auto element_out, auto varlen, auto mask, auto... kernel_options) {
      return fmha_bwd<
        decltype(element),
        decltype(mask),
        varlen,
        decltype(kernel_options)...>
      (
        dOutput,
        query,
        key,
        value,
        output,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seq_len_q,
        max_seq_len_k,
        window_size_left,
        window_size_right);
    };

  auto dispatch_type = [&](auto varlen, auto mask) {
    if (query.dtype() == torch::kFloat16) {
      return dispatch_fmha(cutlass::half_t{}, cutlass::half_t{}, varlen, mask);
    }
    else if (query.dtype() == torch::kBFloat16) {
      return dispatch_fmha(
          cutlass::bfloat16_t{}, cutlass::bfloat16_t{}, varlen, mask);
    }
    else if (query.dtype() == torch::kFloat8_e4m3fn) {
      return dispatch_fmha(
          cutlass::float_e4m3_t{}, cutlass::bfloat16_t{}, varlen, mask);
    }
    TORCH_CHECK(false, "Unsupported dtype for q: ", query.dtype());
  };

  auto dispatch_mask = [&](auto varlen) {
    if (causal) {
      if (bottom_right) {
        return dispatch_type(varlen, CausalForBackwardMask</*kIsQBegin=*/false>{});
      }
      else {
        return dispatch_type(varlen, CausalForBackwardMask</*kIsQBegin=*/true>{});
      }
    }
    else if (local) {
      if (bottom_right) {
        return dispatch_type(varlen, LocalMaskForBackward</*kIsQBegin=*/false>{});
      }
      else {
        return dispatch_type(varlen, LocalMaskForBackward</*kIsQBegin=*/true>{});
      }
    }
    else if (varlen || key.size(1) % 128 != 0) {
      // Use the residual mask for varlen or when K seqlen is not multiple of
      // blockN
      return dispatch_type(varlen, ResidualMaskForBackward{});
    }
    else {
      return dispatch_type(varlen, NoMask{});
    }
  };

  if (max_seq_len_q.has_value()) {
    return dispatch_mask(std::bool_constant<true>{});
  } else {
    TORCH_CHECK(query.dim() == 4, "q must be [B, M, H, D] for fixed length")
    return dispatch_mask(std::bool_constant<false>{});
  }
}

// -------------------------------------------------------------------------------------------------
// Op registration
// -------------------------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("fmha_bwd("
        "    Tensor dOutput, "
        "    Tensor query, "
        "    Tensor key, "
        "    Tensor value, "
        "    Tensor output, "
        "    Tensor softmax_lse, "
        "    Tensor? cu_seqlens_q=None, "
        "    Tensor? cu_seqlens_k=None, "
        "    int? max_seq_len_q=None, "
        "    int? max_seq_len_k=None, "
        "    bool causal=False, "
        "    int window_size_left=-1, "
        "    int window_size_right=-1, "
        "    bool bottom_right=True"
        ") -> (Tensor, Tensor, Tensor)"
  );
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("fmha_bwd", dispatch_fmha_bwd);
}
#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

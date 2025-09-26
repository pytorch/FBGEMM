// @nolint
#include "blackwell_fmha_utils.hpp"
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <
  typename Element,
  typename ElementOut,
  int HeadDim,
  bool kIsVarlen,
  typename ActiveMask
>
std::tuple<at::Tensor, at::Tensor> fmha_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<const at::Tensor>& cu_seqlens_q,
    const std::optional<const at::Tensor>& cu_seqlens_k,
    std::optional<int64_t> max_seq_len_q,
    std::optional<int64_t> max_seq_len_k,
    const std::optional<double> softmax_scale,
    const std::optional<const at::Tensor>& seqlen_kv,
    const int window_size_left,
    const int window_size_right
  ) {
  const auto device = q.device();
  at::cuda::CUDAGuard device_guard(device);

  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;

  // Q K D (H_r H_k) B
  using ProblemShapeRegular =
      cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeVarlen = cute::tuple<
      VariableLength,
      VariableLength,
      int,
      cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeType =
      std::conditional_t<kIsVarlen, ProblemShapeVarlen, ProblemShapeRegular>;

  using StrideQ =
      cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>; // Q D (H_G
                                                                     // H_R B)
  using StrideK =
      cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>; // K D (H_G
                                                                    // H_R B)
  using StrideV = StrideK;
  using StrideO = StrideQ;
  using StrideLSE =
      cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>; // Q   (H_G H_R
                                                                // B)

  static constexpr bool kIsPersistent = true;
  using TileScheduler = std::conditional_t<
      kIsPersistent,
      cutlass::fmha::kernel::PersistentTileScheduler,
      cutlass::fmha::kernel::IndividualTileScheduler>;

  using D_H = cute::Int<HeadDim>;
  using TileShape = Shape<_256, _128, D_H>;

  using Mainloop =
      cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
          Element,
          ElementAccumulatorQK,
          ElementAccumulatorPV,
          TileShape,
          StrideQ,
          StrideK,
          StrideV,
          ActiveMask>;

  using Operation = cutlass::fmha::device::FMHA<
      cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
          ProblemShapeType,
          Mainloop,
          cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
              ElementOut,
              ElementAccumulatorPV,
              typename Mainloop::TileShapePV,
              StrideO,
              StrideLSE>,
          TileScheduler>>;

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

  // Extract dimensions from input tensors
  int H_Q = kIsVarlen ? q.size(1) : q.size(2); // Number of Q heads
  int H_K = kIsVarlen ? k.size(1) : k.size(2); // Number of K heads
  int D = q.size(q.dim() - 1); // Head dimension (D)

  TORCH_CHECK(H_Q % H_K == 0);
  int H_R = H_Q / H_K; // Q heads per K head
  TORCH_CHECK(D == HeadDim);

  // SQ represents SumB(Q) for varlen (jagged len)
  int SQ = kIsVarlen ? q.size(0) : q.size(1);
  int SK = kIsVarlen ? k.size(0) : k.size(1);
  int B = kIsVarlen ? cu_seqlens_q->size(0) - 1 : q.size(0);

  ProblemShapeType problem_shape;
  if constexpr (kIsVarlen) {
    problem_shape = cute::make_tuple(
        VariableLength{
            static_cast<int>(*max_seq_len_q), static_cast<int*>(cu_seqlens_q->data_ptr()), SQ},
        VariableLength{
            static_cast<int>(*max_seq_len_k), static_cast<int*>(cu_seqlens_k->data_ptr()), SK},
        D,
        cute::make_tuple(cute::make_tuple(H_R, H_K), B));
  }
  else {
    problem_shape = cute::make_tuple(
        SQ, SK, D, cute::make_tuple(cute::make_tuple(H_R, H_K), B)
    );
  }

  // Reshape to get strides
  auto B_ = kIsVarlen ? 1 : B;
  auto q_ = q.reshape({B_, SQ, H_K, H_R, D});
  auto k_ = k.reshape({B_, SK, H_K, 1, D}).expand({B_, SK, H_K, H_R, D});
  auto ndim = q_.dim();

  TORCH_CHECK(q_.stride(ndim - 1) == 1, "The head dim in Q must be contiguous");
  TORCH_CHECK(k_.stride(ndim - 1) == 1, "The head dim in K must be contiguous");

  if (H_R != 1) {
    TORCH_CHECK(k_.stride(3) == 0, "The shared K head stride must be zero");
  }

  // Convert torch tensors to CUTLASS format
  // Set up strides for tensors based on dimensions
  // Q shape = (B, Q, H_K, H_R, D)
  StrideQ stride_Q = make_stride(
      static_cast<int>(q_.stride(1)),
      _1{},
      make_stride(
        make_stride(static_cast<int>(q_.stride(3)), static_cast<int>(q_.stride(2))),
        static_cast<int>(q_.stride(0))));

  // K shape = (B, K, H_K, 1, D)
  StrideK stride_K = make_stride(
      static_cast<int>(k_.stride(1)),
      _1{},
      make_stride(
        make_stride(_0{}, static_cast<int>(k_.stride(2))),
        static_cast<int>(k_.stride(0))));
  StrideV stride_V = stride_K;

  // O shape = (B, Q, H_K, H_R, D)
  // O is always contiguous
  StrideO stride_O = make_stride(
      H_Q * D, _1{}, make_stride(make_stride(D, H_R * D), H_Q * D * SQ));

  // LSE shape = (B, H_K, H_R, Q)
  StrideLSE stride_LSE =
      make_stride(_1{}, make_stride(make_stride(SQ, SQ * H_R), SQ * H_Q));

  // The KernelHardwareInfo struct holds the number of SMs on the GPU with a
  // given device ID. This information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device.index();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  // Output tensors
  TensorWrapper<ElementOut> block_O;
  TensorWrapper<ElementAccumulatorPV> block_LSE;
  block_O.reset(q.numel(), kIsVarlen ? D * (*max_seq_len_q) * H_Q : 0);
  int size_LSE = SQ * H_Q * (kIsVarlen ? 1 : B);
  block_LSE.reset(size_LSE);

  typename Operation::Arguments arguments;
  if constexpr (kIsVarlen) {
    get<2, 1>(stride_Q) = 0;
    get<2, 1>(stride_K) = 0;
    get<2, 1>(stride_V) = 0;
    get<2, 1>(stride_O) = 0;
    get<1, 1>(stride_LSE) = 0;
  }
  arguments = {
      problem_shape,
      seqlen_kv.has_value()
          ? static_cast<const int*>(seqlen_kv->data_ptr())
          : nullptr,
      {
          {
              static_cast<Element*>(q.data_ptr()), stride_Q,
              static_cast<Element*>(k.data_ptr()), stride_K,
              static_cast<Element*>(v.data_ptr()), stride_V,
              window_size_left, window_size_right
          },
          static_cast<float>(softmax_scale.value_or(0.0f)) /* softmax_scale */,
          1.0f /* scale_q */,
          1.0f /* scale_k */,
          1.0f /* scale_v */,
          1.0f /* inv_scale_o */,
          window_size_left,
          window_size_right,
      },
      {
          block_O.get(), stride_O,
          block_LSE.get(), stride_LSE
      },
      hw_info
  };

  launch_fmha_op<Operation>(arguments);
  return std::make_tuple(
      block_O.get_data_tensor(q.sizes()),
      block_LSE.get_data_tensor({kIsVarlen ? 1 : B, H_Q, SQ}));
}

std::tuple<at::Tensor, at::Tensor> dispatch_fmha_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const std::optional<at::Tensor>& cu_seqlens_k,
    std::optional<int64_t> max_seq_len_q,
    std::optional<int64_t> max_seq_len_k,
    std::optional<double> softmax_scale,
    bool causal,
    const std::optional<at::Tensor>& seqlen_kv,
    int64_t window_size_left,
    int64_t window_size_right,
    bool bottom_right
  ) {
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

  auto dispatch_fmha = [&](auto element, auto element_out, auto head_dim, auto varlen, auto mask) {
    return fmha_fwd<
      decltype(element),
      decltype(element_out),
      head_dim,
      varlen,
      decltype(mask)>(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seq_len_q,
        max_seq_len_k,
        softmax_scale,
        seqlen_kv,
        window_size_left,
        window_size_right
      );
  };

  auto dispatch_type = [&](auto varlen, auto mask, auto head_dim) {
    if (q.dtype() == torch::kFloat16) {
      return dispatch_fmha(
          cutlass::half_t{}, cutlass::half_t{}, head_dim, varlen, mask);
    }
    else if (q.dtype() == torch::kBFloat16) {
      return dispatch_fmha(
          cutlass::bfloat16_t{}, cutlass::bfloat16_t{}, head_dim, varlen, mask);
    }
    else if (q.dtype() == torch::kFloat8_e4m3fn) {
      // Return BF16 when input is FP8
      return dispatch_fmha(
          cutlass::float_e4m3_t{}, cutlass::bfloat16_t{}, head_dim, varlen, mask);
    }
    TORCH_CHECK(false, "Unsupported dtype for q: ", q.dtype());
  };

  auto dispatch_head_dim = [&](auto varlen, auto mask) {
    if (q.size(q.dim() - 1) == 128) {
      return dispatch_type(varlen, mask, std::integral_constant<int, 128>{});
    }
    else if (q.size(q.dim() - 1) == 64) {
      return dispatch_type(varlen, mask, std::integral_constant<int, 64>{});
    }
    else {
      TORCH_CHECK(false, "Unsupported head dim: ", q.size(q.dim() - 1));
    }
  };

  auto dispatch_mask = [&](auto varlen) {
    if (causal) {
      if (bottom_right) {
        return dispatch_head_dim(varlen, CausalMask</*kIsQBegin=*/false>{});
      }
      else {
        return dispatch_head_dim(varlen, CausalMask</*kIsQBegin=*/true>{});
      }
    }
    else if (local) {
      if (bottom_right) {
        return dispatch_head_dim(varlen, LocalMask</*kIsQBegin=*/false>{});
      }
      else {
        return dispatch_head_dim(varlen, LocalMask</*kIsQBegin=*/true>{});
      }
    }
    else if (varlen || k.size(1) % 128 != 0) {
      // Use the residual mask for varlen or when K seqlen is not multiple of
      // blockN
      return dispatch_head_dim(varlen, ResidualMask{});
    }
    else {
      return dispatch_head_dim(varlen, NoMask{});
    }
  };

  if (max_seq_len_q.has_value()) {
    return dispatch_mask(std::bool_constant<true>{});
  } else {
    return dispatch_mask(std::bool_constant<false>{});
  }
}


// -------------------------------------------------------------------------------------------------
// Op registration
// -------------------------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("fmha_fwd("
        "    Tensor query, "
        "    Tensor key, "
        "    Tensor value, "
        "    Tensor? cu_seqlens_q=None, "
        "    Tensor? cu_seqlens_k=None, "
        "    int? max_seq_len_q=None, "
        "    int? max_seq_len_k=None, "
        "    float? softmax_scale=None, "
        "    bool causal=False, "
        "    Tensor? seqlen_kv=None, "
        "    int window_size_left=-1, "
        "    int window_size_right=-1, "
        "    bool bottom_right=True"
        ") -> (Tensor, Tensor)"
  );
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("fmha_fwd", dispatch_fmha_fwd);
}
#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

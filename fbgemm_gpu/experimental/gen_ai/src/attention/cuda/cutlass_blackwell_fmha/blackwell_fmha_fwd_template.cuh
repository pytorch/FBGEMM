// @nolint
#pragma once
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
    const std::optional<const at::Tensor>& page_table,
    std::optional<int64_t> seqlen_k,
    const int window_size_left,
    const int window_size_right
  ) {
  const auto device = q.device();
  at::cuda::CUDAGuard device_guard(device);

  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;

  bool kIsPaged = false;
  if (page_table && page_table->defined()) {
    kIsPaged = true;
  }

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

  if (kIsPaged) {
    if (kIsVarlen) { // Variable length
      TORCH_CHECK(
          q.dim() == 3,
          "Expect Q shape to be (total_Q_seqlen, num_Q_heads, head_dim) ",
          "Found shape ", q.sizes());
    }
    else { // Fixed Length
      TORCH_CHECK(
          q.dim() == 4,
          "Expect Q shape to be (batch_size, Q_seqlen, num_Q_heads, head_dim). ",
          "Found shape ", q.sizes());
    }
    TORCH_CHECK(
        k.dim() == 4,
        "Expect K shape to be (num_blocks, page_block_size, num_KV_heads, head_dim) ",
        "Found shape ", k.sizes());
    TORCH_CHECK(
        v.dim() == 4,
        "Expect V shape to be (num_blocks, page_block_size, num_KV_heads, head_dim) ",
        "Found shape ", v.sizes());
    TORCH_CHECK(
        page_table.value().dim() == 2,
        "Expect page table shape to be (batch_size, max_num_blocks_per_batch)",
        "Found shape ", page_table.value().sizes());

    int tile_N = static_cast<long>(get<1>(TileShape{}).value);
    TORCH_CHECK((k.size(1) % tile_N) == 0, "Page Block Size should be divisible by N tile size");
    TORCH_CHECK((v.size(1) % tile_N) == 0, "Page Block Size should be divisible by N tile size");

    // For fixed length sequences, seqlen_k should be set.
    if (!kIsVarlen) {
        TORCH_CHECK(seqlen_k.has_value(), "seqlen_k should be set");
    }
  }
  else if (kIsVarlen) {
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
  int H_K = (kIsPaged && kIsVarlen) ? k.size(2)
          : (kIsVarlen ? k.size(1) : k.size(2)); // Number of K heads
  int D = q.size(q.dim() - 1); // Head dimension (D)

  TORCH_CHECK(H_Q % H_K == 0);
  int H_R = H_Q / H_K; // Q heads per K head
  TORCH_CHECK(D == HeadDim);

  // SQ represents SumB(Q) for varlen (jagged len)
  int SQ = kIsVarlen ? q.size(0) : q.size(1);
  int SK = kIsPaged
        ? (kIsVarlen
            ? static_cast<int>(*max_seq_len_k)
            : static_cast<int>(*seqlen_k))
        : (kIsVarlen
            ? k.size(0)
            : k.size(1));
  int B = kIsVarlen ? cu_seqlens_q->size(0) - 1 : q.size(0);

  // Parameters for paged attention.
  int page_table_stride = kIsPaged ? page_table.value().size(1) : 0;
  int num_blocks = kIsPaged ? k.size(0) : 1; // num_blocks
  int page_block_size = kIsPaged ? k.size(1) : 1; // page_block_size
  // num KV tiles > 1 within a page in the case of page_block_size > TileShapeN.
  int num_KV_tiles_per_page = kIsPaged ? k.size(1) / (get<1>(TileShape{}).value) : 1;

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
  auto k_ = (kIsPaged) ? k.reshape({num_blocks, page_block_size, H_K, 1, D}).expand({num_blocks, page_block_size, H_K, H_R, D})
                       : k.reshape({B_, SK, H_K, 1, D}).expand({B_, SK, H_K, H_R, D});
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
  // Strides expressed in logical layout, (K, D, ((H_R, H_K), B)) if non-paged
  // or (page_block_size, D, (H_R, H_K), num_blocks) if paged.
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
    if (!kIsPaged) {
        get<2, 1>(stride_K) = 0;
        get<2, 1>(stride_V) = 0;
    }
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
              kIsPaged
                ? static_cast<int*>(page_table.value().data_ptr())
                : nullptr,
              page_table_stride, num_blocks,
              page_block_size, num_KV_tiles_per_page,
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

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

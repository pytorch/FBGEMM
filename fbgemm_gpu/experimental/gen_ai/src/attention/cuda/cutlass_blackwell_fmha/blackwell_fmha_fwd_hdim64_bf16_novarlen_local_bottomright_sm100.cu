// @nolint
#include "blackwell_fmha_fwd_template.cuh"
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template std::tuple<at::Tensor, at::Tensor> fmha_fwd<
    cutlass::bfloat16_t,
    cutlass::bfloat16_t,
    64,
    false,
    LocalMask<false>
>(
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
);

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

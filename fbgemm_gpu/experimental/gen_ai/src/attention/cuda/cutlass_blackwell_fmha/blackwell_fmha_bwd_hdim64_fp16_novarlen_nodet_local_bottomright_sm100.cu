// @nolint
#include "blackwell_fmha_bwd_template.cuh"
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::half_t,
    LocalMaskForBackward<false>,
    64,
    false,
    false
>(
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
);

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

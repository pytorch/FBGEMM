// @nolint
#include "blackwell_fmha_bwd_template.cuh"
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Explicit template instantiations for BF16 data type
// These instantiations cover the main combinations used by the dispatch
// functions

// BF16 + NoMask + Head Dim 128 instantiations
template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::bfloat16_t, // Element
    NoMask, // ActiveMask
    128, // HeadDim
    false, // kIsVarlen
    false // kIsDeterministic
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
    const int window_size_right);

template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::bfloat16_t, // Element
    NoMask, // ActiveMask
    128, // HeadDim
    false, // kIsVarlen
    true // kIsDeterministic
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
    const int window_size_right);

template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::bfloat16_t, // Element
    NoMask, // ActiveMask
    128, // HeadDim
    true, // kIsVarlen
    false // kIsDeterministic
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
    const int window_size_right);

template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::bfloat16_t, // Element
    NoMask, // ActiveMask
    128, // HeadDim
    true, // kIsVarlen
    true // kIsDeterministic
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
    const int window_size_right);

// BF16 + Head Dim 64 instantiations
template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::bfloat16_t, // Element
    NoMask, // ActiveMask
    64, // HeadDim
    false, // kIsVarlen
    false // kIsDeterministic
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
    const int window_size_right);

template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::bfloat16_t, // Element
    NoMask, // ActiveMask
    64, // HeadDim
    true, // kIsVarlen
    false // kIsDeterministic
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
    const int window_size_right);

// BF16 + ResidualMaskForBackward instantiations
template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::bfloat16_t, // Element
    ResidualMaskForBackward, // ActiveMask
    128, // HeadDim
    false, // kIsVarlen
    false // kIsDeterministic
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
    const int window_size_right);

template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::bfloat16_t, // Element
    ResidualMaskForBackward, // ActiveMask
    128, // HeadDim
    true, // kIsVarlen
    false // kIsDeterministic
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
    const int window_size_right);

// BF16 + CausalForBackwardMask instantiations
template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::bfloat16_t, // Element
    CausalForBackwardMask<true>, // ActiveMask
    128, // HeadDim
    false, // kIsVarlen
    false // kIsDeterministic
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
    const int window_size_right);

template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::bfloat16_t, // Element
    CausalForBackwardMask<false>, // ActiveMask
    128, // HeadDim
    false, // kIsVarlen
    false // kIsDeterministic
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
    const int window_size_right);

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

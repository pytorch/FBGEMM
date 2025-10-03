// @nolint
#include "blackwell_fmha_bwd_template.cuh"
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Explicit template instantiations for FP8 data type
// These instantiations cover the main combinations used by the dispatch
// functions Note: FP8 input type (cutlass::float_e4m3_t) for attention kernels

// FP8 + NoMask + Head Dim 128 instantiations
template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::float_e4m3_t, // Element (FP8)
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
    cutlass::float_e4m3_t, // Element (FP8)
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
    cutlass::float_e4m3_t, // Element (FP8)
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
    cutlass::float_e4m3_t, // Element (FP8)
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

// FP8 + Head Dim 64 instantiations
template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::float_e4m3_t, // Element (FP8)
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
    cutlass::float_e4m3_t, // Element (FP8)
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

// FP8 + ResidualMaskForBackward instantiations
template std::tuple<at::Tensor, at::Tensor, at::Tensor> fmha_bwd<
    cutlass::float_e4m3_t, // Element (FP8)
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
    cutlass::float_e4m3_t, // Element (FP8)
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

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

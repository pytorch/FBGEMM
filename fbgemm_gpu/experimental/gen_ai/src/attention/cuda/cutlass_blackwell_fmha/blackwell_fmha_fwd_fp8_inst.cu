// @nolint
#include "blackwell_fmha_fwd_template.cuh"
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Explicit template instantiations for FP8 data type
// These instantiations cover the main combinations used by the dispatch
// functions Note: FP8 input with BF16 output (as seen in dispatch logic)

// FP8 + Head Dim 128 + NoMask instantiations
template std::tuple<at::Tensor, at::Tensor> fmha_fwd<
    cutlass::float_e4m3_t, // Element (FP8 input)
    cutlass::bfloat16_t, // ElementOut (BF16 output)
    128, // HeadDim
    false, // kIsVarlen
    NoMask // ActiveMask
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
    const std::optional<const at::Tensor>& page_table,
    std::optional<int64_t> seqlen_k,
    const int window_size_left,
    const int window_size_right);

template std::tuple<at::Tensor, at::Tensor> fmha_fwd<
    cutlass::float_e4m3_t, // Element (FP8 input)
    cutlass::bfloat16_t, // ElementOut (BF16 output)
    128, // HeadDim
    true, // kIsVarlen
    NoMask // ActiveMask
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
    const std::optional<const at::Tensor>& page_table,
    std::optional<int64_t> seqlen_k,
    const int window_size_left,
    const int window_size_right);

// FP8 + Head Dim 64 + NoMask instantiations
template std::tuple<at::Tensor, at::Tensor> fmha_fwd<
    cutlass::float_e4m3_t, // Element (FP8 input)
    cutlass::bfloat16_t, // ElementOut (BF16 output)
    64, // HeadDim
    false, // kIsVarlen
    NoMask // ActiveMask
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
    const std::optional<const at::Tensor>& page_table,
    std::optional<int64_t> seqlen_k,
    const int window_size_left,
    const int window_size_right);

template std::tuple<at::Tensor, at::Tensor> fmha_fwd<
    cutlass::float_e4m3_t, // Element (FP8 input)
    cutlass::bfloat16_t, // ElementOut (BF16 output)
    64, // HeadDim
    true, // kIsVarlen
    NoMask // ActiveMask
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
    const std::optional<const at::Tensor>& page_table,
    std::optional<int64_t> seqlen_k,
    const int window_size_left,
    const int window_size_right);

// FP8 + Head Dim 128 + ResidualMask instantiations
template std::tuple<at::Tensor, at::Tensor> fmha_fwd<
    cutlass::float_e4m3_t, // Element (FP8 input)
    cutlass::bfloat16_t, // ElementOut (BF16 output)
    128, // HeadDim
    false, // kIsVarlen
    ResidualMask // ActiveMask
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
    const std::optional<const at::Tensor>& page_table,
    std::optional<int64_t> seqlen_k,
    const int window_size_left,
    const int window_size_right);

template std::tuple<at::Tensor, at::Tensor> fmha_fwd<
    cutlass::float_e4m3_t, // Element (FP8 input)
    cutlass::bfloat16_t, // ElementOut (BF16 output)
    128, // HeadDim
    true, // kIsVarlen
    ResidualMask // ActiveMask
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
    const std::optional<const at::Tensor>& page_table,
    std::optional<int64_t> seqlen_k,
    const int window_size_left,
    const int window_size_right);

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

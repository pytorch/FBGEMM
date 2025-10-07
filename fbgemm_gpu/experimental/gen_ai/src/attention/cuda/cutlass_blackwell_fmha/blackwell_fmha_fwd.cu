// @nolint
#include "blackwell_fmha_fwd_template.cuh"
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

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
    bool bottom_right) {
  // Handle local attention parameters
  bool local = (window_size_left >= 0 || window_size_right >= 0);
  if (local) {
    // If causal is enabled, override window_size_right to 0 for causal+local
    // behavior
    if (causal) {
      window_size_right = 0;
      causal = false; // Use local attention instead of causal
    }
    // Expand -1 window sizes to full sequence length if available
    if (window_size_left < 0) {
      TORCH_CHECK(
          max_seq_len_k.has_value(),
          "window_size_left is negative but max_seq_len_k is not provided");
      window_size_left = max_seq_len_k.value();
    }
    if (window_size_right < 0) {
      TORCH_CHECK(
          max_seq_len_k.has_value(),
          "window_size_right is negative but max_seq_len_k is not provided");
      window_size_right = max_seq_len_k.value();
    }
  }

  auto dispatch_fmha = [&](auto element,
                           auto element_out,
                           auto head_dim,
                           auto varlen,
                           auto mask) {
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
        window_size_right);
  };

  auto dispatch_type = [&](auto varlen, auto mask, auto head_dim) {
    if (q.dtype() == torch::kFloat16) {
      return dispatch_fmha(
          cutlass::half_t{}, cutlass::half_t{}, head_dim, varlen, mask);
    } else if (q.dtype() == torch::kBFloat16) {
      return dispatch_fmha(
          cutlass::bfloat16_t{}, cutlass::bfloat16_t{}, head_dim, varlen, mask);
    } else if (q.dtype() == torch::kFloat8_e4m3fn) {
      // Return BF16 when input is FP8
      return dispatch_fmha(
          cutlass::float_e4m3_t{},
          cutlass::bfloat16_t{},
          head_dim,
          varlen,
          mask);
    }
    TORCH_CHECK(false, "Unsupported dtype for q: ", q.dtype());
  };

  auto dispatch_head_dim = [&](auto varlen, auto mask) {
    if (q.size(q.dim() - 1) == 128) {
      return dispatch_type(varlen, mask, std::integral_constant<int, 128>{});
    } else if (q.size(q.dim() - 1) == 64) {
      return dispatch_type(varlen, mask, std::integral_constant<int, 64>{});
    } else {
      TORCH_CHECK(false, "Unsupported head dim: ", q.size(q.dim() - 1));
    }
  };

  auto dispatch_mask = [&](auto varlen) {
    if (causal) {
      if (bottom_right) {
        return dispatch_head_dim(varlen, CausalMask</*kIsQBegin=*/false>{});
      } else {
        return dispatch_head_dim(varlen, CausalMask</*kIsQBegin=*/true>{});
      }
    } else if (local) {
      if (bottom_right) {
        return dispatch_head_dim(varlen, LocalMask</*kIsQBegin=*/false>{});
      } else {
        return dispatch_head_dim(varlen, LocalMask</*kIsQBegin=*/true>{});
      }
    } else if (varlen || k.size(1) % 128 != 0) {
      // Use the residual mask for varlen or when K seqlen is not multiple of
      // blockN
      return dispatch_head_dim(varlen, ResidualMask{});
    } else {
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
  m.def(
      "fmha_fwd("
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
      ") -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("fmha_fwd", dispatch_fmha_fwd);
}
#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

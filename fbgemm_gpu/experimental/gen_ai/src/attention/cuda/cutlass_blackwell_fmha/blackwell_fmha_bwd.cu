// @nolint
#include "blackwell_fmha_bwd_template.cuh"
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

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
    std::optional<double> softmax_scale,
    bool causal,
    int64_t window_size_left,
    int64_t window_size_right,
    bool bottom_right,
    bool deterministic) {
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
                           auto deterministic,
                           auto mask,
                           auto... kernel_options) {
    return fmha_bwd<
        decltype(element),
        decltype(mask),
        head_dim,
        varlen,
        deterministic,
        decltype(kernel_options)...>(
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
        softmax_scale,
        window_size_left,
        window_size_right);
  };

  auto dispatch_type =
      [&](auto varlen, auto deterministic, auto mask, auto head_dim) {
        if (query.dtype() == torch::kFloat16) {
          return dispatch_fmha(
              cutlass::half_t{},
              cutlass::half_t{},
              head_dim,
              varlen,
              deterministic,
              mask);
        } else if (query.dtype() == torch::kBFloat16) {
          return dispatch_fmha(
              cutlass::bfloat16_t{},
              cutlass::bfloat16_t{},
              head_dim,
              varlen,
              deterministic,
              mask);
        } else if (query.dtype() == torch::kFloat8_e4m3fn) {
          return dispatch_fmha(
              cutlass::float_e4m3_t{},
              cutlass::bfloat16_t{},
              head_dim,
              varlen,
              deterministic,
              mask);
        }
        TORCH_CHECK(false, "Unsupported dtype for q: ", query.dtype());
      };

  auto dispatch_head_dim = [&](auto varlen, auto deterministic, auto mask) {
    if (query.size(query.dim() - 1) == 128) {
      return dispatch_type(
          varlen, deterministic, mask, std::integral_constant<int, 128>{});
    } else if (query.size(query.dim() - 1) == 64) {
      return dispatch_type(
          varlen, deterministic, mask, std::integral_constant<int, 64>{});
    } else {
      TORCH_CHECK(false, "Unsupported head dim: ", query.size(query.dim() - 1));
    }
  };

  auto dispatch_mask = [&](auto varlen, auto deterministic) {
    if (causal) {
      if (bottom_right) {
        return dispatch_head_dim(
            varlen,
            deterministic,
            CausalForBackwardMask</*kIsQBegin=*/false>{});
      } else {
        return dispatch_head_dim(
            varlen, deterministic, CausalForBackwardMask</*kIsQBegin=*/true>{});
      }
    } else if (local) {
      if (bottom_right) {
        return dispatch_head_dim(
            varlen, deterministic, LocalMaskForBackward</*kIsQBegin=*/false>{});
      } else {
        return dispatch_head_dim(
            varlen, deterministic, LocalMaskForBackward</*kIsQBegin=*/true>{});
      }
    } else if (varlen || key.size(1) % 128 != 0) {
      // Use the residual mask for varlen or when K seqlen is not multiple of
      // blockN
      return dispatch_head_dim(
          varlen, deterministic, ResidualMaskForBackward{});
    } else {
      return dispatch_head_dim(varlen, deterministic, NoMask{});
    }
  };

  auto dispatch_deterministic = [&](auto varlen) {
    if (deterministic) {
      return dispatch_mask(varlen, std::bool_constant<true>{});
    } else {
      return dispatch_mask(varlen, std::bool_constant<false>{});
    }
  };

  if (max_seq_len_q.has_value()) {
    return dispatch_deterministic(std::bool_constant<true>{});
  } else {
    TORCH_CHECK(query.dim() == 4, "q must be [B, M, H, D] for fixed length")
    return dispatch_deterministic(std::bool_constant<false>{});
  }
}

// -------------------------------------------------------------------------------------------------
// Op registration
// -------------------------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "fmha_bwd("
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
      "    float? softmax_scale=None, "
      "    bool causal=False, "
      "    int window_size_left=-1, "
      "    int window_size_right=-1, "
      "    bool bottom_right=True, "
      "    bool deterministic=False"
      ") -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("fmha_bwd", dispatch_fmha_bwd);
}
#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

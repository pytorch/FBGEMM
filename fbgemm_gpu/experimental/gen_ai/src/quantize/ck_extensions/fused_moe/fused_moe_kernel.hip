#include <ATen/ATen.h>
#include <torch/library.h>
#include "fused_moe.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <c10/hip/HIPStream.h>

#include <atomic>
#include <cassert>
#include <cmath>
#include <vector>
#include "c10/util/Exception.h"

namespace fbgemm {

// Helper function to get raw pointer from tensor
template <typename T>
T* get_data_ptr(const at::Tensor& tensor) {
  return tensor.defined() ? tensor.data_ptr<T>() : nullptr;
}

at::Tensor fused_moe_impl(
    const at::Tensor& input, // [tokens, hidden_size]
    const at::Tensor&
        gate_up_weight, // [experts, intermediate_size, hidden_size]
    const at::Tensor& down_weight, // [experts, hidden_size, intermediate_size]
    const at::Tensor& topk_ids, // [tokens, topk]
    const at::Tensor& topk_weights, // [tokens, topk]
    const std::optional<at::Tensor> input_scales = {}, // [tokens]
    const std::optional<at::Tensor> gate_up_scales = {}, // [intermediate_size]
    const std::optional<at::Tensor> down_scales = {}, // [intermediate_size]
    const std::optional<at::Tensor> smooth_scales = {}, // [intermediate_size]
    int64_t block_m = 32,
    bool gate_only = true,
    int64_t fused_quant = 0) {

  auto output = at::empty_like(input);

  // Get dimensions
  auto tokens = input.size(0);
  auto hidden_size = input.size(1);
  auto experts = gate_up_weight.size(0);
  // Interface requires that you pass intermediate size. On |gate_only| = False,
  // |gate_up_weight| might be 2 * intermediate size, so extract the size from
  // |down_weight|
  auto intermediate_size = down_weight.size(2);
  auto topk = topk_ids.size(1);
  auto stride = input.stride(0);

  // Create workspace tensors
  int64_t max_num_tokens_padded = topk * tokens + experts * block_m - topk;
  auto sorted_token_ids =
      at::empty({max_num_tokens_padded}, topk_ids.options());
  auto sorted_weights =
      at::empty({max_num_tokens_padded}, topk_weights.options());
  auto sorted_expert_ids = at::empty(
      {(max_num_tokens_padded + block_m - 1) / block_m}, topk_ids.options());
  auto num_sorted_tiles = at::empty({1}, topk_ids.options());

  // Determine precision strings based on tensor dtypes
  auto get_prec_str = [](const at::Tensor& t) -> const char* {
    if (t.dtype() == at::kBFloat16)
      return "bf16";
    if (t.dtype() == at::kHalf)
      return "fp16";
    if (t.dtype() == at::kFloat)
      return "fp32";
    if (t.dtype() == at::kChar)
      return "int8";
    TORCH_CHECK(false, "Unsupported data type");
  };

  auto prec_i = get_prec_str(input);
  auto prec_w = get_prec_str(gate_up_weight);
  auto prec_o = get_prec_str(output);
  auto prec_tkw = get_prec_str(topk_weights);

  // Set up traits structure
  fused_moe_traits traits{
      prec_i,
      prec_w,
      prec_o,
      "fp32", // prec_st (token scale)
      "fp32", // prec_sw (weight scale)
      "fp32", // prec_sq (smooth quant)
      prec_tkw, // prec_kw (topk weight)
      static_cast<int>(block_m),
      1,
      static_cast<int>(gate_only),
      static_cast<int>(fused_quant)};

  // Set up arguments structure
  fused_moe_args args{
      input.data_ptr(),
      input_scales.has_value() ? input_scales->data_ptr() : nullptr,
      gate_up_weight.data_ptr(),
      down_weight.data_ptr(),
      gate_up_scales.has_value() ? gate_up_scales->data_ptr() : nullptr,
      down_scales.has_value() ? down_scales->data_ptr() : nullptr,
      smooth_scales.has_value() ? smooth_scales->data_ptr() : nullptr,
      nullptr,
      output.data_ptr(),
      topk_ids.data_ptr(),
      topk_weights.data_ptr(),
      sorted_token_ids.data_ptr(),
      sorted_weights.data_ptr(),
      sorted_expert_ids.data_ptr(),
      num_sorted_tiles.data_ptr(),
      static_cast<int>(block_m),
      static_cast<int>(hidden_size),
      static_cast<int>(intermediate_size),
      static_cast<int>(tokens),
      static_cast<int>(experts),
      static_cast<int>(topk),
      static_cast<int>(stride)};

  // Call kernel with default stream config
  auto stream = at::cuda::getCurrentHIPStream().stream();
  ck_tile::stream_config stream_cfg{stream};
  fused_moe(traits, args, stream_cfg);

  return output;
}

} // anonymous namespace

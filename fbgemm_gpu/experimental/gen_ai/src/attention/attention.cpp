#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm_gpu::gen_ai::attention {

std::tuple<at::Tensor, at::Tensor, at::Tensor> mqa_attn_splitk_cuda(
    const at::Tensor& XQ,
    const at::Tensor& cache_K,
    const at::Tensor& cache_V,
    const at::Tensor& seq_positions,
    const double qk_scale,
    const int64_t num_split_ks,
    const int64_t num_groups);

} // namespace fbgemm_gpu::gen_ai::attention

TORCH_LIBRARY_FRAGMENT(fbgemm_gpu, m) {
  m.def(
      "mqa_attn_splitk("
      "    Tensor XQ, "
      "    Tensor cache_K, "
      "    Tensor cache_V, "
      "    Tensor seq_positions, "
      "    float qk_scale, "
      "    int num_split_ks, "
      "    int num_int4_kv_groups=1, "
      ") -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA(
      "mqa_attn_splitk",
      fbgemm_gpu::gen_ai::attention::mqa_attn_splitk_cuda
  );
}

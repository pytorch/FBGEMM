#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

Tensor dense_to_jagged_forward(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    const c10::optional<int64_t>& total_L) {
  // D is the embedding dimension
  auto D = dense.size(-1);

  // If total_L is not given then compute it
  int64_t total_L_computed;
  if (total_L.has_value()) {
    total_L_computed = total_L.value();
  } else {
    total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
  }
  auto values = at::empty({total_L_computed, D}, dense.options());
  auto output = at::empty_like(values);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(dense.get_device());

#define DISPATCH_DENSE_TO_JAGGED_CASE(TYPE)                          \
  AT_DISPATCH_CASE(TYPE, [&] {                                       \
    jagged_dense_elementwise_jagged_output_opt_<scalar_t>(           \
        values,                                                      \
        offsets,                                                     \
        dense,                                                       \
        output,                                                      \
        [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t { \
          return y;                                                  \
        });                                                          \
  })

  // clang-format off
  AT_DISPATCH_SWITCH(
      values.scalar_type(),
      "dense_to_jagged_gpu_op_forward",
      DISPATCH_DENSE_TO_JAGGED_CASE(at::ScalarType::Half)
      DISPATCH_DENSE_TO_JAGGED_CASE(at::ScalarType::Int)
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(
          at::ScalarType::Long,
          at::ScalarType::BFloat16,
          [&] {
            jagged_dense_elementwise_jagged_output_<scalar_t>(
                values,
                offsets,
                dense,
                output,
                [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
                  return y;
                }); // device lambda
          } // lambda
          ) // CASE_FLOATING_TYPES_AND
  ); // SWITCH
  // clang-format on

#undef DISPATCH_DENSE_TO_JAGGED_CASE

  return output;
}

} // namespace fbgemm_gpu

JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "dense_to_jagged_forward",
    fbgemm_gpu::dense_to_jagged_forward);

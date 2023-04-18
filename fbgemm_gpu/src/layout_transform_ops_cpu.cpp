/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include <functional>
#include "ATen/Parallel.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

///@defgroup layout-transform-cpu Layout Transformation CPU Operators
///

namespace fbgemm_gpu {

///@ingroup layout-transform-cpu
Tensor recat_embedding_grad_output_mixed_D_cpu(
    const Tensor& grad_output, // [B_local][Sum_T_global(D)]
    const std::vector<int64_t>& dim_sum_per_rank) {
  TORCH_CHECK(grad_output.is_contiguous());
  const auto B_local = grad_output.sizes()[0];

  Tensor sharded_grad_output =
      at::empty({grad_output.numel()}, grad_output.options());

  int n = dim_sum_per_rank.size();
  std::vector<int64_t> accum_dim_sum(n + 1);
  accum_dim_sum[0] = 0;
  std::partial_sum(
      dim_sum_per_rank.begin(), dim_sum_per_rank.end(), &accum_dim_sum[1]);
  const auto global_dim_sum = accum_dim_sum[n];
  TORCH_CHECK(B_local * global_dim_sum == grad_output.numel());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "recat_embedding_gradients", [&] {
        const auto go = grad_output.accessor<scalar_t, 2>();
        auto sgo = sharded_grad_output.accessor<scalar_t, 1>();
        at::parallel_for(
            0, n * B_local, 1, [&](int64_t i_begin, int64_t i_end) {
              const auto dim_begin = i_begin / B_local;
              const auto dim_end = (i_end + B_local - 1) / B_local;
              for (const auto dim : c10::irange(dim_begin, dim_end)) {
                const auto dim_sum = dim_sum_per_rank[dim];
                const auto sgo_offset = B_local * accum_dim_sum[dim];
                scalar_t* dst = &sgo[sgo_offset];
                const scalar_t* src = &go[0][accum_dim_sum[dim]];
                const auto r_begin = (dim == dim_begin) ? i_begin % B_local : 0;
                const auto r_end = (dim == dim_end - 1 && i_end % B_local != 0)
                    ? i_end % B_local
                    : B_local;
                for (const auto r : c10::irange(r_begin, r_end)) {
                  memcpy(
                      dst + r * dim_sum,
                      src + r * global_dim_sum,
                      dim_sum * sizeof(scalar_t));
                }
              }
            });
      });

  return sharded_grad_output;
}

namespace {

using nbit::div_round_up;

std::vector<Tensor> multi_dim_split_cpu(
    const Tensor& ten,
    const std::vector<int64_t>& splits) {
  const int num_dims = splits.size();
  std::vector<int> num_splits(num_dims);
  for (const auto d : c10::irange(num_dims)) {
    num_splits[d] = div_round_up(ten.size(d), splits[d]);
  }
  const int num_total_splits = std::accumulate(
      num_splits.begin(), num_splits.end(), 1, std::multiplies<int>());

  std::vector<Tensor> out_tensors;
  out_tensors.reserve(num_dims);
  // To reduce allocation overhead, allocate one big tensor and slice.
  Tensor out_tensor_container = at::empty({ten.numel()}, ten.options());

  std::vector<int64_t> out_tensor_sizes(num_dims);
  std::vector<int64_t> offsets(num_dims);

  for (const auto split_idx : c10::irange(num_total_splits)) {
    // Compute offset and size of the split.
    auto temp_split_idx = split_idx;
    int64_t out_offset = 0;
    int numel = 1;
    for (int d = num_dims - 1; d >= 0; --d) {
      const auto split_coord = temp_split_idx % num_splits[d];
      temp_split_idx /= num_splits[d];

      out_tensor_sizes[d] =
          std::min(splits[d], ten.size(d) - split_coord * splits[d]);

      offsets[d] = split_coord * splits[d];

      out_offset = out_offset * out_tensor_sizes[d] + offsets[d] * numel;
      numel *= ten.size(d);
    }

    const auto out_numel = std::accumulate(
        out_tensor_sizes.begin(),
        out_tensor_sizes.end(),
        1,
        std::multiplies<int64_t>());

    Tensor out_tensor =
        out_tensor_container.slice(0, out_offset, out_offset + out_numel)
            .view(out_tensor_sizes);

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        ten.scalar_type(),
        "multi_dim_split",
        [&] {
          const scalar_t* ten_ptr = ten.data_ptr<scalar_t>() +
              offsets[num_dims - 1] * ten.stride(num_dims - 1);
          scalar_t* out_ptr = out_tensor.data_ptr<scalar_t>();

          for (const auto i : c10::irange(out_numel / out_tensor.size(-1))) {
            // Compute coordinate within the split.
            const scalar_t* ten_ptr_temp = ten_ptr;
            auto temp_i = i;
            for (int d = num_dims - 2; d >= 0; --d) {
              auto coord = temp_i % out_tensor.size(d);
              temp_i /= out_tensor.size(d);

              ten_ptr_temp += (offsets[d] + coord) * ten.stride(d);
            }

            memcpy(
                out_ptr + i * out_tensor.size(-1),
                ten_ptr_temp,
                out_tensor.size(-1) * sizeof(scalar_t));
          }
        });

    out_tensors.push_back(out_tensor);
  }

  return out_tensors;
}

Tensor multi_dim_cat_cpu(
    const std::vector<Tensor>& tens,
    const std::vector<int64_t>& num_splits) {
  const size_t num_total_splits = std::accumulate(
      num_splits.begin(), num_splits.end(), 1, std::multiplies<int64_t>());
  TORCH_CHECK(tens.size() == num_total_splits);

  const int num_dims = num_splits.size();
  int multiplier = 1;
  std::vector<int64_t> out_tensor_sizes(num_dims);
  for (int d = num_dims - 1; d >= 0; --d) {
    int sum = 0;
    for (int i = 0; i < num_splits[d]; ++i) {
      sum += tens[i * multiplier].size(d);
    }
    out_tensor_sizes[d] = sum;
    multiplier *= num_splits[d];
  }

  TORCH_CHECK(tens.size() > 0);
  Tensor out_tensor = at::empty(out_tensor_sizes, tens[0].options());

  std::vector<int64_t> offsets(num_dims);

  for (const auto split_idx : c10::irange(num_total_splits)) {
    // Compute offset and size of the split.
    auto temp_split_idx = split_idx;
    Tensor ten = *tens[split_idx].expect_contiguous();
    for (int d = num_dims - 1; d >= 0; --d) {
      const int split_coord = temp_split_idx % num_splits[d];
      temp_split_idx /= num_splits[d];

      const int64_t split_size =
          div_round_up(out_tensor.size(d), num_splits[d]);
      TORCH_CHECK(
          std::min(split_size, out_tensor.size(d) - split_coord * split_size) ==
          ten.size(d));

      offsets[d] = split_coord * split_size;
    }

    auto numel = ten.numel();

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        out_tensor.scalar_type(),
        "multi_dim_split",
        [&] {
          const scalar_t* ten_ptr = ten.data_ptr<scalar_t>();
          scalar_t* out_ptr = out_tensor.data_ptr<scalar_t>() +
              offsets[num_dims - 1] * out_tensor.stride(num_dims - 1);

          for (const auto i : c10::irange(numel / ten.size(-1))) {
            // Compute coordinate within the split.
            scalar_t* out_ptr_temp = out_ptr;
            auto temp_i = i;
            for (int d = num_dims - 2; d >= 0; --d) {
              auto coord = temp_i % ten.size(d);
              temp_i /= ten.size(d);

              out_ptr_temp += (offsets[d] + coord) * out_tensor.stride(d);
            }

            memcpy(
                out_ptr_temp,
                ten_ptr + i * ten.size(-1),
                ten.size(-1) * sizeof(scalar_t));
          }
        });
  }

  return out_tensor;
}

} // namespace

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "recat_embedding_grad_output_mixed_D_batch(Tensor grad_output, Tensor dim_sum_per_rank, Tensor cumsum_dim_sum_per_rank) -> Tensor");
  m.def(
      "recat_embedding_grad_output_mixed_D(Tensor grad_output, int[] dim_sum_per_rank) -> Tensor");
  m.def(
      "recat_embedding_grad_output(Tensor grad_output, int[] num_features_per_rank) -> Tensor");
  // multi-dimensional version of torch.split . Python ref code would be
  // split_ten = [ten]
  // for dim, split in enumerate(splits):
  //   temp_split = split_ten
  //   split_ten = []
  //   for t in temp_split:
  //     split_ten.extend(torch.split(t, split, dim=dim))
  // return split_ten
  m.def("multi_dim_split(Tensor ten, int[] splits) -> Tensor[]");
  // multi-dimensional version of torch.cat . Python ref code would be
  // merged_ten = tens
  // for dim, split in reversed(list(enumerate(num_splits))):
  //   temp_split = []
  //   for i in range(0, len(merged_ten), split):
  //     temp_split.append(torch.cat(merged_ten[i : i + split], dim=dim))
  //   merged_ten = temp_split
  // return merged_ten[0]
  m.def("multi_dim_cat(Tensor[] tens, int[] num_splits) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  DISPATCH_TO_CPU(
      "recat_embedding_grad_output_mixed_D",
      fbgemm_gpu::recat_embedding_grad_output_mixed_D_cpu);
  DISPATCH_TO_CPU("multi_dim_split", fbgemm_gpu::multi_dim_split_cpu);
  DISPATCH_TO_CPU("multi_dim_cat", fbgemm_gpu::multi_dim_cat_cpu);
}

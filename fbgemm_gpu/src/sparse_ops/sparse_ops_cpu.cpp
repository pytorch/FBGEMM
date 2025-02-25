/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/MaybeOwned.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "common.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

namespace {

// To avoid multiple threads are touching the same cache line.
// Assume cache line size is 64B and element size is at least 4B like float or
// int32.
constexpr int FALSE_SHARING_PAD = 16;

// Converts sparse tensor to dense tensor with few optimizations to be used with
// histogram binning calibration by feature. (1) Assumes dense_last_dim == 1 (2)
// Does not update default value when length > 1. HBC by feature has a separate
// logic to handle this, but we fold it over here.
template <typename SegmentValueType, typename SegmentLengthType>
void _to_dense_representation(
    const int64_t num_lengths,
    const SegmentValueType* const segment_value_data,
    const SegmentLengthType* const segment_lengths_data,
    SegmentValueType* const dense_segment_value_data) {
  int k = 0;
  for (const auto i : c10::irange(num_lengths)) {
    if (segment_lengths_data[i] == 1) {
      // Add 1 to distinguish between 0 inserted by densification vs. original
      // value.
      dense_segment_value_data[i] = segment_value_data[k] + 1;
    } else {
      dense_segment_value_data[i] = 0;
    }
    k += segment_lengths_data[i];
  }
}

} // namespace

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Custom PackSegments operator that is based on the Caffe2 PackSegments and
// UnpackSegments.
// Needed this to support backward pass.
class PackSegments : public torch::autograd::Function<PackSegments> {
 public:
  static constexpr bool is_traceable = true;
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& t_in,
      const Tensor& lengths,
      at::SymInt max_length) {
    const at::SymInt total_length = t_in.sym_size(0);

    at::AutoDispatchBelowADInplaceOrView guard;

    static auto custom_pack_segments_op =
        at::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::pack_segments", "")
            .typed<at::Tensor(
                const at::Tensor&, const at::Tensor&, const at::SymInt)>();

    Tensor res = custom_pack_segments_op.call(t_in, lengths, max_length);

    ctx->saved_data["max_length"] = max_length;
    ctx->saved_data["total_length"] = total_length;
    ctx->save_for_backward({lengths});

    return {res};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    TORCH_CHECK(grad_output.size() == 2 || grad_output.size() == 1);
    const Tensor& grad = grad_output[0];
    const auto& max_length = ctx->saved_data["max_length"].toSymInt();
    const auto& total_length = ctx->saved_data["total_length"].toSymInt();

    // Retrieve saved variables for backward.
    const auto& saved_variables = ctx->get_saved_variables();
    const auto& lengths = saved_variables[0];

    torch::autograd::variable_list grad_inputs(5);

    static auto custom_pack_segments_backward_op =
        at::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::pack_segments_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::SymInt,
                const at::SymInt)>();

    grad_inputs[0] = custom_pack_segments_backward_op.call(
        grad, lengths, total_length, max_length);
    return grad_inputs;
  }
};

class PackSegmentsV2 : public torch::autograd::Function<PackSegmentsV2> {
 public:
  static constexpr bool is_traceable = true;
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& t_in,
      const Tensor& lengths,
      at::SymInt max_length,
      const bool pad_minf,
      const bool return_presence_mask) {
    const at::SymInt total_length = t_in.sym_size(0);

    at::AutoDispatchBelowADInplaceOrView guard;

    static auto custom_pack_segments_op =
        at::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::pack_segments_v2", "")
            .typed<std::tuple<Tensor, std::optional<Tensor>>(
                const at::Tensor&,
                const at::Tensor&,
                const at::SymInt,
                const bool,
                const bool)>();

    const auto& res = custom_pack_segments_op.call(
        t_in, lengths, max_length, pad_minf, return_presence_mask);

    ctx->saved_data["max_length"] = max_length;
    ctx->saved_data["total_length"] = total_length;
    ctx->save_for_backward({lengths});

    if (return_presence_mask) {
      return {std::get<0>(res), std::get<1>(res).value()};
    }
    return {std::get<0>(res)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    TORCH_CHECK(grad_output.size() == 2 or grad_output.size() == 1);
    const Tensor& grad = grad_output[0];
    const auto& max_length = ctx->saved_data["max_length"].toSymInt();
    const auto& total_length = ctx->saved_data["total_length"].toSymInt();

    // Retrieve saved variables for backward.
    const auto& saved_variables = ctx->get_saved_variables();
    const auto& lengths = saved_variables[0];

    torch::autograd::variable_list grad_inputs(5);

    static auto custom_pack_segments_backward_op =
        at::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::pack_segments_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::SymInt,
                const at::SymInt)>();

    grad_inputs[0] = custom_pack_segments_backward_op.call(
        grad, lengths, total_length, max_length);
    return grad_inputs;
  }
};

Tensor pack_segments_autograd(
    const Tensor& t_in,
    const Tensor& lengths,
    const at::SymInt max_length

) {
  return PackSegments::apply(t_in, lengths, max_length)[0];
}

std::tuple<Tensor, std::optional<Tensor>> pack_segments_autograd_v2(
    const Tensor& t_in,
    const Tensor& lengths,
    const at::SymInt max_length,
    const bool pad_minf,
    const bool return_presence_mask) {
  const auto& res = PackSegmentsV2::apply(
      t_in, lengths, max_length, pad_minf, return_presence_mask);
  std::optional<Tensor> presence_mask;
  if (return_presence_mask) {
    presence_mask = res[1];
  }
  return {res[0], presence_mask};
}

template <typename T>
void prefix_sum(const int length, const T* const array, T* const presum) {
  presum[0] = 0;
  for (const auto i : c10::irange(length)) {
    presum[i + 1] = array[i] + presum[i];
  }
}

// NOTE : _permute_indices_weights_kernel_cpu and _permute_lengths_cpu_kernel
// have to use the same grain size for consistent partitioning across threads.
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
void _permute_2D_indices_weights_kernel_cpu(
    const int32_t T,
    const int32_t B,
    const indices_t* const __restrict__ indices,
    const weights_t* const __restrict__ weights,
    const int32_t* const __restrict__ permute,
    const offsets_t* const __restrict__ input_offsets,
    const int64_t* const __restrict__ output_offsets_per_thread_cumsum,
    indices_t* const __restrict__ permuted_indices,
    weights_t* const __restrict__ permuted_weights,
    const offsets_t* const __restrict__ permuted_lengths) {
  at::parallel_for(
      0, T * B, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        offsets_t output_start = output_offsets_per_thread_cumsum
            [at::get_thread_num() * FALSE_SHARING_PAD];
        int64_t t_begin = tb_begin / B;
        int64_t t_end = (tb_end + B - 1) / B;
        for (const auto t : c10::irange(t_begin, t_end)) {
          int64_t b_begin = (t == t_begin) ? tb_begin % B : 0;
          int64_t b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;
          for (const auto b : c10::irange(b_begin, b_end)) {
            offsets_t permuted_length = permuted_lengths[t * B + b];
            const offsets_t input_start = input_offsets[permute[t] * B + b];
            for (const auto i : c10::irange(permuted_length)) {
              permuted_indices[output_start + i] = indices[input_start + i];
              if (has_weight) {
                permuted_weights[output_start + i] = weights[input_start + i];
              }
            }
            output_start += permuted_length;
          } // for each b
        } // for each t
      }); // parallel_for T * B
}

template <typename index_t>
void _permute_2D_lengths_cpu_kernel(
    const int32_t T,
    const int32_t B,
    const index_t* const __restrict__ lengths,
    int64_t lengths_size,
    const int32_t* const __restrict__ permute,
    index_t* const __restrict__ permuted_lengths,
    index_t* const __restrict__ input_offsets,
    int64_t* const __restrict__ output_offsets_per_thread_cumsum) {
  int num_threads = at::get_num_threads();
  std::vector<int> input_offsets_per_thread_cumsum(
      (num_threads + 1) * FALSE_SHARING_PAD, 0);

  // First parallel for: populate permuted_lengths, and compute per-thread
  // summation of lengths (input_offsets_per_thread_cumsum) and permuted_lengths
  // (output_offsets_per_thread_cumsum)
  at::parallel_for(
      0, T * B, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        index_t current_input_offset = 0;
        // Have a separate loop for summing up lengths because lengths_size
        // can be smaller than T * B.
        for (int tb = tb_begin; tb < std::min(tb_end, lengths_size); ++tb) {
          current_input_offset += lengths[tb];
        }

        index_t current_output_offset = 0;
        int64_t t_begin = tb_begin / B;
        int64_t t_end = (tb_end + B - 1) / B;
        for (const auto t : c10::irange(t_begin, t_end)) {
          int64_t b_begin = (t == t_begin) ? tb_begin % B : 0;
          int64_t b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;
          for (const auto b : c10::irange(b_begin, b_end)) {
            auto permuted_length = lengths[permute[t] * B + b];
            permuted_lengths[t * B + b] = permuted_length;
            current_output_offset += permuted_length;
          }
        }
        input_offsets_per_thread_cumsum
            [(at::get_thread_num() + 1) * FALSE_SHARING_PAD] =
                current_input_offset;
        output_offsets_per_thread_cumsum
            [(at::get_thread_num() + 1) * FALSE_SHARING_PAD] =
                current_output_offset;
      });

  // Inter-thread reduction
  for (const auto t : c10::irange(1, num_threads)) {
    input_offsets_per_thread_cumsum[(t + 1) * FALSE_SHARING_PAD] +=
        input_offsets_per_thread_cumsum[t * FALSE_SHARING_PAD];
    output_offsets_per_thread_cumsum[(t + 1) * FALSE_SHARING_PAD] +=
        output_offsets_per_thread_cumsum[t * FALSE_SHARING_PAD];
  }

  // Second parallel for: populate input_offsets
  // NOTE: this works assuming the partitioning will be the same as the
  // first parallel_for.
  at::parallel_for(
      0, T * B, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        index_t current_input_offset = input_offsets_per_thread_cumsum
            [at::get_thread_num() * FALSE_SHARING_PAD];
        if (tb_begin < lengths_size) {
          input_offsets[tb_begin] = current_input_offset;
        }
        for (const auto tb :
             c10::irange(tb_begin, std::min(tb_end - 1, lengths_size))) {
          current_input_offset += lengths[tb];
          input_offsets[tb + 1] = current_input_offset;
        }
      });
  if (lengths_size >= T * B) {
    input_offsets[T * B] =
        input_offsets_per_thread_cumsum[num_threads * FALSE_SHARING_PAD];
  }

  // Handle cases when lengths_size > T * B
  for (const auto i : c10::irange(T * B, lengths_size)) {
    input_offsets[i + 1] = lengths[i] + input_offsets[i];
  }
}

template <
    bool sequence,
    bool has_weight,
    bool return_bucket_mapping,
    typename offset_t,
    typename index_t,
    typename scalar_t>
void _block_bucketize_sparse_features_cpu_kernel(
    const Tensor& lengths,
    const Tensor& indices,
    const std::optional<Tensor>& weights,
    const bool bucketize_pos,
    const Tensor& block_sizes,
    const std::optional<Tensor>& total_num_blocks,
    const int64_t my_size,
    Tensor new_lengths,
    Tensor new_indices,
    std::optional<Tensor> new_weights,
    std::optional<Tensor> new_pos,
    const std::optional<Tensor>& unbucketize_permute,
    const std::optional<Tensor>& batch_size_per_feature,
    const std::optional<std::vector<at::Tensor>>& block_bucketize_pos,
    const std::optional<Tensor>& bucket_mapping,
    const bool keep_orig_idx) {
  // allocate tensors and buffers
  const auto lengths_size = lengths.numel();
  const auto new_lengths_size = lengths_size * my_size;
  const int32_t T = block_sizes.numel();
  const int32_t B = lengths_size / T;
  auto offsets = at::empty({lengths_size + 1}, lengths.options());
  auto new_offsets = at::empty({new_lengths_size + 1}, lengths.options());
  const offset_t* lengths_data = lengths.data_ptr<offset_t>();
  offset_t* offsets_data = offsets.data_ptr<offset_t>();
  const index_t* indices_data = indices.data_ptr<index_t>();
  scalar_t* weights_data = nullptr;
  scalar_t* new_weights_data = nullptr;
  index_t* new_pos_data = nullptr;
  index_t* unbucketize_permute_data = nullptr;
  index_t* bag_mapping_data = nullptr;
  offset_t* const new_lengths_data = new_lengths.data_ptr<offset_t>();
  offset_t* const new_offsets_data = new_offsets.data_ptr<offset_t>();
  index_t* const new_indices_data = new_indices.data_ptr<index_t>();
  const index_t* const block_sizes_data = block_sizes.data_ptr<index_t>();
  offset_t* batch_sizes_data = nullptr;
  const auto variable_batch_size = batch_size_per_feature.has_value();
  const auto variable_bucket_sizes = block_bucketize_pos.has_value() &&
      block_bucketize_pos.value().size() != 0;
  using uindex_t = std::make_unsigned_t<index_t>;
  using uoffset_t = std::make_unsigned_t<offset_t>;
  std::vector<int64_t> lower_bounds(indices.numel(), 0);

  if constexpr (sequence) {
    unbucketize_permute_data = unbucketize_permute.value().data_ptr<index_t>();
    if constexpr (return_bucket_mapping) {
      bag_mapping_data = bucket_mapping.value().data_ptr<index_t>();
    }
  }
  if constexpr (has_weight) {
    weights_data = weights.value().data_ptr<scalar_t>();
    new_weights_data = new_weights.value().data_ptr<scalar_t>();
  }
  if (bucketize_pos) {
    new_pos_data = new_pos.value().data_ptr<index_t>();
  }

  if (variable_batch_size) {
    batch_sizes_data = batch_size_per_feature.value().data_ptr<offset_t>();
  }

  // count nonzeros
  prefix_sum(lengths_size, lengths_data, offsets_data);
  assert(offsets_data[lengths_size] == indices.numel());
  int64_t cur_offset = 0;
  for (const auto t : c10::irange(T)) {
    const auto blk_size = block_sizes_data[t];
    const auto cur_batch_size = variable_batch_size ? batch_sizes_data[t] : B;
    const index_t* bucketize_offset = nullptr;
    const index_t local_num_blks = total_num_blocks.has_value()
        ? (total_num_blocks.value().data_ptr<index_t>()[t] / my_size)
        : 1;
    const index_t global_num_blks = total_num_blocks.has_value()
        ? (total_num_blocks.value().data_ptr<index_t>()[t])
        : my_size;
    const index_t global_idx_size = blk_size * global_num_blks;
    const index_t local_idx_size = blk_size * local_num_blks;
    int64_t bucket_size = 0;
    if (variable_bucket_sizes) {
      bucketize_offset = block_bucketize_pos.value()[t].data_ptr<index_t>();
      bucket_size = block_bucketize_pos.value()[t].numel();
    }
    for (const auto b : c10::irange(cur_batch_size)) {
      const auto b_t = (variable_batch_size ? cur_offset : t * B) + b;
      const offset_t rowstart = offsets_data[b_t];
      const offset_t rowend = offsets_data[b_t + 1];
      for (const auto i : c10::irange(rowstart, rowend)) {
        // We have use cases using none-hashed raw indices that can be either
        // negative or larger than embedding table hash_size (blk_size *
        // my_size). In cases of none-hashed indices we need to ensure
        // bucketization can distribute them into different ranks and within
        // range of blk_size, we expect the later embedding module to take care
        // of hashing indices calculation.
        uindex_t idx = static_cast<uindex_t>(indices_data[i]);
        if (variable_bucket_sizes) {
          if (blk_size == 0) {
            idx = (idx % global_num_blks) * bucketize_offset[bucket_size - 1] /
                global_num_blks;
          }
          int64_t lb = std::upper_bound(
                           bucketize_offset,
                           bucketize_offset + static_cast<index_t>(bucket_size),
                           idx) -
              bucketize_offset - 1;
          lower_bounds[i] = lb;
          uindex_t p = lb < my_size ? lb : idx % my_size;
          new_lengths_data[p * lengths_size + b_t]++;
        } else {
          uindex_t p = idx < static_cast<uindex_t>(global_idx_size)
              ? idx / local_idx_size
              : (idx % global_num_blks) / local_num_blks;
          new_lengths_data[p * lengths_size + b_t]++;
        }
      }
    }
    cur_offset += cur_batch_size;
  }

  // bucketize nonzeros
  prefix_sum(new_lengths_size, new_lengths_data, new_offsets_data);
  assert(new_offsets_data[new_lengths_size] == new_indices.numel());
  cur_offset = 0;
  for (const auto t : c10::irange(T)) {
    const auto blk_size = block_sizes_data[t];
    const auto cur_batch_size = variable_batch_size ? batch_sizes_data[t] : B;
    const index_t* bucketize_offset = nullptr;
    if (variable_bucket_sizes) {
      bucketize_offset = block_bucketize_pos.value()[t].data_ptr<index_t>();
    }
    const index_t local_num_blks = total_num_blocks.has_value()
        ? (total_num_blocks.value().data_ptr<index_t>()[t] / my_size)
        : 1;
    const index_t global_num_blks = total_num_blocks.has_value()
        ? (total_num_blocks.value().data_ptr<index_t>()[t])
        : my_size;
    const index_t global_idx_size = blk_size * global_num_blks;
    const index_t local_idx_size = blk_size * local_num_blks;
    for (const auto b : c10::irange(cur_batch_size)) {
      const auto b_t = (variable_batch_size ? cur_offset : t * B) + b;
      const offset_t rowstart = offsets_data[b_t];
      const offset_t rowend = offsets_data[b_t + 1];
      for (const auto i : c10::irange(rowstart, rowend)) {
        // We have use cases using none-hashed raw indices that can be either
        // negative or larger than embedding table hash_size (blk_size *
        // my_size). In cases of none-hashed indices we need to ensure
        // bucketization can distribute them into different ranks and within
        // range of blk_size, we expect the later embedding module to take care
        // of hashing indices calculation.
        const uindex_t idx = static_cast<uindex_t>(indices_data[i]);
        uindex_t p, new_idx;
        if (variable_bucket_sizes) {
          int64_t lb = lower_bounds[i];
          p = lb < my_size ? lb : idx % my_size;
          if (keep_orig_idx) {
            new_idx = idx;
          } else if (blk_size == 0) {
            new_idx = idx / global_num_blks;
          } else if (lb < my_size) {
            new_idx = idx - bucketize_offset[lb];
          } else {
            new_idx = idx / my_size;
          }
        } else { // uniform bucket size

          const uindex_t ub = static_cast<uindex_t>(global_idx_size);
          p = idx < ub ? idx / local_idx_size
                       : (idx % global_num_blks) / local_num_blks;
          if (keep_orig_idx) {
            new_idx = idx;
          } else if (idx < ub) {
            new_idx = idx % local_idx_size;
          } else {
            new_idx = idx / global_num_blks;
          }
        }
        const uoffset_t pos = new_offsets_data[p * lengths_size + b_t];
        new_indices_data[pos] = new_idx;
        if (sequence) {
          unbucketize_permute_data[i] = pos;
          if constexpr (return_bucket_mapping) {
            bag_mapping_data[i] = p;
          }
        }
        new_offsets_data[p * lengths_size + b_t]++;
        if (has_weight) {
          new_weights_data[pos] = weights_data[i];
        }
        if (bucketize_pos) {
          new_pos_data[pos] = i - rowstart;
        }
      }
    }
    cur_offset += cur_batch_size;
  }
}

void FloatToBFloat16Quantized_ref(
    const float* const input,
    const size_t numel,
    uint16_t* const output) {
  for (const auto idx : c10::irange(numel)) {
    const float* input_elem = input + idx;
    uint16_t* output_elem = output + idx;
    *output_elem =
        (*reinterpret_cast<const uint32_t*>(input_elem) + (1 << 15)) >> 16;
  }
}

void BFloat16QuantizedToFloat_ref(
    const at::BFloat16* const input,
    const size_t numel,
    float* const output) {
  for (const auto idx : c10::irange(numel)) {
    const at::BFloat16* input_elem = input + idx;
    float* output_elem = output + idx;

    uint32_t val_fp32 =
        static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(input_elem))
        << 16;
    *reinterpret_cast<uint32_t*>(output_elem) = val_fp32;
  }
}

// TODO: replace Half by BFloat16, after BFloat16 is supported by Nvidia NCCL
at::Tensor _float_to_bfloat16_cpu(const at::Tensor& input) {
  TENSOR_ON_CPU(input);

  const auto input_sizes = input.sizes();
  auto output = at::empty(
      input_sizes,
      input.options().dtype(at::kHalf)); // at::kHalf

  FloatToBFloat16Quantized_ref(
      input.data_ptr<float>(),
      input.numel(),
      reinterpret_cast<uint16_t*>(output.data_ptr<at::Half>()));

  return output;
}

// TODO: replace Half by BFloat16, after BFloat16 is supported by Nvidia NCCL
at::Tensor _bfloat16_to_float_cpu(const at::Tensor& input) {
  TENSOR_ON_CPU(input);

  const auto input_sizes = input.sizes();

  auto output = at::empty(input_sizes, input.options().dtype(at::kFloat));

  BFloat16QuantizedToFloat_ref(
      reinterpret_cast<at::BFloat16*>(input.data_ptr<at::Half>()),
      input.numel(),
      output.data_ptr<float>());

  return output;
}

// This function partitions sparse features
// cyclically along the sparse dimension into my_size blocks
template <bool has_weight, typename index_t, typename scalar_t>
void _bucketize_sparse_features_cpu(
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& weights,
    const bool bucketize_pos,
    const int64_t my_size,
    at::Tensor& new_lengths,
    at::Tensor& new_indices,
    std::optional<at::Tensor> new_weights,
    std::optional<at::Tensor> new_pos) {
  TENSOR_ON_CPU(lengths);
  TENSOR_ON_CPU(indices);
  TENSOR_EMPTY_OR_ON_CPU(weights);
  TENSOR_ON_CPU(new_lengths);
  TENSOR_ON_CPU(new_indices);
  TENSOR_EMPTY_OR_ON_CPU(new_weights);
  TENSOR_EMPTY_OR_ON_CPU(new_pos);
  using uindex_t = std::make_unsigned_t<index_t>;

  // allocate tensors and buffers
  const auto lengths_size = lengths.numel();
  const auto new_lengths_size = lengths_size * my_size;
  auto offsets = at::empty({lengths_size + 1}, lengths.options());
  auto new_offsets = at::empty({new_lengths_size + 1}, lengths.options());
  const index_t* lengths_data = lengths.data_ptr<index_t>();
  index_t* offsets_data = offsets.data_ptr<index_t>();
  const index_t* indices_data = indices.data_ptr<index_t>();
  scalar_t* weights_data;
  scalar_t* new_weights_data;
  index_t* new_pos_data;

  index_t* const new_lengths_data = new_lengths.data_ptr<index_t>();
  index_t* const new_offsets_data = new_offsets.data_ptr<index_t>();
  index_t* const new_indices_data = new_indices.data_ptr<index_t>();

  if (has_weight) {
    weights_data = weights.value().data_ptr<scalar_t>();
    new_weights_data = new_weights.value().data_ptr<scalar_t>();
  }
  if (bucketize_pos) {
    new_pos_data = new_pos.value().data_ptr<index_t>();
  }
  // count nonzeros
  prefix_sum(lengths_size, lengths_data, offsets_data);
  assert(offsets_data[lengths_size] == indices.numel());
  for (const auto r : c10::irange(lengths_size)) {
    const index_t rowstart = offsets_data[r];
    const index_t rowend = offsets_data[r + 1];
    for (const auto i : c10::irange(rowstart, rowend)) {
      // Need to handle negative indices if we use raw idices instead of hashed
      // indices, convert to unsigned
      const uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      const uindex_t p = idx % my_size;
      new_lengths_data[p * lengths_size + r]++;
    }
  }
  // bucketize nonzeros
  prefix_sum(new_lengths_size, new_lengths_data, new_offsets_data);
  assert(new_offsets_data[new_lengths_size] == new_indices.numel());
  for (const auto r : c10::irange(lengths_size)) {
    const index_t rowstart = offsets_data[r];
    const index_t rowend = offsets_data[r + 1];
    for (const auto i : c10::irange(rowstart, rowend)) {
      // Need to handle negative indices if we use raw idices instead of hashed
      // indices, convert to unsigned
      const uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      const uindex_t p = idx % my_size;
      const uindex_t new_idx = idx / my_size;
      const uindex_t pos = new_offsets_data[p * lengths_size + r];
      new_indices_data[pos] = new_idx;
      new_offsets_data[p * lengths_size + r]++;
      if (has_weight) {
        new_weights_data[pos] = weights_data[i];
      }
      if (bucketize_pos) {
        new_pos_data[pos] = i - rowstart;
      }
    }
  }
}

std::tuple<Tensor, Tensor, std::optional<Tensor>>
permute_2D_sparse_data_input1D_cpu(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const int64_t& stride,
    const std::optional<Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum) {
  auto [permuted_lengths, permuted_indices, permuted_weights] =
      permute_2D_sparse_data_cpu(
          permute,
          lengths.view({-1, stride}),
          indices,
          weights,
          permuted_lengths_sum);
  return {permuted_lengths.view(-1), permuted_indices, permuted_weights};
}

std::tuple<Tensor, Tensor, std::optional<Tensor>> permute_2D_sparse_data_cpu(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const std::optional<Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum) {
  TENSOR_ON_CPU(permute);
  TENSOR_ON_CPU(lengths);
  TENSOR_ON_CPU(indices);
  if (weights) {
    TENSOR_ON_CPU(weights);
  }
  TORCH_CHECK(lengths.dim() == 2);

  const auto permute_contig = permute.expect_contiguous();
  const auto lengths_contig = lengths.expect_contiguous();
  const auto indices_contig = indices.expect_contiguous();
  // the data to permute over can be less or more with or without
  // repetitions
  const auto T = permute.numel();
  const auto B = lengths.size(1);

  Tensor permuted_lengths;
  Tensor permuted_indices;
  std::optional<Tensor> permuted_weights;

  permuted_lengths = at::empty({T, B}, lengths.options());

  const auto lengths_size = lengths.numel();
  auto input_offsets = at::empty({lengths_size + 1}, lengths.options());

  int num_threads = at::get_num_threads();
  std::vector<int64_t> output_offsets_per_thread_cumsum(
      (num_threads + 1) * FALSE_SHARING_PAD, 0);

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_2D_lengths_cpu_kernel", [&] {
        _permute_2D_lengths_cpu_kernel(
            T,
            B,
            lengths_contig->data_ptr<index_t>(),
            lengths_size,
            permute.data_ptr<int32_t>(),
            permuted_lengths.data_ptr<index_t>(),
            input_offsets.data_ptr<index_t>(),
            output_offsets_per_thread_cumsum.data());
      }); // for each scalar_t

  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size =
        output_offsets_per_thread_cumsum[num_threads * FALSE_SHARING_PAD];
  }
  permuted_indices = at::empty(permuted_indices_size, indices.options());
  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_2D_indices_weights_kernel_1", [&] {
        using offsets_t = index_t;
        FBGEMM_DISPATCH_ALL_TYPES(
            indices.scalar_type(), "permute_2D_indices_weights_kernel_2", [&] {
              using indices_t = scalar_t;
              FBGEMM_DISPATCH_FLOAT_ONLY(
                  weights.has_value() ? weights.value().scalar_type()
                                      : at::ScalarType::Float,
                  "permute_2D_indices_weights_kernel_3",
                  [&] {
                    using weights_t = scalar_t;
                    if (weights.has_value()) {
                      const auto weights_value_contig =
                          weights.value().expect_contiguous();
                      permuted_weights = at::empty(
                          permuted_indices_size, weights.value().options());
                      _permute_2D_indices_weights_kernel_cpu<
                          true,
                          index_t,
                          indices_t,
                          weights_t>(
                          T,
                          B,
                          indices_contig->data_ptr<indices_t>(),
                          weights_value_contig->data_ptr<weights_t>(),
                          permute_contig->data_ptr<int32_t>(),
                          input_offsets.data_ptr<offsets_t>(),
                          output_offsets_per_thread_cumsum.data(),
                          permuted_indices.data_ptr<indices_t>(),
                          permuted_weights->data_ptr<weights_t>(),
                          permuted_lengths.data_ptr<offsets_t>());
                    } else {
                      _permute_2D_indices_weights_kernel_cpu<
                          false,
                          index_t,
                          indices_t,
                          weights_t>(
                          T,
                          B,
                          indices_contig->data_ptr<indices_t>(),
                          nullptr,
                          permute_contig->data_ptr<int32_t>(),
                          input_offsets.data_ptr<offsets_t>(),
                          output_offsets_per_thread_cumsum.data(),
                          permuted_indices.data_ptr<indices_t>(),
                          nullptr,
                          permuted_lengths.data_ptr<offsets_t>());
                    }
                  }); // for each weights_t
            }); // for each indices_t
      }); // for each offsets_t
  return {permuted_lengths, permuted_indices, permuted_weights};
}

// specialization for variable B and T,
// the permute here maps to all items in length.
template <typename index_t>
void _permute_1D_lengths_cpu_kernel(
    const index_t* const __restrict__ lengths,
    int64_t permuted_lengths_size,
    const int32_t* const __restrict__ permute,
    index_t* const __restrict__ permuted_lengths) {
  at::parallel_for(
      0,
      permuted_lengths_size,
      FALSE_SHARING_PAD,
      [&](int64_t tb_begin, int64_t tb_end) {
        // Have a separate loop for summing up lengths
        index_t current_output_offset = 0;
        for (int tb = tb_begin; tb < std::min(tb_end, permuted_lengths_size);
             ++tb) {
          auto permuted_length = lengths[permute[tb]];
          permuted_lengths[tb] = permuted_length;
          current_output_offset += permuted_length;
        }
      });
}

// specialization for variable B and T,
// the permute here maps to all items in length.
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
void _permute_1D_indices_weights_kernel_cpu(
    const offsets_t* const __restrict__ input_offsets,
    const indices_t* const __restrict__ indices,
    const weights_t* const __restrict__ weights,
    const int64_t permuted_lengths_size,
    const int32_t* const __restrict__ permute,
    const offsets_t* const __restrict__ permuted_lengths,
    const offsets_t* const __restrict__ output_offsets,
    indices_t* const __restrict__ permuted_indices,
    weights_t* const __restrict__ permuted_weights) {
  at::parallel_for(
      0,
      permuted_lengths_size,
      FALSE_SHARING_PAD,
      [&](int64_t tb_begin, int64_t tb_end) {
        for (int tb = tb_begin; tb < std::min(tb_end, permuted_lengths_size);
             ++tb) {
          offsets_t permuted_length = permuted_lengths[tb];
          const offsets_t input_start = input_offsets[permute[tb]];
          const offsets_t output_start = output_offsets[tb];
          for (const auto i : c10::irange(permuted_length)) {
            permuted_indices[output_start + i] = indices[input_start + i];
            if (has_weight) {
              permuted_weights[output_start + i] = weights[input_start + i];
            }
          }
        }
      }); // parallel_for T x B, different B across T
}

std::tuple<Tensor, Tensor, std::optional<Tensor>> permute_1D_sparse_data_cpu(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const std::optional<Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum) {
  TENSOR_ON_CPU(permute);
  TENSOR_ON_CPU(lengths);
  TENSOR_ON_CPU(indices);
  TENSOR_ON_CPU(weights);

  const auto permute_contig = permute.expect_contiguous();
  const auto lengths_contig = lengths.expect_contiguous();
  const auto indices_contig = indices.expect_contiguous();
  // the data to permute over can be less or more with or without
  // repetitions
  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;

  const auto permuted_lengths_size = permute.numel();
  permuted_lengths = at::empty({permuted_lengths_size}, lengths.options());

  int num_threads = at::get_num_threads();
  std::vector<int64_t> output_offsets_per_thread_cumsum(
      (num_threads + 1) * FALSE_SHARING_PAD, 0);

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_1D_lengths_cpu_kernel", [&] {
        _permute_1D_lengths_cpu_kernel(
            lengths_contig->data_ptr<index_t>(),
            permuted_lengths_size,
            permute_contig->data_ptr<int32_t>(),
            permuted_lengths.data_ptr<index_t>());
      }); // for each scalar_t

  const auto input_offsets = asynchronous_exclusive_cumsum_cpu(lengths);
  const auto output_offsets =
      asynchronous_complete_cumsum_cpu(permuted_lengths);

  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size =
        output_offsets[permuted_lengths_size].item<int64_t>();
  }

  permuted_indices = at::empty(permuted_indices_size, indices.options());
  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_1D_indices_weights_kernel_1", [&] {
        using offsets_t = index_t;
        FBGEMM_DISPATCH_ALL_TYPES(
            indices.scalar_type(), "permute_1D_indices_weights_kernel_2", [&] {
              using indices_t = scalar_t;
              FBGEMM_DISPATCH_FLOAT_ONLY(
                  weights.has_value() ? weights.value().scalar_type()
                                      : at::ScalarType::Float,
                  "permute_1D_indices_weights_kernel_3",
                  [&] {
                    using weights_t = scalar_t;
                    if (weights.has_value()) {
                      const auto weights_value_contig =
                          weights.value().expect_contiguous();
                      permuted_weights = at::empty(
                          permuted_indices_size, weights.value().options());
                      _permute_1D_indices_weights_kernel_cpu<
                          true,
                          index_t,
                          indices_t,
                          weights_t>(
                          input_offsets.data_ptr<offsets_t>(),
                          indices_contig->data_ptr<indices_t>(),
                          weights_value_contig->data_ptr<weights_t>(),
                          permuted_lengths_size,
                          permute_contig->data_ptr<int32_t>(),
                          permuted_lengths.data_ptr<offsets_t>(),
                          output_offsets.data_ptr<offsets_t>(),
                          permuted_indices.data_ptr<indices_t>(),
                          permuted_weights.data_ptr<weights_t>());
                    } else {
                      _permute_1D_indices_weights_kernel_cpu<
                          false,
                          index_t,
                          indices_t,
                          weights_t>(
                          input_offsets.data_ptr<offsets_t>(),
                          indices_contig->data_ptr<indices_t>(),
                          nullptr,
                          permuted_lengths_size,
                          permute_contig->data_ptr<int32_t>(),
                          permuted_lengths.data_ptr<offsets_t>(),
                          output_offsets.data_ptr<offsets_t>(),
                          permuted_indices.data_ptr<indices_t>(),
                          nullptr);
                    }
                  }); // for each weights_t
            }); // for each indices_t
      }); // for each offsets_t

  return {permuted_lengths, permuted_indices, permuted_weights};
}

template <typename index_t, typename offsets_t>
void _expand_into_jagged_permute_cpu_kernel(
    const offsets_t* const __restrict__ input_offsets,
    const offsets_t* const __restrict__ output_offsets,
    const int64_t permute_size,
    const index_t* const __restrict__ permute,
    index_t* const __restrict__ output_permute) {
  at::parallel_for(
      0, permute_size, FALSE_SHARING_PAD, [&](int64_t t_begin, int64_t t_end) {
        for (int t = t_begin; t < std::min(t_end, permute_size); ++t) {
          offsets_t permute_length = output_offsets[t + 1] - output_offsets[t];
          const offsets_t input_start = input_offsets[permute[t]];
          const offsets_t output_start = output_offsets[t];
          for (const auto i : c10::irange(permute_length)) {
            output_permute[output_start + i] = input_start + i;
          }
        }
      }); // parallel_for T
}

Tensor expand_into_jagged_permute_cpu(
    const Tensor& permute,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    int64_t output_size) {
  TENSOR_ON_CPU(permute);
  TENSOR_ON_CPU(input_offsets);
  TENSOR_ON_CPU(output_offsets);
  TORCH_CHECK(permute.numel() > 0);
  TORCH_CHECK(permute.numel() == input_offsets.numel() - 1);
  TORCH_CHECK(permute.numel() == output_offsets.numel() - 1);

  const auto permute_contig = permute.contiguous();

  const auto permute_size = permute.numel();

  Tensor output_permute = at::empty({output_size}, input_offsets.options());

  AT_DISPATCH_INDEX_TYPES(
      permute.scalar_type(), "expand_into_jagged_permute_cpu", [&] {
        using offset_t = index_t;
        _expand_into_jagged_permute_cpu_kernel(
            input_offsets.data_ptr<offset_t>(),
            output_offsets.data_ptr<offset_t>(),
            permute_size,
            permute.data_ptr<index_t>(),
            output_permute.data_ptr<index_t>());
      });

  return output_permute;
}

template <typename index_t>
void _invert_permute_cpu_kernel(
    const int64_t permute_size,
    const index_t* const __restrict__ permute,
    index_t* const __restrict__ inversed_permute) {
  at::parallel_for(
      0, permute_size, FALSE_SHARING_PAD, [&](int64_t t_begin, int64_t t_end) {
        for (int t = t_begin; t < std::min(t_end, permute_size); ++t) {
          inversed_permute[permute[t]] = t;
        }
      });
}

Tensor invert_permute_cpu(const Tensor& permute) {
  TENSOR_ON_CPU(permute);
  const auto permute_contig = permute.expect_contiguous();
  const auto permute_size = permute.numel();
  Tensor inversed_permute = at::empty_like(permute);

  AT_DISPATCH_INDEX_TYPES(
      permute.scalar_type(), "invert_permute_cpu_kernel", [&] {
        _invert_permute_cpu_kernel<index_t>(
            permute_size,
            permute_contig->data_ptr<index_t>(),
            inversed_permute.data_ptr<index_t>());
      }); // for each scalar_t

  return inversed_permute;
}
template <typename index_t, typename offset_t>
void _populate_bucketized_permute_cpu(
    const offset_t* const length_data,
    const offset_t* const offset_data,
    offset_t* const bucketized_offsets_data,
    const index_t* const bucket_mapping_data,
    index_t* const bucketized_permute_data_out,
    int64_t length_size) {
  for (const auto i : c10::irange(length_size)) {
    const auto length = length_data[i];
    const auto offset = offset_data[i];
    for (const auto j : c10::irange(length)) {
      const auto index = offset + j;
      const auto bucket = bucket_mapping_data[index];
      bucketized_permute_data_out[index] =
          bucketized_offsets_data[bucket * length_size + i]++;
    }
  }
}

Tensor populate_bucketized_permute_cpu(
    const Tensor& lengths,
    const Tensor& bucketized_lengths,
    const Tensor& bucket_mapping) {
  const auto lengths_contig = lengths.expect_contiguous();
  const auto bucketized_lengths_contig = bucketized_lengths.expect_contiguous();
  const auto bucket_mapping_contig = bucket_mapping.expect_contiguous();
  Tensor bucketized_permute = native_empty_like(*bucket_mapping_contig);
  const auto offsets = asynchronous_complete_cumsum_cpu(*lengths_contig);
  const auto bucketized_offsets =
      asynchronous_complete_cumsum_cpu(*bucketized_lengths_contig);
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "populate_bucketized_permute_cpu_1", ([&] {
        using offset_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            bucket_mapping_contig->scalar_type(),
            "populate_bucketized_permute_cpu_2",
            ([&] {
              _populate_bucketized_permute_cpu<index_t, offset_t>(
                  lengths_contig->data_ptr<offset_t>(),
                  offsets.data_ptr<offset_t>(),
                  bucketized_offsets.data_ptr<offset_t>(),
                  bucket_mapping_contig->data_ptr<index_t>(),
                  bucketized_permute.data_ptr<index_t>(),
                  lengths_contig->numel());
            }));
      }));
  return bucketized_permute;
}

std::tuple<
    Tensor,
    Tensor,
    std::optional<Tensor>,
    std::optional<Tensor>,
    std::optional<Tensor>,
    std::optional<Tensor>>
_block_bucketize_sparse_features_cpu(
    const Tensor& lengths,
    const Tensor& indices,
    const bool bucketize_pos,
    const bool sequence,
    const Tensor& block_sizes,
    const std::optional<Tensor>& total_num_blocks,
    const int64_t my_size,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& batch_size_per_feature,
    const int64_t /* max_batch_size */, // Only used in GPU variant
    const std::optional<std::vector<at::Tensor>>& block_bucketize_pos,
    const bool return_bucket_mapping,
    const bool keep_orig_idx) {
  const auto lengths_size = lengths.numel();
  const auto new_lengths_size = lengths_size * my_size;
  auto new_lengths = at::zeros({new_lengths_size}, lengths.options());
  auto new_indices = native_empty_like(indices);
  std::optional<Tensor> new_weights;
  std::optional<Tensor> new_pos;
  std::optional<Tensor> unbucketize_permute;
  std::optional<Tensor> bucket_mapping;
  if (bucketize_pos) {
    new_pos = native_empty_like(indices);
  }
#define LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITH_WEIGHT(      \
    sequence, return_bucket_mapping)                             \
  AT_DISPATCH_INDEX_TYPES(                                       \
      lengths.scalar_type(),                                     \
      "block_bucketize_sparse_features_weights_cpu_1",           \
      [&] {                                                      \
        using offset_t = index_t;                                \
        AT_DISPATCH_INDEX_TYPES(                                 \
            indices.scalar_type(),                               \
            "block_bucketize_sparse_features_weights_cpu_2",     \
            [&] {                                                \
              FBGEMM_DISPATCH_FLOAT_ONLY(                        \
                  weights_value.scalar_type(),                   \
                  "bucketize_sparse_features_weights_cpu_3",     \
                  [&] {                                          \
                    _block_bucketize_sparse_features_cpu_kernel< \
                        sequence,                                \
                        true,                                    \
                        return_bucket_mapping,                   \
                        offset_t,                                \
                        index_t,                                 \
                        scalar_t>(                               \
                        lengths,                                 \
                        indices,                                 \
                        weights,                                 \
                        bucketize_pos,                           \
                        block_sizes,                             \
                        total_num_blocks,                        \
                        my_size,                                 \
                        new_lengths,                             \
                        new_indices,                             \
                        new_weights,                             \
                        new_pos,                                 \
                        unbucketize_permute,                     \
                        batch_size_per_feature,                  \
                        block_bucketize_pos,                     \
                        bucket_mapping,                          \
                        keep_orig_idx);                          \
                  });                                            \
            });                                                  \
      });

#define LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITHOUT_WEIGHT(              \
    sequence, return_bucket_mapping)                                        \
  AT_DISPATCH_INDEX_TYPES(                                                  \
      lengths.scalar_type(), "block_bucketize_sparse_features_cpu_1", [&] { \
        using offset_t = index_t;                                           \
        AT_DISPATCH_INDEX_TYPES(                                            \
            indices.scalar_type(),                                          \
            "block_bucketize_sparse_features_cpu_2",                        \
            [&] {                                                           \
              _block_bucketize_sparse_features_cpu_kernel<                  \
                  sequence,                                                 \
                  false,                                                    \
                  return_bucket_mapping,                                    \
                  offset_t,                                                 \
                  index_t,                                                  \
                  std::nullptr_t>(                                          \
                  lengths,                                                  \
                  indices,                                                  \
                  weights,                                                  \
                  bucketize_pos,                                            \
                  block_sizes,                                              \
                  total_num_blocks,                                         \
                  my_size,                                                  \
                  new_lengths,                                              \
                  new_indices,                                              \
                  new_weights,                                              \
                  new_pos,                                                  \
                  unbucketize_permute,                                      \
                  batch_size_per_feature,                                   \
                  block_bucketize_pos,                                      \
                  bucket_mapping,                                           \
                  keep_orig_idx);                                           \
            });                                                             \
      });
  const auto lengths_sum = indices.numel();
  if (weights.has_value()) {
    Tensor weights_value = weights.value();
    new_weights = native_empty_like(weights_value);
    if (sequence) {
      unbucketize_permute = at::empty({lengths_sum}, indices.options());
      if (return_bucket_mapping) {
        bucket_mapping = at::empty({lengths_sum}, indices.options());
        LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITH_WEIGHT(true, true)
      } else {
        LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITH_WEIGHT(true, false)
      }
    } else {
      if (return_bucket_mapping) {
        bucket_mapping = at::empty({lengths_sum}, indices.options());
        LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITH_WEIGHT(false, true)
      } else {
        LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITH_WEIGHT(false, false)
      }
    }
  } else {
    if (sequence) {
      unbucketize_permute = at::empty({lengths_sum}, indices.options());
      if (return_bucket_mapping) {
        bucket_mapping = at::empty({lengths_sum}, indices.options());
        LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITHOUT_WEIGHT(true, true)
      } else {
        LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITHOUT_WEIGHT(true, false)
      }
    } else {
      if (return_bucket_mapping) {
        bucket_mapping = at::empty({lengths_sum}, indices.options());
        LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITHOUT_WEIGHT(false, true)
      } else {
        LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITHOUT_WEIGHT(false, false)
      }
    }
  }
#undef LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITH_WEIGHT
#undef LAUNCH_BLOCK_BUCKETIZE_SPARSE_FEATURES_WITHOUT_WEIGHT
  return {
      new_lengths,
      new_indices,
      new_weights,
      new_pos,
      unbucketize_permute,
      bucket_mapping};
}

std::tuple<
    Tensor,
    Tensor,
    std::optional<Tensor>,
    std::optional<Tensor>,
    std::optional<Tensor>>
block_bucketize_sparse_features_cpu(
    const Tensor& lengths,
    const Tensor& indices,
    const bool bucketize_pos,
    const bool sequence,
    const Tensor& block_sizes,
    const int64_t my_size,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& batch_size_per_feature,
    const int64_t /* max_batch_size */, // Only used in GPU variant
    const std::optional<std::vector<at::Tensor>>& block_bucketize_pos,
    const bool keep_orig_idx,
    const std::optional<Tensor>& total_num_blocks) {
  Tensor new_lengths;
  Tensor new_indices;
  std::optional<Tensor> new_weights;
  std::optional<Tensor> new_pos;
  std::optional<Tensor> unbucketize_permute;
  std::tie(
      new_lengths,
      new_indices,
      new_weights,
      new_pos,
      unbucketize_permute,
      std::ignore) =
      _block_bucketize_sparse_features_cpu(
          lengths,
          indices,
          bucketize_pos,
          sequence,
          block_sizes,
          total_num_blocks,
          my_size,
          weights,
          batch_size_per_feature,
          -1, /* placeholder for max_batch_size */
          block_bucketize_pos,
          false,
          keep_orig_idx);
  return {new_lengths, new_indices, new_weights, new_pos, unbucketize_permute};
}

std::tuple<
    Tensor,
    Tensor,
    std::optional<Tensor>,
    std::optional<Tensor>,
    std::optional<Tensor>,
    std::optional<Tensor>>
block_bucketize_sparse_features_inference_cpu(
    const Tensor& lengths,
    const Tensor& indices,
    const bool bucketize_pos,
    const bool sequence,
    const Tensor& block_sizes,
    const int64_t my_size,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& batch_size_per_feature,
    const int64_t /* max_batch_size */, // Only used in GPU variant
    const std::optional<std::vector<at::Tensor>>& block_bucketize_pos,
    const bool return_bucket_mapping,
    const bool keep_orig_idx,
    const std::optional<Tensor>& total_num_blocks) {
  return _block_bucketize_sparse_features_cpu(
      lengths,
      indices,
      bucketize_pos,
      sequence,
      block_sizes,
      total_num_blocks,
      my_size,
      weights,
      batch_size_per_feature,
      -1, /* placeholder for max_batch_size */
      block_bucketize_pos,
      return_bucket_mapping,
      keep_orig_idx);
}

// This function partitions sparse features
// cyclically along the sparse dimension into my_size blocks
std::tuple<
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>>
bucketize_sparse_features_cpu(
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const bool bucketize_pos,
    const int64_t my_size,
    const std::optional<at::Tensor>& weights) {
  TENSOR_ON_CPU(lengths);
  TENSOR_ON_CPU(indices);
  TENSOR_ON_CPU(weights);

  const auto lengths_size = lengths.numel();
  const auto new_lengths_size = lengths_size * my_size;
  auto new_lengths = at::zeros({new_lengths_size}, lengths.options());
  auto new_indices = native_empty_like(indices);
  Tensor new_weights;
  Tensor new_pos;
  if (bucketize_pos) {
    new_pos = native_empty_like(indices);
  }
  if (weights.has_value()) {
    Tensor weights_value = weights.value();
    new_weights = native_empty_like(weights_value);
    AT_DISPATCH_INDEX_TYPES(
        indices.scalar_type(), "bucketize_sparse_features_weights_cpu_1", ([&] {
          FBGEMM_DISPATCH_FLOAT_ONLY(
              weights_value.scalar_type(),
              "bucketize_sparse_features_weights_cpu_2",
              ([&] {
                _bucketize_sparse_features_cpu<true, index_t, scalar_t>(
                    lengths,
                    indices,
                    weights,
                    bucketize_pos,
                    my_size,
                    new_lengths,
                    new_indices,
                    new_weights,
                    new_pos);
              }));
        }));
  } else {
    AT_DISPATCH_INDEX_TYPES(
        indices.scalar_type(), "bucketize_sparse_features_cpu", ([&] {
          _bucketize_sparse_features_cpu<false, index_t, std::nullptr_t>(
              lengths,
              indices,
              weights,
              bucketize_pos,
              my_size,
              new_lengths,
              new_indices,
              new_weights,
              new_pos);
        }));
  }
  return {new_lengths, new_indices, new_weights, new_pos};
}

template <typename index_t, typename scalar_t>
void reorder_batched_ad_lengths_(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths,
    Tensor& output,
    const int64_t max_batch_size = 0) {
  const int64_t nB = batch_offsets.numel() - 1;
  const int64_t nT = broadcast_lengths
      ? cat_ad_lengths.numel() / nB
      : cat_ad_lengths.numel() / num_ads_in_batch;
  int64_t output_batch_size = num_ads_in_batch;
  if (max_batch_size > 0) {
    TORCH_CHECK_GE(max_batch_size, num_ads_in_batch);
    TORCH_CHECK_EQ(output.numel(), max_batch_size * nT);
    output_batch_size = max_batch_size;
  }

  const auto* batch_offsets_data = batch_offsets.data_ptr<index_t>();
  const auto* cat_ad_lengths_data = cat_ad_lengths.data_ptr<scalar_t>();
  auto* output_data = output.data_ptr<scalar_t>();
  at::parallel_for(
      0, nB * nT, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        auto b_begin = tb_begin / nT;
        auto b_end = (tb_end + nT - 1) / nT;
        for (const auto b : c10::irange(b_begin, b_end)) {
          const auto num_ads_b =
              batch_offsets_data[b + 1] - batch_offsets_data[b];
          int64_t t_begin = (b == b_begin) ? tb_begin % nT : 0;
          int64_t t_end =
              (b == b_end - 1 && tb_end % nT != 0) ? tb_end % nT : nT;
          const auto data_size = num_ads_b * sizeof(scalar_t);
          for (const auto t : c10::irange(t_begin, t_end)) {
            const int32_t input_segment_start = broadcast_lengths
                ? nT * b + t
                : nT * batch_offsets_data[b] + t * num_ads_b;
            const int32_t output_segment_start =
                t * output_batch_size + batch_offsets_data[b];
            if (broadcast_lengths) {
              for (const auto i : c10::irange(num_ads_b)) {
                output_data[output_segment_start + i] =
                    cat_ad_lengths_data[input_segment_start];
              }
            } else {
              std::memcpy(
                  output_data + output_segment_start,
                  cat_ad_lengths_data + input_segment_start,
                  data_size);
            }
          }
        }
      });
}

Tensor reorder_batched_ad_lengths_cpu_out(
    Tensor& out,
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths,
    const int64_t max_batch_size) {
  TENSOR_ON_CPU(cat_ad_lengths);
  TENSOR_ON_CPU(batch_offsets);
  TENSOR_ON_CPU(out);

  AT_DISPATCH_INDEX_TYPES(
      batch_offsets.scalar_type(),
      "reorder_batched_ad_lengths_cpu_kernel1",
      [&] {
        FBGEMM_DISPATCH_ALL_TYPES(
            cat_ad_lengths.scalar_type(),
            "reorder_batched_ad_lengths_cpu_kernel2",
            [&] {
              reorder_batched_ad_lengths_<index_t, scalar_t>(
                  cat_ad_lengths,
                  batch_offsets,
                  num_ads_in_batch,
                  broadcast_lengths,
                  out,
                  max_batch_size);
            });
      });

  return out;
}

Tensor reorder_batched_ad_lengths_cpu(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths,
    const int64_t max_batch_size) {
  TENSOR_ON_CPU(cat_ad_lengths);
  TENSOR_ON_CPU(batch_offsets);
  Tensor reordered_cat_ad_lengths;
  if (max_batch_size > 0) {
    TORCH_CHECK(!broadcast_lengths);
    TORCH_CHECK(max_batch_size >= num_ads_in_batch);
    TORCH_CHECK(cat_ad_lengths.numel() % num_ads_in_batch == 0);
    reordered_cat_ad_lengths = at::zeros(
        {cat_ad_lengths.numel() / num_ads_in_batch * max_batch_size},
        cat_ad_lengths.options());
  } else {
    reordered_cat_ad_lengths = broadcast_lengths
        ? at::empty(
              {cat_ad_lengths.numel() / (batch_offsets.numel() - 1) *
               num_ads_in_batch},
              cat_ad_lengths.options())
        : at::empty_like(cat_ad_lengths, cat_ad_lengths.options());
  }

  return reorder_batched_ad_lengths_cpu_out(
      reordered_cat_ad_lengths,
      cat_ad_lengths,
      batch_offsets,
      num_ads_in_batch,
      broadcast_lengths,
      max_batch_size);
}

template <typename index_t, typename scalar_t>
void reorder_batched_ad_indices_cpu_(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    Tensor& output) {
  const int64_t nB = batch_offsets.numel() - 1;
  const int64_t nT = (reordered_cat_ad_offsets.numel() - 1) / num_ads_in_batch;

  const auto* batch_offsets_data = batch_offsets.data_ptr<int32_t>();
  const auto* cat_ad_offsets_data = cat_ad_offsets.data_ptr<index_t>();
  const auto* reordered_cat_ad_offsets_data =
      reordered_cat_ad_offsets.data_ptr<index_t>();
  const auto* cat_ad_indices_data = cat_ad_indices.data_ptr<scalar_t>();
  auto* output_data = output.data_ptr<scalar_t>();
  at::parallel_for(
      0, nB * nT, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        auto b_begin = tb_begin / nT;
        auto b_end = (tb_end + nT - 1) / nT;

        for (const auto b : c10::irange(b_begin, b_end)) {
          const auto num_ads_b =
              batch_offsets_data[b + 1] - batch_offsets_data[b];
          int64_t t_begin = (b == b_begin) ? tb_begin % nT : 0;
          int64_t t_end =
              (b == b_end - 1 && tb_end % nT != 0) ? tb_end % nT : nT;
          for (const auto t : c10::irange(t_begin, t_end)) {
            const auto output_segment_offset_start =
                t * num_ads_in_batch + batch_offsets_data[b];
            const auto output_segment_start =
                reordered_cat_ad_offsets_data[output_segment_offset_start];
            const int32_t input_segment_offset_start = broadcast_indices
                ? nT * b + t
                : nT * batch_offsets_data[b] + t * num_ads_b;
            const int32_t input_segment_offset_end = broadcast_indices
                ? input_segment_offset_start + 1
                : input_segment_offset_start + num_ads_b;
            const auto input_segment_start =
                cat_ad_offsets_data[input_segment_offset_start];
            const auto input_segment_end =
                cat_ad_offsets_data[input_segment_offset_end];
            const auto num_elements = input_segment_end - input_segment_start;

            if (broadcast_indices) {
              for (auto j : c10::irange(num_ads_b)) {
                for (auto i : c10::irange(num_elements)) {
                  output_data[output_segment_start + j * num_elements + i] =
                      cat_ad_indices_data[input_segment_start + i];
                }
              }
            } else {
              for (auto i : c10::irange(num_elements)) {
                output_data[output_segment_start + i] =
                    cat_ad_indices_data[input_segment_start + i];
              }
            }
          }
        }
      });
}

template <typename index_t, typename scalar_t>
void cat_reorder_batched_ad_indices_cpu_(
    const Tensor& cat_ad_offsets,
    const std::vector<Tensor>& ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    int64_t max_batch_size,
    Tensor& output) {
  if (max_batch_size <= 0) {
    max_batch_size = num_ads_in_batch;
  }
  const int64_t nB = batch_offsets.numel() - 1;
  const int64_t nT = (reordered_cat_ad_offsets.numel() - 1) / max_batch_size;

  const auto* batch_offsets_data = batch_offsets.data_ptr<int32_t>();
  const auto* cat_ad_offsets_data = cat_ad_offsets.data_ptr<index_t>();
  const auto* reordered_cat_ad_offsets_data =
      reordered_cat_ad_offsets.data_ptr<index_t>();
  auto* output_data = output.data_ptr<scalar_t>();
  at::parallel_for(
      0, nB * nT, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        auto b_begin = tb_begin / nT;
        auto b_end = (tb_end + nT - 1) / nT;
        for (auto b : c10::irange(b_begin, b_end)) {
          const auto* ad_indices_data = ad_indices[b].data_ptr<scalar_t>();
          const auto num_ads_b =
              batch_offsets_data[b + 1] - batch_offsets_data[b];
          int64_t t_begin = (b == b_begin) ? tb_begin % nT : 0;
          int64_t t_end =
              (b == b_end - 1 && tb_end % nT != 0) ? tb_end % nT : nT;
          for (auto t : c10::irange(t_begin, t_end)) {
            const auto output_segment_offset_start =
                t * max_batch_size + batch_offsets_data[b];
            const auto output_segment_start =
                reordered_cat_ad_offsets_data[output_segment_offset_start];
            const int32_t input_segment_offset_start = broadcast_indices
                ? nT * b + t
                : nT * batch_offsets_data[b] + t * num_ads_b;
            const int32_t input_segment_offset_end = broadcast_indices
                ? input_segment_offset_start + 1
                : input_segment_offset_start + num_ads_b;
            const auto based_segment = broadcast_indices
                ? cat_ad_offsets_data[nT * b]
                : cat_ad_offsets_data[nT * batch_offsets_data[b]];
            const auto input_segment_start =
                cat_ad_offsets_data[input_segment_offset_start] - based_segment;
            const auto input_segment_end =
                cat_ad_offsets_data[input_segment_offset_end] - based_segment;
            const auto num_elements = input_segment_end - input_segment_start;
            const auto data_size = num_elements * sizeof(scalar_t);
            if (broadcast_indices) {
              for (auto j : c10::irange(num_ads_b)) {
                std::memcpy(
                    output_data + output_segment_start + j * num_elements,
                    ad_indices_data + input_segment_start,
                    data_size);
              }
            } else {
              std::memcpy(
                  output_data + output_segment_start,
                  ad_indices_data + input_segment_start,
                  data_size);
            }
          }
        }
      });
}

template <typename index_t, typename scalar_t>
void reorder_batched_sequence_embeddings_cpu_(
    const Tensor& cat_sequence_embeddings_offsets,
    const Tensor& cat_sequence_embeddings,
    const Tensor& reordered_cat_sequence_embeddings_offsets,
    const Tensor& batch_offsets,
    const int64_t num_items_in_batch,
    const int32_t dim,
    Tensor& output) {
  const int64_t nB = batch_offsets.numel() - 1;
  const int64_t nT = (reordered_cat_sequence_embeddings_offsets.numel() - 1) /
      num_items_in_batch;

  const auto* batch_offsets_data = batch_offsets.data_ptr<index_t>();
  const auto* cat_sequence_embeddings_offsets_data =
      cat_sequence_embeddings_offsets.data_ptr<index_t>();
  const auto* reordered_cat_sequence_embeddings_offsets_data =
      reordered_cat_sequence_embeddings_offsets.data_ptr<index_t>();
  const auto* cat_sequence_embeddings_data =
      cat_sequence_embeddings.data_ptr<scalar_t>();
  auto* output_data = output.data_ptr<scalar_t>();
  at::parallel_for(
      0, nB * nT, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        auto b_begin = tb_begin / nT;
        auto b_end = (tb_end + nT - 1) / nT;

        for (const auto b : c10::irange(b_begin, b_end)) {
          const auto num_ads_b =
              batch_offsets_data[b + 1] - batch_offsets_data[b];
          int64_t t_begin = (b == b_begin) ? tb_begin % nT : 0;
          int64_t t_end =
              (b == b_end - 1 && tb_end % nT != 0) ? tb_end % nT : nT;
          for (const auto t : c10::irange(t_begin, t_end)) {
            const auto output_segment_offset_start =
                t * num_items_in_batch + batch_offsets_data[b];
            const auto output_segment_start =
                reordered_cat_sequence_embeddings_offsets_data
                    [output_segment_offset_start] *
                dim;
            const int32_t input_segment_offset_start =
                nT * batch_offsets_data[b] + t * num_ads_b;
            const int32_t input_segment_offset_end =
                input_segment_offset_start + num_ads_b;
            const auto input_segment_start =
                cat_sequence_embeddings_offsets_data
                    [input_segment_offset_start] *
                dim;
            const auto input_segment_end =
                cat_sequence_embeddings_offsets_data[input_segment_offset_end] *
                dim;
            const auto num_elements = (input_segment_end - input_segment_start);

            for (auto i : c10::irange(num_elements)) {
              // TODO memcpy once this path is heavily used?
              output_data[output_segment_start + i] =
                  cat_sequence_embeddings_data[input_segment_start + i];
            }
          }
        }
      });
}

Tensor reorder_batched_sequence_embeddings_cpu(
    const Tensor& cat_sequence_embeddings_offsets,
    const Tensor& cat_sequence_embeddings,
    const Tensor& reordered_cat_sequence_embeddings_offsets,
    const Tensor& batch_offsets,
    const int64_t num_items_in_batch) {
  TENSOR_ON_CPU(cat_sequence_embeddings_offsets);
  TENSOR_ON_CPU(cat_sequence_embeddings);
  TENSOR_ON_CPU(reordered_cat_sequence_embeddings_offsets);
  TENSOR_ON_CPU(batch_offsets);
  TORCH_CHECK(cat_sequence_embeddings.dim() == 2);
  // reorder embeddings from (ragged) [B x T x #num_ads_B_{i} x length_{B_{i},
  // t, a})x D] to [T][B][#num_ads_b][length_{b, t, a}][D], i.e.
  // [sum(length_{B_{i}, t, a}), D]
  Tensor reordered_cat_ad_indices = at::empty_like(
      cat_sequence_embeddings, cat_sequence_embeddings.options());

  AT_DISPATCH_INDEX_TYPES(
      cat_sequence_embeddings_offsets.scalar_type(),
      "reorder_batched_sequence_embeddings_cpu_kernel_1",
      [&] {
        FBGEMM_DISPATCH_FLOATING_TYPES(
            cat_sequence_embeddings.scalar_type(),
            "reorder_batched_sequence_embeddings_cpu_kernel_2",
            [&] {
              reorder_batched_sequence_embeddings_cpu_<index_t, scalar_t>(
                  cat_sequence_embeddings_offsets,
                  cat_sequence_embeddings,
                  reordered_cat_sequence_embeddings_offsets,
                  batch_offsets,
                  num_items_in_batch,
                  cat_sequence_embeddings.size(1),
                  reordered_cat_ad_indices);
            });
      });

  return reordered_cat_ad_indices;
}

Tensor reorder_batched_ad_indices_cpu(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    const int64_t num_indices_after_broadcast) {
  TENSOR_ON_CPU(cat_ad_offsets);
  TENSOR_ON_CPU(cat_ad_indices);
  TENSOR_ON_CPU(reordered_cat_ad_offsets);
  TENSOR_ON_CPU(batch_offsets);

  Tensor reordered_cat_ad_indices;
  if (broadcast_indices) {
    TORCH_CHECK_GE(num_indices_after_broadcast, 0);
    reordered_cat_ad_indices =
        at::empty({num_indices_after_broadcast}, cat_ad_indices.options());
  } else {
    reordered_cat_ad_indices =
        at::empty_like(cat_ad_indices, cat_ad_indices.options());
  }
  AT_DISPATCH_INDEX_TYPES(
      cat_ad_offsets.scalar_type(),
      "reorder_batched_ad_indices_cpu_kernel_1",
      [&] {
        FBGEMM_DISPATCH_ALL_TYPES(
            cat_ad_indices.scalar_type(),
            "reorder_batched_ad_indices_cpu_kernel_2",
            [&] {
              reorder_batched_ad_indices_cpu_<index_t, scalar_t>(
                  cat_ad_offsets,
                  cat_ad_indices,
                  reordered_cat_ad_offsets,
                  batch_offsets,
                  num_ads_in_batch,
                  broadcast_indices,
                  reordered_cat_ad_indices);
            });
      });

  return reordered_cat_ad_indices;
}

Tensor cat_reorder_batched_ad_indices_cpu(
    const Tensor& cat_ad_offsets,
    const std::vector<Tensor>& ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    const int64_t total_num_indices,
    const bool pinned_memory,
    const int64_t max_batch_size) {
  TENSOR_ON_CPU(cat_ad_offsets);
  for (const auto& t : ad_indices) {
    TENSOR_ON_CPU(t);
  }
  TENSOR_ON_CPU(reordered_cat_ad_offsets);
  TENSOR_ON_CPU(batch_offsets);
  TORCH_CHECK_GE(total_num_indices, 0);
  Tensor reordered_cat_ad_indices = at::empty(
      {total_num_indices},
      ad_indices[0].options().pinned_memory(pinned_memory));
  return cat_reorder_batched_ad_indices_cpu_out(
      reordered_cat_ad_indices,
      cat_ad_offsets,
      ad_indices,
      reordered_cat_ad_offsets,
      batch_offsets,
      num_ads_in_batch,
      broadcast_indices,
      max_batch_size);
}

Tensor cat_reorder_batched_ad_indices_cpu_out(
    Tensor& out,
    const Tensor& cat_ad_offsets,
    const std::vector<Tensor>& ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    const int64_t max_batch_size) {
  TENSOR_ON_CPU(cat_ad_offsets);
  for (const auto& t : ad_indices) {
    TENSOR_ON_CPU(t);
  }
  TENSOR_ON_CPU(reordered_cat_ad_offsets);
  TENSOR_ON_CPU(batch_offsets);
  AT_DISPATCH_INDEX_TYPES(
      cat_ad_offsets.scalar_type(),
      "cat_reorder_batched_ad_indices_cpu_kernel_1",
      [&] {
        FBGEMM_DISPATCH_ALL_TYPES(
            ad_indices[0].scalar_type(),
            "cat_reorder_batched_ad_indices_cpu_kernel_2",
            [&] {
              cat_reorder_batched_ad_indices_cpu_<index_t, scalar_t>(
                  cat_ad_offsets,
                  ad_indices,
                  reordered_cat_ad_offsets,
                  batch_offsets,
                  num_ads_in_batch,
                  broadcast_indices,
                  max_batch_size,
                  out);
            });
      });

  return out;
}

Tensor offsets_range_cpu(const Tensor& offsets, int64_t range_size) {
  TENSOR_ON_CPU(offsets);
  TENSOR_NDIM_EQUALS(offsets, 1);

  const auto offsets_arg = at::TensorArg(offsets, "offsets", 1);
  checkScalarTypes("_offsets_range_cpu", offsets_arg, {at::kLong, at::kInt});
  auto range = at::empty(range_size, offsets.options());
  if (range_size == 0) {
    return range;
  }
  const auto offsets_contig = offsets.expect_contiguous();
  const auto N = offsets_contig->numel();
  AT_DISPATCH_INDEX_TYPES(
      offsets_contig->scalar_type(), "offsets_range_kernel", [&]() {
        const index_t* offsets_data = offsets_contig->data_ptr<index_t>();
        index_t* range_data = range.data_ptr<index_t>();

        index_t last = range_size;
        for (int64_t i = N - 1; i >= 0; --i) {
          index_t first = offsets_data[i];
          std::iota(range_data + first, range_data + last, 0);
          last = first;
        }
      });

  return range;
}

/// CPU version of batched_unary_embeddings forward pass.
///
/// Sums up `weight` embeddings according to `offsets` and `indices`.
/// `table_offests` is a helper struct to quickly navigate through tables in
/// `weight` -- it is caller's responsibility to keep it in sync with `weight`.
/// Visualization of op semantics: https://fburl.com/9a4uktmb
///
/// This version is only for numerical verification so not optimized for
/// performance.
///
/// @param weight        - Weight for the embeddings.
/// @param table_offsets - Index offsets for each table entry in `weight`.
/// @param offsets       - Offsets for the starting point of each summation.
/// @param indices       - Indices for the embeddings to fetch (from `weight`).
/// @return The sumed embeddings.
Tensor batched_unary_embeddings_forward_cpu(
    const Tensor& weight,
    const Tensor& table_offsets,
    const Tensor& offsets,
    const Tensor& indices) {
  TENSOR_ON_CPU(weight);
  TENSOR_ON_CPU(table_offsets);
  TENSOR_ON_CPU(offsets);
  TENSOR_ON_CPU(indices);

  // N: number of tasks, T: number of tables, B: batch size
  const int32_t N = weight.sizes()[0];
  const int32_t T = table_offsets.numel() - 1;
  const int32_t B = (offsets.numel() - 1) / T;
  TORCH_CHECK(N > 0);
  TORCH_CHECK(T > 0);
  TORCH_CHECK(B > 0);

  // Make sure the index_t are consistent among table_offsets, offsets and
  // indices
  TORCH_CHECK(table_offsets.scalar_type() == offsets.scalar_type());
  TORCH_CHECK(table_offsets.scalar_type() == indices.scalar_type());

  auto output = at::empty({N, B, T}, weight.options());

  AT_DISPATCH_INDEX_TYPES(table_offsets.scalar_type(), "unary_indices", [&] {
    FBGEMM_DISPATCH_FLOATING_TYPES(
        weight.scalar_type(), "batched_unary_embeddings_forward_cpu", [&] {
          const index_t* table_offsets_data = table_offsets.data_ptr<index_t>();
          const index_t* offsets_data = offsets.data_ptr<index_t>();
          const index_t* indices_data = indices.data_ptr<index_t>();
          const index_t sum_E = table_offsets_data[T];
          auto* output_data = output.data_ptr<scalar_t>();
          const auto* weight_data = weight.data_ptr<scalar_t>();

          for (const auto n : c10::irange(N)) {
            for (const auto b : c10::irange(B)) {
              for (const auto t : c10::irange(T)) {
                const index_t indices_start = offsets_data[t * B + b];
                const index_t indices_end = offsets_data[t * B + b + 1];
                float sum = 0;
                for (const auto l : c10::irange(indices_start, indices_end)) {
                  const index_t idx =
                      n * sum_E + table_offsets_data[t] + indices_data[l];
                  // Since we don't care about the performance of CPU impl,
                  // adding the boundary check here. OOB will result in
                  // undefined behavior for GPU impl.
                  TORCH_CHECK(idx < weight.numel());
                  sum += weight_data[idx];
                }
                output_data[(n * B + b) * T + t] = sum;
              }
            }
          }
        });
  });

  return output;
}

template <typename T>
void _histogram_binning_calibration_cpu_kernel(
    const int64_t num_logits,
    const double recalibrate_value,
    const double step,
    const int64_t bin_ctr_in_use_after,
    const double bin_ctr_weight_value,
    const T* const logit_data,
    const double* const bin_num_examples_data,
    const double* const bin_num_positives_data,
    T* const calibrated_prediction_data,
    int64_t* const bin_ids_data) {
  for (const auto i : c10::irange(num_logits)) {
    const T pre_sigmoid = logit_data[i] + recalibrate_value;
    const double uncalibrated = 1.0 / (1.0 + std::exp(-pre_sigmoid));

    bin_ids_data[i] = std::ceil(uncalibrated / step) - 1;

    const auto curr_bin_num_examples = bin_num_examples_data[bin_ids_data[i]];
    if (curr_bin_num_examples > bin_ctr_in_use_after) {
      const auto curr_bin_ctr =
          bin_num_positives_data[bin_ids_data[i]] / curr_bin_num_examples;
      calibrated_prediction_data[i] = curr_bin_ctr * bin_ctr_weight_value +
          uncalibrated * (1.0 - bin_ctr_weight_value);
    } else {
      calibrated_prediction_data[i] = uncalibrated;
    }
  }
}

std::tuple<Tensor, Tensor> histogram_binning_calibration_cpu(
    const Tensor& logit,
    const Tensor& bin_num_examples,
    const Tensor& bin_num_positives,
    double positive_weight,
    double lower_bound,
    double upper_bound,
    int64_t bin_ctr_in_use_after,
    double bin_ctr_weight_value) {
  TENSOR_ON_CPU(logit);
  TENSOR_ON_CPU(bin_num_examples);
  TENSOR_ON_CPU(bin_num_positives);
  TORCH_CHECK(bin_num_examples.numel() == bin_num_positives.numel());

  Tensor calibrated_prediction = at::empty_like(logit);
  Tensor bin_ids = at::empty({logit.numel()}, logit.options().dtype(at::kLong));
  const double recalibrate_value = std::log(positive_weight);
  const double step = (upper_bound - lower_bound) /
      static_cast<double>(bin_num_examples.numel());
  FBGEMM_DISPATCH_FLOATING_TYPES(
      logit.scalar_type(), "histogram_binning_calibration_cpu", [&] {
        _histogram_binning_calibration_cpu_kernel<scalar_t>(
            logit.numel(),
            recalibrate_value,
            step,
            bin_ctr_in_use_after,
            bin_ctr_weight_value,
            logit.data_ptr<scalar_t>(),
            bin_num_examples.data_ptr<double>(),
            bin_num_positives.data_ptr<double>(),
            calibrated_prediction.data_ptr<scalar_t>(),
            bin_ids.data_ptr<int64_t>());
      });

  return std::make_tuple(calibrated_prediction, bin_ids);
}

template <typename LogitType, typename SegmentValueType>
void _histogram_binning_calibration_by_feature_cpu_kernel(
    const int64_t num_logits,
    const int64_t num_bins,
    const int64_t num_segments,
    const double recalibrate_value,
    const double step,
    const int64_t bin_ctr_in_use_after,
    const double bin_ctr_weight_value,
    const LogitType* const logit_data,
    const SegmentValueType* const dense_segment_value_data,
    const double* const bin_num_examples_data,
    const double* const bin_num_positives_data,
    LogitType* const calibrated_prediction_data,
    int64_t* const bin_ids_data) {
  for (const auto i : c10::irange(num_logits)) {
    const LogitType pre_sigmoid = logit_data[i] + recalibrate_value;
    const double uncalibrated = 1.0 / (1.0 + std::exp(-pre_sigmoid));

    const int64_t curr_segment_value =
        dense_segment_value_data[i] > num_segments
        ? 0
        : std::max(0L, dense_segment_value_data[i] * num_bins);

    bin_ids_data[i] = (std::ceil(uncalibrated / step) - 1) + curr_segment_value;

    const auto curr_bin_num_examples = bin_num_examples_data[bin_ids_data[i]];
    if (curr_bin_num_examples > bin_ctr_in_use_after) {
      const auto curr_bin_ctr =
          bin_num_positives_data[bin_ids_data[i]] / curr_bin_num_examples;
      calibrated_prediction_data[i] = curr_bin_ctr * bin_ctr_weight_value +
          uncalibrated * (1.0 - bin_ctr_weight_value);
    } else {
      calibrated_prediction_data[i] = uncalibrated;
    }
  }
}

std::tuple<Tensor, Tensor> histogram_binning_calibration_by_feature_cpu(
    const Tensor& logit,
    const Tensor& segment_value,
    const Tensor& segment_lengths,
    int64_t num_segments,
    const Tensor& bin_num_examples,
    const Tensor& bin_num_positives,
    int64_t num_bins,
    double positive_weight,
    double lower_bound,
    double upper_bound,
    int64_t bin_ctr_in_use_after,
    double bin_ctr_weight_value) {
  TENSOR_ON_CPU(logit);
  TENSOR_ON_CPU(segment_value);
  TENSOR_ON_CPU(segment_lengths);
  TENSOR_ON_CPU(bin_num_examples);
  TENSOR_ON_CPU(bin_num_positives);
  TORCH_CHECK(bin_num_examples.numel() == bin_num_positives.numel());

  // dense_segment_value is used as a temporary storage.
  Tensor dense_segment_value =
      at::empty({logit.numel()}, segment_value.options());
  AT_DISPATCH_INDEX_TYPES(
      segment_value.scalar_type(), "to_dense_representation_cpu_wrapper", [&] {
        using segment_value_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            segment_lengths.scalar_type(), "to_dense_representation_cpu", [&] {
              using segment_length_t = index_t;
              _to_dense_representation<segment_value_t, segment_length_t>(
                  segment_lengths.numel(),
                  segment_value.data_ptr<segment_value_t>(),
                  segment_lengths.data_ptr<segment_length_t>(),
                  dense_segment_value.data_ptr<segment_value_t>());
            });
      });

  Tensor calibrated_prediction = at::empty_like(logit);
  Tensor bin_ids = at::empty({logit.numel()}, logit.options().dtype(at::kLong));
  const double recalibrate_value = std::log(positive_weight);
  const double step =
      (upper_bound - lower_bound) / static_cast<double>(num_bins);
  FBGEMM_DISPATCH_FLOATING_TYPES(
      logit.scalar_type(),
      "histogram_binning_calibration_by_feature_cpu_wrapper",
      [&] {
        using logit_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(
            segment_value.scalar_type(),
            "histogram_binning_calibration_by_feature_cpu",
            [&] {
              using segment_value_t = index_t;
              _histogram_binning_calibration_by_feature_cpu_kernel<
                  logit_t,
                  segment_value_t>(
                  logit.numel(),
                  num_bins,
                  num_segments,
                  recalibrate_value,
                  step,
                  bin_ctr_in_use_after,
                  bin_ctr_weight_value,
                  logit.data_ptr<logit_t>(),
                  dense_segment_value.data_ptr<segment_value_t>(),
                  bin_num_examples.data_ptr<double>(),
                  bin_num_positives.data_ptr<double>(),
                  calibrated_prediction.data_ptr<logit_t>(),
                  bin_ids.data_ptr<int64_t>());
            });
      });

  return std::make_tuple(calibrated_prediction, bin_ids);
}

template <typename LogitType, typename SegmentValueType>
void _generic_histogram_binning_calibration_by_feature_cpu_kernel(
    const int64_t num_logits,
    const int64_t num_bins,
    const int64_t num_segments,
    const double recalibrate_value,
    const int64_t bin_ctr_in_use_after,
    const double bin_ctr_weight_value,
    const LogitType* const logit_data,
    const SegmentValueType* const dense_segment_value_data,
    const double* const bin_num_examples_data,
    const double* const bin_num_positives_data,
    const double* const bin_boundaries,
    LogitType* const calibrated_prediction_data,
    int64_t* const bin_ids_data) {
  for (const auto i : c10::irange(num_logits)) {
    const LogitType pre_sigmoid = logit_data[i] + recalibrate_value;
    const double uncalibrated = 1.0 / (1.0 + std::exp(-pre_sigmoid));

    const int curr_bin_id =
        std::lower_bound(
            bin_boundaries, bin_boundaries + num_bins - 1, uncalibrated) -
        bin_boundaries;

    const int64_t curr_segment_value =
        dense_segment_value_data[i] > num_segments
        ? 0
        : std::max(0L, dense_segment_value_data[i] * num_bins);

    bin_ids_data[i] = curr_bin_id + curr_segment_value;

    const auto curr_bin_num_examples = bin_num_examples_data[bin_ids_data[i]];
    if (curr_bin_num_examples > bin_ctr_in_use_after) {
      const auto curr_bin_ctr =
          bin_num_positives_data[bin_ids_data[i]] / curr_bin_num_examples;
      calibrated_prediction_data[i] = curr_bin_ctr * bin_ctr_weight_value +
          uncalibrated * (1.0 - bin_ctr_weight_value);
    } else {
      calibrated_prediction_data[i] = uncalibrated;
    }
  }
}

std::tuple<Tensor, Tensor> generic_histogram_binning_calibration_by_feature_cpu(
    const Tensor& logit,
    const Tensor& segment_value,
    const Tensor& segment_lengths,
    int64_t num_segments,
    const Tensor& bin_num_examples,
    const Tensor& bin_num_positives,
    const Tensor& bin_boundaries,
    double positive_weight,
    int64_t bin_ctr_in_use_after,
    double bin_ctr_weight_value) {
  TENSOR_ON_CPU(logit);
  TENSOR_ON_CPU(segment_value);
  TENSOR_ON_CPU(segment_lengths);
  TENSOR_ON_CPU(bin_num_examples);
  TENSOR_ON_CPU(bin_num_positives);
  TENSOR_ON_CPU(bin_boundaries);
  TORCH_CHECK(bin_num_examples.numel() == bin_num_positives.numel());
  TORCH_CHECK(
      bin_num_examples.numel() ==
      (num_segments + 1) * (bin_boundaries.numel() + 1));

  // dense_segment_value is used as a temporary storage.
  Tensor dense_segment_value =
      at::empty({logit.numel()}, segment_value.options());
  AT_DISPATCH_INDEX_TYPES(
      segment_value.scalar_type(), "to_dense_representation_cpu_wrapper", [&] {
        using segment_value_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            segment_lengths.scalar_type(), "to_dense_representation_cpu", [&] {
              using segment_length_t = index_t;
              _to_dense_representation<segment_value_t, segment_length_t>(
                  segment_lengths.numel(),
                  segment_value.data_ptr<segment_value_t>(),
                  segment_lengths.data_ptr<segment_length_t>(),
                  dense_segment_value.data_ptr<segment_value_t>());
            });
      });

  Tensor calibrated_prediction = at::empty_like(logit);
  Tensor bin_ids = at::empty({logit.numel()}, logit.options().dtype(at::kLong));
  const double recalibrate_value = std::log(positive_weight);
  FBGEMM_DISPATCH_FLOATING_TYPES(
      logit.scalar_type(),
      "generic_histogram_binning_calibration_by_feature_cpu_wrapper",
      [&] {
        using logit_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(
            segment_value.scalar_type(),
            "generic_histogram_binning_calibration_by_feature_cpu",
            [&] {
              using segment_value_t = index_t;
              _generic_histogram_binning_calibration_by_feature_cpu_kernel<
                  logit_t,
                  segment_value_t>(
                  logit.numel(),
                  bin_boundaries.numel() + 1,
                  num_segments,
                  recalibrate_value,
                  bin_ctr_in_use_after,
                  bin_ctr_weight_value,
                  logit.data_ptr<logit_t>(),
                  dense_segment_value.data_ptr<segment_value_t>(),
                  bin_num_examples.data_ptr<double>(),
                  bin_num_positives.data_ptr<double>(),
                  bin_boundaries.data_ptr<double>(),
                  calibrated_prediction.data_ptr<logit_t>(),
                  bin_ids.data_ptr<int64_t>());
            });
      });

  return std::make_tuple(calibrated_prediction, bin_ids);
}

template <typename scalar_t>
void _segment_sum_csr_cpu_kernel(
    const int num_segments,
    const int batch_size,
    const int* const csr_seg_data,
    const scalar_t* const values_data,
    scalar_t* const output_data) {
  for (const auto i : c10::irange(num_segments)) {
    const int seg_start = csr_seg_data[i] * batch_size;
    const int seg_end = csr_seg_data[i + 1] * batch_size;
    scalar_t v = 0;
    for (const auto j : c10::irange(seg_start, seg_end)) {
      v += values_data[j];
    }
    output_data[i] = v;
  }
}

Tensor segment_sum_csr_cpu(
    const int64_t batch_size,
    const Tensor& csr_seg,
    const Tensor& values) {
  TENSOR_ON_CPU(csr_seg);
  TENSOR_ON_CPU(values);

  auto output = at::empty(csr_seg.numel() - 1, values.options());
  FBGEMM_DISPATCH_ALL_TYPES(values.scalar_type(), "_segment_sum_csr_cpu", [&] {
    _segment_sum_csr_cpu_kernel<scalar_t>(
        csr_seg.numel() - 1,
        batch_size,
        csr_seg.data_ptr<int>(),
        values.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>());
  });
  return output;
}

bool should_prune(
    const Tensor& weights,
    const int64_t num_rows_kept,
    double min_save_ratio) {
  TENSOR_ON_CPU(weights);
  const auto weight_sizes = weights.sizes();

  const int64_t data_byte_size = sizeof(float);
  const int64_t num_cols = weight_sizes[1];

  // Size of the pruned weights tensor.
  const int64_t lut_after_prune_size =
      num_rows_kept * num_cols * data_byte_size;

  constexpr auto index_byte_size = sizeof(int);
  const auto lut_num_row = weight_sizes[0];
  const int64_t compressed_idx_overhead_size = lut_num_row * index_byte_size;

  const int64_t original_size = data_byte_size * weights.numel();
  return (compressed_idx_overhead_size + lut_after_prune_size) <
      min_save_ratio * original_size;
}

// This operator introduces sparsity to a weight matrix by applying
// magnitude based pruning at a row level. The importance level of a row is
// specified using an 'indicator' vector which contains a single value per
// row of the weight matrix.
//
// A row is considered important and not pruned if the indicator value for that
// particular row is greater than the pruning 'threshold' value.
//
// This operator doesn't zero out the pruned rows in-place. Instead, it returns
// a tuple that contains a pruned weights tensor as well as a map that can be
// used to refer the original row in the pruned weights tensor. We refer this
// map as 'compressed indices map' going forward.

// The compressed indices map is an 1D tensor that contains one entry per
// original row in 'weights'. The array index is the index for the original
// non-pruned weight tensor and the value would be the re-mapped index in the
// pruned weights tensor. If the value for a index is -1, it means the
// corresponding row has been pruned from the original weight tensor.

// Arguments:
// 'weights' - the weight tensor that needs to be pruned rowwise.
// 'indicator' - the magnitude for every row of the 'weights' matrix.
// 'threshold' - the pruning threshold that will be used for comparison
//     against the indicator row value.
// 'compressed_indices_dtype' - dtype for the compressed map indices.
//     This should be either int32 or int64.
// 'abs' - whether we should perform abs() on the indicator value or not.
// 'min_non_pruned_rows' - a minimum threshold on the number of rows
//     that should be present after pruning.
// 'min_save_ratio' - a parameter to tradeoff between lookup table CPU overhead
//     with the reduction in memory bandwidth due to pruned rows.
//     Pruning will be skipped for the entire matrix if the physical size of
//     pruned weights and indices mapping is greater than
//     min_save_ratio * weights size.
//     'compressed indices map' will contain a single element [0] in this case.
//
// Returns: a tuple,
// - The first value is the pruned weight tensor whose dtype is float.
// - The second value is a 1D tensor whose dtype is 'compressed_indices_dtype'.
std::tuple<Tensor, Tensor> embedding_bag_rowwise_prune(
    const Tensor& weights,
    const Tensor& indicator,
    const double threshold,
    at::ScalarType compressed_indices_dtype,
    const bool abs,
    const int64_t min_non_pruned_rows,
    const std::optional<double>& min_save_ratio) {
  TENSOR_ON_CPU(weights);
  TENSOR_ON_CPU(indicator);
  TENSOR_NDIM_EQUALS(weights, 2);
  TORCH_CHECK(
      indicator.numel() == weights.sizes()[0],
      "Number of elements in 'indicator' should be equivalent to "
      "number of rows in 'weights'.")
  TORCH_CHECK(
      threshold >= 0.0, "Threshold should be greater than or equal to zero.");
  TORCH_CHECK(
      compressed_indices_dtype == at::ScalarType::Int ||
          compressed_indices_dtype == at::ScalarType::Long,
      "'compressed_indices_dtype' should be Int/Long.");

  const auto indicator_contig = indicator.expect_contiguous();
  const auto indicator_data = indicator_contig->data_ptr<float>();
  auto rowwise_prune_mask = at::empty({indicator.numel()}, at::kBool);
  int num_kept = 0;
  for (const auto i : c10::irange(indicator.numel())) {
    const float val = abs ? std::abs(indicator_data[i]) : indicator_data[i];
    bool should_keep_row = val > threshold;

    // The total number of rows post-pruning should be greater than or equal
    // to 'min_non_pruned_rows'.
    // Skip pruning the current row to satisfy the above criteria.
    if (num_kept < min_non_pruned_rows &&
        num_kept + (indicator.numel() - i) <= min_non_pruned_rows) {
      should_keep_row = true;
    }
    if (!should_keep_row) {
      rowwise_prune_mask[i] = false;
      continue;
    }
    rowwise_prune_mask[i] = true;
    num_kept++;
  }

  if (min_save_ratio.has_value() &&
      !should_prune(weights, min_non_pruned_rows, min_save_ratio.value())) {
    auto compressed_indices_mapping = at::empty({1}, compressed_indices_dtype);
    compressed_indices_mapping[0] = 0;
    return std::tuple<Tensor, Tensor>(weights, compressed_indices_mapping);
  }

  return at::native::_rowwise_prune(
      weights, rowwise_prune_mask, compressed_indices_dtype);
}

Tensor& lengths_range_out(
    Tensor& output,
    const Tensor& t_in,
    const std::optional<std::vector<int64_t>>& shape) {
  TENSOR_ON_CPU(t_in);
  TENSOR_NDIM_EQUALS(t_in, 1);

  const auto t_in_contig = t_in.expect_contiguous();
  const auto num_seq = t_in_contig->numel();

  int64_t output_size;
  if (shape.has_value()) {
    output_size = c10::multiply_integers(shape.value());
  } else {
    // slow path: we need to calculate the output size from the lengths tensor
    output_size = 0;
    AT_DISPATCH_INDEX_TYPES(
        t_in_contig->scalar_type(), "lengths_range_compute_output_size", [&]() {
          const auto* input_data = t_in_contig->data_ptr<index_t>();
          output_size = c10::sum_integers(input_data, input_data + num_seq);
        });
  }

  at::native::resize_(output, {output_size}, std::nullopt);

  AT_DISPATCH_INDEX_TYPES(
      t_in_contig->scalar_type(), "lengths_range_compute", [&]() {
        const auto* input_data = t_in_contig->data_ptr<index_t>();
        auto* output_data = output.data_ptr<index_t>();

        index_t offset = 0;
        for (const auto i : c10::irange(num_seq)) {
          const index_t len = input_data[i];
          index_t* start = output_data + offset;
          TORCH_CHECK((output_size - len) >= offset);
          offset += len;
          TORCH_CHECK(len >= 0 && offset <= output_size);
          std::iota(
              start,
              start + len,
              0); // make the third argument the arg of this operator
        }
      });

  return output;
}

Tensor lengths_range(
    const Tensor& t_in,
    const std::optional<std::vector<int64_t>>& shape) {
  auto output = at::empty({0}, t_in.options());
  return lengths_range_out(output, t_in, shape);
}

// NOTE : _permute_data_kernel_cpu and _permute_lengths_cpu_kernel
// have to use the same grain size for consistent partitioning across threads.
template <bool has_weight, typename index_t, typename scalar_t>
void _permute_data_kernel_cpu(
    const int32_t T,
    const int32_t B,
    const index_t* const __restrict__ indices,
    const scalar_t* const __restrict__ weights,
    const int32_t* const __restrict__ permute,
    const index_t* const __restrict__ input_offsets,
    const int64_t* const __restrict__ output_offsets_per_thread_cumsum,
    index_t* const __restrict__ permuted_indices,
    scalar_t* const __restrict__ permuted_weights,
    const index_t* const __restrict__ permuted_lengths) {
  at::parallel_for(
      0, T * B, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        index_t output_start = output_offsets_per_thread_cumsum
            [at::get_thread_num() * FALSE_SHARING_PAD];
        int64_t t_begin = tb_begin / B;
        int64_t t_end = (tb_end + B - 1) / B;
        for (const auto t : c10::irange(t_begin, t_end)) {
          int64_t b_begin = (t == t_begin) ? tb_begin % B : 0;
          int64_t b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;
          for (const auto b : c10::irange(b_begin, b_end)) {
            index_t permuted_length = permuted_lengths[t * B + b];
            const index_t input_start = input_offsets[permute[t] * B + b];
            for (const auto i : c10::irange(permuted_length)) {
              permuted_indices[output_start + i] = indices[input_start + i];
              if (has_weight) {
                permuted_weights[output_start + i] = weights[input_start + i];
              }
            }
            output_start += permuted_length;
          } // for each b
        } // for each t
      }); // parallel_for T * B
}

std::tuple<Tensor, Tensor, std::optional<Tensor>> permute_sparse_features_cpu(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const std::optional<Tensor>& weights) {
  TENSOR_ON_CPU(permute);
  TENSOR_ON_CPU(lengths);
  TENSOR_ON_CPU(indices);
  TENSOR_ON_CPU(weights);

  // the following implementation requires lengths and indices has the same
  // dtype if usecase comes up that requires different dtype (e.g. int32 for
  // lengths and int64 for indices, this will give a better error msg for
  // debugging
  TENSORS_HAVE_SAME_TYPE(lengths, indices);

  TORCH_CHECK(
      lengths.dim() == 2,
      "The dimension of lengths tensor should be equal to 2 to correctly infer number of features and batch size.");

  const auto permute_contig = permute.expect_contiguous();
  const auto lengths_contig = lengths.expect_contiguous();
  const auto indices_contig = indices.expect_contiguous();
  // the features to permute over can be less or more with or without
  // repetitions
  const auto num_output_features = permute.numel();
  const auto B = lengths.sizes()[1];

  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;

  permuted_lengths = at::empty({num_output_features, B}, lengths.options());

  const auto lengths_size = lengths.numel();
  auto input_offsets = at::empty({lengths_size + 1}, lengths.options());

  int num_threads = at::get_num_threads();
  std::vector<int64_t> output_offsets_per_thread_cumsum(
      (num_threads + 1) * FALSE_SHARING_PAD, 0);

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_lengths_cpu_kernel", ([&] {
        _permute_2D_lengths_cpu_kernel(
            num_output_features,
            B,
            lengths_contig->data_ptr<index_t>(),
            lengths_size,
            permute.data_ptr<int32_t>(),
            permuted_lengths.data_ptr<index_t>(),
            input_offsets.data_ptr<index_t>(),
            output_offsets_per_thread_cumsum.data());
      })); // for each scalar_t

  auto permuted_lengths_sum =
      output_offsets_per_thread_cumsum[num_threads * FALSE_SHARING_PAD];
  permuted_indices = at::empty(permuted_lengths_sum, indices.options());
  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_data_kernel_1", ([&] {
        FBGEMM_DISPATCH_FLOAT_ONLY(
            weights.has_value() ? weights.value().scalar_type()
                                : at::ScalarType::Float,
            "permute_data_kernel_2",
            ([&] {
              if (weights.has_value()) {
                const auto weights_value_contig =
                    weights.value().expect_contiguous();
                permuted_weights =
                    at::empty(permuted_lengths_sum, weights.value().options());
                _permute_data_kernel_cpu<true, index_t, scalar_t>(
                    num_output_features,
                    B,
                    indices_contig->data_ptr<index_t>(),
                    weights_value_contig->data_ptr<scalar_t>(),
                    permute_contig->data_ptr<int32_t>(),
                    input_offsets.data_ptr<index_t>(),
                    output_offsets_per_thread_cumsum.data(),
                    permuted_indices.data_ptr<index_t>(),
                    permuted_weights.data_ptr<scalar_t>(),
                    permuted_lengths.data_ptr<index_t>());
              } else {
                _permute_data_kernel_cpu<false, index_t, scalar_t>(
                    num_output_features,
                    B,
                    indices_contig->data_ptr<index_t>(),
                    nullptr,
                    permute_contig->data_ptr<int32_t>(),
                    input_offsets.data_ptr<index_t>(),
                    output_offsets_per_thread_cumsum.data(),
                    permuted_indices.data_ptr<index_t>(),
                    nullptr,
                    permuted_lengths.data_ptr<index_t>());
              }
            })); // for each scalar_t
      })); // for each index_t
  return {permuted_lengths, permuted_indices, permuted_weights};
}

// A: m, batch_size, k
// B: batch_size, k, n
// bias: batch_size, n
// output: m, batch_size, n
Tensor permute102_baddbmm_permute102_cpu(
    const Tensor& bias,
    const Tensor& A,
    const Tensor& B) {
  TENSOR_ON_CPU(bias);
  TENSOR_ON_CPU(A);
  TENSOR_ON_CPU(B);
  TENSORS_ON_SAME_DEVICE(A, B);
  TENSORS_ON_SAME_DEVICE(A, bias);
  TENSOR_NDIM_EQUALS(A, 3);
  TENSOR_NDIM_EQUALS(B, 3);

  const auto m = A.size(0);
  const auto batch_size = B.size(0);
  const auto n = B.size(2);
  const auto k = A.size(2);
  TORCH_CHECK(B.size(0) == batch_size);
  TORCH_CHECK(B.size(1) == k);
  TORCH_CHECK(bias.size(0) == batch_size);
  TORCH_CHECK(bias.size(1) == n);

  auto output = at::empty({m, batch_size, n}, A.options());

  auto A_permute = at::permute(A, {1, 0, 2});
  auto bias_broadcast = at::unsqueeze(bias, 1);
  output = at::permute(
      at::baddbmm(bias_broadcast, A_permute, B, 1.0, 1.0), {1, 0, 2});

  return output;
}

template <typename index_t>
void _permute_lengths_cpu_kernel(
    const int32_t T,
    const int32_t B,
    const index_t* const __restrict__ lengths,
    int64_t lengths_size,
    const int32_t* const __restrict__ permute,
    index_t* const __restrict__ permuted_lengths,
    index_t* const __restrict__ input_offsets,
    int64_t* const __restrict__ output_offsets_per_thread_cumsum) {
  int num_threads = at::get_num_threads();
  std::vector<int> input_offsets_per_thread_cumsum(
      (num_threads + 1) * FALSE_SHARING_PAD, 0);

  // First parallel for: populate permuted_lengths, and compute per-thread
  // summation of lengths (input_offsets_per_thread_cumsum) and
  // permuted_lengths (output_offsets_per_thread_cumsum)
  at::parallel_for(
      0, T * B, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        index_t current_input_offset = 0;
        // Have a separate loop for summing up lengths because lengths_size
        // can be smaller than T * B.
        for (const auto tb :
             c10::irange(tb_begin, std::min(tb_end, lengths_size))) {
          current_input_offset += lengths[tb];
        }

        index_t current_output_offset = 0;
        int64_t t_begin = tb_begin / B;
        int64_t t_end = (tb_end + B - 1) / B;
        for (const auto t : c10::irange(t_begin, t_end)) {
          int64_t b_begin = (t == t_begin) ? tb_begin % B : 0;
          int64_t b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;
          for (const auto b : c10::irange(b_begin, b_end)) {
            auto permuted_length = lengths[permute[t] * B + b];
            permuted_lengths[t * B + b] = permuted_length;
            current_output_offset += permuted_length;
          }
        }
        input_offsets_per_thread_cumsum
            [(at::get_thread_num() + 1) * FALSE_SHARING_PAD] =
                current_input_offset;
        output_offsets_per_thread_cumsum
            [(at::get_thread_num() + 1) * FALSE_SHARING_PAD] =
                current_output_offset;
      });

  // Inter-thread reduction
  for (const auto t : c10::irange(1, num_threads)) {
    input_offsets_per_thread_cumsum[(t + 1) * FALSE_SHARING_PAD] +=
        input_offsets_per_thread_cumsum[t * FALSE_SHARING_PAD];
    output_offsets_per_thread_cumsum[(t + 1) * FALSE_SHARING_PAD] +=
        output_offsets_per_thread_cumsum[t * FALSE_SHARING_PAD];
  }

  // Second parallel for: populate input_offsets
  // NOTE: this works assuming the partitioning will be the same as the
  // first parallel_for.
  at::parallel_for(
      0, T * B, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        index_t current_input_offset = input_offsets_per_thread_cumsum
            [at::get_thread_num() * FALSE_SHARING_PAD];
        if (tb_begin < lengths_size) {
          input_offsets[tb_begin] = current_input_offset;
        }
        for (const auto tb :
             c10::irange(tb_begin, std::min(tb_end - 1, lengths_size))) {
          current_input_offset += lengths[tb];
          input_offsets[tb + 1] = current_input_offset;
        }
      });
  if (lengths_size >= T * B) {
    input_offsets[T * B] =
        input_offsets_per_thread_cumsum[num_threads * FALSE_SHARING_PAD];
  }

  // Handle cases when lengths_size > T * B
  for (const auto i : c10::irange(T * B, lengths_size)) {
    input_offsets[i + 1] = lengths[i] + input_offsets[i];
  }
}

template <typename index_t, typename scalar_t>
void _permute_embeddings_kernel_cpu(
    const int32_t T,
    const int32_t B,
    const scalar_t* const __restrict__ embeddings,
    const int32_t* const __restrict__ permute,
    const index_t* const __restrict__ input_offsets,
    const int64_t* const __restrict__ output_offsets_per_thread_cumsum,
    scalar_t* const __restrict__ permuted_embeddings,
    const index_t* const __restrict__ permuted_lengths) {
  at::parallel_for(
      0, T * B, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        index_t output_start = output_offsets_per_thread_cumsum
            [at::get_thread_num() * FALSE_SHARING_PAD];
        int64_t t_begin = tb_begin / B;
        int64_t t_end = (tb_end + B - 1) / B;
        for (const auto t : c10::irange(t_begin, t_end)) {
          int64_t b_begin = (t == t_begin) ? tb_begin % B : 0;
          int64_t b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;
          for (const auto b : c10::irange(b_begin, b_end)) {
            index_t permuted_length = permuted_lengths[t * B + b];
            const index_t input_start = input_offsets[permute[t] * B + b];
            for (const auto i : c10::irange(permuted_length)) {
              permuted_embeddings[output_start + i] =
                  embeddings[input_start + i];
            }
            output_start += permuted_length;
          } // for each b
        } // for each t
      }); // parallel_for T * B
}

std::tuple<Tensor, Tensor> permute_sequence_embeddings_cpu(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& embeddings) {
  // wrapper for permute_2D_sparse_data_cpu, kept for BC
  TENSOR_ON_CPU(permute);
  TENSOR_ON_CPU(lengths);
  TENSOR_ON_CPU(embeddings);

  TORCH_CHECK(
      lengths.dim() == 2,
      "The dimension of lengths tensor should be equal to 2"
      "to correctly infer number of features and batch size.");

  Tensor permuted_lengths;
  Tensor permuted_embeddings;
  std::optional<Tensor> weights_dummy;
  std::optional<int64_t> permuted_lengths_sum_dummy;

  const auto T = permute.numel();
  const auto B = lengths.size(1);

  permuted_lengths = at::empty({T, B}, lengths.options());

  // ignore the third element in the tuple
  std::tie(permuted_lengths, permuted_embeddings, std::ignore) =
      fbgemm_gpu::permute_2D_sparse_data_cpu(
          permute,
          lengths,
          embeddings,
          weights_dummy,
          permuted_lengths_sum_dummy);

  return {permuted_lengths, permuted_embeddings};
}

/// Map N dim tensor to N+1 dim based on lengths tensor.
/// Sequences that are shorter than the longest sequence are padded with zeros.
/// @param t_in         N dim Tensor.
/// @param lengths      1D int/long tensor contains the length in each of the
/// output.
/// @param max_length   The pre-defined max_length for the packed segments. -1
/// means autodetect
/// @return packed_tensor
///            packed_tensor        N + 1 dim Tensor where dim(1) is the max
///                                 length, dim(0) is the batch size.
Tensor pack_segments_forward_cpu(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length) {
  TENSOR_NDIM_IS_GE(t_in, 1);
  TENSOR_NDIM_EQUALS(lengths, 1);
  TORCH_CHECK(
      t_in.dtype() == at::ScalarType::Float ||
          t_in.dtype() == at::ScalarType::Double ||
          t_in.dtype() == at::ScalarType::Half ||
          t_in.dtype() == at::ScalarType::BFloat16,
      "t_in must be of type float, double, half, or bfloat16");
  TORCH_CHECK_GT(max_length, 0);

  const auto t_in_cont = t_in.expect_contiguous();
  Tensor packed_tensor;

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "pack_segments_cpu", ([&]() {
        const auto* const lengths_data = lengths.data_ptr<index_t>();

        // Shape of output is batch_size x max_len x ...
        auto shape = t_in_cont->sizes().vec(); // Get copy of current shape
        shape[0] = max_length; // Set first element to max_len
        shape.insert(
            shape.begin(), lengths.numel()); // Insert batch size at beginning
        packed_tensor = at::zeros(shape, t_in_cont->options());

        if (t_in_cont->sizes()[0] == 0) {
          return; // Return empty output (with the proper shape)
        }

        FBGEMM_DISPATCH_ALL_TYPES(
            t_in_cont->scalar_type(), "pack_segments_cpu-packing", ([&]() {
              const auto sizes =
                  t_in_cont->sizes().slice(1, t_in_cont->sizes().size() - 1);
              const auto block_size = c10::multiply_integers(sizes);
              const auto block_bytesize = t_in_cont->itemsize() * block_size;
              const auto* const data_ptr = t_in_cont->data_ptr<scalar_t>();
              auto* const out_data = packed_tensor.data_ptr<scalar_t>();
              int64_t start = 0;
              for (const auto i : c10::irange(lengths.sizes()[0])) {
                const auto len =
                    std::min(static_cast<int64_t>(lengths_data[i]), max_length);
                std::memcpy(
                    out_data + block_size * max_length * i, // dst
                    data_ptr + block_size * start, // src
                    len * block_bytesize);
                start += lengths_data[i];
              }
            }));
      }));

  return packed_tensor;
}

std::tuple<Tensor, std::optional<Tensor>> pack_segments_forward_cpu_v2(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length,
    const bool pad_minf,
    const bool return_presence_mask) {
  TENSOR_NDIM_IS_GE(t_in, 1);
  TENSOR_NDIM_EQUALS(lengths, 1);
  TORCH_CHECK(
      t_in.dtype() == at::ScalarType::Float ||
          t_in.dtype() == at::ScalarType::Half ||
          t_in.dtype() == at::ScalarType::BFloat16 ||
          t_in.dtype() == at::ScalarType::Int ||
          t_in.dtype() == at::ScalarType::Long,
      "t_in must be of type float, half, bfloat16, int or long");
  TORCH_CHECK_GT(max_length, 0);

  const auto t_in_cont = t_in.expect_contiguous();
  Tensor packed_tensor;
  std::optional<Tensor> presence_mask;

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "pack_segments_cpu", ([&]() {
        const auto* const lengths_data = lengths.data_ptr<index_t>();

        // Shape of output is batch_size x max_len x ...
        auto shape = t_in_cont->sizes().vec(); // Get copy of current shape
        shape[0] = max_length; // Set first element to max_len
        shape.insert(
            shape.begin(), lengths.numel()); // Insert batch size at beginning
        if (pad_minf) {
          // Downcasting double infinity to float should still give infinity
          packed_tensor = at::full(
              shape,
              -std::numeric_limits<double>::infinity(),
              t_in_cont->options());
        } else {
          packed_tensor = at::zeros(shape, t_in_cont->options());
        }

        bool* presence_mask_data = nullptr;
        if (return_presence_mask) {
          // Shape of presence is batch_size x max_len
          presence_mask = at::zeros({lengths.numel(), max_length}, at::kBool);
          presence_mask_data = presence_mask->data_ptr<bool>();
        }

        if (t_in_cont->sizes()[0] == 0) {
          return; // Return empty output (with the proper shape)
        }

        FBGEMM_DISPATCH_ALL_TYPES(
            t_in_cont->scalar_type(), "pack_segments_cpu-packing", ([&]() {
              const auto sizes =
                  t_in_cont->sizes().slice(1, t_in_cont->sizes().size() - 1);
              const auto block_size = c10::multiply_integers(sizes);
              const auto block_bytesize = t_in_cont->itemsize() * block_size;
              const auto* const data_ptr = t_in_cont->data_ptr<scalar_t>();
              auto* const out_data = packed_tensor.data_ptr<scalar_t>();
              int64_t start = 0;
              for (const auto i : c10::irange(lengths.sizes()[0])) {
                const auto len =
                    std::min(static_cast<int64_t>(lengths_data[i]), max_length);
                std::memcpy(
                    out_data + block_size * max_length * i, // dst
                    data_ptr + block_size * start, // src
                    len * block_bytesize);
                if (return_presence_mask) {
                  std::fill(
                      presence_mask_data + max_length * i,
                      presence_mask_data + max_length * i + len,
                      true);
                }
                start += lengths_data[i];
              }
            }));
      }));

  return {packed_tensor, presence_mask};
}

/// Map N+1 dim tensor to N dim based on lengths tensor
/// Sequences that are shorter than the longest sequence are padded with zeros.
/// @param data         N+1 dim Tensor.
/// @param lengths      1D int/long tensor contains the length in each of the
/// input.
/// @param total_length Sum of elements in the 1D tensor legnths
/// @param max_length   The pre-defined max_length for the packed segments. -1
/// means autodetect
/// @return unpacked_tensor N-dimensional tensor
Tensor pack_segments_backward_cpu(
    const Tensor& data,
    const Tensor& lengths,
    const int64_t total_length,
    const int64_t max_length) {
  TENSOR_NDIM_IS_GE(data, 2);
  TENSOR_NDIM_EQUALS(lengths, 1);
  TORCH_CHECK(
      data.sizes()[0] == lengths.sizes()[0],
      "LENGTHS and DATA must match in dimension 0");
  TORCH_CHECK(
      data.dtype() == at::ScalarType::Float ||
          data.dtype() == at::ScalarType::Double ||
          data.dtype() == at::ScalarType::Half ||
          data.dtype() == at::ScalarType::BFloat16,
      "data must be of type float, double, half, or bfloat16");
  TORCH_CHECK(
      max_length == data.sizes()[1],
      "max_length should be equal to the second dimension of the packed segments");

  c10::MaybeOwned<Tensor> data_contig = data.expect_contiguous();
  c10::MaybeOwned<Tensor> lengths_contig = lengths.expect_contiguous();
  Tensor unpacked_tensor; // The output tensor

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "unpack_segments_cpu", ([&]() {
        const auto* const lengths_data = lengths_contig->data_ptr<index_t>();

        // Create output tensor of appropriate dimensions
        auto shape = data.sizes().vec();
        shape.erase(shape.begin());
        shape[0] = total_length;
        unpacked_tensor = at::empty(shape, data.options());
        TORCH_CHECK(unpacked_tensor.is_contiguous());

        if (!(data.sizes()[0] &&
              data.sizes()[1])) { // TODO: What does this mean?
          return;
        }

        FBGEMM_DISPATCH_ALL_TYPES(
            data.scalar_type(), "unpack_segments_cpu-unpacking", ([&]() {
              const auto sizes = data.sizes().slice(2, data.sizes().size() - 2);
              const auto block_size = c10::multiply_integers(sizes);
              const auto block_bytesize = data.itemsize() * block_size;
              const auto* const data_ptr = data_contig->data_ptr<scalar_t>();
              auto* const out_data = unpacked_tensor.data_ptr<scalar_t>();

              int64_t start = 0;
              for (const auto i : c10::irange(lengths.sizes()[0])) {
                int64_t len = lengths_data[i];
                len =
                    std::min(static_cast<int64_t>(lengths_data[i]), max_length);
                std::memcpy(
                    out_data + block_size * start, // dst
                    data_ptr + block_size * data.sizes()[1] * i, // src
                    len * block_bytesize);
                start += len;
              }
            }));
      }));

  return unpacked_tensor;
}

Tensor pack_segments_cpu(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length) {
  return pack_segments_forward_cpu(t_in, lengths, max_length);
}

std::tuple<Tensor, std::optional<Tensor>> pack_segments_cpu_v2(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length,
    const bool pad_minf,
    const bool return_presence_mask // https://fburl.com/code/ol14vkbn
) {
  const auto& res = pack_segments_forward_cpu_v2(
      t_in, lengths, max_length, pad_minf, return_presence_mask);
  if (return_presence_mask) {
    return {std::get<0>(res), std::get<1>(res)};
  }
  return {std::get<0>(res), std::nullopt};
}

torch::autograd::variable_list group_index_select_dim0_autograd_impl(
    at::TensorList all_indices_input,
    const int64_t group_size) {
  return GroupIndexSelectDim0Op::apply(all_indices_input, group_size);
}

std::pair<std::vector<Tensor>, std::vector<Tensor>>
group_index_select_dim0_unpack(
    at::TensorList all_indices_input,
    const int64_t group_size) {
  std::vector<Tensor> indices_group;
  std::vector<Tensor> input_group;

  indices_group.reserve(group_size);
  input_group.reserve(group_size);

  for (const auto i : c10::irange(group_size)) {
    indices_group.push_back(all_indices_input[i]);
    input_group.push_back(all_indices_input[group_size + i]);
  }

  TORCH_CHECK(group_size == static_cast<int64_t>(indices_group.size()));

  return std::make_pair(input_group, indices_group);
}

torch::autograd::variable_list group_index_select_dim0(
    at::TensorList input_group,
    at::TensorList indices_group) {
  const auto group_size = indices_group.size();
  std::vector<Tensor> output_group;

  if (group_size == 0) {
    return std::vector<Tensor>();
  }

  // Pack input_group and indices_group into TensorList
  std::vector<Tensor> all_indices_input_vec;
  all_indices_input_vec.reserve(group_size * 2);

  for (const Tensor& index : indices_group) {
    all_indices_input_vec.push_back(index);
  }
  for (const Tensor& input : input_group) {
    all_indices_input_vec.push_back(input);
  }

  at::TensorList all_indices_input_tensor = all_indices_input_vec;

  static auto forward_op =
      at::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::group_index_select_dim0_gpu_impl", "")
          .typed<decltype(group_index_select_dim0_autograd_impl)>();
  auto res = forward_op.call(all_indices_input_tensor, group_size);
  TORCH_CHECK(res.size() == group_size + 2);
  // only return the outputs (the first group_size elements)
  res.resize(group_size);
  return res;
}

torch::autograd::variable_list group_index_select_dim0_forward_impl_cpu(
    at::TensorList all_indices_input,
    const int64_t group_size) {
  // Unpack from TensorList
  auto [input_group, indices_group] =
      group_index_select_dim0_unpack(all_indices_input, group_size);

  std::vector<Tensor> output_group;
  for (const auto i : c10::irange(group_size)) {
    output_group.push_back(
        at::index_select(input_group[i], 0, indices_group[i]));
  }

  // to match return format in CUDA implementation
  // (group_size outputs, 1 args_tensor, 1 saved_data)
  output_group.push_back(at::empty({0}, at::TensorOptions().dtype(at::kLong)));
  output_group.push_back(at::empty({0}, at::TensorOptions().dtype(at::kLong)));
  return output_group;
}

torch::autograd::variable_list group_index_select_dim0_backward_impl_cpu(
    at::TensorList all_inputs,
    c10::SymIntArrayRef output_shape_group_ref) {
  TORCH_CHECK(all_inputs.size() > 2);
  // all input size =  group_size * 2 (from grads, indices)
  // + 1 args_tensor + 1 saved_data + 1 first output
  const int64_t group_size = (all_inputs.size() - 3) / 2;

  auto grad_output_group = std::vector<Tensor>(
      all_inputs.cbegin(), all_inputs.cbegin() + group_size);

  auto indices_group = std::vector<Tensor>(
      all_inputs.cbegin() + group_size, all_inputs.cbegin() + 2 * group_size);

  const Tensor fwd_input = all_inputs[2 * group_size + 2];
  const int64_t output_dim = fwd_input.dim();

  std::vector<int64_t> output_shape_group;
  output_shape_group.reserve(output_shape_group_ref.size());
  for (const auto& i : output_shape_group_ref) {
    output_shape_group.push_back(i.as_int_unchecked());
  }

  // Returning 3 outputs - to match as CUDA implementation:
  // 1) group_size Variable()'s for indices -> fill with empty tensors
  // 2) group_size gradients for inputs
  // 3) 1 Variable() for group_size (this will be added in autograd backward)
  std::vector<Tensor> outputs;
  outputs.reserve(group_size * 2 + 1);

  // 1) Add group_size Variable()'s for indices
  // c10::irange cannot be used in here as it
  // triggers a build error of i being an unused variable.
  // Add empty tensor with zero size here to make __torch_dispatch__ work for
  // the backward op. Those empty tensors will be replaced with
  // torch::autograd::Variable() in autograd backward.
  for (auto i = 0; i < group_size; i++) {
    outputs.push_back(at::empty({0}, at::TensorOptions().dtype(at::kLong)));
  }

  // 2) group_size gradients for inputs
  for (const auto i : c10::irange(group_size)) {
    // get input grad shape
    const auto grad_input_shape = std::vector<int64_t>(
        output_shape_group.begin() + i * output_dim,
        output_shape_group.begin() + (i + 1) * output_dim);

    // initialize grad input
    auto grad_input = at::zeros({grad_input_shape}, fwd_input.options());
    const auto grad_output = grad_output_group[i];
    const auto indices = indices_group[i];

    // add gradient for each index in the group
    for (const auto j : c10::irange(indices.numel())) {
      const auto index = indices[j];
      grad_input[index] += grad_output[j];
    }
    outputs.push_back(grad_input);
  }
  return outputs;
}

torch::autograd::variable_list group_index_select_dim0_decomposed(
    at::TensorList input_group,
    at::TensorList indices_group) {
  int num_groups = input_group.size();
  TORCH_CHECK(num_groups == (int)indices_group.size())
  std::vector<Tensor> output_group;
  for (const auto i : c10::irange(num_groups)) {
    output_group.push_back(
        at::index_select(input_group[i], 0, indices_group[i]));
  }
  return output_group;
}

torch::autograd::variable_list GroupIndexSelectDim0Op::forward(
    torch::autograd::AutogradContext* ctx,
    at::TensorList all_indices_input,
    const int64_t group_size) {
  at::AutoDispatchBelowADInplaceOrView guard;
  static auto forward_op =
      at::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::group_index_select_dim0_gpu_impl", "")
          .typed<decltype(group_index_select_dim0_forward_impl_cpu)>();
  auto result = forward_op.call(all_indices_input, group_size);
  TORCH_CHECK(static_cast<int64_t>(result.size()) == group_size + 2);

  auto [input_group, indices_group] =
      group_index_select_dim0_unpack(all_indices_input, group_size);
  const auto input_dim = input_group[0].dim();
  std::vector<c10::SymInt> input_shape_group;
  input_shape_group.reserve(group_size * input_dim);

  for (const auto i : c10::irange(group_size)) {
    const auto& input = input_group[i];
    // Copy input shape
    auto input_shape = input.sym_sizes().vec();
    input_shape_group.insert(
        input_shape_group.end(), input_shape.begin(), input_shape.end());
  }

  for (int i = 0; i < input_shape_group.size(); i++) {
    ctx->saved_data["input_shape_group_" + std::to_string(i)] =
        input_shape_group[i];
  }
  // save indices, args_tensor, saved_data
  auto saved_tensors = std::vector<at::Tensor>(indices_group);
  saved_tensors.insert(
      saved_tensors.end(), result.cbegin() + group_size, result.cend());
  saved_tensors.push_back(input_group[0]);
  ctx->save_for_backward(saved_tensors);

  return result;
}

torch::autograd::variable_list GroupIndexSelectDim0Op::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output_group) {
  TORCH_CHECK(grad_output_group.size() >= 2);
  if (grad_output_group.size() == 2) {
    // empty outputs
    return torch::autograd::variable_list(1);
  }
  // remove redundant grads
  auto group_size = grad_output_group.size() - 2;
  grad_output_group.resize(group_size);

  auto saved_tensors = ctx->get_saved_variables();
  TORCH_CHECK(saved_tensors.size() == group_size + 3);
  std::vector<c10::SymInt> output_shape_group;
  int i = 0;
  while (true) {
    if (ctx->saved_data.find("input_shape_group_" + std::to_string(i)) ==
        ctx->saved_data.end()) {
      break;
    }
    output_shape_group.push_back(
        ctx->saved_data["input_shape_group_" + std::to_string(i)].toSymInt());
    i++;
  }
  grad_output_group.insert(
      grad_output_group.end(), saved_tensors.begin(), saved_tensors.end());

  static auto backward_op =
      at::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::group_index_select_dim0_gpu_backward", "")
          .typed<decltype(group_index_select_dim0_backward_impl_cpu)>();
  auto res = backward_op.call(grad_output_group, output_shape_group);
  // 1) Add group_size Variable()'s for indices
  // Replace all empty tensors with Variable(). This must be done after the
  // op.call to make __torch_dispatch__ work for the backward op.
  std::fill(res.begin(), res.begin() + group_size, torch::autograd::Variable());
  // 3) Add 1 Variable() for group_size
  res.push_back({});
  return res;
}

namespace {
Tensor index_select_dim0(
    const Tensor& input,
    const Tensor& indices,
    std::optional<int64_t> /*consecutive_range_start*/,
    std::optional<int64_t> /*consecutive_range_length*/,
    std::optional<bool> /*skip_indices_sorting_fwd*/) {
  return at::index_select(input, 0, indices);
}

Tensor bottom_k_per_row(
    const Tensor& input,
    const Tensor& k_offsets,
    const bool requires_unique) {
  auto num_cols = input.size(-1);
  Tensor input_reshaped = input.reshape({-1, num_cols});
  auto input_accessor = input_reshaped.accessor<int64_t, 2>();
  auto k_offsets_accessor = k_offsets.accessor<int64_t, 1>();

  // Assume fixed k is used if there are only two offsets
  bool use_fixed_k = k_offsets.size(0) == 2;
  const int64_t fixed_k =
      use_fixed_k ? k_offsets_accessor[1] - k_offsets_accessor[0] : 0;

  // Create output tensor
  Tensor output = at::empty(
      {use_fixed_k ? input_reshaped.size(0) * fixed_k
                   : k_offsets_accessor[k_offsets.numel() - 1]},
      input.options());
  auto output_accessor = output.accessor<int64_t, 1>();

  at::parallel_for(
      0, input_reshaped.size(0), 1, [&](int64_t start, int64_t end) {
        for (const auto i : c10::irange(start, end)) {
          auto start_k_offset =
              use_fixed_k ? i * fixed_k : k_offsets_accessor[i];
          auto k = use_fixed_k ? fixed_k
                               : k_offsets_accessor[i + 1] - start_k_offset;
          TORCH_CHECK(k >= 0);

          if (k == 0) {
            continue;
          }

          if (requires_unique) {
            std::set<int64_t> s;

            for (auto j : c10::irange(num_cols)) {
              s.insert(input_accessor[i][j]);
              if (s.size() == static_cast<size_t>(k)) {
                break;
              }
            }
            TORCH_CHECK(
                s.size() == static_cast<size_t>(k),
                "too skewed distribution (alpha too big)")
            int j = 0;
            for (int64_t x : s) {
              output_accessor[start_k_offset + j] = x;
              ++j;
            }
          } else {
            for (auto j : c10::irange(k)) {
              output_accessor[start_k_offset + j] = input_accessor[i][j];
            }
          }
        }
      });

  return use_fixed_k ? output.reshape({input.size(0), -1, fixed_k}) : output;
}

} // namespace

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
#ifdef HAS_IMPL_ABSTRACT_PYSTUB
  m.impl_abstract_pystub(
      "fbgemm_gpu.sparse_ops",
      "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_py");
#endif
  m.def(
      "permute_sparse_data(Tensor permute, Tensor lengths, Tensor values, Tensor? weights=None, SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
  m.def(
      "permute_2D_sparse_data(Tensor permute, Tensor lengths, Tensor values, Tensor? weights=None, SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)",
      {PT2_COMPLIANT_TAG});
  m.def(
      "permute_2D_sparse_data_input1D(Tensor permute, Tensor lengths, Tensor values, SymInt stride, Tensor? weights=None, SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
  m.def(
      "permute_1D_sparse_data(Tensor permute, Tensor lengths, Tensor values, Tensor? weights=None, SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)",
      {PT2_COMPLIANT_TAG});
  m.def("invert_permute(Tensor permute) -> Tensor");
  m.def(
      "expand_into_jagged_permute(Tensor permute, Tensor input_offset, Tensor output_offset, SymInt output_size) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "populate_bucketized_permute(Tensor lengths, Tensor bucketized_lengths, Tensor bucket_mapping) -> Tensor");
  m.def(
      "block_bucketize_sparse_features(Tensor lengths, Tensor indices, bool bucketize_pos, bool sequence, Tensor block_sizes, SymInt my_size, Tensor? weights=None, Tensor? batch_size_per_feature=None, SymInt max_B= -1, Tensor[]? block_bucketize_pos=None, bool keep_orig_idx=False, Tensor? total_num_blocks=None) -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?)");
  m.def(
      "block_bucketize_sparse_features_inference(Tensor lengths, Tensor indices, bool bucketize_pos, bool sequence, Tensor block_sizes, SymInt my_size, Tensor? weights=None, Tensor? batch_size_per_feature=None, SymInt max_B= -1, Tensor[]? block_bucketize_pos=None, bool return_bucket_mapping=False, bool keep_orig_idx=False, Tensor? total_num_blocks=None) -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?)");
  m.def(
      "bucketize_sparse_features(Tensor lengths, Tensor indices, bool bucketize_pos, SymInt my_size, Tensor? weights=None) -> (Tensor, Tensor, Tensor?, Tensor?)");
  m.def(
      "reorder_batched_sequence_embeddings(Tensor cat_sequence_embeddings_offsets, Tensor cat_sequence_embeddings, Tensor reordered_cat_sequence_embeddings_offsets, Tensor batch_offsets, SymInt num_items_in_batch) -> Tensor");
  m.def(
      "reorder_batched_ad_lengths(Tensor cat_ad_lengths, Tensor batch_offsets, SymInt num_ads_in_batch, bool broadcast_lengths=False, SymInt max_batch_size=0) -> Tensor");
  m.def(
      "reorder_batched_ad_indices(Tensor cat_ad_offsets, Tensor cat_ad_indices, Tensor reordered_cat_ad_offsets, Tensor batch_offsets, SymInt num_ads_in_batch, bool broadcast_indices=False, SymInt num_indices_after_broadcast=-1) -> Tensor");
  m.def(
      "cat_reorder_batched_ad_indices(Tensor cat_ad_offsets, Tensor[] cat_ad_indices, Tensor reordered_cat_ad_offsets, Tensor batch_offsets, SymInt num_ads_in_batch, bool broadcast_indices, SymInt total_num_indices, bool pinned_memory=False, SymInt max_batch_size=0) -> Tensor");
  m.def("offsets_range(Tensor offsets, SymInt range_size) -> Tensor");
  m.def(
      "batched_unary_embeddings(Tensor weight, Tensor table_offsets, Tensor offsets, Tensor indices) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "histogram_binning_calibration(Tensor logit, Tensor bin_num_examples, Tensor bin_num_positives, float positive_weight, float lower_bound, float upper_bound, SymInt bin_ctr_in_use_after, float bin_ctr_weight_value) -> (Tensor, Tensor)");
  m.def(
      "histogram_binning_calibration_by_feature(Tensor logit, Tensor segment_value, Tensor segment_lengths, SymInt num_segments, Tensor bin_num_examples, Tensor bin_num_positives, SymInt num_bins, float positive_weight, float lower_bound, float upper_bound, SymInt bin_ctr_in_use_after, float bin_ctr_weight_value) -> (Tensor, Tensor)");
  m.def(
      "generic_histogram_binning_calibration_by_feature(Tensor logit, Tensor segment_value, Tensor segment_lengths, SymInt num_segments, Tensor bin_num_examples, Tensor bin_num_positives, Tensor bin_boundaries, float positive_weight, SymInt bin_ctr_in_use_after, float bin_ctr_weight_value) -> (Tensor, Tensor)");
  m.def(
      "segment_sum_csr(SymInt batch_size, Tensor csr_seg, Tensor values) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "embedding_bag_rowwise_prune(Tensor weight, Tensor indicator, float threshold, ScalarType compressed_indices_dtype, bool abs=True, SymInt min_num_rows=0, float? min_save_ratio=1.0) -> (Tensor, Tensor)");
  m.def("lengths_range(Tensor t_in, SymInt[]? shape=None) -> Tensor");
  m.def(
      "lengths_range_out(Tensor output, Tensor t_in, SymInt[]? shape=None) -> Tensor");
  m.def(
      "permute_sparse_features(Tensor permute, Tensor lengths, Tensor indices, Tensor? weights=None) -> (Tensor, Tensor, Tensor?)",
      {PT2_COMPLIANT_TAG});
  m.def("Bfloat16QuantizedToFloat(Tensor input) -> Tensor");
  m.def("FloatToBfloat16Quantized(Tensor input) -> Tensor");
  m.def(
      "permute102_baddbmm_permute102(Tensor bias, Tensor A, Tensor B) -> Tensor");
  m.def(
      "permute_sequence_embeddings(Tensor permute, Tensor lengths, Tensor embeddings) -> (Tensor, Tensor)");
  m.def(
      "pack_segments(Tensor t_in, Tensor lengths, SymInt max_length) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "pack_segments_v2(Tensor t_in, Tensor lengths, SymInt max_length, bool pad_minf=False, bool return_presence_mask=False) -> (Tensor packed_segments, Tensor? presence_mask)",
      {PT2_COMPLIANT_TAG});
  m.def(
      "pack_segments_backward(Tensor data, Tensor lengths, SymInt total_length, SymInt max_length) -> Tensor");
  // A specialization of at::index_select for selecting dim 0
  //
  // The consecutive_range_start and consecutive_range_length arguments are for
  // the special case where indices are selected from a consecutive range
  // [consecutive_range_start, consecutive_range_start +
  // consecutive_range_length).
  //
  // For the consecutive indices range case, we can skip the unique indices
  // computation step in the backward operation because we can infer them from
  // the consecutive indices range.  This assumption saves computation as well
  // as a host-device synchronization that occurs in the unique operation of
  // Torch.
  //
  // If indices are not selected from a consecutive range, we perform the
  // unique indices computation step in the backward operation.
  //
  // skip_indices_sorting_fwd is for skipping indices sorting in forward
  m.def(
      "index_select_dim0(Tensor input, Tensor indices, SymInt? consecutive_range_start=0, SymInt? consecutive_range_length=0, bool? skip_indices_sorting_fwd=None) -> Tensor");
  m.def(
      "group_index_select_dim0(Tensor[] input_group, Tensor[] indices_group) -> Tensor[]",
      {PT2_COMPLIANT_TAG});
  // group_index_select_dim0_gpu helper functions - not defined for CPU!
  m.def(
      "group_index_select_dim0_gpu_impl(Tensor[] inputs, int group_size) -> Tensor[]");
  m.def(
      "group_index_select_dim0_gpu_backward(Tensor[] inputs, SymInt[] output_shape_group) -> Tensor[]");
  // This is an one-off op to be used in split_embedding_utils.py for zipf
  // generation w/o replacement along dim=-1. If requires_unique=True, find
  // smallest unique k.  If the number of unique elements is less than k,
  // errors out. If requires_unique=False, copy the top k elements into a new
  // buffer. If k_offsets's length is 2, assume that k is fixed (using length =
  // 2 instead of 1 to trigger the fixed-k assumption to keep the k_offsets
  // semantic).
  m.def(
      "bottom_k_per_row(Tensor input, Tensor k_offsets, bool requires_unique) -> Tensor");
  m.def(
      "keyed_jagged_index_select_dim1(Tensor values, Tensor lengths, Tensor offsets, Tensor indices, SymInt batch_size, Tensor? weights=None, SymInt? selected_lengths_sum=None) -> Tensor[]",
      {PT2_COMPLIANT_TAG});
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  DISPATCH_TO_CPU(
      "permute_sparse_data", fbgemm_gpu::permute_2D_sparse_data_cpu);
  DISPATCH_TO_CPU(
      "permute_2D_sparse_data", fbgemm_gpu::permute_2D_sparse_data_cpu);
  DISPATCH_TO_CPU(
      "permute_2D_sparse_data_input1D",
      fbgemm_gpu::permute_2D_sparse_data_input1D_cpu);
  DISPATCH_TO_CPU(
      "permute_1D_sparse_data", fbgemm_gpu::permute_1D_sparse_data_cpu);
  DISPATCH_TO_CPU("invert_permute", fbgemm_gpu::invert_permute_cpu);
  DISPATCH_TO_CPU(
      "expand_into_jagged_permute", fbgemm_gpu::expand_into_jagged_permute_cpu);
  DISPATCH_TO_CPU(
      "populate_bucketized_permute",
      fbgemm_gpu::populate_bucketized_permute_cpu);
  DISPATCH_TO_CPU(
      "block_bucketize_sparse_features",
      fbgemm_gpu::block_bucketize_sparse_features_cpu);
  DISPATCH_TO_CPU(
      "block_bucketize_sparse_features_inference",
      fbgemm_gpu::block_bucketize_sparse_features_inference_cpu);
  DISPATCH_TO_CPU(
      "bucketize_sparse_features", fbgemm_gpu::bucketize_sparse_features_cpu);
  DISPATCH_TO_CPU(
      "reorder_batched_ad_lengths", fbgemm_gpu::reorder_batched_ad_lengths_cpu);
  DISPATCH_TO_CPU(
      "reorder_batched_ad_indices", fbgemm_gpu::reorder_batched_ad_indices_cpu);
  DISPATCH_TO_CPU(
      "cat_reorder_batched_ad_indices",
      fbgemm_gpu::cat_reorder_batched_ad_indices_cpu);
  DISPATCH_TO_CPU(
      "reorder_batched_sequence_embeddings",
      fbgemm_gpu::reorder_batched_sequence_embeddings_cpu);
  DISPATCH_TO_CPU("offsets_range", fbgemm_gpu::offsets_range_cpu);
  DISPATCH_TO_CPU(
      "batched_unary_embeddings",
      fbgemm_gpu::batched_unary_embeddings_forward_cpu);
  DISPATCH_TO_CPU(
      "histogram_binning_calibration",
      fbgemm_gpu::histogram_binning_calibration_cpu);
  DISPATCH_TO_CPU(
      "histogram_binning_calibration_by_feature",
      fbgemm_gpu::histogram_binning_calibration_by_feature_cpu);
  DISPATCH_TO_CPU(
      "generic_histogram_binning_calibration_by_feature",
      fbgemm_gpu::generic_histogram_binning_calibration_by_feature_cpu);
  DISPATCH_TO_CPU("segment_sum_csr", fbgemm_gpu::segment_sum_csr_cpu);
  DISPATCH_TO_CPU(
      "embedding_bag_rowwise_prune", fbgemm_gpu::embedding_bag_rowwise_prune);
  DISPATCH_TO_CPU("lengths_range", fbgemm_gpu::lengths_range);
  DISPATCH_TO_CPU("lengths_range_out", fbgemm_gpu::lengths_range_out);
  DISPATCH_TO_CPU(
      "permute_sparse_features", fbgemm_gpu::permute_sparse_features_cpu);
  DISPATCH_TO_CPU(
      "FloatToBfloat16Quantized", fbgemm_gpu::_float_to_bfloat16_cpu);
  DISPATCH_TO_CPU(
      "Bfloat16QuantizedToFloat", fbgemm_gpu::_bfloat16_to_float_cpu);
  DISPATCH_TO_CPU(
      "permute102_baddbmm_permute102",
      fbgemm_gpu::permute102_baddbmm_permute102_cpu);
  DISPATCH_TO_CPU(
      "permute_sequence_embeddings",
      fbgemm_gpu::permute_sequence_embeddings_cpu);
  DISPATCH_TO_CPU("pack_segments", fbgemm_gpu::pack_segments_cpu);
  DISPATCH_TO_CPU("pack_segments_v2", fbgemm_gpu::pack_segments_cpu_v2);
  DISPATCH_TO_CPU(
      "pack_segments_backward", fbgemm_gpu::pack_segments_backward_cpu);
  DISPATCH_TO_CPU("index_select_dim0", fbgemm_gpu::index_select_dim0);
  DISPATCH_TO_CPU(
      "group_index_select_dim0", fbgemm_gpu::group_index_select_dim0);
  DISPATCH_TO_CPU(
      "group_index_select_dim0_gpu_impl",
      fbgemm_gpu::group_index_select_dim0_forward_impl_cpu);
  DISPATCH_TO_CPU(
      "group_index_select_dim0_gpu_backward",
      fbgemm_gpu::group_index_select_dim0_backward_impl_cpu);
  DISPATCH_TO_CPU("bottom_k_per_row", fbgemm_gpu::bottom_k_per_row);
}

TORCH_LIBRARY_IMPL(fbgemm, Autograd, m) {
  m.impl("pack_segments", &fbgemm_gpu::pack_segments_autograd);
  m.impl("pack_segments_v2", &fbgemm_gpu::pack_segments_autograd_v2);
  m.impl(
      "group_index_select_dim0_gpu_impl",
      &fbgemm_gpu::group_index_select_dim0_autograd_impl);
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradCPU, m) {
  m.impl("group_index_select_dim0", &fbgemm_gpu::group_index_select_dim0);
  m.impl(
      "group_index_select_dim0_gpu_impl",
      &fbgemm_gpu::group_index_select_dim0_autograd_impl);
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl(
      "group_index_select_dim0", TORCH_FN(fbgemm_gpu::group_index_select_dim0));
}

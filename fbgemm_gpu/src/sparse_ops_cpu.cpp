/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <functional>

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <torch/library.h>
#include "ATen/Parallel.h"

#include "fbgemm_gpu/sparse_ops_utils.h"

namespace {

// To avoid multiple threads are touching the same cache line.
// Assume cache line size is 64B and element size is at least 4B like float or
// int32.
constexpr int FALSE_SHARING_PAD = 16;

} // namespace

using Tensor = at::Tensor;

namespace fbgemm_gpu {

Tensor native_empty_like(const Tensor& self) {
  return at::native::empty_like(
      self,
      optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt(),
      c10::nullopt);
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
void _permute_indices_weights_kernel_cpu(
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
    typename offset_t,
    typename index_t,
    typename scalar_t>
void _block_bucketize_sparse_features_cpu(
    Tensor lengths,
    Tensor indices,
    c10::optional<Tensor> weights,
    bool bucketize_pos,
    Tensor block_sizes,
    int64_t my_size,
    Tensor new_lengths,
    Tensor new_indices,
    c10::optional<Tensor> new_weights,
    c10::optional<Tensor> new_pos,
    c10::optional<Tensor> unbucketize_permute) {
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
  scalar_t* weights_data;
  scalar_t* new_weights_data;
  index_t* new_pos_data;
  index_t* unbucketize_permute_data;
  offset_t* new_lengths_data = new_lengths.data_ptr<offset_t>();
  offset_t* new_offsets_data = new_offsets.data_ptr<offset_t>();
  index_t* new_indices_data = new_indices.data_ptr<index_t>();
  index_t* block_sizes_data = block_sizes.data_ptr<index_t>();
  using uindex_t = std::make_unsigned_t<index_t>;
  using uoffset_t = std::make_unsigned_t<offset_t>;

  if (sequence) {
    unbucketize_permute_data = unbucketize_permute.value().data_ptr<index_t>();
  }
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
  for (const auto t : c10::irange(T)) {
    auto blk_size = block_sizes_data[t];
    for (const auto b : c10::irange(B)) {
      const auto b_t = t * B + b;
      const offset_t rowstart = offsets_data[b_t];
      const offset_t rowend = offsets_data[b_t + 1];
      for (const auto i : c10::irange(rowstart, rowend)) {
        // We have use cases using none-hashed raw indices that can be either
        // negative or larger than embedding table hash_size (blk_size *
        // my_size). In cases of none-hashed indices we need to ensure
        // bucketization can distribute them into different ranks and within
        // range of blk_size, we expect the later embedding module to take care
        // of hashing indices calculation.
        const auto idx = static_cast<int64_t>(indices_data[i]);
        const auto p =
            idx < blk_size * my_size ? idx / blk_size : idx % my_size;
        new_lengths_data[p * lengths_size + b_t]++;
      }
    }
  }

  // bucketize nonzeros
  prefix_sum(new_lengths_size, new_lengths_data, new_offsets_data);
  assert(new_offsets_data[new_lengths_size] == new_indices.numel());
  for (const auto t : c10::irange(T)) {
    auto blk_size = block_sizes_data[t];
    for (const auto b : c10::irange(B)) {
      const auto b_t = t * B + b;
      const offset_t rowstart = offsets_data[b_t];
      const offset_t rowend = offsets_data[b_t + 1];
      for (const auto i : c10::irange(rowstart, rowend)) {
        // We have use cases using none-hashed raw indices that can be either
        // negative or larger than embedding table hash_size (blk_size *
        // my_size). In cases of none-hashed indices we need to ensure
        // bucketization can distribute them into different ranks and within
        // range of blk_size, we expect the later embedding module to take care
        // of hashing indices calculation.
        const auto idx = static_cast<int64_t>(indices_data[i]);
        const auto p =
            idx < blk_size * my_size ? idx / blk_size : idx % my_size;
        const uindex_t new_idx = idx % blk_size;
        const uoffset_t pos = new_offsets_data[p * lengths_size + b_t];
        new_indices_data[pos] = new_idx;
        if (sequence) {
          unbucketize_permute_data[i] = pos;
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
  }
}

std::tuple<Tensor, Tensor, c10::optional<Tensor>> permute_sparse_data_cpu(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum) {
  TENSOR_ON_CPU(permute);
  TENSOR_ON_CPU(lengths);
  TENSOR_ON_CPU(indices);
  TENSOR_ON_CPU(weights);

  const auto permute_contig = permute.expect_contiguous();
  const auto lengths_contig = lengths.expect_contiguous();
  const auto indices_contig = indices.expect_contiguous();
  // the data to permute over can be less or more with or without
  // repetitions
  const auto T = permute.numel();
  const auto B = lengths.view({lengths.sizes()[0], -1}).sizes()[1];

  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;

  permuted_lengths = at::empty({T, B}, lengths.options());

  const auto lengths_size = lengths.numel();
  auto input_offsets = at::empty({lengths_size + 1}, lengths.options());

  int num_threads = at::get_num_threads();
  std::vector<int64_t> output_offsets_per_thread_cumsum(
      (num_threads + 1) * FALSE_SHARING_PAD, 0);

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_lengths_cpu_kernel", ([&] {
        _permute_lengths_cpu_kernel(
            T,
            B,
            lengths_contig->data_ptr<index_t>(),
            lengths_size,
            permute.data_ptr<int32_t>(),
            permuted_lengths.data_ptr<index_t>(),
            input_offsets.data_ptr<index_t>(),
            output_offsets_per_thread_cumsum.data());
      })); // for each scalar_t

  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size =
        output_offsets_per_thread_cumsum[num_threads * FALSE_SHARING_PAD];
  }
  permuted_indices = at::empty(permuted_indices_size, indices.options());
  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_indices_weights_kernel_1", ([&] {
        using offsets_t = index_t;
        AT_DISPATCH_ALL_TYPES(
            indices.scalar_type(), "permute_indices_weights_kernel_2", ([&] {
              using indices_t = scalar_t;
              AT_DISPATCH_FLOATING_TYPES(
                  weights.has_value() ? weights.value().scalar_type()
                                      : at::ScalarType::Float,
                  "permute_indices_weights_kernel_3",
                  ([&] {
                    using weights_t = scalar_t;
                    if (weights.has_value()) {
                      const auto weights_value_contig =
                          weights.value().expect_contiguous();
                      permuted_weights = at::empty(
                          permuted_indices_size, weights.value().options());
                      _permute_indices_weights_kernel_cpu<
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
                          permuted_weights.data_ptr<weights_t>(),
                          permuted_lengths.data_ptr<offsets_t>());
                    } else {
                      _permute_indices_weights_kernel_cpu<
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
                  })); // for each weights_t
            })); // for each indices_t
      })); // for each offsets_t
  return {permuted_lengths, permuted_indices, permuted_weights};
}

std::tuple<
    Tensor,
    Tensor,
    c10::optional<Tensor>,
    c10::optional<Tensor>,
    c10::optional<Tensor>>
block_bucketize_sparse_features_cpu(
    Tensor lengths,
    Tensor indices,
    bool bucketize_pos,
    bool sequence,
    Tensor block_sizes,
    int64_t my_size,
    c10::optional<Tensor> weights) {
  const auto lengths_size = lengths.numel();
  const auto new_lengths_size = lengths_size * my_size;
  auto new_lengths = at::zeros({new_lengths_size}, lengths.options());
  auto new_indices = native_empty_like(indices);
  Tensor new_weights;
  Tensor new_pos;
  Tensor unbucketize_permute;
  if (bucketize_pos) {
    new_pos = native_empty_like(indices);
  }
  if (weights.has_value()) {
    const auto lengths_sum = indices.numel();
    Tensor weights_value = weights.value();
    new_weights = native_empty_like(weights_value);
    if (sequence) {
      unbucketize_permute = at::empty({lengths_sum}, indices.options());
      AT_DISPATCH_INDEX_TYPES(
          lengths.scalar_type(),
          "block_bucketize_sparse_features_weights_cpu_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(),
                "block_bucketize_sparse_features_weights_cpu_2",
                ([&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "bucketize_sparse_features_weights_cpu_3",
                      ([&] {
                        _block_bucketize_sparse_features_cpu<
                            true,
                            true,
                            offset_t,
                            index_t,
                            scalar_t>(
                            lengths,
                            indices,
                            weights,
                            bucketize_pos,
                            block_sizes,
                            my_size,
                            new_lengths,
                            new_indices,
                            new_weights,
                            new_pos,
                            unbucketize_permute);
                      }));
                }));
          }));
    } else {
      AT_DISPATCH_INDEX_TYPES(
          lengths.scalar_type(),
          "block_bucketize_sparse_features_weights_cpu_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(),
                "block_bucketize_sparse_features_weights_cpu_2",
                ([&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "bucketize_sparse_features_weights_cpu_3",
                      ([&] {
                        _block_bucketize_sparse_features_cpu<
                            false,
                            true,
                            offset_t,
                            index_t,
                            scalar_t>(
                            lengths,
                            indices,
                            weights,
                            bucketize_pos,
                            block_sizes,
                            my_size,
                            new_lengths,
                            new_indices,
                            new_weights,
                            new_pos,
                            unbucketize_permute);
                      }));
                }));
          }));
    }
  } else {
    if (sequence) {
      const auto lengths_sum = indices.numel();
      unbucketize_permute = at::empty({lengths_sum}, indices.options());
      AT_DISPATCH_INDEX_TYPES(
          lengths.scalar_type(), "block_bucketize_sparse_features_cpu_1", ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(),
                "block_bucketize_sparse_features_cpu_2",
                ([&] {
                  _block_bucketize_sparse_features_cpu<
                      true,
                      false,
                      offset_t,
                      index_t,
                      std::nullptr_t>(
                      lengths,
                      indices,
                      weights,
                      bucketize_pos,
                      block_sizes,
                      my_size,
                      new_lengths,
                      new_indices,
                      new_weights,
                      new_pos,
                      unbucketize_permute);
                }));
          }));
    } else {
      AT_DISPATCH_INDEX_TYPES(
          lengths.scalar_type(), "block_bucketize_sparse_features_cpu_1", ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(),
                "block_bucketize_sparse_features_cpu_2",
                ([&] {
                  _block_bucketize_sparse_features_cpu<
                      false,
                      false,
                      offset_t,
                      index_t,
                      std::nullptr_t>(
                      lengths,
                      indices,
                      weights,
                      bucketize_pos,
                      block_sizes,
                      my_size,
                      new_lengths,
                      new_indices,
                      new_weights,
                      new_pos,
                      unbucketize_permute);
                }));
          }));
    }
  }
  return {new_lengths, new_indices, new_weights, new_pos, unbucketize_permute};
}

// 1D exclusive scan: output[i] = input[i-1] + input[i-2] + input[i-3]
// Used as a helper to several functions below.
template <class T, class U>
U exclusive_scan_ptrs_cpu(
    const int64_t N,
    const T* const input,
    U* const output) {
  U cumsum = 0;
  for (const auto i : c10::irange(N)) {
    output[i] = cumsum;
    cumsum += input[i];
  }
  return cumsum;
}

Tensor asynchronous_exclusive_cumsum_cpu(const Tensor& t_in) {
  TENSOR_ON_CPU(t_in);

  const auto t_in_contig = t_in.expect_contiguous();
  auto output = native_empty_like(*t_in_contig);
  AT_DISPATCH_ALL_TYPES(
      t_in_contig->type(), "asynchronous_exclusive_cumsum_cpu_kernel", ([&] {
        exclusive_scan_ptrs_cpu(
            t_in_contig->numel(),
            t_in_contig->data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
      }));
  return output;
}

Tensor asynchronous_inclusive_cumsum_cpu(const Tensor& t_in) {
  TENSOR_ON_CPU(t_in);

  const auto t_in_contig = t_in.expect_contiguous();
  auto output = native_empty_like(*t_in_contig);
  AT_DISPATCH_ALL_TYPES(
      t_in_contig->type(), "asynchronous_inclusive_cumsum_cpu_kernel", ([&] {
        scalar_t cumsum = 0;
        const auto* input_ptr = t_in_contig->data_ptr<scalar_t>();
        const auto N = t_in_contig->numel();
        auto* output_ptr = output.data_ptr<scalar_t>();

        for (const auto i : c10::irange(N)) {
          cumsum += input_ptr[i];
          output_ptr[i] = cumsum;
        }
      }));
  return output;
}

Tensor asynchronous_complete_cumsum_cpu(const Tensor& t_in) {
  TENSOR_ON_CPU(t_in);
  TORCH_CHECK(t_in.dim() == 1);

  const auto t_in_contig = t_in.expect_contiguous();
  auto output = at::zeros({t_in.numel() + 1}, t_in.options());
  AT_DISPATCH_ALL_TYPES(
      t_in_contig->type(), "asynchronous_complete_cumsum_cpu_kernel", ([&] {
        const auto N = t_in_contig->numel();
        const auto last_sum = exclusive_scan_ptrs_cpu(
            N, t_in_contig->data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
        output.data_ptr<scalar_t>()[N] = last_sum;
      }));
  return output;
}

template <class T>
void reorder_batched_ad_lengths_(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    Tensor& output) {
  const int64_t nB = batch_offsets.numel() - 1;
  const int64_t nT = cat_ad_lengths.numel() / num_ads_in_batch;

  const auto* batch_offsets_data = batch_offsets.data_ptr<T>();

  for (auto b = 0; b < nB; b++) {
    const auto num_ads_b = batch_offsets_data[b + 1] - batch_offsets_data[b];
    for (auto t = 0; t < nT; t++) {
      const int32_t input_segment_start =
          nT * batch_offsets_data[b] + t * num_ads_b;
      const int32_t output_segment_start =
          t * num_ads_in_batch + batch_offsets_data[b];
      for (auto i = 0; i < num_ads_b; i++) {
        output[output_segment_start + i] =
            cat_ad_lengths[input_segment_start + i];
      }
    }
  }
}

Tensor reorder_batched_ad_lengths_cpu(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch) {
  TENSOR_ON_CPU(cat_ad_lengths);
  TENSOR_ON_CPU(batch_offsets);

  Tensor reordered_cat_ad_lengths = at::empty_like(cat_ad_lengths);
  AT_DISPATCH_ALL_TYPES(
      batch_offsets.type(), "reorder_batched_ad_lengths_cpu_kernel", ([&] {
        reorder_batched_ad_lengths_<scalar_t>(
            cat_ad_lengths,
            batch_offsets,
            num_ads_in_batch,
            reordered_cat_ad_lengths);
      }));

  return reordered_cat_ad_lengths;
}

template <class T>
void reorder_batched_ad_indices_cpu_(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    Tensor& output) {
  const int64_t nB = batch_offsets.numel() - 1;
  const int64_t nT = (cat_ad_offsets.numel() - 1) / num_ads_in_batch;

  const auto* batch_offsets_data = batch_offsets.data_ptr<int32_t>();
  const auto* cat_ad_offsets_data = cat_ad_offsets.data_ptr<int32_t>();
  const auto* reordered_cat_ad_offsets_data =
      reordered_cat_ad_offsets.data_ptr<int32_t>();
  const auto* cat_ad_indices_data = cat_ad_indices.data_ptr<T>();

  for (auto b = 0; b < nB; b++) {
    const auto num_ads_b = batch_offsets_data[b + 1] - batch_offsets_data[b];
    for (auto t = 0; t < nT; t++) {
      const int32_t input_segment_offset_start =
          nT * batch_offsets_data[b] + t * num_ads_b;
      const int32_t input_segment_offset_end =
          nT * batch_offsets_data[b] + t * num_ads_b + num_ads_b;

      const auto input_segment_start =
          cat_ad_offsets_data[input_segment_offset_start];
      const auto input_segment_end =
          cat_ad_offsets_data[input_segment_offset_end];

      const auto output_segment_offset_start =
          t * num_ads_in_batch + batch_offsets_data[b];
      const auto output_segment_start =
          reordered_cat_ad_offsets_data[output_segment_offset_start];

      for (auto i = 0; i < input_segment_end - input_segment_start; i++) {
        output[output_segment_start + i] =
            cat_ad_indices_data[input_segment_start + i];
      }
    }
  }
}

Tensor reorder_batched_ad_indices_cpu(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch) {
  TENSOR_ON_CPU(cat_ad_offsets);
  TENSOR_ON_CPU(cat_ad_indices);
  TENSOR_ON_CPU(reordered_cat_ad_offsets);
  TENSOR_ON_CPU(batch_offsets);

  Tensor reordered_cat_ad_indices = at::empty_like(cat_ad_indices);
  AT_DISPATCH_ALL_TYPES(
      cat_ad_indices.type(), "reorder_batched_ad_indices_cpu_kernel", ([&] {
        reorder_batched_ad_indices_cpu_<scalar_t>(
            cat_ad_offsets,
            cat_ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            num_ads_in_batch,
            reordered_cat_ad_indices);
      }));

  return reordered_cat_ad_indices;
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

  // Only supporting limited data types for now.
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float);

  // Make sure the index_t are consistent among table_offsets, offsets and
  // indices
  TORCH_CHECK(table_offsets.scalar_type() == offsets.scalar_type());
  TORCH_CHECK(table_offsets.scalar_type() == indices.scalar_type());

  auto output = at::empty({N, B, T}, weight.options());
  auto* output_data = output.data_ptr<float>();
  const auto* weight_data = weight.data_ptr<float>();

  AT_DISPATCH_INDEX_TYPES(
      table_offsets.scalar_type(), "unary_indices", ([&] {
        const index_t* table_offsets_data = table_offsets.data_ptr<index_t>();
        const index_t* offsets_data = offsets.data_ptr<index_t>();
        const index_t* indices_data = indices.data_ptr<index_t>();
        const index_t sum_E = table_offsets_data[T];

        for (const auto n : c10::irange(N)) {
          for (const auto b : c10::irange(B)) {
            for (const auto t : c10::irange(T)) {
              const index_t indices_start = offsets_data[t * B + b];
              const index_t indices_end = offsets_data[t * B + b + 1];
              float sum = 0;
              for (const auto l : c10::irange(indices_start, indices_end)) {
                const index_t idx =
                    n * sum_E + table_offsets_data[t] + indices_data[l];
                // Since we don't care about the performance of CPU impl, adding
                // the boundary check here. OOB will result in undefined
                // behavior for GPU impl.
                TORCH_CHECK(idx < weight.numel());
                sum += weight_data[idx];
              }
              output_data[(n * B + b) * T + t] = sum;
            }
          }
        }
      }));

  return output;
}

template <typename index_t, typename scalar_t>
void jagged_2d_to_dense_forward_kernel(
    int32_t B,
    int32_t max_L,
    int32_t D,
    const index_t* offsets,
    const scalar_t* values_data,
    scalar_t* padded_values_data) {
  const auto block_size = max_L * D;
  const auto embedding_byte_size = D * sizeof(scalar_t);
  for (auto b = 0; b < B; ++b) {
    auto start_idx = offsets[b];
    auto end_idx = offsets[b + 1];
    auto length = end_idx - start_idx;
    if (length > max_L) {
      length = max_L;
    }
    auto padding_length = max_L - length;
    memcpy(
        &padded_values_data[b * block_size],
        &values_data[start_idx * D],
        length * embedding_byte_size);
    memset(
        &padded_values_data[b * block_size + length * D],
        0,
        padding_length * embedding_byte_size);
  }
}

Tensor
jagged_2d_to_dense_forward_cpu(Tensor values, Tensor offsets, int64_t max_L) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  const auto B = offsets.numel() - 1;
  const auto D = values.size(1);
  const auto values_contig = values.expect_contiguous();
  const auto offsets_contig = offsets.expect_contiguous();

  if (values.size(0) == 0) {
    return at::zeros({B, max_L, D}, values.options());
  }

  auto padded_values = at::empty({B, max_L, D}, values.options());
  AT_DISPATCH_INDEX_TYPES(
      offsets_contig->scalar_type(),
      "jagged_2d_to_dense_forward_by_offsets",
      ([&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            values_contig->scalar_type(),
            "jagged_2d_to_dense_forward_by_values",
            ([&]() {
              jagged_2d_to_dense_forward_kernel(
                  B,
                  max_L,
                  D,
                  offsets_contig->data_ptr<index_t>(),
                  values_contig->data_ptr<scalar_t>(),
                  padded_values.data_ptr<scalar_t>());
            }));
      }));

  return padded_values;
}

template <typename index_t, typename scalar_t>
void jagged_1d_to_dense_kernel(
    int64_t B,
    int64_t max_L,
    scalar_t padding_value,
    const index_t* offsets,
    const scalar_t* values_data,
    scalar_t* padded_values_data) {
  const auto block_size = max_L;
  const auto value_byte_size = sizeof(scalar_t);
  for (auto b = 0; b < B; ++b) {
    auto start_idx = offsets[b];
    auto end_idx = offsets[b + 1];
    auto length = end_idx - start_idx;
    if (length > max_L) {
      // Guard against the case that lengths[b] > max_L. This is
      // a valid use case.
      length = max_L;
    }
    auto padding_length = max_L - length;
    memcpy(
        &padded_values_data[b * block_size],
        &values_data[start_idx],
        length * value_byte_size);
    for (int l = 0, offset = b * block_size + length; l < padding_length;
         ++l, ++offset) {
      padded_values_data[offset] = padding_value;
    }
  }
}

Tensor jagged_1d_to_dense_cpu(
    Tensor values,
    Tensor offsets,
    int64_t max_L,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  const auto B = offsets.numel() - 1;
  const auto values_contig = values.expect_contiguous();
  const auto offsets_contig = offsets.expect_contiguous();

  if (values.size(0) == 0 && padding_value == 0) {
    return at::zeros({B, max_L}, values.options());
  }

  auto padded_values = at::empty({B, max_L}, values.options());
  AT_DISPATCH_INDEX_TYPES(
      offsets_contig->scalar_type(), "jagged_1d_to_dense_1", ([&]() {
        AT_DISPATCH_ALL_TYPES(
            values_contig->scalar_type(), "jagged_1d_to_dense_2", ([&]() {
              jagged_1d_to_dense_kernel<index_t, scalar_t>(
                  B,
                  max_L,
                  padding_value,
                  offsets_contig->data_ptr<index_t>(),
                  values_contig->data_ptr<scalar_t>(),
                  padded_values.data_ptr<scalar_t>());
            }));
      }));

  return padded_values;
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logit.type(), "histogram_binning_calibration_cpu", [&]() {
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

template <typename T>
void _histogram_binning_calibration_by_feature_cpu_kernel(
    const int64_t num_logits,
    const int64_t num_bins,
    const int64_t num_segments,
    const int64_t num_lengths,
    const double recalibrate_value,
    const double step,
    const int64_t bin_ctr_in_use_after,
    const double bin_ctr_weight_value,
    const T* const logit_data,
    const int64_t* const segment_value_data,
    const int64_t* const segment_lengths_data,
    const double* const bin_num_examples_data,
    const double* const bin_num_positives_data,
    int64_t* const dense_segment_value_data,
    T* const calibrated_prediction_data,
    int64_t* const bin_ids_data) {
  int k = 0;
  for (const auto i : c10::irange(num_lengths)) {
    if (segment_lengths_data[i] > 0) {
      // Add 1 to distinguish between 0 inserted by densification vs. original
      // value.
      dense_segment_value_data[i] = segment_value_data[k] + 1;
      ++k;
    }
  }

  for (const auto i : c10::irange(num_logits)) {
    const T pre_sigmoid = logit_data[i] + recalibrate_value;
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
      at::zeros({logit.numel()}, segment_value.options());
  Tensor calibrated_prediction = at::empty_like(logit);
  Tensor bin_ids = at::empty({logit.numel()}, logit.options().dtype(at::kLong));
  const double recalibrate_value = std::log(positive_weight);
  const double step =
      (upper_bound - lower_bound) / static_cast<double>(num_bins);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logit.type(), "histogram_binning_calibration_by_feature_cpu", [&]() {
        _histogram_binning_calibration_by_feature_cpu_kernel<scalar_t>(
            logit.numel(),
            num_bins,
            num_segments,
            segment_lengths.numel(),
            recalibrate_value,
            step,
            bin_ctr_in_use_after,
            bin_ctr_weight_value,
            logit.data_ptr<scalar_t>(),
            segment_value.data_ptr<int64_t>(),
            segment_lengths.data_ptr<int64_t>(),
            bin_num_examples.data_ptr<double>(),
            bin_num_positives.data_ptr<double>(),
            dense_segment_value.data_ptr<int64_t>(),
            calibrated_prediction.data_ptr<scalar_t>(),
            bin_ids.data_ptr<int64_t>());
      });

  return std::make_tuple(calibrated_prediction, bin_ids);
}

template <typename T>
void _generic_histogram_binning_calibration_by_feature_cpu_kernel(
    const int64_t num_logits,
    const int64_t num_bins,
    const int64_t num_segments,
    const int64_t num_lengths,
    const double recalibrate_value,
    const int64_t bin_ctr_in_use_after,
    const double bin_ctr_weight_value,
    const T* const logit_data,
    const int64_t* const segment_value_data,
    const int64_t* const segment_lengths_data,
    const double* const bin_num_examples_data,
    const double* const bin_num_positives_data,
    const double* const bin_boundaries,
    int64_t* const dense_segment_value_data,
    T* const calibrated_prediction_data,
    int64_t* const bin_ids_data) {
  int k = 0;
  for (const auto i : c10::irange(num_lengths)) {
    if (segment_lengths_data[i] > 0) {
      // Add 1 to distinguish between 0 inserted by densification vs. original
      // value.
      dense_segment_value_data[i] = segment_value_data[k] + 1;
      ++k;
    }
  }

  for (const auto i : c10::irange(num_logits)) {
    const T pre_sigmoid = logit_data[i] + recalibrate_value;
    const double uncalibrated = 1.0 / (1.0 + std::exp(-pre_sigmoid));

    const int curr_bin_id =
        std::lower_bound(
            bin_boundaries, bin_boundaries + num_bins, uncalibrated) -
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
      at::zeros({logit.numel()}, segment_value.options());
  Tensor calibrated_prediction = at::empty_like(logit);
  Tensor bin_ids = at::empty({logit.numel()}, logit.options().dtype(at::kLong));
  const double recalibrate_value = std::log(positive_weight);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logit.type(),
      "generic_histogram_binning_calibration_by_feature_cpu",
      [&]() {
        _generic_histogram_binning_calibration_by_feature_cpu_kernel<scalar_t>(
            logit.numel(),
            bin_boundaries.numel() + 1,
            num_segments,
            segment_lengths.numel(),
            recalibrate_value,
            bin_ctr_in_use_after,
            bin_ctr_weight_value,
            logit.data_ptr<scalar_t>(),
            segment_value.data_ptr<int64_t>(),
            segment_lengths.data_ptr<int64_t>(),
            bin_num_examples.data_ptr<double>(),
            bin_num_positives.data_ptr<double>(),
            bin_boundaries.data_ptr<double>(),
            dense_segment_value.data_ptr<int64_t>(),
            calibrated_prediction.data_ptr<scalar_t>(),
            bin_ids.data_ptr<int64_t>());
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
  AT_DISPATCH_ALL_TYPES(values.type(), "_segment_sum_csr_cpu", ([&] {
                          _segment_sum_csr_cpu_kernel<scalar_t>(
                              csr_seg.numel() - 1,
                              batch_size,
                              csr_seg.data_ptr<int>(),
                              values.data_ptr<scalar_t>(),
                              output.data_ptr<scalar_t>());
                        }));
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
    const c10::optional<double>& min_save_ratio) {
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

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "permute_sparse_data(Tensor permute, Tensor lengths, Tensor values, Tensor? weights=None, int? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
  m.def(
      "block_bucketize_sparse_features(Tensor lengths, Tensor indices, bool bucketize_pos, bool sequence, Tensor block_sizes, int my_size, Tensor? weights=None) -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?)");
  m.def("asynchronous_exclusive_cumsum(Tensor t_in) -> Tensor");
  m.def("asynchronous_inclusive_cumsum(Tensor t_in) -> Tensor");
  m.def("asynchronous_complete_cumsum(Tensor t_in) -> Tensor");
  m.def(
      "reorder_batched_ad_lengths(Tensor cat_ad_lengths, Tensor batch_offsets, int num_ads_in_batch) -> Tensor");
  m.def(
      "reorder_batched_ad_indices(Tensor cat_ad_offsets, Tensor cat_ad_indices, Tensor reordered_cat_ad_offsets, Tensor batch_offsets, int num_ads_in_batch) -> Tensor");
  m.def("offsets_range(Tensor offsets, int range_size) -> Tensor");
  m.def(
      "batched_unary_embeddings(Tensor weight, Tensor table_offsets, Tensor offsets, Tensor indices) -> Tensor");
  m.def(
      "jagged_2d_to_dense(Tensor values, Tensor offsets, int max_sequence_length) -> Tensor");
  m.def(
      "jagged_1d_to_dense(Tensor values, Tensor offsets, int max_sequence_length, int padding_value) -> Tensor");

  m.def(
      "stacked_jagged_2d_to_dense_forward(Tensor values, Tensor lengths, int[] offset_per_key, int[] max_lengths_per_key) -> (Tensor[], Tensor[])");
  m.def(
      "stacked_jagged_2d_to_dense_backward(int B, int D, int total_L, Tensor[] grad_padded_values_per_key, Tensor[] offsets_tensor_per_key, int[] offset_per_key) -> Tensor");

  m.def(
      "stacked_jagged_1d_to_dense(Tensor values, Tensor lengths, int[] offset_per_key, int[] max_lengths_per_key, int padding_value) -> Tensor[]");
  m.def(
      "histogram_binning_calibration(Tensor logit, Tensor bin_num_examples, Tensor bin_num_positives, float positive_weight, float lower_bound, float upper_bound, int bin_ctr_in_use_after, float bin_ctr_weight_value) -> (Tensor, Tensor)");
  m.def(
      "histogram_binning_calibration_by_feature(Tensor logit, Tensor segment_value, Tensor segment_lengths, int num_segments, Tensor bin_num_examples, Tensor bin_num_positives, int num_bins, float positive_weight, float lower_bound, float upper_bound, int bin_ctr_in_use_after, float bin_ctr_weight_value) -> (Tensor, Tensor)");
  m.def(
      "generic_histogram_binning_calibration_by_feature(Tensor logit, Tensor segment_value, Tensor segment_lengths, int num_segments, Tensor bin_num_examples, Tensor bin_num_positives, Tensor bin_boundaries, float positive_weight, int bin_ctr_in_use_after, float bin_ctr_weight_value) -> (Tensor, Tensor)");
  m.def(
      "segment_sum_csr(int batch_size, Tensor csr_seg, Tensor values) -> Tensor");
  m.def(
      "embedding_bag_rowwise_prune(Tensor weight, Tensor indicator, float threshold, ScalarType compressed_indices_dtype, bool abs=True, int min_num_rows=0, float? min_save_ratio=1.0) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl("permute_sparse_data", fbgemm_gpu::permute_sparse_data_cpu);
  m.impl(
      "block_bucketize_sparse_features",
      fbgemm_gpu::block_bucketize_sparse_features_cpu);
  m.impl(
      "asynchronous_exclusive_cumsum",
      fbgemm_gpu::asynchronous_exclusive_cumsum_cpu);
  m.impl(
      "asynchronous_inclusive_cumsum",
      fbgemm_gpu::asynchronous_inclusive_cumsum_cpu);
  m.impl(
      "asynchronous_complete_cumsum",
      fbgemm_gpu::asynchronous_complete_cumsum_cpu);
  m.impl(
      "reorder_batched_ad_lengths", fbgemm_gpu::reorder_batched_ad_lengths_cpu);
  m.impl(
      "reorder_batched_ad_indices", fbgemm_gpu::reorder_batched_ad_indices_cpu);
  m.impl("offsets_range", fbgemm_gpu::offsets_range_cpu);
  m.impl(
      "batched_unary_embeddings",
      fbgemm_gpu::batched_unary_embeddings_forward_cpu);
  m.impl("jagged_2d_to_dense", fbgemm_gpu::jagged_2d_to_dense_forward_cpu);
  m.impl("jagged_1d_to_dense", fbgemm_gpu::jagged_1d_to_dense_cpu);
  m.impl(
      "histogram_binning_calibration",
      fbgemm_gpu::histogram_binning_calibration_cpu);
  m.impl(
      "histogram_binning_calibration_by_feature",
      fbgemm_gpu::histogram_binning_calibration_by_feature_cpu);
  m.impl(
      "generic_histogram_binning_calibration_by_feature",
      fbgemm_gpu::generic_histogram_binning_calibration_by_feature_cpu);
  m.impl("segment_sum_csr", fbgemm_gpu::segment_sum_csr_cpu);
  m.impl(
      "embedding_bag_rowwise_prune", fbgemm_gpu::embedding_bag_rowwise_prune);
}

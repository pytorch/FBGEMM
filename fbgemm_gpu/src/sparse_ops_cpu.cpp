/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <torch/library.h>
#include "ATen/Parallel.h"

#include "fbgemm_gpu/sparse_ops_utils.h"

namespace at {

namespace {
// To avoid multiple threads are touching the same cache line.
// Assume cache line size is 64B and element size is at least 4B like float or
// int32.
constexpr int FALSE_SHARING_PAD = 16;

} // namespace

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
            [get_thread_num() * FALSE_SHARING_PAD];
        int64_t t_begin = tb_begin / B;
        int64_t t_end = (tb_end + B - 1) / B;
        for (int t = t_begin; t < t_end; ++t) {
          int64_t b_begin = (t == t_begin) ? tb_begin % B : 0;
          int64_t b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;
          for (int b = b_begin; b < b_end; ++b) {
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
  int num_threads = get_num_threads();
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
        for (int t = t_begin; t < t_end; ++t) {
          int64_t b_begin = (t == t_begin) ? tb_begin % B : 0;
          int64_t b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;
          for (int b = b_begin; b < b_end; ++b) {
            auto permuted_length = lengths[permute[t] * B + b];
            permuted_lengths[t * B + b] = permuted_length;
            current_output_offset += permuted_length;
          }
        }
        input_offsets_per_thread_cumsum
            [(get_thread_num() + 1) * FALSE_SHARING_PAD] = current_input_offset;
        output_offsets_per_thread_cumsum
            [(get_thread_num() + 1) * FALSE_SHARING_PAD] =
                current_output_offset;
      });

  // Inter-thread reduction
  for (int t = 1; t < num_threads; ++t) {
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
            [get_thread_num() * FALSE_SHARING_PAD];
        if (tb_begin < lengths_size) {
          input_offsets[tb_begin] = current_input_offset;
        }
        for (int tb = tb_begin; tb < std::min(tb_end - 1, lengths_size); ++tb) {
          current_input_offset += lengths[tb];
          input_offsets[tb + 1] = current_input_offset;
        }
      });
  if (lengths_size >= T * B) {
    input_offsets[T * B] =
        input_offsets_per_thread_cumsum[num_threads * FALSE_SHARING_PAD];
  }

  // Handle cases when lengths_size > T * B
  for (auto i = T * B; i < lengths_size; ++i) {
    input_offsets[i + 1] = lengths[i] + input_offsets[i];
  }
}

template <bool sequence, bool has_weight, typename offset_t, typename index_t, typename scalar_t>
void _block_bucketize_sparse_features_cpu(
    at::Tensor lengths,
    at::Tensor indices,
    c10::optional<at::Tensor> weights,
    bool bucketize_pos,
    at::Tensor block_sizes,
    int64_t my_size,
    at::Tensor new_lengths,
    at::Tensor new_indices,
    c10::optional<at::Tensor> new_weights,
    c10::optional<at::Tensor> new_pos,
    c10::optional<at::Tensor> unbucketize_permute) {
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
        const uindex_t idx = static_cast<uindex_t>(indices_data[i]);
        const uindex_t p =
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
        const uindex_t idx = static_cast<uindex_t>(indices_data[i]);
        const uindex_t p =
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

  int num_threads = get_num_threads();
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
                                      : ScalarType::Float,
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
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>
block_bucketize_sparse_features_cpu(
    at::Tensor lengths,
    at::Tensor indices,
    bool bucketize_pos,
    bool sequence,
    at::Tensor block_sizes,
    int64_t my_size,
    c10::optional<at::Tensor> weights) {
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

} // namespace at

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "permute_sparse_data(Tensor permute, Tensor lengths, Tensor values, Tensor? weights=None, int? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
  m.def(
      "block_bucketize_sparse_features(Tensor lengths, Tensor indices, bool bucketize_pos, bool sequence, Tensor block_sizes, int my_size, Tensor? weights=None) -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?)");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl("permute_sparse_data", at::permute_sparse_data_cpu);
  m.impl("block_bucketize_sparse_features", at::block_bucketize_sparse_features_cpu);
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

DLL_PUBLIC Tensor linearize_cache_indices_cpu(
    const Tensor& /*cache_hash_size_cumsum*/,
    const Tensor& indices,
    const Tensor& /*offsets*/,
    const std::optional<Tensor>& /*B_offsets*/,
    const int64_t /*max_B*/,
    const int64_t /*indices_base_offset*/) {
  return at::empty_like(indices);
}

DLL_PUBLIC Tensor linearize_cache_indices_from_row_idx_cpu(
    Tensor /*cache_hash_size_cumsum*/,
    Tensor /*update_table_indices*/,
    Tensor update_row_indices) {
  return at::empty_like(update_row_indices);
}

DLL_PUBLIC Tensor linearize_cache_indices_meta(
    const Tensor& /*cache_hash_size_cumsum*/,
    const Tensor& indices,
    const Tensor& /*offsets*/,
    const std::optional<Tensor>& /*B_offsets*/,
    const int64_t /*max_B*/,
    const int64_t /*indices_base_offset*/) {
  return at::empty_like(indices, indices.options().dtype(at::kLong));
}

/**
 * CPU implementation for computing unique indices from a 1D tensor of linear
 * indices.
 *
 * This function processes a tensor of linear indices and returns the unique
 * values along with optional metadata (counts and inverse mapping). The
 * implementation uses stable sorting to ensure deterministic ordering of
 * duplicate values, matching the reference Python implementation.
 *
 * Example:
 *     Input:
 *         linear_indices = [20, 0, 10, 10, 0]
 *         max_indices = 20
 *         compute_count = true
 *         compute_inverse_indices = true
 *     Output:
 *         unique_indices = [0, 10, 20, x, x]  (dtype: int64, x is
 *             uninitialized)
 *         unique_indices_length = [3]  (dtype: int32)
 *         unique_indices_count = [2, 2, 1, x, x]  (dtype: int32, 0 appears 2
 *             times, 10 appears 2 times, 20 appears 1 time)
 *         linear_index_positions_sorted = [1, 4, 2, 3, 0]  (dtype: int32,
 *             positions that sort the input:
 *             linear_indices[[1,4,2,3,0]] = [0,0,10,10,20])
 *
 * @param linear_indices 1D input tensor containing linear indices to process
 *     (dtype: int32 or int64). Must be 1D and have at most INT32_MAX
 *     elements.
 * @param max_indices Maximum number of unique indices expected (dtype: int64,
 *     currently unused, present to match GPU interface and API compatibility).
 * @param compute_count If true, computes and returns the count of each unique
 *     index in the output (dtype: bool).
 * @param compute_inverse_indices If true, computes the original positions of
 *     elements in sorted order using stable sort (dtype: bool).
 *
 * @return A tuple containing:
 *     - unique_indices_output: Tensor of size `linear_indices` that stores
 *       unique values in sorted order (dtype: same as input; first `num_unique`
 *       elements are valid, rest are uninitialized)
 *     - unique_indices_length: Tensor of size 1 containing number of unique
 *       indices (dtype: int32)
 *     - unique_indices_count: Optional tensor (if compute_count=true) of size
 *       `linear_indices` that contains an occurrence count for each unique
 *       value (dtype: int32), else std::nullopt
 *     - linear_index_positions_sorted: Optional tensor (dtype: int32) (if
 *       compute_inverse_indices=true) of size `linear_indices` that contains
 *       original positions such that
 * linear_indices[linear_index_positions_sorted[i]] is the i postion in the
 * sorted order.
 *
 */
DLL_PUBLIC
std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>
get_unique_indices_cpu_impl(
    const Tensor& linear_indices,
    const int64_t /*max_indices*/,
    const bool compute_count,
    const bool compute_inverse_indices) {
  TORCH_CHECK(linear_indices.dim() == 1, "linear_indices must be 1D");
  TORCH_CHECK(linear_indices.numel() < std::numeric_limits<int32_t>::max());

  const int32_t N = linear_indices.numel();

  // Handle empty input
  if (N == 0) {
    return std::tuple{
        at::empty_like(linear_indices),
        at::zeros({1}, linear_indices.options().dtype(at::kInt)),
        compute_count ? std::optional<Tensor>(at::zeros(
                            {0}, linear_indices.options().dtype(at::kInt)))
                      : std::nullopt,
        compute_inverse_indices
            ? std::optional<Tensor>(
                  at::zeros({0}, linear_indices.options().dtype(at::kInt)))
            : std::nullopt};
  }

  // Use torch::unique to get unique indices
  Tensor unique_indices;
  Tensor inverse_indices;
  Tensor counts;

  if (compute_count || compute_inverse_indices) {
    std::tie(unique_indices, inverse_indices, counts) = at::unique_dim(
        linear_indices,
        /*dim=*/0,
        /*sorted=*/true,
        /*return_inverse=*/true,
        /*return_counts=*/true);
  } else {
    unique_indices = std::get<0>(at::unique_dim(
        linear_indices,
        /*dim=*/0,
        /*sorted=*/true,
        /*return_inverse=*/false,
        /*return_counts=*/false));
  }

  // Prepare output tensors
  const int32_t num_unique = unique_indices.numel();
  auto unique_indices_length =
      at::tensor({num_unique}, linear_indices.options().dtype(at::kInt));

  // Resize unique_indices to match same size as input
  auto unique_indices_output = at::zeros_like(linear_indices);
  unique_indices_output.slice(0, 0, num_unique).copy_(unique_indices);

  std::optional<Tensor> unique_indices_count = std::nullopt;
  std::optional<Tensor> linear_index_positions_sorted;

  if (compute_count) {
    // Resize counts to match same size as input
    unique_indices_count =
        at::zeros({N}, linear_indices.options().dtype(at::kInt));
    unique_indices_count->slice(0, 0, num_unique).copy_(counts.to(at::kInt));
  }

  if (compute_inverse_indices) {
    auto sort_indices = at::argsort(
        linear_indices, /*stable=*/true, /*dim=*/0, /*descending=*/false);

    // Convert to int32
    linear_index_positions_sorted = sort_indices.to(at::kInt);
  }

  return std::tuple{
      unique_indices_output,
      unique_indices_length,
      unique_indices_count,
      linear_index_positions_sorted};
}

DLL_PUBLIC
std::tuple<Tensor, Tensor, std::optional<Tensor>> get_unique_indices_cpu(
    const Tensor& linear_indices,
    const int64_t max_indices,
    const bool compute_count) {
  const auto ret = get_unique_indices_cpu_impl(
      linear_indices,
      max_indices,
      compute_count,
      /*compute_inverse_indices=*/false);

  return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
}

DLL_PUBLIC
std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>
get_unique_indices_with_inverse_cpu(
    const Tensor& linear_indices,
    const int64_t max_indices,
    const bool compute_count,
    const bool compute_inverse_indices) {
  return get_unique_indices_cpu_impl(
      linear_indices, max_indices, compute_count, compute_inverse_indices);
}

} // namespace fbgemm_gpu

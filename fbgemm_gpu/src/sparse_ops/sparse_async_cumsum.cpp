/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include "common.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

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

void asynchronous_exclusive_cumsum_cpu_out(Tensor& t_out, const Tensor& t_in) {
  TENSOR_ON_CPU(t_in);
  TENSOR_ON_CPU(t_out);

  const auto t_in_contig = t_in.expect_contiguous();
  at::native::resize_(t_out, t_in_contig->sizes(), std::nullopt);

  FBGEMM_DISPATCH_ALL_TYPES(
      t_in_contig->scalar_type(),
      "asynchronous_exclusive_cumsum_cpu_kernel",
      [&] {
        exclusive_scan_ptrs_cpu(
            t_in_contig->numel(),
            t_in_contig->const_data_ptr<scalar_t>(),
            t_out.mutable_data_ptr<scalar_t>());
      });
}

Tensor asynchronous_exclusive_cumsum_cpu(const Tensor& t_in) {
  TENSOR_ON_CPU(t_in);

  const auto t_in_contig = t_in.expect_contiguous();
  auto output = native_empty_like(*t_in_contig);
  asynchronous_exclusive_cumsum_cpu_out(output, *t_in_contig);
  return output;
}

Tensor asynchronous_inclusive_cumsum_cpu(const Tensor& t_in) {
  TENSOR_ON_CPU(t_in);

  const auto t_in_contig = t_in.expect_contiguous();
  auto output = native_empty_like(*t_in_contig);
  FBGEMM_DISPATCH_ALL_TYPES(
      t_in_contig->scalar_type(),
      "asynchronous_inclusive_cumsum_cpu_kernel",
      [&] {
        scalar_t cumsum = 0;
        const auto* input_ptr = t_in_contig->const_data_ptr<scalar_t>();
        const auto N = t_in_contig->numel();
        auto* output_ptr = output.mutable_data_ptr<scalar_t>();

        for (const auto i : c10::irange(N)) {
          cumsum += input_ptr[i];
          output_ptr[i] = cumsum;
        }
      });
  return output;
}

Tensor asynchronous_complete_cumsum_cpu_out(Tensor& t_out, const Tensor& t_in) {
  TENSOR_ON_CPU(t_in);
  TENSOR_ON_CPU(t_out);
  const auto num_dims = t_in.dim();
  TORCH_CHECK(num_dims == 1 || num_dims == 2);
  const auto t_in_contig = t_in.expect_contiguous();

  FBGEMM_DISPATCH_ALL_TYPES(
      t_in_contig->scalar_type(),
      "asynchronous_complete_cumsum_cpu_kernel",
      [&] {
        if (num_dims == 1) {
          const auto N = t_in_contig->numel();
          t_out.mutable_data_ptr<scalar_t>()[N] = exclusive_scan_ptrs_cpu(
              N,
              t_in_contig->const_data_ptr<scalar_t>(),
              t_out.mutable_data_ptr<scalar_t>());
        } else {
          const auto num_vecs = t_in_contig->size(0);
          const auto N = t_in_contig->size(1);
          at::parallel_for(0, num_vecs, 1, [&](int64_t start, int64_t end) {
            for (const auto i : c10::irange(start, end)) {
              scalar_t* out_ptr =
                  t_out.mutable_data_ptr<scalar_t>() + i * (N + 1);
              out_ptr[N] = exclusive_scan_ptrs_cpu(
                  N, t_in_contig->const_data_ptr<scalar_t>() + i * N, out_ptr);
            }
          });
        }
      });
  return t_out;
}

Tensor asynchronous_complete_cumsum_cpu(const Tensor& t_in) {
  const auto num_dims = t_in.dim();
  TORCH_CHECK(num_dims == 1 || num_dims == 2);
  auto output = num_dims == 1
      ? at::empty({t_in.numel() + 1}, t_in.options())
      : at::empty({t_in.size(0), t_in.size(1) + 1}, t_in.options());

  return asynchronous_complete_cumsum_cpu_out(output, t_in);
}

int64_t check_segmented_cumsum_inputs(
    const Tensor& t_in,
    at::IntArrayRef segment_sizes) {
  TORCH_CHECK(t_in.dim() == 1, "t_in must be 1-D, got ", t_in.dim(), "-D");
  const auto numel = t_in.numel();
  int64_t total_size = 0;
  for (const auto size : segment_sizes) {
    TORCH_CHECK(size >= 0, "segment_sizes must be non-negative, got ", size);
    // Checked before adding, so the sum cannot overflow.
    TORCH_CHECK(
        size <= numel - total_size,
        "segment_sizes sum to more than the ",
        numel,
        " elements of t_in");
    total_size += size;
  }
  TORCH_CHECK(
      total_size == numel,
      "segment_sizes sum to ",
      total_size,
      " but t_in has ",
      numel,
      " elements");
  return total_size;
}

// Complete cumsum of each segment of a 1-D tensor, concatenated: equivalent to
// cat([asynchronous_complete_cumsum(s) for s in t_in.split(segment_sizes)]).
Tensor asynchronous_segmented_complete_cumsum_cpu(
    const Tensor& t_in,
    at::IntArrayRef segment_sizes) {
  TENSOR_ON_CPU(t_in);
  const auto total_size = check_segmented_cumsum_inputs(t_in, segment_sizes);

  const auto t_in_contig = t_in.expect_contiguous();
  auto output = at::empty(
      {total_size + static_cast<int64_t>(segment_sizes.size())},
      t_in.options());

  FBGEMM_DISPATCH_ALL_TYPES(
      t_in_contig->scalar_type(),
      "asynchronous_segmented_complete_cumsum_cpu_kernel",
      [&] {
        const auto* in_ptr = t_in_contig->const_data_ptr<scalar_t>();
        auto* out_ptr = output.mutable_data_ptr<scalar_t>();
        for (const auto size : segment_sizes) {
          out_ptr[size] = exclusive_scan_ptrs_cpu(size, in_ptr, out_ptr);
          in_ptr += size;
          out_ptr += size + 1;
        }
      });
  return output;
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "asynchronous_exclusive_cumsum(Tensor t_in) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "asynchronous_inclusive_cumsum(Tensor t_in) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "asynchronous_complete_cumsum(Tensor t_in) -> Tensor",
      {PT2_COMPLIANT_TAG});
  m.def(
      "asynchronous_segmented_complete_cumsum(Tensor t_in, SymInt[] segment_sizes) -> Tensor",
      {PT2_COMPLIANT_TAG});
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  DISPATCH_TO_CPU(
      "asynchronous_exclusive_cumsum",
      fbgemm_gpu::asynchronous_exclusive_cumsum_cpu);
  DISPATCH_TO_CPU(
      "asynchronous_inclusive_cumsum",
      fbgemm_gpu::asynchronous_inclusive_cumsum_cpu);
  DISPATCH_TO_CPU(
      "asynchronous_complete_cumsum",
      fbgemm_gpu::asynchronous_complete_cumsum_cpu);
  DISPATCH_TO_CPU(
      "asynchronous_segmented_complete_cumsum",
      fbgemm_gpu::asynchronous_segmented_complete_cumsum_cpu);
}

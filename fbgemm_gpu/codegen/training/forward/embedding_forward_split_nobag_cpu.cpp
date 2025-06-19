/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/utils/cpu_utils.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/ops_utils.h"

#if FBGEMM_GPU_MEMCHECK
#define FBGEMM_MEM_CHECK_ONLY
#else
#define FBGEMM_MEM_CHECK_ONLY maybe_unused
#endif

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

template <
    typename weights_t,
    typename index_t,
    typename offset_t,
    typename output_t>
void split_embedding_nobag_codegen_forward_cpu_kernel(
    const Tensor& weights,
    const Tensor& weights_offsets,
    int64_t D,
    const Tensor& hash_size_cumsum,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& output) {
  TORCH_CHECK(weights.is_contiguous());
  Tensor indices_contig = indices.contiguous();
  Tensor offsets_contig = offsets.contiguous();

  const auto weights_offsets_data = weights_offsets.accessor<int64_t, 1>();
  const auto hash_size_cumsum_data = hash_size_cumsum.accessor<int64_t, 1>();
  const auto indices_data = indices.data_ptr<index_t>();
  const auto offsets_data = offsets.data_ptr<offset_t>();
  const auto weights_data = weights.data_ptr<weights_t>();
  auto output_data = output.data_ptr<output_t>();

  int64_t T = weights_offsets.size(0);
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK_GE(B, 0);

  at::parallel_for(0, T, 0, [&](int64_t t_begin, int64_t t_end) {
    for (const auto t : c10::irange(t_begin, t_end)) {
      int64_t hash_size = 0;
      int64_t t_temp = static_cast<int64_t>(t) + 1;
      do {
        hash_size = hash_size_cumsum_data[t_temp] - hash_size_cumsum_data[t];
        ++t_temp;
      } while (hash_size == 0);

      const auto table_begin = weights_offsets_data[t];

      bool success = true;
      at::parallel_for(0, B, 0, [&](int64_t b_begin, int64_t b_end) {
        for (const auto b : c10::irange(b_begin, b_end)) {
          const auto indices_start = offsets_data[t * B + b];
          const auto indices_end = offsets_data[t * B + b + 1];
          for (auto i = indices_start; i < indices_end; ++i) {
            const auto idx = indices_data[i];
            if (idx < 0 || idx >= hash_size) {
              success = false;
              continue;
            }
            const auto embedding_offset = table_begin + idx * D;
            for (const auto d : c10::irange(D)) {
              output_data[i * D + d] =
                  static_cast<output_t>(weights_data[embedding_offset + d]);
            }
          }
        }
      });

      if (!success) {
        fbgemm_gpu::report_embedding_error(
            static_cast<int>(t),
            static_cast<int>(B),
            0,
            static_cast<int>(B),
            offsets_data,
            indices_data,
            hash_size);
      }
    }
  });
}

Tensor split_embedding_nobag_codegen_forward_cpu(
    const Tensor& weights,
    const Tensor& weights_offsets,
    int64_t D,
    const Tensor& hash_size_cumsum,
    const Tensor& indices,
    const Tensor& offsets,
    int64_t output_dtype) {
  int64_t num_indices = indices.size(0);
  auto options = weights.options();
  if (output_dtype == static_cast<int64_t>(SparseType::FP32)) {
    options = weights.options().dtype(at::kFloat);
  } else if (output_dtype == static_cast<int64_t>(SparseType::FP16)) {
    options = weights.options().dtype(at::kHalf);
  } else if (output_dtype == static_cast<int64_t>(SparseType::BF16)) {
    options = weights.options().dtype(at::kBFloat16);
  }
  Tensor output = at::empty({num_indices, D}, options);

  // Dispatch based on indices, offsets, and output types
  FBGEMM_DISPATCH_FLOAT_AND_HALF(
      output.scalar_type(), "split_embedding_nobag_cpu_forward_1", [&]() {
        using output_t = scalar_t;

        FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
            weights.scalar_type(), "split_embedding_nobag_cpu_forward_2", [&] {
              using weights_t = scalar_t;

              AT_DISPATCH_INDEX_TYPES(
                  offsets.scalar_type(),
                  "split_embedding_nobag_cpu_forward_3",
                  [&] {
                    using offset_t = index_t;

                    AT_DISPATCH_INDEX_TYPES(
                        indices.scalar_type(),
                        "split_embedding_nobag_cpu_forward_4",
                        [&] {
                          split_embedding_nobag_codegen_forward_cpu_kernel<
                              weights_t,
                              index_t,
                              offset_t,
                              output_t>(
                              weights,
                              weights_offsets,
                              D,
                              hash_size_cumsum,
                              indices,
                              offsets,
                              output);
                        });
                  });
            });
      });

  return output;
}

namespace {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "split_embedding_nobag_codegen_forward_cpu(Tensor weights, "
      "                                          Tensor weights_offsets, "
      "                                          int D, "
      "                                          Tensor hash_size_cumsum, "
      "                                          Tensor indices, "
      "                                          Tensor offsets, "
      "                                          int output_dtype) -> Tensor");

  DISPATCH_TO_CPU(
      "split_embedding_nobag_codegen_forward_cpu",
      split_embedding_nobag_codegen_forward_cpu);
}
} // namespace

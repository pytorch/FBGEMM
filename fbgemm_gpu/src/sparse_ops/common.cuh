/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/sparse_ops.cuh"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/utils/cuda_block_count.h"
#include "fbgemm_gpu/utils/cuda_utilities.cuh"
#include "fbgemm_gpu/utils/ops_utils.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDADeviceAssertion.h>
#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// clang-format off
#include "fbgemm_gpu/utils/cub_namespace_prefix.cuh"
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/utils/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/utils/binary_search_range.cuh"
#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/log2.h"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

#ifdef USE_ROCM
#include <hipblas/hipblas.h>
#endif

#ifdef USE_ROCM
#define LDG(ptr) (*(ptr))
#else
#define LDG(ptr) (__ldg(ptr))
#endif

using Tensor = at::Tensor;

namespace fbgemm_gpu {

constexpr int MAX_ELEMENTS_PER_THREAD = 4;

// Whether the sparse permute kernels instantiate their device-side bounds
// asserts. This is a COMPILE-time switch, not a runtime one. Selecting it at
// runtime forces the compiler to emit both the debug and the non-debug
// specialization of every kernel in the dispatch matrix. For the permute_2D
// data kernels that is offsets(2) x indices(5) x weights(6) = 60 for each of
// the two weighted launches, plus offsets(2) x indices(5) = 10 for the
// unweighted one, so 130 kernels become 260 -- and that doubling pushed large
// PyTorch test binaries past the 2 GiB PC-relative relocation limit
// (T283951345).
//
// The lengths kernels have much smaller matrices (permute_2D: indices(2),
// permute_1D: permute(2) x indices(2)), so their doubling is not what blew the
// limit. They share this constant anyway so that every device assert in the
// sparse permute ops is toggled by one mechanism rather than each growing its
// own runtime-selected launch macro.
//
// Build with -DFBGEMM_DEBUG_PERMUTE_DEVICE_ASSERT to instantiate the asserts;
// leaving it out keeps the fused two-stream copy optimizable. The host-side
// checks stay runtime-gated on FBGEMM_DEBUG_PERMUTE=1 (see
// is_debug_permute_enabled below) and cost no instantiations -- they are the
// checks to reach for first, since they fail fast at the offending call and
// name both values.
#ifdef FBGEMM_DEBUG_PERMUTE_DEVICE_ASSERT
inline constexpr bool kPermuteDeviceAssert = true;
#else
inline constexpr bool kPermuteDeviceAssert = false;
#endif

// Opt-in host-side validation for the sparse permute ops. Every check below
// performs a D2H sync, so it is gated behind an env var which is off by
// default. Enable with FBGEMM_DEBUG_PERMUTE=1 for additional debugging with a
// performance penalty. When disabled, callers see a single cached-bool branch
// and no syncs.
inline bool is_debug_permute_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("FBGEMM_DEBUG_PERMUTE");
    if (v == nullptr || v[0] == '\0') {
      return false;
    }
    // Accept common truthy spellings (case-insensitive) so users who set
    // "true"/"yes"/"on"/"1" all get debugging. Anything else counts as
    // disabled; warn so a typo or unsupported value isn't silently ignored.
    std::string val(v);
    std::transform(val.begin(), val.end(), val.begin(), [](unsigned char c) {
      return std::tolower(c);
    });
    if (val == "1" || val == "true" || val == "yes" || val == "on") {
      return true;
    }
    TORCH_WARN(
        "[DEBUG permute] FBGEMM_DEBUG_PERMUTE=\"",
        v,
        "\" is not a recognized truthy value; debug validation stays disabled. "
        "Use one of: 1, true, yes, on.");
    return false;
  }();
  return enabled;
}

// Stage 1: validate the raw inputs. permute indexes lengths / input_offsets, so
// an out-of-range value would cause an out-of-bounds read; lengths must be
// non-negative and indices must cover the full input sum.
// `permute_index_bound` is the exclusive upper bound for permute values
// (lengths.numel() for 1D, lengths.size(0) for 2D). `weights` is the optional
// per-index weights argument; its presence and column width are logged (not
// validated).
inline void debug_check_permute_inputs(
    const char* tag,
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    int64_t permute_index_bound,
    const std::optional<at::Tensor>& weights) {
  const auto permute_min = permute.min().item<int64_t>();
  const auto permute_max = permute.max().item<int64_t>();
  const auto lengths_min = lengths.min().item<int64_t>();
  const auto lengths_max = lengths.max().item<int64_t>();
  const auto lengths_sum = lengths.sum().item<int64_t>();
  const auto indices_numel = indices.numel();
  // indices can be empty (e.g. all lengths zero); min/max on an empty tensor
  // throws, so only read them when there is data. When empty, min/max are
  // logged as a placeholder 0 -- the numel=0 printed first disambiguates it.
  const auto indices_min =
      indices_numel > 0 ? indices.min().item<int64_t>() : 0;
  const auto indices_max =
      indices_numel > 0 ? indices.max().item<int64_t>() : 0;
  // weights is optional and parallel to indices; log presence and the per-index
  // column width (weights_columns), but do not validate it.
  const bool has_weight = weights.has_value();
  const auto weights_columns =
      (has_weight && weights->dim() > 1) ? weights->size(1) : 1;
  TORCH_WARN(
      "[DEBUG ",
      tag,
      "] STAGE1 checking inputs | permute: numel=",
      permute.numel(),
      " min=",
      permute_min,
      " max=",
      permute_max,
      " | lengths: numel=",
      lengths.numel(),
      " min=",
      lengths_min,
      " max=",
      lengths_max,
      " sum=",
      lengths_sum,
      " | indices: numel=",
      indices_numel,
      " min=",
      indices_min,
      " max=",
      indices_max,
      " | weights: present=",
      has_weight,
      " columns=",
      weights_columns);
  TORCH_CHECK(
      permute_min >= 0 && permute_max < permute_index_bound,
      "[DEBUG ",
      tag,
      "] FAILED in Stage 1: permute out of range [0, ",
      permute_index_bound,
      "): min=",
      permute_min,
      " max=",
      permute_max,
      " -> lengths kernel will read OOB (corrupt caller input)");
  TORCH_CHECK(
      lengths_min >= 0,
      "[DEBUG ",
      tag,
      "] FAILED in Stage 1: negative input length min=",
      lengths_min,
      " (corrupt caller input)");
  // indices holds all input segments, so it must cover the full input sum; an
  // undersized indices tensor means the data kernel reads OOB from it (stale
  // metadata: lengths claims more elements than indices has).
  TORCH_CHECK(
      indices_numel >= lengths_sum,
      "[DEBUG ",
      tag,
      "] FAILED in Stage 1: indices undersized: indices.numel()=",
      indices_numel,
      " < lengths.sum()=",
      lengths_sum,
      " -> data kernel reads OOB from indices (stale/mismatched metadata)");
  // Per-element upper bound: sum(lengths) == indices.numel() and all lengths
  // are non-negative, so no single length can exceed the total. A lone
  // garbage-huge length that the aggregate sum check might not isolate is
  // pinpointed here.
  TORCH_CHECK(
      lengths_max <= indices_numel,
      "[DEBUG ",
      tag,
      "] FAILED in Stage 1: input length exceeds total indices: lengths.max()=",
      lengths_max,
      " > indices.numel()=",
      indices_numel,
      " (garbage/stale length value)");
}

// Stage 2: validate permuted_lengths after the lengths kernel. Each entry is a
// copy of an input length, so values must be non-negative and no larger than
// the total number of indices. Returns the sum so Stage 3 can reuse it without
// a second D2H sync.
inline int64_t debug_check_permuted_lengths(
    const char* tag,
    const at::Tensor& permuted_lengths,
    int64_t indices_numel) {
  const auto pl_min = permuted_lengths.min().item<int64_t>();
  const auto pl_max = permuted_lengths.max().item<int64_t>();
  const auto pl_sum = permuted_lengths.sum().item<int64_t>();
  TORCH_WARN(
      "[DEBUG ",
      tag,
      "] STAGE2 checking permuted_lengths | numel=",
      permuted_lengths.numel(),
      " min=",
      pl_min,
      " max=",
      pl_max,
      " sum=",
      pl_sum);
  TORCH_CHECK(
      pl_min >= 0 && pl_max <= indices_numel,
      "[DEBUG ",
      tag,
      "] FAILED in Stage 2: permuted_lengths corrupted after lengths kernel: min=",
      pl_min,
      " max=",
      pl_max,
      " sum=",
      pl_sum,
      " indices.numel()=",
      indices_numel);
  return pl_sum;
}

// Stage 3: validate the offsets. A complete cumsum must start at 0, be
// monotonic non-decreasing, and its last element must equal the sum of
// permuted_lengths. Checking the whole array, not just the last element,
// isolates a cumsum that is wrong in the middle but right at the end.
// `permuted_lengths_sum` is the independent Stage 2 result, reused to avoid an
// extra D2H sync.
inline void debug_check_output_offsets(
    const char* tag,
    const at::Tensor& output_offsets,
    int64_t permuted_lengths_sum) {
  // asynchronous_complete_cumsum_gpu returns at least one element, but guard
  // empty defensively so a debug indexing throw never masks the real
  // validation failure (symmetric with the min/max guards elsewhere here).
  const auto oo_numel = output_offsets.numel();
  const auto oo_last = oo_numel > 0 ? output_offsets[-1].item<int64_t>() : 0;
  const auto oo_first = oo_numel > 0 ? output_offsets[0].item<int64_t>() : 0;
  // When output_offsets has a single element, diff() is empty and min() would
  // throw, so min_delta is set to a placeholder 0 (numel is printed first to
  // disambiguate) and the monotonic check passes vacuously.
  const auto min_delta = output_offsets.numel() > 1
      ? output_offsets.diff().min().item<int64_t>()
      : 0;
  TORCH_WARN(
      "[DEBUG ",
      tag,
      "] STAGE3 checking offsets | output_offsets: numel=",
      output_offsets.numel(),
      " dtype=",
      output_offsets.scalar_type(),
      " [0]=",
      oo_first,
      " [-1]=",
      oo_last,
      " min_delta=",
      min_delta,
      " permuted_lengths.sum()=",
      permuted_lengths_sum);
  TORCH_CHECK(
      oo_last == permuted_lengths_sum,
      "[DEBUG ",
      tag,
      "] FAILED in Stage 3: cumsum mismatch output_offsets[-1]=",
      oo_last,
      " != permuted_lengths.sum()=",
      permuted_lengths_sum,
      " (complete_cumsum bug or permuted_lengths mutated between STAGE2 and cumsum)");
  TORCH_CHECK(
      oo_first == 0,
      "[DEBUG ",
      tag,
      "] FAILED in Stage 3: output_offsets[0]=",
      oo_first,
      " != 0 (corrupt complete_cumsum)");
  TORCH_CHECK(
      min_delta >= 0,
      "[DEBUG ",
      tag,
      "] FAILED in Stage 3: output_offsets not monotonic, min delta=",
      min_delta,
      " (corrupt complete_cumsum)");
}

template <
    typename scalar_t,
    int ndim,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
at::PackedTensorAccessor32<scalar_t, ndim, PtrTraits>
dummy_packed_accessor32() {
  std::array<int64_t, ndim> zeros{};
  return {nullptr, zeros.data(), zeros.data()};
}

template <
    typename scalar_t,
    int ndim,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
pta::PackedTensorAccessor64<scalar_t, ndim, PtrTraits>
dummy_packed_accessor64() {
  std::array<int64_t, ndim> zeros{};
  return {nullptr, zeros.data(), zeros.data()};
}

} // namespace fbgemm_gpu

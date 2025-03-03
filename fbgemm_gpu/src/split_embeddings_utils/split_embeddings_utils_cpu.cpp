/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>
#include "fbgemm_gpu/split_embeddings_utils.h"
#include "fbgemm_gpu/utils/ops_utils.h"

using Tensor = at::Tensor;

/// Find number of bits to accommodate this value
///
/// num_bits           number of bits to needed to accommodate
///                    e.g., the function returns 3 if `n` is between 4 (100)
///                    and 7 (111) as 3 bits are required to represent the
///                    number.
///
/// @param n           positive decimal number
///
DLL_PUBLIC int32_t get_num_bits(int32_t n) {
  TORCH_CHECK(n > 0, "Expect n to be positive but got ", n);
  return static_cast<int32_t>(std::floor(std::log2(n) + 1));
}

/// Calculates number of bits to accommodate batch size (B) and table (T) from
/// T. We first calculate how many bits needed for T and the rest is for B,
/// since T does not change once TBE is initialized but B can change.
///
/// info_B_num_bits     Number of bits needed for accommodate batch size
/// info_B_mask         Bit mask for information of B
/// @param T            Number of tables (features)
/// @param B            Batch size
///
DLL_PUBLIC std::tuple<int32_t, uint32_t> get_info_B_num_bits_from_T(
    int32_t T,
    int32_t B = 1) {
  TORCH_CHECK(B > 0, "B must be positive. Got B = ", B);
  TORCH_CHECK(T > 0, "T must be positive. Got T = ", T);
  const int32_t info_T_num_bits = get_num_bits(T);
  const int32_t info_B_num_bits = DEFAULT_INFO_NUM_BITS - info_T_num_bits;
  const uint32_t info_B_mask = (1u << info_B_num_bits) - 1;
  TORCH_CHECK(
      B <= info_B_mask,
      "Not enough infos bits to accommodate T and B. T = ",
      T,
      " takes ",
      info_T_num_bits,
      " and info_B_num_bits is ",
      info_B_num_bits,
      ". Expect max_B = ",
      info_B_mask,
      "but got B ",
      B);

  return {info_B_num_bits, info_B_mask};
}

DLL_PUBLIC std::tuple<int32_t, uint32_t> adjust_info_B_num_bits(
    int32_t B,
    int32_t T) {
  int32_t info_B_num_bits = DEFAULT_INFO_B_NUM_BITS;
  uint32_t info_B_mask = DEFAULT_INFO_B_MASK;
  uint32_t max_T = MAX_T;
  uint32_t max_B = MAX_B;
  bool invalid_T = T > max_T;
  bool invalid_B = B > max_B;

  TORCH_CHECK(
      !(invalid_T && invalid_B),
      "Not enough infos bits to accommodate T and B. Default num bits = ",
      DEFAULT_INFO_NUM_BITS);

  if (invalid_T) {
    // Reduce info_B_num_bits
    while (invalid_T && !invalid_B && info_B_num_bits > 0) {
      info_B_num_bits--;
      max_T = ((max_T + 1) << 1) - 1;
      max_B = ((max_B + 1) >> 1) - 1;
      invalid_T = T > max_T;
      invalid_B = B > max_B;
    }
  } else if (invalid_B) {
    // Increase info_B_num_bits
    while (!invalid_T && invalid_B && info_B_num_bits < DEFAULT_INFO_NUM_BITS) {
      info_B_num_bits++;
      max_T = ((max_T + 1) >> 1) - 1;
      max_B = ((max_B + 1) << 1) - 1;
      invalid_T = T > max_T;
      invalid_B = B > max_B;
    }
  }

  TORCH_CHECK(
      !invalid_T && !invalid_B,
      "Not enough infos bits to accommodate T and B. Default num bits = ",
      DEFAULT_INFO_NUM_BITS);

  // Recompute info_B_mask using new info_B_num_bits
  info_B_mask = (1u << info_B_num_bits) - 1;

  return {info_B_num_bits, info_B_mask};
}

namespace {

std::tuple<Tensor /*row_output_offsets*/, Tensor /*b_t_map*/>
generate_vbe_metadata_cpu(
    const Tensor& B_offsets,
    const Tensor& B_offsets_rank_per_feature,
    const Tensor& output_offsets_feature_rank,
    const Tensor& D_offsets,
    const int64_t D,
    const bool nobag,
    const c10::SymInt max_B_feature_rank,
    const int64_t info_B_num_bits,
    const c10::SymInt total_B) {
  Tensor row_output_offsets = output_offsets_feature_rank;
  Tensor b_t_map = B_offsets_rank_per_feature;
  return {row_output_offsets, b_t_map};
}

std::tuple<int64_t, int64_t>
get_infos_metadata_cpu(Tensor unused, int64_t B, int64_t T) {
  return get_info_B_num_bits_from_T(T, B);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "transpose_embedding_input("
      "    Tensor hash_size_cumsum, "
      "    int total_hash_size_bits, "
      "    Tensor indices, "
      "    Tensor offsets, "
      "    bool nobag=False, "
      "    Tensor? vbe_b_t_map=None, "
      "    int info_B_num_bits=26, "
      "    int info_B_mask=0x2FFFFFF, "
      "    int total_unique_indices=-1, "
      "    bool is_index_select=False, "
      "    Tensor? total_L_offsets=None, "
      "    int fixed_L_per_warp=0, "
      "    int num_warps_per_feature=0"
      ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("get_infos_metadata(Tensor unused, int B, int T) -> (int, int)");
  m.def(
      "generate_vbe_metadata("
      "    Tensor B_offsets, "
      "    Tensor B_offsets_rank_per_feature, "
      "    Tensor output_offsets_feature_rank, "
      "    Tensor D_offsets, "
      "    int D, "
      "    bool nobag, "
      "    SymInt max_B_feature_rank, "
      "    int info_B_num_bits, "
      "    SymInt total_B"
      ") -> (Tensor, Tensor)");
  DISPATCH_TO_CPU("generate_vbe_metadata", generate_vbe_metadata_cpu);
  DISPATCH_TO_CPU("get_infos_metadata", get_infos_metadata_cpu);
}

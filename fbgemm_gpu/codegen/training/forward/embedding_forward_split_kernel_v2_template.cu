/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

{#-
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
// Note: clang-format off doesn't work with this templaterized code,
// so we need to keep lint-ignore-every.
// See https://fburl.com/dw9ljh4h
#}

{%- set wdesc =  "weighted" if weighted else "unweighted" %}
#include "fbgemm_gpu/embedding_forward_template_helpers.cuh"

using namespace fbgemm_gpu;

constexpr uint32_t VEC_WIDTH = 4;

enum SAVED_PARAMS {
  P_indices = 0,
  P_weights,
  P_outputs,
  {%- if weighted %}
  P_index_weights,
  {%- endif %}
  P_offsets,
  P_num_offsets,
  P_load_D,
  P_total_load_D
};
{%- if weighted %}
constexpr uint32_t SAVED_PARAMS_CNT = 8;
{%- else %}
constexpr uint32_t SAVED_PARAMS_CNT = 7;
{%- endif %}
enum LXU_CACHE_PARAMS {
  P_lxu_cache_weights = SAVED_PARAMS_CNT,
  P_lxu_cache_locations = SAVED_PARAMS_CNT + 1};
constexpr uint32_t LXU_PARAMS_CNT = 2;

#define SMEM_PTR_BASE(TYPE) \
  (reinterpret_cast<TYPE>(smem + WEIGHT_PTR_OFFSET) + threadIdx.y * kWarpSize)

#define SMEM_GENERIC_PTR SMEM_PTR_BASE(uintptr_t*)

#define SMEM_EMB_WEIGHT_PTR SMEM_PTR_BASE(const emb_vec_t**)

#define SMEM_EMB_WEIGHT_DATA(SMEM_IDX, WEIGHT_IDX) \
  (SMEM_PTR_BASE(const emb_vec_t**)[SMEM_IDX])[WEIGHT_IDX]

#define SMEM_CACHE_WEIGHT_PTR SMEM_PTR_BASE(const cache_vec_t**)

#define SMEM_CACHE_WEIGHT_DATA(SMEM_IDX, WEIGHT_IDX) \
  (SMEM_PTR_BASE(const cache_vec_t**)[SMEM_IDX])[WEIGHT_IDX]

// This avoid type conversion of denom in div_round_up
#define DIV_ROUND_UP(numer, denom) ((numer + denom - 1) / denom)

#define ACC_ADD_OR_FMA(WEIGHT, INDEX_WEIGHT) \
  {%- if weighted %}
  accumulator.fma(WEIGHT, INDEX_WEIGHT);
  {%- else %}
  accumulator.add(WEIGHT);
  {%- endif %}

template <typename T>
struct Vec4Type {};

template <>
struct Vec4Type<float> {
  using type = float4;
};

template <>
struct Vec4Type<at::Half> {
  using type = float2;
};

template <>
struct Vec4Type<at::BFloat16> {
  using type = float2;
};

template <>
struct Vec4Type<uint8_t> {
  using type = uint8_t;
};

template <typename T>
using vec4_type = typename Vec4Type<T>::type;

template<uint32_t LOWER_BIT_CNT, uint32_t WARP_MASK>
__inline__ __device__ void get_next_bag_boundary_and_L(
    const uint32_t bag_boundary,
    int32_t* const next_boundary,
    uint32_t* const L) {
  const int32_t tid = bag_boundary & WARP_MASK;
  if (tid < kWarpSize) {
    const auto prev_boundary = *next_boundary;
    *next_boundary = shfl_sync(bag_boundary, tid) >> LOWER_BIT_CNT;
    *L = (*next_boundary) - prev_boundary;
  }
  else {
    *next_boundary = -1;
  }
}

template <
  typename index_t,
  typename emb_t,
  typename emb_vec_t,
  typename output_vec_t,
  uint32_t STEP
  >
__inline__ __device__ void process_all_indices_no_pooling(
    long* const smem,
    const bool process_d,
    const uint32_t params_offset) {
  constexpr uint32_t TOTAL_L = kWarpSize; // caller needs to ensure this

  const auto* __restrict__ indices =
    *reinterpret_cast<index_t**>(&smem[params_offset + SAVED_PARAMS::P_indices]);
  const auto* __restrict__ weights =
    *reinterpret_cast<emb_vec_t**>(&smem[params_offset + SAVED_PARAMS::P_weights]);
  {%- if weighted %}
  const auto* index_weights = reinterpret_cast<float*>(smem[params_offset + SAVED_PARAMS::P_index_weights]);
  {%- endif %}
  const auto load_D = static_cast<uint32_t>(smem[params_offset + SAVED_PARAMS::P_load_D]);
  const auto total_load_D = static_cast<uint32_t>(smem[params_offset + SAVED_PARAMS::P_total_load_D]);

  // Each thread loads a separate weight ptr
  const auto weight_ptrs = reinterpret_cast<uintptr_t>(&weights[indices[threadIdx.x] * load_D]);

  // Assuming kWarpSize is a multiple of STEP
  for (uint32_t l_start = 0; l_start < TOTAL_L; l_start += STEP) {
    Vec4StepT<STEP, emb_t> vecs;
    #pragma unroll
    for (uint32_t j = 0; j < STEP; ++j) {
      // Get weight pointer
      const auto* ptr = reinterpret_cast<const emb_vec_t*>(
          shfl_sync(weight_ptrs, l_start + j)) + threadIdx.x;
      // Load row
      if (process_d) {
        vecs.load(ptr, j);
      }
    }

    auto* const __restrict__ output =
      *reinterpret_cast<output_vec_t**>(&smem[params_offset + SAVED_PARAMS::P_outputs]) + l_start * total_load_D + threadIdx.x;

    if (process_d) {
      // Write to output (not pooling)
      #pragma unroll
      for (uint32_t j = 0; j < STEP; ++j) {
        {%- if weighted %}
        const auto index_weight = index_weights[l_start + j];
        vecs.index_weighted_store(j, &output[j * total_load_D], index_weight);
        {%- else %}
        vecs.index_store(j, &output[j * total_load_D]);
        {%- endif %}
      }
    }
  }
}

template <
  typename emb_t,
  typename output_vec_t,
  uint32_t STEP,
  uint32_t BOUNDARY_IDX_BIT_CNT,
  uint32_t WARP_MASK
  >
__inline__ __device__ void write_loop_small_Ls(
    long* const smem,
    uint32_t* const write_idx,
    uint32_t* const bag_boundary,
    int32_t* const next_boundary,
    uint32_t* const L,
    Vec4StepT<STEP, emb_t>* const accumulator,
    const uint32_t params_offset,
    const uint32_t l,
    const bool process_d,
    const bool mean_pooling) {
  // The loop writes to accumulated results or zeros to the output buffer.
  // When threads first enter this loop, they write the accumulated results.
  // Then, if next_boundary is still equal to l + 1, they write zeros. If all
  // the outputs (up to 32 outputs) are written, next_boundary will be -1.
  while (*next_boundary == l + 1) {
    output_vec_t* __restrict__ const output =
      *reinterpret_cast<output_vec_t**>(&smem[params_offset + SAVED_PARAMS::P_outputs]) + *write_idx + threadIdx.x;

    const auto total_load_D = static_cast<uint32_t>(smem[params_offset + SAVED_PARAMS::P_total_load_D]);
    *write_idx += total_load_D;

    // Write the output
    if (process_d) {
      if (mean_pooling && *L != 0) {
        accumulator->div(*L);
      }
      accumulator->store(output);
    }

    // Reset accumulator
    accumulator->reset();

    // Increment boundary index
    *bag_boundary += 1; // boundary value in the upper bits is unaffected
    get_next_bag_boundary_and_L<BOUNDARY_IDX_BIT_CNT, WARP_MASK>(*bag_boundary, next_boundary, L);
  }
}

template <
  typename index_t,
  typename emb_t,
  typename emb_vec_t,
  typename cache_t,
  typename cache_vec_t,
  typename output_vec_t,
  bool USE_CACHE_WEIGHTS,
  bool USE_MIXED_TYPE_CACHE,
  uint32_t WEIGHT_PTR_OFFSET,
  uint32_t STEP,
  uint32_t STEP_MASK,
  uint32_t LOAD_GROUP_SIZE // unused
  >
__noinline__ __device__ void process_all_indices_small_Ls(
    long* const smem,
    const uint32_t total_L,
    const bool process_d,
    const bool mean_pooling,
    const uint32_t params_offset,
    const uint32_t max_D_cache) {
  Vec4StepT<STEP, emb_t> accumulator;

  if (total_L <= 0) {
    if (process_d) {
      output_vec_t* __restrict__ const output =
        *reinterpret_cast<output_vec_t**>(&smem[params_offset + SAVED_PARAMS::P_outputs]) + threadIdx.x;
      const uint32_t num_offsets = smem[params_offset + SAVED_PARAMS::P_num_offsets];
      const uint32_t total_load_D = smem[params_offset + SAVED_PARAMS::P_total_load_D];
      // Write zeros to the sample that L = 0
      Vec4StepT<1, emb_t> accumulator;
      for (uint32_t i = 0; i < num_offsets; ++i) {
        accumulator.store(output + i * total_load_D);
      }
    }
    return;
  }

  // Determine the indices at which we need to write to output, i.e. the bag boundaries
  const auto* __restrict__ offsets =
    *reinterpret_cast<index_t**>(&smem[params_offset + SAVED_PARAMS::P_offsets]);
  const uint32_t num_offsets = smem[params_offset + SAVED_PARAMS::P_num_offsets];

  // The first offset assigned to this warp
  index_t offset_start = offsets[0];
  uint32_t bag_boundary = 0;
  if (threadIdx.x < num_offsets) {
    bag_boundary = offsets[threadIdx.x + 1] - offset_start;
  }

  // Check the special case where each thread needs to process exactly one input
  // If UVM cache is used, fall back to the generic function
  if (!USE_CACHE_WEIGHTS &&
      ballot_sync(bag_boundary == threadIdx.x + 1) == kFullWarpMask) {
    process_all_indices_no_pooling<index_t, emb_t, emb_vec_t, output_vec_t, STEP>(
        smem, process_d, params_offset);
    return;
  }

  // Each thread needs to keep two states:
  // 1) The index before/after which we need to write outputs. i.e. bag boundaries.
  //    Note that there are at most kWarpSize such indices because this function can
  //    handle at most kWarpsize offsets. Each thread will keep one of those values.
  //
  // 2) The next bag boundary index to check. In the beginning, the first boundary
  //    will be checked. Over the iterations, we will be moving to the next boundaries
  //    as we write outputs. This index will correspond to the threadIdx we need to
  //    read the #1 data from.
  //
  // Optimization:
  // The max value for #1 is the max number of indices that can be handled by this warp.
  // The max value for #2 is kWarpSize.
  // We will use a single variable to keep track of both: The lower 8 bits will store #2,
  // whereas the upper 24 bits will store #1. Here, we assume that kWarpSize < 256, but this
  // can be easily changed if needed.
  constexpr uint32_t BOUNDARY_IDX_BIT_CNT = 8;
  constexpr uint32_t WARP_MASK = (1u << BOUNDARY_IDX_BIT_CNT) - 1;
  bag_boundary = bag_boundary << BOUNDARY_IDX_BIT_CNT;

  uint32_t write_idx = 0; // Index for output
  int32_t next_boundary = 0; // Can be -1 if all outputs are written
  uint32_t L = 0; // Required for mean pooling

  // Compute the first boundary and L
  get_next_bag_boundary_and_L<BOUNDARY_IDX_BIT_CNT, WARP_MASK>(bag_boundary, &next_boundary, &L);

  while (next_boundary == 0) {
    auto * __restrict__ const output = *reinterpret_cast<output_vec_t**>(&smem[params_offset + SAVED_PARAMS::P_outputs]);
    const auto total_load_D = static_cast<uint32_t>(smem[params_offset + SAVED_PARAMS::P_total_load_D]);
    if (process_d) {
      Vec4StepT<1, emb_t> accumulator;
      accumulator.store(output + write_idx + threadIdx.x);
    }
    write_idx += total_load_D;

    // Increment the bag boundary index value
    bag_boundary += 1; // Note that the boundary value in the upper bits is unaffected
    get_next_bag_boundary_and_L<BOUNDARY_IDX_BIT_CNT, WARP_MASK>(bag_boundary, &next_boundary, &L);
  }

  uint32_t l_start = 0;
  // The ith LSB bit is set to 1 if the ith row needs to be looked up from cache
  uint32_t cache_look_up_bits = 0;
  for (; l_start < total_L; l_start += STEP) {
    if ((l_start % kWarpSize) == 0) {
      const uint32_t l = l_start + threadIdx.x;

      // We expect the following to be loaded from shared memory instead of consuming registers
      const auto* __restrict__ indices =
        *reinterpret_cast<index_t**>(&smem[params_offset + SAVED_PARAMS::P_indices]);
      const auto* __restrict__ weights =
        *reinterpret_cast<emb_vec_t**>(&smem[params_offset + SAVED_PARAMS::P_weights]);
      const auto load_D = static_cast<uint32_t>(smem[params_offset + SAVED_PARAMS::P_load_D]);

      syncwarp(); // Ensure that all warps read the previous value before overwriting it
      if (USE_CACHE_WEIGHTS) {
        auto cache_idx = kCacheLocationMissing;
        if (l < total_L) {
          cache_idx = reinterpret_cast<int32_t*>(smem[params_offset + LXU_CACHE_PARAMS::P_lxu_cache_locations])[l];
          const cache_t* lxu_cache_weights =
            reinterpret_cast<const cache_t*>(smem[params_offset + LXU_CACHE_PARAMS::P_lxu_cache_weights]);
          SMEM_GENERIC_PTR[threadIdx.x] = cache_idx != kCacheLocationMissing ?
            reinterpret_cast<uintptr_t>(&lxu_cache_weights[cache_idx * max_D_cache]) :
            reinterpret_cast<uintptr_t>(&weights[indices[l] * load_D]);
        }
        if (!std::is_same<emb_t, cache_t>::value) {
          cache_look_up_bits = ballot_sync(cache_idx != kCacheLocationMissing);
        }
      }
      else {
        SMEM_EMB_WEIGHT_PTR[threadIdx.x] = (l < total_L) ?
          reinterpret_cast<const emb_vec_t*>(&weights[indices[l] * load_D]) : nullptr;
      }
      syncwarp(); // Ensure that all weight pointers are written
    }

    // Make sure that all threads execute the same code
    if (l_start + STEP > total_L) {
      break;
    }

    const auto cache_look_up_bits_step = cache_look_up_bits & STEP_MASK;
    if (USE_MIXED_TYPE_CACHE && cache_look_up_bits_step != 0) {
      if (cache_look_up_bits_step == STEP_MASK) {
        #pragma unroll
        for (uint32_t j = 0; j < STEP; ++j) {
          const auto smem_offset = (l_start % kWarpSize) + j;
          if (process_d) {
            {%- if weighted %}
            const auto index_weight =
              reinterpret_cast<float*>(smem[params_offset + SAVED_PARAMS::P_index_weights])[l_start + j];
            {%- endif %}
            // Load STEP rows from lx_cache_weights
            const auto* weight = &SMEM_CACHE_WEIGHT_DATA(smem_offset, threadIdx.x);
            ACC_ADD_OR_FMA(weight, index_weight)
          }

          // Write to the output buffer at the boundary
          write_loop_small_Ls<emb_t, output_vec_t, STEP, BOUNDARY_IDX_BIT_CNT, WARP_MASK>(
              smem,
              &write_idx,
              &bag_boundary,
              &next_boundary,
              &L,
              &accumulator,
              params_offset,
              l_start + j,
              process_d,
              mean_pooling);
        }
        cache_look_up_bits >>= STEP;
      }
      else {
        #pragma unroll
        for (uint32_t j = 0; j < STEP; ++j) {
          const auto smem_offset = (l_start % kWarpSize) + j;
          if (process_d) {
            {%- if weighted %}
            const auto index_weight =
              reinterpret_cast<float*>(smem[params_offset + SAVED_PARAMS::P_index_weights])[l_start + j];
            {%- endif %}
            // Load and accumulate STEP rows for UVM caching that emb_t and cache_t
            // are not the same and rows within STEPS are read from different
            // locations. It is unlikely that the compiler will be able to unroll
            // the loop below because of the runtime conditionals
            if (cache_look_up_bits & 1u) {
              const auto* weight = &SMEM_CACHE_WEIGHT_DATA(smem_offset, threadIdx.x);
              ACC_ADD_OR_FMA(weight, index_weight)
            }
            else {
              const auto* weight = &SMEM_EMB_WEIGHT_DATA(smem_offset, threadIdx.x);
              ACC_ADD_OR_FMA(weight, index_weight)
            }
          }

          // Write to the output buffer at the boundary
          write_loop_small_Ls<emb_t, output_vec_t, STEP, BOUNDARY_IDX_BIT_CNT, WARP_MASK>(
              smem,
              &write_idx,
              &bag_boundary,
              &next_boundary,
              &L,
              &accumulator,
              params_offset,
              l_start + j,
              process_d,
              mean_pooling);

          cache_look_up_bits >>= 1;
        }
      }
    }
    else {
      if (process_d) {
        // Load STEP rows
        #pragma unroll
        for (uint32_t j = 0; j < STEP; ++j) {
          const auto smem_offset = (l_start % kWarpSize) + j;
          accumulator.load(&SMEM_EMB_WEIGHT_DATA(smem_offset, threadIdx.x), j);
        }
      }

      #pragma unroll
      for (uint32_t j = 0; j < STEP; ++j) {
        // Accumulate rows
        if (process_d) {
          {%- if weighted %}
          const auto index_weight =
            reinterpret_cast<float*>(smem[params_offset + SAVED_PARAMS::P_index_weights])[l_start + j];
          accumulator.index_fma(j, index_weight);
          {%- else %}
          accumulator.index_add(j);
          {%- endif %}
        }

        // Write to the output buffer at the boundary
        write_loop_small_Ls<emb_t, output_vec_t, STEP, BOUNDARY_IDX_BIT_CNT, WARP_MASK>(
            smem,
            &write_idx,
            &bag_boundary,
            &next_boundary,
            &L,
            &accumulator,
            params_offset,
            l_start + j,
            process_d,
            mean_pooling);
      }

      if (USE_MIXED_TYPE_CACHE) {
        cache_look_up_bits >>= STEP;
      }
    }
  }

  // Process the remaining indices (less than STEP)
  for (uint32_t j = 0; j < total_L - l_start; j++) {
    // Load and accumulate rows
    if (process_d) {
      {%- if weighted %}
      const auto index_weight =
        reinterpret_cast<float*>(smem[params_offset + SAVED_PARAMS::P_index_weights])[l_start + j];
      {%- endif %}
      if (USE_MIXED_TYPE_CACHE && cache_look_up_bits & 1u) {
        const auto* weight = &SMEM_CACHE_WEIGHT_DATA((l_start % kWarpSize) + j, threadIdx.x);
        ACC_ADD_OR_FMA(weight, index_weight)
      }
      else {
        const auto* weight = &SMEM_EMB_WEIGHT_DATA((l_start % kWarpSize) + j, threadIdx.x);
        ACC_ADD_OR_FMA(weight, index_weight)
      }
      if (USE_MIXED_TYPE_CACHE) {
        cache_look_up_bits >>= 1;
      }
    }

    // Write to the output buffer at the boundary
    write_loop_small_Ls<emb_t, output_vec_t, STEP, BOUNDARY_IDX_BIT_CNT, WARP_MASK>(
        smem,
        &write_idx,
        &bag_boundary,
        &next_boundary,
        &L,
        &accumulator,
        params_offset,
        l_start + j,
        process_d,
        mean_pooling);
  }
}

template <
  typename index_t,
  typename emb_t,
  typename emb_vec_t,
  typename cache_t,
  typename cache_vec_t,
  typename output_vec_t,
  bool USE_CACHE_WEIGHTS,
  bool USE_MIXED_TYPE_CACHE,
  uint32_t WEIGHT_PTR_OFFSET,
  uint32_t STEP,
  uint32_t STEP_MASK,
  uint32_t LOAD_GROUP_SIZE
  >
__noinline__ __device__ void process_all_indices_large_Ls(
    long* const smem,
    const uint32_t L,
    const bool process_d,
    const bool mean_pooling,
    const uint32_t params_offset,
    const uint32_t max_D_cache) {

#define SMEM_OFFSET \
    (IS_FULL_WARP ? j : ((threadIdx.x / LOAD_GROUP_SIZE) + (j * NUM_LOAD_GROUPS)))

#define WEIGHT_OFFSET \
    (IS_FULL_WARP ? threadIdx.x : (threadIdx.x % LOAD_GROUP_SIZE))

  constexpr uint32_t NUM_LOAD_GROUPS = kWarpSize / LOAD_GROUP_SIZE;
  constexpr bool IS_FULL_WARP = LOAD_GROUP_SIZE == kWarpSize;

  Vec4StepT<STEP, emb_t> accumulator;

  uint32_t l_start = 0;
  // The ith LSB bit is set to 1 if the ith row needs to be looked up from cache
  uint32_t cache_look_up_bits = 0;
  for (;
      l_start < L;
      l_start += (IS_FULL_WARP ? STEP : (NUM_LOAD_GROUPS * STEP))) {
    if ((l_start % kWarpSize) == 0) {
      const uint32_t l = l_start + threadIdx.x;

      // We expect the following to be loaded from shared memory instead of consuming registers
      const auto* __restrict__ indices =
        *reinterpret_cast<index_t**>(&smem[params_offset + SAVED_PARAMS::P_indices]);
      const auto* __restrict__ weights =
        *reinterpret_cast<emb_vec_t**>(&smem[params_offset + SAVED_PARAMS::P_weights]);
      const auto load_D = static_cast<uint32_t>(smem[params_offset + SAVED_PARAMS::P_load_D]);

      syncwarp(); // Ensure that all warps read the previous value before overwriting it
      if (USE_CACHE_WEIGHTS) {
        int32_t cache_idx = kCacheLocationMissing;
        if (l < L) {
          cache_idx = reinterpret_cast<int32_t*>(smem[params_offset + LXU_CACHE_PARAMS::P_lxu_cache_locations])[l];
          const auto* lxu_cache_weights =
            reinterpret_cast<const cache_t*>(smem[params_offset + LXU_CACHE_PARAMS::P_lxu_cache_weights]);
          SMEM_GENERIC_PTR[threadIdx.x] = cache_idx != kCacheLocationMissing ?
            reinterpret_cast<uintptr_t>(&lxu_cache_weights[cache_idx * max_D_cache]) :
            reinterpret_cast<uintptr_t>(&weights[indices[l] * load_D]);
        }
        if (!std::is_same<emb_t, cache_t>::value) {
          cache_look_up_bits = ballot_sync(cache_idx != kCacheLocationMissing);
          // Shift cache_look_up_bits based on group_id
          cache_look_up_bits >>= static_cast<uint32_t>(threadIdx.x / LOAD_GROUP_SIZE);
        }
      }
      else {
        SMEM_EMB_WEIGHT_PTR[threadIdx.x] = (l < L) ?
          reinterpret_cast<const emb_vec_t*>(&weights[indices[l] * load_D]) : nullptr;
      }
      syncwarp(); // Ensure that all weight pointers are written
    }

    // Make sure that all threads execute the same code
    if (l_start + (IS_FULL_WARP ? STEP : (NUM_LOAD_GROUPS * STEP)) > L) {
      break;
    }

    if (process_d) {
      const auto cache_look_up_bits_step = cache_look_up_bits & STEP_MASK;
      if (USE_MIXED_TYPE_CACHE && cache_look_up_bits_step != 0) {
        {%- if weighted %}
        const auto* index_weights =
          reinterpret_cast<const float*>(smem[params_offset + SAVED_PARAMS::P_index_weights]) + l_start;
        {%- endif %}
        if (cache_look_up_bits_step == STEP_MASK) {
          // Load STEP rows from lxu_cache_weights
          #pragma unroll
          for (uint32_t j = 0; j < STEP; ++j) {
            const auto* weight =
              &SMEM_CACHE_WEIGHT_DATA((l_start % kWarpSize) + SMEM_OFFSET, WEIGHT_OFFSET);
            ACC_ADD_OR_FMA(weight, index_weights[SMEM_OFFSET])
          }
          // Bypass the hip clang error of "shift count >= width of type"
          cache_look_up_bits >>= std::min(STEP * NUM_LOAD_GROUPS, 31u);
        }
        else {
          // Load and accumulate STEP rows for UVM caching that emb_t and cache_t
          // are not the same and rows within STEPS are read from different
          // locations. It is unlikely that the compiler will be able to unroll
          // the loop below because of the runtime conditionals
          #pragma unroll
          for (uint32_t j = 0; j < STEP; ++j) {
            if (cache_look_up_bits & 1u) {
              // Look up from lxu_cache_weights
              const auto* weight =
                &SMEM_CACHE_WEIGHT_DATA((l_start % kWarpSize) + SMEM_OFFSET, WEIGHT_OFFSET);
              ACC_ADD_OR_FMA(weight, index_weights[SMEM_OFFSET])
            }
            else {
              // Look up from dev_weights/uvm_weights
              const auto* weight =
                &SMEM_EMB_WEIGHT_DATA((l_start % kWarpSize) + SMEM_OFFSET, WEIGHT_OFFSET);
              ACC_ADD_OR_FMA(weight, index_weights[SMEM_OFFSET])
            }
            cache_look_up_bits >>= NUM_LOAD_GROUPS;
          }
        }
      }
      else {
        // Load STEP rows from dev_weights
        #pragma unroll
        for (uint32_t j = 0; j < STEP; ++j) {
          accumulator.load(
              &SMEM_EMB_WEIGHT_DATA(
                (l_start % kWarpSize) + SMEM_OFFSET,
                WEIGHT_OFFSET),
              j);
        }

        // Accumulate rows
        {%- if weighted %}
        accumulator.weighted_sum(
            reinterpret_cast<const float*>(smem[params_offset + SAVED_PARAMS::P_index_weights]) + l_start,
            IS_FULL_WARP ? 0 : (threadIdx.x / LOAD_GROUP_SIZE),
            IS_FULL_WARP ? 1 : NUM_LOAD_GROUPS);
        {%- else %}
        accumulator.sum();
        {%- endif %}

        if (USE_MIXED_TYPE_CACHE) {
          // Bypass the hip clang error of "shift count >= width of type"
          cache_look_up_bits >>= std::min(STEP * NUM_LOAD_GROUPS, 31u);
        }
      }
    }
  }

  if (process_d) {
    // Process the remaining indices (less than STEP)
    for (uint32_t j = (IS_FULL_WARP ? 0 : (threadIdx.x / LOAD_GROUP_SIZE));
         j < L - l_start;
         j += (IS_FULL_WARP ? 1 : NUM_LOAD_GROUPS)) {
      // Load and accumulate rows
      const auto weight_offset = WEIGHT_OFFSET;
      {%- if weighted %}
      const auto index_weight =
        reinterpret_cast<float*>(smem[params_offset + SAVED_PARAMS::P_index_weights])[l_start + j];
      {%- endif %}
      if (USE_MIXED_TYPE_CACHE && cache_look_up_bits & 1u) {
        const auto* weight =
          &SMEM_CACHE_WEIGHT_DATA((l_start % kWarpSize) + j, weight_offset);
        ACC_ADD_OR_FMA(weight, index_weight)
      }
      else {
        const auto* weight =
          &SMEM_EMB_WEIGHT_DATA((l_start % kWarpSize) + j, weight_offset);
        ACC_ADD_OR_FMA(weight, index_weight)
      }
      if (USE_MIXED_TYPE_CACHE) {
        cache_look_up_bits >>= NUM_LOAD_GROUPS;
      }
    }
  }

  // Sync accumulator when subwarp is used
  if (!IS_FULL_WARP) {
    for (uint32_t i = NUM_LOAD_GROUPS >> 1; i > 0; i >>= 1) {
      const auto src = i * LOAD_GROUP_SIZE;
      accumulator.acc[0] += shfl_down_sync(accumulator.acc[0], src);
      accumulator.acc[1] += shfl_down_sync(accumulator.acc[1], src);
      accumulator.acc[2] += shfl_down_sync(accumulator.acc[2], src);
      accumulator.acc[3] += shfl_down_sync(accumulator.acc[3], src);
    }
  }

  // Write results to output
  if (process_d && (IS_FULL_WARP || threadIdx.x < LOAD_GROUP_SIZE)) {
    auto* __restrict__ const output =
      *reinterpret_cast<output_vec_t**>(&smem[params_offset + SAVED_PARAMS::P_outputs]) +
      (IS_FULL_WARP ? threadIdx.x : (threadIdx.x % LOAD_GROUP_SIZE));
    if (mean_pooling) {
      accumulator.div(L);
    }
    accumulator.store(output);
  }

#undef SMEM_OFFSET
#undef WEIGHT_OFFSET

}

template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    bool USE_LXU_CACHE
    >
__launch_bounds__(kForwardMaxThreads, 2048 / kForwardMaxThreads)
__global__ void split_embedding_codegen_forward_{{ wdesc }}_v2_kernel(
    const emb_t* __restrict__ const dev_weights,
    const emb_t* __restrict__ const uvm_weights,
    const cache_t* __restrict__ const lxu_cache_weights,
    const int32_t* __restrict__ const weights_placements,
    const uint32_t B,
    const uint32_t T,
    const bool mean_pooling,
    const uint32_t max_D_cache,
    const FixedDivisor fd_num_warps_per_table,
    const index_t* __restrict__ const indices,
    {%- if weighted %}
    const float* __restrict__ const index_weights,
    {%- endif %}
    const index_t* __restrict__ const  offsets,
    const uint32_t* __restrict__ const D_offsets,
    const int64_t* __restrict__ const weights_offsets,
    const int32_t* __restrict__ const lxu_cache_locations,
    output_t* __restrict__ const output) {
    using emb_vec_t = vec4_type<emb_t>;
    using cache_vec_t = vec4_type<cache_t>;
    using output_vec_t = vec4_type<output_t>;

    constexpr uint32_t NUM_WARPS = kForwardMaxThreads / kWarpSize;
    constexpr uint32_t NUM_OFFSETS_PER_WARP = kWarpSize;
    constexpr uint32_t NUM_PARAMS = SAVED_PARAMS_CNT + (USE_LXU_CACHE ? LXU_PARAMS_CNT : 0);
    constexpr uint32_t STEP = 4;
    __shared__ long smem[NUM_PARAMS * NUM_WARPS + kForwardMaxThreads];
    const uint32_t params_offset = NUM_PARAMS * threadIdx.y;

    const int32_t global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int32_t t;
    int32_t table_warp_id;
    fd_num_warps_per_table.DivMod(global_warp_id, &t, &table_warp_id);

    if (t >= T) {
      return;
    }

    const auto total_L = offsets[(t + 1) * B] - offsets[t * B];
    const auto is_zero_total_L = total_L == 0;

    // Short circuit for all zeros
    if (is_zero_total_L) {
      const uint32_t D_start = D_offsets[t] / VEC_WIDTH;
      const uint32_t load_D = (D_offsets[t + 1] / VEC_WIDTH) - D_start;
      const uint32_t num_warps_per_row = DIV_ROUND_UP(load_D, kWarpSize);
      if (table_warp_id >= num_warps_per_row * B) {
        return;
      }
      const uint32_t load_d = (table_warp_id % num_warps_per_row) * kWarpSize;
      if (load_d + threadIdx.x < load_D) {
        const uint32_t b = table_warp_id / num_warps_per_row;
        const uint32_t total_load_D = D_offsets[T] / VEC_WIDTH;

        output_vec_t* output_ptr = reinterpret_cast<output_vec_t*>(output) +
            D_start + b * total_load_D + load_d + threadIdx.x;

        // Write zeros to output
        Vec4StepT<1, emb_t> accumulator;
        accumulator.store(output_ptr);
      }
      return;
    }

    // Use the small-L optimization if average L <= 8
    const auto is_small_L = total_L <= (static_cast<index_t>(B) * 8);
    const uint32_t num_warps_for_small_L = DIV_ROUND_UP(B, NUM_OFFSETS_PER_WARP);

    // Early exit for small-L to avoid D_offsets reads
    // if table_warp_id > B * max(num_warps_per_row) / NUM_OFFSETS_PER_WARP
    // max(num_warps_per_row) = 8 (for D = 1024)
    // NUM_OFFSETS_PER_WARP = 32
    // Return if table_warp_id > ceil(B / 32) * 8
    if (is_small_L && table_warp_id >= num_warps_for_small_L * 8) {
      return;
    }

    uint32_t load_D;
    uint32_t D_start;
    uint32_t total_load_D;

    if (threadIdx.x == 0) {
      D_start = D_offsets[t] / VEC_WIDTH;
      load_D = (D_offsets[t + 1] / VEC_WIDTH) - D_start;
    }
    load_D = shfl_sync(load_D, 0);

    const uint32_t num_warps_per_row = DIV_ROUND_UP(load_D, kWarpSize);

    if (table_warp_id >= num_warps_per_row * (is_small_L ? num_warps_for_small_L : B)) {
      return;
    }

    // Compute d (same for all Ls)
    const uint32_t load_d = (table_warp_id % num_warps_per_row) * kWarpSize;
    // Compute sample ID
    const uint32_t b = table_warp_id / num_warps_per_row * (is_small_L ? NUM_OFFSETS_PER_WARP : 1);
    uint32_t L;
    uint32_t row_start;
    bool use_lxu_cache = USE_LXU_CACHE;

    if (threadIdx.x == 0) {
      // If the small-L optimization is used, each warp processes up to 32
      // rows. Otherwise, each warp processes only 1 row.
      auto num_offsets = is_small_L ? (B - b < NUM_OFFSETS_PER_WARP ? B - b : NUM_OFFSETS_PER_WARP) : 1;
      row_start = offsets[t * B + b];
      L = offsets[t * B + b + num_offsets] - row_start;
      total_load_D = (D_offsets[T] / VEC_WIDTH);
      // Manual register spilling (Store state registers in shared memory to
      // free up some registers for compiler optimizations)
      if (L > 1 || is_small_L) {
        *reinterpret_cast<const index_t**>(&smem[params_offset + SAVED_PARAMS::P_indices]) =
          indices + row_start;
        const auto placement = static_cast<PlacementType>(weights_placements[t]);
        const auto weight_offset = weights_offsets[t];
        const emb_t* weight = placement == PlacementType::DEVICE ?
          &dev_weights[weight_offset] : &uvm_weights[weight_offset];
        *reinterpret_cast<const emb_vec_t**>(&smem[params_offset + SAVED_PARAMS::P_weights]) =
          reinterpret_cast<const emb_vec_t*>(weight) + load_d;
        *reinterpret_cast<output_vec_t**>(&smem[params_offset + SAVED_PARAMS::P_outputs]) =
          reinterpret_cast<output_vec_t*>(output) + D_start + b * total_load_D + load_d;
        {%- if weighted %}
        *reinterpret_cast<const float**>(&smem[params_offset + SAVED_PARAMS::P_index_weights]) =
          index_weights + row_start;
        {%- endif %}
        if (is_small_L) {
          *reinterpret_cast<const index_t**>(&smem[params_offset + SAVED_PARAMS::P_offsets]) = &offsets[t * B + b];
          smem[params_offset + SAVED_PARAMS::P_num_offsets] = num_offsets;
        }
        smem[params_offset + SAVED_PARAMS::P_load_D] = load_D;
        smem[params_offset + SAVED_PARAMS::P_total_load_D] = total_load_D;
        if (USE_LXU_CACHE) {
          if (placement == PlacementType::MANAGED_CACHING) {
            *reinterpret_cast<const cache_t**>(&smem[params_offset + LXU_CACHE_PARAMS::P_lxu_cache_weights]) =
              lxu_cache_weights + (load_d * VEC_WIDTH);
            *reinterpret_cast<const int32_t**>(&smem[params_offset + LXU_CACHE_PARAMS::P_lxu_cache_locations]) =
              lxu_cache_locations + row_start;
          }
          else {
            use_lxu_cache = false;
          }
        }
      }
    }

    L = shfl_sync(L, 0);
    use_lxu_cache = shfl_sync(use_lxu_cache, 0);

#define INVOKE_PROCESS_ALL_INDICES_HELPER(USE_CACHE, KERNEL_TYPE, TAIL_WARP_SIZE, STEP_MASK) \
    process_all_indices_## KERNEL_TYPE< \
      index_t, \
      emb_t, \
      emb_vec_t, \
      cache_t, \
      cache_vec_t, \
      output_vec_t, \
      USE_CACHE, \
      USE_CACHE && !std::is_same<emb_t, cache_t>::value, \
      NUM_PARAMS * NUM_WARPS, \
      STEP, \
      STEP_MASK, \
      TAIL_WARP_SIZE \
    >( \
        smem, \
        L, \
        load_d + (threadIdx.x % TAIL_WARP_SIZE) < load_D, \
        mean_pooling, \
        params_offset, \
        max_D_cache)

#define INVOKE_PROCESS_ALL_INDICES(...) \
    if (use_lxu_cache) { \
      INVOKE_PROCESS_ALL_INDICES_HELPER(true, __VA_ARGS__); \
    } \
    else { \
      INVOKE_PROCESS_ALL_INDICES_HELPER(false, __VA_ARGS__); \
    }

    if (is_small_L) {
      INVOKE_PROCESS_ALL_INDICES(small_Ls, 32, 0xf)
      return;
    }

    // Special cases
    if (L <= 1) {
      total_load_D = shfl_sync(total_load_D, 0);
      D_start = shfl_sync(D_start, 0);
      output_vec_t* output_ptr = reinterpret_cast<output_vec_t*>(output) +
          D_start + b * total_load_D + load_d + threadIdx.x;
      if (L == 0) {
        if (load_d + threadIdx.x < load_D) {
          // Write zeros to output
          Vec4StepT<1, emb_t> accumulator;
          accumulator.store(output_ptr);
        }
      }
      else {
        row_start = shfl_sync(row_start, 0);
        if (load_d + threadIdx.x < load_D) {
          const auto placement = static_cast<PlacementType>(weights_placements[t]);
          const auto weight_offset = weights_offsets[t];
          Vec4StepT<1, emb_t> accumulator;
          {%- if weighted %}
          const auto index_weight = index_weights[row_start];
          {%- endif %}
          const int32_t cache_idx = USE_LXU_CACHE && placement == PlacementType::MANAGED_CACHING ?
            lxu_cache_locations[row_start] : kCacheLocationMissing;
          if (USE_LXU_CACHE &&
              placement == PlacementType::MANAGED_CACHING &&
              cache_idx != kCacheLocationMissing) {
            const auto* weight =
              reinterpret_cast<const cache_vec_t*>(&lxu_cache_weights[cache_idx * max_D_cache]);
            const auto weight_vec_offset = load_d + threadIdx.x;
            ACC_ADD_OR_FMA(weight + weight_vec_offset, index_weight);
          }
          else {
            const auto* weight = reinterpret_cast<const emb_vec_t*>(placement == PlacementType::DEVICE ?
                &dev_weights[weight_offset] : &uvm_weights[weight_offset]);
            const auto weight_vec_offset = load_d + indices[row_start] * load_D + threadIdx.x;
            // Load and store data
            ACC_ADD_OR_FMA(weight + weight_vec_offset, index_weight)
          }
          accumulator.store(output_ptr);
        }
      }
      return;
    }

    // Tail warp
    // STEP_MASK computation assumes STEP = 4
    {% if not weighted %}
    if (load_D - load_d < kWarpSize) {
      const auto tail_warp_size = load_D % kWarpSize;
      if (tail_warp_size <= 8) {
        INVOKE_PROCESS_ALL_INDICES(large_Ls, 8, 0x1111)
      }
      else if (tail_warp_size <= 16) {
        INVOKE_PROCESS_ALL_INDICES(large_Ls, 16, 0x55)
      }
      else {
        INVOKE_PROCESS_ALL_INDICES(large_Ls, 32, 0xf)
      }
    }
    else {
      INVOKE_PROCESS_ALL_INDICES(large_Ls, 32, 0xf)
    }
    {% else %}
    INVOKE_PROCESS_ALL_INDICES(large_Ls, 32, 0xf)
    {% endif %}

#undef INVOKE_PROCESS_ALL_INDICES_HELPER
#undef INVOKE_PROCESS_ALL_INDICES

}

/*
    Explicitly instantiate the kernel function template.  The instantiations are
    based on the types enumerated by DISPATCH_EMB_CACHE_TYPES macro used in
    embedding_forward_split_template.cu
*/

{%- for output_type in ['float', 'at::Half', 'at::BFloat16'] %}
{%- for emb_type in ['float', 'at::Half'] %}
{%- for cache_type in ['float', 'at::Half'] %}
{%- for use_cache in ['true', 'false'] %}

template __launch_bounds__(kForwardMaxThreads, 2048 / kForwardMaxThreads)
__global__ void split_embedding_codegen_forward_{{ wdesc }}_v2_kernel
<
    {{ emb_type }},
    {{ cache_type }},
    {{ output_type }},
    int64_t, // index_t
    {{ use_cache }}
> (
    const {{ emb_type }}* __restrict__ const dev_weights,
    const {{ emb_type }}* __restrict__ const uvm_weights,
    const {{ cache_type }}* __restrict__ const lxu_cache_weights,
    const int32_t* __restrict__ const weights_placements,
    const uint32_t B,
    const uint32_t T,
    const bool mean_pooling,
    const uint32_t max_D_cache,
    const FixedDivisor fd_num_warps_per_table,
    const int64_t* __restrict__ const indices,
    {%- if weighted %}
    const float* __restrict__ const index_weights,
    {%- endif %}
    const int64_t* __restrict__ const  offsets,
    const uint32_t* __restrict__ const D_offsets,
    const int64_t* __restrict__ const weights_offsets,
    const int32_t* __restrict__ const lxu_cache_locations,
    {{ output_type }}* __restrict__ const output);

{%- endfor %}
{%- endfor %}
{%- endfor %}
{%- endfor %}

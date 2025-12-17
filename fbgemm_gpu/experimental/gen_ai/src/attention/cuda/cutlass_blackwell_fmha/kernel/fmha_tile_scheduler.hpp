// @nolint
/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/


#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::fmha::kernel {

////////////////////////////////////////////////////////////////////////////////

// Standard Individual Tile Scheduler (No Split-K)
// Used by non-decode kernels (dense, varlen, etc.)
struct IndividualTileScheduler {

  struct Params {
    dim3 grid;
  };

  bool valid_ = true;

  CUTLASS_DEVICE
  IndividualTileScheduler(Params const&) {}

  template<class ProblemSize, class ClusterShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size, KernelHardwareInfo hw_info,
      ClusterShape const& cluster_shape, TileShape const& tile_shape) {
    using namespace cute;
    dim3 grid(round_up(ceil_div(size<0>(problem_size), size<0>(tile_shape)), size<0>(cluster_shape)), size<3,0>(problem_size), size<3,1>(problem_size));
    return Params{ grid };
  }

  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    return make_coord(blockIdx.x, _0{}, make_coord(blockIdx.y, blockIdx.z));
  }

  CUTLASS_DEVICE
  IndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////

// Split-K Decode Scheduler 
//
// This scheduler implements split-K attention where each sequence is divided
// into fixed-size chunks of 1024 tokens along the K/V sequence dimension.
// 
// Grid Layout:
//   - grid.x: Q tiles (typically 1 for decode with single query token)
//   - grid.y: H_k * num_splits (packed dimension combining heads and splits)
//   - grid.z: Batch dimension
//
// Benefits:
//   - Better parallelism for long sequences 
//   - More efficient SM utilization with multiple splits per sequence
//   - Deterministic outputs when split size is fixed (batch-size invariant)

struct IndividualTileSchedulerSplitK {

  struct Params {
    dim3 grid;
    int num_splits;
    int h_k;
  };

  bool valid_ = true;
  Params params_;

  CUTLASS_DEVICE
  IndividualTileSchedulerSplitK(Params const& params) : params_(params) {}

  template<class ProblemSize, class ClusterShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size, KernelHardwareInfo hw_info,
      ClusterShape const& cluster_shape, TileShape const& tile_shape,
      int splitk_size,
      int window_size = -1) {
    using namespace cute;
    
    // Calculate number of splits based on sequence length
    // If splitk_size is 0, default to no splitting (num_splits = 1)
    int sk = size<1>(problem_size);  // sequence length
    
    // For sliding window attention, use window_size as the effective seqlen
    // This ensures the grid only launches CTAs for splits within the window
    int effective_seqlen = sk;
    if (window_size > 0 && window_size < sk) {
      effective_seqlen = window_size;
    }
    
    int num_splits;
    if (splitk_size <= 0) {
      num_splits = 1;  // No splitting
    } else {
      num_splits = (effective_seqlen + splitk_size - 1) / splitk_size;  // ceil division
    }
    
    // Grid is now (Q_tiles, H_k * SplitK, B)
    // We pack H_k and SplitK into grid.y to maintain 3D grid
    int h_k = size<3,0>(problem_size);
    dim3 grid(
        round_up(ceil_div(size<0>(problem_size), size<0>(tile_shape)), size<0>(cluster_shape)), 
        h_k * num_splits,  // Pack H_k and SplitK into y dimension
        size<3,1>(problem_size)  // Batch (B)
    );
    return Params{ grid, num_splits, h_k };
  }

  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    // Unpack the packed y dimension to get H_k and SplitK indices
    int packed_y = blockIdx.y;
    int h_k_idx = packed_y % params_.h_k;
    int split_k_idx = packed_y / params_.h_k;
    
    // Return coordinate with split_k_idx in the second position
    // This matches the expected coordinate structure for gen kernel
    return make_coord(blockIdx.x, split_k_idx, make_coord(h_k_idx, blockIdx.z));
  }

  CUTLASS_DEVICE
  IndividualTileSchedulerSplitK& operator++() {
    valid_ = false;
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////

struct PersistentTileScheduler {

  struct Params {
    int num_blocks;
    FastDivmod divmod_m_block;
    FastDivmod divmod_h;
    FastDivmod divmod_b;

    KernelHardwareInfo hw_info;
  };

  int block_idx = 0;
  Params params;

  CUTLASS_DEVICE
  PersistentTileScheduler(Params const& params) : block_idx(blockIdx.x), params(params) {}

  template<class ProblemSize, class ClusterShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size, KernelHardwareInfo hw_info,
      ClusterShape const& cluster_shape, TileShape const& tile_shape,
      int splitk_size = 0,
      int window_size = -1) {
    using namespace cute;
    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    hw_info.sm_count = sm_count;

    int num_m_blocks = cutlass::round_up(ceil_div(size<0>(problem_size), size<0>(tile_shape)), size<0>(cluster_shape));
    int num_blocks = num_m_blocks * size<3,0>(problem_size) * size<3,1>(problem_size);

    return Params {
      num_blocks,
      { num_m_blocks}, { size<3,0>(problem_size) }, { size<3,1>(problem_size) },
      hw_info
    };
  }

  static dim3 get_grid_shape(Params const& params) {
    dim3 grid(std::min(params.num_blocks, params.hw_info.sm_count), 1, 1);
    return grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return block_idx < params.num_blocks;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    int block_decode = block_idx;
    int m_block, bidb, bidh;
    params.divmod_m_block(block_decode, m_block, block_decode);
    params.divmod_b(block_decode, bidb, block_decode);
    params.divmod_h(block_decode, bidh, block_decode);
    return make_coord(m_block, _0{}, make_coord(bidh, bidb));
  }

  CUTLASS_DEVICE
  PersistentTileScheduler& operator++() {
    block_idx += gridDim.x;
    return *this;
  }
};


////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::fmha::kernel

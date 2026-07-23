/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/fast_math.h"

#include "named_barrier.hpp"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

class SingleTileSchedulerBwd {
 public:
  // Host side kernel arguments
  struct Arguments {
    int const num_blocks_m, num_head, num_batch;
  };

  // Device side kernel params
  struct Params {};

  static Params to_underlying_arguments(Arguments const& args) {
    return {};
  }

  static dim3 get_grid_dim(Arguments const& args) {
    return {uint32_t(args.num_blocks_m), uint32_t(args.num_head), uint32_t(args.num_batch)};
  }

  struct WorkTileInfo {
    int M_idx = 0;
    int H_idx = 0;
    int B_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return is_valid_tile;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t> get_block_coord(
        Params const& params) const {
      return {M_idx, H_idx, B_idx};
    }
  };

  CUTLASS_DEVICE
  SingleTileSchedulerBwd() {}

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work() const {
    return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work)
      const {}

  template <bool IsProducer = false>
  CUTLASS_DEVICE WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {-1, -1, -1, false};
  }
};

} // namespace flash

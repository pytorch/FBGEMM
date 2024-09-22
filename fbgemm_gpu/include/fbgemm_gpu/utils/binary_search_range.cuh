/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace fbgemm_gpu {

template <typename scalar_t>
__device__ __forceinline__ void binary_search_range(
    int* found,
    const scalar_t* arr,
    const scalar_t target,
    const int num_entries) {
  const int last_entry = num_entries - 1;
  int start = 0, end = last_entry;
  int found_ = -1;
  while (start <= end) {
    int mid = start + (end - start) / 2;
    scalar_t mid_offset = arr[mid];
    if (target == mid_offset) {
      if (mid != last_entry && target != arr[last_entry]) {
        // Do linear scan in case of duplicate data (We assume that the
        // number of duplicates is small.  This can we very bad if the
        // number of duplicates is large)
        for (int i = mid + 1; i < num_entries; i++) {
          if (target != arr[i]) {
            found_ = i;
            break;
          }
        }
      }
      break;
    } else if (target < mid_offset) {
      if (mid == 0) {
        found_ = 0;
        break;
      } else if (mid - 1 >= 0 && target > arr[mid - 1]) {
        found_ = mid;
        break;
      }
      end = mid - 1;
    } else {
      if (mid + 1 <= last_entry && target < arr[mid + 1]) {
        found_ = mid + 1;
        break;
      }
      start = mid + 1;
    }
  }
  *found = found_;
}

} // namespace fbgemm_gpu

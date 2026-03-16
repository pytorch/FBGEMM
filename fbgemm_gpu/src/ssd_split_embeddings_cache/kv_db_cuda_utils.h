/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <cuda_runtime.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

namespace kv_db_utils {

/// @ingroup embedding-ssd
///
/// @brief A host function for `cudaLaunchHostFunc`
///
/// A common host function for `cudaLaunchHostFunc`, i.e.,
/// `cudaHostFn_t`. This function casts `functor` into a void function,
/// invokes it and then deletes it (the deletion occurs in another thread).
///
/// Unlike `cudaStreamAddCallback`, `cudaLaunchHostFunc` does not hold the
/// CUDA driver mutex during execution, allowing concurrent CUDA API calls
/// from other threads (e.g., NCCL kernel launches on other streams).
///
/// @param functor A functor that will be called
///
/// @return None
void cuda_host_func(void* functor);

}; // namespace kv_db_utils

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

namespace kv_db_utils {

/// @brief A callback function for `cudaStreamAddCallback`
///
/// A common callback function for `cudaStreamAddCallback`, i.e.,
/// `cudaStreamCallback_t callback`. This function casts `functor`
/// into a void function, invokes it and then delete it (the deletion
/// occurs in another thread)
///
/// @param stream (`cudaStream_t`) CUDA stream that
///               `cudaStreamAddCallback` operates on
/// @param status (`cudaError_t`) CUDA status
/// @param functor (`void*`) A functor that will be called
///
/// @return None
void cuda_callback_func(cudaStream_t stream, cudaError_t status, void* functor);

}; // namespace kv_db_utils

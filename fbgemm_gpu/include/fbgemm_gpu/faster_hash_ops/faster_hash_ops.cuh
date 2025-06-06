/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

/// @defgroup faster-hash-ops CUDA Operators
/// The following are CUDA Operators

namespace fbgemm_gpu {

using at::Tensor;

///@ingroup faster-hash-ops
///
/// CUDA implementation of zero collision hash
///
/// @param output the output tensor that will be modified in place
/// @param evict_slots the slots that will be evicted
/// @param input the input tensor
/// @param identities the identity tensor
/// @param max_probe the maximum number of probes
/// @param circular_probe whether to use circular probe
/// @param cur_hour the current hour
/// @param readonly whether to use readonly mode
/// @param support_evict whether to support evict
/// @param local_sizes the local sizes tensor
/// @param offsets the offsets tensor
/// @param hash_identity whether to hash the identity
/// @param metadata the metadata tensor
/// @param disable_fallback whether to disable fallback
/// @param input_metadata the input metadata tensor
/// @param eviction_threshold the eviction threshold
/// @param eviction_policy the eviction policy
/// @param opt_in_prob the opt-in probability
/// @param num_reserved_slots the number of reserved slots
/// @param opt_in_rands the opt-in randoms tensor
///
/// @return None
template <typename TInput, typename TIdentity>
void _zero_collision_hash_cuda(
    Tensor& output,
    Tensor& evict_slots,
    const Tensor& input,
    Tensor& identities,
    int64_t max_probe,
    bool circular_probe,
    int64_t cur_hour,
    bool readonly,
    bool support_evict,
    const std::optional<Tensor>& local_sizes,
    const std::optional<Tensor>& offsets,
    int32_t hash_identity,
    const std::optional<Tensor>& metadata,
    bool disable_fallback,
    const std::optional<Tensor>& input_metadata,
    int64_t eviction_threshold,
    int64_t eviction_policy,
    int64_t opt_in_prob,
    int64_t num_reserved_slots,
    const std::optional<Tensor>& opt_in_rands);

///@ingroup faster-hash-ops
///
/// CUDA implementation of murmurhash3
///
/// @param input the input tensor
/// @param y the y value
/// @param seed the seed value

/// @return the output tensor
Tensor murmur_hash3_cuda(const Tensor& input, int64_t y, int64_t seed);

} // namespace fbgemm_gpu

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

/// @defgroup faster-hash-ops-cuda CUDA Operators
/// The following are CUDA Operators

namespace fbgemm_gpu {

using at::Tensor;

/// @ingroup faster-hash-ops-cuda
///
/// @brief CUDA implementation of zero collision hash
/// This function performs zero collision hash on the input feature IDs in the
/// input tensor and returns the remapped IDs in the output tensor. It also
/// updates the metadata table if the eviction policy is enabled.
/// Specifically, it performs the following steps:
/// 1. For each input feature ID, it computes the hash value using the
/// MurmurHash3 algorithm. And the hash value will be forwarded to the identity
/// table (tensor named identities).
/// 2. Check if the slot in the identity table indexed by the hash value is
/// empty. If it is empty, the feature ID will be inserted into the slot and the
/// hash value will be returned as the remapped ID.
/// 3. If the slot is not empty, it will linearly probe the next slot until it
/// finds an empty slot or reaches the maximum number of probes. If an empty
/// slot is found, the feature ID will be inserted into that slot and the index
/// of the empty slot will be returned as the remapped ID.
/// 4. If no empty slot is found, it will find the evictable slot based on the
/// eviction policy and evict the feature ID in that slot. Then, it will insert
/// the current feature ID into the evicted slot and return the index of the
/// evicted slot as the remapped ID. The metadata table will also be updated
/// accordingly.
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
/// @return None (the output tensor will be modified in place)
///
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

///@ingroup faster-hash-ops-cuda
///
/// @brief Murmur hash operator for CUDA device
///
/// This function implements the Murmur hash algorithm. Given an input tensor
/// a y value and a seed value, it returns the hash value of the input tensor.
/// The hash value is calculated using the Murmur hash3 x64 algorithm
/// implemented in the `murmur_hash3_2x64` function in `common_utils.cuh`.
///
/// @param input the input tensor
/// @param y the y value
/// @param seed the seed value

/// @return the output tensor
Tensor murmur_hash3_cuda(const Tensor& input, int64_t y, int64_t seed);

} // namespace fbgemm_gpu

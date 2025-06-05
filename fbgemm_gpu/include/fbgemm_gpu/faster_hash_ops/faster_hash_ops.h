/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {

using at::Tensor;

/// @defgroup faster-hash-ops CPP Operators
///

/// @ingroup faster-hash-ops
///
/// Create buffers for identity table and metadata table for ZCH
///
/// @param size The target tensor dimensions
/// @param support_evict Whether to support eviction
/// @param device The device to allocate the tensor on
/// @param long_type Whether to use long type for the tensor
///
/// @return A tuple of two tensors, the first tensor is the
// identity table and the second tensor is the metadata table
std::tuple<Tensor, Tensor> create_zch_buffer_cpu(
    const int64_t size,
    bool support_evict,
    std::optional<at::Device> device,
    bool long_type);

/// @ingroup faster-hash-ops
///
/// Murmur hash operator for CPU
///
/// @param input The input tensor
/// @param y The y value
/// @param seed The seed value
///
/// @return The output hash value
Tensor murmur_hash3_cpu(const Tensor& input, int64_t y, int64_t seed);

/// @ingroup faster-hash-ops
///
/// Zero collision hash operator for CPU
///
/// @param input The input tensor
/// @param identities The identity table
/// @param max_probe The maximum number of probes
/// @param circular_probe Whether to use circular probe
/// @param exp_hours The number of hours before identity table item's
/// expirition
/// @param readonly Whether to use readonly mode
/// @param local_sizes The local sizes tensor
/// @param offsets The offsets tensor
/// @param metadata The metadata tensor
/// @param output_on_uvm Whether to output on UVM
/// @param disable_fallback Whether to disable fallback
/// @param _modulo_identity_DPRECATED The modulo identity
/// @param input_metadata The input metadata tensor
/// @param eviction_threshold The eviction threshold
/// @param eviction_policy The eviction policy
/// @param opt_in_prob The opt-in probability
/// @param num_reserved_slots The number of reserved slots
/// @param opt_in_rands The opt-in randoms tensor
///
/// @return A tuple of two tensors, the first tensor is the
/// output tensor and the second tensor is the slots to be evicted
std::tuple<Tensor, Tensor> zero_collision_hash_cpu(
    const Tensor& input,
    Tensor& identities,
    int64_t max_probe,
    bool circular_probe,
    int64_t exp_hours,
    bool readonly,
    const std::optional<Tensor>& local_sizes,
    const std::optional<Tensor>& offsets,
    const std::optional<Tensor>& metadata,
    bool /* output_on_uvm */,
    bool disable_fallback,
    bool _modulo_identity_DPRECATED,
    const std::optional<Tensor>& input_metadata,
    int64_t eviction_threshold,
    int64_t /* eviction_policy */,
    int64_t opt_in_prob,
    int64_t num_reserved_slots,
    const std::optional<Tensor>& opt_in_rands);

/// @ingroup faster-hash-ops
///
/// Zero collision hash operator for data on meta device
///
/// @param input The input tensor
/// @param identities The identity table
/// @param max_probe The maximum number of probes
/// @param circular_probe Whether to use circular probe
/// @param exp_hours The number of hours before identity table item's expirition
/// @param readonly Whether to use readonly mode
/// @param local_sizes The local sizes tensor
/// @param offsets The offsets tensor
/// @param metadata The metadata tensor
/// @param output_on_uvm Whether to output on UVM
/// @param disable_fallback Whether to disable fallback
/// @param _modulo_identity_DPRECATED The modulo identity
/// @param input_metadata The input metadata tensor
/// @param eviction_threshold The eviction threshold
/// @param eviction_policy The eviction policy
/// @param opt_in_prob The opt-in probability
/// @param num_reserved_slots The number of reserved slots
/// @param opt_in_rands The opt-in randoms tensor
///
/// @return A tuple of two tensors, the first tensor is the
/// output tensor and the second tensor is the slots to be evicted
std::tuple<Tensor, Tensor> zero_collision_hash_meta(
    const Tensor& input,
    Tensor& /* identities */,
    int64_t /* max_probe */,
    bool /* circular_probe */,
    int64_t /* exp_hours */,
    bool /* readonly */,
    const std::optional<Tensor>& /* local_sizes */,
    const std::optional<Tensor>& /* offsets */,
    const std::optional<Tensor>& /* metadata */,
    bool /* output_on_uvm */,
    bool /* disable_fallback */,
    bool /* _modulo_identity_DPRECATED */,
    const std::optional<Tensor>& /* input_metadata */,
    int64_t /* eviction_threshold */,
    int64_t /* eviction_policy */,
    int64_t /* opt_in_prob */,
    int64_t /* num_reserved_slots */,
    const std::optional<Tensor>& /* opt_in_rands */);

/// @ingroup faster-hash-ops
///
/// Murmur hash operator for Meta device
///
/// @param input The input tensor
/// @param y The y value
/// @param seed The seed value
Tensor murmur_hash3_meta(const Tensor& input, int64_t y, int64_t seed);

// /// @ingroup faster-hash-ops
// ///
// /// process one item for zero collision hash
// ///
// /// @param input The input tensor
// /// @param output The output tensor
// /// @param identities The identity table
// /// @param modulo The modulo
// /// @param max_probe The maximum number of probes
// /// @param local_sizes The local sizes tensor
// /// @param offsets The offsets tensor
// /// @param opt_in_prob The opt-in probability
// /// @param num_reserved_slots The number of reserved slots
// ///
// /// @return A template with the following parameters:
// /// DISABLE_FALLBACK: Whether to disable fallback
// /// HASH_IDENTITY: The hash identity
// /// CIRCULAR_PROBE: Whether to use circular probe
// /// HAS_OFFSET: Whether to have offset
// /// - TInput: The type of the input tensor
// /// - TIdentity: The type of the identity table
// template <
//     bool DISABLE_FALLBACK,
//     int32_t HASH_IDENTITY,
//     bool CIRCULAR_PROBE,
//     bool HAS_OFFSET,
//     typename TInput,
//     typename TIdentity>
// void process_item_zch(
//     const at::PackedTensorAccessor64<TInput, 1>& input,
//     at::PackedTensorAccessor64<int64_t, 1> output,
//     const at::PackedTensorAccessor64<TIdentity, 2>& identities,
//     int64_t modulo,
//     int64_t max_probe,
//     const int64_t* const local_sizes,
//     const int64_t* const offsets,
//     int64_t opt_in_prob,
//     int64_t num_reserved_slots)

} // namespace fbgemm_gpu

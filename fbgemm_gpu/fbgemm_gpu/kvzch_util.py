# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from torchrec.modules.embedding_configs import (
    CountBasedEvictionPolicy,
    CountTimestampMixedEvictionPolicy,
    FeatureL2NormBasedEvictionPolicy,
    NoEvictionPolicy,
    TimestampBasedEvictionPolicy,
    VirtualTableEvictionPolicy,
)


def parse_metadata_tensor(metadata_tensor: torch.Tensor):
    """
    Parses a kvzch metadata tensor where each element encodes three pieces of information
    packed into a single 64-bit integer.
    The 64-bit integer layout is as follows:
    - The lower 32 bits (bits 0-31) represent the 'timestamp', stored as a uint32.
      This timestamp is typically in seconds and can represent a range of over 120 years.
    - The upper 32 bits (bits 32-63) encode two fields packed together:
        * The lower 31 bits of this upper half (bits 32-62 overall) represent 'count',
          a 31-bit unsigned integer indicating a usage count or score.
        * The highest bit of this upper half (bit 63 overall) represents 'used',
          a boolean flag indicating whether the block is currently in use.
    This function extracts these three components from each 64-bit integer in the tensor:
    - 'timestamps' as a uint32 array
    - 'counts' as a uint32 array (31 bits used)
    - 'used' as a boolean array
    Args:
        metadata_tensor (torch.Tensor): A 1D tensor of dtype torch.int64, where each
                                        element encodes timestamp, count, and used flag.
    Returns:
        tuple: (timestamps, counts, used)
            - timestamps (tensor): int64 array of timestamps extracted from the tensor.
            - counts (tensor): int64 array of counts extracted from the tensor.
            - used (tensor): boolean array indicating usage flags extracted from the tensor.
    """
    assert metadata_tensor.dtype == torch.int64
    timestamps = metadata_tensor & 0xFFFFFFFF  # Extract lower 32 bits as timestamp
    count_used = (
        metadata_tensor >> 32
    )  # Extract upper 32 bits containing count and used
    counts = count_used & 0x7FFFFFFF  # Lower 31 bits of upper half as count
    used = ((count_used >> 31) & 1).to(
        torch.bool
    )  # Highest bit of upper half as used flag
    return timestamps, counts, used


def get_kv_zch_eviction_mask(
    metadata_tensor: torch.Tensor,
    eviction_policy: VirtualTableEvictionPolicy,
):
    """
    Returns a boolean mask indicating which blocks should be evicted from the KV cache.
    The eviction policy is determined by the 'eviction_policy' argument.
    Args:
        metadata_tensor (torch.Tensor): A 1D tensor of dtype torch.int64, where each
                                        element encodes timestamp, count, and used flag.
        eviction_policy (VirtualTableEvictionPolicy): The eviction policy to use.
    Returns:
        torch.Tensor: A 1D boolean tensor of the same size as 'metadata_tensor', where False indicates a block should be evicted.
    """

    eviction_mask = torch.ones_like(
        metadata_tensor, dtype=torch.bool
    )  # Initialize mask to True (keep all blocks)
    if isinstance(eviction_policy, NoEvictionPolicy):
        return eviction_mask

    # Parse the metadata tensor to extract timestamps, counts, and used flags
    timestamps, counts, _ = parse_metadata_tensor(metadata_tensor)

    # Apply the eviction policy to determine which blocks should be evicted
    # Check which policy is being used
    if isinstance(eviction_policy, CountBasedEvictionPolicy):
        inference_eviction_threshold = eviction_policy.inference_eviction_threshold
        eviction_mask = counts >= inference_eviction_threshold

    elif isinstance(eviction_policy, TimestampBasedEvictionPolicy):
        inference_eviction_ttl_mins = eviction_policy.inference_eviction_ttl_mins
        if inference_eviction_ttl_mins != 0:  # eviction_ttl_mins == 0 means no eviction
            current_time = int(time.time())
            eviction_mask = (
                current_time - timestamps
            ) <= inference_eviction_ttl_mins * 60

    elif isinstance(eviction_policy, CountTimestampMixedEvictionPolicy):
        inference_eviction_threshold = eviction_policy.inference_eviction_threshold
        inference_eviction_ttl_mins = eviction_policy.inference_eviction_ttl_mins
        current_time = int(time.time())
        eviction_ttl_secs = inference_eviction_ttl_mins * 60
        if inference_eviction_threshold == 0:
            count_mask = torch.ones_like(counts, dtype=torch.bool)
        else:
            count_mask = counts >= inference_eviction_threshold

        if inference_eviction_ttl_mins == 0:
            timestamp_mask = torch.ones_like(counts, dtype=torch.bool)
        else:
            timestamp_mask = (current_time - timestamps) <= eviction_ttl_secs
        eviction_mask = count_mask & timestamp_mask

    elif isinstance(eviction_policy, FeatureL2NormBasedEvictionPolicy):
        # Feature L2 norm-based eviction logic
        # No op for now
        pass
    else:
        raise ValueError("Unsupported eviction policy")

    return eviction_mask

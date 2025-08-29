# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch


def _scaled_mm_helper(
    a, b, scale_a_factor=1.0, scale_b_factor=1.0, out_dtype=torch.float32
):
    """
    Helper function to perform scaled matrix multiplication with proper memory layout.

    Args:
        a: First matrix tensor (2D)
        b: Second matrix tensor (2D)
        scale_a_factor: Scale factor for matrix a
        scale_b_factor: Scale factor for matrix b
        out_dtype: Output data type

    Returns:
        Result of scaled matrix multiplication
    """
    # Ensure proper memory layout for cuBLASLt
    # cuBLASLt requires one matrix to be row-major and the other column-major
    a = a.contiguous()  # row-major
    b = b.t().contiguous().t()  # force column-major layout

    # Convert to fp8 if needed
    a_fp8 = a.to(torch.float8_e4m3fn) if a.dtype != torch.float8_e4m3fn else a
    b_fp8 = b.to(torch.float8_e4m3fn)

    # Create scale tensors
    scale_a = torch.tensor(scale_a_factor, device=a.device, dtype=torch.float32)
    scale_b = torch.tensor(scale_b_factor, device=a.device, dtype=torch.float32)

    return torch._scaled_mm(
        a_fp8,
        b_fp8,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
        use_fast_accum=True,
    )


def _batched_scaled_mm(
    a_batch, b_batch, scale_a_factor=1.0, scale_b_factor=1.0, handle_mqa_gqa=True
):
    """
    Handle batched scaled matrix multiplication with optional MQA/GQA support.

    Args:
        a_batch: Batched tensor [batch_size, ...]
        b_batch: Batched tensor [batch_size, ...] or single tensor for broadcasting
        scale_a_factor: Scale factor for a
        scale_b_factor: Scale factor for b
        handle_mqa_gqa: Whether to handle Multi-Query/Grouped-Query Attention head repetition
    """
    # a: [batch_heads_q, seq_q, dim], b: [batch_heads_kv, dim, seq_k]
    batch_heads_a = a_batch.shape[0]
    batch_heads_b = b_batch.shape[0]

    # Handle MQA/GQA head repetition if needed
    if handle_mqa_gqa and (batch_heads_a != batch_heads_b):
        # Repeat k/v heads to match q heads (from fp8_matmul logic)
        assert (
            batch_heads_a % batch_heads_b == 0
        ), f"q_heads ({batch_heads_a}) must be divisible by kv_heads ({batch_heads_b})"
        repeat_factor = batch_heads_a // batch_heads_b
        # Repeat each kv head to match q heads
        b_batch = b_batch.repeat_interleave(repeat_factor, dim=0)

    # Batched processing loop
    results = []
    for i in range(a_batch.shape[0]):
        a_2d = a_batch[i]
        b_2d = b_batch[i] if b_batch.dim() > 2 else b_batch
        result = _scaled_mm_helper(a_2d, b_2d, scale_a_factor, scale_b_factor)
        results.append(result)

    return torch.stack(results, dim=0)


def _apply_sm_scale(tensor_batch, scale_factor):
    """
    Apply scale factor to a batched tensor using scaled_mm with identity matrix.

    Args:
        tensor_batch: Tensor of shape [batch_heads, seq_len, dim]
        scale_factor: Scale factor to apply

    Returns:
        Scaled tensor with same shape and dtype as input
    """
    # Create identity matrix once
    identity = torch.eye(
        tensor_batch.shape[-1], device=tensor_batch.device, dtype=tensor_batch.dtype
    ).unsqueeze(0)

    # Use batched helper
    scaled_result = _batched_scaled_mm(
        tensor_batch, identity, scale_a_factor=scale_factor, scale_b_factor=1.0
    )
    return scaled_result.to(tensor_batch.dtype)


def _fp8_matmul(a, b, scale_a_factor=1.0, scale_b_factor=1.0):
    """
    Helper function for scaled_mm with custom scaling support.
    """
    # torch._scaled_mm expects 2D matrices, so we need to handle batching
    # Also handle MQA/GQA where a and b may have different batch dimensions

    if a.dim() == 3 and b.dim() == 3:
        return _batched_scaled_mm(
            a, b, scale_a_factor, scale_b_factor, handle_mqa_gqa=True
        )
    else:
        # Fallback for 2D case
        return _scaled_mm_helper(a, b, scale_a_factor, scale_b_factor)


def attention_ref_fp8(
    q, k, v, causal=False, scale_p=False, reorder_ops=False, use_direct_softmax=False
):
    """
    FP8 attention reference implementation using torch._scaled_mm.

    Args:
        q: Query tensor of shape (batch_size, seqlen_q, nheads, head_dim)
        k: Key tensor of shape (batch_size, seqlen_k, nheads_k, head_dim)
        v: Value tensor of shape (batch_size, seqlen_k, nheads_k, head_dim)
        causal: Whether to apply causal masking
        scale_p: Whether to scale probabilities (unused in current implementation)
        reorder_ops: If True, apply sm_scale to k instead of q (following attention_ref pattern)
        use_direct_softmax: If True, use torch.softmax directly instead of manual exp/sum steps

    Returns:
        Output tensor of shape (batch_size, seqlen_q, nheads, head_dim)
    """
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    q_, k_, v_ = T(q), T(k), T(v)

    # S = QK using scaled_mm with scales
    sm_scale = 1.0 / math.sqrt(q.shape[-1])

    if reorder_ops:
        # Apply sm_scale to q
        q_scaled = _apply_sm_scale(q_, sm_scale)
        S = _fp8_matmul(q_scaled, k_.transpose(-1, -2))
    else:
        # Apply sm_scale to k (original behavior)
        k_scaled = _apply_sm_scale(k_, sm_scale)
        S = _fp8_matmul(q_, k_scaled.transpose(-1, -2))

    S = S.to(float)

    if use_direct_softmax:
        # Use torch.softmax directly
        # Results in different numerics and mismatched with cutlass implementation
        P = torch.softmax(S, dim=-1)
        row_sum = torch.ones_like(
            P.sum(dim=-1, keepdim=True)
        )  # softmax already normalized
    else:
        # Manual softmax computation with separate steps
        row_max = S.max(dim=-1, keepdim=True)[0]
        S_shifted = S - row_max
        P = torch.exp(S_shifted)
        assert torch.all(P > 0)
        assert torch.all(P <= 1.0)

        # Compute row_sum before converting to fp8 since sum is not supported on fp8
        row_sum = P.sum(dim=-1, keepdim=True)

    P = P.to(torch.float8_e4m3fn)

    # P @ V using scaled_mm with scales=1
    numerator = _fp8_matmul(P, v_)
    out = numerator / row_sum

    # Reshape back
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    out = out.permute((0, 2, 1, 3))
    return out.to(dtype=torch.bfloat16)

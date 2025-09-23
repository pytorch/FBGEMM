# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import logging
import unittest
from typing import Optional, Union

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import torch
from hypothesis import given, settings, strategies as st
from hypothesis.database import InMemoryExampleDatabase

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)


def perHeadAmax(xqkv: torch.Tensor, varseq_batch: torch.Tensor, B: int) -> torch.Tensor:
    """
    Calculate AMAX values from xqkv tensor per batch and head.

    Args:
        xqkv: [B_T, HH, D_H] interleaved Q, K, V data
        varseq_batch: [B_T] batch indices for each token
        B: batch size

    Returns:
        amax_values: [B, HH] AMAX values per batch and head
    """
    device = xqkv.device
    B_T, HH, D_H = xqkv.shape

    amax_values = torch.zeros(B, HH, device=device, dtype=torch.float32)

    for b in range(B):
        # Find tokens belonging to this batch
        batch_mask = varseq_batch == b
        if torch.any(batch_mask):
            batch_tokens = xqkv[batch_mask]  # [num_tokens_in_batch, HH, D_H]
            # Take max across both token and feature dimensions for each head
            batch_amax_per_head = torch.max(torch.abs(batch_tokens), dim=0)[
                0
            ]  # [HH, D_H]
            batch_amax_per_head = torch.max(batch_amax_per_head, dim=1)[0]  # [HH]
            amax_values[b] = batch_amax_per_head.float()

    return amax_values


def calc_scale(amax_value: float, multiplier: float = 8.0) -> float:
    """
    Calculate quantization scale following CUDA kernel recipe exactly:
    1. Use provided AMAX value (from xqkv_amax_head)
    2. val_ = amax * multiplier
    3. Clamp to 12000: val_ = fminf(val_, 12000)
    4. Compute scale: scale = fmaxf(val_ / FP8_E4M3_MAX, min_scaling_factor)

    Args:
        amax_value: Pre-computed AMAX value from xqkv_amax_head
        multiplier: Scaling multiplier (8.0 for Q, 64.0 for KV)

    Returns:
        Computed quantization scale
    """
    FP8_E4M3_MAX = 448.0
    min_scaling_factor = 1.0 / (FP8_E4M3_MAX * 512.0)

    val_ = amax_value * multiplier
    val_ = min(val_, 12000.0)
    scale = max(val_ / FP8_E4M3_MAX, min_scaling_factor)

    return scale


def quantize(data: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Quantize tensor data to FP8 following CUDA kernel recipe exactly:
    1. Convert to float32 first (like CUDA kernel does)
    2. Apply scaling: multiply by inv_scale (like CUDA kernel: val * inv_scale)
    3. Clamp to fp8 range before conversion (like CUDA kernel: fmaxf/fminf)
    4. Convert to fp8 (this matches the CUDA kernel's __nv_fp8_e4m3 conversion)

    Args:
        data: Input tensor data to quantize
        scale: Quantization scale value

    Returns:
        Quantized FP8 tensor
    """
    if scale <= 0:
        return torch.zeros_like(data, dtype=torch.float8_e4m3fn)

    # Convert to float32 first (like CUDA kernel does)
    data_f32 = data.to(torch.float32)
    inv_scale = 1.0 / scale

    # Apply scaling: multiply by inv_scale (like CUDA kernel: val * inv_scale)
    quantized_data = data_f32 * inv_scale

    # Clamp to fp8 range before conversion (like CUDA kernel: fmaxf/fminf)
    quantized_data = torch.clamp(quantized_data, -448.0, 448.0)

    # Convert to fp8 (this matches the CUDA kernel's __nv_fp8_e4m3 conversion)
    return quantized_data.to(torch.float8_e4m3fn)


def quantize_qkv_per_head_python_reference(
    xqkv_amax_row: torch.Tensor,
    xqkv: torch.Tensor,
    varseq_seqpos: torch.Tensor,
    varseq_batch: torch.Tensor,
    B: int,
    N_H: int,
    N_KVH: int = 1,
    precalc: Optional[torch.Tensor] = None,
    cache_K: Optional[torch.Tensor] = None,
    cache_V: Optional[torch.Tensor] = None,
) -> Union[
    tuple[torch.Tensor, torch.Tensor],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]:
    """
    Python reference implementation of quantize_qkv_per_head.

    This function mimics the CUDA kernel behavior:
    1. Extract xq from xqkv (first N_H heads)
    2. Quantize xq to fp8 format
    3. If precalc provided: Extract K,V and quantize them, write to cache for prefill tokens
    4. Return quantization scales per batch and KV head

    Args:
        xqkv_amax_row: [B, N_H + 2*N_KVH] AMAX values per head
        xqkv: [B_T, N_H + 2*N_KVH, D_H] interleaved Q, K, V data
        varseq_seqpos: [B_T] sequence positions
        varseq_batch: [B_T] batch indices for each token
        B: batch size
        N_H: number of query heads
        N_KVH: number of key-value heads (must be 1)
        precalc: [B_T] boolean tensor, False means need to calculate KV for this token (prefill)
        cache_K: [B, MAX_T, N_KVH, D_H] KV cache for K (written to if precalc)
        cache_V: [B, MAX_T, N_KVH, D_H] KV cache for V (written to if precalc)

    Returns:
        xq_scale: [B, N_KVH] quantization scales for Q
        xq_quantized: [B_T, N_H, D_H] quantized Q tensor
        If precalc provided:
            k_scale: [B, N_KVH] quantization scales for K
            v_scale: [B, N_KVH] quantization scales for V
            updated_cache_K/V: Updated KV caches
    """
    device = xqkv.device
    B_T, HH, D_H = xqkv.shape

    # Extract Q from xqkv (first N_H heads)
    xq = xqkv[:, :N_H, :]  # [B_T, N_H, D_H]

    # Initialize output tensors
    xq_scale = torch.zeros(B, N_KVH, dtype=torch.float32, device=device)
    xq_quantized = torch.zeros_like(xq, dtype=torch.float8_e4m3fn)

    # Initialize KV scale and cache outputs if precalc is provided
    k_scale = None
    v_scale = None
    updated_cache_K = None
    updated_cache_V = None
    xk = None
    xv = None

    if precalc is not None:
        k_scale = torch.zeros(B, N_KVH, dtype=torch.float32, device=device)
        v_scale = torch.zeros(B, N_KVH, dtype=torch.float32, device=device)
        if cache_K is not None:
            updated_cache_K = cache_K.clone()
        if cache_V is not None:
            updated_cache_V = cache_V.clone()

        # Extract K and V from xqkv for prefill
        xk = xqkv[:, N_H : N_H + N_KVH, :]  # [B_T, N_KVH, D_H]
        xv = xqkv[:, N_H + N_KVH : N_H + 2 * N_KVH, :]  # [B_T, N_KVH, D_H]

    # Process each batch
    for b in range(B):
        # Find tokens belonging to this batch
        batch_mask = varseq_batch == b
        if not torch.any(batch_mask):
            continue

        batch_tokens = xq[batch_mask]  # [num_tokens_in_batch, N_H, D_H]

        # For each KV head (should be 1 for now)
        for kvh in range(N_KVH):
            # For Q heads: CUDA kernel computes max across ALL Q head AMAXs for this batch
            # Match CUDA kernel behavior: val = fmaxf(val, xqkv_amax_head[b * HH + hh]) for all Q heads
            q_max_amax = 0.0
            for h in range(N_H):
                q_max_amax = max(q_max_amax, xqkv_amax_row[b, h].item())

            # Calculate Q scale using AMAX from xqkv_amax_row (matches CUDA kernel)
            scale = calc_scale(q_max_amax, multiplier=1.0)

            xq_scale[b, kvh] = scale

            # Quantize batch tokens to fp8 using reusable function
            xq_quantized[batch_mask] = quantize(batch_tokens, scale)

            # Handle KV quantization for prefill case - match CUDA kernel behavior exactly
            if (
                precalc is not None
                and xk is not None
                and xv is not None
                and k_scale is not None
                and v_scale is not None
            ):
                # Check if any token in this batch needs KV calculation (precalc=False)
                batch_tokens_list = torch.where(batch_mask)[
                    0
                ]  # Get token indices for this batch

                # Skip if all tokens in this batch are decode tokens (precalc=True)
                if torch.all(precalc[batch_tokens_list]):
                    continue

                # For K and V: CUDA kernel uses individual head AMAX directly
                # K head is at index N_H + kvh, V head is at index N_H + N_KVH + kvh
                k_amax = xqkv_amax_row[b, N_H + kvh].item()  # K head AMAX
                v_amax = xqkv_amax_row[b, N_H + N_KVH + kvh].item()  # V head AMAX

                # Calculate K and V scales using AMAX from xqkv_amax_row (matches CUDA kernel)
                k_scale_val = calc_scale(k_amax, multiplier=64.0)
                v_scale_val = calc_scale(v_amax, multiplier=64.0)

                k_data = xk[batch_mask, kvh, :]  # [num_tokens_in_batch, D_H]
                v_data = xv[batch_mask, kvh, :]  # [num_tokens_in_batch, D_H]

                # Store scales (for verification)
                k_scale[b, kvh] = k_scale_val
                v_scale[b, kvh] = v_scale_val

                # Process each token individually for KV cache writing
                for token_idx, global_token_idx in enumerate(batch_tokens_list):
                    # Only write to cache for prefill tokens (precalc=False)
                    if not precalc[global_token_idx].item():
                        seqpos_t = varseq_seqpos[global_token_idx].item()

                        # Get individual token data
                        k_token_data = k_data[token_idx, :]  # [D_H]
                        v_token_data = v_data[token_idx, :]  # [D_H]

                        # Quantize and write to cache using reusable function
                        if updated_cache_K is not None and updated_cache_V is not None:
                            updated_cache_K[b, int(seqpos_t), kvh, :] = quantize(
                                k_token_data, k_scale_val
                            )
                            updated_cache_V[b, int(seqpos_t), kvh, :] = quantize(
                                v_token_data, v_scale_val
                            )

    if precalc is None:
        return xq_scale, xq_quantized
    else:
        # Ensure non-optional types for return
        assert k_scale is not None
        assert v_scale is not None
        assert updated_cache_K is not None
        assert updated_cache_V is not None
        return (
            xq_scale,
            xq_quantized,
            k_scale,
            v_scale,
            updated_cache_K,
            updated_cache_V,
        )


def create_test_tensors(
    B: int,
    B_T: int,
    N_H: int,
    N_KVH: int,
    D_H: int,
    MAX_T: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Helper function to create test tensors with proper shapes and dtypes."""

    HH = N_H + 2 * N_KVH

    # Generate B numbers whose sum is B_T
    # Start with equal distribution, then add remaining tokens randomly
    base_tokens_per_batch = B_T // B
    remaining_tokens = B_T % B

    tokens_per_batch = [base_tokens_per_batch] * B
    # Distribute remaining tokens randomly among batches
    for _ in range(remaining_tokens):
        batch_idx = int(torch.randint(0, B, (1,)).item())
        tokens_per_batch[batch_idx] += 1

    # Create varseq_batch as contiguous blocks: [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, ...]
    varseq_batch = []
    for batch_idx in range(B):
        varseq_batch.extend([batch_idx] * tokens_per_batch[batch_idx])
    varseq_batch = torch.tensor(varseq_batch, device=device, dtype=torch.int32)

    # Create xqkv tensor with interleaved Q, K, V data
    # Use realistic range [-10, 10]
    xqkv = (
        torch.rand(B_T, HH, D_H, device=device, dtype=torch.bfloat16) * 20.0 - 10.0
    )  # Range [-10, 10]

    # Compute xqkv_amax_row from actual xqkv data using perHeadAmax function
    # xqkv_amax works by grouping all tokens of the same batch together
    # and taking the max of each head in this batch resulting in [B, H] tensor
    xqkv_amax_row = perHeadAmax(xqkv, varseq_batch, B)

    # Create sequence position tensor: [b0_start, b0_start + 1, ..., b1_start, ...]
    # For now, choose start pos as 0 for all batch lanes
    varseq_seqpos = []
    for batch_idx in range(B):
        batch_start_pos = 0  # Start position is 0 for all batches for now
        num_tokens = tokens_per_batch[batch_idx]
        # Generate sequence positions for this batch: [start, start+1, start+2, ...]
        batch_seqpos = list(range(batch_start_pos, batch_start_pos + num_tokens))
        varseq_seqpos.extend(batch_seqpos)
    varseq_seqpos = torch.tensor(varseq_seqpos, device=device, dtype=torch.int32)

    # Create cache tensors for K and V
    cache_K = torch.zeros(
        B, MAX_T, N_KVH, D_H, device=device, dtype=torch.float8_e4m3fn
    )
    cache_V = torch.zeros(
        B, MAX_T, N_KVH, D_H, device=device, dtype=torch.float8_e4m3fn
    )

    # Create output tensor for Q
    XQ_O = torch.zeros(B_T, N_H, D_H, device=device, dtype=torch.float8_e4m3fn)

    return xqkv_amax_row, xqkv, varseq_seqpos, cache_K, cache_V, XQ_O, varseq_batch


class QuantizeQKVPerHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        device = torch.accelerator.current_accelerator()
        assert device is not None
        cls.device = device

        # Check if FP8 is supported
        try:
            torch.zeros(1, device=cls.device, dtype=torch.float8_e4m3fn)
            cls.fp8_supported = True
        except Exception:
            cls.fp8_supported = False

    def _test_against_reference(
        self,
        B: int,
        B_T: int,
        N_H: int,
        N_KVH: int,
        D_H: int,
        MAX_T: int,
        precalc_pattern: Optional[torch.Tensor] = None,
    ) -> None:
        """Helper method to test CUDA function against Python reference."""
        # Create test tensors
        (xqkv_amax_row, xqkv, varseq_seqpos, cache_K, cache_V, XQ_O, varseq_batch) = (
            create_test_tensors(B, B_T, N_H, N_KVH, D_H, MAX_T, self.device)
        )

        # Create precalc tensor from pattern if provided
        precalc = None
        if precalc_pattern is not None:
            precalc = precalc_pattern[varseq_batch]

        # Prepare KV scale tensors for prefill case
        k_scale = None
        v_scale = None
        if precalc is not None:
            k_scale = torch.zeros(B, N_KVH, dtype=torch.float32, device=self.device)
            v_scale = torch.zeros(B, N_KVH, dtype=torch.float32, device=self.device)

        # Get Python reference result
        if precalc is None:
            # Decode case
            ref_q_scale, ref_q_quantized = quantize_qkv_per_head_python_reference(
                xqkv_amax_row, xqkv, varseq_seqpos, varseq_batch, B, N_H, N_KVH
            )
            ref_k_scale = None
            ref_v_scale = None
            ref_cache_K = None
            ref_cache_V = None
        else:
            # Prefill case
            (
                ref_q_scale,
                ref_q_quantized,
                ref_k_scale,
                ref_v_scale,
                ref_cache_K,
                ref_cache_V,
            ) = quantize_qkv_per_head_python_reference(
                xqkv_amax_row,
                xqkv,
                varseq_seqpos,
                varseq_batch,
                B,
                N_H,
                N_KVH,
                precalc,
                cache_K.clone(),
                cache_V.clone(),
            )

        # Call the actual CUDA function
        if precalc is None:
            cuda_q_scale = torch.ops.fbgemm.quantize_qkv_per_head(
                xqkv_amax_row,
                xqkv,
                varseq_seqpos,
                varseq_batch,
                None,
                cache_K,
                cache_V,
                XQ_O,
                B,
            )
        else:
            cuda_q_scale = torch.ops.fbgemm.quantize_qkv_per_head(
                xqkv_amax_row,
                xqkv,
                varseq_seqpos,
                varseq_batch,
                precalc,
                cache_K,
                cache_V,
                XQ_O,
                B,
                qparam_k=k_scale,
                qparam_v=v_scale,
            )

        # Verify Q quantization results match
        self.assertEqual(cuda_q_scale.shape, ref_q_scale.shape)
        self.assertEqual(XQ_O.shape, ref_q_quantized.shape)
        torch.testing.assert_close(cuda_q_scale, ref_q_scale, rtol=1e-6, atol=1e-8)
        # Use exact matching for FP8 tensors
        self.assertTrue(torch.equal(ref_q_quantized, XQ_O))

        # Verify KV results for prefill case
        if precalc is not None:
            assert ref_k_scale is not None
            assert ref_v_scale is not None
            assert ref_cache_K is not None
            assert ref_cache_V is not None
            assert k_scale is not None
            assert v_scale is not None

            torch.testing.assert_close(k_scale, ref_k_scale, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(v_scale, ref_v_scale, rtol=1e-6, atol=1e-8)
            # Use exact matching for FP8 tensors
            self.assertTrue(torch.equal(cache_K, ref_cache_K))
            self.assertTrue(torch.equal(cache_V, ref_cache_V))

    @unittest.skipIf(
        torch.version.hip is not None
        or not torch.cuda.is_available()
        or torch.cuda.device_count() == 0
        or torch.cuda.get_device_properties(0).major < 9,
        "CUDA is not available or no GPUs detected",
    )
    @settings(deadline=None, database=InMemoryExampleDatabase())
    @given(
        B=st.sampled_from([1, 2, 3]),
        B_T=st.sampled_from([4, 8, 16]),
        N_H=st.sampled_from([4, 8]),
        N_KVH=st.sampled_from([1]),
        D_H=st.sampled_from([128]),
        MAX_T=st.sampled_from([512, 1024, 2048]),
    )
    def test_quantize_qkv_per_head_q_only(
        self, B: int, B_T: int, N_H: int, N_KVH: int, D_H: int, MAX_T: int
    ) -> None:
        """Test quantize_qkv_per_head against Python reference implementation for decode case."""
        if not self.fp8_supported:
            self.skipTest("FP8 not supported on this device")

        self._test_against_reference(
            B, B_T, N_H, N_KVH, D_H, MAX_T, precalc_pattern=None
        )

    @unittest.skipIf(
        torch.version.hip is not None
        or not torch.cuda.is_available()
        or torch.cuda.device_count() == 0
        or torch.cuda.get_device_properties(0).major < 9,
        "CUDA is not available or no GPUs detected",
    )
    @settings(deadline=None, database=InMemoryExampleDatabase())
    @given(
        test_case=st.sampled_from(
            [
                ([False, True, False], "mixed_prefill_decode"),
                ([False, False, False], "all_prefill"),
                ([True, True, True], "all_decode"),
            ]
        )
    )
    def test_quantize_qkv_per_head_kv_cache_write(
        self, test_case: tuple[list[bool], str]
    ) -> None:
        """Test prefill case where KV cache is quantized and written for precalc=False tokens."""
        if not self.fp8_supported:
            self.skipTest("FP8 not supported on this device")

        precalc_pattern, test_name = test_case

        # Create precalc pattern tensor
        # False means this token needs processing (prefill) - will quantize and write K, V to cache
        # True means this token is already processed (decode) - will quantize Q only
        precalc_pattern_tensor = torch.tensor(
            precalc_pattern, device=self.device, dtype=torch.bool
        )

        # Test with precalc pattern parameter
        self._test_against_reference(
            3, 128, 8, 1, 128, 512, precalc_pattern=precalc_pattern_tensor
        )


if __name__ == "__main__":
    unittest.main()

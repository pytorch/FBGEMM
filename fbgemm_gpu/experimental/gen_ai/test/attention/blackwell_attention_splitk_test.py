# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
import unittest
from typing import Optional

import torch
from fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha import (
    cutlass_blackwell_fmha_decode_forward,
)
from parameterized import parameterized

from .test_utils import attention_ref


DEBUG = False
SEED = 42

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")

skip_cuda_lt_sm100 = unittest.skipIf(
    compute_capability < (10, 0), "Only support sm100+"
)
skip_rocm = unittest.skipIf(torch.version.hip is not None, "Does not support ROCm")


@skip_cuda_lt_sm100
@skip_rocm
class SplitKTest(unittest.TestCase):
    """Test suite for SplitK attention implementation."""

    def _abs_max(self, t: torch.Tensor) -> float:
        """Compute the maximum absolute value in a tensor."""
        return t.abs().max().item()

    def _generate_qkv(
        self,
        batch_size: int,
        seqlen_q: int,
        seqlen_k: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random Q, K, V tensors for testing."""
        q = torch.randn(
            batch_size,
            seqlen_q,
            q_heads,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        k = torch.randn(
            batch_size,
            seqlen_k,
            kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        v = torch.randn(
            batch_size,
            seqlen_k,
            kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        return q, k, v

    def _reference_splitk_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_splits: int,
        split_size: int,
        sm_scale: Optional[float] = None,
        window_size: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reference implementation for SplitK attention.

        This reshapes the K/V sequence dimension into [num_splits, split_size] and
        processes each split independently, then combines the results.

        Args:
            q: Query tensor [B, Sq, H, D]
            k: Key tensor [B, Sk, H_kv, D]
            v: Value tensor [B, Sk, H_kv, D]
            num_splits: Number of splits along the sequence dimension
            split_size: Size of each split
            sm_scale: Softmax scale (defaults to 1/sqrt(D))

        Returns:
            out_ref: Output tensor [B, H, num_splits, D]
            lse_ref: LSE tensor [B, num_splits, H]
        """
        B, Sq, _, D = q.shape
        _, Sk, H_kv, _ = k.shape

        # Apply sliding window: only use last window_size tokens
        if window_size > 0 and window_size < Sk:
            # Slice K and V to only include the last window_size tokens
            k = k[:, -window_size:, :, :]
            v = v[:, -window_size:, :, :]
            Sk = window_size

        # Pad K and V to make them divisible by split_size
        padded_Sk = num_splits * split_size
        if Sk < padded_Sk:
            pad_size = padded_Sk - Sk
            k_padded = torch.cat(
                [k, torch.zeros(B, pad_size, H_kv, D, dtype=k.dtype, device=k.device)],
                dim=1,
            )
            v_padded = torch.cat(
                [v, torch.zeros(B, pad_size, H_kv, D, dtype=v.dtype, device=v.device)],
                dim=1,
            )
        else:
            k_padded = k[:, :padded_Sk]
            v_padded = v[:, :padded_Sk]

        # Reshape K and V: [B, padded_Sk, H_kv, D] -> [B, num_splits, split_size, H_kv, D]
        k_reshaped = k_padded.view(B, num_splits, split_size, H_kv, D)
        v_reshaped = v_padded.view(B, num_splits, split_size, H_kv, D)

        # For decode case (Sq == 1), we process each split independently
        # Output will be [B, H, num_splits, D]
        # LSE will be [B, num_splits, H]
        output_splits = []
        lse_splits = []

        for split_idx in range(num_splits):
            # Get K and V for this split: [B, split_size, H_kv, D]
            k_split = k_reshaped[:, split_idx, :, :, :]
            v_split = v_reshaped[:, split_idx, :, :, :]

            # Calculate valid length for this split
            start_idx = split_idx * split_size
            end_idx = min((split_idx + 1) * split_size, Sk)
            valid_len = end_idx - start_idx

            # Create padding mask for this split if needed
            if valid_len < split_size:
                # Create mask: True for valid positions, False for padding
                key_padding_mask = torch.zeros(
                    B, split_size, dtype=torch.bool, device=q.device
                )
                key_padding_mask[:, :valid_len] = True
            else:
                key_padding_mask = None

            # Run attention for this split
            # q: [B, Sq, H, D]
            # k_split: [B, split_size, H_kv, D]
            # v_split: [B, split_size, H_kv, D]
            out_split, _, lse_split = attention_ref(
                q,
                k_split,
                v_split,
                key_padding_mask=key_padding_mask,
                causal=True,  # Decode case is causal
                upcast=True,
                softmax_scale=sm_scale,
                return_lse=True,
            )

            # out_split is [B, Sq, H, D]
            # lse_split is [B, H, Sq]
            # For decode case (Sq == 1), squeeze to [B, H, D] and [B, H]
            if Sq == 1:
                out_split = out_split.squeeze(1)  # [B, H, D]
                lse_split = lse_split.squeeze(-1)  # [B, H]

            output_splits.append(out_split)
            lse_splits.append(lse_split)

        # Stack outputs from all splits
        # Each out_split is [B, H, D]
        # Stack to get [B, H, num_splits, D]
        out_ref = torch.stack(output_splits, dim=2)  # [B, H, num_splits, D]

        # Stack LSE from all splits
        # Each lse_split is [B, H]
        # Stack to get [B, num_splits, H]
        lse_ref = torch.stack(lse_splits, dim=1)  # [B, num_splits, H]

        return out_ref, lse_ref

    def _merge_attentions_ref(
        self,
        attn_split: torch.Tensor,
        lse_split: torch.Tensor,
        debug: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Python reference implementation for merging attention outputs.

        This implements the standard log-sum-exp based reduction to combine
        partial attention outputs from split-K computation.

        Reference: xformers/tests/test_fmha_merge_attentions.py

        Args:
            attn_split: Attention outputs [split_k, B, M, (G,) H, Kq]
            lse_split: LSE values [split_k, B, (G,) H, M]
            debug: If True, print intermediate values

        Returns:
            attn_out: Combined attention output [B, M, (G,) H, Kq]
            lse_out: Combined LSE [B, (G,) H, M]
        """
        is_bmghk = len(attn_split.shape) == 6
        if not is_bmghk:
            attn_split = attn_split.unsqueeze(3)
            lse_split = lse_split.unsqueeze(2)

        if debug:
            print(
                f"    [merge] After unsqueeze - attn_split: {attn_split.shape}, lse_split: {lse_split.shape}"
            )

        # Move LSE to match attention shape for broadcasting
        # lse_split: [split_k, B, G, H, M] -> [split_k, B, M, G, H, 1]
        lse_split = lse_split[..., None].moveaxis(4, 2)

        if debug:
            print(f"    [merge] After moveaxis - lse_split: {lse_split.shape}")
            print(f"    [merge] lse_split values: {lse_split[:, 0, 0, 0, 0, 0]}")

        # Compute max LSE across splits for numerical stability
        lse_max, _ = torch.max(lse_split, dim=0)  # [B, M, G, H, 1]

        if debug:
            print(f"    [merge] lse_max: {lse_max[0, 0, 0, 0, 0]}")

        # Compute normalized sum of exponentials
        sumexp_normalized = torch.exp(lse_split - lse_max)  # [split_k, B, M, G, H, 1]
        denominator = sumexp_normalized.sum(dim=0)  # [B, M, G, H, 1]

        if debug:
            print(
                f"    [merge] sumexp_normalized per split: {sumexp_normalized[:, 0, 0, 0, 0, 0]}"
            )
            print(f"    [merge] denominator: {denominator[0, 0, 0, 0, 0]}")
            # Print weights (what fraction each split contributes)
            weights = sumexp_normalized / denominator
            print(f"    [merge] weights per split: {weights[:, 0, 0, 0, 0, 0]}")

        # Weighted sum of attention outputs
        numerator = (sumexp_normalized * attn_split).sum(dim=0)  # [B, M, G, H, K]

        # Final output
        attn_out = numerator / denominator  # [B, M, G, H, Kq]
        lse_out = lse_max + torch.log(denominator)
        lse_out = lse_out.squeeze(4).permute(0, 2, 3, 1)  # [B, G, H, M]

        if not is_bmghk:
            attn_out = attn_out.squeeze(2)  # [B, M, H, Kq]
            lse_out = lse_out.squeeze(1)  # [B, H, M]

        return attn_out, lse_out

    def _run_splitk_correctness_check(
        self,
        batch_size: int,
        seqlen_k: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        seqlen_kv_tensor: Optional[torch.Tensor] = None,
        tolerance: float = 1e-3,
        window_size: int = -1,
    ) -> None:
        """
        Core split-K correctness check comparing kernel output against reference.

        This helper runs the split-K kernel and compares its output against a
        pure Python reference implementation.

        Args:
            batch_size: Batch size
            seqlen_k: Maximum sequence length for K/V
            q_heads: Number of query heads
            kv_heads: Number of key/value heads
            head_dim: Head dimension
            seqlen_kv_tensor: Optional per-batch sequence lengths for varlen support.
                              If None, uses seqlen_k for all batches.
            tolerance: Maximum allowed difference between kernel and reference outputs
        """
        device = torch.accelerator.current_accelerator()
        assert device is not None

        torch.manual_seed(SEED)

        seqlen_q = 1  # Decode case
        dtype = torch.bfloat16

        # Generate random Q, K, V
        q, k, v = self._generate_qkv(
            batch_size, seqlen_q, seqlen_k, q_heads, kv_heads, head_dim, device, dtype
        )

        # Calculate split parameters
        SPLIT_SIZE = 1024

        # For sliding window, effective seqlen is min(window_size, seqlen_k)
        effective_seqlen = seqlen_k
        if window_size > 0 and window_size < seqlen_k:
            effective_seqlen = window_size
        num_splits = (effective_seqlen + SPLIT_SIZE - 1) // SPLIT_SIZE

        # Use provided seqlen_kv or create uniform tensor
        if seqlen_kv_tensor is None:
            seqlen_kv = torch.full(
                (batch_size,),
                seqlen_k,
                dtype=torch.int32,
                device=device,
            )
        else:
            seqlen_kv = seqlen_kv_tensor

        if DEBUG:
            print("\nTest parameters:")
            print(f"  Batch size: {batch_size}")
            print(f"  Max sequence length (K): {seqlen_k}")
            print(f"  Window size: {window_size}")
            print(f"  Effective seqlen: {effective_seqlen}")
            print(f"  Q heads: {q_heads}, KV heads: {kv_heads}")
            print(f"  Head dim: {head_dim}")
            print(f"  Num splits: {num_splits}, Split size: {SPLIT_SIZE}")
            print(f"  seqlen_kv: {seqlen_kv.tolist()}")

        # ========================================
        # Run splitk CUTLASS kernel
        # ========================================
        sm_scale = 1.0 / math.sqrt(head_dim)

        # Convert window_size to window_left/window_right format
        window_left = window_size
        window_right = 0 if window_size > 0 else -1

        out_test, lse_test = cutlass_blackwell_fmha_decode_forward(
            q,
            k,
            v,
            seqlen_kv=seqlen_kv,
            split_k_size=SPLIT_SIZE,
            window_left=window_left,
            window_right=window_right,
        )

        # ========================================
        # Run _reference_splitk_attention (pure Python reference)
        # ========================================
        out_ref, lse_ref = self._reference_splitk_attention(
            q, k, v, num_splits, SPLIT_SIZE, sm_scale, window_size
        )
        # out_ref: [B, H, num_splits, D]
        # lse_ref: [B, num_splits, H]

        if DEBUG:
            print("\nOutput shapes:")
            print(f"  Test output: {out_test.shape}")
            print(f"  Reference output: {out_ref.shape}")
            if lse_test is not None:
                print(f"  LSE test: {lse_test.shape}")
            print(f"  LSE ref: {lse_ref.shape}")
            print(f" Dtypes: {out_test.dtype=}, {out_ref.dtype=}")
            print(f" Dtypes: {lse_test.dtype=}, {lse_ref.dtype=}")

        # Verify output layout: [B, 1, H, num_splits, D]
        expected_shape = (batch_size, 1, q_heads, num_splits, head_dim)
        self.assertEqual(
            out_test.shape,
            expected_shape,
            f"Output shape mismatch: expected {expected_shape}, got {out_test.shape}",
        )
        # Squeeze Q dimension for comparison with reference
        out_test = out_test.squeeze(1)  # [B, H, num_splits, D]

        # Compare outputs
        max_diff = self._abs_max(out_test - out_ref)

        if DEBUG:
            print("\nNumerical comparison:")
            print(f"  Max absolute difference: {max_diff}")
            print("\n  Test output splits (first batch, first head, first 5 dims):")
            for split_idx in range(num_splits):
                print(f"    Split {split_idx}: {out_test[0, 0, split_idx, :5]}")
            print(
                "\n  Reference output splits (first batch, first head, first 5 dims):"
            )
            for split_idx in range(num_splits):
                print(f"    Split {split_idx}: {out_ref[0, 0, split_idx, :5]}")

            # Check if splits are duplicated (kernel bug indicator)
            if num_splits > 1:
                splits_equal = torch.allclose(
                    out_test[0, 0, 0, :], out_test[0, 0, 1, :], rtol=1e-2
                )
                if splits_equal:
                    print("\n  ⚠️  WARNING: Splits 0 and 1 are nearly identical!")
                    print(
                        "      This indicates the kernel is NOT processing different K/V chunks per split."
                    )
                    print(
                        "      Check split_k_idx usage in the kernel's K/V pointer calculations."
                    )

        # Verify no NaN or Inf values
        self.assertFalse(torch.isnan(out_test).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(out_test).any(), "Output contains Inf values")

        # Allow some tolerance due to numerical precision differences
        self.assertLessEqual(
            max_diff,
            tolerance,
            f"Output difference too large: {max_diff} > {tolerance}",
        )

        if DEBUG:
            print(f"✓ Test passed with max difference: {max_diff}")

    def _run_splitk_merged_vs_full_attention_check(
        self,
        batch_size: int,
        seqlen_k: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        seqlen_kv_tensor: Optional[torch.Tensor] = None,
        output_tolerance: float = 1e-2,
        lse_tolerance: float = 1e-3,
        window_size: int = -1,
    ) -> None:
        """
        Core test comparing split-K merged output against full (non-split) attention.

        This helper validates end-to-end split-K correctness by:
        1. Running full attention with split_k_size=0 (reference)
        2. Running split-K attention with split_k_size=SPLIT_SIZE
        3. Merging the split-K partial outputs using log-sum-exp reduction
        4. Verifying the merged output matches the full attention output

        Supports varlen through the seqlen_kv_tensor parameter.

        Args:
            batch_size: Batch size
            seqlen_k: Maximum sequence length for K/V
            q_heads: Number of query heads
            kv_heads: Number of key/value heads
            head_dim: Head dimension
            seqlen_kv_tensor: Optional per-batch sequence lengths for varlen support.
                              If None, uses seqlen_k for all batches.
            output_tolerance: Maximum allowed difference for output comparison
            lse_tolerance: Maximum allowed difference for LSE comparison
            window_size: Sliding window size (-1 for disabled)
        """
        device = torch.accelerator.current_accelerator()
        assert device is not None

        torch.manual_seed(SEED)

        seqlen_q = 1  # Decode case
        dtype = torch.bfloat16

        # Generate random Q, K, V
        q, k, v = self._generate_qkv(
            batch_size, seqlen_q, seqlen_k, q_heads, kv_heads, head_dim, device, dtype
        )

        # Calculate split parameters
        SPLIT_SIZE = 1024

        # For sliding window, effective seqlen is min(window_size, seqlen_k)
        effective_seqlen = seqlen_k
        if window_size > 0 and window_size < seqlen_k:
            effective_seqlen = window_size
        num_splits = (effective_seqlen + SPLIT_SIZE - 1) // SPLIT_SIZE

        # Use provided seqlen_kv or create uniform tensor
        if seqlen_kv_tensor is None:
            seqlen_kv = torch.full(
                (batch_size,),
                seqlen_k,
                dtype=torch.int32,
                device=device,
            )
        else:
            seqlen_kv = seqlen_kv_tensor

        # Convert window_size to window_left/window_right format
        window_left = window_size
        window_right = 0 if window_size > 0 else -1

        if DEBUG:
            print("\nSplit-K merged vs full attention test:")
            print(f"  Batch size: {batch_size}")
            print(f"  Max sequence length (K): {seqlen_k}")
            print(f"  Window size: {window_size}")
            print(f"  Effective seqlen: {effective_seqlen}")
            print(f"  Q heads: {q_heads}, KV heads: {kv_heads}")
            print(f"  Head dim: {head_dim}")
            print(f"  Num splits: {num_splits}, Split size: {SPLIT_SIZE}")
            print(f"  seqlen_kv: {seqlen_kv.tolist()}")

        # ========================================
        # Run full attention with split_k_size=0
        # ========================================
        out_full, lse_full = cutlass_blackwell_fmha_decode_forward(
            q,
            k,
            v,
            seqlen_kv=seqlen_kv,
            split_k_size=0,  # No split-K
            use_heuristic=False,  # Disable heuristic to truly disable split-K
            window_left=window_left,
            window_right=window_right,
        )
        # out_full: [B, 1, H, D]
        # lse_full: [B, H, 1]

        if DEBUG:
            print(f"\n  Full attention output shape: {out_full.shape}")
            print(f"  Full attention LSE shape: {lse_full.shape}")

        # ========================================
        # Run split-K attention (using heuristic to compute split size)
        # ========================================
        out_split, lse_split = cutlass_blackwell_fmha_decode_forward(
            q.clone(),
            k.clone(),
            v.clone(),
            seqlen_kv=seqlen_kv,
            # Let heuristic compute optimal split size (use_heuristic=True by default)
            split_k_size=SPLIT_SIZE,
            window_left=window_left,
            window_right=window_right,
        )
        # out_split: [B, 1, H, num_splits, D]
        # lse_split: [B, num_splits, H, 1]
        num_splits = out_split.shape[3]  # Get actual num_splits from output

        # Check for NaN values and print which batch/split indices have them
        if torch.isnan(out_split).any():
            nan_mask = torch.isnan(out_split)
            for batch_idx in range(batch_size):
                for split_idx in range(num_splits):
                    if nan_mask[batch_idx, :, :, split_idx, :].any():
                        print(
                            f"  ⚠️  NaN detected in out_split: batchIdx={batch_idx}, splitIdx={split_idx}"
                        )
                        torch.set_printoptions(threshold=float("inf"))
                        print(out_split[batch_idx, :, :, split_idx, :])
                        # out_split[batch_idx, :, :, split_idx, :].zero_()

        # Verify no NaN or Inf values in split output
        self.assertFalse(
            torch.isnan(out_split).any(), "Split output contains NaN values"
        )
        self.assertFalse(
            torch.isinf(out_split).any(), "Split output contains Inf values"
        )

        # ========================================
        # Merge split-K outputs
        # ========================================
        # Reshape for merge function:
        # out_split: [B, 1, H, num_splits, D] -> [num_splits, B, 1, H, D]
        # lse_split: [B, num_splits, H, 1] -> [num_splits, B, H, 1]
        out_chunks_stacked = out_split.permute(
            3, 0, 1, 2, 4
        )  # [num_splits, B, 1, H, D]
        lse_chunks_stacked = lse_split.permute(1, 0, 2, 3)  # [num_splits, B, H, 1]

        if DEBUG:
            print("\n  Reshaped for merge:")
            print(f"    out_chunks_stacked: {out_chunks_stacked.shape}")
            print(f"    lse_chunks_stacked: {lse_chunks_stacked.shape}")

        out_merged, lse_merged = self._merge_attentions_ref(
            out_chunks_stacked, lse_chunks_stacked, debug=DEBUG
        )
        # out_merged: [B, 1, H, D]
        # lse_merged: [B, H, 1]

        if DEBUG:
            print(f"\n  Final merged output shape: {out_merged.shape}")
            print(f"  Final merged LSE shape: {lse_merged.shape}")

        # ========================================
        # Compare outputs
        # ========================================
        # out_full has shape [B, 1, H, 1, D] - squeeze the num_splits dimension
        # out_merged has shape [B, 1, H, D] from merge function
        out_full_squeezed = out_full.squeeze(3)  # [B, 1, H, D]
        lse_full_squeezed = lse_full.squeeze(1)  # [B, H, 1]

        # Assert shapes are identical before comparison
        self.assertEqual(
            out_merged.shape,
            out_full_squeezed.shape,
            f"Shape mismatch: out_merged {out_merged.shape} vs out_full {out_full_squeezed.shape}",
        )
        self.assertEqual(
            lse_merged.shape,
            lse_full_squeezed.shape,
            f"Shape mismatch: lse_merged {lse_merged.shape} vs lse_full {lse_full_squeezed.shape}",
        )

        # Compare outputs
        out_merged_bf16 = out_merged.to(torch.bfloat16)
        out_diff = self._abs_max(out_merged_bf16.float() - out_full_squeezed.float())
        lse_diff = self._abs_max(lse_merged - lse_full_squeezed)

        if DEBUG:
            print("\nOverall comparison:")
            print(f"  Output max diff: {out_diff}")
            print(f"  LSE max diff: {lse_diff}")

        # Verify outputs match within tolerance
        self.assertLessEqual(
            out_diff,
            output_tolerance,
            f"Merged output differs from full attention: {out_diff} > {output_tolerance}",
        )
        self.assertLessEqual(
            lse_diff,
            lse_tolerance,
            f"Merged LSE differs from full attention: {lse_diff} > {lse_tolerance}",
        )

        if DEBUG:
            print("✓ Split-K merged matches full attention")

    @parameterized.expand(
        [
            (batch_size, seqlen_k, q_heads, kv_heads, head_dim)
            for batch_size in [1, 2, 4]
            for seqlen_k in [1024, 2048, 3072, 4096]  # Test various sequence lengths
            for q_heads in [8, 16]
            for kv_heads in [1, 2]  # Test MQA and GQA
            for head_dim in [128]
        ]
    )
    def test_splitk_decode(
        self,
        batch_size: int,
        seqlen_k: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        """
        Test SplitK attention for decode case (seqlen_q == 1).

        This test validates that the SplitK implementation produces correct results
        by comparing against a reference implementation that reshapes the sequence
        dimension into splits.
        """
        self._run_splitk_correctness_check(
            batch_size, seqlen_k, q_heads, kv_heads, head_dim
        )

    def test_splitk_output_layout(self) -> None:
        """
        Test that SplitK attention produces outputs in the correct layout.

        Expected layouts:
        - Output: [B, H, num_splits, D]
        - LSE: [B, num_splits, H]
        """
        device = torch.accelerator.current_accelerator()
        assert device is not None

        torch.manual_seed(SEED)

        # Test parameters
        batch_size = 2
        seqlen_q = 1
        seqlen_k = 2048
        q_heads = 8
        kv_heads = 1
        head_dim = 128
        dtype = torch.bfloat16

        # Generate Q, K, V
        q, k, v = self._generate_qkv(
            batch_size, seqlen_q, seqlen_k, q_heads, kv_heads, head_dim, device, dtype
        )

        # Calculate expected number of splits
        SPLIT_SIZE = 1024
        num_splits = (seqlen_k + SPLIT_SIZE - 1) // SPLIT_SIZE

        # Prepare seqlen_kv
        seqlen_kv = torch.full(
            (batch_size,),
            seqlen_k,
            dtype=torch.int32,
            device=device,
        )

        # Run SplitK attention
        out, lse = cutlass_blackwell_fmha_decode_forward(
            q,
            k,
            v,
            seqlen_kv=seqlen_kv,
            split_k_size=SPLIT_SIZE,
        )

        # Verify output layout: [B, 1, H, num_splits, D]
        expected_out_shape = (batch_size, 1, q_heads, num_splits, head_dim)
        self.assertEqual(
            out.shape,
            expected_out_shape,
            f"Output shape should be [B={batch_size}, 1, H={q_heads}, num_splits={num_splits}, D={head_dim}]",
        )

        # Verify LSE layout: [B, num_splits, H, 1]
        if lse is not None:
            expected_lse_shape = (batch_size, num_splits, q_heads, 1)
            self.assertEqual(
                lse.shape,
                expected_lse_shape,
                f"LSE shape should be [B={batch_size}, num_splits={num_splits}, H={q_heads}, 1]",
            )

        if DEBUG:
            print("\n✓ Layout test passed:")
            print(f"  Output shape: {out.shape}")
            if lse is not None:
                print(f"  LSE shape: {lse.shape}")

    def test_non_splitk_output_layout(self) -> None:
        """
        Test that non-split-k attention produces outputs in the correct layout.

        When split_k_size <= 0, expected layouts:
        - Output: [B, 1, H, D] with bfloat16 dtype (same as input q)
        - LSE: [B, H, 1] with float32 dtype
        """
        device = torch.accelerator.current_accelerator()
        assert device is not None

        torch.manual_seed(SEED)

        # Test parameters
        batch_size = 2
        seqlen_q = 1
        seqlen_k = 2048
        q_heads = 8
        kv_heads = 1
        head_dim = 128
        dtype = torch.bfloat16

        # Generate Q, K, V
        q, k, v = self._generate_qkv(
            batch_size, seqlen_q, seqlen_k, q_heads, kv_heads, head_dim, device, dtype
        )

        # Prepare seqlen_kv
        seqlen_kv = torch.full(
            (batch_size,),
            seqlen_k,
            dtype=torch.int32,
            device=device,
        )

        # Run non-split attention (split_k_size = 0)
        out, lse = cutlass_blackwell_fmha_decode_forward(
            q,
            k,
            v,
            seqlen_kv=seqlen_kv,
            split_k_size=0,  # Disable split-k
            use_heuristic=False,  # Disable heuristic to truly disable split-K
        )

        # Verify output layout: [B, 1, H, 1, D] (with num_splits=1)
        expected_out_shape = (batch_size, 1, q_heads, 1, head_dim)
        self.assertEqual(
            out.shape,
            expected_out_shape,
            f"Non-split output shape should be [B={batch_size}, 1, H={q_heads}, 1, D={head_dim}]",
        )

        # Verify output dtype is bfloat16 (same as input for non-split case)
        self.assertEqual(
            out.dtype,
            torch.bfloat16,
            f"Non-split output dtype should be bfloat16, got {out.dtype}",
        )

        # Verify LSE layout: [B, 1, H, 1] (with num_splits=1)
        if lse is not None:
            expected_lse_shape = (batch_size, 1, q_heads, 1)
            self.assertEqual(
                lse.shape,
                expected_lse_shape,
                f"Non-split LSE shape should be [B={batch_size}, 1, H={q_heads}, 1]",
            )

            # Verify LSE dtype is float32
            self.assertEqual(
                lse.dtype,
                torch.float32,
                f"LSE dtype should be float32, got {lse.dtype}",
            )

        if DEBUG:
            print("\n✓ Non-split layout test passed:")
            print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
            if lse is not None:
                print(f"  LSE shape: {lse.shape}, dtype: {lse.dtype}")

    @parameterized.expand(
        [
            (batch_size, seqlen_k, q_heads, kv_heads, head_dim)
            for batch_size in [1]
            for seqlen_k in [1024, 2048]
            for q_heads in [1, 8]
            for kv_heads in [1]
            for head_dim in [128]
        ]
    )
    def test_splitk_merged_vs_full_attention(
        self,
        batch_size: int,
        seqlen_k: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        """
        Test that split-K outputs, when merged, match full (non-split) attention.

        This is an end-to-end correctness test for the split-K kernel that:
        1. Runs full attention with split_k_size=0 (reference)
        2. Runs split-K attention with split_k_size=SPLIT_SIZE
        3. Merges the split-K partial outputs using log-sum-exp reduction
        4. Verifies the merged output matches the full attention output

        This validates that the split-K kernel correctly computes partial
        attention outputs that can be combined to produce the correct final result.
        """
        self._run_splitk_merged_vs_full_attention_check(
            batch_size=batch_size,
            seqlen_k=seqlen_k,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )

    @parameterized.expand(
        [
            (batch_size, max_seqlen_k, varlen)
            for batch_size in [1, 2, 4]
            for max_seqlen_k in [
                512,
                1023,
                1024,
                1025,
                2047,
                2048,
                3000,
                4096,
            ]
            for varlen in [True]
        ]
    )
    def test_splitk_boundary_cases(
        self, batch_size: int, max_seqlen_k: int, varlen: bool
    ) -> None:
        """
        Test SplitK attention at boundary cases of split sizes with varlen support.

        This ensures correct behavior when:
        - Sequence length is exactly at or near split boundaries (e.g., 1024, 2048)
        - Different batch elements have different sequence lengths (varlen=True)

        Uses the merged vs full attention comparison which supports varlen.
        """
        device = torch.accelerator.current_accelerator()
        assert device is not None

        # Test parameters
        q_heads = 8
        kv_heads = 1
        head_dim = 128

        # Create seqlen_kv tensor
        if varlen:
            # Generate varying sequence lengths for each batch element
            # Range from half of max_seqlen_k to max_seqlen_k
            torch.manual_seed(SEED)
            min_seqlen = max(1, max_seqlen_k // 2)
            print(f"min_seqlen: {min_seqlen}")
            seqlen_kv = torch.randint(
                min_seqlen,
                max_seqlen_k + 1,
                (batch_size,),
                dtype=torch.int32,
                device=device,
            )
            # Ensure at least one element has the max length
            seqlen_kv[0] = max_seqlen_k
            # Test varlen with shorter sequence that creates empty splits
            # When seqlen_kv < max_seqlen_k, some splits will have no valid data
            if batch_size > 1:
                seqlen_kv[-1] = min_seqlen  # Use shorter sequence to test empty splits
            print(f"seqlen_kv: {seqlen_kv.tolist()}")
        else:
            seqlen_kv = torch.full(
                (batch_size,),
                max_seqlen_k,
                dtype=torch.int32,
                device=device,
            )

        if DEBUG:
            print("\nBoundary test:")
            print(
                f"  batch_size={batch_size}, max_seqlen_k={max_seqlen_k}, varlen={varlen}"
            )
            print(f"  seqlen_kv: {seqlen_kv.tolist()}")

        self._run_splitk_merged_vs_full_attention_check(
            batch_size=batch_size,
            seqlen_k=max_seqlen_k,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seqlen_kv_tensor=seqlen_kv,
        )

        if DEBUG:
            print(
                f"✓ Boundary test passed for max_seqlen_k={max_seqlen_k}, varlen={varlen}"
            )

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (batch_size, seqlen_k, window_size, varlen)
            for batch_size in [1, 2]
            for seqlen_k in [4096, 8192]
            for window_size in [
                # -1,  # Disabled (no windowing)
                # 512,  # Window < splitk_size (single split)
                1024,  # Window == splitk_size
                1025,  # Creates 2 splits with 1 token in split 1
                1536,  # Window between 1 and 2 splits
                2048,  # Window == 2 * splitk_size (2 full splits)
                # 8192,  # Window > seqlen_k (no effect)
            ]
            for varlen in [False, True]
        ]
    )
    def test_splitk_sliding_window(
        self,
        batch_size: int,
        seqlen_k: int,
        window_size: int,
        varlen: bool,
    ) -> None:
        """
        Test SplitK attention with sliding window.

        Tests scenarios from the implementation plan:
        - window_size <= 0: Disabled, use full sequence
        - window_size >= seqlen_k: Use full sequence, offset=0
        - window_size < splitk_size: Single split covers entire window
        - window_size spanning splits: Multiple splits, each with correct range
        - Varlen batches: Each batch has different offset
        """
        device = torch.accelerator.current_accelerator()
        assert device is not None

        q_heads = 8
        kv_heads = 1
        head_dim = 128

        # Create seqlen_kv tensor
        if varlen:
            torch.manual_seed(SEED)
            min_seqlen = max(1, seqlen_k // 2)
            seqlen_kv = torch.randint(
                min_seqlen,
                seqlen_k + 1,
                (batch_size,),
                dtype=torch.int32,
                device=device,
            )
            seqlen_kv[0] = seqlen_k  # At least one has max length
            print(f"seqlen_kv: {seqlen_kv.tolist()}")
        else:
            seqlen_kv = torch.full(
                (batch_size,),
                1200,
                dtype=torch.int32,
                device=device,
            )

        if DEBUG:
            print("\nSliding window + Split-K test:")
            print(f"  batch_size={batch_size}, seqlen_k={seqlen_k}")
            print(f"  window_size={window_size}, varlen={varlen}")
            print(f"  seqlen_kv: {seqlen_kv.tolist()}")

        self._run_splitk_merged_vs_full_attention_check(
            batch_size=batch_size,
            seqlen_k=seqlen_k,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seqlen_kv_tensor=seqlen_kv,
            window_size=window_size,
        )

        if DEBUG:
            print(
                f"✓ Sliding window + Split-K test passed for "
                f"seqlen_k={seqlen_k}, window_size={window_size}, varlen={varlen}"
            )


if __name__ == "__main__":
    unittest.main()

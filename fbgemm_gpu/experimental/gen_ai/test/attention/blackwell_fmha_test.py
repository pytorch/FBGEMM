# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import unittest
from typing import cast, Optional

import torch
from einops import rearrange

from fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha import (
    _cutlass_blackwell_fmha_forward,
    cutlass_blackwell_fmha_decode_forward,
    cutlass_blackwell_fmha_func,
)
from parameterized import parameterized

from .attention_ref_fp8 import attention_ref_fp8
from .test_utils import attention_ref, generate_qkv, generate_random_padding_mask


DEBUG = False
SEED = 2

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")

skip_cuda_lt_sm100 = unittest.skipIf(
    compute_capability < (10, 0), "Only support sm100+"
)
skip_rocm = unittest.skipIf(torch.version.hip is not None, "Does not support ROCm")


class CutlassBlackwellFMHATest(unittest.TestCase):
    def _abs_max(self, t: torch.Tensor):
        return t.abs().max().item()

    def _allclose(
        self,
        t_test: torch.Tensor,
        t_ref: torch.Tensor,
        t_pt: torch.Tensor,
    ) -> None:
        assert t_test.shape == t_ref.shape == t_pt.shape

        ratio = 2.0

        # Calculate all differences
        test_ref_diff = self._abs_max(t_test - t_ref)
        test_pt_diff = self._abs_max(t_test - t_pt)
        pt_ref_diff = self._abs_max(t_pt - t_ref)

        if DEBUG:
            # Debug: Print the differences
            print(f"DEBUG: Max absolute difference vs ref: {test_ref_diff}")
            print(f"DEBUG: Max absolute difference vs pt: {test_pt_diff}")
            print(f"DEBUG: Max absolute difference pt vs ref: {pt_ref_diff}")
            print(
                f"DEBUG: Tolerance check: {test_ref_diff} <= {ratio * pt_ref_diff + 1e-5}"
            )

        # First assertion with gap information
        tolerance_threshold = ratio * pt_ref_diff + 1e-4
        assert test_ref_diff <= tolerance_threshold, (
            f"Tolerance check failed: max_diff={test_ref_diff:.6f} > "
            f"threshold={tolerance_threshold:.6f}, gap={test_ref_diff - tolerance_threshold:.6f}"
        )

        # sanity checks
        assert test_ref_diff <= 0.5, (
            f"Max difference vs ref too large: {test_ref_diff:.6f} > 0.5, "
            f"gap={test_ref_diff - 0.5:.6f}"
        )
        assert pt_ref_diff <= 0.5, (
            f"Max difference pt vs ref too large: {pt_ref_diff:.6f} > 0.5, "
            f"gap={pt_ref_diff - 0.5:.6f}"
        )

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
        q = torch.randn(
            batch_size,
            seqlen_q,
            q_heads,
            head_dim,
            dtype=dtype if dtype != torch.float8_e4m3fn else torch.float,
            device=device,
            requires_grad=True,
        )
        k, v = (
            torch.randn(
                batch_size,
                seqlen_k,
                kv_heads,
                head_dim,
                dtype=dtype if dtype != torch.float8_e4m3fn else torch.float,
                device=device,
                requires_grad=True,
            )
            for _ in range(2)
        )
        if dtype == torch.float8_e4m3fn:
            q = q.to(torch.float8_e4m3fn)
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)
        return q, k, v

    # Generates K and V for paged attention for fixed length sequences.
    def _generate_qkv_paged(
        self,
        batch_size: int,
        seqlen_q: int,
        seqlen_k: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        page_block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        num_blocks = math.ceil(seqlen_k / page_block_size) * batch_size
        q = torch.randn(
            batch_size,
            seqlen_q,
            q_heads,
            head_dim,
            dtype=dtype if dtype != torch.float8_e4m3fn else torch.float,
            device=device,
            requires_grad=True,
        )
        k_paged, v_paged = (
            torch.randn(
                num_blocks,
                page_block_size,
                kv_heads,
                head_dim,
                dtype=dtype if dtype != torch.float8_e4m3fn else torch.float,
                device=device,
                requires_grad=True,
            )
            for _ in range(2)
        )
        page_table = rearrange(
            torch.randperm(num_blocks, dtype=torch.int32, device=device),
            "(b nblocks) -> b nblocks",
            b=batch_size,
        )
        if DEBUG:
            print(f"page_table: {page_table.size()}")

        k = rearrange(
            # pytorch 1.12 doesn't have indexing with int32
            k_paged[page_table.to(dtype=torch.long).flatten()],
            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
            b=batch_size,
        )[:, :seqlen_k]

        v = rearrange(
            v_paged[page_table.to(dtype=torch.long).flatten()],
            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
            b=batch_size,
        )[:, :seqlen_k]

        if dtype == torch.float8_e4m3fn:
            q = q.to(torch.float8_e4m3fn)
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)
            k_paged = k_paged.to(torch.float8_e4m3fn)
            v_paged = v_paged.to(torch.float8_e4m3fn)
        return q, k, v, k_paged, v_paged, page_table

    # Reshapes K and V for paged attention for variable length sequences.
    def _reshape_for_paged_attention(
        self,
        k_unpad: torch.Tensor,
        v_unpad: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = cu_seqlens_k.shape[0] - 1
        num_heads = k_unpad.shape[1]
        head_dim = k_unpad.shape[2]

        # Compute number of blocks per sequence
        seq_lens = [cu_seqlens_k[i + 1] - cu_seqlens_k[i] for i in range(batch_size)]
        num_blocks_per_seq = [math.ceil(len / page_block_size) for len in seq_lens]
        max_blocks = max(num_blocks_per_seq)

        # Prepare block table: [batch_size, max_blocks], fill with -1 for unused blocks
        page_table = torch.full(
            (batch_size, max_blocks),
            -1,
            dtype=torch.int32,
            device=k_unpad.device,
        )

        k_blocks_list = []
        v_blocks_list = []
        block_idx = 0
        for i in range(batch_size):
            start = cu_seqlens_k[i]
            end = cu_seqlens_k[i + 1]
            seq_len = end - start
            num_blocks = num_blocks_per_seq[i]

            # Pad sequence to multiple of page_block_size
            pad_len = num_blocks * page_block_size - seq_len
            k_seq = k_unpad[start:end]  # [seq_len, num_heads, head_dim]
            v_seq = v_unpad[start:end]

            if pad_len > 0:
                k_seq = torch.cat(
                    [
                        k_seq,
                        torch.zeros(
                            (int(pad_len.item()), num_heads, head_dim),
                            device=k_unpad.device,
                            dtype=k_unpad.dtype,
                        ),
                    ],
                    dim=0,
                )
                v_seq = torch.cat(
                    [
                        v_seq,
                        torch.zeros(
                            (int(pad_len.item()), num_heads, head_dim),
                            device=v_unpad.device,
                            dtype=v_unpad.dtype,
                        ),
                    ],
                    dim=0,
                )

            # Reshape to [num_blocks, page_block_size, num_heads, head_dim]
            k_seq_blocks = k_seq.view(num_blocks, page_block_size, num_heads, head_dim)
            v_seq_blocks = v_seq.view(num_blocks, page_block_size, num_heads, head_dim)

            k_blocks_list.append(k_seq_blocks)
            v_blocks_list.append(v_seq_blocks)

            # Fill page table
            page_table[i, :num_blocks] = torch.arange(
                block_idx,
                block_idx + num_blocks,
                device=k_unpad.device,
                dtype=torch.int32,
            )
            block_idx += num_blocks

        # Concatenate all blocks: [num_blocks_total, page_block_size, num_heads, head_dim]
        k_blocks = torch.cat(k_blocks_list, dim=0).to(
            device=k_unpad.device, dtype=k_unpad.dtype
        )
        v_blocks = torch.cat(v_blocks_list, dim=0).to(
            device=v_unpad.device, dtype=v_unpad.dtype
        )

        return k_blocks, v_blocks, page_table

    def _execute_cutlass_blackwell_attn_decode(
        self,
        batch_size: int,
        kv_padding: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        window_size: tuple[int, int],
        sm_scale: Optional[float],
        use_full_seqlen: bool = False,
    ) -> None:
        device = torch.accelerator.current_accelerator()
        assert device is not None
        torch.manual_seed(SEED)
        seqlen_q = 1
        causal = True
        assert seqlen_q <= kv_padding

        q, k, v = self._generate_qkv(
            batch_size,
            seqlen_q,
            kv_padding,
            q_heads,
            kv_heads,
            head_dim,
            device,
            dtype,
        )
        # Generate random key padding mask using utility function
        # For FP8, use full sequences (no variable lengths) since FP8 ref doesn't support masks
        if use_full_seqlen:
            key_padding_mask = generate_random_padding_mask(
                kv_padding, batch_size, device, mode="full", zero_lengths=False
            )
        else:
            key_padding_mask = generate_random_padding_mask(
                kv_padding, batch_size, device, mode="random", zero_lengths=False
            )
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="full", zero_lengths=False
        )

        (
            _q_unpad,
            _k_unpad,
            _v_unpad,
            _cu_seqlens_q,
            _cu_seqlens_k,
            _seqused_q,
            _seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q_for_ref,
            k_for_ref,
            v_for_ref,
            _output_pad_fn,
            _dq_pad_fn,
            _dk_pad_fn,
        ) = generate_qkv(
            q,
            k,
            v,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
        )

        # Run reference attention (also capture LSE for validation)
        out_baseline, _, lse_ref = attention_ref(
            q,
            k,
            v,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            causal=causal,
            window_size=window_size,
            upcast=True,
            softmax_scale=sm_scale,
            return_lse=True,
        )
        if dtype == torch.float8_e4m3fn:
            # reference implementation only supports decode case (seqlen_q == 1)
            # FP8 reference doesn't support masks or LSE
            out_ref = attention_ref_fp8(q, k, v)
            out_pt = attention_ref_fp8(q, k, v, reorder_ops=True)

        else:
            out_ref = out_baseline
            out_pt, _ = attention_ref(
                q,
                k,
                v,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                causal=causal,
                window_size=window_size,
                reorder_ops=True,
                upcast=False,
                softmax_scale=sm_scale,
            )
        if DEBUG:
            print(f"KV padding (constant): {kv_padding}")
            print(f"seqlen_kv (variable): {_seqused_k}")
        # Run decode-specific kernel
        out, lse = cutlass_blackwell_fmha_decode_forward(
            q,
            k,
            v,
            seqlen_kv=_seqused_k,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seq_len_q=None,
            max_seq_len_k=None,
            softmax_scale=sm_scale,
            causal=causal,
            window_left=window_size[0],
            window_right=window_size[1],
            bottom_right=True,
            split_k_size=0,
            use_heuristic=False,
        )

        # Output is [B, 1, H, 1, D] - squeeze num_splits dimension
        out = out.squeeze(3)  # [B, 1, H, D]
        # LSE is [B, 1, H, 1] - squeeze num_splits and adjust to [B, H, 1]
        lse = lse.squeeze(1)  # [B, H, 1]

        if DEBUG:
            print("cutlass_blackwell_fmha_decode_forward completed successfully!")

        # Compare outputs
        self._allclose(out, out_ref, out_pt)
        # Validate LSE correctness (skip for FP8 since ref doesn't support LSE)
        if dtype != torch.float8_e4m3fn:
            assert (
                lse.shape == lse_ref.shape
            ), f"LSE shape mismatch: {lse.shape} vs {lse_ref.shape}"

            lse_diff = (lse - lse_ref).abs().max().item()
            if DEBUG:
                print(f"LSE shape from kernel: {lse.shape}")
                print(f"LSE shape from reference: {lse_ref.shape}")
                print(f"Max LSE difference: {lse_diff}")

            assert (
                lse_diff <= 1e-2
            ), f"LSE comparison failed: max_diff={lse_diff:.6f} > 1e-2"

    def _execute_cutlass_blackwell_attn_dense(
        self,
        batch_size: int,
        seqlen_q: int,
        seqlen_k: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        page_block_size: int,
        dtype: torch.dtype,
        causal: bool,
        window_size: tuple[int, int],
        fwd_only: bool,
        deterministic: bool,
        sm_scale: Optional[float],
        is_paged: Optional[bool],
        use_compile: bool = False,
    ) -> None:
        device = torch.accelerator.current_accelerator()
        assert device is not None
        torch.manual_seed(SEED)
        assert seqlen_q <= seqlen_k

        # Initialize deterministic variables
        out_d = None
        out_paged: torch.Tensor | None = None
        k_paged: torch.Tensor | None = None
        v_paged: torch.Tensor | None = None
        page_table: torch.Tensor | None = None

        if is_paged:
            q, k, v, k_paged, v_paged, page_table = self._generate_qkv_paged(
                batch_size,
                seqlen_q,
                seqlen_k,
                q_heads,
                kv_heads,
                head_dim,
                page_block_size,
                device,
                dtype,
            )
        else:
            q, k, v = self._generate_qkv(
                batch_size,
                seqlen_q,
                seqlen_k,
                q_heads,
                kv_heads,
                head_dim,
                device,
                dtype,
            )

        # Initialize seqlen_kv for generation phase (seqlen_q == 1)
        seqlen_kv = None
        if seqlen_q == 1:
            seqlen_kv = torch.full(
                (batch_size,),
                seqlen_k,
                dtype=torch.int32,
                device=torch.accelerator.current_accelerator(),
            )

        # Run reference attention
        out_baseline, _ = attention_ref(
            q,
            k,
            v,
            causal=causal,
            window_size=window_size,
            upcast=True,
            softmax_scale=sm_scale,
        )
        if dtype == torch.float8_e4m3fn:
            # reference implementation only supports decode case (seqlen_q == 1)
            out_ref = attention_ref_fp8(q, k, v)
            out_pt = attention_ref_fp8(q, k, v, reorder_ops=True)

        else:
            out_ref = out_baseline
            out_pt, _ = attention_ref(
                q,
                k,
                v,
                causal=causal,
                window_size=window_size,
                reorder_ops=True,
                upcast=False,
                softmax_scale=sm_scale,
            )

        # Run tested kernel
        func_to_test = cutlass_blackwell_fmha_func
        if use_compile:
            func_to_test = torch.compile(func_to_test, fullgraph=True)
        if is_paged:
            assert k_paged is not None and v_paged is not None
            out_paged = func_to_test(
                q,
                k_paged,
                v_paged,
                causal=causal,
                window_size=window_size,
                seqlen_kv=seqlen_kv,
                page_table=page_table,
                seqlen_k=seqlen_k,
                deterministic=deterministic,
                softmax_scale=sm_scale,
            )

        out = func_to_test(
            q,
            k,
            v,
            causal=causal,
            window_size=window_size,
            seqlen_kv=seqlen_kv,
            page_table=None,
            seqlen_k=seqlen_k,
            deterministic=deterministic,
            softmax_scale=sm_scale,
        )

        if DEBUG:
            print("cutlass_blackwell_fmha_func completed successfully!")

        # Follow FlashAttention's numerical evaluation
        # Compare outputs
        if is_paged:
            # Compare paged output with both reference and non paged output
            self._allclose(out_paged, out_ref, out_pt)
            self._allclose(out_paged, out, out_pt)
        else:
            self._allclose(out, out_ref, out_pt)

        if deterministic:
            # Rerun the test. The outputs must be bit-wise exact
            out_d = func_to_test(
                q,
                cast(torch.Tensor, k_paged) if is_paged else k,
                cast(torch.Tensor, v_paged) if is_paged else v,
                causal=causal,
                window_size=window_size,
                seqlen_kv=seqlen_kv,
                page_table=page_table if is_paged else None,
                seqlen_k=seqlen_k,
                deterministic=deterministic,
                softmax_scale=sm_scale,
            )
            assert torch.equal(out, out_d)

        if fwd_only:
            return

        # Generate gradient tensor
        g = torch.rand_like(out)
        (
            dq,
            dk,
            dv,
            # Run attention backwards
        ) = torch.autograd.grad(out, (q, k, v), g)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)

        # Compare input gradients
        self._allclose(dq, dq_ref, dq_pt)
        self._allclose(dk, dk_ref, dk_pt)
        self._allclose(dv, dv_ref, dv_pt)

        if deterministic:
            # Rerun the test. The outputs must be bit-wise exact
            (
                dq_d,
                dk_d,
                dv_d,
            ) = torch.autograd.grad(out_d, (q, k, v), g)
            assert torch.equal(dq, dq_d)
            assert torch.equal(dk, dk_d)
            assert torch.equal(dv, dv_d)

    def _execute_cutlass_blackwell_attn_varlen(
        self,
        batch_size: int,
        seqlen_q: int,
        seqlen_k: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        page_block_size: int,
        dtype: torch.dtype,
        causal: bool,
        window_size: tuple[int, int],
        fwd_only: bool,
        deterministic: bool,
        sm_scale: Optional[float],
        is_paged: Optional[bool],
        use_compile: bool = False,
    ) -> None:
        device = torch.accelerator.current_accelerator()
        assert device is not None

        torch.manual_seed(SEED)

        out_paged: torch.Tensor | None = None
        k_paged: torch.Tensor | None = None
        v_paged: torch.Tensor | None = None
        page_table: torch.Tensor | None = None

        # Initialize deterministic variables
        out_unpad_d = None
        q_ref, k_ref, v_ref = self._generate_qkv(
            batch_size,
            seqlen_q,
            seqlen_k,
            q_heads,
            kv_heads,
            head_dim,
            device,
            dtype,
        )

        q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        # It fails with zero_lengths=True
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="random", zero_lengths=False
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, device, mode="random", zero_lengths=False
        )

        # Always have seqlen_k >= seqlen_q
        key_padding_mask[:, :seqlen_q] |= query_padding_mask

        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            _,
            _,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            kvpacked=False,
            query_unused_mask=None,
            key_unused_mask=None,
        )

        if is_paged:
            k_paged, v_paged, page_table = self._reshape_for_paged_attention(
                k_unpad, v_unpad, cu_seqlens_k, page_block_size
            )

        # Run attention forwards
        out_ref, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
            softmax_scale=sm_scale,
        )

        out_pt, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
            softmax_scale=sm_scale,
        )

        func_to_test = cutlass_blackwell_fmha_func
        if use_compile:
            func_to_test = torch.compile(func_to_test, fullgraph=True)
        if is_paged:
            assert k_paged is not None and v_paged is not None
            out_unpad_paged = func_to_test(
                q_unpad,
                k_paged,
                v_paged,
                causal=causal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seq_len_q=max_seqlen_q,
                max_seq_len_k=max_seqlen_k,
                page_table=page_table,
                window_size=window_size,
                deterministic=deterministic,
                softmax_scale=sm_scale,
            )
            out_paged = output_pad_fn(out_unpad_paged)

        out_unpad = func_to_test(
            q_unpad,
            k_unpad,
            v_unpad,
            causal=causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            page_table=None,
            window_size=window_size,
            deterministic=deterministic,
            softmax_scale=sm_scale,
        )
        out = output_pad_fn(out_unpad)

        # Follow FlashAttention's numerical evaluation
        # Compare outputs
        if is_paged:
            # Compare paged output with both reference and non paged output
            self._allclose(out_paged, out_ref, out_pt)
            self._allclose(out_paged, out, out_pt)
        else:
            self._allclose(out, out_ref, out_pt)

        if deterministic:
            # Rerun the test. The outputs must be bit-wise exact
            out_unpad_d = func_to_test(
                q_unpad,
                cast(torch.Tensor, k_paged) if is_paged else k_unpad,
                cast(torch.Tensor, v_paged) if is_paged else v_unpad,
                causal=causal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seq_len_q=max_seqlen_q,
                max_seq_len_k=max_seqlen_k,
                page_table=page_table,
                window_size=window_size,
                deterministic=deterministic,
                softmax_scale=sm_scale,
            )
            out_d = output_pad_fn(out_unpad_d)
            assert torch.equal(out, out_d)

        if fwd_only:
            return

        g_unpad = torch.randn_like(out_unpad)
        dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(
            out_unpad, (q_unpad, k_unpad, v_unpad), g_unpad
        )
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)

        g = output_pad_fn(g_unpad)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)

        # Compare input gradients
        self._allclose(dq, dq_ref, dq_pt)
        self._allclose(dk, dk_ref, dk_pt)
        self._allclose(dv, dv_ref, dv_pt)

        if deterministic:
            # Rerun the test. The outputs must be bit-wise exact
            dq_unpad_d, dk_unpad_d, dv_unpad_d = torch.autograd.grad(
                out_unpad_d, (q_unpad, k_unpad, v_unpad), g_unpad
            )
            dq_d = dq_pad_fn(dq_unpad_d)
            dk_d = dk_pad_fn(dk_unpad_d)
            dv_d = dk_pad_fn(dv_unpad_d)
            assert torch.equal(dq, dq_d)
            assert torch.equal(dk, dk_d)
            assert torch.equal(dv, dv_d)

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (
                dtype,
                kv_padding,
                batch_size,
                is_mqa,
                window_size,
                head_dim,
                num_groups,
            )
            for dtype in [torch.bfloat16, torch.float8_e4m3fn]
            for kv_padding in [64, 128, 256, 1024, 8192]
            for batch_size in [1, 2]
            for is_mqa in [True, False]
            for window_size in [(-1, -1), (0, 0), (128, 0), (1024, 0)]
            for head_dim in [128, 64]
            for num_groups in [1, 2]
        ]
    )
    def test_decode(
        self,
        dtype: torch.dtype,
        kv_padding: int,
        batch_size: int,
        is_mqa: bool,
        window_size: tuple[int, int],
        head_dim: int,
        num_groups: int = 1,
        q_heads: int = 8,
    ) -> None:
        if DEBUG:
            print(
                f"Running test_decode with params: "
                f"dtype={dtype}, kv_padding={kv_padding}, batch_size={batch_size}, "
                f"is_mqa={is_mqa}, window_size={window_size}, head_dim={head_dim}, "
                f"num_groups={num_groups}, q_heads={q_heads}"
            )

        # Skip test for known numerical precision issues with FP8 and head_dim=64 in GQA mode
        if dtype == torch.float8_e4m3fn and (head_dim == 64 or window_size[0] >= 0):
            self.skipTest("Skip: Numerical precision issue with FP8, head_dim=64")

        self._execute_cutlass_blackwell_attn_decode(
            batch_size,
            kv_padding,
            q_heads,
            kv_heads=num_groups if is_mqa else q_heads,
            head_dim=head_dim,
            dtype=dtype,
            window_size=window_size,
            sm_scale=None,
            # FP8 ref doesn't support variable lengths, use full sequences
            use_full_seqlen=(dtype == torch.float8_e4m3fn),
        )

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (batch_size, kv_padding, window_left, q_heads, kv_heads, head_dim)
            for batch_size in [1, 2, 4]
            for kv_padding in [512, 1024, 2048, 4096]
            for window_left in [
                -1,  # Disabled (no windowing)
                128,  # Small window
                256,  # Medium window
                512,  # Window < kv_padding for some cases
                1024,  # Window == kv_padding for some cases
                2048,  # Window > kv_padding for some cases
            ]
            for q_heads in [4]
            for kv_heads in [1, 2]  # Test MQA and GQA
            for head_dim in [128]
        ]
    )
    def test_decode_sliding_window(
        self,
        batch_size: int,
        kv_padding: int,
        window_left: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        """
        Test decode attention with sliding window.

        Tests scenarios:
        - window_left <= 0: Disabled, use full sequence
        - window_left >= kv_padding: Use full sequence (no effect)
        - window_left < kv_padding: Only attend to last window_left tokens
        """
        if DEBUG:
            print(
                f"Running test_decode_sliding_window with params: "
                f"batch_size={batch_size}, kv_padding={kv_padding}, "
                f"window_left={window_left}, q_heads={q_heads}, "
                f"kv_heads={kv_heads}, head_dim={head_dim}"
            )

        # Convert window_left to window_size tuple format
        # For decode, right window is typically 0 (or -1 if window is disabled)
        window_right = 0 if window_left > 0 else -1
        window_size = (window_left, window_right)

        self._execute_cutlass_blackwell_attn_decode(
            batch_size=batch_size,
            kv_padding=kv_padding,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            window_size=window_size,
            sm_scale=None,
            use_full_seqlen=False,
        )

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (batch_size, kv_padding, window_left)
            for batch_size in [1, 2]
            for kv_padding in [1024, 2048]
            for window_left in [
                1,  # Minimum window (only last token)
                kv_padding - 1,  # Window = seqlen - 1
                kv_padding,  # Window == seqlen (no effect)
                kv_padding + 1,  # Window > seqlen (no effect)
            ]
        ]
    )
    def test_decode_sliding_window_edge_cases(
        self,
        batch_size: int,
        kv_padding: int,
        window_left: int,
    ) -> None:
        """
        Test sliding window at boundary conditions.

        Tests edge cases:
        - window_left = 1: Minimum window (only attend to last token)
        - window_left = kv_padding - 1: Window just under full sequence
        - window_left = kv_padding: Window equals sequence length
        - window_left = kv_padding + 1: Window exceeds sequence length
        """
        if DEBUG:
            print(
                f"Running test_decode_sliding_window_edge_cases with params: "
                f"batch_size={batch_size}, kv_padding={kv_padding}, "
                f"window_left={window_left}"
            )

        q_heads = 8
        kv_heads = 1
        head_dim = 128

        # Convert window_left to window_size tuple format
        window_right = 0 if window_left > 0 else -1
        window_size = (window_left, window_right)

        self._execute_cutlass_blackwell_attn_decode(
            batch_size=batch_size,
            kv_padding=kv_padding,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            window_size=window_size,
            sm_scale=None,
            use_full_seqlen=False,
        )

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (
                kv_padding,
                batch_size,
                q_heads,
                causal,
                window_size,
                head_dim,
                sm_scale,
            )
            for kv_padding in [128, 256, 512, 1024]
            for batch_size in [2, 8]
            for q_heads in [8, 16]
            for causal in [True, False]
            for window_size in [(-1, -1), (128, 0), (200, 0)]
            for head_dim in [128]
            for sm_scale in [None]
        ]
    )
    def test_jagged_vs_padded_kv(
        self,
        kv_padding: int,
        batch_size: int,
        q_heads: int,
        causal: bool,
        window_size: tuple[int, int],
        head_dim: int,
        sm_scale: Optional[float],
    ) -> None:
        """
        Test comparing two scenarios:
        a) Jagged KV: Only pass cu_seqlen, not seqlen_kv
        b) Padded KV: Pass cu_seqlen as [0, max_t, 2*max_t, ...] and seqlen_kv same as seqlen_k in scenario a

        The outputs should be identical if the implementation is correct.
        """
        # batch_size = 2
        # kv_padding = 128
        seqlen_q = kv_padding  # Maximum sequence length (padded size)
        device = torch.accelerator.current_accelerator()
        kv_heads = 1
        dtype = torch.bfloat16

        torch.manual_seed(SEED)

        # Create tensors
        q_padded = torch.randn(
            batch_size,
            seqlen_q,
            q_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        ).to(dtype)

        # Create full-sized KV tensors (padded to max length)
        k_padded = torch.randn(
            batch_size,
            kv_padding,
            kv_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        ).to(dtype)
        v_padded = torch.randn(
            batch_size,
            kv_padding,
            kv_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        ).to(dtype)

        k_padding_mask = generate_random_padding_mask(
            kv_padding, batch_size, device, mode="random", zero_lengths=False
        )
        q_padding_mask = generate_random_padding_mask(
            kv_padding, batch_size, device, mode="third", zero_lengths=False
        )
        # # Always have seqlen_k >= seqlen_q
        k_padding_mask[:, :seqlen_q] |= q_padding_mask
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k_jagged,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            _,
            _,
        ) = generate_qkv(
            q_padded,
            k_padded,
            v_padded,
            q_padding_mask,
            k_padding_mask,
        )
        # Create variable length sequences
        cu_seqlens_k_padded = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )

        # Set cumulative sequence lengths
        for i in range(batch_size + 1):
            cu_seqlens_k_padded[i] = i * kv_padding

        assert torch.all(
            seqused_k <= kv_padding
        ), "Actual sequence lengths must be less than or equal to max sequence length"

        if DEBUG:
            print("\n=== Testing Jagged KV vs Padded KV with seqlen_kv ===")
            print(f"cu_seqlen_q: {cu_seqlens_q}")
            print(f"jagged cu_seqlens_k: {cu_seqlens_k_jagged}")
            print(f"padded cu_seqlens_k: {cu_seqlens_k_padded}")
            print(f"seqlen_kv: {seqused_k}")
            print(f"max_seqlen_q: {max_seqlen_q}")
            print(f"max_seqlen_k: {max_seqlen_k}")
            print(f"q_unpad: {q_unpad.shape}")

        # Scenario A: Jagged KV with cu_seqlens_k
        out_jagged = cutlass_blackwell_fmha_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k_jagged,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
            softmax_scale=sm_scale,
        )

        # # Scenario B: Padded KV with seqlen_kv
        # # Run with padded KV and seqlen_kv
        k_ = k_padded.view(-1, kv_heads, head_dim)
        v_ = v_padded.view(-1, kv_heads, head_dim)

        out_padded = cutlass_blackwell_fmha_func(
            q_unpad,
            k_,
            v_,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k_padded,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            causal=causal,
            window_size=window_size,
            seqlen_kv=seqused_k,
            softmax_scale=sm_scale,
        )
        if DEBUG:
            print(f"out_jagged: {out_jagged}")
            print(f"k_: {k_.shape}")
            print(f"v_: {v_.shape}")
            print(f"out_padded: {out_padded}")

        # # Compare outputs
        diff = (out_jagged - out_padded).abs().max().item()
        self.assertLess(
            diff,
            1e-5,
            "Jagged KV and Padded KV with seqlen_kv produced significantly different outputs",
        )

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (
                seqlen_q,
                offset_q,
                batch_size,
                causal,
                is_gqa,
                is_varlen,
                kv_heads,
                window_size,
                head_dim,
                sm_scale,
            )
            for seqlen_q, offset_q in [
                (101, 0),
                (111, 2),
                (256, 0),
                (1024, 0),
                (113, 90),
                (128, 90),
                (256, 90),
                (256, 128),
                (1024, 128),
            ]
            for batch_size in [1, 2, 8]
            for causal in [False, True]
            for is_gqa in [False, True]
            for is_varlen in [False, True]
            for kv_heads in [1, 2, 3, 4]
            for window_size in [(-1, -1), (128, 0), (200, 0)]
            for head_dim in [64, 128]
            for sm_scale in [None]
        ]
    )
    def test_forward(
        self,
        seqlen_q: int,
        offset_q: int,
        batch_size: int,
        causal: bool,
        is_gqa: bool,
        is_varlen: bool,
        kv_heads: int,
        window_size: tuple[int, int],
        head_dim: int,
        sm_scale: Optional[float],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        random.seed(SEED)
        seqlen_k = offset_q + seqlen_q
        if seqlen_k > seqlen_q:
            causal = True
        test_func = (
            self._execute_cutlass_blackwell_attn_varlen
            if is_varlen
            else self._execute_cutlass_blackwell_attn_dense
        )
        q_heads_per_kv_head = random.randint(2, 8) if is_gqa else 1
        test_func(
            batch_size,
            seqlen_q,
            seqlen_k,
            q_heads=kv_heads * q_heads_per_kv_head,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_block_size=0,
            dtype=dtype,
            causal=causal,
            window_size=window_size,
            fwd_only=True,
            deterministic=False,
            sm_scale=sm_scale,
            is_paged=False,
        )

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (
                seqlen_q,
                offset_q,
                batch_size,
                causal,
                is_gqa,
                is_varlen,
                kv_heads,
                window_size,
                head_dim,
                sm_scale,
                page_block_size,
            )
            for seqlen_q, offset_q in [
                (101, 0),
                (113, 90),
                (256, 0),
                (256, 128),
                (1024, 25),
            ]
            for batch_size in [1, 4]
            for causal in [False, True]
            for is_gqa in [False, True]
            for is_varlen in [False, True]
            for kv_heads in [1, 2]
            for window_size in [(-1, -1), (23, 0)]
            for head_dim in [64, 128]
            for sm_scale in [None]
            for page_block_size in [128, 256]
        ]
    )
    def test_paged_forward(
        self,
        seqlen_q: int,
        offset_q: int,
        batch_size: int,
        causal: bool,
        is_gqa: bool,
        is_varlen: bool,
        kv_heads: int,
        window_size: tuple[int, int],
        head_dim: int,
        sm_scale: Optional[float],
        page_block_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        random.seed(SEED)
        seqlen_k = offset_q + seqlen_q
        if seqlen_k > seqlen_q:
            causal = True
        test_func = (
            self._execute_cutlass_blackwell_attn_varlen
            if is_varlen
            else self._execute_cutlass_blackwell_attn_dense
        )
        q_heads_per_kv_head = random.randint(2, 8) if is_gqa else 1
        test_func(
            batch_size,
            seqlen_q,
            seqlen_k,
            q_heads=kv_heads * q_heads_per_kv_head,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_block_size=page_block_size,
            dtype=dtype,
            causal=causal,
            window_size=window_size,
            fwd_only=True,
            deterministic=False,
            sm_scale=sm_scale,
            is_paged=True,
        )

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (
                batch_size,
                seqlen,
                offset,
                kv_heads,
                causal,
                is_gqa,
                is_varlen,
                window_size,
                deterministic,
                head_dim,
                sm_scale,
            )
            for batch_size in [2]
            for seqlen, offset in [
                (8, 0),
                (103, 0),
                (256, 0),
                (256, 1024),
                (1024, 8192),
            ]
            for kv_heads in [2]
            for causal in [True, False]
            for is_gqa in [True]
            for is_varlen in [True]
            # Include small window sizes that trigger the barrier coordination bug
            for window_size in [
                (-1, -1),
                (256, -1),
                (128, 128),
                (512, -1),
                (128, -1),
                (-1, 128),
            ]
            for deterministic in [False, True]
            for head_dim in [64, 128]
            for sm_scale in [None]
        ]
    )
    def test_backward(
        self,
        batch_size: int,
        seqlen: int,
        offset: int,
        kv_heads: int,
        causal: bool,
        is_gqa: bool,
        is_varlen: bool,
        window_size: tuple[int, int],
        deterministic: bool,
        head_dim: int,
        sm_scale: Optional[float],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if DEBUG:
            # Print test parameters for debugging
            print(
                f"Running test_backward with params: "
                f"batch_size={batch_size}, seqlen={seqlen}, offset={offset}, "
                f"kv_heads={kv_heads}, causal={causal}, is_gqa={is_gqa}, "
                f"is_varlen={is_varlen}, window_size={window_size}, "
                f"deterministic={deterministic}, head_dim={head_dim}, "
                f"sm_scale={sm_scale}, dtype={dtype}"
            )

        random.seed(SEED)
        test_func = (
            self._execute_cutlass_blackwell_attn_varlen
            if is_varlen
            else self._execute_cutlass_blackwell_attn_dense
        )
        q_heads_per_kv_head = random.randint(2, 8) if is_gqa else 1
        test_func(
            batch_size,
            seqlen,
            seqlen + offset,
            q_heads=kv_heads * q_heads_per_kv_head,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_block_size=0,
            dtype=dtype,
            causal=causal,
            window_size=window_size,
            fwd_only=False,
            deterministic=deterministic,
            sm_scale=sm_scale,
            is_paged=False,
        )

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (
                batch_size,
                seqlen,
                offset,
                kv_heads,
                causal,
                is_gqa,
                is_varlen,
                window_size,
                deterministic,
                head_dim,
            )
            for batch_size in [2]
            for seqlen, offset in [
                (256, 0),
                (256, 1024),
            ]
            for kv_heads in [2]
            for causal in [True, False]
            for is_gqa in [True]
            for is_varlen in [True]
            for window_size in [(-1, -1), (128, 128)]
            for deterministic in [False, True]
            for head_dim in [64, 128]
        ]
    )
    def test_sm_scale(
        self,
        batch_size: int,
        seqlen: int,
        offset: int,
        kv_heads: int,
        causal: bool,
        is_gqa: bool,
        is_varlen: bool,
        window_size: tuple[int, int],
        deterministic: bool,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Test custom softmax scale (1.0 / head_dim)."""
        sm_scale = 1.0 / head_dim

        random.seed(SEED)
        test_func = (
            self._execute_cutlass_blackwell_attn_varlen
            if is_varlen
            else self._execute_cutlass_blackwell_attn_dense
        )
        q_heads_per_kv_head = random.randint(2, 8) if is_gqa else 1
        test_func(
            batch_size,
            seqlen,
            seqlen + offset,
            q_heads=kv_heads * q_heads_per_kv_head,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_block_size=0,
            dtype=dtype,
            causal=causal,
            window_size=window_size,
            fwd_only=False,
            deterministic=deterministic,
            sm_scale=sm_scale,
            is_paged=False,
        )

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (
                is_varlen,
                is_mqa,
                seqlen_q,
            )
            for is_varlen in [False, True]
            for is_mqa in [False, True]
            for seqlen_q in [1, 64]
        ]
    )
    def test_compile(
        self,
        is_varlen: bool,
        is_mqa: bool,
        seqlen_q: int,
    ):
        # Skip decode (seqlen_q=1) with dense attention (is_varlen=False) due to shape mismatch
        if not is_varlen and seqlen_q == 1:
            self.skipTest("Skip: Decode with dense attention has shape mismatch issue")

        test_func = (
            self._execute_cutlass_blackwell_attn_varlen
            if is_varlen
            else self._execute_cutlass_blackwell_attn_dense
        )
        q_heads = 8
        kv_heads = 2 if is_mqa else q_heads
        batch_size = 2
        seqlen_k = 128
        kv_heads = 2
        head_dim = 128
        dtype = torch.bfloat16
        causal = True
        # Decode kernel does not support sliding window attention yet
        window_size = (-1, -1)
        deterministic = False
        # Backward pass is not supported for generation phase (sq=1)
        is_decode = seqlen_q == 1
        fwd_only = is_decode
        # Decode kernel does not support sm_scale
        sm_scale = None if is_decode else 1.0 / head_dim

        if is_decode:
            self._execute_cutlass_blackwell_attn_decode(
                batch_size,
                seqlen_k,
                q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                window_size=window_size,
                sm_scale=None,
            )
            return

        test_func(
            batch_size,
            seqlen_q,
            seqlen_k,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_block_size=0,
            dtype=dtype,
            causal=causal,
            window_size=window_size,
            fwd_only=fwd_only,
            deterministic=deterministic,
            sm_scale=sm_scale,
            is_paged=False,
        )

    @skip_cuda_lt_sm100
    @skip_rocm
    @parameterized.expand(
        [
            (
                seqlen_q,
                offset_q,
                batch_size,
                causal,
                kv_heads,
                window_size,
                head_dim,
                sm_scale,
            )
            for seqlen_q, offset_q in [
                (1, 0),  # Decode path
                (101, 0),
                (256, 0),
                (1024, 0),
                (128, 90),
            ]
            for batch_size in [1, 2]
            for causal in [False, True]
            for kv_heads in [1, 2]
            for window_size in [(-1, -1), (128, 0)]
            for head_dim in [64, 128]
            for sm_scale in [None]
        ]
    )
    def test_lse_correctness(
        self,
        seqlen_q: int,
        offset_q: int,
        batch_size: int,
        causal: bool,
        kv_heads: int,
        window_size: tuple[int, int],
        head_dim: int,
        sm_scale: Optional[float],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Test LSE correctness by calling internal forward functions directly
        and comparing with reference implementation.
        """
        device = torch.accelerator.current_accelerator()
        assert device is not None
        torch.manual_seed(SEED)

        seqlen_k = offset_q + seqlen_q
        if seqlen_k > seqlen_q:
            causal = True

        q_heads = kv_heads * 2  # Use GQA for testing

        # Generate test data
        q, k, v = self._generate_qkv(
            batch_size,
            seqlen_q,
            seqlen_k,
            q_heads,
            kv_heads,
            head_dim,
            device,
            dtype,
        )

        # Compute reference LSE
        _, _, lse_ref = attention_ref(
            q,
            k,
            v,
            causal=causal,
            window_size=window_size,
            upcast=True,
            softmax_scale=sm_scale,
            return_lse=True,
        )

        # Call _cutlass_blackwell_fmha_forward (returns tuple with LSE)
        _, lse_fwd = _cutlass_blackwell_fmha_forward(
            q,
            k,
            v,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seq_len_q=None,
            max_seq_len_k=None,
            softmax_scale=sm_scale,
            causal=causal,
            seqlen_kv=None,
            page_table=None,
            seqlen_k=None,
            window_left=window_size[0],
            window_right=window_size[1],
            bottom_right=True,
        )

        # Validate LSE shapes match before comparing values
        assert (
            lse_fwd.shape == lse_ref.shape
        ), f"LSE shape mismatch: {lse_fwd.shape} vs {lse_ref.shape}"

        if DEBUG:
            print(f"LSE shape from kernel: {lse_fwd.shape}")
            print(f"LSE shape from reference: {lse_ref.shape}")
            print(f"Max LSE difference: {(lse_fwd - lse_ref).abs().max().item()}")

        # Compare LSE values with tolerance
        lse_diff = (lse_fwd - lse_ref).abs().max().item()
        assert (
            lse_diff <= 1e-2
        ), f"LSE comparison failed: max_diff={lse_diff:.6f} > 1e-2"

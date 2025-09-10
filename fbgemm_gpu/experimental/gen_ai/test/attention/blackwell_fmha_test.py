# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest
from typing import Tuple

import hypothesis.strategies as st
import torch

from fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha import (
    cutlass_blackwell_fmha_func,
)
from hypothesis import given, HealthCheck, settings, Verbosity
from parameterized import parameterized

from .attention_ref_fp8 import attention_ref_fp8
from .test_utils import attention_ref, generate_qkv, generate_random_padding_mask

common_settings = {
    "verbosity": Verbosity.verbose,
    "max_examples": 20,
    "deadline": None,
    "suppress_health_check": [HealthCheck.filter_too_much, HealthCheck.data_too_large],
}

DEBUG = False

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
        if DEBUG:
            # Debug: Print the differences
            print(
                f"DEBUG: Max absolute difference vs ref: {self._abs_max(t_test - t_ref)}"
            )
            print(
                f"DEBUG: Max absolute difference vs pt: {self._abs_max(t_test - t_pt)}"
            )
            print(
                f"DEBUG: Max absolute difference pt vs ref: {self._abs_max(t_pt - t_ref)}"
            )
            print(
                f"DEBUG: Tolerance check: {self._abs_max(t_test - t_ref)} <= {2 * self._abs_max(t_pt - t_ref) + 1e-5}"
            )
        assert self._abs_max(t_test - t_ref) <= 2 * self._abs_max(t_pt - t_ref) + 1e-4

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _execute_cutlass_blackwell_attn_dense(
        self,
        batch_size: int,
        seqlen_q: int,
        seqlen_k: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        causal: bool,
        window_size: tuple[int, int],
        fwd_only: bool,
        deterministic: bool,
    ) -> None:
        device = torch.accelerator.current_accelerator()
        assert device is not None
        assert seqlen_q <= seqlen_k
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
            q, k, v, causal=causal, window_size=window_size, upcast=True
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
            )

        # Run tested kernel
        out = cutlass_blackwell_fmha_func(
            q,
            k,
            v,
            causal=causal,
            window_size=window_size,
            seqlen_kv=seqlen_kv,
            deterministic=deterministic,
        )
        if DEBUG:
            print("cutlass_blackwell_fmha_func completed successfully!")

        # Follow FlashAttention's numerical evaluation
        # Compare outputs
        self._allclose(out, out_ref, out_pt)

        if deterministic:
            # Rerun the test. The outputs must be bit-wise exact
            out_d = cutlass_blackwell_fmha_func(
                q,
                k,
                v,
                causal=causal,
                window_size=window_size,
                seqlen_kv=seqlen_kv,
                deterministic=deterministic,
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
        dtype: torch.dtype,
        causal: bool,
        window_size: tuple[int, int],
        fwd_only: bool,
        deterministic: bool,
    ) -> None:
        device = torch.accelerator.current_accelerator()
        assert device is not None
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

        # Run attention forwards
        out_ref, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
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
        )

        out_unpad = cutlass_blackwell_fmha_func(
            q_unpad,
            k_unpad,
            v_unpad,
            causal=causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            window_size=window_size,
            deterministic=deterministic,
        )
        out = output_pad_fn(out_unpad)

        # Follow FlashAttention's numerical evaluation
        # Compare outputs
        self._allclose(out, out_ref, out_pt)

        if deterministic:
            # Rerun the test. The outputs must be bit-wise exact
            out_unpad_d = cutlass_blackwell_fmha_func(
                q_unpad,
                k_unpad,
                v_unpad,
                causal=causal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seq_len_q=max_seqlen_q,
                max_seq_len_k=max_seqlen_k,
                window_size=window_size,
                deterministic=deterministic,
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
                seqlen_k,
                batch_size,
                is_mqa,
                window_size,
            )
            for seqlen_k in [64, 128, 256, 1024]
            for batch_size in [1, 2]
            for is_mqa in [True]
            for window_size in [(-1, -1), (0, 0), (0, 128), (128, 0), (1024, 0)]
        ]
    )
    def test_decode(
        self,
        seqlen_k: int,
        batch_size: int,
        is_mqa: bool,
        window_size: tuple[int, int],
        q_heads: int = 8,
        dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        seqlen_q = 1
        causal = True
        assert (
            dtype == torch.float8_e4m3fn
        ), "Gen Kernel only supports float8_e4m3fn for now"
        self._execute_cutlass_blackwell_attn_dense(
            batch_size,
            seqlen_q,
            seqlen_k,
            q_heads,
            kv_heads=1 if is_mqa else q_heads,
            head_dim=128,
            dtype=dtype,
            causal=causal,
            window_size=window_size,
            fwd_only=True,
            deterministic=False,
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
            )
            for kv_padding in [128, 256, 512, 1024]
            for batch_size in [2, 8]
            for q_heads in [8, 16]
            for causal in [True, False]
            for window_size in [(-1, -1), (0, 0), (0, 128), (128, 0), (1024, 0)]
        ]
    )
    def test_jagged_vs_padded_kv(
        self,
        kv_padding: int,
        batch_size: int,
        q_heads: int,
        causal: bool,
        window_size: tuple[int, int] = (-1, -1),
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
        head_dim = 128
        dtype = torch.bfloat16

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
            for window_size in [(-1, -1), (0, 0), (0, 128), (128, 0), (1024, 0)]
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
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
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
            head_dim=128,
            dtype=dtype,
            causal=causal,
            window_size=window_size,
            fwd_only=True,
            deterministic=False,
        )

    @skip_cuda_lt_sm100
    @skip_rocm
    @given(
        batch_size=st.integers(min_value=1, max_value=128),
        seqlen=st.integers(min_value=8, max_value=1024),
        kv_heads=st.integers(min_value=1, max_value=4),
        dtype=st.sampled_from([torch.bfloat16]),
        causal=st.booleans(),
        is_varlen=st.booleans(),
        is_gqa=st.booleans(),
        window_size=st.sampled_from(
            [(-1, -1), (128, 0), (256, 0), (128, 128), (512, 0)]
        ),
        deterministic=st.booleans(),
    )
    @settings(**common_settings)
    def test_backward(
        self,
        batch_size: int,
        seqlen: int,
        kv_heads: int,
        dtype: torch.dtype,
        causal: bool,
        is_varlen: bool,
        is_gqa: bool,
        window_size: tuple[int, int],
        deterministic: bool,
    ) -> None:
        test_func = (
            self._execute_cutlass_blackwell_attn_varlen
            if is_varlen
            else self._execute_cutlass_blackwell_attn_dense
        )
        q_heads_per_kv_head = random.randint(2, 8) if is_gqa else 1
        test_func(
            batch_size,
            seqlen,
            seqlen,
            q_heads=kv_heads * q_heads_per_kv_head,
            kv_heads=kv_heads,
            head_dim=128,
            dtype=dtype,
            causal=causal,
            window_size=window_size,
            fwd_only=False,
            deterministic=deterministic,
        )

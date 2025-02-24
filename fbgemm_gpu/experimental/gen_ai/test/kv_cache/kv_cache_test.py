# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import logging
import unittest
from enum import Enum, unique
from typing import List, Optional, Tuple

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import torch
from hypothesis import given, settings, strategies as st

try:
    from xformers.attn_bias_utils import pack_kv_cache
    from xformers.ops import fmha

    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

if HAS_XFORMERS:
    from rope_padded import rope_padded


@unique
class LogicalDtype(Enum):
    bf16 = 0
    fp8 = 1
    int4 = 2


logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _get_varseq_batch_seqpos(
    seqlens_q: List[int], seqlens_kv: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    varseq_batch[i] is batch index of query i
    varseq_seqpos[i] is the offset of the last key which query i attends to
    """

    varseq_batch = torch.cat(
        [
            torch.as_tensor([i for _ in range(len_q)], dtype=torch.int, device="cuda")
            for i, len_q in enumerate(seqlens_q)
        ]
    )
    varseq_seqpos = torch.cat(
        [
            torch.as_tensor(
                [len_kv - len_q + t for t in range(len_q)],
                dtype=torch.int,
                device="cuda",
            )
            for len_q, len_kv in zip(seqlens_q, seqlens_kv)
        ]
    )
    return varseq_batch, varseq_seqpos


class KVCacheTests(unittest.TestCase):
    @settings(deadline=None)
    @given(
        num_groups=st.sampled_from([1, 2, 4, 8]),
        MAX_T=st.sampled_from([8000, 16384]),
        N_KVH_L=st.sampled_from([1, 2]),
    )
    @unittest.skipIf(
        not HAS_XFORMERS,
        "Skip when xformers is not available",
    )
    def test_int4_kv_cache(self, num_groups: int, MAX_T: int, N_KVH_L: int) -> None:
        N_H_L = 2
        T = 2
        B = 2
        D_H = 128
        # D = 8192
        # D_H = 128
        # B = 16
        # PROMPT_T = 1024

        xq = (
            torch.randn(size=(B * T, N_H_L, D_H), dtype=torch.bfloat16, device="cuda")
            * 0.01
        )
        xk = (
            torch.randn(size=(B * T, N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda")
            * 0.01
        )
        xv = (
            torch.randn(size=(B * T, N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda")
            * 0.01
        )
        varseq_seqpos = torch.cat(
            [
                torch.as_tensor(list(range(T)), dtype=torch.int, device="cuda")
                for b in range(B)
            ]
        )
        varseq_batch = torch.cat(
            [
                torch.as_tensor([b for _ in range(T)], dtype=torch.int, device="cuda")
                for b in range(B)
            ]
        )
        attn_bias = (
            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=[T for _ in range(B)],
                kv_padding=MAX_T,
                kv_seqlen=[T for _ in range(B)],
            )
        )
        attn_bias.k_seqinfo.to(torch.device("cuda"))
        assert attn_bias.k_seqinfo.seqlen.shape == (B,)
        assert attn_bias.k_seqinfo.seqlen.tolist() == [T for _ in range(B)]

        theta = 10000.0
        cache_k_bf16 = torch.zeros(
            size=(B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda"
        )
        cache_v_bf16 = torch.zeros(
            size=(B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda"
        )

        xq_out_bf16 = torch.compile(torch.ops.fbgemm.rope_qkv_varseq_prefill)(
            xq,
            xk,
            xv,
            cache_k_bf16,
            cache_v_bf16,
            varseq_batch,
            varseq_seqpos,
            theta,
        )
        qparam_offset = 4 * num_groups

        cache_k_int4 = torch.zeros(
            size=(B, MAX_T, N_KVH_L, int(D_H // 2) + qparam_offset),
            dtype=torch.uint8,
            device="cuda",
        )
        cache_v_int4 = torch.zeros(
            size=(B, MAX_T, N_KVH_L, int(D_H // 2) + qparam_offset),
            dtype=torch.uint8,
            device="cuda",
        )
        xq_out = torch.compile(torch.ops.fbgemm.rope_qkv_varseq_prefill)(
            xq,
            xk,
            xv,
            cache_k_int4,
            cache_v_int4,
            varseq_batch,
            varseq_seqpos,
            theta,
            num_groups=num_groups,
            cache_logical_dtype_int=LogicalDtype.int4.value,
        )
        torch.testing.assert_close(xq_out_bf16, xq_out)

        dequantized_cache = torch.compile(torch.ops.fbgemm.dequantize_int4_cache)(
            cache_k_int4,
            cache_v_int4,
            attn_bias.k_seqinfo.seqlen,
            num_groups=num_groups,
        )
        cache_k, cache_v = dequantized_cache

        torch.testing.assert_close(
            cache_k[:, :T], cache_k_bf16[:, :T], atol=1.0e-2, rtol=1.0e-2
        )
        torch.testing.assert_close(
            cache_v[:, :T], cache_v_bf16[:, :T], atol=1.0e-2, rtol=1.0e-2
        )

    @settings(deadline=None)
    @given(
        MAX_T=st.sampled_from([8000, 16384]),
        N_KVH_L=st.sampled_from([1, 2]),
    )
    @unittest.skipIf(
        not torch.cuda.is_available()
        or (
            torch.version.cuda
            and torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9
        )
        or (torch.version.hip)
        or not HAS_XFORMERS,
        "Skip when H100 is not available",
    )
    def test_fp8_kv_cache(self, MAX_T: int, N_KVH_L: int) -> None:
        N_H_L = 2
        T = 2
        B = 2
        D_H = 128

        xq = (
            torch.cat(
                [
                    torch.randn(N_H_L, D_H, dtype=torch.bfloat16, device="cuda") * (i)
                    for i in range(B * T)
                ]
            )
        ).view(B * T, N_H_L, D_H)
        scale_step = 0.01 / B / T
        shift_step = 5 * scale_step
        xk_rows = [
            scale_step
            * (i + 1)
            * torch.randn(size=(N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda")
            + i * shift_step
            for i in range(B * T)
        ]
        xv_rows = [
            scale_step
            * (i + 1)
            * torch.randn(size=(N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda")
            + i * shift_step
            for i in range(B * T)
        ]

        xk = (torch.cat(xk_rows)).view(B * T, N_KVH_L, D_H)

        xv = (torch.cat(xv_rows)).view(B * T, N_KVH_L, D_H)
        varseq_seqpos = torch.cat(
            [
                torch.as_tensor(list(range(T)), dtype=torch.int, device="cuda")
                for b in range(B)
            ]
        )
        varseq_batch = torch.cat(
            [
                torch.as_tensor([b for _ in range(T)], dtype=torch.int, device="cuda")
                for b in range(B)
            ]
        )
        attn_bias = (
            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=[T for _ in range(B)],
                kv_padding=MAX_T,
                kv_seqlen=[T for _ in range(B)],
            )
        )
        attn_bias.k_seqinfo.to(torch.device("cuda"))
        assert attn_bias.k_seqinfo.seqlen.shape == (B,)
        assert attn_bias.k_seqinfo.seqlen.tolist() == [T for _ in range(B)]

        theta = 10000.0
        cache_k_bf16 = torch.zeros(
            size=(B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda"
        )
        cache_v_bf16 = torch.zeros(
            size=(B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda"
        )

        xq_out_bf16 = torch.compile(torch.ops.fbgemm.rope_qkv_varseq_prefill)(
            xq,
            xk,
            xv,
            cache_k_bf16,
            cache_v_bf16,
            varseq_batch,
            varseq_seqpos,
            theta,
        )
        qparam_offset = 4

        cache_k_fp8 = torch.zeros(
            size=(B, MAX_T, N_KVH_L, int(D_H) + qparam_offset),
            dtype=torch.uint8,
            device="cuda",
        )
        cache_v_fp8 = torch.zeros(
            size=(B, MAX_T, N_KVH_L, int(D_H) + qparam_offset),
            dtype=torch.uint8,
            device="cuda",
        )
        xq_out = torch.compile(torch.ops.fbgemm.rope_qkv_varseq_prefill)(
            xq,
            xk,
            xv,
            cache_k_fp8,
            cache_v_fp8,
            varseq_batch,
            varseq_seqpos,
            theta,
            cache_logical_dtype_int=LogicalDtype.fp8.value,
        )
        torch.testing.assert_close(xq_out_bf16, xq_out)

        dequantized_cache = torch.compile(torch.ops.fbgemm.dequantize_fp8_cache)(
            cache_k_fp8,
            cache_v_fp8,
            attn_bias.k_seqinfo.seqlen,
        )
        cache_k, cache_v = dequantized_cache

        torch.testing.assert_close(
            cache_k[:, :T], cache_k_bf16[:, :T], atol=1.0e-2, rtol=5.0e-2
        )
        torch.testing.assert_close(
            cache_v[:, :T], cache_v_bf16[:, :T], atol=1.0e-2, rtol=5.0e-2
        )

    @settings(deadline=None)
    @given(
        prefill=st.booleans(),
        rope_theta=st.sampled_from([None, 10000.0]),
        MAX_T=st.sampled_from([4000, 8192]),
        B=st.sampled_from([1, 128]),
        BLOCK_N=st.sampled_from([64, 128, 256]),
    )
    @unittest.skipIf(
        not HAS_XFORMERS,
        "Skip when xformers is not available",
    )
    def test_positional_encoding_with_paged_attention(
        self,
        prefill: bool,
        rope_theta: Optional[float],
        MAX_T: int,
        B: int,
        BLOCK_N: int,
    ) -> None:

        N_H_L = 1
        N_KVH_L = 8
        D_H = 128
        torch.manual_seed(100)

        kv_seqlens = torch.randint(low=0, high=MAX_T, size=(B,)).tolist()
        q_seqlens = kv_seqlens if prefill else [1 for _ in range(B)]
        seq_positions = torch.tensor(
            [x - 1 for x in kv_seqlens], device="cuda", dtype=torch.int32
        )
        total_length_q = sum(q_seqlens)

        cache_k = torch.randn(
            (B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda"
        )
        cache_v = torch.randn_like(cache_k)

        block_tables, packed_cache_k, packed_cache_v = pack_kv_cache(
            cache_k,
            cache_v,
            [x + 1 for x in seq_positions],
            BLOCK_N=BLOCK_N,
        )

        assert packed_cache_k.is_contiguous()
        assert packed_cache_v.is_contiguous()

        xqkv = torch.randn(
            total_length_q,
            N_H_L + 2 * N_KVH_L,
            D_H,
            dtype=torch.bfloat16,
            device="cuda",
        )
        xq = xqkv[:, :N_H_L, :]
        # This clone is to avoid a weirdness in torch.compile:
        # because as far as the signature of rope_qkv_varseq_prefill
        # goes, xk could be modified (but in fact isn't because we
        # aren't using write_k_back=True) and torch.compile takes
        # action to be careful when a function has aliased parameters
        # one of which is modified. The function merge_view_inputs in
        # runtime_wrappers.py inside torch compile has an IndexError
        # without this clone.
        xk = xqkv[:, N_H_L : N_H_L + N_KVH_L, :].clone()
        xv = xqkv[:, N_H_L + N_KVH_L :, :]

        xpos_gamma: float = 0.8
        xpos_scale_base: float = 4096.0
        xpos_theta: float = 500000.0
        xpos_exponent_offset = 0

        assert cache_k.is_contiguous()
        assert cache_v.is_contiguous()

        B_T = total_length_q
        assert xq.shape == (B_T, N_H_L, D_H)
        assert xk.shape == (B_T, N_KVH_L, D_H)
        assert xv.shape == (B_T, N_KVH_L, D_H)

        assert cache_k.shape == (B, MAX_T, N_KVH_L, D_H)
        assert cache_v.shape == (B, MAX_T, N_KVH_L, D_H)

        if prefill:
            seqpos_args = _get_varseq_batch_seqpos(q_seqlens, kv_seqlens)
        else:
            seqpos_args = (seq_positions,)

        if rope_theta is not None:
            func = (
                torch.compile(torch.ops.fbgemm.rope_qkv_varseq_prefill)
                if prefill
                else torch.compile(torch.ops.fbgemm.rope_qkv_decoding)
            )
            xq_out_ref = func(
                xq,
                xk,
                xv,
                cache_k,
                cache_v,
                *seqpos_args,
                rope_theta,
                num_groups=0,
            )
            xq_out_paged = func(
                xq,
                xk,
                xv,
                packed_cache_k,
                packed_cache_v,
                *seqpos_args,
                rope_theta,
                num_groups=0,
                block_tables=block_tables,
                page_size=BLOCK_N,
            )
        else:
            func = (
                torch.compile(torch.ops.fbgemm.xpos_qkv_varseq_prefill)
                if prefill
                else torch.compile(torch.ops.fbgemm.xpos_qkv_decoding)
            )
            xq_out_ref = func(
                xq,
                xk,
                xv,
                cache_k,
                cache_v,
                *seqpos_args,
                theta=xpos_theta,
                gamma=xpos_gamma,
                scale_base=xpos_scale_base,
                exponent_offset=xpos_exponent_offset,
                num_groups=0,
            )
            xq_out_paged = func(
                xq,
                xk,
                xv,
                packed_cache_k,
                packed_cache_v,
                *seqpos_args,
                xpos_theta,
                xpos_gamma,
                xpos_scale_base,
                xpos_exponent_offset,
                num_groups=0,
                block_tables=block_tables,
                page_size=BLOCK_N,
            )
        torch.testing.assert_close(xq_out_ref, xq_out_paged)

        for b in range(B):
            num_blocks = (kv_seqlens[b] + BLOCK_N - 1) // BLOCK_N
            for logical_idx in range(num_blocks):
                len_to_compare = min(kv_seqlens[b] - logical_idx * BLOCK_N, BLOCK_N)
                for kv_ref, kv_packed in (
                    (cache_k, packed_cache_k),
                    (cache_v, packed_cache_v),
                ):
                    physical_idx = block_tables[b][logical_idx]
                    logical_start = logical_idx * BLOCK_N
                    physical_start = physical_idx * BLOCK_N
                    ref_vals = kv_ref[
                        b,
                        logical_start : logical_start + len_to_compare,
                    ]
                    packed_vals = kv_packed[0][
                        physical_start : physical_start + len_to_compare
                    ]
                    torch.testing.assert_close(ref_vals, packed_vals)

    @settings(deadline=None)
    @given(
        prefill=st.booleans(),
        rope_theta=st.sampled_from([10000.0]),
        MAX_T=st.sampled_from([8192]),
        B=st.sampled_from([128]),
        BLOCK_N=st.sampled_from([256]),
    )
    @unittest.skipIf(
        not HAS_XFORMERS,
        "Skip when xformers is not available",
    )
    def test_rope_positional_encoding_only(
        self,
        prefill: bool,
        rope_theta: float,
        MAX_T: int,
        B: int,
        BLOCK_N: int,
    ) -> None:
        N_H_L = 1
        N_KVH_L = 8
        D_H = 128
        torch.manual_seed(100)

        kv_seqlens = torch.randint(low=0, high=MAX_T, size=(B,)).tolist()
        q_seqlens = kv_seqlens if prefill else [1 for _ in range(B)]
        seq_positions = torch.tensor(
            [x - 1 for x in kv_seqlens], device="cuda", dtype=torch.int32
        )
        total_length_q = sum(q_seqlens)

        cache_k = torch.randn(
            (B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda"
        )
        cache_v = torch.randn_like(cache_k)

        xqkv = torch.randn(
            total_length_q,
            N_H_L + 2 * N_KVH_L,
            D_H,
            dtype=torch.bfloat16,
            device="cuda",
        )
        xq = xqkv[:, :N_H_L, :]
        xk = xqkv[:, N_H_L : N_H_L + N_KVH_L, :].clone()
        xv = xqkv[:, N_H_L + N_KVH_L :, :]

        assert cache_k.is_contiguous()
        assert cache_v.is_contiguous()

        B_T = total_length_q
        assert xq.shape == (B_T, N_H_L, D_H)
        assert xk.shape == (B_T, N_KVH_L, D_H)
        assert xv.shape == (B_T, N_KVH_L, D_H)

        assert cache_k.shape == (B, MAX_T, N_KVH_L, D_H)
        assert cache_v.shape == (B, MAX_T, N_KVH_L, D_H)

        if prefill:
            seqpos_args = _get_varseq_batch_seqpos(q_seqlens, kv_seqlens)
        else:
            seqpos_args = (seq_positions,)

        func = (
            torch.compile(torch.ops.fbgemm.rope_qkv_varseq_prefill)
            if prefill
            else torch.compile(torch.ops.fbgemm.rope_qkv_decoding)
        )
        xq_out = func(
            xq,
            xk,
            xv,
            cache_k,
            cache_v,
            *seqpos_args,
            rope_theta,
            num_groups=0,
        )
        xq_out = xq_out.view(1, xq_out.shape[0], xq_out.shape[1], xq_out.shape[2])
        attn_bias = (
            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=q_seqlens,
                kv_padding=MAX_T,
                kv_seqlen=kv_seqlens,
            )
        )
        attn_bias.k_seqinfo.to(torch.device("cuda"))
        xq = xq.view(1, xq.shape[0], N_H_L, D_H)
        xk = xk.view(1, xk.shape[0], N_KVH_L, D_H)
        xv = xv.view(1, xv.shape[0], N_KVH_L, D_H)
        cache_k = cache_k.view(1, B * MAX_T, N_KVH_L, D_H)
        cache_v = cache_k.view(1, B * MAX_T, N_KVH_L, D_H)
        xq_out_ref = rope_padded(
            xq=xq,
            xk=xk,
            xv=xv,
            cache_k=cache_k,
            cache_v=cache_v,
            attn_bias=attn_bias,
            theta=rope_theta,
        )

        torch.testing.assert_close(xq_out, xq_out_ref, atol=0.01, rtol=0.01)

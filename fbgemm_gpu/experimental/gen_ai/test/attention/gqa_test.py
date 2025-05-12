#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from enum import Enum, unique
from typing import List, Tuple

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import assume, given, settings, Verbosity

try:
    from xformers.ops import fmha
    from xformers.ops.fmha import triton_splitk

    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False


VERBOSITY: Verbosity = Verbosity.verbose


@unique
class LogicalDtype(Enum):
    bf16 = 0
    fp8 = 1
    int4 = 2


def quant_int4_dequant_bf16(
    in_tensor: torch.Tensor, num_groups: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A util function for quantizing a tensor from from a float type (including
    FP32, FP16, BF16) to INT4 and then dequantize the INT4 result to BF16
    (i.e., fake quantization)
    """
    in_shape = in_tensor.shape
    in_tensor = in_tensor.reshape(
        *in_shape[:-1], num_groups, in_shape[-1] // num_groups
    )

    # Find max and min for each group
    max_vals = torch.max(in_tensor, dim=-1, keepdim=True).values
    min_vals = torch.min(in_tensor, dim=-1, keepdim=True).values

    # Compute scale and shift
    scale: torch.Tensor = (max_vals - min_vals) / 15
    shift = torch.min(in_tensor, dim=-1, keepdim=True).values
    scale = scale.to(torch.float16)
    shift = shift.to(torch.float16)
    shift_expand = shift.expand(in_tensor.shape)
    scale_expand = scale.expand(in_tensor.shape)

    # Scale and shift
    in_bytes = ((in_tensor - shift_expand) / scale_expand).to(torch.uint8)

    # Get only 4 bits
    in_int4 = in_bytes & 0xF

    # Pack int4 in uint8
    in_int4_packed = in_int4[..., ::2] + (in_int4[..., 1::2] << 4)

    # Concat data
    scale_shift = torch.concat(
        [scale.view(torch.uint8), shift.view(torch.uint8)], dim=-1
    )
    in_quant = torch.concat(
        [
            scale_shift.flatten(start_dim=-2),
            in_int4_packed.flatten(start_dim=-2),
        ],
        dim=-1,
    )

    # Dequantize tensor for reference
    in_fp16 = in_int4.to(torch.float16)

    # Convert type based on the CUDA implementation
    in_quant_dequant_fp16 = (in_fp16 * scale_expand) + shift_expand
    in_quant_dequant_fp32 = in_quant_dequant_fp16.to(torch.float)
    in_quant_dequant_bf16 = in_quant_dequant_fp32.to(torch.bfloat16)

    return in_quant, in_quant_dequant_bf16.view(*in_shape)


def gqa_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    seq_lens: List[int],
    qk_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The reference GQA implementation
    """
    (B, T, H, D) = Q.shape
    (_, MAX_T, Hk, D) = K.shape
    (_, MAX_T, Hv, D) = V.shape
    assert T == 1
    assert Hk == Hv
    Y = torch.zeros_like(Q)
    attn_out = torch.zeros(B, H, MAX_T)
    for b in range(B):
        max_t = seq_lens[b]
        for h in range(H):
            # now, compute fused attention
            Q_ = Q[b, 0, h, :]
            assert Q_.shape == (D,)
            K_ = K[b, :max_t, 0, :]
            assert K_.shape == (max_t, D)
            S = (Q_.view(1, D) @ K_.T) * qk_scale  # 1.0 / np.sqrt(D)
            # max_qk_acc = torch.max(S)
            # softmax_denominator = torch.exp(S - max_qk_acc).sum()
            assert S.shape == (1, max_t)
            P = torch.nn.functional.softmax(S, dim=-1)

            assert P.shape == (1, max_t)

            V_ = V[b, :max_t, 0, :]
            assert V_.shape == (max_t, D)
            O_ = P.view(1, max_t) @ V_
            assert O_.shape == (1, D)
            Y[b, 0, h, :] = O_
            attn_out[b, h, :max_t] = P
    return Y, attn_out


class Int4GQATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        device = torch.accelerator.current_accelerator()
        assert device is not None
        cls.device = device

    @unittest.skipIf(
        torch.version.hip is not None,
        "gqa_attn_splitk with use_tensor_cores=True is not supported on ROCm",
    )
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
        "Skip when CUDA is not available or CUDA compute capability is less than 8",
    )
    @settings(verbosity=VERBOSITY, max_examples=40, deadline=None)
    # pyre-ignore
    @given(
        int4_kv=st.booleans(),
        num_groups=st.sampled_from([1, 4]),
        B=st.integers(min_value=1, max_value=128),
        MAX_T=st.integers(min_value=4, max_value=128),
        N_H_L=st.integers(min_value=1, max_value=128),
    )
    def test_gqa(
        self,
        int4_kv: bool,
        num_groups: int,
        B: int,
        MAX_T: int,
        N_H_L: int,
    ) -> None:
        """
        Test correctness of torch.ops.fbgemm.gqa_attn_splitk against the
        reference GQA implementation (testing both BF16 and INT4 KV caches)
        """

        # Constants
        D_H = 128
        N_KVH_L = 1  # gqa_attn_splitk only supports 1 currently
        SEQ_POSITION = MAX_T - 2

        seq_positions = torch.tensor(
            [SEQ_POSITION for _ in range(B)], device=self.device
        ).int()
        kv_seqlens = [seq_position + 1 for seq_position in seq_positions]
        q = torch.randn((B, 1, N_H_L, D_H), dtype=torch.bfloat16, device=self.device)

        # Generate KV cache
        cache_k = torch.randn(
            (B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device=self.device
        )
        cache_v = torch.randn_like(cache_k)
        if int4_kv:
            cache_k, cache_k_ref = quant_int4_dequant_bf16(cache_k, num_groups)
            cache_v, cache_v_ref = quant_int4_dequant_bf16(cache_v, num_groups)
            cache_k_ref = cache_k_ref.cpu().float()
            cache_v_ref = cache_v_ref.cpu().float()
        else:
            cache_k_ref = cache_k.cpu().float()
            cache_v_ref = cache_v.cpu().float()

        # Compute qk_scale
        qk_scale = 1.0 / np.sqrt(D_H)

        # Run reference
        z_ref, attn_out_ref = gqa_reference(
            q.cpu().float(),
            cache_k_ref,
            cache_v_ref,
            kv_seqlens,
            qk_scale=qk_scale,
        )

        # Run test
        for split_k in [1, 2, 4, 8, 13, 16]:
            z, _, _ = torch.ops.fbgemm.gqa_attn_splitk(
                q,
                cache_k,
                cache_v,
                seq_positions,
                qk_scale=qk_scale,
                num_split_ks=split_k,
                kv_cache_quant_num_groups=num_groups,
            )
            torch.testing.assert_close(
                z.cpu().bfloat16(),
                z_ref.cpu().bfloat16(),
                atol=2.0e-2,
                rtol=6.0e-3,
            )

    @settings(deadline=None)
    @given(
        dtype=st.sampled_from(["bf16", "fp8", "int4"]),
        num_groups=st.sampled_from([1, 2, 4, 8]),
        args=st.sampled_from(
            [
                (1, 16, 8, 128),
                (7, 16, 7, 128),
                (7, 16, 128, 128),
                (13, 4, 1, 128),
                (13, 4, 16, 128),
                (13, 4, 23, 128),
                (13, 4, 50, 128),
                (111, 121, 8, 128),
                (111, 121, 16, 128),
                (111, 121, 19, 128),
                (111, 121, 128, 128),
                (111, 121, 31, 128),
                (0, 16, 8, 128),
            ]
        ),
        mqa=st.booleans(),
        validate_p_inf_exp=st.booleans(),
    )
    @unittest.skipIf(
        torch.version.hip is not None,
        "gqa_attn_splitk with use_tensor_cores=True is not supported on ROCm",
    )
    # pyre-fixme[56]
    @unittest.skipIf(
        not torch.cuda.is_available()
        # or torch.cuda.get_device_capability()[0] < 8
        or not HAS_XFORMERS,
        "Skip when CUDA is not available or xformers is not available",
    )
    def test_mqa_main(  # noqa C901
        self,
        dtype: str,
        num_groups: int,
        args: Tuple[int, int, int, int],
        mqa: bool,
        validate_p_inf_exp: bool,
    ) -> None:
        """
        Compare various MQA kernels against the reference implementation, with and without
        int4 K/V cache quantization.
        When int4_kv is True, comparison with reference implementation is done like this:
        - random bf16 K/V are generated
        - they are placed into the "original" K/V cache
        - they are also quantized and placed into "quantized" int4 K/V cache
        - then the "quantized" K/V cache is dequantized to get "reference" K/V cache
        - Finally, we compare reference MQA on "reference" K/V cache
          with the output of MQA-kernel-under-test on the "quantized" one
        This way we eliminate the quantization error from the comparison and verify only
        the correctness of the implementation.
        """
        # TODO : FP8
        assume(not validate_p_inf_exp or args[0] != 0)

        B, MAX_T, N_H_L, D_H = args

        def mqa_reference(
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            seq_lens: List[int],
            qk_scale: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            (B, T, H, D) = Q.shape
            (_, MAX_T, Hk, D) = K.shape
            (_, MAX_T, Hv, D) = V.shape
            assert T == 1
            assert Hk == Hv
            if mqa:
                assert Hk == 1
            else:
                assert Hk == H
            Y = torch.zeros_like(Q)
            attn_out = torch.zeros(B, H, MAX_T)
            for b in range(B):
                max_t = seq_lens[b]
                for h in range(H):
                    # now, compute fused attention
                    Q_ = Q[b, 0, h, :]
                    assert Q_.shape == (D,)
                    K_ = K[b, :max_t, 0 if mqa else h, :]
                    assert K_.shape == (max_t, D)
                    S = (Q_.view(1, D) @ K_.T) * qk_scale  # 1.0 / np.sqrt(D)
                    # max_qk_acc = torch.max(S)
                    # softmax_denominator = torch.exp(S - max_qk_acc).sum()
                    assert S.shape == (1, max_t)
                    P = torch.nn.functional.softmax(S, dim=-1)

                    assert P.shape == (1, max_t)

                    V_ = V[b, :max_t, 0 if mqa else h, :]
                    assert V_.shape == (max_t, D)
                    O_ = P.view(1, max_t) @ V_
                    assert O_.shape == (1, D)
                    # weighted sum of columns
                    # if b == 0 and h == 0:
                    #     logger.info(S / qk_scale)
                    #     logger.info(P)
                    #     logger.info(O_)
                    Y[b, 0, h, :] = O_
                    attn_out[b, h, :max_t] = P
            return Y, attn_out

        N_KVH_L = 1 if mqa else N_H_L
        SEQ_POSITION = MAX_T - 2
        seq_positions = torch.tensor(
            [SEQ_POSITION for _ in range(B)], device=self.device
        ).int()
        kv_seqlens = [seq_position + 1 for seq_position in seq_positions]
        q = torch.randn((B, 1, N_H_L, D_H), dtype=torch.bfloat16, device=self.device)
        if validate_p_inf_exp:
            # Validate inf exp of P.  Set the first value of the query to be
            # very large.  This will cause some of the QK^T results (i.e., P)
            # to be large.  If P is not shifted by the max value before passing
            # to exp, the test will fail since the power of the exponent is too
            # large causing the result to be inf
            q[0][0][0][0] = 1000
        # cache_x_ref is for input to reference implementation
        if dtype in ["fp8", "int4"]:

            if dtype == "fp8":
                num_groups = 1
                qparam_offset = 4 * num_groups
                D_H_KV = D_H + qparam_offset
                l_dtype = LogicalDtype.fp8.value
            else:
                qparam_offset = 4 * num_groups
                D_H_KV = D_H // 2 + qparam_offset
                l_dtype = LogicalDtype.int4.value

            xq = torch.randn(
                (B * SEQ_POSITION, N_H_L, D_H), dtype=torch.bfloat16, device=self.device
            )
            xk = torch.randn(
                (B * SEQ_POSITION, N_KVH_L, D_H),
                dtype=torch.bfloat16,
                device=self.device,
            )
            xv = torch.randn_like(xk)

            cache_k_orig = torch.zeros(
                size=(B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device=self.device
            )
            cache_v_orig = torch.zeros_like(cache_k_orig)
            # Create and fill quantized K/V cache
            cache_k = torch.zeros(
                B,
                MAX_T,
                N_KVH_L,
                D_H_KV,
                dtype=torch.uint8,
                device=self.device,
            )
            cache_v = torch.zeros_like(cache_k)

            if B > 0:
                varseq_seqpos = torch.cat(
                    [
                        torch.as_tensor(
                            list(range(SEQ_POSITION)),
                            dtype=torch.int,
                            device=self.device,
                        )
                        for b in range(B)
                    ]
                )
                varseq_batch = torch.cat(
                    [
                        torch.as_tensor(
                            [b for _ in range(SEQ_POSITION)],
                            dtype=torch.int,
                            device=self.device,
                        )
                        for b in range(B)
                    ]
                )

                theta = 10000.0
                # Create and fill non-quantized K/V cache
                torch.ops.fbgemm.rope_qkv_varseq_prefill(
                    xq,
                    xk,
                    xv,
                    cache_k_orig,
                    cache_v_orig,
                    varseq_batch,
                    varseq_seqpos,
                    theta,
                )
                torch.ops.fbgemm.rope_qkv_varseq_prefill(
                    xq,
                    xk,
                    xv,
                    cache_k,
                    cache_v,
                    varseq_batch,
                    varseq_seqpos,
                    theta,
                    num_groups=num_groups,
                    cache_logical_dtype_int=l_dtype,
                )
            if dtype == "fp8":
                cache_k_ref, cache_v_ref = torch.ops.fbgemm.dequantize_fp8_cache(
                    cache_k.view(torch.uint8).contiguous(),
                    cache_v.view(torch.uint8).contiguous(),
                    torch.tensor(kv_seqlens, device=cache_k.device, dtype=torch.int32),
                )
            else:
                cache_k_ref, cache_v_ref = torch.ops.fbgemm.dequantize_int4_cache(
                    cache_k.view(torch.uint8).contiguous(),
                    cache_v.view(torch.uint8).contiguous(),
                    torch.tensor(kv_seqlens, device=cache_k.device, dtype=torch.int32),
                    num_groups=num_groups,
                )
            # May be assert that cache_k_ref is close to cache_k_orig to eleminate error due to quantization
            cache_k_ref = cache_k_ref.cpu().float()
            cache_v_ref = cache_v_ref.cpu().float()
        else:
            # Not quantized cache
            l_dtype = LogicalDtype.bf16.value
            D_H_KV = D_H
            cache_k = torch.randn(
                (B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device=self.device
            )
            cache_v = torch.randn_like(cache_k)
            cache_k_ref = cache_k.cpu().float()
            cache_v_ref = cache_v.cpu().float()

        qk_scale = 1.0 / np.sqrt(D_H)
        z_ref, attn_out_ref = mqa_reference(
            q.cpu().float(),
            cache_k_ref,
            cache_v_ref,
            kv_seqlens,
            qk_scale=qk_scale,
        )

        z = torch.ops.fbgemm.mqa_attn(
            q,
            cache_k,
            cache_v,
            seq_positions,
            qk_scale=qk_scale,
            num_groups=num_groups,
            cache_logical_dtype_int=l_dtype,
        )
        torch.testing.assert_close(
            z.cpu().bfloat16(), z_ref.cpu().bfloat16(), atol=2.0e-3, rtol=6.0e-3
        )
        if mqa and num_groups in [1, 4]:
            for split_k in [1, 2, 4, 8, 13, 16]:
                z, _, _ = torch.ops.fbgemm.gqa_attn_splitk(
                    q,
                    cache_k,
                    cache_v,
                    seq_positions,
                    qk_scale=qk_scale,
                    num_split_ks=split_k,
                    kv_cache_quant_num_groups=num_groups,
                    use_tensor_cores=True,
                    cache_logical_dtype_int=l_dtype,
                )
                torch.testing.assert_close(
                    z.cpu().bfloat16(),
                    z_ref.cpu().bfloat16(),
                    atol=2.0e-2,
                    rtol=6.0e-3,
                )

        if dtype == "fp8":
            return
        int4_kv = dtype == "int4"

        # TODO: Remove the following gqa testing. Looks redundant.
        # gqa_attn_splitk gives numerical errors on H100 for group-wise quantization
        gqa_attn_splitk_known_failure = (
            int4_kv
            and self.device.type == "cuda"
            and num_groups > 1
            and torch.cuda.get_device_capability(q.device) >= (9, 0)
        )
        # gqa_attn_splitk requires multiquery
        if mqa:
            for split_k in [1, 2, 4, 8, 13, 16]:
                if not gqa_attn_splitk_known_failure:
                    z_split_k, _, _ = torch.ops.fbgemm.gqa_attn_splitk(
                        q,
                        cache_k,
                        cache_v,
                        seq_positions,
                        qk_scale=qk_scale,
                        num_split_ks=split_k,
                        kv_cache_quant_num_groups=num_groups,
                        use_tensor_cores=False,
                    )
                    torch.testing.assert_close(
                        z_split_k.cpu().bfloat16(),
                        z_ref.cpu().bfloat16(),
                        atol=2.0e-2,
                        rtol=6.0e-3,
                    )
                if num_groups == 1 or num_groups == 4:
                    z, _, _ = torch.ops.fbgemm.gqa_attn_splitk(
                        q,
                        cache_k,
                        cache_v,
                        seq_positions,
                        qk_scale=qk_scale,
                        num_split_ks=split_k,
                        kv_cache_quant_num_groups=num_groups,
                        use_tensor_cores=True,
                    )
                    torch.testing.assert_close(
                        z.cpu().bfloat16(),
                        z_ref.cpu().bfloat16(),
                        atol=2.0e-2,
                        rtol=6.0e-3,
                    )

        # Not testing B = 0 with Triton as it might not be supported
        # Not testing N_H_L >= 128 as Triton ops failed with the illegal
        # memory access error
        if B == 0 or N_H_L >= 128:
            return

        # TODO: Add fmha.flash.FwOp when it supports padded mask
        for op in (
            [
                triton_splitk.FwOp,
                triton_splitk.FwOp_S16,
                triton_splitk.FwOp_S32,
                triton_splitk.FwOp_S64,
                triton_splitk.FwOp_S128,
            ]
            + [fmha.cutlass.FwOp]
            if not int4_kv
            else []
        ):
            if int4_kv:
                op = type(
                    f"{op.__name__}_{num_groups}",
                    (op,),
                    {"NUM_GROUPS": num_groups},
                )
            attn_bias = (
                fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                    q_seqlen=[1] * B,
                    kv_padding=MAX_T,
                    kv_seqlen=[s.item() for s in kv_seqlens],
                )
            )

            axq = q.view(1, B * 1, N_H_L, D_H)
            axk = cache_k.view(1, B * MAX_T, N_KVH_L, D_H_KV).expand(
                1, B * MAX_T, N_H_L, D_H_KV
            )
            axv = cache_v.view(1, B * MAX_T, N_KVH_L, D_H_KV).expand(
                1, B * MAX_T, N_H_L, D_H_KV
            )
            y = fmha.memory_efficient_attention_forward(
                axq,
                axk.view(torch.int32) if int4_kv else axk,
                axv.view(torch.int32) if int4_kv else axv,
                attn_bias,
                op=op,
            )
            torch.testing.assert_close(
                y.view(*z_ref.shape),
                z_ref.bfloat16().to(self.device),
                atol=1.0e-2,
                rtol=1.0e-2,
            )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha import (
    cutlass_blackwell_fmha_func,
)


def _allclose(
    t_1: torch.Tensor,
    t_2: torch.Tensor,
) -> tuple[float, float]:
    diff = t_1 - t_2
    return diff.abs().max().item(), diff.abs().sum().item()


def _generate_inputs(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.accelerator.current_accelerator()
    assert device is not None
    assert seqlen_q <= seqlen_k
    q = torch.randn(
        batch_size,
        seqlen_q,
        q_heads,
        head_dim,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    ).to(dtype)
    k = torch.randn(
        batch_size,
        seqlen_k,
        kv_heads,
        head_dim,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    ).to(dtype)
    v = torch.randn(
        batch_size,
        seqlen_k,
        kv_heads,
        head_dim,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    ).to(dtype)
    g = torch.rand_like(q)
    return q, k, v, g


def _execute_cutlass_blackwell_attn_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Run tested kernel
    out = cutlass_blackwell_fmha_func(q, k, v, causal=causal, seqlen_kv=None)
    (
        dq,
        dk,
        dv,
        # Run attention backwards
    ) = torch.autograd.grad(out, (q, k, v), g)

    return out, dq, dk, dv


def test_dense(
    batch_size=4,
    seqlen_q=8192,
    seqlen_k=8192,
    q_heads=8,
    kv_heads=8,
    head_dim=128,
    dtype=torch.bfloat16,
    causal=False,
):
    q, k, v, do = _generate_inputs(
        batch_size, seqlen_q, seqlen_k, q_heads, kv_heads, head_dim, dtype
    )
    o1, dq1, dk1, dv1 = _execute_cutlass_blackwell_attn_dense(q, k, v, do, causal)
    o2, dq2, dk2, dv2 = _execute_cutlass_blackwell_attn_dense(q, k, v, do, causal)
    print(
        f"Testing {batch_size} {seqlen_q} {seqlen_k} {q_heads} {kv_heads} {head_dim} {dtype} {causal}"
    )
    print(f"output difference (max and sum): {_allclose(o1, o2)}")
    print(f"dq difference (max and sum): {_allclose(dq1, dq2)}")
    print(f"dk difference (max and sum): {_allclose(dk1, dk2)}")
    print(f"dv difference (max and sum): {_allclose(dv1, dv2)}")
    print("")


torch.use_deterministic_algorithms(True)

for seq in [1024, 2048, 4096, 8192]:
    test_dense(seqlen_q=seq, seqlen_k=seq, causal=False)
    test_dense(seqlen_q=seq, seqlen_k=seq, causal=True)

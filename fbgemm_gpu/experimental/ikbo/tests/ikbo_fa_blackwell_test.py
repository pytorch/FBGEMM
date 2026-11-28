# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import sys

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip("CUDA GPU unavailable", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("Blackwell (sm_100 / sm_103) only", allow_module_level=True)

from ikbo.ops.tlx_ikbo_fa_blackwell import (
    plan_flash_attn_ikbo_blackwell,
    tlx_flash_attn_ikbo_blackwell,
)
from ikbo.ops.tlx_ikbo_fa_blackwell_1cta import ikbo_fa_1cta, make_tile_table
from ikbo.ops.tlx_ikbo_fa_blackwell_2cta import ikbo_fa_2cta

DEVICE = "cuda"
DTYPE = torch.float16


def broadcast_sdpa(
    query, key, value, cand_to_user_index, n_seed, num_heads, d_head, max_seq_len
):
    q = query.view(-1, n_seed, num_heads, d_head).permute(0, 2, 1, 3)
    k = torch.index_select(
        key.view(-1, max_seq_len, num_heads, d_head), 0, cand_to_user_index
    ).permute(0, 2, 1, 3)
    v = torch.index_select(
        value.view(-1, max_seq_len, num_heads, d_head), 0, cand_to_user_index
    ).permute(0, 2, 1, 3)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def prepare_inputs(
    B,
    n_seed,
    num_heads,
    d_head,
    max_seq_len,
    low_num_cands_per_user,
    high_num_cands_per_user,
):
    random.seed(0)
    counts = []
    total = 0
    while total < B:
        # odd and even candidate counts with equal chance, as in ikbo_fa_test.py
        cur = min(
            random.randint(low_num_cands_per_user, high_num_cands_per_user)
            + random.randint(0, 1),
            B - total,
        )
        counts.append(cur)
        total += cur
    counts = torch.tensor(counts)
    cand_to_user_index = torch.repeat_interleave(
        torch.arange(counts.numel()), counts
    ).to(dtype=torch.int32, device=DEVICE)
    num_users = counts.numel()
    query = torch.randn((B * n_seed, num_heads, d_head), device=DEVICE, dtype=DTYPE)
    key = torch.randn(
        (num_users * max_seq_len, num_heads, d_head), device=DEVICE, dtype=DTYPE
    )
    value = torch.randn(
        (num_users * max_seq_len, num_heads, d_head), device=DEVICE, dtype=DTYPE
    )
    return query, key, value, cand_to_user_index


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("B", [512, 1024, 2048])
@pytest.mark.parametrize("n_seed", [64])
@pytest.mark.parametrize("num_heads", [1, 2, 4, 6])
@pytest.mark.parametrize("d_head", [128])
@pytest.mark.parametrize("max_seq_len", [500, 512, 1000, 1024, 2000, 2048])
@pytest.mark.parametrize("cand_to_user_ratio", [4, 10, 70])
def test_tlx_ikbo_fa_blackwell(
    B, n_seed, num_heads, d_head, max_seq_len, cand_to_user_ratio
):
    query, key, value, idx = prepare_inputs(
        B,
        n_seed,
        num_heads,
        d_head,
        max_seq_len,
        cand_to_user_ratio,
        cand_to_user_ratio,
    )
    plan = plan_flash_attn_ikbo_blackwell(idx, n_seed, num_heads, max_seq_len)
    # ratio 4 / 10 exercise the 1-CTA kernel, ratio 70 the 2-CTA kernel (with the tails-last plan from 1024 on)
    assert plan.use_2cta == (cand_to_user_ratio >= 20)
    out = tlx_flash_attn_ikbo_blackwell(
        query, key, value, idx, n_seed, max_seq_len, plan=plan
    )
    out = out.view(B, n_seed, num_heads, d_head).permute(0, 2, 1, 3)
    ref = broadcast_sdpa(query, key, value, idx, n_seed, num_heads, d_head, max_seq_len)
    torch.testing.assert_close(ref, out, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("max_seq_len", [512, 4096])
def test_tlx_ikbo_fa_blackwell_no_plan(max_seq_len):
    query, key, value, idx = prepare_inputs(1024, 64, 2, 128, max_seq_len, 64, 64)
    out = tlx_flash_attn_ikbo_blackwell(query, key, value, idx, 64, max_seq_len)
    out = out.view(1024, 64, 2, 128).permute(0, 2, 1, 3)
    ref = broadcast_sdpa(query, key, value, idx, 64, 2, 128, max_seq_len)
    torch.testing.assert_close(ref, out, atol=1e-3, rtol=1e-4)


# ---------------------------------------------------------------------------------------------------------------------
# End-to-end accuracy beyond the upstream grid. Criterion as in ikbo_lce_test.py: the kernel's max error vs an fp32
# ground truth must stay within ERROR_MULTIPLIER x SDPA's max error (in the input dtype) vs the same ground truth.
ERROR_MULTIPLIER = 2.0
ERROR_FLOOR = 2e-3


def fp32_attention(
    query, key, value, cand_to_user_index, n_seed, max_seq_len, chunk=256
):
    B = cand_to_user_index.numel()
    H, D = query.shape[1], query.shape[2]
    q = query.view(B, n_seed, H, D)
    k = key.view(-1, max_seq_len, H, D)
    v = value.view(-1, max_seq_len, H, D)
    out = torch.empty((B, n_seed, H, D), device=query.device, dtype=torch.float32)
    for c0 in range(0, B, chunk):
        idx = cand_to_user_index[c0 : c0 + chunk].long()
        qc = q[c0 : c0 + chunk].float().permute(0, 2, 1, 3)
        kc = k[idx].float().permute(0, 2, 1, 3)
        vc = v[idx].float().permute(0, 2, 1, 3)
        p = torch.softmax(torch.matmul(qc, kc.transpose(-1, -2)) * D**-0.5, -1)
        out[c0 : c0 + chunk] = torch.matmul(p, vc).permute(0, 2, 1, 3)
    return out.view(B * n_seed, H, D)


def check_vs_fp32(out, query, key, value, cand_to_user_index, n_seed, max_seq_len):
    H, D = query.shape[1], query.shape[2]
    ref = fp32_attention(query, key, value, cand_to_user_index, n_seed, max_seq_len)
    sdpa = (
        broadcast_sdpa(query, key, value, cand_to_user_index, n_seed, H, D, max_seq_len)
        .permute(0, 2, 1, 3)
        .reshape(-1, H, D)
        .float()
    )
    baseline_err = (sdpa - ref).abs().max().item()
    kernel_err = (out.float() - ref).abs().max().item()
    threshold = max(ERROR_MULTIPLIER * baseline_err, ERROR_FLOOR)
    assert (
        kernel_err <= threshold
    ), f"kernel error {kernel_err:.4e} > {ERROR_MULTIPLIER}x SDPA error {baseline_err:.4e}"


def inputs_from_counts(
    counts, n_seed, num_heads, max_seq_len, dtype=DTYPE, dist="randn", device=DEVICE
):
    B = sum(counts)
    idx = (
        torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
        .int()
        .to(device)
    )
    q = torch.randn((B * n_seed, num_heads, 128), device=device)
    k = torch.randn((len(counts) * max_seq_len, num_heads, 128), device=device)
    v = torch.randn((len(counts) * max_seq_len, num_heads, 128), device=device)
    if dist == "sharp":  # logits std ~8: near one-hot rows
        q *= 8
    elif dist == "flat":  # near-uniform rows over the whole sequence
        q *= 0.02
    elif dist == "v_offset":  # output ~8 + small
        v += 8.0
    elif dist == "huge_logits":  # logits up to ~+-300
        q *= 30
    return q.to(dtype), k.to(dtype), v.to(dtype), idx


def random_counts(B, lo, hi, empty_frac=0.0, seed=0):
    rng = random.Random(seed)
    counts = []
    while sum(counts) < B:
        if rng.random() < empty_frac:
            counts.append(0)
        else:
            counts.append(min(rng.randint(lo, hi), B - sum(counts)))
    return counts


def all_kernels(query, key, value, idx, n_seed, max_seq_len):
    """The public entry point plus both kernels forced, natural and tails-last tile order."""
    H = query.shape[1]
    t1 = make_tile_table(idx, n_seed, H, max_seq_len, block_m=256)
    t2 = make_tile_table(idx, n_seed, H, max_seq_len, block_m=512)
    full = t2[:, 2] > 256
    t2_tails_last = torch.cat([t2[full], t2[~full]], 0).contiguous()
    return {
        "public": tlx_flash_attn_ikbo_blackwell(
            query, key, value, idx, n_seed, max_seq_len
        ),
        "1cta": ikbo_fa_1cta(query, key, value, max_seq_len, t1),
        "2cta": ikbo_fa_2cta(query, key, value, max_seq_len, t2),
        "2cta_tails_last": ikbo_fa_2cta(
            query, key, value, max_seq_len, t2_tails_last, clc_lead=4
        ),
    }


LAYOUTS = {
    "one_cand_per_user": lambda: [1] * 512,
    "1_to_8": lambda: random_counts(512, 1, 8),
    "2_to_3": lambda: random_counts(512, 2, 3),
    "7_to_9": lambda: random_counts(512, 7, 9),
    "15_to_17": lambda: random_counts(512, 15, 17),
    "single_user": lambda: [512],
    "1_to_300": lambda: random_counts(512, 1, 300),
    "with_empty_users": lambda: [0, 0]
    + random_counts(512, 1, 80, empty_frac=0.3)
    + [0],
}


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("max_seq_len", [1, 129, 640, 1000, 4097])
def test_tlx_ikbo_fa_blackwell_layouts(layout, max_seq_len):
    """Irregular candidate layouts (full / half / partial tiles of both tilings, users without candidates) and KV tail
    classes (single row, 1 row past a block, long), through every kernel path."""
    query, key, value, idx = inputs_from_counts(LAYOUTS[layout](), 64, 2, max_seq_len)
    for out in all_kernels(query, key, value, idx, 64, max_seq_len).values():
        check_vs_fp32(out, query, key, value, idx, 64, max_seq_len)


@pytest.mark.parametrize("num_heads", [3, 8])
@pytest.mark.parametrize("n_seed", [128, 192, 256])
def test_tlx_ikbo_fa_blackwell_heads_seeds(num_heads, n_seed):
    query, key, value, idx = inputs_from_counts(
        random_counts(256, 1, 40), n_seed, num_heads, 1000
    )
    for out in all_kernels(query, key, value, idx, n_seed, 1000).values():
        check_vs_fp32(out, query, key, value, idx, n_seed, 1000)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dist", ["randn", "sharp", "flat", "v_offset", "huge_logits"])
@pytest.mark.parametrize("cands", [(63, 65), (2, 9)])
def test_tlx_ikbo_fa_blackwell_dtypes(dtype, dist, cands):
    query, key, value, idx = inputs_from_counts(
        random_counts(512, *cands), 64, 2, 2048, dtype=dtype, dist=dist
    )
    plan = plan_flash_attn_ikbo_blackwell(idx, 64, 2, 2048)
    out = tlx_flash_attn_ikbo_blackwell(query, key, value, idx, 64, 2048, plan=plan)
    check_vs_fp32(out, query, key, value, idx, 64, 2048)


@pytest.mark.parametrize("cands", [(63, 65), (2, 5)])
def test_tlx_ikbo_fa_blackwell_deterministic_and_graph(cands):
    query, key, value, idx = inputs_from_counts(random_counts(512, *cands), 64, 2, 4096)
    plan = plan_flash_attn_ikbo_blackwell(idx, 64, 2, 4096)
    o1 = tlx_flash_attn_ikbo_blackwell(query, key, value, idx, 64, 4096, plan=plan)
    o2 = tlx_flash_attn_ikbo_blackwell(query, key, value, idx, 64, 4096, plan=plan)
    assert torch.equal(o1, o2)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = tlx_flash_attn_ikbo_blackwell(query, key, value, idx, 64, 4096, plan=plan)
    for i in range(3):
        torch.manual_seed(10 + i)
        for t in (query, key, value):
            t.copy_(torch.randn_like(t))
        g.replay()
        check_vs_fp32(out, query, key, value, idx, 64, 4096)


@pytest.mark.parametrize("cands", [(63, 65), (2, 5), (1, 1)])
def test_tlx_ikbo_fa_blackwell_no_out_of_bounds_writes(cands):
    query, key, value, idx = inputs_from_counts(random_counts(300, *cands), 64, 2, 1000)
    pad = 512
    buf = torch.full(
        (query.shape[0] + 2 * pad, 2, 128), 7.0, device=DEVICE, dtype=DTYPE
    )
    out = buf[pad : pad + query.shape[0]]
    for kernel, block_m in [(ikbo_fa_1cta, 256), (ikbo_fa_2cta, 512)]:
        buf.fill_(float("nan"))
        buf[:pad] = 7.0
        buf[pad + query.shape[0] :] = 7.0
        kernel(
            query,
            key,
            value,
            1000,
            make_tile_table(idx, 64, 2, 1000, block_m=block_m),
            out=out,
        )
        assert bool((buf[:pad] == 7.0).all()) and bool(
            (buf[pad + query.shape[0] :] == 7.0).all()
        )
        assert bool(torch.isfinite(out).all()), "not every output row was written"
        check_vs_fp32(out, query, key, value, idx, 64, 1000)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_tlx_ikbo_fa_blackwell_multi_device():
    for d in [1, 0]:
        dev = torch.device(DEVICE, d)
        with torch.cuda.device(dev):
            for cands in [(63, 65), (2, 5)]:
                query, key, value, idx = inputs_from_counts(
                    random_counts(512, *cands), 64, 2, 1000, device=dev
                )
                out = tlx_flash_attn_ikbo_blackwell(query, key, value, idx, 64, 1000)
                assert out.device == dev
                check_vs_fp32(out, query, key, value, idx, 64, 1000)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_tlx_ikbo_fa_blackwell_inputs_on_non_current_device():
    """Inputs on cuda:1 while cuda:0 is current and a side stream is current on cuda:1 (both kernels)."""
    dev = torch.device(DEVICE, 1)
    side = torch.cuda.Stream(device=dev)
    prev_stream = torch.cuda.current_stream(dev)
    prev_device = torch.cuda.current_device()
    for cands in [(63, 65), (2, 5)]:
        query, key, value, idx = inputs_from_counts(
            random_counts(512, *cands), 64, 2, 1000, device=dev
        )
        side.wait_stream(torch.cuda.current_stream(dev))
        try:
            torch.cuda.set_stream(side)  # also makes cuda:1 current
            torch.cuda.set_device(0)
            out = tlx_flash_attn_ikbo_blackwell(query, key, value, idx, 64, 1000)
            assert out.device == dev
            check_vs_fp32(out, query, key, value, idx, 64, 1000)
        finally:
            torch.cuda.set_stream(prev_stream)
            torch.cuda.set_device(prev_device)


def test_tlx_ikbo_fa_blackwell_plan_with_host_counts():
    """num_cands_per_user may be a CPU tensor (precomputed host counts)."""
    counts = random_counts(512, 1, 40)
    query, key, value, idx = inputs_from_counts(counts, 64, 2, 1000)
    plan = plan_flash_attn_ikbo_blackwell(
        idx, 64, 2, 1000, num_cands_per_user=torch.tensor(counts)
    )
    assert torch.equal(
        plan.tile_table, plan_flash_attn_ikbo_blackwell(idx, 64, 2, 1000).tile_table
    )
    out = tlx_flash_attn_ikbo_blackwell(query, key, value, idx, 64, 1000, plan=plan)
    check_vs_fp32(out, query, key, value, idx, 64, 1000)


def test_tlx_ikbo_fa_blackwell_rejects_unsorted_index():
    idx = torch.tensor([1, 1, 0, 0, 0], dtype=torch.int32, device=DEVICE)
    with pytest.raises(AssertionError, match="sorted"):
        plan_flash_attn_ikbo_blackwell(idx, 64, 2, 512)


def test_tlx_ikbo_fa_blackwell_rejects_mismatched_plan():
    query, key, value, idx = inputs_from_counts(random_counts(256, 1, 40), 64, 2, 1000)
    plan = plan_flash_attn_ikbo_blackwell(idx, 64, 2, 1000)
    with pytest.raises(AssertionError, match="different layout"):
        tlx_flash_attn_ikbo_blackwell(query, key, value, idx, 64, 500, plan=plan)
    q4 = torch.randn((query.shape[0], 4, 128), device=DEVICE, dtype=DTYPE)
    k4 = torch.randn((key.shape[0], 4, 128), device=DEVICE, dtype=DTYPE)
    with pytest.raises(AssertionError, match="different layout"):
        tlx_flash_attn_ikbo_blackwell(q4, k4, k4, idx, 64, 1000, plan=plan)
    with pytest.raises(AssertionError, match="mixed dtypes"):
        tlx_flash_attn_ikbo_blackwell(
            query, key.bfloat16(), value, idx, 64, 1000, plan=plan
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

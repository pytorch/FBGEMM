# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Accuracy tests for the Blackwell TLX IKBO LCE kernel. Same philosophy as ikbo_lce_test.py: the kernel's error vs an
# fp32 ground truth must not exceed cuBLAS fp16's error vs the same ground truth.

import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip("CUDA GPU unavailable", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("Blackwell (sm_100 / sm_103) only", allow_module_level=True)

import ikbo.ops.tlx_ikbo_lce_blackwell as lce_module
from ikbo.ops.tlx_ikbo_lce_blackwell import pack_weight, tlx_ikbo_lce_blackwell
from ikbo.ops.torch_lce import torch_decomposed_lce

DEVICE = "cuda"
DTYPE = torch.float16
PAD_UNIT = 8
ERROR_MULTIPLIER = 1.0
ERROR_FLOOR = 1e-4


def _prepare_inputs(B, M, N, K_USER, K_CAND, cand_to_user_ratio):
    k_user = ((K_USER + PAD_UNIT - 1) // PAD_UNIT) * PAD_UNIT
    k_cand = ((K_CAND + PAD_UNIT - 1) // PAD_UNIT) * PAD_UNIT
    num_users = (B + cand_to_user_ratio - 1) // cand_to_user_ratio
    cand_to_user_index = (torch.arange(B, device=DEVICE) // cand_to_user_ratio).int()
    cw_cand = torch.randn((M, k_cand), device=DEVICE, dtype=DTYPE)
    cw_user = torch.randn((M, k_user), device=DEVICE, dtype=DTYPE)
    e_cand = torch.randn((B, k_cand, N), device=DEVICE, dtype=DTYPE)
    e_user = torch.randn((num_users, k_user, N), device=DEVICE, dtype=DTYPE)
    return cw_cand, cw_user, e_cand, e_user, cand_to_user_index


def _fp32_reference(inputs):
    """fp32 ground truth, the cuBLAS fp16 baseline's max error vs it, and the allowed kernel error."""
    cw_c, cw_u, e_c, e_u, idx = inputs
    ref_fp32 = torch_decomposed_lce(
        cw_c.float(), cw_u.float(), e_c.float(), e_u.float(), idx
    )
    ref_fp16 = torch_decomposed_lce(cw_c, cw_u, e_c, e_u, idx)
    baseline_err = (ref_fp16.float() - ref_fp32).abs().max().item()
    return ref_fp32, baseline_err, max(ERROR_MULTIPLIER * baseline_err, ERROR_FLOOR)


def _check_vs_fp32(out, inputs):
    ref_fp32, baseline_err, threshold = _fp32_reference(inputs)
    kernel_err = (out.float() - ref_fp32).abs().max().item()
    assert (
        kernel_err <= threshold
    ), f"error exceeds {ERROR_MULTIPLIER}x cuBLAS fp16 baseline: kernel={kernel_err:.4e}, baseline={baseline_err:.4e}"


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("B", [512, 1024])
@pytest.mark.parametrize("M", [128, 433])
@pytest.mark.parametrize("N", [256])
@pytest.mark.parametrize("K_USER", [1024, 1025, 1184])
@pytest.mark.parametrize("K_CAND", [1023, 1024, 872])
@pytest.mark.parametrize("cand_to_user_ratio", [10, 70, 100, 1000])
def test_tlx_ikbo_lce_blackwell(B, M, N, K_USER, K_CAND, cand_to_user_ratio):
    inputs = _prepare_inputs(B, M, N, K_USER, K_CAND, cand_to_user_ratio)
    _check_vs_fp32(tlx_ikbo_lce_blackwell(*inputs), inputs)


def test_tlx_ikbo_lce_blackwell_packed_weights():
    inputs = _prepare_inputs(1024, 433, 256, 1178, 866, 70)
    cw_c, cw_u, *rest = inputs
    _check_vs_fp32(
        tlx_ikbo_lce_blackwell(pack_weight(cw_c), pack_weight(cw_u), *rest), inputs
    )


def test_tlx_ikbo_lce_blackwell_repeated_calls():
    """The kernel's cross-call counters live in a cached workspace: repeated calls on the same shapes with new data,
    and calls that change the user count (workspace eviction), must stay correct."""
    for i, ratio in enumerate([70, 70, 10, 20, 30, 40, 50, 60, 80, 100, 128, 70, 10]):
        torch.manual_seed(i)
        inputs = _prepare_inputs(512, 433, 256, 1024, 872, ratio)
        _check_vs_fp32(tlx_ikbo_lce_blackwell(*inputs), inputs)


# ---------------------------------------------------------------------------------------------------------------------
# End-to-end accuracy beyond the upstream grid: tile-boundary shapes, irregular user layouts, dtypes / value ranges,
# CUDA graphs and devices. Same criterion: max error vs fp32 <= cuBLAS's max error (in the input dtype) vs fp32.


def _inputs_from_counts(
    M, K_USER, K_CAND, counts, dtype=DTYPE, dist="randn", shuffle=False, device=DEVICE
):
    B = sum(counts)
    idx = (
        torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
        .int()
        .to(device)
    )
    if shuffle:  # the kernel does not need candidates grouped by user
        idx = idx[torch.randperm(B, device=device)]
    cw_cand = torch.randn((M, K_CAND), device=device)
    cw_user = torch.randn((M, K_USER), device=device)
    e_cand = torch.randn((B, K_CAND, 256), device=device)
    e_user = torch.randn((len(counts), K_USER, 256), device=device)
    if dist == "init":  # W ~ N(0, 1/K), embeddings ~ N(0, 0.05^2)
        cw_cand /= K_CAND**0.5
        cw_user /= K_USER**0.5
        e_cand *= 0.05
        e_user *= 0.05
    elif dist == "offset":  # correlated, non-cancelling sums
        cw_cand = cw_cand.abs() / K_CAND**0.5
        cw_user = cw_user.abs() / K_USER**0.5
        e_cand += 1.0
        e_user += 1.0
    return tuple(t.to(dtype) for t in (cw_cand, cw_user, e_cand, e_user)) + (idx,)


def _random_counts(B, lo, hi, empty_frac=0.0, seed=0):
    rng = random.Random(seed)
    counts = []
    while sum(counts) < B:
        if rng.random() < empty_frac:
            counts.append(0)
        else:
            counts.append(min(rng.randint(lo, hi), B - sum(counts)))
    return counts


@pytest.mark.parametrize("M", [1, 64, 65, 192, 257, 513, 1000])
@pytest.mark.parametrize("K_USER,K_CAND", [(8, 8), (1184, 56), (64, 4096)])
def test_tlx_ikbo_lce_blackwell_tile_edges(M, K_USER, K_CAND):
    """Every last-m-tile class (64 / 128 / 192 / 256 columns, split into power-of-two MMA pieces), tiny M, single and
    partial K steps, long K."""
    inputs = _inputs_from_counts(M, K_USER, K_CAND, _random_counts(256, 20, 40))
    _check_vs_fp32(tlx_ikbo_lce_blackwell(*inputs), inputs)


@pytest.mark.parametrize(
    "layout",
    [
        "one_cand_per_user",
        "single_user",
        "random_1_100",
        "with_empty_users",
        "shuffled",
        "B1",
        "B3",
    ],
)
def test_tlx_ikbo_lce_blackwell_layouts(layout):
    counts, shuffle = {
        "one_cand_per_user": ([1] * 1024, False),
        "single_user": ([1024], False),
        "random_1_100": (_random_counts(1024, 1, 100), False),
        "with_empty_users": (
            [0, 0] + _random_counts(1024, 1, 50, empty_frac=0.3) + [0],
            False,
        ),
        "shuffled": (_random_counts(1024, 1, 90), True),
        "B1": ([1], False),
        "B3": ([2, 1], False),
    }[layout]
    inputs = _inputs_from_counts(433, 1184, 872, counts, shuffle=shuffle)
    _check_vs_fp32(tlx_ikbo_lce_blackwell(*inputs), inputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dist", ["randn", "init", "offset"])
@pytest.mark.parametrize("M", [256, 433])
def test_tlx_ikbo_lce_blackwell_dtypes(dtype, dist, M):
    inputs = _inputs_from_counts(
        M, 1184, 872, _random_counts(1024, 60, 80), dtype=dtype, dist=dist
    )
    _check_vs_fp32(tlx_ikbo_lce_blackwell(*inputs), inputs)


def test_tlx_ikbo_lce_blackwell_deterministic():
    inputs = _inputs_from_counts(433, 1184, 872, _random_counts(1024, 1, 100))
    assert torch.equal(tlx_ikbo_lce_blackwell(*inputs), tlx_ikbo_lce_blackwell(*inputs))


def test_tlx_ikbo_lce_blackwell_cuda_graph():
    """Graph replays read the cross-call counters from device memory: replays with new data, interleaved with eager
    calls on the same workspace, must stay correct."""
    static = list(_inputs_from_counts(433, 1184, 872, _random_counts(512, 1, 100)))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        tlx_ikbo_lce_blackwell(*static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = tlx_ikbo_lce_blackwell(*static)
    for i in range(6):
        torch.manual_seed(100 + i)
        for t in static[:4]:
            t.copy_(torch.randn_like(t))
        if i % 2:
            _check_vs_fp32(tlx_ikbo_lce_blackwell(*static), static)
        g.replay()
        _check_vs_fp32(out, static)
    for _ in range(50):
        g.replay()
    _check_vs_fp32(out, static)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_tlx_ikbo_lce_blackwell_multi_device():
    for d in [1, 0, 1]:
        dev = torch.device(DEVICE, d)
        with torch.cuda.device(dev):
            inputs = _inputs_from_counts(
                433, 1184, 872, _random_counts(512, 1, 100, seed=d), device=dev
            )
            out = tlx_ikbo_lce_blackwell(*inputs)
            assert out.device == dev
            _check_vs_fp32(out, inputs)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_tlx_ikbo_lce_blackwell_inputs_on_non_current_device():
    """Inputs on cuda:1 while cuda:0 is current and a side stream is current on cuda:1: the kernel must run on
    cuda:1's current stream, ordered with the work that reads its output there."""
    dev = torch.device(DEVICE, 1)
    inputs = _inputs_from_counts(
        433, 1184, 872, _random_counts(512, 1, 100), device=dev
    )
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    prev_stream = torch.cuda.current_stream(dev)
    prev_device = torch.cuda.current_device()
    try:
        torch.cuda.set_stream(side)  # also makes cuda:1 current
        torch.cuda.set_device(0)
        out = tlx_ikbo_lce_blackwell(*inputs)
        assert out.device == dev
        _check_vs_fp32(out, inputs)  # runs on side, cuda:1's current stream
    finally:
        torch.cuda.set_stream(prev_stream)
        torch.cuda.set_device(prev_device)


def test_tlx_ikbo_lce_blackwell_bad_index_gives_nan():
    """Out-of-range user indices must not hang the kernel (candidates wait on per-user flags): their rows are NaN,
    every other row is exact."""
    inputs = list(_inputs_from_counts(433, 1184, 872, _random_counts(512, 1, 100)))
    U = inputs[3].shape[0]
    bad = torch.tensor([3, 100, 511])
    good = torch.ones(512, dtype=torch.bool)
    good[bad] = False
    inputs[4] = inputs[4].clone()
    inputs[4][bad.to(DEVICE)] = torch.tensor(
        [U, -1, 1 << 30], dtype=torch.int32, device=DEVICE
    )
    out = tlx_ikbo_lce_blackwell(*inputs)
    assert bool(torch.isnan(out[bad.to(DEVICE)]).all())
    sub = inputs[:2] + [
        inputs[2][good.to(DEVICE)],
        inputs[3],
        inputs[4][good.to(DEVICE)],
    ]
    _check_vs_fp32(out[good.to(DEVICE)], sub)


def test_tlx_ikbo_lce_blackwell_streams():
    """Calls from two streams share the device workspace and are serialized on it."""
    inputs = [
        _inputs_from_counts(433, 1184, 872, _random_counts(512, 1, 100, seed=i))
        for i in range(2)
    ]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    outs = []
    for i in range(8):
        with torch.cuda.stream(streams[i % 2]):
            outs.append(tlx_ikbo_lce_blackwell(*inputs[i % 2]))
    torch.cuda.synchronize()
    for i, out in enumerate(outs):
        _check_vs_fp32(out, inputs[i % 2])


def test_tlx_ikbo_lce_blackwell_threads():
    """Host threads on different streams share the device workspace. Small grids with long K let kernels from two
    streams run at the same time, and a short thread switch interval makes threads interleave inside the launch
    path; without the per-device launch lock this gives wrong outputs within a few thousand calls.
    """
    n_threads, calls = 4, 2000
    inputs = [
        _prepare_inputs(8 + 4 * t, 200, 256, 4096, 4096, 4) for t in range(n_threads)
    ]
    refs = [_fp32_reference(x) for x in inputs]
    for x in inputs:  # compile and size the workspace before the threads start
        tlx_ikbo_lce_blackwell(*x)
    torch.cuda.synchronize()
    start = threading.Barrier(n_threads)

    def worker(t):
        ref_fp32, _, _ = refs[t]
        stream = torch.cuda.Stream(device=ref_fp32.device)
        stream.wait_stream(torch.cuda.default_stream(ref_fp32.device))
        start.wait()
        worst = 0.0
        with torch.cuda.stream(stream):
            for _ in range(calls // 50):
                outs = [tlx_ikbo_lce_blackwell(*inputs[t]) for _ in range(50)]
                errs = torch.stack([(o.float() - ref_fp32).abs().max() for o in outs])
                worst = max(worst, errs.max().item())
        return worst

    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        with ThreadPoolExecutor(n_threads) as pool:
            worst = list(pool.map(worker, range(n_threads)))
    finally:
        sys.setswitchinterval(switch_interval)
    for t in range(n_threads):
        assert worst[t] <= refs[t][2], f"thread {t}: max error {worst[t]:.4e}"


def test_tlx_ikbo_lce_blackwell_capture_needs_warm_workspace():
    inputs = _inputs_from_counts(433, 1184, 872, _random_counts(512, 1, 100))
    lce_module._workspaces.clear()
    g = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="before capturing"):
        with torch.cuda.graph(g):
            tlx_ikbo_lce_blackwell(*inputs)
    _check_vs_fp32(tlx_ikbo_lce_blackwell(*inputs), inputs)


def test_tlx_ikbo_lce_blackwell_graph_survives_workspace_growth():
    """A captured graph keeps using its workspace after a larger eager call replaces the device workspace."""
    static = list(_inputs_from_counts(433, 1184, 872, _random_counts(256, 1, 100)))
    tlx_ikbo_lce_blackwell(*static)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = tlx_ikbo_lce_blackwell(*static)
    big = _inputs_from_counts(1000, 1184, 872, [1] * 600)  # grows the workspace
    _check_vs_fp32(tlx_ikbo_lce_blackwell(*big), big)
    torch.cuda.empty_cache()
    for i in range(3):
        torch.manual_seed(50 + i)
        for t in static[:4]:
            t.copy_(torch.randn_like(t))
        g.replay()
        _check_vs_fp32(out, static)
        _check_vs_fp32(tlx_ikbo_lce_blackwell(*big), big)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

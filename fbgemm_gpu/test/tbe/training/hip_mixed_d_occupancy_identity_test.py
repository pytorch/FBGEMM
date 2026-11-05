# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Bit-identity test for TBE_ROCM_HIP_MIXED_D_OCCUPANCY (permanent gfx950 test next to the <3, 8> pin).

The occupancy route runs the hip_mixed_d warp backward as <..., 3, 8, false> instead of <..., 4, 8, false> for tables
with max_D <= 96. The rowwise-Adagrad update spells out the sum of squares (a packed float2 FMA chain, contraction off)
in <3, 8> and in the vecs 0-2 that <4, 8> shares with it (genscript/optimizers.py), so the two must update weights
and momentum bit for bit on any toolchain. test_default_route_golden also pins both routes to digests recorded on the
internal toolchain before that change, so neither path's numerics can move unnoticed.

Each route runs in its own spawned process (the gates are function-local statics read once per process) with the same
seed, 3 successive forward + backward + update steps per case, the target types (FP16 weights and cache, BF16 output
and grad, int32 indices: the target kernel is <c10::Half, c10::BFloat16, c10::Half, int, 3|4, 8, false>), and returns weights, momentum and the backward kernel names. Cases: a mixed-D table set (96 x 6,
24 x 2, 8 x 3, 4 x 3; scaled down in rows) with SR on / off and high / low duplication, D {32, 64, 96, 96}, and a
max_D 128 control; two of the mixed-D cases also run max_norm > 0 (the weight-norm sum of squares is pinned too), and
one runs the weighted TBE (per-sample weights), whose hip_mixed_d <3, 8> instantiation carries the same pin.
"""

import hashlib
import json
import multiprocessing as mp
import os
import unittest

os.environ.setdefault("FBGEMM_NO_JK", "1")

import torch  # noqa: E402

MIXED_DIMS = (
    [96] * 6 + [24] * 2 + [8] * 3 + [4] * 3
)  # mixed D: 6 x 96, 2 x 24, 3 x 8, 3 x 4
B = 512
LR, EPS = 0.021, 1e-4
SEED = 2026
STEPS = 3
MAX_NORM = (
    0.1  # below the D = 96 row norms of the init (~0.28), above the D = 4 ones (~0.06)
)
GATED = {
    "mix_sr_highdup": (MIXED_DIMS, True, False),
    "mix_nosr_highdup": (MIXED_DIMS, False, False),
    "mix_sr_lowdup": (MIXED_DIMS, True, True),
    "mix_nosr_lowdup": (MIXED_DIMS, False, True),
    "d32_64_96_96_sr": ([32, 64, 96, 96], True, False),
    # max_norm > 0 runs the weight-norm renormalisation, whose weight sum of squares is pinned the same way.
    "mix_sr_maxnorm": (MIXED_DIMS, True, False, MAX_NORM),
    "mix_nosr_maxnorm": (MIXED_DIMS, False, False, MAX_NORM),
    # The warp template emits hip_mixed_d for the weighted TBE too; its <3, 8> instantiation carries the same pin.
    "mix_sr_weighted": (MIXED_DIMS, True, False, 0.0, True),
}
CONTROL = {"d64_64_128_128_sr_control": ([64, 64, 128, 128], True, False)}
# Golden only (default route A): the mixed-D group at a production-like batch and 10x the rows.
PRODSHAPE = {"prodshape_mix_sr": (MIXED_DIMS, True, False, 0.0, False, 4096, 10)}
# Default route (A, pin off) digests recorded from the internal build BEFORE the <4, 8> pattern change, keyed by
# torch.version.hip. Another toolchain (e.g. OSS ROCm) has no entry and skips; re-record on a ROCm upgrade.
GOLDEN_SOURCE = (
    "testrun 844425476272633 at 18f422a3c0d5 (parent 7a6c337d06da, identity-test-only change), "
    "route A = pin off, recorded before the <4, 8> pattern change"
)
GOLDEN: dict = {
    "7.0.51831": {
        "d32_64_96_96_sr": "a40175c69e96efec8318f70d94f17c5e5229fb144e560e7aa341901f28892ac0",
        "d64_64_128_128_sr_control": "93afb075e815780549b705e7e6d7dfc9b2cda8f95959fe65c3db99af35f9ac5c",
        "mix_nosr_highdup": "f0e982f62f9240427702c022a46dccde0723c50b7862182eb0714b912df70326",
        "mix_nosr_lowdup": "cccde3f9030406a52acace2c981e792604e27ff75ca19d99354a16fad98c6fb5",
        "mix_nosr_maxnorm": "89b958db5d8b0eff7730f7741dc78b161fd1fa804a73750f3ea0d1e78eed3d9a",
        "mix_sr_highdup": "446d2141b8ca6b5190211486f534ca5c602c544370b38e66268747692df0a495",
        "mix_sr_lowdup": "a4a9df84cfee45e756381701f4419b689a5b1fec86289959718c236339132ce5",
        "mix_sr_maxnorm": "57d1ced839b1db4c14ee9d8a6bcf4b2561f0b4f586b420222b1c698e752a36c6",
        "mix_sr_weighted": "676645c95080f1f50bfcc4aac81ea8f992e2d9e1bd85be50bab45f3c0788f5d0",
        "prodshape_mix_sr": "1c91902bf24a93d44758feb2e7875f7a0143fbdc1ef9bc9d62e138772dd34a3c",
    },
}


def _rows(d, scale=1):
    return (2000 if d >= 32 else 400) * scale


# The target instantiation: FP16 weights and cache, BF16 grad, int32 indices.
V4 = "c10::Half, c10::BFloat16, c10::Half, int, 4, 8, false>"
V3 = "c10::Half, c10::BFloat16, c10::Half, int, 3, 8, false>"
WEIGHTED_CASES = {"mix_sr_weighted"}


def kernel_kind_ok(case, names):
    """Weighted cases must run the weighted kernel only, every other case the unweighted kernel only."""
    if case in WEIGHTED_CASES:
        return bool(names) and all(
            "_weighted_kernel_warp_per_row_1" in n and "unweighted" not in n
            for n in names
        )
    return bool(names) and all("unweighted_kernel_warp_per_row_1" in n for n in names)


WARP_PREFIX = "hip_mixed_d_split_embedding_backward_codegen_rowwise_adagrad_"
WARP_SUFFIX = "weighted_kernel_warp_per_row_1<"  # matches both unweighted and weighted


def is_warp(name):
    """True for the hip_mixed_d rowwise-Adagrad warp backward kernel, weighted or unweighted. Profiler names carry
    the return type ('void hip_mixed_d_...'), so match by substring."""
    return WARP_PREFIX in name and WARP_SUFFIX in name


REQUIRED_ARCH = "gfx950"  # MI350: the pin was derived and verified on this arch
REQUIRE_ENV = "HIP_MIXED_D_REQUIRE_GFX950"


def gate_decision(arch: str, require: str) -> str:
    """'run' on gfx950. Elsewhere 'skip', unless HIP_MIXED_D_REQUIRE_GFX950=1 asks to run anyway (the route
    processes then fail on the arch check, so a wrong device cannot pass silently)."""
    if arch.startswith(REQUIRED_ARCH) or require == "1":
        return "run"
    return "skip"


def _rocm() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def _route_env(route: str) -> None:
    os.environ["FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL"] = "1"
    os.environ["TBE_ROCM_HIP_MIXED_D_EIGHTH_WAVE"] = "1"
    os.environ["TBE_ROCM_HIP_MIXED_D_UNIFORM"] = "1"
    if route == "B":
        os.environ["TBE_ROCM_HIP_MIXED_D_OCCUPANCY"] = "1"
    else:
        os.environ.pop("TBE_ROCM_HIP_MIXED_D_OCCUPANCY", None)


def _inputs(dims, low_dup, step, batch=B, rows_scale=1):
    g = torch.Generator().manual_seed(7 + step)
    lens, ids, grads = [], [], []
    for f in range(len(dims)):
        if low_dup:
            L = torch.randint(0, 4, (batch,), generator=g)
            u = torch.rand(int(L.sum()), generator=g, dtype=torch.float64)
        else:
            L = torch.randint(0, 9, (batch,), generator=g)
            L[::13] = 48
            u = torch.rand(int(L.sum()), generator=g, dtype=torch.float64) ** 3
        lens.append(L)
        rows = _rows(dims[f], rows_scale)
        ids.append((u * rows).long().clamp_(max=rows - 1))
        grads.append(
            torch.randn((batch, dims[f]), generator=g)
        )  # arbitrary FP32: every rounding point is exercised
    return lens, ids, grads


def _run_case(dims, sr, low_dup, max_norm=0.0, weighted=False, batch=B, rows_scale=1):
    from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
    from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
        BoundsCheckMode,
        EmbeddingLocation,
        PoolingMode,
    )
    from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
        ComputeDevice,
        SplitTableBatchedEmbeddingBagsCodegen,
    )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tbe = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[
            (r, d, EmbeddingLocation.DEVICE, ComputeDevice.CUDA)
            for r, d in zip([_rows(x, rows_scale) for x in dims], dims)
        ],
        optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        weights_precision=SparseType.FP16,
        output_dtype=SparseType.BF16,
        stochastic_rounding=sr,
        pooling_mode=PoolingMode.SUM,
        bounds_check_mode=BoundsCheckMode.FATAL,
        learning_rate=LR,
        eps=EPS,
        max_norm=max_norm,
        embedding_table_index_type=torch.int32,
        embedding_table_offset_type=torch.int32,
    ).cuda()
    with torch.no_grad():
        for t, w in enumerate(tbe.split_embedding_weights()):
            g = torch.Generator(device="cuda").manual_seed(1234 + t)
            w.copy_(
                (torch.rand(w.shape, generator=g, device="cuda") * 0.1 - 0.05).to(
                    w.dtype
                )
            )
    steps, names = [], set()
    for s in range(STEPS):
        lens, ids, grads = _inputs(dims, low_dup, s, batch, rows_scale)
        lengths = torch.cat(lens)
        offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
        offsets[1:] = torch.cumsum(lengths, 0)
        idx, off = torch.cat(ids).int().cuda(), offsets.int().cuda()
        # CPU generator on purpose: same per-sample weights in both route processes.
        gen = torch.Generator().manual_seed(99 + s)
        psw = None
        if weighted:
            psw = torch.rand(idx.numel(), generator=gen)
            psw = psw.cuda()  # noqa: CITRINE(redundant_cuda_to_device)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as p:
            out = tbe(idx, off, per_sample_weights=psw) if weighted else tbe(idx, off)
            out.backward(torch.cat(grads, dim=1).to(torch.bfloat16).cuda())
            torch.cuda.synchronize()
        names |= {e.name for e in p.events() if is_warp(e.name)}
        steps.append(
            {
                "weights": [
                    x.detach().cpu().clone() for x in tbe.split_embedding_weights()
                ],
                "momentum": [
                    [m.detach().cpu().clone() for m in st]
                    for st in tbe.split_optimizer_states()
                ],
            }
        )
    return {"steps": steps, "kernels": sorted(names)}


def _route_main(route, path):
    _route_env(route)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(
        f"route {route}: device {torch.cuda.get_device_name(0)} arch {arch}", flush=True
    )
    if not arch.startswith(REQUIRED_ARCH):
        raise RuntimeError(
            f"route {route}: device arch {arch}, need {REQUIRED_ARCH} (MI350); the measurement is void"
        )
    res = {c: _run_case(*v) for c, v in {**GATED, **CONTROL}.items()}
    if route == "A":
        res.update({c: _run_case(*v) for c, v in PRODSHAPE.items()})
    res["_arch"] = arch
    torch.save(res, path)


def digest(case_res) -> str:
    """sha256 over every step's weights and momentum bytes of one case."""
    h = hashlib.sha256()
    for st in case_res["steps"]:
        for w in st["weights"]:
            h.update(w.contiguous().numpy().tobytes())
        for table in st["momentum"]:
            for m in table:
                h.update(m.contiguous().numpy().tobytes())
    return h.hexdigest()


def ulp16(a, b):
    """Largest ulp distance between two FP16 tensors (sign-magnitude mapped to ordered integers)."""

    def ordered(x):
        i = x.view(torch.int16).int()
        return torch.where(i < 0, -32768 - i, i)

    return int((ordered(a) - ordered(b)).abs().max().item()) if a.numel() else 0


def first_diff(a, b):
    """Flat index of the first element whose raw bits differ (FP16 weights or FP32 momentum), or -1."""
    bits = torch.int16 if a.dtype == torch.float16 else torch.int32
    m = (a.view(bits) != b.view(bits)).flatten()
    nz = torch.nonzero(m)
    return int(nz[0].item()) if nz.numel() else -1


def signed(a, b):
    """Signed B - A (route B = <3, 8> arm minus route A = <4, 8> control) over the differing elements:
    (n_up, n_down, mean signed relative difference). Symmetric rounding gives n_up ~ n_down and a mean near 0.
    """
    af, bf = a.float(), b.float()
    m = af != bf
    if not bool(m.any()):
        return 0, 0, 0.0
    d = (bf - af)[m]
    rel = d / af[m].abs().clamp_min(1e-30)
    return (
        int((d > 0).sum().item()),
        int((d < 0).sum().item()),
        float(rel.mean().item()),
    )


def compare(a, b, cases):
    """Rows (case, step, tensor, identical, differing, numel, max ulp for weights / max rel for momentum,
    n_up, n_down, mean signed rel diff (B - A))."""
    rows = []
    for c in cases:
        for s, (sa, sb) in enumerate(zip(a[c]["steps"], b[c]["steps"])):
            for t, (wa, wb) in enumerate(zip(sa["weights"], sb["weights"])):
                diff = int((wa.view(torch.int16) != wb.view(torch.int16)).sum().item())
                rows.append(
                    (c, s, f"w{t}", diff == 0, diff, wa.numel(), ulp16(wa, wb))
                    + signed(wa, wb)
                )
            for t, (ma, mb) in enumerate(zip(sa["momentum"], sb["momentum"])):
                for k, (x, y) in enumerate(zip(ma, mb)):
                    diff = int(
                        (x.view(torch.int32) != y.view(torch.int32)).sum().item()
                    )
                    rel = (
                        float(((x - y).abs() / y.abs().clamp_min(1e-30)).max().item())
                        if diff
                        else 0.0
                    )
                    rows.append(
                        (c, s, f"m{t}.{k}", diff == 0, diff, x.numel(), rel)
                        + signed(x, y)
                    )
    return rows


def totals(rows):
    """Per tensor kind (w / m): total differing, total up, total down, and the up share."""
    out = {}
    for kind in ("w", "m"):
        r = [x for x in rows if x[2].startswith(kind)]
        up, down = sum(x[7] for x in r), sum(x[8] for x in r)
        out[kind] = (
            sum(x[4] for x in r),
            up,
            down,
            (up / (up + down)) if up + down else float("nan"),
        )
    return out


def _table(rows):
    return "\n".join(" | ".join(str(x) for x in r) for r in rows)


class HelpersTest(unittest.TestCase):
    def test_is_warp(self) -> None:
        self.assertTrue(
            is_warp(
                WARP_PREFIX
                + "unweighted_kernel_warp_per_row_1<c10::Half, c10::BFloat16, float, long, 3, 8, false>(...)"
            )
        )
        self.assertTrue(
            is_warp(
                WARP_PREFIX
                + "weighted_kernel_warp_per_row_1<c10::Half, c10::BFloat16, float, long, 4, 8, false>(...)"
            )
        )
        self.assertTrue(
            is_warp(
                "void "
                + WARP_PREFIX
                + "unweighted_kernel_warp_per_row_1<c10::Half, c10::BFloat16, c10::Half, int, 3, 8, false>(...)"
            )
        )  # the profiler's form
        self.assertFalse(
            is_warp(
                "split_embedding_backward_codegen_rowwise_adagrad_unweighted_kernel_warp_per_row_1<...>"
            )
        )
        self.assertFalse(is_warp(WARP_PREFIX + "unweighted_kernel_cta_per_row_1<...>"))

    def test_target_strings_and_kind(self) -> None:
        unw = "void " + WARP_PREFIX + "unweighted_kernel_warp_per_row_1<" + V3 + "(...)"
        wtd = "void " + WARP_PREFIX + "weighted_kernel_warp_per_row_1<" + V4 + "(...)"
        other = (
            "void "
            + WARP_PREFIX
            + "unweighted_kernel_warp_per_row_1<c10::Half, c10::BFloat16, float, long, 3, 8, false>(...)"
        )
        self.assertTrue(V3 in unw and V4 in wtd)
        self.assertFalse(
            V3 in other
        )  # a <..., float, long, 3, 8, false> kernel is not the target one
        self.assertTrue(kernel_kind_ok("mix_sr_weighted", [wtd]))
        self.assertFalse(kernel_kind_ok("mix_sr_weighted", [unw]))
        self.assertTrue(kernel_kind_ok("mix_sr_highdup", [unw]))
        self.assertFalse(kernel_kind_ok("mix_sr_highdup", [wtd]))
        self.assertFalse(kernel_kind_ok("mix_sr_highdup", []))

    def test_gfx950_gate(self) -> None:
        self.assertEqual(gate_decision("gfx950:sramecc+:xnack-", ""), "run")
        self.assertEqual(gate_decision("gfx942:sramecc+:xnack-", ""), "skip")
        self.assertEqual(gate_decision("gfx942:sramecc+:xnack-", "0"), "skip")
        self.assertEqual(gate_decision("gfx942:sramecc+:xnack-", "1"), "run")

    def test_first_diff(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        y = x.clone()
        self.assertEqual(first_diff(x, y), -1)
        y[2] = torch.tensor(3.001, dtype=torch.float16)
        self.assertEqual(first_diff(x, y), 2)
        m = torch.tensor([[0.5, 0.25], [1.0, 2.0]])
        n = m.clone()
        n[1, 0] = 1.0000001
        self.assertEqual(first_diff(m, n), 2)

    def test_ulp_and_compare(self) -> None:
        one = torch.tensor([1.0], dtype=torch.float16)
        nxt = torch.tensor(
            [1.0009765625], dtype=torch.float16
        )  # one FP16 ulp above 1.0
        self.assertEqual(ulp16(one, one), 0)
        self.assertEqual(ulp16(one, nxt), 1)
        self.assertEqual(
            ulp16(
                torch.tensor([0.0], dtype=torch.float16),
                torch.tensor([-0.0], dtype=torch.float16),
            ),
            0,
        )
        tiny = torch.tensor([2.0**-24], dtype=torch.float16)
        self.assertEqual(ulp16(tiny, -tiny), 2)
        a = {
            "c": {
                "steps": [{"weights": [one], "momentum": [[torch.tensor([1.0])]]}],
                "kernels": [],
            }
        }
        b = {
            "c": {
                "steps": [{"weights": [nxt], "momentum": [[torch.tensor([1.0])]]}],
                "kernels": [],
            }
        }
        rows = compare(a, b, ["c"])
        self.assertEqual(rows[0][:5], ("c", 0, "w0", False, 1))
        self.assertEqual(rows[0][6], 1)
        self.assertEqual(rows[0][7:9], (1, 0))  # B above A: one element up
        self.assertAlmostEqual(rows[0][9], 2.0**-10, places=6)
        self.assertEqual(rows[1][:5], ("c", 0, "m0.0", True, 0))
        self.assertEqual(rows[1][7:], (0, 0, 0.0))
        x = torch.tensor([1.0, 2.0, 4.0, 8.0])
        y = torch.tensor([1.0, 2.5, 3.0, 8.0])
        self.assertEqual(signed(x, y)[:2], (1, 1))  # one up, one down
        t = totals(rows + [("c", 1, "w1", False, 2, 10, 1, 0, 2, -0.1)])
        self.assertEqual(t["w"], (3, 1, 2, 1 / 3))


@unittest.skipUnless(_rocm(), "ROCm only")
class BitwiseABTest(unittest.TestCase):
    res: dict = {}

    @classmethod
    def setUpClass(cls) -> None:
        import tempfile

        arch = torch.cuda.get_device_properties(0).gcnArchName
        if gate_decision(arch, os.environ.get(REQUIRE_ENV, "")) == "skip":
            raise unittest.SkipTest(
                f"device arch {arch}: the pin is verified on {REQUIRED_ARCH} (MI350) only; set {REQUIRE_ENV}=1 to run"
            )

        d = tempfile.mkdtemp(dir=os.environ.get("BITWISE_DIR"))
        ctx = mp.get_context("spawn")
        for route in ("A", "B"):
            path = os.path.join(d, f"{route}.pt")
            p = ctx.Process(target=_route_main, args=(route, path))
            p.start()
            p.join()
            assert p.exitcode == 0, f"route {route} process exited {p.exitcode}"
            cls.res[route] = torch.load(path)

    def test_device_is_gfx950(self) -> None:
        for route in ("A", "B"):
            print(f"route {route} arch {self.res[route]['_arch']}")
            self.assertTrue(
                self.res[route]["_arch"].startswith(REQUIRED_ARCH),
                (route, self.res[route]["_arch"]),
            )

    def test_routes_take_different_kernels(self) -> None:
        a, b = self.res["A"], self.res["B"]
        for c in GATED:
            print(
                f"{c}: A {[n.split('(')[0][-24:] for n in a[c]['kernels']]} B {[n.split('(')[0][-24:] for n in b[c]['kernels']]}"
            )
            self.assertTrue(
                a[c]["kernels"] and all(V4 in n for n in a[c]["kernels"]),
                (c, a[c]["kernels"]),
            )
            self.assertTrue(
                b[c]["kernels"] and all(V3 in n for n in b[c]["kernels"]),
                (c, b[c]["kernels"]),
            )
            for r in (a, b):
                self.assertTrue(
                    kernel_kind_ok(c, r[c]["kernels"]), (c, r[c]["kernels"])
                )
        for c in CONTROL:
            for r in (a, b):
                self.assertTrue(
                    r[c]["kernels"] and all(V4 in n for n in r[c]["kernels"]),
                    (c, r[c]["kernels"]),
                )

    def test_default_route_golden(self) -> None:
        """Pin off (<4, 8>) and pin on (<3, 8>) must keep the bits recorded before the <4, 8> pattern change."""
        hip = str(torch.version.hip)
        got = {c: digest(self.res["A"][c]) for c in {**GATED, **CONTROL, **PRODSHAPE}}
        # Pin on (route B) must give the same bits on the cases it runs (B == A held at the recording revision).
        got_b = {c: digest(self.res["B"][c]) for c in {**GATED, **CONTROL}}
        print("GOLDEN_RECORD " + json.dumps({hip: got, "B": got_b}, sort_keys=True))
        if hip not in GOLDEN:
            raise unittest.SkipTest(
                f"no golden digests for HIP {hip} ({GOLDEN_SOURCE}); the digests above are this run's"
            )
        self.assertEqual(got, GOLDEN[hip], GOLDEN_SOURCE)
        self.assertEqual(got_b, {c: GOLDEN[hip][c] for c in got_b}, GOLDEN_SOURCE)

    def test_control_identical(self) -> None:
        rows = compare(self.res["A"], self.res["B"], list(CONTROL))
        self.assertTrue(
            all(r[3] for r in rows),
            "control differs:\n" + _table([r for r in rows if not r[3]]),
        )

    def test_gated_identical(self) -> None:
        rows = compare(self.res["A"], self.res["B"], list(GATED))
        tot = totals(rows)
        ident = all(r[3] for r in rows)
        # Verdict first: CI keeps only the first few hundred characters of output.
        print(
            f"VERDICT gated_bit_identical={ident} w_diff={tot['w'][0]} m_diff={tot['m'][0]} cases={len(GATED)} steps={STEPS}"
        )
        first = []
        for c in GATED:
            for s, (sa, sb) in enumerate(
                zip(self.res["A"][c]["steps"], self.res["B"][c]["steps"])
            ):
                for t, (wa, wb) in enumerate(zip(sa["weights"], sb["weights"])):
                    i = first_diff(wa, wb)
                    if i >= 0:
                        first.append(f"{c} s{s} w{t} first {i} ulp {ulp16(wa, wb)}")
                for t, (ma, mb) in enumerate(zip(sa["momentum"], sb["momentum"])):
                    for k, (x, y) in enumerate(zip(ma, mb)):
                        i = first_diff(x, y)
                        if i >= 0:
                            first.append(f"{c} s{s} m{t}.{k} first {i}")
        art = os.environ.get("TEST_RESULT_ARTIFACTS_DIR")
        print(f"TEST_RESULT_ARTIFACTS_DIR set={bool(art)}")
        if art:
            import json

            os.makedirs(art, exist_ok=True)
            with open(
                os.path.join(art, "hip_mixed_d_occupancy_identity.json"), "w"
            ) as fh:
                json.dump(
                    {
                        "gated_bit_identical": ident,
                        "totals": {k: list(v) for k, v in tot.items()},
                        "first": first,
                        "rows": [list(map(str, r)) for r in rows],
                    },
                    fh,
                )
        print(
            "case | step | tensor | identical | differing | numel | max ulp (w) / max rel (m) | up | down | mean signed rel (B-A)\n"
            + _table(rows)
        )
        self.assertTrue(
            ident,
            "NOT bit-identical (the <3, 8> pin no longer matches <4, 8>):\n"
            + "\n".join(first[:40])
            + "\n"
            + _table([r for r in rows if not r[3]]),
        )


if __name__ == "__main__":
    unittest.main()

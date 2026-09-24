#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
import sys
import unittest

import torch


def _check_native_permutations(device: str) -> None:
    # Python autograd registrations can normalize an undefined Tensor to None.
    # Keep this process free of those registrations, as in native inference.
    assert "fbgemm_gpu" not in sys.modules
    calls = {
        "permute_2D_sparse_data": "permute, lengths, values, weights",
        "permute_1D_sparse_data": "permute, lengths.view(-1), values, weights",
        "permute_sparse_data": "permute, lengths, values, weights",
        "permute_2D_sparse_data_input1D": (
            "permute, lengths.view(-1), values, 1, weights"
        ),
        "permute_sparse_features": "permute, lengths, values, weights",
        "permute_2D_sparse_preallocated_out": (
            "permute, lengths, values, weights, values.numel(), "
            "torch.empty_like(lengths), torch.empty_like(values), weights_out"
        ),
    }
    passed = 0
    for op, arguments in calls.items():
        # Check None inside TorchScript before crossing the Python boundary,
        # and pass the result directly to another native permutation.
        compilation_source = f"""
def permute_twice(permute: Tensor, lengths: Tensor, values: Tensor,
                  weights: Optional[Tensor]):
    weights_out: Optional[Tensor] = None
    if weights is not None:
        weights_out = torch.empty_like(weights)
    out_lengths, out_values, out_weights = torch.ops.fbgemm.{op}({arguments})
    out_weights_none = out_weights is None
    twice_lengths, twice_values, twice_weights = torch.ops.fbgemm.permute_2D_sparse_data(
        permute, out_lengths.view(2, 1), out_values, out_weights)
    return (out_lengths.view(2, 1), out_values, out_weights, out_weights_none,
            twice_lengths, twice_values, twice_weights, twice_weights is None)
"""
        functions = torch.jit.CompilationUnit(compilation_source)
        for dtype in (torch.int32, torch.int64):
            for sizes in ([2, 1], [0, 2], [0, 0]):
                for weighted in (False, True):
                    case = (
                        f"{device}, {op}, {dtype}, lengths={sizes}, weighted={weighted}"
                    )
                    print(case, flush=True)
                    permute = torch.tensor([1, 0], dtype=torch.int32, device=device)
                    lengths = torch.tensor(sizes, dtype=dtype, device=device).view(2, 1)
                    values = torch.arange(sum(sizes), dtype=dtype, device=device)
                    weights = values.float() * 0.25 if weighted else None
                    result = functions.permute_twice(permute, lengths, values, weights)
                    expected_values = torch.cat(
                        (values[sizes[0] :], values[: sizes[0]])
                    )
                    torch.testing.assert_close(result[0], lengths.flip(0))
                    torch.testing.assert_close(result[1], expected_values)
                    torch.testing.assert_close(result[4], lengths)
                    torch.testing.assert_close(result[5], values)
                    assert result[3] == (not weighted), case
                    assert result[7] == (not weighted), case
                    if weighted:
                        torch.testing.assert_close(
                            result[2], expected_values.float() * 0.25
                        )
                        torch.testing.assert_close(result[6], weights)
                    else:
                        assert result[2] is None, case
                        assert result[6] is None, case
                    passed += 1
    print(f"{passed} native TorchScript cases passed on {device}", flush=True)


class PermuteOptionalWeightsTest(unittest.TestCase):
    def _run_native_test(self, device: str, flat: bool) -> None:
        # Resolve the shared libraries in the usual test environment, but load
        # only their native registrations in a fresh process below.
        import fbgemm_gpu

        libraries = (
            sorted(torch.ops.loaded_libraries)
            if getattr(fbgemm_gpu, "open_source", False)
            else ["//deeplearning/fbgemm/fbgemm_gpu:sparse_ops"]
        )

        # These flags are cached by C++; a fresh process tests both kernel paths
        # independently of any other permutation tests in the parent process.
        env = os.environ.copy()
        env["FBGEMM_FLAT_PERMUTE_1D"] = str(int(flat))
        env["FBGEMM_FLAT_PERMUTE_2D"] = str(int(flat))
        script = """
import json
import runpy
import sys
import torch

for library in json.loads(sys.argv[1]):
    torch.ops.load_library(library)
runpy.run_path(sys.argv[2])["_check_native_permutations"](sys.argv[3])
"""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    json.dumps(libraries),
                    os.path.abspath(__file__),
                    device,
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(
                f"Native permutation tests timed out: {error.stdout}\n{error.stderr}"
            )
        self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
        self.assertIn("72 native TorchScript cases passed", result.stdout)

    def test_cpu(self) -> None:
        self._run_native_test("cpu", flat=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_cuda(self) -> None:
        for flat in (False, True):
            with self.subTest(flat=flat):
                self._run_native_test("cuda", flat=flat)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import unittest
from typing import Any
from unittest import mock

import torch
from fbgemm_gpu.tbe.ssd import training

ENV = "TORCHREC_KVZCH_OWNER_LOCAL_CACHE_INT8_FUSED_CAST"


def _cache(
    device: torch.device,
    cache_dtype: torch.dtype = torch.float16,
    output_dtype: torch.dtype = torch.bfloat16,
    dim: int = 128,
    **kwargs: Any,
) -> training._DirectMappedEnrichmentCache:
    return training._DirectMappedEnrichmentCache(
        embedding_dim=dim,
        total_hash_size=1 << 20,
        cache_dtype=cache_dtype,
        device=device,
        cache_size_mb=1,
        read_populate=False,
        read_populate_max_lookups=0,
        int8=True,
        output_dtype=output_dtype,
        **kwargs,
    )


class OwnerLocalInt8FusedCastDispatchTest(unittest.TestCase):
    def test_default_off_keeps_storage_dtype_and_reference_gather(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            cache = _cache(torch.device("cpu"))
        self.assertIsNone(cache._fused_cast_dtype)
        addresses = torch.zeros(2, dtype=torch.int64)
        workspace = (
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(2, 128, dtype=torch.float16),
            addresses,
        )
        with (
            mock.patch.object(cache, "_get_workspaces", return_value=workspace),
            mock.patch.object(torch.ops.fbgemm, "kvzch_direct_cache_lookup_int8"),
            mock.patch.object(
                torch.ops.fbgemm, "kvzch_direct_cache_gather_int8"
            ) as gather,
            mock.patch.object(
                training, "_kvzch_direct_cache_gather_int8_cast"
            ) as fused,
        ):
            result = cache.lookup(torch.arange(2), mock.Mock())
        self.assertEqual(result.dtype, torch.float16)
        gather.assert_called_once_with(result, addresses, torch.float16, 0)
        fused.assert_not_called()
        self.assertEqual(
            cache.lookup(torch.empty(0, dtype=torch.int64), mock.Mock()).dtype,
            torch.float16,
        )

    def test_explicit_opt_in_and_fp32_precedence(self) -> None:
        for mode in ("1", "triton"):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, {ENV: mode}):
                enabled = _cache(torch.device("cpu"))
                same_dtype = _cache(torch.device("cpu"), output_dtype=torch.float16)
                fp32 = _cache(
                    torch.device("cpu"),
                    output_dtype=torch.float32,
                    fused_float_gather=True,
                )
                self.assertEqual(enabled._fused_cast_dtype, torch.bfloat16)
                self.assertIsNone(same_dtype._fused_cast_dtype)
                self.assertTrue(fp32._fused_float_gather)
                self.assertIsNone(fp32._fused_cast_dtype)
                self.assertEqual(
                    enabled.lookup(
                        torch.empty(0, dtype=torch.int64), mock.Mock()
                    ).dtype,
                    torch.bfloat16,
                )
        with mock.patch.dict(os.environ, {ENV: "0"}):
            self.assertIsNone(_cache(torch.device("cpu"))._fused_cast_dtype)

    def _launch(
        self, env: dict[str, str], launches: int = 1, max_ctas: int = 0
    ) -> tuple[mock.MagicMock, mock.MagicMock]:
        compiled = mock.MagicMock(function=1234)
        kernel = mock.MagicMock()
        kernel.__getitem__.return_value = mock.MagicMock(return_value=compiled)
        set_attr = mock.MagicMock(return_value=0)
        training._smem_carveout_set.clear()
        with (
            mock.patch.dict(os.environ, {ENV: "triton", **env}, clear=True),
            mock.patch.object(
                training,
                "_get_kvzch_gather_int8_cast_triton_kernel",
                return_value=kernel,
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=mock.MagicMock(multi_processor_count=4),
            ),
            mock.patch.object(training, "_cu_func_set_attribute", set_attr),
            mock.patch.object(torch.cuda, "device") as device_guard,
        ):
            for _ in range(launches):
                training._kvzch_direct_cache_gather_int8_cast(
                    torch.empty(1000, 132, dtype=torch.bfloat16),
                    torch.zeros(1000, dtype=torch.int64),
                    torch.float16,
                    max_ctas,
                )
        self.assertEqual(device_guard.call_count, launches)
        device_guard.assert_called_with(torch.device("cpu"))
        return kernel, set_attr

    def test_launch_defaults_and_fractional_grid(self) -> None:
        kernel, set_attr = self._launch({})
        kernel.__getitem__.assert_called_once_with((63,))
        launch = kernel.__getitem__.return_value.call_args.kwargs
        self.assertEqual(
            (launch["BM"], launch["num_warps"], launch["D"], launch["BD"]),
            (16, 4, 132, 256),
        )
        set_attr.assert_not_called()
        kernel, _ = self._launch(
            {
                ENV + "_PROGRAMS_PER_SM": "0.5",
                ENV + "_BM": "128",
                ENV + "_NUM_WARPS": "8",
            }
        )
        kernel.__getitem__.assert_called_once_with((2,))
        self.assertEqual(kernel.__getitem__.return_value.call_args.kwargs["BM"], 128)
        self.assertEqual(
            kernel.__getitem__.return_value.call_args.kwargs["num_warps"], 8
        )
        kernel, _ = self._launch({}, max_ctas=3)
        kernel.__getitem__.assert_called_once_with((3,))

    def _assert_invalid_launch_config(self, suffix: str, value: str) -> None:
        output = torch.empty(2, 128, dtype=torch.bfloat16)
        addresses = torch.zeros(2, dtype=torch.int64)
        with (
            mock.patch.dict(
                os.environ, {ENV: "triton", ENV + suffix: value}, clear=True
            ),
            mock.patch.dict(sys.modules, {"triton": None, "triton.language": None}),
            mock.patch.object(torch.cuda, "get_device_properties") as properties,
            mock.patch.object(
                training, "_get_kvzch_gather_int8_cast_triton_kernel"
            ) as kernel,
        ):
            with self.assertRaisesRegex(ValueError, ENV + suffix):
                training._kvzch_direct_cache_gather_int8_cast(
                    output, addresses, torch.float16
                )
        properties.assert_not_called()
        kernel.assert_not_called()

    def test_unsupported_row_dtype_rejected_before_cuda_or_triton(self) -> None:
        for row_dtype in (torch.float64, torch.int8, torch.bool):
            with (
                self.subTest(row_dtype=row_dtype),
                mock.patch.dict(os.environ, {ENV: "triton"}, clear=True),
                mock.patch.dict(sys.modules, {"triton": None, "triton.language": None}),
                mock.patch.object(torch.cuda, "get_device_properties") as properties,
                mock.patch.object(
                    training, "_get_kvzch_gather_int8_cast_triton_kernel"
                ) as kernel,
            ):
                with self.assertRaisesRegex(ValueError, ENV + ".*row_dtype"):
                    training._kvzch_direct_cache_gather_int8_cast(
                        torch.empty(2, 128, dtype=torch.bfloat16),
                        torch.zeros(2, dtype=torch.int64),
                        row_dtype,
                    )
                properties.assert_not_called()
                kernel.assert_not_called()

    def test_bm_rejects_invalid_values_before_cuda_or_triton(self) -> None:
        for bm in ("0", "3", "-1", "-16", "abc", "", "16.0"):
            with self.subTest(bm=bm):
                self._assert_invalid_launch_config("_BM", bm)

    def test_num_warps_rejects_invalid_values_before_cuda_or_triton(self) -> None:
        for warps in ("0", "3", "64", "-4", "abc", "", "4.0"):
            with self.subTest(warps=warps):
                self._assert_invalid_launch_config("_NUM_WARPS", warps)

    def test_programs_per_sm_rejects_invalid_values_before_cuda_or_triton(self) -> None:
        for programs in ("0", "-1", "nan", "inf", "-inf", "abc", ""):
            with self.subTest(programs=programs):
                self._assert_invalid_launch_config("_PROGRAMS_PER_SM", programs)

    def test_launch_config_accepts_defaults_v55_and_valid_boundaries(self) -> None:
        cases = (
            ({}, (16, 4, 16.0)),
            ({ENV + "_PROGRAMS_PER_SM": "4"}, (16, 4, 4.0)),
            (
                {
                    ENV + "_BM": "1",
                    ENV + "_NUM_WARPS": "1",
                    ENV + "_PROGRAMS_PER_SM": "0.5",
                },
                (1, 1, 0.5),
            ),
            ({ENV + "_BM": "128", ENV + "_NUM_WARPS": "32"}, (128, 32, 16.0)),
        )
        for env, expected in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(
                    training._kvzch_int8_fused_cast_launch_config(), expected
                )

    def test_carveout_set_once_per_compiled_kernel(self) -> None:
        kernel, set_attr = self._launch({ENV + "_SMEM_CARVEOUT": "100"}, launches=3)
        self.assertEqual(kernel.__getitem__.call_count, 3)
        set_attr.assert_called_once_with(1234, 9, 100)

    def test_carveout_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            self._launch({ENV + "_SMEM_CARVEOUT": "101"})

    def test_carveout_driver_failure_is_not_cached(self) -> None:
        training._smem_carveout_set.clear()
        with mock.patch.object(training, "_cu_func_set_attribute", return_value=1):
            with self.assertRaises(RuntimeError):
                training._set_smem_carveout(1234, 100)
        self.assertNotIn(1234, training._smem_carveout_set)


@unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA device")
class OwnerLocalInt8FusedCastNumericTest(unittest.TestCase):
    def _check(
        self,
        cache_dtype: torch.dtype,
        out_dtype: torch.dtype,
        dim: int = 128,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.accelerator.current_accelerator(True)
        assert device is not None
        self.assertEqual(device.type, "cuda")
        with mock.patch.dict(os.environ, {ENV: "triton"}):
            cache = _cache(device, cache_dtype, out_dtype, dim)
        self.assertEqual(cache._fused_cast_dtype, out_dtype)
        self.assertEqual(cache.cache_weights.size(1), dim + 8)
        table = (torch.randn(4096, dim, device=device) * 3 + 1).to(cache_dtype)
        table[0].zero_()
        table[2].fill_(-1)
        resident = torch.arange(0, 4096, 2, device=device)
        # Exercise the real stable INT8 writer, including its D-byte qparam offset.
        cache.populate_from_write(resident, table[resident])
        indices = torch.cat(
            (
                torch.arange(4096, device=device),
                torch.randint(0, 4096, (45907,), device=device),
            )
        )

        def read_backend_into(
            keys_cpu: torch.Tensor, rows: torch.Tensor, count_cpu: torch.Tensor
        ) -> None:
            torch.cuda.synchronize(device)
            count = int(count_cpu[0])
            keys = keys_cpu[:count].to(device)
            rows[:count].copy_(table[keys])

        with mock.patch.dict(os.environ, {ENV: "triton"}):
            fused = cache.lookup(indices, read_backend_into)
        hits, queries = cache.last_hit_counts()
        self.assertGreater(hits, 0)
        self.assertLess(hits, queries)
        cache._fused_cast_dtype = None
        reference = cache.lookup(indices, read_backend_into)
        self.assertEqual(fused.dtype, out_dtype)
        self.assertEqual(reference.dtype, cache_dtype)
        self.assertTrue(
            torch.equal(
                fused.view(torch.uint8), reference.to(out_dtype).view(torch.uint8)
            )
        )

    def test_triton_matches_reference_bitwise(self) -> None:
        for dim in (128, 132):
            for cache_dtype, output_dtype in (
                (torch.float16, torch.bfloat16),
                (torch.float16, torch.float32),
                (torch.bfloat16, torch.float16),
                (torch.float32, torch.bfloat16),
            ):
                with self.subTest(
                    dim=dim, cache_dtype=cache_dtype, output_dtype=output_dtype
                ):
                    self._check(cache_dtype, output_dtype, dim=dim)

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires two CUDA devices")
    def test_non_current_device_matches_reference_bitwise(self) -> None:
        current = torch.cuda.current_device()
        other = (current + 1) % torch.cuda.device_count()
        self._check(torch.float16, torch.bfloat16, device=torch.device(f"cuda:{other}"))
        self.assertEqual(torch.cuda.current_device(), current)

    def test_shape_knobs_match_reference_bitwise(self) -> None:
        for programs, bm, warps in (
            ("8", "64", "4"),
            ("4", "128", "8"),
            ("1", "16", "4"),
            ("0.5", "128", "8"),
        ):
            with (
                self.subTest(programs=programs, bm=bm, warps=warps),
                mock.patch.dict(
                    os.environ,
                    {
                        ENV + "_PROGRAMS_PER_SM": programs,
                        ENV + "_BM": bm,
                        ENV + "_NUM_WARPS": warps,
                    },
                ),
            ):
                self._check(torch.float16, torch.bfloat16)

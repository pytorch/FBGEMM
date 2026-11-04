# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import sys
import unittest
from typing import Any
from unittest import mock

import torch
from fbgemm_gpu.tbe.ssd import training

ENV = "TORCHREC_KVZCH_OWNER_LOCAL_CACHE_INT8_FUSED_CAST"
_HAS_TRITON: bool = importlib.util.find_spec("triton") is not None


def _cache(
    device: torch.device,
    cache_dtype: torch.dtype = torch.float16,
    output_dtype: torch.dtype = torch.bfloat16,
    dim: int = 128,
    read_populate: bool = False,
    **kwargs: Any,
) -> training._DirectMappedEnrichmentCache:
    return training._DirectMappedEnrichmentCache(
        embedding_dim=dim,
        total_hash_size=1 << 20,
        cache_dtype=cache_dtype,
        device=device,
        cache_size_mb=1,
        read_populate=read_populate,
        read_populate_max_lookups=0,
        int8=True,
        output_dtype=output_dtype,
        **kwargs,
    )


def _four_sms() -> Any:
    return mock.patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=mock.MagicMock(multi_processor_count=4),
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
            mock.patch.object(
                torch.ops.fbgemm, "kvzch_direct_cache_lookup_int8", create=True
            ),
            mock.patch.object(
                torch.ops.fbgemm, "kvzch_direct_cache_gather_int8", create=True
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
            with (
                self.subTest(mode=mode),
                mock.patch.dict(os.environ, {ENV: mode}),
                _four_sms(),
            ):
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

    def test_unknown_mode_fails_at_construction(self) -> None:
        for mode in ("true", "on", "Triton"):
            with (
                self.subTest(mode=mode),
                mock.patch.dict(os.environ, {ENV: mode}, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, ENV):
                    _cache(torch.device("cpu"))

    def test_launch_config_is_resolved_once_at_construction(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {ENV: "triton", ENV + "_BM": "64", ENV + "_SMEM_CARVEOUT": "50"},
                clear=True,
            ),
            _four_sms() as properties,
        ):
            cache = _cache(torch.device("cpu"))
        self.assertEqual(properties.call_count, 1)
        self.assertEqual(
            cache._fused_cast_launch,
            training._Int8FusedCastLaunch(
                row_dtype=torch.float16,
                use_cxx_op=False,
                bm=64,
                num_warps=4,
                block_d=128,
                max_programs=64,
                smem_carveout=50,
            ),
        )
        workspace = (
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(2, 128, dtype=torch.float16),
            torch.zeros(2, dtype=torch.int64),
        )
        # Lookups neither read the environment nor query the device again.
        with (
            mock.patch.dict(os.environ, {ENV: "bogus", ENV + "_BM": "3"}),
            mock.patch.object(cache, "_get_workspaces", return_value=workspace),
            mock.patch.object(
                torch.ops.fbgemm, "kvzch_direct_cache_lookup_int8", create=True
            ),
            mock.patch.object(torch.cuda, "get_device_properties") as properties,
            mock.patch.object(
                training, "_kvzch_direct_cache_gather_int8_cast"
            ) as fused,
        ):
            result = cache.lookup(torch.arange(2), mock.Mock())
        properties.assert_not_called()
        fused.assert_called_once_with(result, workspace[5], cache._fused_cast_launch)
        self.assertEqual(result.dtype, torch.bfloat16)

    def _launch(
        self, env: dict[str, str], launches: int = 1, max_ctas: int = 0
    ) -> tuple[mock.MagicMock, mock.MagicMock]:
        compiled = mock.MagicMock(function=1234)
        kernel = mock.MagicMock()
        kernel.__getitem__.return_value = mock.MagicMock(return_value=compiled)
        set_attr = mock.MagicMock(return_value=0)
        training._smem_carveout_set.clear()
        with mock.patch.dict(os.environ, env, clear=True), _four_sms():
            launch = training._kvzch_int8_fused_cast_launch_config(
                "triton", torch.float16, 132, torch.device("cpu"), max_ctas
            )
        with (
            mock.patch.dict(sys.modules, {"triton": None, "triton.language": None}),
            mock.patch.object(
                training,
                "_get_kvzch_gather_int8_cast_triton_kernel",
                return_value=(kernel, {torch.float16: "fp16"}),
            ),
            mock.patch.object(training, "_cu_func_set_attribute", set_attr),
            mock.patch.object(torch.cuda, "device") as device_guard,
        ):
            for _ in range(launches):
                training._kvzch_direct_cache_gather_int8_cast(
                    torch.empty(1000, 132, dtype=torch.bfloat16),
                    torch.zeros(1000, dtype=torch.int64),
                    launch,
                )
        self.assertEqual(device_guard.call_count, launches)
        device_guard.assert_called_with(torch.device("cpu"))
        return kernel, set_attr

    def test_launch_defaults_and_fractional_grid(self) -> None:
        kernel, set_attr = self._launch({})
        kernel.__getitem__.assert_called_once_with((63,))
        launch = kernel.__getitem__.return_value.call_args.kwargs
        self.assertEqual(
            (
                launch["BM"],
                launch["num_warps"],
                launch["D"],
                launch["BD"],
                launch["ROW_T"],
            ),
            (16, 4, 132, 256, "fp16"),
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
                _cache(torch.device("cpu"))
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
                    training._kvzch_int8_fused_cast_launch_config(
                        "triton", row_dtype, 128, torch.device("cpu"), 0
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

    def test_smem_carveout_rejects_invalid_values_before_cuda_or_triton(
        self,
    ) -> None:
        for carveout in ("-1", "101", "abc", "50.0"):
            with self.subTest(carveout=carveout):
                self._assert_invalid_launch_config("_SMEM_CARVEOUT", carveout)

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
                self.assertEqual(training._kvzch_int8_fused_cast_shape(), expected)

    def test_carveout_set_once_per_compiled_kernel(self) -> None:
        kernel, set_attr = self._launch({ENV + "_SMEM_CARVEOUT": "100"}, launches=3)
        self.assertEqual(kernel.__getitem__.call_count, 3)
        set_attr.assert_called_once_with(1234, 9, 100)

    def test_carveout_zero_is_applied(self) -> None:
        _, set_attr = self._launch({ENV + "_SMEM_CARVEOUT": "0"})
        set_attr.assert_called_once_with(1234, 9, 0)

    def test_carveout_driver_failure_is_not_cached(self) -> None:
        training._smem_carveout_set.clear()
        with mock.patch.object(training, "_cu_func_set_attribute", return_value=1):
            with self.assertRaises(RuntimeError):
                training._set_smem_carveout(1234, 100)
        self.assertNotIn(1234, training._smem_carveout_set)


@unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA device")
@unittest.skipIf(not _HAS_TRITON, "requires triton")
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

    def test_reserved_sms_keep_the_cuda_gather(self) -> None:
        device = torch.accelerator.current_accelerator(True)
        assert device is not None
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        with mock.patch.dict(os.environ, {ENV: "triton"}, clear=True):
            capped = _cache(device, gather_reserved_sms=sms - 2)
        self.assertEqual(capped._gather_max_ctas, 2)
        self.assertIsNone(capped._fused_cast_dtype)
        self.assertIsNone(capped._fused_cast_launch)

    def test_read_populate_matches_reference_bitwise(self) -> None:
        device = torch.accelerator.current_accelerator(True)
        assert device is not None
        caches = []
        for env in ({ENV: "triton"}, {}):
            with mock.patch.dict(os.environ, env, clear=True):
                caches.append(_cache(device, read_populate=True))
        fused_cache, reference_cache = caches
        self.assertEqual(fused_cache._fused_cast_dtype, torch.bfloat16)
        self.assertIsNone(reference_cache._fused_cast_dtype)
        table = (torch.randn(8192, 128, device=device) * 3 + 1).half()

        def read_backend_into(
            keys_cpu: torch.Tensor, rows: torch.Tensor, count_cpu: torch.Tensor
        ) -> None:
            torch.cuda.synchronize(device)
            count = int(count_cpu[0])
            rows[:count].copy_(table[keys_cpu[:count].to(device)])

        generator = torch.Generator(device=device).manual_seed(5)
        for _ in range(3):
            # Duplicates make a miss insert race a hit on the same slot, which
            # the gather-before-insert ordering must keep correct.
            indices = torch.randint(
                0, 8192, (20000,), device=device, generator=generator
            )
            # Racing inserts can pick different winners, so start each round
            # from the same state.
            reference_cache.cache_state.copy_(fused_cache.cache_state)
            reference_cache.cache_weights.copy_(fused_cache.cache_weights)
            fused = fused_cache.lookup(indices, read_backend_into)
            reference = reference_cache.lookup(indices, read_backend_into)
            self.assertEqual(
                fused_cache.last_hit_counts(), reference_cache.last_hit_counts()
            )
            self.assertEqual(fused.dtype, torch.bfloat16)
            self.assertTrue(
                torch.equal(
                    fused.view(torch.uint8),
                    reference.to(torch.bfloat16).view(torch.uint8),
                )
            )
        hits, queries = fused_cache.last_hit_counts()
        self.assertGreater(hits, 0)
        self.assertLess(hits, queries)

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


def _slot(key: int, num_slots: int) -> int:
    """cache_slot() in split_embeddings_cache_cuda.cuh."""
    mask = (1 << 64) - 1
    h = key & mask
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & mask
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & mask
    h ^= h >> 33
    return h % num_slots


def _dram_backend(dim: int, num_rows: int) -> Any:
    enrichment_config = torch.classes.fbgemm.EnrichmentConfig(
        4,  # IN_MEMORY_TEST_ONLY
        "",
        "",
        dim,
        1,  # THRIFT_FLOAT
    )
    return torch.classes.fbgemm.DramKVEmbeddingCacheWrapper(
        max_D=dim,
        uniform_init_lower=-0.1,
        uniform_init_upper=0.1,
        num_shards=8,
        num_threads=8,
        row_storage_bitwidth=16,
        hash_size_cumsum=torch.tensor([0, num_rows]),
        disable_random_init=True,
        enrichment_config=enrichment_config,
    )


@unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA device")
class ExclusiveOwnerCacheTest(unittest.TestCase):
    def test_insert_claims_only_empty_slots_and_reports_them(self) -> None:
        device = torch.accelerator.current_accelerator(check_available=True)
        assert device is not None
        dim, slots = 128, 4096
        cache_state = torch.full((slots, 1), -1, dtype=torch.int64, device=device)
        cache_weights = torch.zeros(slots, dim + 8, dtype=torch.uint8, device=device)
        last_access = torch.zeros(slots, dtype=torch.int32, device=device)
        occupied_key = 7
        cache_state[_slot(occupied_key, slots)] = 10**9
        keys = torch.arange(1000, device=device)
        rows = torch.randn(1000, dim, device=device).half()
        rows[3].zero_()
        promoted = torch.full((1000,), -1, dtype=torch.int64, device=device)
        count = torch.zeros(1, dtype=torch.int32, device=device)

        torch.ops.fbgemm.kvzch_exclusive_cache_insert_int8(
            cache_state,
            cache_weights,
            last_access,
            keys,
            rows,
            None,
            promoted,
            count,
            1234,
        )

        n = int(count.item())
        claimed = set(promoted[:n].tolist())
        self.assertNotIn(occupied_key, claimed)
        self.assertNotIn(3, claimed)
        self.assertEqual(cache_state[_slot(occupied_key, slots)].item(), 10**9)
        # Each claimed key sits in its hashed slot with its access time; a key
        # that lost its slot to another one in the batch was not reported.
        for key in claimed:
            slot = _slot(key, slots)
            self.assertEqual(cache_state[slot].item(), key)
            self.assertEqual(last_access[slot].item(), 1234)
        self.assertEqual(int((cache_state >= 0).sum().item()), n + 1)

        # Hits stamp their slot; misses (even address) do not.
        slot = _slot(next(iter(claimed)), slots)
        hit = cache_weights.data_ptr() + slot * (dim + 8) | 1
        miss = cache_weights.data_ptr() + 2 * (dim + 8)
        torch.ops.fbgemm.kvzch_exclusive_cache_touch(
            torch.tensor([hit, miss], device=device),
            cache_weights,
            last_access,
            5678,
        )
        self.assertEqual(last_access[slot].item(), 5678)
        self.assertNotEqual(last_access[2].item(), 5678)

    def test_lookup_moves_rows_to_hbm_and_reset_moves_them_back(self) -> None:
        device = torch.accelerator.current_accelerator(check_available=True)
        assert device is not None
        dim, num_rows = 128, 2048
        backend = _dram_backend(dim, num_rows)
        table = (torch.randn(num_rows, dim) * 0.3).half()
        ids = torch.arange(num_rows)
        backend.set(ids, table, torch.tensor([num_rows]))
        backend.wait_util_filling_work_done()
        cache = training._ExclusiveDirectMappedEnrichmentCache(
            embedding_dim=dim,
            total_hash_size=1 << 20,
            cache_dtype=torch.float16,
            device=device,
            cache_size_mb=4,
            backend=backend,
        )

        def read_backend_into(
            keys_cpu: torch.Tensor, rows: torch.Tensor, count_cpu: torch.Tensor
        ) -> None:
            backend.get_cuda(keys_cpu, rows, count_cpu)

        indices = ids.to(device)
        first = cache.lookup(indices, read_backend_into)
        torch.cuda.synchronize()
        backend.exclusive_hbm_wait()
        in_hbm = int(backend.exclusive_hbm_stats()[0])
        self.assertGreater(in_hbm, num_rows * 0.8)
        self.assertEqual(int(backend.get_dram_kv_perf(1, 1)[22]), num_rows - in_hbm)
        torch.testing.assert_close(first.cpu(), table, rtol=0, atol=0)

        second = cache.lookup(indices, read_backend_into)
        hits, queries = cache.last_hit_counts()
        self.assertEqual(hits, in_hbm)
        self.assertEqual(queries, num_rows)
        # HBM rows are INT8: within half a quantization step of the row range.
        step = (table.float().amax(1) - table.float().amin(1)) / 255
        err = (second.cpu().float() - table.float()).abs().amax(1)
        self.assertTrue(bool((err <= step / 2 + 1e-3).all()))

        # No lookup reached a row that lived only in HBM.
        self.assertEqual(int(backend.exclusive_hbm_stats()[7]), 0)

        # dper_lib create_ssd_rdb_checkpoint walks every module and treats one
        # with `_ssd_db`, `_lookups` or `_lookup` as a sharded container.
        for name in ("_ssd_db", "_lookups", "_lookup", "_emb_module", "_emb_modules"):
            self.assertFalse(hasattr(cache, name), name)

        cache.reset()
        self.assertEqual(int(backend.exclusive_hbm_stats()[0]), 0)
        self.assertEqual(int(backend.get_dram_kv_perf(1, 1)[22]), num_rows)
        third = cache.lookup(indices, read_backend_into)
        torch.testing.assert_close(third, second, rtol=0, atol=0)

    def test_lookups_return_each_ids_own_row_through_promotion_and_rewrites(
        self,
    ) -> None:
        """Every lookup returns the id's own row (exact from DRAM, within half
        an INT8 step from HBM), never zeros or another id's row: while
        promotions are in flight, after they land, with duplicate ids and slot
        collisions, after DRAM rewrites of HBM rows, and through the write-side
        fill."""
        device = torch.accelerator.current_accelerator(check_available=True)
        assert device is not None
        dim, num_rows = 128, 50000
        backend = _dram_backend(dim, num_rows)
        table = (torch.randn(num_rows, dim) * 0.3).half()
        ids = torch.arange(num_rows)
        backend.set(ids, table, torch.tensor([num_rows]))
        backend.wait_util_filling_work_done()
        # ~7k slots for 50k ids: most promotions collide.
        cache = training._ExclusiveDirectMappedEnrichmentCache(
            embedding_dim=dim,
            total_hash_size=1 << 20,
            cache_dtype=torch.float16,
            device=device,
            cache_size_mb=1,
            backend=backend,
            promote_max_rows=4096,
        )

        def read_backend_into(
            keys_cpu: torch.Tensor, rows: torch.Tensor, count_cpu: torch.Tensor
        ) -> None:
            backend.get_cuda(keys_cpu, rows, count_cpu)

        def check(expected: torch.Tensor, indices: torch.Tensor) -> None:
            out = cache.lookup(indices.to(device), read_backend_into).cpu().float()
            want = expected[indices].float()
            step = (want.amax(1) - want.amin(1)) / 255
            err = (out - want).abs().amax(1)
            bad = (err > step / 2 + 1e-3).nonzero().flatten()
            self.assertEqual(
                bad.numel(),
                0,
                f"{bad.numel()} rows wrong, e.g. id {indices[bad[:5]].tolist()} "
                f"err {err[bad[:5]].tolist()}",
            )

        gen = torch.Generator().manual_seed(0)
        for _ in range(20):
            # Duplicates and random order; no wait between lookups, so rows are
            # read while their promotions are still queued.
            check(table, torch.randint(0, num_rows, (30000,), generator=gen))
        torch.cuda.synchronize()
        backend.exclusive_hbm_wait()
        self.assertGreater(int(backend.exclusive_hbm_stats()[0]), 1000)
        check(table, ids)

        # Rewrite half of the rows, HBM rows included: lookups see the new rows.
        rewritten = ids[::2].contiguous()
        fresh = table.clone()
        fresh[rewritten] = (torch.randn(rewritten.numel(), dim) * 0.3).half()
        backend.set(rewritten, fresh[rewritten], torch.tensor([rewritten.numel()]))
        backend.wait_util_filling_work_done()
        backend.exclusive_hbm_wait()
        self.assertGreater(int(backend.exclusive_hbm_stats()[5]), 0)
        for _ in range(5):
            check(fresh, torch.randint(0, num_rows, (30000,), generator=gen))

        # The write-side fill promotes rows read back from DRAM.
        fill = torch.randint(0, num_rows, (20000,), generator=gen).unique()
        rows = fresh[fill].to(device)
        cache.populate_from_write(fill.to(device), rows)
        torch.cuda.synchronize()
        backend.exclusive_hbm_wait()
        check(fresh, ids)
        self.assertEqual(int(backend.exclusive_hbm_stats()[7]), 0)

    @unittest.skipIf(
        os.environ.get("KVZCH_EXCLUSIVE_HBM_BENCH") is None,
        "set KVZCH_EXCLUSIVE_HBM_BENCH=1",
    )
    def test_benchmark_probe_touch_insert_at_ot_scale(self) -> None:
        """GPU cost per lookup of the exclusive cache at OT scale: 5M indices
        into a 12000 MB cache (~85M slots) that is ~70% full."""
        device = torch.accelerator.current_accelerator(check_available=True)
        assert device is not None
        dim = 128
        slot_bytes = dim + 8 + 8 + 4
        slots = 12000 * 1024 * 1024 // slot_bytes
        cache_state = torch.full((slots, 1), -1, dtype=torch.int64, device=device)
        cache_weights = torch.zeros(slots, dim + 8, dtype=torch.uint8, device=device)
        last_access = torch.zeros(slots, dtype=torch.int32, device=device)
        filled = int(slots * 1.2)
        chunk = 1 << 22
        promoted = torch.empty(chunk, dtype=torch.int64, device=device)
        count = torch.zeros(1, dtype=torch.int32, device=device)
        rows = torch.randn(chunk, dim, device=device).half()
        for start in range(0, filled, chunk):
            keys = torch.arange(start, min(filled, start + chunk), device=device)
            torch.ops.fbgemm.kvzch_exclusive_cache_insert_int8(
                cache_state,
                cache_weights,
                last_access,
                keys,
                rows[: keys.numel()],
                None,
                promoted,
                count,
                1,
            )
        occupancy = float((cache_state >= 0).float().mean().item())

        n = 5_000_000
        indices = torch.randint(0, 2 * filled, (n,), device=device)
        action_indices = torch.empty(n, dtype=torch.int64, device=device)
        actions_count = torch.zeros(1, dtype=torch.int32, device=device)
        inserted = torch.randn(n, dim, device=device).half()
        row_addresses = torch.empty(n, dtype=torch.int64, device=device)
        output = torch.empty(n, dim, dtype=torch.float16, device=device)
        promoted = torch.empty(n, dtype=torch.int64, device=device)
        cap = torch.full((1,), 262144, dtype=torch.int32, device=device)

        def timed(fn: Any) -> float:
            for _ in range(3):
                fn()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 10

        probe_ms = timed(
            lambda: torch.ops.fbgemm.kvzch_direct_cache_lookup_int8(
                indices,
                cache_state,
                cache_weights,
                action_indices,
                actions_count,
                inserted,
                row_addresses,
            )
        )
        hit_rate = 1 - int(actions_count.item()) / n
        gather_ms = timed(
            lambda: torch.ops.fbgemm.kvzch_direct_cache_gather_int8(
                output, row_addresses, torch.float16, 0
            )
        )
        touch_ms = timed(
            lambda: torch.ops.fbgemm.kvzch_exclusive_cache_touch(
                row_addresses, cache_weights, last_access, 2
            )
        )
        insert_ms = timed(
            lambda: torch.ops.fbgemm.kvzch_exclusive_cache_insert_int8(
                cache_state,
                cache_weights,
                last_access,
                action_indices,
                inserted,
                cap,
                promoted,
                count,
                3,
            )
        )
        print(
            f"exclusive owner cache, {slots} slots ({occupancy:.0%} full), "
            f"{n} indices, hit rate {hit_rate:.0%}: probe {probe_ms:.3f} ms, "
            f"gather {gather_ms:.3f} ms, touch {touch_ms:.3f} ms, "
            f"insert (cap 262144) {insert_ms:.3f} ms"
        )

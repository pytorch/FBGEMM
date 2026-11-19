#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import random
import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    _reconcile_empty_lxu_cache_weights,
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.cache.cache_config import CacheAlgorithm
from fbgemm_gpu.tbe.config.embedding_config import EmbeddingLocation
from fbgemm_gpu.tbe.utils import generate_requests, round_up
from hypothesis import assume, example, given, settings, Verbosity

from .. import common  # noqa E402
from ..common import MAX_EXAMPLES, MAX_EXAMPLES_LONG_RUNNING, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable, optests


VERBOSITY: Verbosity = Verbosity.verbose

# pyre-ignore
additional_decorators: dict[str, list[Callable[..., Any]]] = {
    "test_faketensor__test_nbit_forward_cpu_seq_int4": {
        unittest.skip(
            "Operator outputs int4 tensors which do not support opcheck tests"
        ),
    },
    "test_pt2_compliant_tag_fbgemm_int_nbit_split_embedding_codegen_lookup_function": {
        unittest.skip(
            "Operator outputs int4 tensors which do not support opcheck tests"
        ),
    },
}


@optests.generate_opcheck_tests(fast=True, additional_decorators=additional_decorators)
class NBitSplitEmbeddingsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @example(
        T=2,
        D=32,
        B=4,
        log_E=3,
        L=3,
        weights_ty=SparseType.INT8,
        output_dtype=SparseType.BF16,
    )
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        output_dtype=st.sampled_from(
            [
                SparseType.FP16,
                SparseType.BF16,
                SparseType.INT8,
            ]
        ),
    )
    @settings(
        verbosity=VERBOSITY,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    def test_nbit_split_embedding_weights_with_scale_and_bias(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_ty: SparseType,
        output_dtype: SparseType,
    ) -> None:
        D_alignment = max(weights_ty.align_size() for t in range(T))
        D_alignment = max(D_alignment, output_dtype.align_size())
        D = round_up(D, D_alignment)
        # ROCm kernels support BF16 output; CUDA kernels require sm80+.
        assume(
            output_dtype != SparseType.BF16
            or (
                torch.cuda.is_available()
                and (
                    torch.version.hip is not None
                    or torch.cuda.get_device_capability() >= (8, 0)
                )
            )
        )
        Ds = [
            round_up(
                np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                D_alignment,
            )
            for _ in range(T)
        ]
        Ds = [D] * T
        E = int(10**log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]

        weights_ty_list = [weights_ty] * T
        managed = [EmbeddingLocation.DEVICE] * T
        op = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    W_TY,
                    EmbeddingLocation(M),
                )
                for (E, D, M, W_TY) in zip(Es, Ds, managed, weights_ty_list)
            ],
            output_dtype=output_dtype,
            device=torch.cuda.current_device(),
        )
        # Initialize the random weights for int nbit table split embedding bag
        op.fill_random_weights()

        # sync weights between two ops
        split_weights = op.split_embedding_weights()
        split_weights_with_scale_bias = op.split_embedding_weights_with_scale_bias(
            split_scale_bias_mode=2
        )
        for t in range(T):
            weights, scale_bias = split_weights[t]
            weights2, scale, bias = split_weights_with_scale_bias[t]
            torch.testing.assert_close(weights2, weights)
            if scale is None:
                self.assertIsNone(scale_bias)
                self.assertIsNone(bias)
            else:
                torch.testing.assert_close(
                    scale.cpu(),
                    torch.tensor(
                        # pyre-ignore: ['Undefined attribute : Optional type has no attribute `__...
                        scale_bias[:, : scale_bias.size(1) // 2]
                        .contiguous()
                        .cpu()
                        .numpy()
                        .view(np.float16)
                    ),
                )
                torch.testing.assert_close(
                    # pyre-ignore: ['Undefined attribute : Optional type has no attribute `cpu`.']
                    bias.cpu(),
                    torch.tensor(
                        scale_bias[:, scale_bias.size(1) // 2 :]
                        .contiguous()
                        .cpu()
                        .numpy()
                        .view(np.float16)
                    ),
                )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
        emulate_pruning=st.booleans(),
        indices_dtype=st.sampled_from([torch.int, torch.int64]),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_int_nbit_split_embedding_uvm_caching_codegen_lookup_function(
        self,
        weights_ty: SparseType,
        emulate_pruning: bool,
        indices_dtype: torch.dtype,
    ) -> None:
        # TODO: support direct-mapped in int_nbit_split_embedding_uvm_caching_codegen_lookup_function
        # This test is for int_nbit_split_embedding_uvm_caching_codegen_lookup_function.
        # We run IntNBitTableBatchedEmbeddingBagsCodegen with UVM_CACHING, and then
        # run int_nbit_split_embedding_uvm_caching_codegen_lookup_function with the
        # exact same cache configuration. As both use the same logic, the result
        # as well as cache state should match.

        # Currently, int_nbit_split_embedding_uvm_caching_codegen_lookup_function supports only LRU.
        cache_algorithm = CacheAlgorithm.LRU
        associativity = 32  # Currently, hard-coded 32-way set associative.
        current_device: torch.device = torch.device(torch.cuda.current_device())

        T = random.randint(1, 5)
        B = random.randint(1, 128)
        L = random.randint(1, 20)
        D = random.randint(2, 256)
        log_E = random.randint(3, 5)

        iters = 3
        E = int(10**log_E)

        D_alignment = (
            1 if weights_ty.bit_rate() % 8 == 0 else int(8 / weights_ty.bit_rate())
        )
        D = round_up(D, D_alignment)

        # Currently, int_nbit_split_embedding_uvm_caching_codegen_lookup_function supports only all UVM or all UVM_CACHING.
        Ds = [D] * T
        Es = [E] * T
        managed_caching = [EmbeddingLocation.MANAGED_CACHING] * T

        # Note both cc_ref and cc use caching.
        cc_ref = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", E, D, weights_ty, M) for (E, D, M) in zip(Es, Ds, managed_caching)],
            cache_algorithm=cache_algorithm,
            indices_dtype=indices_dtype,
        )
        cc_ref.fill_random_weights()

        # cc is only for cache states; we test int_nbit_split_embedding_uvm_caching_codegen_lookup_function directly;
        # hence, no need to synchronize cc's weights with cc_ref's.
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            [("", E, D, weights_ty, M) for (E, D, M) in zip(Es, Ds, managed_caching)],
            cache_algorithm=cache_algorithm,
            indices_dtype=indices_dtype,
        )
        cc.fill_random_weights()

        # weights_placement for all UVM case.
        managed_uvm = [EmbeddingLocation.MANAGED] * T
        placement_uvm = torch.tensor(
            managed_uvm, device=current_device, dtype=torch.int32
        )

        # zero size HBM cache for UVM case.
        zero_size_cache_weights = torch.zeros(
            0, 0, device=current_device, dtype=torch.uint8
        )

        requests = generate_requests(
            iters, B, T, L, min(Es), reuse=0.1, emulate_pruning=emulate_pruning
        )
        for req in requests:
            indices, offsets = req.unpack_2()
            indices = indices.to(dtype=indices_dtype)
            offsets = offsets.to(dtype=indices_dtype)
            output_ref = cc_ref(indices, offsets)

            # int_nbit_split_embedding_uvm_caching_codegen_lookup_function for UVM_CACHING.
            # using weights and other params from cc_ref, but
            # cache states from cc.
            output_uvm_caching = torch.ops.fbgemm.int_nbit_split_embedding_uvm_caching_codegen_lookup_function(
                dev_weights=(
                    cc_ref.weights_host if cc_ref.host_size > 0 else cc_ref.weights_dev
                ),
                uvm_weights=cc_ref.weights_uvm,
                weights_placements=cc_ref.weights_placements,
                weights_offsets=cc_ref.weights_offsets,
                weights_tys=cc_ref.weights_tys,
                D_offsets=cc_ref.D_offsets,
                total_D=cc_ref.total_D,
                max_int2_D=cc_ref.max_int2_D,
                max_int4_D=cc_ref.max_int4_D,
                max_int8_D=cc_ref.max_int8_D,
                max_float16_D=cc_ref.max_float16_D,
                max_float32_D=cc_ref.max_float32_D,
                indices=indices,
                offsets=offsets,
                pooling_mode=int(cc_ref.pooling_mode),
                indice_weights=None,
                output_dtype=cc_ref.output_dtype,
                lxu_cache_weights=cc.lxu_cache_weights,  # cc, not cc_ref.
                lxu_cache_locations=torch.empty(0, dtype=torch.int32).fill_(-1),
                row_alignment=cc_ref.row_alignment,
                max_float8_D=cc_ref.max_float8_D,
                fp8_exponent_bits=cc_ref.fp8_exponent_bits,
                fp8_exponent_bias=cc_ref.fp8_exponent_bias,
                # Additional args for UVM_CACHING: using cc, not cc_ref.
                cache_hash_size_cumsum=cc.cache_hash_size_cumsum,
                total_cache_hash_size=cc.total_cache_hash_size,
                cache_index_table_map=cc.cache_index_table_map,
                lxu_cache_state=cc.lxu_cache_state,
                lxu_state=cc.lxu_state,
            )
            torch.testing.assert_close(output_uvm_caching, output_ref, equal_nan=True)
            # cache status; we use the exact same logic, but still assigning ways in a associative cache can be
            # arbitrary. We compare sum along ways in each set, instead of expecting exact tensor match.
            cache_weights_ref = torch.reshape(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Tensor | Module`.
                cc_ref.lxu_cache_weights,
                [-1, associativity],
            )
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Tensor | Module`.
            cache_weights = torch.reshape(cc.lxu_cache_weights, [-1, associativity])
            torch.testing.assert_close(
                torch.sum(cache_weights_ref, 1),
                torch.sum(cache_weights, 1),
                equal_nan=True,
            )
            torch.testing.assert_close(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Tensor | Module`.
                torch.sum(cc.lxu_cache_state, 1),
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Tensor | Module`.
                torch.sum(cc_ref.lxu_cache_state, 1),
                equal_nan=True,
            )
            # lxu_state can be different as time_stamp values can be different.
            # we check the entries with max value.
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Tensor | Module`.
            max_timestamp_ref = torch.max(cc_ref.lxu_state)
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Tensor | Module`.
            max_timestamp_uvm_caching = torch.max(cc.lxu_state)
            x = cc_ref.lxu_state == max_timestamp_ref
            y = cc.lxu_state == max_timestamp_uvm_caching
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[bool,
            #  Tensor]`.
            torch.testing.assert_close(torch.sum(x, 1), torch.sum(y, 1))

            # int_nbit_split_embedding_uvm_caching_codegen_lookup_function for UVM.
            output_uvm = torch.ops.fbgemm.int_nbit_split_embedding_uvm_caching_codegen_lookup_function(
                dev_weights=(
                    cc_ref.weights_host if cc_ref.host_size > 0 else cc_ref.weights_dev
                ),
                uvm_weights=cc_ref.weights_uvm,
                weights_placements=placement_uvm,  # all UVM weights placement.
                weights_offsets=cc_ref.weights_offsets,
                weights_tys=cc_ref.weights_tys,
                D_offsets=cc_ref.D_offsets,
                total_D=cc_ref.total_D,
                max_int2_D=cc_ref.max_int2_D,
                max_int4_D=cc_ref.max_int4_D,
                max_int8_D=cc_ref.max_int8_D,
                max_float16_D=cc_ref.max_float16_D,
                max_float32_D=cc_ref.max_float32_D,
                indices=indices,
                offsets=offsets,
                pooling_mode=int(cc_ref.pooling_mode),
                indice_weights=None,
                output_dtype=cc_ref.output_dtype,
                lxu_cache_weights=zero_size_cache_weights,  # empty HBM cache.
                lxu_cache_locations=torch.empty(0, dtype=torch.int32).fill_(-1),
                row_alignment=cc_ref.row_alignment,
                max_float8_D=cc_ref.max_float8_D,
                fp8_exponent_bits=cc_ref.fp8_exponent_bits,
                fp8_exponent_bias=cc_ref.fp8_exponent_bias,
                # Additional args for UVM_CACHING; not needed for UVM.
                cache_hash_size_cumsum=None,
                total_cache_hash_size=None,
                cache_index_table_map=None,
                lxu_cache_state=None,
                lxu_state=None,
            )
            torch.testing.assert_close(output_uvm, output_ref, equal_nan=True)

    @given(
        weights_ty=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
                SparseType.INT8,
                SparseType.INT4,
                SparseType.INT2,
            ]
        ),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_int_nbit_split_embedding_cpu_mixed_indices_offsets_dtypes(
        self,
        weights_ty: SparseType,
    ) -> None:
        T = random.randint(1, 5)
        B = random.randint(1, 128)
        L = random.randint(1, 20)
        D = random.randint(2, 256)
        log_E = random.randint(3, 5)

        iters = 4
        E = int(10**log_E)

        D_alignment = (
            1 if weights_ty.bit_rate() % 8 == 0 else int(8 / weights_ty.bit_rate())
        )
        D = round_up(D, D_alignment)

        Ds = [D] * T
        Es = [E] * T
        cpu_locations = [EmbeddingLocation.HOST] * T

        requests = generate_requests(
            iters, B, T, L, min(Es), reuse=0.1, emulate_pruning=False, use_cpu=True
        )
        dtypes_combo = [
            (torch.int64, torch.int64),
            (torch.int32, torch.int32),
            (torch.int32, torch.int64),
            (torch.int64, torch.int32),
        ]
        for i, req in enumerate(requests):
            indices, offsets = req.unpack_2()
            indices_dtype, offsets_dtype = dtypes_combo[i]

            cc = IntNBitTableBatchedEmbeddingBagsCodegen(
                [("", E, D, weights_ty, M) for (E, D, M) in zip(Es, Ds, cpu_locations)],
                device=torch.device("cpu"),
                indices_dtype=indices_dtype,
            )
            cc.fill_random_weights()

            indices = indices.to(indices_dtype)
            offsets = offsets.to(offsets_dtype)
            _ = cc(indices, offsets)

    def _host_only_tbe(
        self, device: torch.device
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        return IntNBitTableBatchedEmbeddingBagsCodegen(
            [
                ("", 100, 16, SparseType.INT8, EmbeddingLocation.HOST),
                ("", 50, 32, SparseType.FP16, EmbeddingLocation.HOST),
            ],
            device=device,
        )

    def _host_only_requests(self) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, 50, (24,), dtype=torch.int32)
        offsets = torch.tensor([0, 2, 7, 11, 15, 20, 24], dtype=torch.int32)
        return indices, offsets

    def test_host_only_tbe_has_no_lxu_cache_weights(self) -> None:
        cc = self._host_only_tbe(torch.device("cpu"))
        cc.fill_random_weights()
        self.assertIsNone(cc.lxu_cache_weights)
        self.assertNotIn("lxu_cache_weights", cc.state_dict())
        indices, offsets = self._host_only_requests()
        self.assertEqual(cc(indices, offsets).size(), (3, 16 + 32))

    def test_host_only_tbe_prefetch_and_reset_cache_states_are_no_ops(self) -> None:
        cc = self._host_only_tbe(torch.device("cpu"))
        cc.fill_random_weights()
        indices, offsets = self._host_only_requests()
        expected = cc(indices, offsets)
        cc.prefetch(indices, offsets)
        cc.reset_cache_states()
        self.assertIsNone(cc.lxu_cache_weights)
        torch.testing.assert_close(
            cc(indices, offsets), expected, rtol=0, atol=0, equal_nan=True
        )

    def test_host_only_tbe_forward_does_not_pop_lxu_cache_locations(self) -> None:
        cc = self._host_only_tbe(torch.device("cpu"))
        cc.fill_random_weights()
        indices, offsets = self._host_only_requests()
        expected = cc(indices, offsets)
        cc.lxu_cache_locations_list = MagicMock()
        torch.testing.assert_close(
            cc(indices, offsets), expected, rtol=0, atol=0, equal_nan=True
        )
        cc.lxu_cache_locations_list.pop.assert_not_called()

    def test_reconcile_empty_lxu_cache_weights(self) -> None:
        empty = torch.zeros(0, 0, dtype=torch.uint8)
        non_empty = torch.zeros(4, 8, dtype=torch.uint8)
        no_cache = torch.nn.Module()
        no_cache.register_buffer("lxu_cache_weights", None)
        empty_cache = torch.nn.Module()
        empty_cache.register_buffer("lxu_cache_weights", empty)
        with_cache = torch.nn.Module()
        with_cache.register_buffer("lxu_cache_weights", non_empty)
        key = "tbes.0.lxu_cache_weights"
        other_key = "tbes.1.lxu_cache_weights"
        for test_name, module, state_dict, expected_keys in [
            ("no_cache_drops_empty_key", no_cache, {key: empty}, []),
            ("no_cache_keeps_non_empty_key", no_cache, {key: non_empty}, [key]),
            ("cache_keeps_empty_key", with_cache, {key: empty}, [key]),
            ("cache_keeps_non_empty_key", with_cache, {key: non_empty}, [key]),
            ("no_cache_keeps_other_prefix", no_cache, {other_key: empty}, [other_key]),
            ("no_cache_without_key", no_cache, {}, []),
            ("empty_cache_adds_missing_key", empty_cache, {}, [key]),
            ("empty_cache_keeps_existing_key", empty_cache, {key: empty}, [key]),
            ("cache_does_not_add_missing_key", with_cache, {}, []),
        ]:
            with self.subTest(test_name):
                _reconcile_empty_lxu_cache_weights(module, state_dict, "tbes.0.")
                self.assertEqual(list(state_dict), expected_keys)
        state_dict = {}
        _reconcile_empty_lxu_cache_weights(empty_cache, state_dict, "tbes.0.")
        self.assertEqual(state_dict[key].shape, empty.shape)

    def test_module_with_empty_lxu_cache_weights_loads_cpu_state_dict(
        self,
    ) -> None:
        cpu_model = torch.nn.Module()
        cpu_model.tbes = torch.nn.ModuleList([self._host_only_tbe(torch.device("cpu"))])
        state_dict = cpu_model.state_dict()
        self.assertNotIn("tbes.0.lxu_cache_weights", state_dict)
        model = torch.nn.Module()
        model.tbes = torch.nn.ModuleList([self._host_only_tbe(torch.device("cpu"))])
        tbe = model.tbes[0]
        tbe.register_buffer("lxu_cache_weights", torch.zeros(0, 0, dtype=torch.uint8))
        model.load_state_dict(state_dict, strict=True)
        self.assertEqual(tbe.lxu_cache_weights.numel(), 0)

    def test_host_only_tbe_loads_state_dict_with_empty_lxu_cache_weights(
        self,
    ) -> None:
        cc = self._host_only_tbe(torch.device("cpu"))
        state_dict = cc.state_dict()
        state_dict["lxu_cache_weights"] = torch.zeros(0, 0, dtype=torch.uint8)
        cc.load_state_dict(state_dict, strict=True)
        self.assertIsNone(cc.lxu_cache_weights)
        self.assertIn("lxu_cache_weights", state_dict)

    def test_nested_host_only_tbe_loads_state_dict_with_empty_lxu_cache_weights(
        self,
    ) -> None:
        model = torch.nn.Module()
        model.tbes = torch.nn.ModuleList([self._host_only_tbe(torch.device("cpu"))])
        state_dict = model.state_dict()
        state_dict["tbes.0.lxu_cache_weights"] = torch.zeros(0, 0, dtype=torch.uint8)
        model.load_state_dict(state_dict, strict=True)
        self.assertIsNone(model.tbes[0].lxu_cache_weights)

    def test_host_only_tbe_rejects_state_dict_with_non_empty_lxu_cache_weights(
        self,
    ) -> None:
        cc = self._host_only_tbe(torch.device("cpu"))
        state_dict = cc.state_dict()
        state_dict["lxu_cache_weights"] = torch.zeros(4, 8, dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key.*lxu_cache_weights"):
            cc.load_state_dict(state_dict, strict=True)

    def test_host_only_tbe_moved_from_meta_to_cpu_has_no_lxu_cache_weights(
        self,
    ) -> None:
        cc = self._host_only_tbe(torch.device("meta"))
        cc.initialize_weights()
        self.assertIsNone(cc.lxu_cache_weights)
        cc.move_to_device_with_cache(torch.device("cpu"), cache_load_factor=0.0)
        self.assertEqual(cc.current_device, torch.device("cpu"))
        self.assertIsNone(cc.lxu_cache_weights)
        cc.fill_random_weights()
        indices, offsets = self._host_only_requests()
        self.assertEqual(cc(indices, offsets).size(), (3, 16 + 32))

    def test_device_tbe_moved_from_meta_to_cpu_has_no_lxu_cache_weights(
        self,
    ) -> None:
        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            [
                ("", 100, 16, SparseType.INT8, EmbeddingLocation.DEVICE),
                ("", 50, 32, SparseType.FP16, EmbeddingLocation.DEVICE),
            ],
            device=torch.device("meta"),
        )
        cc.initialize_weights()
        self.assertFalse(cc.use_cpu)
        self.assertIsNotNone(cc.lxu_cache_weights)
        cc.move_to_device_with_cache(torch.device("cpu"), cache_load_factor=0.0)
        self.assertTrue(cc.use_cpu)
        self.assertIsNone(cc.lxu_cache_weights)
        self.assertNotIn("lxu_cache_weights", cc.state_dict())
        cc.fill_random_weights()
        indices, offsets = self._host_only_requests()
        self.assertEqual(cc(indices, offsets).size(), (3, 16 + 32))


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import gc
import tempfile
import unittest
from typing import Any, Dict
from unittest import TestCase

import fbgemm_gpu  # noqa E402
import torch
import torch.testing
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import BackendType
from fbgemm_gpu.utils.loader import load_torch_module
from hypothesis import given, settings, strategies as st, Verbosity

from .. import common  # noqa E402
from ..common import open_source, running_in_oss

if not open_source:
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings",
    )

MAX_EXAMPLES = 20
MAX_D = 256
default_settings: Dict[str, Any] = {
    "verbosity": Verbosity.verbose,
    "max_examples": MAX_EXAMPLES,
    "deadline": None,
}


@unittest.skipIf(*running_in_oss)
class KvTensorWrapperTest(TestCase):

    def create_db(
        self,
        ssd_directory: str,
        backend_type: BackendType,
        max_D: int,
        D: int,
        E: int,
        weights_precision: SparseType,
    ) -> object:
        if backend_type == BackendType.SSD:
            return torch.classes.fbgemm.EmbeddingRocksDBWrapper(
                ssd_directory,
                8,  # num_shards
                8,  # num_threads
                0,  # ssd_memtable_flush_period,
                0,  # ssd_memtable_flush_offset,
                4,  # ssd_l0_files_per_compact,
                max_D,  # embedding_dim
                0,  # ssd_rate_limit_mbps,
                1,  # ssd_size_ratio,
                8,  # ssd_compaction_trigger,
                536870912,  # 512MB ssd_write_buffer_size,
                8,  # ssd_max_write_buffer_num,
                -0.01,  # ssd_uniform_init_lower
                0.01,  # ssd_uniform_init_upper
                weights_precision.bit_rate(),  # row_storage_bitwidth
                10 * (2**20),  # block cache size
                enable_async_update=False,
                table_dims=torch.tensor([D]),
                hash_size_cumsum=torch.tensor([0, E]),
            )
        elif backend_type == BackendType.DRAM:
            return torch.classes.fbgemm.DramKVEmbeddingCacheWrapper(
                max_D=max_D,
                uniform_init_lower=-0.1,
                uniform_init_upper=0.1,
                num_shards=8,
                num_threads=32,
                row_storage_bitwidth=weights_precision.bit_rate(),
                table_dims=torch.tensor([D]),
                hash_size_cumsum=torch.tensor([0, E]),
                enable_async_update=False,
            )

    # pyre-ignore[56]
    @given(
        precision=st.sampled_from(
            [
                (SparseType.FP16, 16),
                (SparseType.FP32, 32),
            ]
        ),
        D=st.integers(min_value=64, max_value=MAX_D),
        backend_type=st.sampled_from([BackendType.DRAM, BackendType.SSD]),
    )
    @settings(**default_settings)
    def test_read_tensor_using_wrapper_from_db(
        self, precision: tuple[SparseType, int], D: int, backend_type: BackendType
    ) -> None:
        E = int(1e4)
        max_D = MAX_D  # max emb dimension seen by rocksDB
        N = 1000
        weights_precision, dtype_width = precision
        weights_dtype = weights_precision.as_dtype()

        with tempfile.TemporaryDirectory() as ssd_directory:
            ssd_db = self.create_db(
                ssd_directory, backend_type, max_D, D, E, weights_precision
            )

            # create random index tensor with size N
            indices = torch.randperm(N)
            # insert the weights with the corresponding indices into the table
            weights = torch.arange(N * D, dtype=weights_dtype).view(N, D)
            padded_weights = torch.nn.functional.pad(weights, (0, max_D - D))
            count = torch.tensor([N])
            ssd_db.set(indices, padded_weights, count)

            # create a view tensor wrapper
            snapshot = (
                ssd_db.create_snapshot() if backend_type == BackendType.SSD else None
            )
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, D], weights.dtype, 0, snapshot
            )
            if backend_type == BackendType.SSD:
                tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
            elif backend_type == BackendType.DRAM:
                tensor_wrapper.set_dram_db_wrapper(ssd_db)
            self.assertEqual(tensor_wrapper.shape, [E, D])

            # read one row as an extreme case
            narrowed = tensor_wrapper.narrow(0, 0, 1)

            # table has a total of E rows
            # load 1000 rows at a time
            step = 1000
            for i in range(0, E, step):
                narrowed = tensor_wrapper.narrow(0, i, step)
                for weight_ind, v in enumerate(indices):
                    j = v.item()
                    if j < i or j >= i + step:
                        continue
                    self.assertTrue(
                        torch.equal(narrowed[j % step], weights[weight_ind]),
                        msg=(
                            f"Tensor value mismatch at row {j}:\n"
                            f"actual\n{narrowed[j % step]}\n\nexpected\n{weights[weight_ind]}"
                        ),
                    )

            del tensor_wrapper
            del snapshot
            if backend_type == BackendType.SSD:
                self.assertEqual(ssd_db.get_snapshot_count(), 0)

    # pyre-ignore[56]
    @given(
        precision=st.sampled_from(
            [
                (SparseType.FP16, 16),
                (SparseType.FP32, 32),
            ]
        ),
        D=st.integers(min_value=64, max_value=MAX_D),
        backend_type=st.sampled_from([BackendType.DRAM, BackendType.SSD]),
    )
    @settings(**default_settings)
    def test_write_tensor_to_db(
        self, precision: tuple[SparseType, int], D: int, backend_type: BackendType
    ) -> None:
        E = int(1e4)  # num total rows
        max_D = MAX_D  # max emb dimension seen by rocksDB
        N = 1000  # window size
        weights_precision, dtype_width = precision
        weights_dtype = weights_precision.as_dtype()

        table_offsets = [0, E]

        with tempfile.TemporaryDirectory() as ssd_directory:
            ssd_db = self.create_db(
                ssd_directory, backend_type, max_D, D, E, weights_precision
            )
            weights = [
                torch.randn(N * D, dtype=weights_dtype).view(N, D),
                torch.randn(N * D, dtype=weights_dtype).view(N, D),
            ]

            for table_idx, offset in enumerate(table_offsets):
                # no snapshot needed for writing to rocksdb
                tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                    [E, D], weights[table_idx].dtype, offset
                )
                if backend_type == BackendType.SSD:
                    tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
                elif backend_type == BackendType.DRAM:
                    tensor_wrapper.set_dram_db_wrapper(ssd_db)
                step = N
                for i in range(0, E, step):
                    tensor_wrapper.set_range(0, i, step, weights[table_idx])

            # create a view tensor wrapper
            snapshot = (
                ssd_db.create_snapshot() if backend_type == BackendType.SSD else None
            )

            for table_idx, offset in enumerate(table_offsets):
                wrong_tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                    [E, D], weights[table_idx].dtype, 1, snapshot
                )
                tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                    [E, D], weights[table_idx].dtype, offset, snapshot
                )
                if backend_type == BackendType.SSD:
                    wrong_tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
                    tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
                elif backend_type == BackendType.DRAM:
                    wrong_tensor_wrapper.set_dram_db_wrapper(ssd_db)
                    tensor_wrapper.set_dram_db_wrapper(ssd_db)
                self.assertEqual(tensor_wrapper.shape, [E, D])

                # table has a total of E rows
                # load 1000 rows at a time
                step = N
                for i in range(0, E, step):
                    narrowed = tensor_wrapper.narrow(0, i, step)
                    self.assertTrue(
                        torch.equal(narrowed, weights[table_idx]),
                        msg=(
                            f"Tensor value mismatch :\n"
                            f"actual\n{narrowed}\n\nexpected\n{weights[table_idx]}"
                        ),
                    )

                    wrong_narrowed = wrong_tensor_wrapper.narrow(0, i, step)
                    self.assertTrue(
                        not torch.equal(wrong_narrowed, weights[table_idx]),
                        msg=(
                            f"Tensor value shouldn't match :\n"
                            f"actual\n{wrong_narrowed}\n\nexpected\n{weights[table_idx]}"
                        ),
                    )
                del wrong_tensor_wrapper
                del tensor_wrapper

            del snapshot
            if backend_type == BackendType.SSD:
                self.assertEqual(ssd_db.get_snapshot_count(), 0)

    # pyre-ignore[56]
    @given(
        precision=st.sampled_from(
            [
                (SparseType.FP16, 16),
                (SparseType.FP32, 32),
            ]
        ),
        D=st.integers(min_value=64, max_value=MAX_D),
        backend_type=st.sampled_from([BackendType.DRAM, BackendType.SSD]),
    )
    @settings(**default_settings)
    def test_discrete_id_weights_io(
        self, precision: tuple[SparseType, int], D: int, backend_type: BackendType
    ) -> None:
        E = int(1e4)  # num total rows
        max_D = MAX_D  # max emb dimension seen by rocksDB
        N = 1000  # window size
        weights_precision, dtype_width = precision
        weights_dtype = weights_precision.as_dtype()

        table_offsets = [0, N]

        with tempfile.TemporaryDirectory() as ssd_directory:
            ssd_db = self.create_db(
                ssd_directory, backend_type, max_D, D, E, weights_precision
            )
            indices = torch.randperm(N)
            weights = [
                torch.randn(N * D, dtype=weights_dtype).view(N, D),
                torch.randn(N * D, dtype=weights_dtype).view(N, D),
            ]

            new_weights_after_snapshot = (
                torch.randn(N, D, dtype=weights_dtype)
                if backend_type == BackendType.SSD
                else None
            )

            # no snapshot needed for writing to rocksdb
            for table_idx, offset in enumerate(table_offsets):
                tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                    [E, D], weights[table_idx].dtype, offset
                )
                if backend_type == BackendType.SSD:
                    tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
                elif backend_type == BackendType.DRAM:
                    tensor_wrapper.set_dram_db_wrapper(ssd_db)
                tensor_wrapper.set_weights_and_ids(weights[table_idx], indices)

            # create a view tensor wrapper
            snapshot = (
                ssd_db.create_snapshot() if backend_type == BackendType.SSD else None
            )

            for table_idx, offset in enumerate(table_offsets):
                tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                    [E, D], weights[table_idx].dtype, offset
                )
                if backend_type == BackendType.SSD:
                    tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
                    tensor_wrapper.set_weights_and_ids(
                        new_weights_after_snapshot, indices
                    )
                elif backend_type == BackendType.DRAM:
                    tensor_wrapper.set_dram_db_wrapper(ssd_db)
            for table_idx, offset in enumerate(table_offsets):
                wrong_tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                    [E, D], weights[table_idx].dtype, 1, snapshot
                )
                tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                    [E, D], weights[table_idx].dtype, offset, snapshot
                )
                if backend_type == BackendType.SSD:
                    wrong_tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
                    tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
                elif backend_type == BackendType.DRAM:
                    wrong_tensor_wrapper.set_dram_db_wrapper(ssd_db)
                    tensor_wrapper.set_dram_db_wrapper(ssd_db)
                self.assertEqual(tensor_wrapper.shape, [E, D])

                wrong_out_weights = wrong_tensor_wrapper.get_weights_by_ids(indices)
                self.assertTrue(
                    not torch.equal(wrong_out_weights, weights[table_idx]),
                    msg=(
                        f"Tensor value should be mismatch but actually matches:\n"
                        f"actual\n{wrong_out_weights}\n\nexpected\n{weights[table_idx]}"
                    ),
                )

                out_weights = tensor_wrapper.get_weights_by_ids(indices)
                self.assertTrue(
                    torch.equal(out_weights, weights[table_idx]),
                    msg=(
                        f"Tensor value mismatch :\n"
                        f"actual\n{out_weights}\n\nexpected\n{weights[table_idx]}"
                    ),
                )
                del tensor_wrapper
                del wrong_tensor_wrapper

            del snapshot
            if backend_type == BackendType.SSD:
                self.assertEqual(ssd_db.get_snapshot_count(), 0)

    # pyre-ignore[56]
    @given(
        precision=st.sampled_from(
            [
                (SparseType.FP16, 16),
                (SparseType.FP32, 32),
            ]
        ),
        D=st.integers(min_value=8, max_value=16),
        backend_type=st.sampled_from([BackendType.DRAM, BackendType.SSD]),
    )
    @settings(**default_settings)
    def test_narrow_mapping_offset_to_weight_id(
        self, precision: tuple[SparseType, int], D: int, backend_type: BackendType
    ) -> None:
        E = int(1e4)  # num total rows
        max_D = MAX_D  # max emb dimension seen by rocksDB
        # N = 1000  # window size
        N = 10
        weights_precision, dtype_width = precision
        weights_dtype = weights_precision.as_dtype()

        with tempfile.TemporaryDirectory() as ssd_directory:
            ssd_db = self.create_db(
                ssd_directory, backend_type, max_D, D, E, weights_precision
            )
            indices = torch.randperm(N)
            weights = torch.arange(N * D, dtype=weights_dtype).view(N, D)
            new_weights_after_snapshot = torch.randn(N, D, dtype=weights_dtype)

            # no snapshot needed for writing to rocksdb
            tensor_wrapper0 = torch.classes.fbgemm.KVTensorWrapper(
                [E, D], weights.dtype, 0
            )
            if backend_type == BackendType.SSD:
                tensor_wrapper0.set_embedding_rocks_dp_wrapper(ssd_db)
            elif backend_type == BackendType.DRAM:
                tensor_wrapper0.set_dram_db_wrapper(ssd_db)
            tensor_wrapper0.set_weights_and_ids(weights, indices)

            # create a view tensor wrapper
            snapshot = (
                ssd_db.create_snapshot() if backend_type == BackendType.SSD else None
            )
            tensor_wrapper0.set_weights_and_ids(new_weights_after_snapshot, indices)
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, D], weights.dtype, 0, snapshot, indices
            )
            if backend_type == BackendType.SSD:
                tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
            elif backend_type == BackendType.DRAM:
                tensor_wrapper.set_dram_db_wrapper(ssd_db)
            self.assertEqual(tensor_wrapper.shape, [E, D])
            indices_copy = indices.clone()
            del indices
            gc.collect()

            # read one row as an extreme case
            narrowed = tensor_wrapper.narrow(0, 0, 1)
            expected_narrowed = tensor_wrapper.get_weights_by_ids(indices_copy[0:1])
            self.assertTrue(
                torch.equal(narrowed, expected_narrowed),
                msg=(
                    f"Tensor value mismatch :\n"
                    f"actual\n{narrowed}\n\nexpected\n{expected_narrowed}"
                ),
            )

            # table has a total of E rows
            # load 1000 rows at a time
            step = 1000
            for i in range(0, E, step):
                narrowed_weights = tensor_wrapper.narrow(0, i, step)
                sliced_indices = indices_copy[i : i + step]
                expected_weights = tensor_wrapper.get_weights_by_ids(sliced_indices)

                self.assertTrue(
                    torch.equal(narrowed_weights, expected_weights),
                    msg=(
                        f"Tensor value mismatch :\n"
                        f"actual\n{narrowed}\n\nexpected\n{weights}"
                    ),
                )
            del tensor_wrapper0
            del tensor_wrapper
            del snapshot
            if backend_type == BackendType.SSD:
                self.assertEqual(ssd_db.get_snapshot_count(), 0)

    # pyre-ignore[56]
    @given(
        precision=st.sampled_from(
            [
                (SparseType.FP16, 16),
                (SparseType.FP32, 32),
            ]
        ),
        D=st.integers(min_value=64, max_value=MAX_D - 10),  # D = max_D - extra_width
        backend_type=st.sampled_from([BackendType.SSD, BackendType.DRAM]),
    )
    @settings(**default_settings)
    def test_read_weights_with_specified_width(
        self, precision: tuple[SparseType, int], D: int, backend_type: BackendType
    ) -> None:
        E = int(1e4)
        extra_width = 10
        width_offset = 10
        max_D = MAX_D
        N = 1000
        weights_precision, dtype_width = precision
        weights_dtype = weights_precision.as_dtype()

        with tempfile.TemporaryDirectory() as ssd_directory:
            ssd_db = self.create_db(
                ssd_directory, backend_type, max_D, D, E, weights_precision
            )
            # create random index tensor with size N
            indices = torch.arange(N)
            # insert the weights with the corresponding indices into the table
            weights = torch.arange(N * D, dtype=weights_dtype).view(N, D)
            padded_weights = torch.nn.functional.pad(weights, (0, max_D - D), value=1.0)

            count = torch.tensor([N])
            ssd_db.set(indices, padded_weights, count)

            """
            create a new case with kvt dim == weight dim
            test different width offset to read the weights out
            """
            # case 0
            snapshot = (
                ssd_db.create_snapshot() if backend_type == BackendType.SSD else None
            )
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, D], weights.dtype, 0, snapshot
            )
            if backend_type == BackendType.SSD:
                tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
            elif backend_type == BackendType.DRAM:
                tensor_wrapper.set_dram_db_wrapper(ssd_db)
            self.assertEqual(tensor_wrapper.shape, [E, D])

            narrowed = tensor_wrapper.narrow(0, 0, 1)
            weight_by_id = tensor_wrapper.get_weights_by_ids(torch.tensor([0]))
            self.assertTrue(torch.equal(narrowed[0], weights[0]))
            self.assertTrue(torch.equal(weight_by_id[0], weights[0]))

            # case 1
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, D - width_offset],
                weights.dtype,
                0,
                snapshot,
                width_offset=width_offset,
            )
            if backend_type == BackendType.SSD:
                tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
            elif backend_type == BackendType.DRAM:
                tensor_wrapper.set_dram_db_wrapper(ssd_db)
            narrowed = tensor_wrapper.narrow(0, 0, 1)
            weight_by_id = tensor_wrapper.get_weights_by_ids(torch.tensor([0]))
            self.assertTrue(torch.equal(narrowed[0], weights[0][width_offset:]))
            self.assertTrue(torch.equal(weight_by_id[0], weights[0][width_offset:]))

            """
            create a new case with kvt dim > weight dim
            we should only get upto weight dim and the rest should be 0s
            """
            new_D = D + extra_width
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, new_D], weights.dtype, 0, snapshot
            )

            if backend_type == BackendType.SSD:
                tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
            elif backend_type == BackendType.DRAM:
                tensor_wrapper.set_dram_db_wrapper(ssd_db)
            self.assertEqual(tensor_wrapper.shape, [E, new_D])

            narrowed = tensor_wrapper.narrow(0, 0, 1)
            weight_by_id = tensor_wrapper.get_weights_by_ids(torch.tensor([0]))
            if backend_type == BackendType.SSD:
                self.assertTrue(torch.equal(narrowed[0], padded_weights[0][:new_D]))
            elif backend_type == BackendType.DRAM:
                self.assertTrue(torch.equal(narrowed[0], padded_weights[0][:new_D]))

            width_offset = 10
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, new_D - width_offset],
                weights.dtype,
                0,
                snapshot,
                width_offset=width_offset,
            )
            if backend_type == BackendType.SSD:
                tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
            elif backend_type == BackendType.DRAM:
                tensor_wrapper.set_dram_db_wrapper(ssd_db)
            narrowed = tensor_wrapper.narrow(0, 0, 1)
            weight_by_id = tensor_wrapper.get_weights_by_ids(torch.tensor([0]))
            self.assertTrue(
                torch.equal(narrowed[0], padded_weights[0][width_offset:new_D])
            )
            self.assertTrue(
                torch.equal(weight_by_id[0], padded_weights[0][width_offset:new_D])
            )

            # non existing rows, we should expect bytes beyond new_D to be 0s
            narrowed = tensor_wrapper.narrow(0, N, 1)
            weight_by_id = tensor_wrapper.get_weights_by_ids(torch.tensor([N]))
            self.assertTrue(all(narrowed[0][D:] == 0))
            self.assertTrue(all(weight_by_id[0][D:] == 0))

            # with width offset
            narrowed = tensor_wrapper.narrow(0, N, 1)
            weight_by_id = tensor_wrapper.get_weights_by_ids(torch.tensor([N]))
            self.assertTrue(all(narrowed[0][D - width_offset :] == 0))
            self.assertTrue(all(weight_by_id[0][D - width_offset :] == 0))

            del tensor_wrapper
            del snapshot
            if backend_type == BackendType.SSD:
                self.assertEqual(ssd_db.get_snapshot_count(), 0)

    def test_dram_kv_and_rdb_snapshot_check(self) -> None:
        max_D = MAX_D  # max emb dimension seen by rocksDB
        D = 64
        E = int(1e4)
        weights_precision = SparseType.FP16
        dtype_width = 16
        weights_dtype = weights_precision.as_dtype()

        dram_kv = torch.classes.fbgemm.DramKVEmbeddingCacheWrapper(
            max_D=max_D,
            uniform_init_lower=-0.1,
            uniform_init_upper=0.1,
            num_shards=8,
            num_threads=32,
            row_storage_bitwidth=weights_precision.bit_rate(),
        )

        with tempfile.TemporaryDirectory() as ssd_directory:
            ssd_db = torch.classes.fbgemm.EmbeddingRocksDBWrapper(
                ssd_directory,
                8,  # num_shards
                8,  # num_threads
                0,  # ssd_memtable_flush_period,
                0,  # ssd_memtable_flush_offset,
                4,  # ssd_l0_files_per_compact,
                max_D,  # embedding_dim
                0,  # ssd_rate_limit_mbps,
                1,  # ssd_size_ratio,
                8,  # ssd_compaction_trigger,
                536870912,  # 512MB ssd_write_buffer_size,
                8,  # ssd_max_write_buffer_num,
                -0.01,  # ssd_uniform_init_lower
                0.01,  # ssd_uniform_init_upper
                dtype_width,  # row_storage_bitwidth
                10 * (2**20),  # block cache size
            )
            snapshot = ssd_db.create_snapshot()
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, D], weights_dtype, 0, snapshot
            )
            tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
            tensor_wrapper.narrow(0, 0, 1)
            tensor_wrapper.get_weights_by_ids(torch.tensor([1]))

            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, D], weights_dtype, 0
            )
            tensor_wrapper.set_embedding_rocks_dp_wrapper(ssd_db)
            with self.assertRaises(RuntimeError):
                tensor_wrapper.narrow(0, 0, 1)
            with self.assertRaises(RuntimeError):
                tensor_wrapper.get_weights_by_ids(torch.tensor([1]))

            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, D], weights_dtype, 0
            )
            tensor_wrapper.set_dram_db_wrapper(dram_kv)
            tensor_wrapper.narrow(0, 0, 1)
            tensor_wrapper.get_weights_by_ids(torch.tensor([1]))

            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                [E, D], weights_dtype, 0, snapshot
            )
            tensor_wrapper.set_dram_db_wrapper(dram_kv)

            with self.assertRaises(RuntimeError):
                tensor_wrapper.narrow(0, 0, 1)
            with self.assertRaises(RuntimeError):
                tensor_wrapper.get_weights_by_ids(torch.tensor([1]))

    def test_dram_kv_read_only_mode(self) -> None:
        max_D = MAX_D  # max emb dimension seen dram backend
        D = 64
        N = 10  # window size
        E = int(1e4)
        weights_precision = SparseType.FP16
        weights_dtype = weights_precision.as_dtype()

        dram_kv = torch.classes.fbgemm.DramKVEmbeddingCacheWrapper(
            max_D=max_D,
            uniform_init_lower=-0.1,
            uniform_init_upper=0.1,
            num_shards=8,
            num_threads=32,
            row_storage_bitwidth=weights_precision.bit_rate(),
        )

        # create random index tensor with size N
        indices = torch.arange(N)
        # insert the weights with the corresponding indices into the table
        weights = torch.arange(N * D, dtype=weights_dtype).view(N, D)
        padded_weights = torch.nn.functional.pad(weights, (0, max_D - D), value=1.0)
        count = torch.tensor([N])
        dram_kv.set(indices, padded_weights, count)

        tensor_wrapper_read_only = torch.classes.fbgemm.KVTensorWrapper(
            shape=[E, D], dtype=weights_dtype, row_offset=0, read_only=True
        )
        tensor_wrapper_read_only.set_dram_db_wrapper(dram_kv)

        # Get the weights that are already stored in the DRAM KV cache
        narrowed_weights = tensor_wrapper_read_only.narrow(0, 0, N)
        weights_by_ids = tensor_wrapper_read_only.get_weights_by_ids(indices)
        self.assertTrue(
            torch.equal(narrowed_weights, weights),
            msg=(
                f"Tensor value mismatch :\n"
                f"actual\n{narrowed_weights}\n\nexpected\n{weights}"
            ),
        )
        self.assertTrue(
            torch.equal(weights_by_ids, weights),
            msg=(
                f"Tensor value mismatch :\n"
                f"actual\n{weights_by_ids}\n\nexpected\n{weights}"
            ),
        )

        # Try to set_range() on a read-only tensor wrapper, which should be no-op
        insert_weight = torch.randn(D, dtype=weights_dtype).view(1, D)
        tensor_wrapper_read_only.set_range(0, N, 1, insert_weight)

        # narrow from the above, which should not match the original weights
        narrowed_weight = tensor_wrapper_read_only.narrow(0, N, 1)
        self.assertTrue(
            not torch.equal(narrowed_weight, insert_weight),
            msg=(
                f"Tensor value should not match :\n"
                f"actual\n{narrowed_weight}\n\nexpected\n{insert_weight}"
            ),
        )

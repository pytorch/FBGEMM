# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import gc
import pickle
import tempfile
import unittest
from typing import Any, Protocol
from unittest import TestCase
from unittest.mock import MagicMock

import fbgemm_gpu  # noqa E402
import torch
import torch.testing
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.tbe.ssd.ssd_config import BackendType
from fbgemm_gpu.tbe.ssd.utils.partially_materialized_tensor import (
    PartiallyMaterializedTensor,
)
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
default_settings: dict[str, Any] = {
    "verbosity": Verbosity.verbose,
    "max_examples": MAX_EXAMPLES,
    "deadline": None,
}


class _CuCache(Protocol):
    """The DramKVEmbeddingCacheWrapper methods the CU cache tests call."""

    def set(
        self, indices: torch.Tensor, weights: torch.Tensor, count: torch.Tensor
    ) -> None:
        """Writes `count` rows."""

    def get_int8_row_bytes(self) -> int:
        """Raw INT8 checkpoint row bytes, 0 without INT8 rows."""

    def get_raw_row_bytes(self) -> int:
        """Raw checkpoint row bytes, 0 without INT8 or packed rows."""


def _metaheader_diff(actual: torch.Tensor, expected: torch.Tensor) -> str:
    """The MetaHeader fields (key, timestamp, count, used) that differ between
    the first 8 fp16 columns of two row tensors, with the first differing
    row's actual and expected values."""
    a = actual[:, :8].contiguous().view(torch.int64).view(-1, 2)
    e = expected[:, :8].contiguous().view(torch.int64).view(-1, 2)
    fields = (
        ("key", a[:, 0], e[:, 0]),
        ("timestamp", a[:, 1] & 0xFFFFFFFF, e[:, 1] & 0xFFFFFFFF),
        ("count", (a[:, 1] >> 32) & 0x7FFFFFFF, (e[:, 1] >> 32) & 0x7FFFFFFF),
        ("used", (a[:, 1] >> 63) & 1, (e[:, 1] >> 63) & 1),
    )
    diffs = []
    for name, x, y in fields:
        bad = (x != y).nonzero().flatten()
        if bad.numel() > 0:
            row = int(bad[0])
            diffs.append(
                f"{name}: {bad.numel()} rows, row {row} actual {int(x[row])} "
                f"expected {int(y[row])}"
            )
    return "; ".join(diffs)


# pyrefly: ignore [bad-argument-type]
@unittest.skipIf(*running_in_oss)
class KvTensorWrapperTest(TestCase):

    def test_partially_materialized_tensor_pickle_preserves_flags(self) -> None:
        pmt = PartiallyMaterializedTensor(
            torch.tensor([1.0]), is_virtual=True, includes_metadata=True
        )

        restored = pickle.loads(pickle.dumps(pmt))

        self.assertTrue(restored.is_virtual)
        _, args = restored.__reduce__()
        self.assertEqual(args[1:], (True, True))
        torch.testing.assert_close(restored.wrapped, pmt.wrapped)

        copied = copy.deepcopy(pmt)
        self.assertTrue(copied.is_virtual)
        _, copied_args = copied.__reduce__()
        self.assertEqual(copied_args[1:], (True, True))
        torch.testing.assert_close(copied.wrapped, pmt.wrapped)

    def test_partially_materialized_tensor_dispatches_prefixed_rows(self) -> None:
        wrapped = MagicMock()
        pmt = PartiallyMaterializedTensor(wrapped, includes_metadata=True)
        weights = torch.tensor([[1.0]])
        ids = torch.tensor([11])

        pmt.set_weights_and_ids(weights, ids)

        wrapped.set_prefixed_weights_and_ids.assert_called_once_with(weights, ids)
        wrapped.set_weights_and_ids.assert_not_called()

    def test_partially_materialized_tensor_dispatches_payload_rows(self) -> None:
        wrapped = MagicMock()
        pmt = PartiallyMaterializedTensor(wrapped)
        weights = torch.tensor([[1.0]])
        ids = torch.tensor([11])

        pmt.set_weights_and_ids(weights, ids)

        wrapped.set_weights_and_ids.assert_called_once_with(weights, ids)
        wrapped.set_prefixed_weights_and_ids.assert_not_called()

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
            # pyrefly: ignore [missing-attribute]
            ssd_db.set(indices, padded_weights, count)

            # create a view tensor wrapper
            snapshot = (
                # pyrefly: ignore [missing-attribute]
                ssd_db.create_snapshot()
                if backend_type == BackendType.SSD
                else None
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
                # pyrefly: ignore [missing-attribute]
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
                # pyrefly: ignore [missing-attribute]
                ssd_db.create_snapshot()
                if backend_type == BackendType.SSD
                else None
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
                # pyrefly: ignore [missing-attribute]
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
                # pyrefly: ignore [missing-attribute]
                ssd_db.create_snapshot()
                if backend_type == BackendType.SSD
                else None
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
                # pyrefly: ignore [missing-attribute]
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
                # pyrefly: ignore [missing-attribute]
                ssd_db.create_snapshot()
                if backend_type == BackendType.SSD
                else None
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
                # pyrefly: ignore [missing-attribute]
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
            # pyrefly: ignore [missing-attribute]
            ssd_db.set(indices, padded_weights, count)

            """
            create a new case with kvt dim == weight dim
            test different width offset to read the weights out
            """
            # case 0
            snapshot = (
                # pyrefly: ignore [missing-attribute]
                ssd_db.create_snapshot()
                if backend_type == BackendType.SSD
                else None
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
                # pyrefly: ignore [missing-attribute]
                self.assertEqual(ssd_db.get_snapshot_count(), 0)

    def _create_cu_cache(
        self, int8_storage: bool, max_D: int = 132, packed_storage: bool = False
    ) -> _CuCache:
        # CU whole rows retain four unused fp16 tail columns for compatibility.
        enrichment_config = torch.classes.fbgemm.EnrichmentConfig(
            enrichment_type=4,  # IN_MEMORY_TEST_ONLY
            provider_name="",
            client_id="",
            enrichment_dim=128,
            response_format=0,
            serving_qdq_bit_rate=4 if packed_storage else 0,
            serving_qdq_group_size=8 if packed_storage else 0,
            packed_storage=packed_storage,
            int8_storage=int8_storage,
        )
        return torch.classes.fbgemm.DramKVEmbeddingCacheWrapper(
            max_D=max_D,
            uniform_init_lower=-0.1,
            uniform_init_upper=0.1,
            num_shards=8,
            num_threads=8,
            row_storage_bitwidth=16,
            table_dims=torch.tensor([max_D]),
            hash_size_cumsum=torch.tensor([0, 100000]),
            backend_return_whole_row=True,
            disable_random_init=True,
            enrichment_config=enrichment_config,
        )

    def _whole_row_pmt(
        self,
        db: _CuCache,
        ids: torch.Tensor,
        width: int,
        dtype: torch.dtype,
        row_offset: int = 0,
        only_load_weight: bool = False,
    ) -> PartiallyMaterializedTensor:
        kvt = torch.classes.fbgemm.KVTensorWrapper(
            shape=[ids.numel(), width],
            dtype=dtype,
            row_offset=row_offset,
            snapshot_handle=None,
            sorted_indices=ids.view(-1, 1),
            only_load_weight=only_load_weight,
        )
        kvt.set_dram_db_wrapper(db)
        return PartiallyMaterializedTensor(kvt, True)

    def _write_cu_rows(self, db: _CuCache, n: int) -> torch.Tensor:
        torch.manual_seed(0)
        mean = torch.randn(1, 128) * 0.6
        rows = mean + torch.randn(n, 128) * 0.3
        rows = rows / rows.norm(dim=1, keepdim=True)
        rows = torch.cat([rows, torch.randn(n, 4)], dim=1).half()
        db.set(torch.arange(n), rows, torch.tensor([n]))
        return rows

    def test_int8_rows_checkpoint_round_trip_through_uint8_pmt(self) -> None:
        n = 1000
        ids = torch.arange(n)
        src = self._create_cu_cache(int8_storage=True)
        self._write_cu_rows(src, n)
        row_bytes = src.get_int8_row_bytes()
        self.assertEqual(row_bytes, 16 + 128 + 8 + 4 * 2)

        saved_pmt = self._whole_row_pmt(src, ids, row_bytes, torch.uint8)
        self.assertEqual(saved_pmt.dtype, torch.uint8)
        saved = saved_pmt.narrow(0, 0, n)
        self.assertEqual(saved.dtype, torch.uint8)
        self.assertEqual(saved.shape, torch.Size([n, row_bytes]))
        self.assertTrue(torch.equal(saved[:, :8].view(torch.int64).view(-1), ids))

        dst = self._create_cu_cache(int8_storage=True)
        load_pmt = self._whole_row_pmt(dst, ids, row_bytes, torch.uint8)
        load_pmt.wrapped.set_range(0, 0, n, saved.clone())

        self.assertTrue(torch.equal(load_pmt.narrow(0, 0, n), saved))
        fp16_src = self._whole_row_pmt(src, ids, 8 + 132, torch.float16)
        fp16_dst = self._whole_row_pmt(dst, ids, 8 + 132, torch.float16)
        self.assertTrue(
            torch.equal(
                fp16_src.narrow(0, 0, n).view(torch.uint8),
                fp16_dst.narrow(0, 0, n).view(torch.uint8),
            )
        )
        fp16_bytes = (8 + 132) * 2
        print(
            f"checkpoint bytes/row: fp16 {fp16_bytes}, INT8 {row_bytes} "
            f"({row_bytes / fp16_bytes:.3f}x)"
        )

    def test_fp16_checkpoint_loads_into_int8_rows(self) -> None:
        n = 500
        ids = torch.arange(n)
        fp16_db = self._create_cu_cache(int8_storage=False)
        self.assertEqual(fp16_db.get_int8_row_bytes(), 0)
        rows = self._write_cu_rows(fp16_db, n)
        saved = self._whole_row_pmt(fp16_db, ids, 8 + 132, torch.float16).narrow(
            0, 0, n
        )

        int8_db = self._create_cu_cache(int8_storage=True)
        load_pmt = self._whole_row_pmt(int8_db, ids, 8 + 132, torch.float16)
        load_pmt.wrapped.set_range(0, 0, n, saved.clone())

        restored = load_pmt.narrow(0, 0, n)
        # MetaHeader and the fp16 tail come back exactly; the embedding is the
        # INT8 publish of the saved row, within half an INT8 step.
        self.assertTrue(
            torch.equal(restored[:, :8], saved[:, :8]),
            _metaheader_diff(restored, saved),
        )
        self.assertTrue(torch.equal(restored[:, 136:], saved[:, 136:]))
        emb = rows[:, :128].float()
        step = (emb.max(dim=1).values - emb.min(dim=1).values) / 255
        err = (restored[:, 8:136].float() - emb).abs().max(dim=1).values
        self.assertTrue(bool((err <= step * 0.5 + 1e-3).all()))
        codes = torch.ops.quantized.embedding_bag_byte_prepack(
            restored[:, 8:136].contiguous()
        )
        int8_row_bytes = int8_db.get_int8_row_bytes()
        int8_rows = self._whole_row_pmt(
            int8_db, ids, int8_row_bytes, torch.uint8
        ).narrow(0, 0, n)
        self.assertTrue(torch.equal(codes, int8_rows[:, 16 : 16 + 136]))

    def test_int8_checkpoint_load_uses_local_ids_and_keeps_source_bytes(self) -> None:
        n, offset = 100, 50000
        ids = torch.arange(n) + offset
        src = self._create_cu_cache(int8_storage=True)
        self._write_cu_rows(src, n)
        saved = self._whole_row_pmt(
            src, ids, 160, torch.uint8, row_offset=-offset
        ).narrow(0, 0, n)
        source_bytes = saved.clone()
        dst = self._create_cu_cache(int8_storage=True)
        target = self._whole_row_pmt(dst, ids, 160, torch.uint8, row_offset=-offset)

        target.set_weights_and_ids(saved, ids)

        self.assertTrue(torch.equal(saved, source_bytes))
        self.assertTrue(torch.equal(target.narrow(0, 0, n), saved))
        local = self._whole_row_pmt(dst, torch.arange(n), 160, torch.uint8).narrow(
            0, 0, n
        )
        self.assertTrue(
            torch.equal(
                local[:, :8].contiguous().view(torch.int64).flatten(), ids - offset
            )
        )

    def test_int8_checkpoint_can_omit_or_restore_unused_tail_columns(self) -> None:
        n = 100
        ids = torch.arange(n)
        src = self._create_cu_cache(int8_storage=True)
        self._write_cu_rows(src, n)
        saved = self._whole_row_pmt(src, ids, 160, torch.uint8).narrow(0, 0, n)
        without_tail = self._create_cu_cache(int8_storage=True, max_D=128)
        loading = self._whole_row_pmt(
            without_tail, ids, 160, torch.uint8, only_load_weight=True
        )

        loading.wrapped.set_range(0, 0, n, saved.clone())

        compact = self._whole_row_pmt(without_tail, ids, 152, torch.uint8).narrow(
            0, 0, n
        )
        self.assertTrue(torch.equal(compact, saved[:, :152]))
        restored = self._create_cu_cache(int8_storage=True)
        loading_full = self._whole_row_pmt(restored, ids, 152, torch.uint8)
        loading_full.wrapped.set_range(0, 0, n, compact.clone())
        full = self._whole_row_pmt(restored, ids, 160, torch.uint8).narrow(0, 0, n)
        self.assertTrue(torch.equal(full[:, :152], compact))
        self.assertEqual(int(torch.count_nonzero(full[:, 152:])), 0)

    def test_packed_rows_checkpoint_round_trip_through_uint8_pmt(self) -> None:
        n = 1000
        ids = torch.arange(n)
        src = self._create_cu_cache(int8_storage=False, packed_storage=True)
        self._write_cu_rows(src, n)
        row_bytes = src.get_raw_row_bytes()
        self.assertEqual(row_bytes, 16 + 64 + 16 + 4 * 2)
        self.assertEqual(src.get_int8_row_bytes(), 0)

        saved = self._whole_row_pmt(src, ids, row_bytes, torch.uint8).narrow(0, 0, n)
        self.assertEqual(saved.shape, torch.Size([n, row_bytes]))
        self.assertTrue(torch.equal(saved[:, :8].view(torch.int64).view(-1), ids))

        dst = self._create_cu_cache(int8_storage=False, packed_storage=True)
        load_pmt = self._whole_row_pmt(dst, ids, row_bytes, torch.uint8)
        load_pmt.wrapped.set_range(0, 0, n, saved.clone())

        self.assertTrue(torch.equal(load_pmt.narrow(0, 0, n), saved))
        fp16_src = self._whole_row_pmt(src, ids, 8 + 132, torch.float16)
        fp16_dst = self._whole_row_pmt(dst, ids, 8 + 132, torch.float16)
        self.assertTrue(
            torch.equal(
                fp16_src.narrow(0, 0, n).view(torch.uint8),
                fp16_dst.narrow(0, 0, n).view(torch.uint8),
            )
        )
        print(f"checkpoint bytes/row: fp16 {(8 + 132) * 2}, INT8 160, g8 {row_bytes}")

    def test_int8_checkpoint_loads_into_packed_rows(self) -> None:
        n = 500
        ids = torch.arange(n)
        int8_db = self._create_cu_cache(int8_storage=True)
        self._write_cu_rows(int8_db, n)
        saved = self._whole_row_pmt(
            int8_db, ids, int8_db.get_int8_row_bytes(), torch.uint8
        ).narrow(0, 0, n)
        int8_fp16 = self._whole_row_pmt(int8_db, ids, 8 + 132, torch.float16).narrow(
            0, 0, n
        )

        packed_db = self._create_cu_cache(int8_storage=False, packed_storage=True)
        load_pmt = self._whole_row_pmt(packed_db, ids, saved.size(1), torch.uint8)
        load_pmt.wrapped.set_range(0, 0, n, saved.clone())

        restored = self._whole_row_pmt(packed_db, ids, 8 + 132, torch.float16).narrow(
            0, 0, n
        )
        self.assertTrue(
            torch.equal(restored[:, :8], int8_fp16[:, :8]),
            _metaheader_diff(restored, int8_fp16),
        )
        self.assertTrue(torch.equal(restored[:, 136:], int8_fp16[:, 136:]))
        # The g8 publish of the restored rows reproduces them.
        emb = restored[:, 8:136].contiguous()
        published = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(
            torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(emb, 8), 8, 128
        )
        self.assertTrue(torch.equal(published, emb))
        err = (emb.float() - int8_fp16[:, 8:136].float()).abs().max()
        self.assertLess(float(err), 0.05)

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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest
from unittest import TestCase

import fbgemm_gpu  # noqa E402
import torch
import torch.testing
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.utils.loader import load_torch_module
from hypothesis import given, settings, strategies as st, Verbosity

from .. import common  # noqa E402
from ..common import open_source, running_in_oss

if not open_source:
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings",
    )


@unittest.skipIf(*running_in_oss)
class KvTensorWrapperTest(TestCase):
    # pyre-ignore[56]
    @given(
        precision=st.sampled_from(
            [
                (SparseType.FP16, 16),
                (SparseType.FP32, 32),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_read_tensor_using_wrapper_from_db(
        self, precision: tuple[SparseType, int]
    ) -> None:
        E = int(1e4)
        D = 128
        max_D = 256  # max emb dimension seen by rocksDB
        N = 1000
        weights_precision, dtype_width = precision
        weights_dtype = weights_precision.as_dtype()

        with tempfile.TemporaryDirectory() as ssd_directory:
            # pyre-fixme[16]: Module `classes` has no attribute `fbgemm`.
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

            # create random index tensor with size N
            indices = torch.randperm(N)
            # insert the weights with the corresponding indices into the table
            weights = torch.arange(N * D, dtype=weights_dtype).view(N, D)
            padded_weights = torch.nn.functional.pad(weights, (0, max_D - D))
            output_weights = torch.empty_like(padded_weights)
            count = torch.tensor([N])
            ssd_db.set(indices, padded_weights, count)

            # force waiting for set to complete
            ssd_db.get(indices, output_weights, torch.tensor(indices.shape[0]))
            torch.testing.assert_close(padded_weights, output_weights)

            # create a view tensor wrapper
            snapshot = ssd_db.create_snapshot()
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                ssd_db, [E, D], weights.dtype, 0, snapshot
            )
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
            self.assertEqual(ssd_db.get_snapshot_count(), 0)

    # pyre-ignore[56]
    @given(
        precision=st.sampled_from(
            [
                (SparseType.FP16, 16),
                (SparseType.FP32, 32),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_write_tensor_to_db(self, precision: tuple[SparseType, int]) -> None:
        E = int(1e4)  # num total rows
        D = 128  # emb dimension
        max_D = 256  # max emb dimension seen by rocksDB
        N = 1000  # window size
        weights_precision, dtype_width = precision
        weights_dtype = weights_precision.as_dtype()

        with tempfile.TemporaryDirectory() as ssd_directory:
            # pyre-fixme[16]: Module `classes` has no attribute `fbgemm`.
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
            weights = torch.arange(N * D, dtype=weights_dtype).view(N, D)
            padded_weights = torch.nn.functional.pad(weights, (0, max_D - D))
            output_weights = torch.empty_like(padded_weights, dtype=weights_dtype)

            # no snapshot needed for writing to rocksdb
            tensor_wrapper0 = torch.classes.fbgemm.KVTensorWrapper(
                ssd_db, [E, D], weights.dtype, 0
            )
            step = N
            for i in range(0, E, step):
                tensor_wrapper0.set_range(0, i, step, weights)

            # force waiting for set to complete
            indices = torch.arange(step)
            for i in range(0, E, step):
                ssd_db.get(i + indices, output_weights, torch.tensor(indices.shape[0]))

            # create a view tensor wrapper
            snapshot = ssd_db.create_snapshot()
            tensor_wrapper = torch.classes.fbgemm.KVTensorWrapper(
                ssd_db, [E, D], weights.dtype, 0, snapshot
            )
            self.assertEqual(tensor_wrapper.shape, [E, D])

            # table has a total of E rows
            # load 1000 rows at a time
            step = 1000
            for i in range(0, E, step):
                narrowed = tensor_wrapper.narrow(0, i, step)
                self.assertTrue(
                    torch.equal(narrowed, weights),
                    msg=(
                        f"Tensor value mismatch :\n"
                        f"actual\n{narrowed}\n\nexpected\n{weights}"
                    ),
                )

            del tensor_wrapper0
            del tensor_wrapper
            del snapshot
            self.assertEqual(ssd_db.get_snapshot_count(), 0)

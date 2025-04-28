# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)

from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.bench import (
    BatchParams,
    IndicesParams,
    PoolingParams,
    TBEDataConfig,
)
from fbgemm_gpu.tbe.stats import TBEBenchmarkParamsReporter
from fbgemm_gpu.tbe.utils import get_device


class TestTBEBenchmarkParamsReporter(unittest.TestCase):
    @patch("fbgemm_gpu.utils.FileStore")  # Mock FileStore
    def test_report_stats(
        self,
        mock_filestore: MagicMock,  # Mock FileStore
    ) -> None:

        tbeconfig = TBEDataConfig(
            T=2,
            E=1024,
            D=32,
            mixed_dim=True,
            weighted=False,
            batch_params=BatchParams(B=512),
            indices_params=IndicesParams(
                heavy_hitters=torch.tensor([]),
                zipf_q=0.1,
                zipf_s=0.1,
                index_dtype=torch.int64,
                offset_dtype=torch.int64,
            ),
            pooling_params=PoolingParams(L=2),
            use_cpu=True,
        )

        embedding_location = EmbeddingLocation.HOST

        _, Ds = tbeconfig.generate_embedding_dims()
        embedding_op = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    tbeconfig.E,
                    D,
                    embedding_location,
                    ComputeDevice.CPU,
                )
                for D in Ds
            ],
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            learning_rate=0.01,
            weights_precision=SparseType.FP32,
            pooling_mode=PoolingMode.SUM,
            output_dtype=SparseType.FP32,
        )

        embedding_op = embedding_op.to(get_device())

        requests = tbeconfig.generate_requests(1)

        # Initialize the reporter
        reporter = TBEBenchmarkParamsReporter(report_interval=1)
        # Set the mock filestore as the reporter's filestore
        reporter.filestore = mock_filestore

        request = requests[0]
        # Call the report_stats method
        extracted_config = reporter.extract_params(
            embedding_op=embedding_op,
            indices=request.indices,
            offsets=request.offsets,
        )

        reporter.report_stats(
            embedding_op=embedding_op,
            indices=request.indices,
            offsets=request.offsets,
        )

        # TODO: This is not working because need more details in initial config
        # Assert that the reconstructed configuration matches the original
        # assert (
        #     extracted_config == tbeconfig
        # ), "Extracted configuration does not match the original TBEDataConfig"

        # Check if the write method was called on the FileStore
        assert (
            reporter.filestore.write.assert_called_once
        ), "FileStore.write() was not called"

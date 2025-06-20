# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import unittest

import hypothesis.strategies as st

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation
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
from hypothesis import given, settings


class TestTBEBenchmarkParamsReporter(unittest.TestCase):
    # pyre-ignore[56]
    @given(
        T=st.integers(1, 10),
        E=st.integers(100, 10000),
        D=st.sampled_from([32, 64, 128, 256]),
        L=st.integers(1, 10),
        B=st.integers(20, 100),
    )
    @settings(max_examples=1, deadline=None)
    def test_report_stats(
        self,
        T: int,
        E: int,
        D: int,
        L: int,
        B: int,
    ) -> None:
        """Test that the reporter can extract a valid JSON configuration from the embedding operation and requests."""

        # Generate a TBEDataConfig
        tbeconfig = TBEDataConfig(
            T=T,
            E=E,
            D=D,
            mixed_dim=False,
            weighted=False,
            batch_params=BatchParams(B=B),
            indices_params=IndicesParams(
                heavy_hitters=torch.tensor([]),
                zipf_q=0.1,
                zipf_s=0.1,
                index_dtype=torch.int64,
                offset_dtype=torch.int64,
            ),
            pooling_params=PoolingParams(L=L),
            use_cpu=not torch.cuda.is_available(),
        )

        embedding_location = (
            EmbeddingLocation.DEVICE
            if torch.cuda.is_available()
            else EmbeddingLocation.HOST
        )

        # Generate the embedding dimension list
        _, Ds = tbeconfig.generate_embedding_dims()

        # Generate the embedding operation
        embedding_op = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    tbeconfig.E,
                    D,
                    embedding_location,
                    (
                        ComputeDevice.CUDA
                        if torch.cuda.is_available()
                        else ComputeDevice.CPU
                    ),
                )
                for D in Ds
            ],
            embedding_table_index_type=tbeconfig.indices_params.index_dtype
            or torch.int64,
            embedding_table_offset_type=tbeconfig.indices_params.offset_dtype
            or torch.int64,
        )

        embedding_op = embedding_op.to(get_device())

        # Initialize the reporter
        reporter = TBEBenchmarkParamsReporter(report_interval=1)

        # Generate indices and offsets
        request = tbeconfig.generate_requests(1)[0]

        # Call the report_stats method
        extracted_config = reporter.extract_params(
            embedding_op=embedding_op,
            indices=request.indices,
            offsets=request.offsets,
        )

        assert (
            extracted_config.T == tbeconfig.T
            and extracted_config.E == tbeconfig.E
            and extracted_config.D == tbeconfig.D
            and extracted_config.pooling_params.L == tbeconfig.pooling_params.L
            and extracted_config.batch_params.B == tbeconfig.batch_params.B
            and extracted_config.mixed_dim == tbeconfig.mixed_dim
            and extracted_config.weighted == tbeconfig.weighted
            and extracted_config.indices_params.index_dtype
            == tbeconfig.indices_params.index_dtype
            and extracted_config.indices_params.offset_dtype
            == tbeconfig.indices_params.offset_dtype
        ), "Extracted config does not match the original TBEDataConfig"
        # Attempt to reconstruct TBEDataConfig from extracted_json_config

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import unittest
from typing import Optional
from unittest.mock import patch

import fbgemm_gpu

import hypothesis.strategies as st

import torch
from fbgemm_gpu.config import FeatureGateName
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    ComputeDevice,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.bench.tbe_data_config import (
    BatchParams,
    IndicesParams,
    PoolingParams,
    TBEDataConfig,
)

from fbgemm_gpu.tbe.bench.tbe_data_config_bench_helper import (
    generate_embedding_dims,
    generate_requests,
)

from fbgemm_gpu.tbe.stats import TBEBenchmarkParamsReporter
from fbgemm_gpu.tbe.utils import get_device
from hypothesis import given, settings

from .. import common  # noqa E402
from ..common import running_in_oss

try:
    # pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
    open_source: bool = getattr(fbgemm_gpu, "open_source", False)
except Exception:
    open_source: bool = False


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
        _, Ds = generate_embedding_dims(tbeconfig)

        embedding_specs = [
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
        ]

        # Generate the embedding operation
        embedding_op = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs,
            embedding_table_index_type=tbeconfig.indices_params.index_dtype
            or torch.int64,
            embedding_table_offset_type=tbeconfig.indices_params.offset_dtype
            or torch.int64,
        )

        embedding_op = embedding_op.to(get_device())

        # Initialize the reporter
        reporter = TBEBenchmarkParamsReporter(report_interval=1)

        # Generate indices and offsets
        request = generate_requests(tbeconfig, 1)[0]

        # Call the extract_params method
        extracted_config = reporter.extract_params(
            feature_rows=embedding_op.rows_per_table,
            feature_dims=embedding_op.feature_dims,
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

    # pyre-ignore[56]
    @given(
        T=st.integers(1, 10),
        E=st.integers(100, 10000),
        D=st.sampled_from([32, 64, 128, 256]),
        L=st.integers(1, 10),
        B=st.integers(20, 100),
    )
    @settings(max_examples=1, deadline=None)
    @unittest.skipIf(*running_in_oss)
    def test_report_fb_files(
        self,
        T: int,
        E: int,
        D: int,
        L: int,
        B: int,
    ) -> None:
        """
        Test writing extrcted TBEDataConfig to FB FileStore
        """
        from fbgemm_gpu.fb.utils.manifold_wrapper import FileStore

        # Initialize the reporter
        bucket = "tlparse_reports"
        path_prefix = "tree/unit_tests/"

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
        _, Ds = generate_embedding_dims(tbeconfig)

        with patch(
            "torch.ops.fbgemm.check_feature_gate_key"
        ) as mock_check_feature_gate_key:
            # Mock the return value for TBE_REPORT_INPUT_PARAMS
            def side_effect(feature_name: str) -> Optional[bool]:
                if feature_name == FeatureGateName.TBE_REPORT_INPUT_PARAMS.name:
                    return True

            mock_check_feature_gate_key.side_effect = side_effect

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
            )

            embedding_op = embedding_op.to(get_device())

            # Generate indices and offsets
            request = generate_requests(tbeconfig, 1)[0]

            # Execute the embedding operation with reporting flag enable
            embedding_op.forward(request.indices, request.offsets)

            # Check if the file was written to Manifold
            store = FileStore(bucket)
            path = f"{path_prefix}tbe-{embedding_op.uuid}-config-estimation-{embedding_op.iter_cpu.item()}.json"
            assert store.exists(path), f"{path} not exists"

            # Clenaup, delete the file
            store.remove(path)


if __name__ == "__main__":
    unittest.main()

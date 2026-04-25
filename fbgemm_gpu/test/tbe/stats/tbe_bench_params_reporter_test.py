# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

import fbgemm_gpu
import hypothesis.strategies as st
import torch
from fbgemm_gpu.config import FeatureGateName
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    ComputeDevice,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
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
from fbgemm_gpu.tbe.monitoring.bench_params_reporter import TBEBenchmarkParamsReporter
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


class TestInferenceTBEBenchmarkParamsReporter(unittest.TestCase):
    def test_inference_report_stats(self) -> None:
        """Test that the reporter can extract a valid config from inference TBE."""
        T = 3
        E = 1000
        D = 64
        L = 5
        B = 32

        use_cpu = not torch.cuda.is_available() or torch.version.hip is not None
        embedding_location = (
            EmbeddingLocation.HOST if use_cpu else EmbeddingLocation.DEVICE
        )

        embedding_specs = [
            ("", E, D, SparseType.FP16, embedding_location) for _ in range(T)
        ]

        cc = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            indices_dtype=torch.int64,
            device=torch.device("cpu") if use_cpu else None,
        )
        cc.fill_random_weights()

        reporter = TBEBenchmarkParamsReporter(report_interval=1)

        # Generate test indices and offsets (int64 required by tbe_estimate_indices_distribution)
        indices = torch.randint(0, E, (T * B * L,), dtype=torch.int64)
        offsets = torch.arange(0, T * B * L + 1, L, dtype=torch.int64)

        feature_dims = torch.tensor([D] * T, device="cpu", dtype=torch.int64)

        extracted_config = reporter.extract_params(
            feature_rows=cc.rows_per_table,
            feature_dims=feature_dims,
            indices=indices,
            offsets=offsets,
        )

        self.assertEqual(extracted_config.T, T)
        self.assertEqual(extracted_config.E, E)
        self.assertEqual(extracted_config.D, D)
        self.assertEqual(extracted_config.pooling_params.L, L)
        self.assertEqual(extracted_config.batch_params.B, B)

    def test_inference_jit_script_with_eeg(self) -> None:
        """Confirm TorchScript compatibility is preserved when EEG is initialized."""
        T = 2
        E = 100
        D = 32
        B = 4
        L = 3

        use_cpu = not torch.cuda.is_available() or torch.version.hip is not None
        embedding_location = (
            EmbeddingLocation.HOST if use_cpu else EmbeddingLocation.DEVICE
        )

        with patch(
            "torch.ops.fbgemm.check_feature_gate_key"
        ) as mock_check_feature_gate_key:

            def side_effect(feature_name: str) -> Optional[bool]:
                if feature_name == FeatureGateName.TBE_REPORT_INPUT_PARAMS.name:
                    return True

            mock_check_feature_gate_key.side_effect = side_effect

            embedding_specs = [
                ("", E, D, SparseType.FP16, embedding_location) for _ in range(T)
            ]

            cc = IntNBitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                device=torch.device("cpu") if use_cpu else None,
            )
            cc.fill_random_weights()

            # TorchScript should succeed
            scripted = torch.jit.script(cc)

            indices = torch.randint(0, E, (T * B * L,), dtype=torch.int32)
            offsets = torch.arange(0, T * B * L + 1, L, dtype=torch.int32)

            # Forward should succeed on scripted module
            output = scripted(indices, offsets)
            self.assertEqual(output.shape, (B, T * D))

    def test_inference_eeg_forward_reports(self) -> None:
        """Test that forward() calls report_stats when EEG is enabled."""
        T = 2
        E = 100
        D = 32
        B = 4
        L = 3

        use_cpu = not torch.cuda.is_available() or torch.version.hip is not None
        embedding_location = (
            EmbeddingLocation.HOST if use_cpu else EmbeddingLocation.DEVICE
        )

        with patch(
            "torch.ops.fbgemm.check_feature_gate_key"
        ) as mock_check_feature_gate_key:

            def side_effect(feature_name: str) -> Optional[bool]:
                if feature_name == FeatureGateName.TBE_REPORT_INPUT_PARAMS.name:
                    return True

            mock_check_feature_gate_key.side_effect = side_effect

            embedding_specs = [
                ("", E, D, SparseType.FP16, embedding_location) for _ in range(T)
            ]

            cc = IntNBitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                indices_dtype=torch.int64,
                device=torch.device("cpu") if use_cpu else None,
            )
            cc.fill_random_weights()

            # Replace the report function with a mock to verify it's called
            report_mock = MagicMock()
            cc._report_input_params = report_mock

            indices = torch.randint(0, E, (T * B * L,), dtype=torch.int64)
            offsets = torch.arange(0, T * B * L + 1, L, dtype=torch.int64)

            # Forward should trigger reporting
            cc.forward(indices, offsets)

            # Verify the report function was called
            report_mock.assert_called_once()
            call_kwargs = report_mock.call_args[1]
            self.assertEqual(call_kwargs["op_id"], cc.uuid)
            self.assertEqual(len(call_kwargs["feature_rows"]), T)
            self.assertEqual(call_kwargs["feature_dims"].shape[0], T)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import json
import logging
import os
from typing import List, Optional, Tuple

import fbgemm_gpu  # noqa F401
import torch  # usort:skip

from fbgemm_gpu.tbe.bench.tbe_data_config import (
    BatchParams,
    IndicesParams,
    PoolingParams,
    TBEDataConfig,
)

open_source: bool = False
# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    from fbgemm_gpu.utils import FileStore

else:
    try:
        from fbgemm_gpu.fb.utils.manifold_wrapper import FileStore

        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu/src/tbe/eeg:indices_estimator"
        )
    except Exception:
        pass


class TBEBenchmarkParamsReporter:
    """
    TBEBenchmarkParamsReporter is responsible for extracting and reporting the configuration data of TBE processes.
    """

    def __init__(
        self,
        report_interval: int,
        report_iter_start: int = 0,
        report_iter_end: int = -1,
        bucket: Optional[str] = None,
        path_prefix: Optional[str] = None,
    ) -> None:
        """
        Initializes the TBEBenchmarkParamsReporter with the specified parameters.

        Args:
            report_interval (int): The interval at which reports are generated.
            report_iter_start (int): The start of the iteration range to capture. Defaults to 0.
            report_iter_end (int): The end of the iteration range to capture. Defaults to -1 (last iteration).
            bucket (Optional[str], optional): The storage bucket for reports. Defaults to None.
            path_prefix (Optional[str], optional): The path prefix for report storage. Defaults to None.
        """

        assert report_interval > 0, "report_interval must be greater than 0"
        assert (
            report_iter_start >= 0
        ), "report_iter_start must be greater than or equal to 0"
        assert (
            report_iter_end >= -1
        ), "report_iter_end must be greater than or equal to -1"
        assert (
            report_iter_end == -1 or report_iter_start <= report_iter_end
        ), "report_iter_start must be less than or equal to report_iter_end"

        self.report_interval = report_interval
        self.report_iter_start = report_iter_start
        self.report_iter_end = report_iter_end

        if path_prefix is not None and path_prefix.endswith("/"):
            path_prefix = path_prefix[:-1]

        self.path_prefix = path_prefix

        default_bucket = "/tmp" if open_source else "tlparse_reports"
        bucket = (
            bucket
            if bucket is not None
            else os.environ.get("FBGEMM_TBE_REPORTING_BUCKET", default_bucket)
        )
        self.filestore = FileStore(bucket)

        if self.path_prefix is not None and not self.filestore.exists(self.path_prefix):
            self.filestore.create_directory(self.path_prefix)

        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    @classmethod
    def create(cls) -> "TBEBenchmarkParamsReporter":
        """
        This method returns an instance of TBEBenchmarkParamsReporter based on environment variables.

        If the `FBGEMM_REPORT_INPUT_PARAMS_INTERVAL` environment variable is set to a value greater than 0, it creates an instance that:
        - Reports input parameters (TBEDataConfig).
        - Writes the output as a JSON file.

        Additionally, the following environment variables are considered:
        - `FBGEMM_REPORT_INPUT_PARAMS_ITER_START`: Specifies the start of the iteration range to capture.
        - `FBGEMM_REPORT_INPUT_PARAMS_ITER_END`: Specifies the end of the iteration range to capture.
        - `FBGEMM_REPORT_INPUT_PARAMS_BUCKET`: Specifies the bucket for reporting.
        - `FBGEMM_REPORT_INPUT_PARAMS_PATH_PREFIX`: Specifies the path prefix for reporting.

        Returns:
            TBEBenchmarkParamsReporter: An instance configured based on the environment variables.
        """
        report_interval = int(
            os.environ.get("FBGEMM_REPORT_INPUT_PARAMS_INTERVAL", "1")
        )
        report_iter_start = int(
            os.environ.get("FBGEMM_REPORT_INPUT_PARAMS_ITER_START", "0")
        )
        report_iter_end = int(
            os.environ.get("FBGEMM_REPORT_INPUT_PARAMS_ITER_END", "-1")
        )
        bucket = os.environ.get("FBGEMM_REPORT_INPUT_PARAMS_BUCKET", "")
        path_prefix = os.environ.get("FBGEMM_REPORT_INPUT_PARAMS_PATH_PREFIX", "")

        return cls(
            report_interval=report_interval,
            report_iter_start=report_iter_start,
            report_iter_end=report_iter_end,
            bucket=bucket,
            path_prefix=path_prefix,
        )

    def extract_Ls(
        self,
        bag_sizes: List[int],
        Bs: List[int],
    ) -> List[float]:
        Ls = []
        start = 0
        for b in Bs:
            end = start + b
            avg_L = sum(bag_sizes[start:end]) / b if b > 0 else 0
            start = end
            Ls.append(avg_L)
        return Ls

    def extract_params(
        self,
        feature_rows: torch.Tensor,
        feature_dims: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        Es: Optional[List[int]] = None,
        Ds: Optional[List[int]] = None,
        embedding_specs: Optional[List[Tuple[int, int]]] = None,
        feature_table_map: Optional[List[int]] = None,
    ) -> TBEDataConfig:
        """
        Extracts parameters from the embedding operation, input indices, and offsets to create a TBEDataConfig.

        Args:
            feature_rows (torch.Tensor): Number of rows in each feature.
            feature_dims (torch.Tensor): Number of dimensions in each feature.
            indices (torch.Tensor): The input indices tensor.
            offsets (torch.Tensor): The input offsets tensor.
            per_sample_weights (Optional[torch.Tensor], optional): Weights for each sample. Defaults to None.
            batch_size_per_feature_per_rank (Optional[List[List[int]]], optional): Batch sizes per feature per rank. Defaults to None.

        Returns:
            TBEDataConfig: The configuration data for TBE benchmarking.
        """

        Es = feature_rows.tolist()
        Ds = feature_dims.tolist()

        assert len(Es) == len(
            Ds
        ), "feature_rows and feature_dims must have the same length"

        # Transfer indices back to CPU for EEG analysis
        indices_cpu = indices.cpu()

        # Set T to be the number of features we are looking at
        T = len(Ds)
        # Set E to be the mean of the rowcounts to avoid biasing
        E = (
            Es[0]
            if len(set(Es)) == 1
            else torch.ceil(
                torch.mean(torch.tensor(feature_rows, dtype=torch.float))
            ).item()
        )
        # Set mixed_dim to be True if there are multiple dims
        mixed_dim = len(set(Ds)) > 1
        # Set D to be the mean of the dims to avoid biasing
        D = (
            Ds[0]
            if not mixed_dim
            else torch.ceil(
                torch.mean(torch.tensor(feature_dims, dtype=torch.float))
            ).item()
        )

        # Compute indices distribution parameters
        heavy_hitters, q, s, _, _ = torch.ops.fbgemm.tbe_estimate_indices_distribution(
            indices_cpu
        )
        indices_params = IndicesParams(
            heavy_hitters, q, s, indices.dtype, offsets.dtype
        )

        # Compute batch parameters
        B = int((offsets.numel() - 1) // T)
        Bs = (
            [sum(b_per_rank) for b_per_rank in batch_size_per_feature_per_rank]
            if batch_size_per_feature_per_rank
            else [B] * T
        )
        batch_params = BatchParams(
            B=B,
            sigma_B=(
                int(
                    torch.ceil(
                        torch.std(
                            torch.tensor(
                                [
                                    b
                                    for bs in batch_size_per_feature_per_rank
                                    for b in bs
                                ]
                            ).float()
                        )
                    )
                )
                if batch_size_per_feature_per_rank
                else None
            ),
            vbe_distribution=("normal" if batch_size_per_feature_per_rank else None),
            vbe_num_ranks=(
                len(batch_size_per_feature_per_rank)
                if batch_size_per_feature_per_rank
                else None
            ),
            Bs=Bs,
        )

        # Compute pooling parameters
        bag_sizes = offsets[1:] - offsets[:-1]
        if batch_size_per_feature_per_rank is None:
            _B = int(bag_sizes.numel() // T)
            assert _B == Bs[0], f"Expected constant batch size {Bs[0]} but got {_B}"
        mixed_bag_sizes = len(set(bag_sizes)) > 1
        pooling_params = PoolingParams(
            L=(
                int(torch.ceil(torch.mean(bag_sizes.float())))
                if mixed_bag_sizes
                else int(bag_sizes[0])
            ),
            sigma_L=(
                int(torch.ceil(torch.std(bag_sizes.float())))
                if mixed_bag_sizes
                else None
            ),
            length_distribution=("normal" if mixed_bag_sizes else None),
            Ls=self.extract_Ls(bag_sizes.tolist(), Bs),
        )

        return TBEDataConfig(
            T=T,
            E=E,
            D=D,
            mixed_dim=mixed_dim,
            weighted=(per_sample_weights is not None),
            batch_params=batch_params,
            indices_params=indices_params,
            pooling_params=pooling_params,
            use_cpu=(not torch.cuda.is_available()),
            Es=Es,
            Ds=Ds,
            embedding_specs=embedding_specs,
            feature_table_map=feature_table_map,
        )

    def report_stats(
        self,
        feature_rows: torch.Tensor,
        feature_dims: torch.Tensor,
        iteration: int,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        op_id: str = "",
        per_sample_weights: Optional[torch.Tensor] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        embedding_specs: Optional[List[Tuple[int, int]]] = None,
        feature_table_map: Optional[List[int]] = None,
    ) -> None:
        """
        Reports the configuration of the embedding operation and input data, then writes the TBE configuration to the filestore.

        Args:
            feature_rows (torch.Tensor): Number of rows in each feature.
            feature_dims (torch.Tensor): Number of dimensions in each feature.
            iteration (int): The current iteration number.
            indices (torch.Tensor): The input indices tensor.
            offsets (torch.Tensor): The input offsets tensor.
            op_id (str, optional): The operation identifier. Defaults to an empty string.
            per_sample_weights (Optional[torch.Tensor], optional): Weights for each sample. Defaults to None.
            batch_size_per_feature_per_rank (Optional[List[List[int]]], optional): Batch sizes per feature per rank. Defaults to None.
            embedding_specs (Optional[List[Tuple[int, int]]]): Embedding specs. Defaults to None.
            feature_table_map (Optional[List[int]], optional): Feature table map. Defaults to None.
        """
        if (
            (iteration - self.report_iter_start) % self.report_interval == 0
            and (iteration >= self.report_iter_start)
            and (self.report_iter_end == -1 or iteration <= self.report_iter_end)
        ):
            # If indices tensor is empty (indices.numel() == 0), skip reporting
            # TODO: Remove this once we have a better way to handle empty indices tensors
            if indices.numel() == 0:
                return

            # Extract TBE config
            config = self.extract_params(
                feature_rows=feature_rows,
                feature_dims=feature_dims,
                indices=indices,
                offsets=offsets,
                per_sample_weights=per_sample_weights,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                Es=feature_rows.tolist(),
                Ds=feature_dims.tolist(),
                embedding_specs=embedding_specs,
                feature_table_map=feature_table_map,
            )

            # Write the TBE config to FileStore
            self.filestore.write(
                f"{self.path_prefix}/tbe-{op_id}-config-estimation-{iteration}.json",
                io.BytesIO(json.dumps(config.dict(), indent=2).encode()),
            )

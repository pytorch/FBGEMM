#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import logging
import os
from typing import List, Optional

import fbgemm_gpu  # noqa F401
import numpy as np
import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    DenseTableBatchedEmbeddingBagsCodegen,  # noqa
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.bench import (
    BatchParams,
    IndicesParams,
    PoolingParams,
    TBEDataConfig,
)

# pyre-ignore[16]
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    from fbgemm_gpu.utils import FileStore
else:
    from fbgemm_gpu.fb.utils import FileStore


class TBEBenchmarkParamsReporter:
    def __init__(
        self,
        report_interval: int,
        report_once: bool = False,
        bucket: Optional[str] = None,
        path_prefix: Optional[str] = None,
    ) -> None:
        self.report_interval = report_interval
        self.report_once = report_once

        default_bucket = "/tmp" if open_source else "tlparse_reports"
        bucket = (
            bucket
            if bucket
            else os.environ.get("FBGEMM_TBE_REPORTING_BUCKET", default_bucket)
        )
        self.filestore = FileStore(bucket)

        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def report_stats(
        self,
        embedding_op: SplitTableBatchedEmbeddingBagsCodegen,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> None:
        """
        Print input stats (for debugging purpose only)

        Args:
            indices (Tensor): Input indices
            offsets (Tensor): Input offsets
            per_sample_weights (Optional[Tensor]): Input per
                sample weights
        """
        if embedding_op.iter.item() % self.report_interval == 0:
            pass

        # Transfer indices back to CPU for EEG analysis
        indices_cpu = indices.cpu()

        # Extract embedding table specs
        embedding_specs = [
            embedding_op.embedding_specs[t] for t in embedding_op.feature_table_map
        ]
        rowcounts = [embedding_spec[0] for embedding_spec in embedding_specs]
        dims = [embedding_spec[1] for embedding_spec in embedding_specs]

        # Set T to be the number of features we are looking at
        T = len(embedding_op.feature_table_map)
        # Set E to be the median of the rowcounts to avoid biasing the
        E = rowcounts[0] if len(set(rowcounts)) == 1 else np.ceil((np.mean(rowcounts)))
        # Set mixed_dim to be True if there are multiple dims
        mixed_dim = len(set(dims)) > 1
        # Set D to be the median of the dims to avoid biasing
        D = dims[0] if mixed_dim else np.ceil((np.mean(dims)))

        # Compute indices distribution parameters
        heavy_hitters, q, s, _, _ = torch.ops.fbgemm.tbe_estimate_indices_distribution(
            indices_cpu
        )
        indices_params = IndicesParams(
            heavy_hitters, q, s, indices.dtype, offsets.dtype
        )

        # Compute batch parameters
        batch_params = BatchParams(
            B=((offsets.numel() - 1) // T),
            sigma_B=(
                np.ceil(
                    np.std([b for bs in batch_size_per_feature_per_rank for b in bs])
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
        )

        # Compute pooling parameters
        bag_sizes = offsets[1:] - offsets[:-1]
        mixed_bag_sizes = len(set(bag_sizes)) > 1
        pooling_params = PoolingParams(
            L=np.ceil(np.mean(bag_sizes)) if mixed_bag_sizes else bag_sizes[0],
            sigma_L=(np.ceil(np.std(bag_sizes)) if mixed_bag_sizes else None),
            length_distribution=("normal" if mixed_bag_sizes else None),
        )

        config = TBEDataConfig(
            T=T,
            E=E,
            D=D,
            mixed_dim=mixed_dim,
            weighted=(per_sample_weights is not None),
            batch_params=batch_params,
            indices_params=indices_params,
            pooling_params=pooling_params,
            use_cpu=(not torch.cuda.is_available()),
        )

        # Write the TBE config to FileStore
        self.filestore.write(
            f"tbe-{embedding_op.uuid}-config-estimation-{embedding_op.iter.item()}.json",
            io.BytesIO(config.json(format=True).encode()),
        )

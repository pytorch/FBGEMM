#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
from typing import Any, Optional

import click
import torch

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    EmbeddingLocation,
    PoolingMode,
)


@dataclasses.dataclass(frozen=True)
class EmbeddingOpsCommonConfig:
    # Precision of the embedding weights
    weights_dtype: SparseType
    # Precision of the embedding cache
    cache_dtype: Optional[SparseType]
    # Precision of the embedding output
    output_dtype: SparseType
    # Enable stochastic rounding when performing quantization
    stochastic_rounding: bool
    # Pooling operation to perform
    pooling_mode: PoolingMode
    # Use host-mapped UVM buffers
    uvm_host_mapped: bool
    # Memory location of the embeddings
    embedding_location: EmbeddingLocation
    # Bounds check mode
    bounds_check_mode: BoundsCheckMode

    # pyre-ignore [3]
    def validate(self):
        return self

    def split_args(self) -> dict[str, Any]:
        return {
            "weights_precision": self.weights_dtype,
            "stochastic_rounding": self.stochastic_rounding,
            "output_dtype": self.output_dtype,
            "pooling_mode": self.pooling_mode,
            "bounds_check_mode": self.bounds_check_mode,
            "uvm_host_mapped": self.uvm_host_mapped,
        }


class EmbeddingOpsCommonConfigLoader:
    @classmethod
    # pyre-ignore [2]
    def options(cls, func) -> click.Command:
        options = [
            click.option(
                "--emb-weights-dtype",
                type=SparseType,
                default=SparseType.FP32,
                help="Precision of the embedding weights",
            ),
            click.option(
                "--emb-cache-dtype",
                type=SparseType,
                default=None,
                help="Precision of the embedding cache",
            ),
            click.option(
                "--emb-output-dtype",
                type=SparseType,
                default=SparseType.FP32,
                help="Precision of the embedding output",
            ),
            click.option(
                "--emb-stochastic-rounding",
                is_flag=True,
                default=False,
                help="Enable stochastic rounding when performing quantization",
            ),
            click.option(
                "--emb-pooling-mode",
                type=click.Choice(["sum", "mean", "none"], case_sensitive=False),
                default="sum",
                help="Pooling operation to perform",
            ),
            click.option(
                "--emb-uvm-host-mapped",
                is_flag=True,
                default=False,
                help="Use host-mapped UVM buffers",
            ),
            click.option(
                "--emb-location",
                default="device",
                type=click.Choice(EmbeddingLocation.str_values(), case_sensitive=False),
                help="Memory location of the embeddings",
            ),
            click.option(
                "--emb-bounds-check",
                type=int,
                default=BoundsCheckMode.WARNING.value,
                help="Bounds check mode"
                f"Available modes: FATAL={BoundsCheckMode.FATAL.value}, "
                f"WARNING={BoundsCheckMode.WARNING.value}, "
                f"IGNORE={BoundsCheckMode.IGNORE.value}, "
                f"NONE={BoundsCheckMode.NONE.value}",
            ),
        ]

        for option in reversed(options):
            func = option(func)
        return func

    @classmethod
    def load(cls, context: click.Context) -> EmbeddingOpsCommonConfig:
        params = context.params

        weights_dtype = params["emb_weights_dtype"]
        cache_dtype = params["emb_cache_dtype"]
        output_dtype = params["emb_output_dtype"]
        stochastic_rounding = params["emb_stochastic_rounding"]
        pooling_mode = PoolingMode.from_str(str(params["emb_pooling_mode"]))
        uvm_host_mapped = params["emb_uvm_host_mapped"]
        bounds_check_mode = BoundsCheckMode(params["emb_bounds_check"])

        embedding_location = EmbeddingLocation.from_str(str(params["emb_location"]))
        if (
            embedding_location is EmbeddingLocation.DEVICE
            and not torch.cuda.is_available()
        ):
            embedding_location = EmbeddingLocation.HOST

        return EmbeddingOpsCommonConfig(
            weights_dtype,
            cache_dtype,
            output_dtype,
            stochastic_rounding,
            pooling_mode,
            uvm_host_mapped,
            embedding_location,
            bounds_check_mode,
        ).validate()

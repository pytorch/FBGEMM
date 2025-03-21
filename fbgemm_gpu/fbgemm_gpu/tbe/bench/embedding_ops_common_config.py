#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import json
from typing import Any, Dict, Optional

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

    @classmethod
    # pyre-ignore [3]
    def from_dict(cls, data: Dict[str, Any]):
        data["weights_dtype"] = SparseType[data["weights_dtype"]]
        data["cache_dtype"] = (
            SparseType[data["cache_dtype"]] if data.get("cache_dtype", None) else None
        )
        data["output_dtype"] = SparseType[data["output_dtype"]]
        data["pooling_mode"] = PoolingMode[data["pooling_mode"]]
        data["embedding_location"] = EmbeddingLocation[data["embedding_location"]]
        data["bounds_check_mode"] = BoundsCheckMode[data["bounds_check_mode"]]
        return cls(**data)

    @classmethod
    # pyre-ignore [3]
    def from_json(cls, data: str):
        return cls.from_dict(json.loads(data))

    def dict(self) -> Dict[str, Any]:
        tmp = dataclasses.asdict(self)
        tmp["weights_dtype"] = self.weights_dtype.name
        tmp["cache_dtype"] = self.cache_dtype.name if self.cache_dtype else None
        tmp["output_dtype"] = self.output_dtype.name
        tmp["pooling_mode"] = self.pooling_mode.name
        tmp["embedding_location"] = self.embedding_location.name
        tmp["bounds_check_mode"] = self.bounds_check_mode.name
        return tmp

    def json(self, format: bool = False) -> str:
        return json.dumps(self.dict(), indent=(2 if format else -1), sort_keys=True)

    # pyre-ignore [3]
    def validate(self):
        return self

    def split_args(self) -> Dict[str, Any]:
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
    # pyre-ignore [2,3]
    def options(cls, emb_location: bool = True):
        # pyre-ignore [2]
        def decorator(func) -> click.Command:
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

            if emb_location:
                options.append(
                    click.option(
                        "--emb-location",
                        default="device",
                        type=click.Choice(
                            ["device", "managed", "managed_caching"],
                            case_sensitive=False,
                        ),
                        help="Memory location of the embeddings",
                    )
                )

            for option in reversed(options):
                func = option(func)
            return func

        return decorator

    @classmethod
    def load(cls, context: click.Context) -> EmbeddingOpsCommonConfig:
        params = context.params

        weights_dtype = params["emb_weights_dtype"]
        cache_dtype = params["emb_cache_dtype"]
        output_dtype = params["emb_output_dtype"]
        stochastic_rounding = params["emb_stochastic_rounding"]
        pooling_mode = PoolingMode[str(params["emb_pooling_mode"]).upper()]
        uvm_host_mapped = params["emb_uvm_host_mapped"]
        bounds_check_mode = BoundsCheckMode(params["emb_bounds_check"])

        embedding_location = EmbeddingLocation[str(params["emb_location"]).upper()]
        if (
            embedding_location is EmbeddingLocation.DEVICE
            and not torch.cuda.is_available()
        ):
            embedding_location = EmbeddingLocation.HOST

        print(f"\n\n\n{weights_dtype}")

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

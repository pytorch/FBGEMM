#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import logging
import re
from enum import Enum

import click
import torch
import yaml

# fmt:skip
from fbgemm_gpu.tbe.bench.tbe_data_config import (
    BatchParams,
    IndicesParams,
    PoolingParams,
    TBEDataConfig,
)


@dataclasses.dataclass(frozen=True)
class TBEDataConfigHelperText(Enum):
    # Config File
    TBE_CONFIG = "TBE data configuration filepath.  If provided, all other `--tbe-*` options are ignored."

    # Table Parameters
    TBE_NUM_TABLES = "Number of tables (T)"
    TBE_NUM_EMBEDDINGS = "Number of embeddings (E)"
    TBE_EMBEDDING_DIM = "Embedding dimensions (D)"
    TBE_MIXED_DIM = "Use mixed dimensions"
    TBE_WEIGHTED = "Flag to indicate if the table is weighted"

    # Batch Parameters
    TBE_BATCH_SIZE = "Batch size (B)"
    TBE_BATCH_VBE_SIGMA = "Standard deviation of B for VBE"
    TBE_BATCH_VBE_DIST = "VBE distribution (choices: 'uniform', 'normal')"
    TBE_BATCH_VBE_RANKS = "Number of ranks for VBE"

    # Indices Parameters
    TBE_INDICES_HITTERS = "Heavy hitters for indices (comma-delimited list of floats)"
    TBE_INDICES_ZIPF = "Zipf distribution parameters for indices generation (q, s)"
    TBE_INDICES_DTYPE = "The dtype of the table indices (choices: '32', '64')"
    TBE_OFFSETS_DTYPE = "The dtype of the table offsets (choices: '32', '64')"

    # Pooling Parameters
    TBE_POOLING_SIZE = "Bag size / pooling factor (L)"
    TBE_POOLING_VL_SIGMA = "Standard deviation of L for variable bag size"
    TBE_POOLING_VL_DIST = (
        "Variable bag size distribution (choices: 'uniform', 'normal')"
    )
    TBE_EMBEDDING_SPECS = "Embedding Specs which is List[Tuple[int, int, EmbeddingLocation, ComputeDevice]]"
    TBE_FEATURE_TABLE_MAP = "Mapping of feature-table"


class TBEDataConfigLoader:
    @classmethod
    # pyre-ignore [2]
    def options(cls, func) -> click.Command:
        options = [
            # Config File
            click.option(
                "--tbe-config",
                type=str,
                required=False,
                help=TBEDataConfigHelperText.TBE_CONFIG.value,
            ),
            # Table Parameters
            click.option(
                "--tbe-num-tables",
                type=int,
                default=32,
                help=TBEDataConfigHelperText.TBE_NUM_TABLES.value,
            ),
            click.option(
                "--tbe-num-embeddings",
                type=int,
                default=int(1e5),
                help=TBEDataConfigHelperText.TBE_NUM_EMBEDDINGS.value,
            ),
            click.option(
                "--tbe-num-embeddings-list",
                type=str,
                required=False,
                default=None,
                help="Comma-separated list of number of embeddings (Es)",
            ),
            click.option(
                "--tbe-embedding-dim",
                type=int,
                default=128,
                help=TBEDataConfigHelperText.TBE_EMBEDDING_DIM.value,
            ),
            click.option(
                "--tbe-embedding-dim-list",
                type=str,
                required=False,
                default=None,
                help="Comma-separated list of number of Embedding dimensions (D)",
            ),
            click.option(
                "--tbe-mixed-dim",
                is_flag=True,
                default=False,
                help=TBEDataConfigHelperText.TBE_MIXED_DIM.value,
            ),
            click.option(
                "--tbe-weighted",
                is_flag=True,
                default=False,
                help=TBEDataConfigHelperText.TBE_WEIGHTED.value,
            ),
            click.option(
                "--tbe-max-indices",
                type=int,
                required=False,
                default=None,
                help="(Optional) Maximum number of indices, will be calculated if not provided",
            ),
            # Batch Parameters
            click.option(
                "--tbe-batch-size",
                type=int,
                default=512,
                help=TBEDataConfigHelperText.TBE_BATCH_SIZE.value,
            ),
            click.option(
                "--tbe-batch-sizes-list",
                type=str,
                required=False,
                default=None,
                help="List Batch sizes per feature (Bs)",
            ),
            click.option(
                "--tbe-batch-vbe-sigma",
                type=int,
                required=False,
                help=TBEDataConfigHelperText.TBE_BATCH_VBE_SIGMA.value,
            ),
            click.option(
                "--tbe-batch-vbe-dist",
                type=click.Choice(["uniform", "normal"]),
                required=False,
                help=TBEDataConfigHelperText.TBE_BATCH_VBE_DIST.value,
            ),
            click.option(
                "--tbe-batch-vbe-ranks",
                type=int,
                required=False,
                help=TBEDataConfigHelperText.TBE_BATCH_VBE_RANKS.value,
            ),
            # Indices Parameters
            click.option(
                "--tbe-indices-hitters",
                type=str,
                default="",
                help=TBEDataConfigHelperText.TBE_INDICES_HITTERS.value,
            ),
            click.option(
                "--tbe-indices-zipf",
                type=(float, float),
                default=(0.1, 0.1),
                help=TBEDataConfigHelperText.TBE_INDICES_ZIPF.value,
            ),
            click.option(
                "--tbe-indices-dtype",
                type=click.Choice(["32", "64"]),
                default="64",
                help=TBEDataConfigHelperText.TBE_INDICES_DTYPE.value,
            ),
            click.option(
                "--tbe-offsets-dtype",
                type=click.Choice(["32", "64"]),
                default="64",
                help=TBEDataConfigHelperText.TBE_OFFSETS_DTYPE.value,
            ),
            # Pooling Parameters
            click.option(
                "--tbe-pooling-size",
                type=int,
                default=20,
                help=TBEDataConfigHelperText.TBE_POOLING_SIZE.value,
            ),
            click.option(
                "--tbe-pooling-vl-sigma",
                type=int,
                required=False,
                help=TBEDataConfigHelperText.TBE_POOLING_VL_SIGMA.value,
            ),
            click.option(
                "--tbe-pooling-vl-dist",
                type=click.Choice(["uniform", "normal"]),
                required=False,
                help=TBEDataConfigHelperText.TBE_POOLING_VL_DIST.value,
            ),
            click.option(
                "--tbe-embedding-specs",
                type=str,
                required=False,
                help=TBEDataConfigHelperText.TBE_EMBEDDING_SPECS.value,
            ),
            click.option(
                "--tbe-feature-table-map",
                type=str,
                required=False,
                help=TBEDataConfigHelperText.TBE_FEATURE_TABLE_MAP.value,
            ),
        ]

        for option in reversed(options):
            func = option(func)
        return func

    @classmethod
    def load_from_file(cls, filepath: str) -> TBEDataConfig:
        with open(filepath, "r") as f:
            if filepath.endswith(".yaml") or filepath.endswith(".yml"):
                data = yaml.safe_load(f)
                return TBEDataConfig.from_dict(data).validate()
            else:
                return TBEDataConfig.from_json(f.read()).validate()

    @classmethod
    def load_from_context(cls, context: click.Context) -> TBEDataConfig:
        params = context.params

        # Read table parameters
        T = params["tbe_num_tables"]  # number of features
        E = params["tbe_num_embeddings"]  # feature_rows
        if params["tbe_num_embeddings_list"] is not None:
            Es = [int(x) for x in params["tbe_num_embeddings_list"].split(",")]
            T = len(Es)
            E = sum(Es) // T  # average E
        else:
            Es = None
        D = params["tbe_embedding_dim"]
        if params["tbe_embedding_dim_list"] is not None:
            Ds = [int(x) for x in params["tbe_embedding_dim_list"].split(",")]
            assert (
                len(Ds) == T
            ), f"Expected tbe_embedding_dim_list to have {T} elements, but got {len(Ds)}"
            D = sum(Ds) // T  # average D
        else:
            Ds = None

        mixed_dim = params["tbe_mixed_dim"]
        weighted = params["tbe_weighted"]
        if params["tbe_max_indices"] is not None:
            max_indices = params["tbe_max_indices"]
        else:
            max_indices = None

        # Read batch parameters
        B = params["tbe_batch_size"]
        sigma_B = params["tbe_batch_vbe_sigma"]
        vbe_distribution = params["tbe_batch_vbe_dist"]
        vbe_num_ranks = params["tbe_batch_vbe_ranks"]
        if params["tbe_batch_sizes_list"] is not None:
            Bs = [int(x) for x in params["tbe_batch_sizes_list"].split(",")]
            B = sum(Bs) // T  # average B
        else:
            B = params["tbe_batch_size"]
            Bs = None
        batch_params = BatchParams(B, sigma_B, vbe_distribution, vbe_num_ranks, Bs)

        # Parse embedding_specs: "(E,D),(E,D),..." or "(E,D,loc,dev),(E,D,loc,dev),..."
        # Only the first two values (E, D) are extracted.
        embedding_specs = None
        feature_table_map = None
        if params["tbe_embedding_specs"] is not None:
            try:
                tuples = re.findall(r"\(([^)]+)\)", params["tbe_embedding_specs"])
                if tuples:
                    embedding_specs = [
                        (int(t.split(",")[0].strip()), int(t.split(",")[1].strip()))
                        for t in tuples
                    ]
            except (ValueError, IndexError):
                logging.warning("Failed to parse embedding_specs. Setting to None.")
        if params["tbe_feature_table_map"] is not None:
            feature_table_map = [
                int(x) for x in params["tbe_feature_table_map"].split(",")
            ]

        # Read indices parameters
        heavy_hitters = (
            torch.tensor([float(x) for x in params["tbe_indices_hitters"].split(",")])
            if params["tbe_indices_hitters"]
            else torch.tensor([])
        )
        zipf_q, zipf_s = params["tbe_indices_zipf"]
        index_dtype = (
            torch.int32 if int(params["tbe_indices_dtype"]) == 32 else torch.int64
        )
        offset_dtype = (
            torch.int32 if int(params["tbe_offsets_dtype"]) == 32 else torch.int64
        )
        indices_params = IndicesParams(
            heavy_hitters, zipf_q, zipf_s, index_dtype, offset_dtype
        )

        # Read pooling parameters
        L = params["tbe_pooling_size"]
        sigma_L = params["tbe_pooling_vl_sigma"]
        length_distribution = params["tbe_pooling_vl_dist"]
        pooling_params = PoolingParams(L, sigma_L, length_distribution)

        return TBEDataConfig(
            T,
            E,
            D,
            mixed_dim,
            weighted,
            batch_params,
            indices_params,
            pooling_params,
            not torch.cuda.is_available(),
            Es,
            Ds,
            max_indices,
            embedding_specs,
            feature_table_map,
        ).validate()

    @classmethod
    def load(cls, context: click.Context) -> TBEDataConfig:
        tbe_config_filepath = context.params["tbe_config"]
        if tbe_config_filepath is not None:
            return cls.load_from_file(tbe_config_filepath)
        else:
            return cls.load_from_context(context)

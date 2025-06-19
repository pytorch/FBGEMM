#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import click
import torch
import yaml

from .tbe_data_config import TBEDataConfig
from .tbe_data_config_param_models import BatchParams, IndicesParams, PoolingParams


class TBEDataConfigLoader:
    @classmethod
    # pyre-ignore [2]
    def options(cls, func) -> click.Command:
        options = [
            ####################################################################
            # Config File
            ####################################################################
            click.option(
                "--tbe-config",
                type=str,
                required=False,
                help="TBE data configuration filepath.  If provided, all other `--tbe-*` options are ignored.",
            ),
            ####################################################################
            # Table Parameters
            ####################################################################
            click.option(
                "--tbe-num-tables",
                type=int,
                default=32,
                help="Number of tables (T)",
            ),
            click.option(
                "--tbe-num-embeddings",
                type=int,
                default=int(1e5),
                help="Number of embeddings (E)",
            ),
            click.option(
                "--tbe-embedding-dim",
                type=int,
                default=128,
                help="Embedding dimensions (D)",
            ),
            click.option(
                "--tbe-mixed-dim",
                is_flag=True,
                default=False,
                help="Use mixed dimensions",
            ),
            click.option(
                "--tbe-weighted",
                is_flag=True,
                default=False,
                help="Whether the table is weighted or not",
            ),
            ####################################################################
            # Batch Parameters
            ####################################################################
            click.option(
                "--tbe-batch-size", type=int, default=512, help="Batch size (B)"
            ),
            click.option(
                "--tbe-batch-vbe-sigma",
                type=int,
                required=False,
                help="Standard deviation of B for VBE",
            ),
            click.option(
                "--tbe-batch-vbe-dist",
                type=click.Choice(["uniform", "normal"]),
                required=False,
                help="VBE distribution",
            ),
            click.option(
                "--tbe-batch-vbe-ranks",
                type=int,
                required=False,
                help="Number of ranks for VBE",
            ),
            ####################################################################
            # Indices Parameters
            ####################################################################
            click.option(
                "--tbe-indices-hitters",
                type=str,
                default="",
                help="TBE heavy hitter indices (comma-delimited list of floats)",
            ),
            click.option(
                "--tbe-indices-zipf",
                type=(float, float),
                default=(0.1, 0.1),
                help="Zipf distribution parameters for indices generation (q, s)",
            ),
            click.option(
                "--tbe-indices-dtype",
                type=click.Choice(["32", "64"]),
                default="64",
                help="The dtype of the table indices",
            ),
            click.option(
                "--tbe-offsets-dtype",
                type=click.Choice(["32", "64"]),
                default="64",
                help="The dtype of the table offsets",
            ),
            ####################################################################
            # Pooling Parameters
            ####################################################################
            click.option(
                "--tbe-pooling-size",
                type=int,
                default=20,
                help="Bag size / pooling factor (L)",
            ),
            click.option(
                "--tbe-pooling-vl-sigma",
                type=int,
                required=False,
                help="Standard deviation of B for VBE",
            ),
            click.option(
                "--tbe-pooling-vl-dist",
                type=click.Choice(["uniform", "normal"]),
                required=False,
                help="Pooling factor distribution",
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
        T = params["tbe_num_tables"]
        E = params["tbe_num_embeddings"]
        D = params["tbe_embedding_dim"]
        mixed_dim = params["tbe_mixed_dim"]
        weighted = params["tbe_weighted"]

        # Read batch parameters
        B = params["tbe_batch_size"]
        sigma_B = params["tbe_batch_vbe_sigma"]
        vbe_distribution = params["tbe_batch_vbe_dist"]
        vbe_num_ranks = params["tbe_batch_vbe_ranks"]
        batch_params = BatchParams(B, sigma_B, vbe_distribution, vbe_num_ranks)

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
        ).validate()

    @classmethod
    def load(cls, context: click.Context) -> TBEDataConfig:
        tbe_config_filepath = context.params["tbe_config"]
        if tbe_config_filepath is not None:
            return cls.load_from_file(tbe_config_filepath)
        else:
            return cls.load_from_context(context)

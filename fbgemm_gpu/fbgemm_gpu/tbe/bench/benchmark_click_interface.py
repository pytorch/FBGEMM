#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import click

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import BoundsCheckMode

from .bench_config import TBEBenchmarkingHelperText
from .tbe_data_config_loader import TBEDataConfigHelperText


class TbeBenchClickInterface:
    @classmethod
    # pyre-ignore [2]
    def common_options(cls, func) -> click.Command:
        options = [
            click.option(
                "--alpha",
                default=1.0,
                help="The alpha value used for the benchmark, default is 1.0. Recommended value: alpha=1.15 for training and alpha=1.09 for inference",
            ),
            click.option(
                "--batch-size",
                default=512,
                help=TBEDataConfigHelperText.TBE_BATCH_SIZE.value + " Default is 512.",
            ),
            click.option(
                "--weights-precision",
                type=SparseType,
                default=SparseType.FP32,
                help="The precision type for weights, default is FP32.",
            ),
            click.option(
                "--stoc",
                is_flag=True,
                default=False,
                help="Flag to enable stochastic rounding, default is False.",
            ),
            click.option(
                "--iters",
                default=100,
                help=TBEBenchmarkingHelperText.BENCH_ITERATIONS.value
                + " Default is 100.",
            ),
            click.option(
                "--warmup-runs",
                default=0,
                help=(
                    TBEBenchmarkingHelperText.BENCH_WARMUP_ITERATIONS.value
                    + " Default is 0."
                ),
            ),
            click.option(  # Note: Original default for uvm bencmark is 0.1
                "--reuse",
                default=0.0,
                help="The inter-batch indices reuse rate for the benchmark, default is 0.0.",
            ),
            click.option(
                "--flush-gpu-cache-size-mb",
                default=0,
                help=TBEBenchmarkingHelperText.BENCH_FLUSH_GPU_CACHE_SIZE.value,
            ),
        ]

        for option in reversed(options):
            func = option(func)
        return func

    @classmethod
    # pyre-ignore [2]
    def table_options(cls, func) -> click.Command:
        options = [
            click.option(
                "--bag-size",
                default=20,
                help=TBEDataConfigHelperText.TBE_POOLING_SIZE.value + " Default is 20.",
            ),
            click.option(
                "--embedding-dim",
                default=128,
                help=TBEDataConfigHelperText.TBE_EMBEDDING_DIM.value
                + " Default is 128.",
            ),
            click.option(
                "--mixed",
                is_flag=True,
                default=False,
                help=TBEDataConfigHelperText.TBE_MIXED_DIM.value + " Default is False.",
            ),
            click.option(
                "--num-embeddings",
                default=int(1e5),
                help=TBEDataConfigHelperText.TBE_NUM_EMBEDDINGS.value
                + " Default is 1e5.",
            ),
            click.option(
                "--num-tables",
                default=32,
                help=TBEDataConfigHelperText.TBE_NUM_TABLES.value + " Default is 32.",
            ),
            click.option(
                "--tables",
                type=str,
                default=None,
                help="Comma-separated list of table numbers Default is None.",
            ),
        ]

        for option in reversed(options):
            func = option(func)
        return func

    @classmethod
    # pyre-ignore [2]
    def device_options(cls, func) -> click.Command:
        options = [
            click.option(
                "--cache-precision",
                type=SparseType,
                default=None,
                help="The precision type for cache, default is None.",
            ),
            click.option(
                "--managed",
                type=click.Choice(
                    ["device", "managed", "managed_caching"], case_sensitive=False
                ),
                default="device",
                help="The managed option for embedding location. Choices are 'device', 'managed', or 'managed_caching'. Default is 'device'.",
            ),
            click.option(
                "--row-wise/--no-row-wise",
                default=True,
                help="Flag to enable or disable row-wise optimization, default is enabled. Use --no-row-wise to disable.",
            ),
            click.option(
                "--weighted",
                is_flag=True,
                default=False,
                help=TBEDataConfigHelperText.TBE_WEIGHTED.value + " Default is False.",
            ),
            click.option(
                "--pooling",
                type=click.Choice(["sum", "mean", "none"], case_sensitive=False),
                default="sum",
                help="The pooling method to use. Choices are 'sum', 'mean', or 'none'. Default is 'sum'.",
            ),
            click.option(
                "--bounds-check-mode",
                type=int,
                default=BoundsCheckMode.NONE.value,
                help="The bounds check mode, default is NONE. Options are: FATAL (0) - Raise an exception (CPU) or device-side assert (CUDA), WARNING (1) - Log the first out-of-bounds instance per kernel, and set to zero, IGNORE (2) - Set to zero, NONE (3) - No bounds checks, V2_IGNORE (4) - IGNORE with V2 enabled, V2_WARNING (5) - WARNING with V2 enabled, V2_FATAL (6) - FATAL with V2 enabled.",
            ),
        ]

        for option in reversed(options):
            func = option(func)
        return func

    @classmethod
    # pyre-ignore [2]
    def vbe_options(cls, func) -> click.Command:
        options = [
            click.option(
                "--bag-size-list",
                type=str,
                default="20",
                help="A comma-separated list of bag sizes for each table, default is '20'.",
            ),
            click.option(
                "--bag-size-sigma-list",
                type=str,
                default="None",
                help="A comma-separated list of bag size standard deviations for generating bag sizes (one std per table). If set, the benchmark will treat --bag-size-list as a list of bag size means. Default is 'None'.",
            ),
        ]

        for option in reversed(options):
            func = option(func)
        return func

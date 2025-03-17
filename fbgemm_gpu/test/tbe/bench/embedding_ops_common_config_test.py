#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest
from typing import List

import click
import torch
from click.testing import CliRunner
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.tbe.bench import (
    EmbeddingOpsCommonConfig,
    EmbeddingOpsCommonConfigLoader,
)


def rand_int(min_value: int, max_value: int) -> int:
    return torch.randint(min_value, max_value, (1,)).tolist()[0]


def clean_command(command: str) -> List[str]:
    return [x for x in command.strip().split() if x]


@click.command()
@EmbeddingOpsCommonConfigLoader.options()
@click.pass_context
# pyre-ignore [2]
def read_config(context: click.Context, **kwargs) -> None:
    config = EmbeddingOpsCommonConfigLoader.load(context)
    # NOTE: This is a hack to pass the parsed config from inside the click CLI
    # runtime back to the test harness
    raise Exception(config.json())


class EmbeddingOpsCommonConfigTest(unittest.TestCase):
    def test_serialization(self) -> None:
        config = EmbeddingOpsCommonConfig(
            SparseType.FP32,
            SparseType.FP16,
            SparseType.BF16,
            random.random() < 0.5,
            PoolingMode.SUM,
            random.random() < 0.5,
            EmbeddingLocation.MANAGED,
            BoundsCheckMode.WARNING,
        )

        assert EmbeddingOpsCommonConfig.from_json(config.json()) == config

    def test_config_load(self) -> None:
        config = EmbeddingOpsCommonConfig(
            SparseType.FP32,
            SparseType.FP16,
            SparseType.BF16,
            random.random() < 0.5,
            PoolingMode.SUM,
            random.random() < 0.5,
            EmbeddingLocation.MANAGED,
            BoundsCheckMode.WARNING,
        )

        args = clean_command(
            # pyre-ignore [16]
            f"""
            --emb-weights-dtype {config.weights_dtype.value}
            --emb-cache-dtype {config.cache_dtype.value}
            --emb-output-dtype {config.output_dtype.value}
            {"--emb-stochastic-rounding" if config.stochastic_rounding else ""}
            --emb-pooling-mode {config.pooling_mode.name}
            {"--emb-uvm-host-mapped" if config.uvm_host_mapped else ""}
            --emb-location {config.embedding_location.name}
            --emb-bounds-check  {config.bounds_check_mode.value}
            """
        )

        runner = CliRunner()
        result = runner.invoke(read_config, args)
        print(str(result.stderr_bytes))
        print(str(result.exception))
        assert EmbeddingOpsCommonConfig.from_json(str(result.exception)) == config

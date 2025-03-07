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
from fbgemm_gpu.tbe.bench import (
    BatchParams,
    IndicesParams,
    PoolingParams,
    TBEDataConfig,
    TBEDataConfigLoader,
)


def rand_int(min_value: int, max_value: int) -> int:
    return torch.randint(min_value, max_value, (1,)).tolist()[0]


def clean_command(command: str) -> List[str]:
    return [x for x in command.strip().split() if x]


@click.command()
@TBEDataConfigLoader.options
@click.pass_context
# pyre-ignore [2]
def read_tbe_config(context: click.Context, **kwargs) -> None:
    config = TBEDataConfigLoader.load(context)
    # NOTE: This is a hack to pass the parsed config from inside the click CLI
    # runtime back to the test harness
    raise Exception(f"{config.json()}")


class TBEDataConfigLoaderTest(unittest.TestCase):
    def test_read_tbe_config_options(self) -> None:
        config = TBEDataConfig(
            rand_int(10, 100),
            rand_int(10, 100),
            rand_int(10, 100),
            random.random() < 0.5,
            random.random() < 0.5,
            BatchParams(
                rand_int(10, 100), rand_int(10, 100), "normal", rand_int(10, 100)
            ),
            IndicesParams(
                torch.rand(20, dtype=torch.float32),
                torch.rand(1).tolist()[0],
                torch.rand(1).tolist()[0],
                torch.int32,
                torch.int64,
            ),
            PoolingParams(rand_int(10, 100), rand_int(10, 100), "normal"),
            not torch.cuda.is_available(),
        )

        args = clean_command(
            f"""
            --tbe-num-tables {config.T}
            --tbe-num-embeddings {config.E}
            --tbe-embedding-dim {config.D}
            {"--tbe-mixed-dim" if config.mixed_dim else ""}
            {"--tbe-weighted" if config.weighted else ""}
            --tbe-batch-size {config.batch_params.B}
            --tbe-batch-vbe-sigma {config.batch_params.sigma_B}
            --tbe-batch-vbe-dist {config.batch_params.vbe_distribution}
            --tbe-batch-vbe-ranks {config.batch_params.vbe_num_ranks}
            --tbe-indices-hitters {",".join([str(x) for x in config.indices_params.heavy_hitters.tolist()])}
            --tbe-indices-zipf {config.indices_params.zipf_q} {config.indices_params.zipf_s}
            --tbe-indices-dtype {32 if config.indices_params.index_dtype == torch.int32 else 64}
            --tbe-offsets-dtype {32 if config.indices_params.offset_dtype == torch.int32 else 64}
            --tbe-pooling-size {config.pooling_params.L}
            --tbe-pooling-vl-sigma {config.pooling_params.sigma_L}
            --tbe-pooling-vl-dist {config.pooling_params.length_distribution}
            """
        )

        runner = CliRunner()
        result = runner.invoke(read_tbe_config, args)
        print(str(result.stderr_bytes))
        print(str(result.exception))
        assert TBEDataConfig.from_json(str(result.exception)) == config

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import click
import torch

from fbgemm_gpu.tbe.bench import IndicesParams


@click.command()
@click.option("--indices", required=True, help="Indices tensor file (*.pt)")
def cli(indices: str) -> None:
    """
    Fetch GitHub commits from a repository with flexible time ranges
    """

    indices = torch.load(indices)
    heavy_hitters, q, s, _, _ = torch.ops.fbgemm.tbe_estimate_indices_distribution(
        indices
    )

    params = IndicesParams(
        heavy_hitters=heavy_hitters, zipf_q=q, zipf_s=s, index_dtype=indices.dtype
    )

    print(params.json(format=True))


if __name__ == "__main__":
    cli()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Tuple

import click
import torch

from fbgemm_gpu.tbe.bench import IndicesParams


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--indices", required=True, help="Indices tensor file (*.pt)")
def estimate(indices: str) -> None:
    """
    Estimate the distribution of indices given a tensor file

    Parameters:
        indices (str): Indices tensor file (*.pt)

    Returns:
        None

    Example:
        estimate --indices="indices.pt"
    """

    indices = torch.load(indices)
    heavy_hitters, q, s, max_index, num_indices = (
        torch.ops.fbgemm.tbe_estimate_indices_distribution(indices)
    )

    params = IndicesParams(
        heavy_hitters=heavy_hitters, zipf_q=q, zipf_s=s, index_dtype=indices.dtype
    )

    print(params.json(format=True), f"max_index={max_index}\nnum_indices={num_indices}")


@cli.command()
@click.option(
    "--hitters",
    type=str,
    default="",
    help="TBE heavy hitter indices (comma-delimited list of floats)",
)
@click.option(
    "--zipf",
    type=(float, float),
    default=(0.1, 0.1),
    help="Zipf distribution parameters for indices generation (q, s)",
)
@click.option(
    "-e",
    "--max-index",
    type=int,
    default=20,
    help="Max index value (E)",
)
@click.option(
    "-n",
    "--num-indices",
    type=int,
    default=20,
    help="Target number of indices to generate",
)
@click.option(
    "--output",
    type=str,
    required=True,
    help="Tensor filepath (*.pt) to save the generated indices",
)
def generate(
    hitters: str,
    zipf: Tuple[float, float],
    max_index: int,
    num_indices: int,
    output: str,
) -> None:
    """
    Generates a tensor of indices given the indices distribution parameters

    Parameters:
        hitters (str): heavy hitter indices (comma-delimited list of floats)

        zipf (Tuple[float, float]): Zipf distribution parameters for indices generation (q, s)

        max_index (int): Max index value (E)

        num_indices (int): Target number of indices to generate

        output (str): Tensor filepath (*.pt) to save the generated indices

    Returns:
        None

    Example:
        generate --hitters="2,4,6" --zipf="1.1,1.1" --max-index=10 --num-indices=100 --output="generated_indices.pt"
    """
    assert max_index > 0, "Max index value (E) must be greater than 0"
    assert num_indices > 0, "Target number of indices must be greater than 0"
    assert zipf[0] > 0, "Zipf parameter q must be greater than 0.0"
    assert zipf[1] > 0, "Zipf parameter s must be greater than 0.0"
    assert output != "", "Output file path must be provided"

    try:
        _hitters: List[float] = (
            [float(x) for x in hitters.split(",")] if hitters else []
        )
    except Exception as e:
        raise AssertionError(
            f'Error: {e}. Please ensure to use comma-delimited list of floats, e.g., --hitters="2,4,6". '
        )

    heavy_hitters = torch.tensor(_hitters)
    assert heavy_hitters.numel() <= 20, "The number of heavy hitters should be <= 20"

    indices = torch.ops.fbgemm.tbe_generate_indices_from_distribution(
        heavy_hitters, zipf[0], zipf[1], max_index, num_indices
    )

    print(f"Generated indices: {indices}")
    torch.save(indices, output)
    print(f"Saved indices to: {output}")


if __name__ == "__main__":
    cli()

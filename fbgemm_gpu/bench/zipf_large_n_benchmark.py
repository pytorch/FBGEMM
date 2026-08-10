#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Manual large-grid overflow reproduction for ``fbgemm.zipf_cuda``."""

import logging

import click
import fbgemm_gpu.sparse_ops  # noqa: F401
import torch

logger: logging.Logger = logging.getLogger(__name__)


@click.command()
@click.option("--alpha", default=1.5, show_default=True, type=float)
@click.option("--n", default=(1 << 32) + 1, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
def main(alpha: float, n: int, seed: int) -> None:
    """Run the allocation-heavy grid-overflow reproduction once."""
    if not torch.cuda.is_available():
        raise RuntimeError("zipf_cuda requires a CUDA or ROCm accelerator")
    if n <= 1 << 32:
        raise ValueError("n must exceed 2**32 to reproduce the grid-overflow case")

    logger.warning(
        "Allocating an int64 output with %d elements (approximately %.1f GiB)",
        n,
        n * 8 / (1 << 30),
    )
    output = torch.ops.fbgemm.zipf_cuda(alpha, n, seed)

    if output.shape != (n,) or output.dtype is not torch.int64:
        raise RuntimeError(
            f"unexpected output contract: shape={output.shape}, dtype={output.dtype}"
        )

    # Citrine C3: create sentinel indices directly on the output device.
    indices = torch.tensor([0, n // 2, n - 1], device=output.device)
    if not bool(torch.all(output[indices] >= 0)):
        raise RuntimeError("zipf_cuda left a sampled output position uninitialized")

    logger.info("Large-grid reproduction completed successfully")


if __name__ == "__main__":
    main()

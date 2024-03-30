#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import sys
from typing import Any

try:
    from .common import CodeTemplate
    from .optimizers import rowwise_adagrad
    from .scripts_argsparse import args
except ImportError:
    # pyre-ignore[21]
    from common import CodeTemplate

    # pyre-ignore[21]
    from optimizers import rowwise_adagrad

    # pyre-ignore[21]
    from scripts_argsparse import args


class EmbeddingOptimizerGenerator:
    @staticmethod
    def generate(**kwargs: Any) -> None:
        """
        Generate embedding optimizer code blocks (host, CUDA host, CUDA kernel,
        and header files) given the optimizer's parameters.
        """

        optimizer = kwargs.get("optimizer")
        kwargs["optimizer_class_name"] = "".join(
            [optim.capitalize() for optim in optimizer.split("_")]
        )
        kwargs["args"] = kwargs["args"].cuda

        # Generate CUDA host code
        CodeTemplate.load("embedding_optimizer_split_template.cu").write(
            f"gen_embedding_optimizer_{optimizer}_split_cuda.cu", **kwargs
        )

        # Generate CUDA kernel code
        CodeTemplate.load("embedding_optimizer_split_kernel_template.cu").write(
            f"gen_embedding_optimizer_{optimizer}_split_kernel.cu", **kwargs
        )

        # Generate host code
        CodeTemplate.load("embedding_optimizer_split_host_template.cpp").write(
            f"gen_embedding_optimizer_{optimizer}_split.cpp", **kwargs
        )

        # Generates Python invoker for CUDA
        CodeTemplate.load("split_embedding_optimizer_codegen.template").write(
            f"split_embedding_optimizer_{optimizer}.py",
            is_fbcode=args.is_fbcode,
            **kwargs,
        )

        # Generate optimizer kernel headers
        CodeTemplate.load("embedding_optimizer_split_device_kernel_template.cuh").write(
            f"gen_embedding_optimizer_{optimizer}_split_device_kernel.cuh", **kwargs
        )


def main() -> None:
    optimizers = [rowwise_adagrad()]

    for optimizer in optimizers:
        EmbeddingOptimizerGenerator.generate(**optimizer)


if __name__ == "__main__":
    print(f"[GENERATE OPTIMIZERS] {sys.argv}")
    main()

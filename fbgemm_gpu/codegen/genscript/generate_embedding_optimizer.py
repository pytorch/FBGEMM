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
    def generate_embedding_optimizer(**kwargs: Any) -> None:
        """
        Generate embedding optimizer code blocks (host, CUDA host, CUDA kernel,
        and header files) given the optimizer's parameters.
        """

        optimizer = kwargs.get("optimizer")
        kwargs["optimizer_class_name"] = "".join(
            [optim.capitalize() for optim in optimizer.split("_")]
        )
        kwargs["args"] = kwargs["args"].cuda

        PREFIX = "training/optimizer"

        for template_filepath, filename in [
            (  # CUDA host code
                f"{PREFIX}/embedding_optimizer_split_template.cu",
                f"gen_embedding_optimizer_{optimizer}_split_cuda.cu",
            ),
            (  # CUDA kernel code
                f"{PREFIX}/embedding_optimizer_split_kernel_template.cu",
                f"gen_embedding_optimizer_{optimizer}_split_kernel.cu",
            ),
            (  # CPU code
                f"{PREFIX}/embedding_optimizer_split_host_template.cpp",
                f"gen_embedding_optimizer_{optimizer}_split.cpp",
            ),
            (  # Optimizer kernel headers
                f"{PREFIX}/embedding_optimizer_split_device_kernel_template.cuh",
                f"gen_embedding_optimizer_{optimizer}_split_device_kernel.cuh",
            ),
            (  # Python kernel invokers
                "training/python/split_embedding_optimizer_codegen.template",
                f"split_embedding_optimizer_{optimizer}.py",
            ),
        ]:
            CodeTemplate.load(template_filepath).write(
                filename, is_fbcode=args.is_fbcode, **kwargs
            )

    @staticmethod
    def generate() -> None:
        optimizers = [rowwise_adagrad()]

        for optimizer in optimizers:
            EmbeddingOptimizerGenerator.generate_embedding_optimizer(**optimizer)

        CodeTemplate.copy_to_root("training/python/optimizer_args.py")


def main() -> None:
    EmbeddingOptimizerGenerator.generate()


if __name__ == "__main__":
    print(f"[GENERATE OPTIMIZERS]: {sys.argv}")
    main()

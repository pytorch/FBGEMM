#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa F401

from typing import Optional

try:
    # Internal
    from .embedding_common_code_generator import *
except ImportError:
    # OSS
    from embedding_common_code_generator import *


def _generate(**kwargs: Any) -> None:
    gen_args = kwargs["args"]
    kwargs["args"] = gen_args["cuda"]
    optimizer = kwargs.get("optimizer")

    # Generate cuda host code
    template = env.get_template("embedding_optimizer_split_template.cu")
    write(
        f"gen_embedding_optimizer_{optimizer}_split_cuda.cu", template.render(**kwargs)
    )

    # Generate host code
    template = env.get_template("embedding_optimizer_split_host_template.cpp")
    write(f"gen_embedding_optimizer_{optimizer}_split.cpp", template.render(**kwargs))

    template = env.get_template("embedding_optimizer_split_kernel_template.cu")
    write(
        f"gen_embedding_optimizer_{optimizer}_split_kernel.cu",
        template.render(**kwargs),
    )

    # Generates Python invoker for CUDA
    template = env.get_template("split_embedding_optimizer_codegen.template")
    write(
        f"split_embedding_optimizer_{optimizer}.py",
        template.render(is_fbcode=args.is_fbcode, **kwargs),
    )

    # Generate optimizer kernel
    template = env.get_template("embedding_optimizer_split_device_kernel_template.cuh")
    write(
        f"gen_embedding_optimizer_{optimizer}_split_device_kernel.cuh",
        template.render(**kwargs),
    )


def generate(**kwargs: Any) -> None:
    _generate(
        optimizer_class_name="".join(
            [optim.capitalize() for optim in kwargs["optimizer"].split("_")]
        ),
        **kwargs,
    )


def optimizer_codegen(
    install_dir: Optional[str] = None, is_fbcode: Optional[bool] = None
) -> None:
    if install_dir is not None and len(install_dir) != 0:
        args.install_dir = install_dir
    if is_fbcode is not None:
        args.is_fbcode = is_fbcode

    # Generate optimizers
    generate(**(rowwise_adagrad()))


def main() -> None:
    optimizer_codegen()


if __name__ == "__main__":
    main()

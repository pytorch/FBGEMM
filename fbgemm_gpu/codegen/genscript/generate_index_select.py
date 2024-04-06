#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import re
import sys
from typing import Optional

try:
    from .common import CodeTemplate
    from .optimizer_args import FLOAT, OptimizerArgsSet
except ImportError:
    # pyre-ignore[21]
    from common import CodeTemplate

    # pyre-ignore[21]
    from optimizer_args import FLOAT, OptimizerArgsSet


class IndexSelectGenerator:
    @staticmethod
    def generate() -> None:
        optargs = OptimizerArgsSet.create([(FLOAT, "unused")])
        for template_file, generated_file in [
            (
                "training/forward/embedding_forward_split_template.cu",
                "gen_batch_index_select_dim0_forward_codegen_cuda.cu",
            ),
            (
                "training/forward/embedding_forward_split_kernel_template.cu",
                "gen_batch_index_select_dim0_forward_kernel.cu",
            ),
            (
                "training/forward/embedding_forward_split_kernel_nobag_small_template.cu",
                "gen_batch_index_select_dim0_forward_kernel_small.cu",
            ),
            (
                "training/backward/embedding_backward_split_template.cu",
                "gen_batch_index_select_dim0_backward_codegen_cuda.cu",
            ),
            (
                "training/backward/embedding_backward_split_kernel_cta_template.cu",
                "gen_batch_index_select_dim0_backward_kernel_cta.cu",
            ),
            (
                "training/backward/embedding_backward_split_kernel_warp_template.cu",
                "gen_batch_index_select_dim0_backward_kernel_warp.cu",
            ),
            (
                "training/backward/embedding_backward_split_device_kernel_template.cuh",
                "gen_embedding_backward_batch_index_select_split_device_kernel.cuh",
            ),
        ]:
            CodeTemplate.load(template_file).write(
                generated_file,
                weighted=False,
                dense=True,
                vbe=False,
                nobag=True,
                is_index_select=True,
                gen_once=False,
                kdesc="batch_index_select",
                args=optargs.cuda,
            )

        CodeTemplate.load(
            "training/backward/embedding_backward_split_grad_template.cu"
        ).write(
            "gen_embedding_backward_split_grad_index_select.cu",
            is_index_select=True,
        )

        # Generate common backward device kernels (generate only once)
        CodeTemplate.load(
            "training/backward/embedding_backward_split_device_kernel_template.cuh"
        ).write(
            "gen_embedding_backward_common_split_device_kernel.cuh",
            gen_once=True,
        )


def main() -> None:
    IndexSelectGenerator.generate()


if __name__ == "__main__":
    print(f"[INDEX SELECT GENERATOR]: {sys.argv}")
    main()

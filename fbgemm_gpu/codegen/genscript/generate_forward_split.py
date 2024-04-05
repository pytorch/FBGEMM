#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import sys
from typing import List

try:
    from .common import CodeTemplate
except ImportError:
    # pyre-ignore[21]
    from common import CodeTemplate


class ForwardSplitGenerator:
    @staticmethod
    def render_forward_templates(
        template_filepath: str,
        filename_format: str,
        dense_options: List[bool],
        nobag_options: List[bool],
        vbe_options: List[bool],
    ) -> None:
        template = CodeTemplate.load(template_filepath)
        for dense in dense_options:
            for weighted in [True, False]:
                for nobag in nobag_options:
                    for vbe in vbe_options:
                        if (not nobag or (not weighted and not vbe)) and (
                            not dense or not vbe
                        ):
                            dense_desc = f"{ 'dense' if dense else 'split'}"
                            weight_desc = (
                                f"{ 'weighted' if weighted else 'unweighted' }"
                            )
                            nobag_desc = f"{ '_nobag' if nobag else '' }"
                            vbe_desc = f"{ '_vbe' if vbe else '' }"

                            template.write(
                                filename_format.format(
                                    f"{ dense_desc }_{ weight_desc }{ nobag_desc }{ vbe_desc }"
                                ),
                                dense=dense,
                                weighted=weighted,
                                nobag=nobag,
                                vbe=vbe,
                                is_index_select=False,
                            )

    @staticmethod
    def generate_pt2_wrappers() -> None:
        # Generate PT2 forward wrapper (CUDA)
        CodeTemplate.load(
            "training/pt2/embedding_split_host_pt2_cuda_wrapper_template.cpp",
        ).write(
            f"gen_embedding_forward_split_pt2_cuda_wrapper.cpp",
            has_gpu_support=True,
            is_forward=True,
            has_vbe_support=True,
        )

        # Generate PT2 forward wrapper (CPU)
        CodeTemplate.load(
            "training/pt2/embedding_split_host_pt2_cpu_wrapper_template.cpp",
        ).write(
            f"gen_embedding_forward_split_pt2_cpu_wrapper.cpp",
            has_cpu_support=True,
            is_forward=True,
        )

    @staticmethod
    def generate_small_kernels() -> None:
        # Generate the small kernels (for nobag only) for the forward splits
        template = CodeTemplate.load(
            "training/forward/embedding_forward_split_kernel_nobag_small_template.cu"
        )
        for dense in [True, False]:
            wdesc = f"{ 'dense' if dense else 'split' }"
            template.write(
                f"gen_embedding_forward_{wdesc}_unweighted_nobag_kernel_small.cu",
                dense=dense,
                is_index_select=False,
            )

    @staticmethod
    def generate_kernels() -> None:
        # Generate the CUDA host code
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_template.cu",
            "gen_embedding_forward_{}_codegen_cuda.cu",
            dense_options=[True, False],
            nobag_options=[False],  # nobag is not used
            vbe_options=[True, False],
        )

        # Generate the meta kernels
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_meta_template.cpp",
            "gen_embedding_forward_{}_codegen_meta.cpp",
            dense_options=[True, False],
            nobag_options=[False],  # nobag is not used
            vbe_options=[True, False],
        )

        # Generate the CUDA kernels
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_kernel_template.cu",
            "gen_embedding_forward_{}_kernel.cu",
            dense_options=[True, False],
            nobag_options=[True, False],
            vbe_options=[True, False],
        )

        # Generate the v2 CUDA kernels
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_kernel_v2_template.cu",
            "gen_embedding_forward_{}_v2_kernel.cu",
            dense_options=[False],  # dense is not supported
            nobag_options=[False],  # nobag is not supported
            vbe_options=[False],  # vbe is not supported
        )

    @staticmethod
    def generate() -> None:
        ForwardSplitGenerator.generate_kernels()
        ForwardSplitGenerator.generate_small_kernels()
        ForwardSplitGenerator.generate_pt2_wrappers()


def main() -> None:
    ForwardSplitGenerator.generate()


if __name__ == "__main__":
    print(f"[GENERATE FORWARD SPLIT]: {sys.argv}")
    main()

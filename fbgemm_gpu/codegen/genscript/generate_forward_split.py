#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import argparse
import itertools
import sys
from typing import List

try:
    from .common import CodeTemplate
    from .optimizer_args import annotation_dict
except ImportError:
    # pyre-ignore[21]
    from common import CodeTemplate

    # pyre-ignore[21]
    from optimizer_args import annotation_dict


class ForwardSplitGenerator:
    @staticmethod
    def render_forward_templates(
        template_filepath: str,
        filename_format: str,
        dense_options: List[bool],
        nobag_options: List[bool],
        vbe_options: List[bool],
        ssd_options: List[bool],
        is_gwd: bool = False,
    ) -> None:
        template = CodeTemplate.load(template_filepath)
        weighted_options = [True, False]

        for dense, weighted, nobag, vbe, ssd in itertools.product(
            dense_options, weighted_options, nobag_options, vbe_options, ssd_options
        ):
            if nobag and (weighted or vbe):
                continue
            if dense and ssd:
                continue
            if ssd and is_gwd:
                continue

            desc = "".join(
                [
                    f"{ 'dense' if dense else ('ssd' if ssd else 'split') }",
                    f"{ '_weighted' if weighted else '_unweighted' }",
                    f"{ '_nobag' if nobag else '' }",
                    f"{ '_vbe' if vbe else '' }",
                ]
            )
            fname = filename_format.format(desc)
            template.write(
                fname,
                dense=dense,
                weighted=weighted,
                nobag=nobag,
                vbe=vbe,
                ssd=ssd,
                is_index_select=False,
                is_gwd=is_gwd,
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
            schema_annotation=annotation_dict,
        )

        # Generate PT2 forward wrapper (CPU)
        CodeTemplate.load(
            "training/pt2/embedding_split_host_pt2_cpu_wrapper_template.cpp",
        ).write(
            f"gen_embedding_forward_split_pt2_cpu_wrapper.cpp",
            has_cpu_support=True,
            is_forward=True,
            has_vbe_support=True,
            schema_annotation=annotation_dict,
        )

        # Generate SSD PT2 forward wrapper (CUDA)
        CodeTemplate.load(
            "training/pt2/embedding_split_host_pt2_cuda_wrapper_template.cpp",
        ).write(
            f"gen_embedding_forward_ssd_pt2_cuda_wrapper.cpp",
            has_gpu_support=True,
            is_forward=True,
            has_vbe_support=True,
            ssd=True,
            schema_annotation=annotation_dict,
        )

    @staticmethod
    def generate_small_kernels() -> None:
        # Generate the small kernels (for nobag only) for the forward splits
        template = CodeTemplate.load(
            "training/forward/embedding_forward_split_kernel_nobag_small_template.cu"
        )
        for dense in [True, False]:
            for ssd in [True, False]:
                ddesc = f"{ 'dense' if dense else ('ssd' if ssd else 'split') }"
                template.write(
                    f"gen_embedding_forward_{ ddesc }_unweighted_nobag_kernel_small.cu",
                    dense=dense,
                    ssd=ssd,
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
            ssd_options=[True, False],
        )
        # Generate the CUDA host code for global weight decay
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_template.cu",
            "gen_embedding_forward_{}_gwd_codegen_cuda.cu",
            dense_options=[False],
            nobag_options=[False],  # nobag is not used
            vbe_options=[True, False],
            is_gwd=True,
            ssd_options=[False],
        )

        # Generate the meta kernels
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_meta_template.cpp",
            "gen_embedding_forward_{}_codegen_meta.cpp",
            dense_options=[True, False],
            nobag_options=[False],  # nobag is not used
            vbe_options=[True, False],
            ssd_options=[True, False],
        )

        # Generate the CUDA kernels
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_kernel_template.cu",
            "gen_embedding_forward_{}_kernel.cu",
            dense_options=[True, False],
            nobag_options=[True, False],
            vbe_options=[True, False],
            ssd_options=[True, False],
        )
        # Generate the global weight decay CUDA kernels
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_kernel_template.cu",
            "gen_embedding_forward_{}_gwd_kernel.cu",
            dense_options=[False],
            nobag_options=[False],
            vbe_options=[True, False],
            ssd_options=[False],
            is_gwd=True,
        )

        # Generate the v2 CUDA kernels
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_kernel_v2_template.cu",
            "gen_embedding_forward_{}_v2_kernel.cu",
            dense_options=[False],  # dense is not supported
            nobag_options=[False],  # nobag is not supported
            vbe_options=[False],  # vbe is not supported
            ssd_options=[False],  # ssd is not supported
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

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import sys

try:
    from .optimizers import *
    from .common import CodeTemplate
    from .optimizer_args import OptimizerArgsSet
    from .scripts_argsparse import args
except ImportError:
    from optimizers import *

    # pyre-ignore[21]
    from common import CodeTemplate

    # pyre-ignore[21]
    from optimizer_args import OptimizerArgsSet

    # pyre-ignore[21]
    from scripts_argsparse import args


class BackwardSplitGenerator:
    @staticmethod
    def render_backward_templates(
        template_filepath: str,
        optimizer: str,
        filename_format: str,
        kwargs: Dict[str, Any],
    ) -> None:
        if not kwargs.get("has_gpu_support"):
            return
        vbe_options = [True, False] if kwargs.get("has_vbe_support") else [False]
        template = CodeTemplate.load(template_filepath)

        for weighted in [True, False]:
            for nobag in [True, False]:
                for vbe in vbe_options:
                    if (not nobag or (not weighted and not vbe)) and (
                        not kwargs.get("dense") or not vbe
                    ):
                        wdesc = f"{ 'weighted' if weighted else 'unweighted' }{ '_nobag' if nobag else '' }{ '_vbe' if vbe else '' }"
                        template.write(
                            filename_format.format(optimizer, wdesc),
                            weighted=weighted,
                            nobag=nobag,
                            vbe=vbe,
                            is_index_select=False,
                            kdesc=wdesc,
                            **kwargs,
                        )

    @staticmethod
    def generate_backward_split_gpu(**kwargs: Any) -> None:
        """
        Generate CUDA variants of the TBE backward split operators
        """

        optimizer = kwargs.get("optimizer")
        # Generate the backward split kernels
        for template_filepath, filename_format in [
            (
                "training/backward/embedding_backward_split_template.cu",
                "gen_embedding_backward_{}_split_{}_cuda.cu",
            ),
            (
                "training/backward/embedding_backward_split_meta_template.cpp",
                "gen_embedding_backward_{}_split_{}_meta.cpp",
            ),
            (
                "training/backward/embedding_backward_split_kernel_cta_template.cu",
                "gen_embedding_backward_{}_split_{}_kernel_cta.cu",
            ),
            (
                "training/backward/embedding_backward_split_kernel_warp_template.cu",
                "gen_embedding_backward_{}_split_{}_kernel_warp.cu",
            ),
        ]:
            BackwardSplitGenerator.render_backward_templates(
                template_filepath,
                optimizer,
                filename_format,
                kwargs,
            )

        # Generate optimizer kernel
        CodeTemplate.load(
            "training/optimizer/embedding_optimizer_split_device_kernel_template.cuh"
        ).write(
            f"gen_embedding_optimizer_{optimizer}_split_device_kernel.cuh", **kwargs
        )

        # Generate the backward splits (non-dense)
        # We generate only the API to preserve the backward compatibility if
        # has_gpu_support=True
        if not kwargs.get("dense"):
            # Generate CUDA autograd, PT2 unified autograd, and PT2 backward wrapper
            for template_filepath, filename in [
                (
                    "training/backward/embedding_backward_split_host_template.cpp",
                    f"gen_embedding_backward_split_{optimizer}.cpp",
                ),
                (
                    "training/pt2/embedding_split_host_pt2_autograd_template.cpp",
                    f"gen_embedding_split_{optimizer}_pt2_autograd.cpp",
                ),
                (
                    "training/pt2/embedding_split_host_pt2_cuda_wrapper_template.cpp",
                    f"gen_embedding_backward_split_{optimizer}_pt2_cuda_wrapper.cpp",
                ),
            ]:
                CodeTemplate.load(template_filepath).write(
                    filename, is_forward=False, **kwargs
                )

            if kwargs.get("has_cpu_support") or kwargs.get("has_gpu_support"):
                # Generates Python invoker for CUDA + CPU, and PT2
                template = CodeTemplate.load(
                    "training/python/split_embedding_codegen_lookup_invoker.template"
                )
                for filename in [
                    f"lookup_{optimizer}.py",
                    f"lookup_{optimizer}_pt2.py",
                ]:
                    template.write(filename, is_fbcode=args.is_fbcode, **kwargs)

    @staticmethod
    def generate_backward_split_cpu(**kwargs: Any) -> None:
        """
        Generate CPU variants of the TBE backward split operators
        """

        optimizer = kwargs.get("optimizer")

        # Generate the backward splits
        if kwargs.get("has_cpu_support"):
            CodeTemplate.load(
                "training/backward/embedding_backward_split_cpu_approx_template.cpp"
                if "approx" in optimizer
                else "training/backward/embedding_backward_split_cpu_template.cpp"
            ).write(f"gen_embedding_backward_{optimizer}_split_cpu.cpp", **kwargs)

        # Generate the backward splits (non-dense)
        if not kwargs.get("dense"):
            for template_filepath, filename in [
                (
                    "training/backward/embedding_backward_split_host_cpu_template.cpp",
                    f"gen_embedding_backward_split_{optimizer}_cpu.cpp",
                ),
                (
                    "training/pt2/embedding_split_host_pt2_cpu_wrapper_template.cpp",
                    f"gen_embedding_backward_split_{optimizer}_pt2_cpu_wrapper.cpp",
                ),
            ]:
                CodeTemplate.load(template_filepath).write(
                    filename, is_forward=False, **kwargs
                )

    @staticmethod
    def generate_backward_split(**kwargs: Any) -> None:
        gen_args = kwargs["args"]
        kwargs["args_pt2"] = gen_args.any

        kwargs["args"] = gen_args.cuda
        BackwardSplitGenerator.generate_backward_split_gpu(**kwargs)

        kwargs["args"] = gen_args.cpu
        BackwardSplitGenerator.generate_backward_split_cpu(**kwargs)

    @staticmethod
    def generate_backward_device() -> None:
        # Generate backward device kernels based on weighted (True/False), VBE
        # (True/False), no bag (True/False)
        template_filepath = (
            "training/backward/embedding_backward_split_device_kernel_template.cuh"
        )

        BackwardSplitGenerator.render_backward_templates(
            template_filepath,
            "",
            "{}gen_embedding_backward_{}_split_device_kernel.cuh",
            {
                "has_gpu_support": True,
                "has_vbe_support": True,
                "dense": False,
                "gen_once": False,
            },
        )

        # Generate common backward device kernels (generate only once)
        CodeTemplate.load(template_filepath).write(
            "gen_embedding_backward_common_split_device_kernel.cuh",
            gen_once=True,
        )

    @staticmethod
    def generate_backward_grad() -> None:
        # Generate the common grad functions
        CodeTemplate.load(
            "training/backward/embedding_backward_split_grad_template.cu"
        ).write(
            "gen_embedding_backward_split_grad_embedding_ops.cu", is_index_select=False
        )

    @staticmethod
    def generate_backward_indices() -> None:
        template = CodeTemplate.load(
            "training/backward/embedding_backward_split_indice_weights_template.cu"
        )
        for dense in [True, False]:
            template.write(
                f"gen_embedding_backward_{'dense' if dense else 'split'}_indice_weights_codegen_cuda.cu",
                dense=dense,
            )

    @staticmethod
    def generate_python_sources() -> None:
        CodeTemplate.load("training/python/__init__.template").write("__init__.py")
        CodeTemplate.copy_to_root("training/python/lookup_args.py")

    @staticmethod
    def generate() -> None:
        # Generate backwards and optimizers
        optimizers = [
            dense(),
            adagrad(),
            adam(),
            lamb(),
            lars_sgd(),
            partial_rowwise_adam(),
            partial_rowwise_lamb(),
            rowwise_adagrad(),
            approx_rowwise_adagrad(),
            rowwise_adagrad_with_weight_decay(),
            approx_rowwise_adagrad_with_weight_decay(),
            rowwise_adagrad_with_counter(),
            approx_rowwise_adagrad_with_counter(),
            rowwise_weighted_adagrad(),
            sgd(),
            approx_sgd(),
            none_optimizer(),
        ]

        for optimizer in optimizers:
            BackwardSplitGenerator.generate_backward_split(**optimizer)

        # Generate common device kernels for backwards
        BackwardSplitGenerator.generate_backward_device()

        # Generate forwards and specialized backwards
        BackwardSplitGenerator.generate_backward_grad()
        BackwardSplitGenerator.generate_backward_indices()

        BackwardSplitGenerator.generate_python_sources()


def main() -> None:
    BackwardSplitGenerator.generate()


if __name__ == "__main__":
    print(f"[GENERAATE BACKWARD SPLIT]: {sys.argv}")
    main()

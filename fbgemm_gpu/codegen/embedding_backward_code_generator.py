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
    # Internal
    from .embedding_common_code_generator import *
except ImportError:
    # OSS
    from embedding_common_code_generator import *

import re


def generate_backward_embedding_cuda(
    template_filepath: str,
    optimizer: str,
    filename_format: str,
    kwargs: Dict[str, Any],
) -> None:
    if not kwargs.get("has_gpu_support"):
        return
    template = env.get_template(template_filepath)
    vbe_options = [True, False] if kwargs.get("has_vbe_support") else [False]
    for weighted in [True, False]:
        for nobag in [True, False]:
            for vbe in vbe_options:
                if (not nobag or (not weighted and not vbe)) and (
                    not kwargs.get("dense") or not vbe
                ):
                    wdesc = f"{ 'weighted' if weighted else 'unweighted' }{ '_nobag' if nobag else '' }{ '_vbe' if vbe else '' }"
                    filename = filename_format.format(optimizer, wdesc)
                    write(
                        filename,
                        template.render(
                            weighted=weighted,
                            nobag=nobag,
                            vbe=vbe,
                            is_index_select=False,
                            kdesc=wdesc,
                            **kwargs,
                        ),
                    )
                    print(f"[Backward Split] [{optimizer}]: {filename}")


def generate(**kwargs: Any) -> None:
    optimizer = kwargs.get("optimizer")
    gen_args = kwargs["args"]

    #
    # Generate GPU variants of the operators
    #
    kwargs["args"] = gen_args["cuda"]
    kwargs["args_pt2"] = gen_args["any_device"]

    # Generate the backward splits
    generate_backward_embedding_cuda(
        "embedding_backward_split_template.cu",
        optimizer,
        "gen_embedding_backward_{}_split_{}_cuda.cu",
        kwargs,
    )

    generate_backward_embedding_cuda(
        "embedding_backward_split_meta_template.cpp",
        optimizer,
        "gen_embedding_backward_{}_split_{}_meta.cpp",
        kwargs,
    )

    # Generate the cta_per_row kernels for the backward splits
    generate_backward_embedding_cuda(
        "embedding_backward_split_kernel_cta_template.cu",
        optimizer,
        "gen_embedding_backward_{}_split_{}_kernel_cta.cu",
        kwargs,
    )

    # Generate the warp_per_row kernels for the backward splits
    generate_backward_embedding_cuda(
        "embedding_backward_split_kernel_warp_template.cu",
        optimizer,
        "gen_embedding_backward_{}_split_{}_kernel_warp.cu",
        kwargs,
    )

    # Generate optimizer kernel
    template = env.get_template("embedding_optimizer_split_device_kernel_template.cuh")
    filename = f"gen_embedding_optimizer_{optimizer}_split_device_kernel.cuh"
    write(filename, template.render(**kwargs))

    # Generate the backward splits (non-dense)
    # We generate only the API to preserve the backward compatibility if
    # has_gpu_support=True
    if not kwargs.get("dense"):
        # TO DO: deprecate
        # Generate CUDA Autograd
        template = env.get_template("embedding_backward_split_host_template.cpp")
        filename = f"gen_embedding_backward_split_{optimizer}.cpp"
        write(filename, template.render(**kwargs))
        print(f"[Backward Split] [{optimizer}]: {filename}")

        if kwargs.get("has_cpu_support") or kwargs.get("has_gpu_support"):
            # Generates Python invoker for CUDA + CPU
            template = env.get_template(
                "split_embedding_codegen_lookup_invoker.template"
            )
            filename = f"lookup_{optimizer}.py"
            write(filename, template.render(is_fbcode=args.is_fbcode, **kwargs))
            print(f"[Backward Split] [{optimizer}]: {filename}")

        # Generate PT2 unified Autograd
        template_pt2 = env.get_template(
            "embedding_split_host_pt2_autograd_template.cpp"
        )
        filename_pt2 = f"gen_embedding_split_{optimizer}_pt2_autograd.cpp"
        write(filename_pt2, template_pt2.render(**kwargs))
        print(f"[Backward Split] [{optimizer}]: {filename_pt2}")

        # Generate PT2 backward wrapper
        template_pt2 = env.get_template(
            "embedding_split_host_pt2_cuda_wrapper_template.cpp"
        )
        filename_pt2 = f"gen_embedding_backward_split_{optimizer}_pt2_cuda_wrapper.cpp"
        write(filename_pt2, template_pt2.render(is_forward=False, **kwargs))
        print(f"[Backward Split] [{optimizer}]: {filename_pt2}")

        if kwargs.get("has_cpu_support") or kwargs.get("has_gpu_support"):
            # Generate Python invoker for CUDA + CPU PT2
            template_pt2 = env.get_template(
                "split_embedding_codegen_lookup_invoker.template"
            )
            filename_pt2 = f"lookup_{optimizer}_pt2.py"
            write(filename_pt2, template_pt2.render(is_fbcode=args.is_fbcode, **kwargs))
            print(f"[Backward Split] [{optimizer}]: {filename_pt2}")

    #
    # Generate CPU variants of the operators
    #
    kwargs["args"] = gen_args["cpu"]
    kwargs["args_pt2"] = gen_args["any_device"]

    # Generate the backward splits
    if kwargs.get("has_cpu_support"):
        is_approx = "approx" in optimizer
        template = (
            env.get_template("embedding_backward_split_cpu_approx_template.cpp")
            if is_approx
            else env.get_template("embedding_backward_split_cpu_template.cpp")
        )
        filename = f"gen_embedding_backward_{optimizer}_split_cpu.cpp"
        write(filename, template.render(**kwargs))
        print(f"[Backward Split] [{optimizer}]: {filename}")

    # Generate the backward splits (non-dense)
    if not kwargs.get("dense"):
        template = env.get_template("embedding_backward_split_host_cpu_template.cpp")
        filename = f"gen_embedding_backward_split_{optimizer}_cpu.cpp"
        write(filename, template.render(**kwargs))
        print(f"[Backward Split] [{optimizer}]: {filename}")

        # Generate PT2 backward wrapper functions
        template_pt2 = env.get_template(
            "embedding_split_host_pt2_cpu_wrapper_template.cpp"
        )
        filename_pt2 = f"gen_embedding_backward_split_{optimizer}_pt2_cpu_wrapper.cpp"
        write(filename_pt2, template_pt2.render(is_forward=False, **kwargs))
        print(f"[Backward Split] [{optimizer}]: {filename_pt2}")


# Format the way to generate PackedTensorAccessors
def make_pta_acc_format(pta_str_list: List[str], func_name: str) -> List[str]:
    new_str_list = []
    for pta_str in pta_str_list:
        if "packed_accessor" in pta_str:
            match = re.search(
                r"([a-zA-z0-9_]*)[.]packed_accessor([3|6][2|4])<(.*)>\(\)", pta_str
            )
            assert match is not None and len(match.groups()) == 3
            tensor, acc_nbits, args = match.groups()
            if "acc_type" in args:
                match = re.search("at::acc_type<([a-zA-Z_]*), true>", args)
                assert match is not None and len(match.groups()) == 1
                new_type = match.group(1)
                args = re.sub("at::acc_type<[a-zA-Z_]*, true>", new_type, args)
                func_name_suffix = "_ACC_TYPE"
            else:
                func_name_suffix = ""
            new_str_list.append(
                f"{func_name}{func_name_suffix}({tensor}, {args}, {acc_nbits})"
            )
        else:
            new_str_list.append(pta_str)
    return new_str_list


def replace_pta_namespace(pta_str_list: List[str]) -> List[str]:
    return [
        pta_str.replace("at::PackedTensorAccessor", "pta::PackedTensorAccessor")
        for pta_str in pta_str_list
    ]


def backward_indices() -> None:
    template = env.get_template("embedding_backward_split_indice_weights_template.cu")
    src_cu = template.render()
    write("gen_embedding_backward_split_indice_weights_codegen_cuda.cu", src_cu)
    src_cu = template.render(dense=True)
    write("gen_embedding_backward_dense_indice_weights_codegen_cuda.cu", src_cu)


def backward_dense() -> None:
    generate(
        optimizer="dense",
        dense=True,
        args=make_args(
            [
                (FLOAT, "unused"),
            ]
        ),
        split_precomputation=split_precomputation,
        split_weight_update=split_weight_update,
        split_post_update="",
        split_weight_update_cpu=split_weight_update_cpu,
        has_cpu_support=False,
        has_gpu_support=True,
        has_vbe_support=False,
    )


def generate_forward_embedding_cuda(
    template_filepath: str,
    filename_format: str,
    dense_options: List[bool],
    nobag_options: List[bool],
    vbe_options: List[bool],
) -> None:
    template = env.get_template(template_filepath)
    for dense in dense_options:
        for weighted in [True, False]:
            for nobag in nobag_options:
                for vbe in vbe_options:
                    if (not nobag or (not weighted and not vbe)) and (
                        not dense or not vbe
                    ):
                        dense_desc = f"{ 'dense' if dense else 'split'}"
                        weight_desc = f"{ 'weighted' if weighted else 'unweighted' }"
                        nobag_desc = f"{ '_nobag' if nobag else '' }"
                        vbe_desc = f"{ '_vbe' if vbe else '' }"
                        desc = (
                            f"{ dense_desc }_{ weight_desc }{ nobag_desc }{ vbe_desc }"
                        )
                        filename = filename_format.format(desc)
                        write(
                            filename,
                            template.render(
                                dense=dense,
                                weighted=weighted,
                                nobag=nobag,
                                vbe=vbe,
                                is_index_select=False,
                            ),
                        )
                        print(f"[Forward Split]: {filename}")


def forward_split() -> None:
    # Generate the forward splits
    generate_forward_embedding_cuda(
        "embedding_forward_split_template.cu",
        "gen_embedding_forward_{}_codegen_cuda.cu",
        dense_options=[True, False],
        nobag_options=[False],  # nobag is not used
        vbe_options=[True, False],
    )

    generate_forward_embedding_cuda(
        "embedding_forward_split_meta_template.cpp",
        "gen_embedding_forward_{}_codegen_meta.cpp",
        dense_options=[True, False],
        nobag_options=[False],  # nobag is not used
        vbe_options=[True, False],
    )

    # Generate the kernels for the forward splits
    generate_forward_embedding_cuda(
        "embedding_forward_split_kernel_template.cu",
        "gen_embedding_forward_{}_kernel.cu",
        dense_options=[True, False],
        nobag_options=[True, False],
        vbe_options=[True, False],
    )

    # Generate the kernels for the forward splits v2
    generate_forward_embedding_cuda(
        "embedding_forward_split_kernel_v2_template.cu",
        "gen_embedding_forward_{}_v2_kernel.cu",
        dense_options=[False],  # dense is not supported
        nobag_options=[False],  # nobag is not supported
        vbe_options=[False],  # vbe is not supported
    )

    # Generate the small kernels (for nobag only) for the forward splits
    template = env.get_template(
        "embedding_forward_split_kernel_nobag_small_template.cu"
    )
    for dense in [True, False]:
        wdesc = f"{ 'dense' if dense else 'split' }"
        filename = f"gen_embedding_forward_{wdesc}_unweighted_nobag_kernel_small.cu"
        write(filename, template.render(dense=dense, is_index_select=False))
        print(f"[Forward Split]: {filename}")

    # Generate PT2 forward wrapper cuda
    template_pt2 = env.get_template(
        "embedding_split_host_pt2_cuda_wrapper_template.cpp",
    )
    filename_pt2 = f"gen_embedding_forward_split_pt2_cuda_wrapper.cpp"
    write(
        filename_pt2,
        template_pt2.render(
            has_gpu_support=True,
            is_forward=True,
            has_vbe_support=True,
        ),
    )
    print(f"[Forward Split]: {filename_pt2}")

    # Generate PT2 forward wrapper cpu
    template_pt2 = env.get_template(
        "embedding_split_host_pt2_cpu_wrapper_template.cpp",
    )
    filename_pt2 = f"gen_embedding_forward_split_pt2_cpu_wrapper.cpp"
    write(
        filename_pt2,
        template_pt2.render(
            has_cpu_support=True,
            is_forward=True,
        ),
    )
    print(f"[Forward Split]: {filename_pt2}")


def backward_device_kernel() -> None:
    # Generate backward device kernels based on weighted (True/False), VBE
    # (True/False), no bag (True/False)
    template_filepath = "embedding_backward_split_device_kernel_template.cuh"
    generate_backward_embedding_cuda(
        template_filepath=template_filepath,
        optimizer="",
        filename_format="{}gen_embedding_backward_{}_split_device_kernel.cuh",
        kwargs={
            "has_gpu_support": True,
            "has_vbe_support": True,
            "dense": False,
            "gen_once": False,
        },
    )

    # Generate common backward device kernels (generate only once)
    template = env.get_template(template_filepath)
    write(
        "gen_embedding_backward_common_split_device_kernel.cuh",
        template.render(gen_once=True),
    )


# TODO: Separate this function into another codegen script
def index_select() -> None:
    kwargs = make_args([(FLOAT, "unused")])
    kwargs["args"] = kwargs["cuda"]
    for templ_file, gen_file in [
        (
            "embedding_forward_split_template.cu",
            "gen_batch_index_select_dim0_forward_codegen_cuda.cu",
        ),
        (
            "embedding_forward_split_kernel_template.cu",
            "gen_batch_index_select_dim0_forward_kernel.cu",
        ),
        (
            "embedding_forward_split_kernel_nobag_small_template.cu",
            "gen_batch_index_select_dim0_forward_kernel_small.cu",
        ),
        (
            "embedding_backward_split_template.cu",
            "gen_batch_index_select_dim0_backward_codegen_cuda.cu",
        ),
        (
            "embedding_backward_split_kernel_cta_template.cu",
            "gen_batch_index_select_dim0_backward_kernel_cta.cu",
        ),
        (
            "embedding_backward_split_kernel_warp_template.cu",
            "gen_batch_index_select_dim0_backward_kernel_warp.cu",
        ),
        (
            "embedding_backward_split_device_kernel_template.cuh",
            "gen_embedding_backward_batch_index_select_split_device_kernel.cuh",
        ),
    ]:
        template = env.get_template(templ_file)
        write(
            gen_file,
            template.render(
                weighted=False,
                dense=True,
                vbe=False,
                nobag=True,
                is_index_select=True,
                gen_once=False,
                kdesc="batch_index_select",
                **kwargs,
            ),
        )

    template = env.get_template("embedding_backward_split_grad_template.cu")
    write(
        "gen_embedding_backward_split_grad_index_select.cu",
        template.render(is_index_select=True),
    )

    # Generate common backward device kernels (generate only once)
    template = env.get_template("embedding_backward_split_device_kernel_template.cuh")
    write(
        "gen_embedding_backward_common_split_device_kernel.cuh",
        template.render(gen_once=True),
    )


def backward_grad() -> None:
    # Generate the common grad functions
    template = env.get_template("embedding_backward_split_grad_template.cu")
    write(
        "gen_embedding_backward_split_grad_embedding_ops.cu",
        template.render(is_index_select=False),
    )


def backward_indices() -> None:
    template = env.get_template("embedding_backward_split_indice_weights_template.cu")
    src_cu = template.render()
    write("gen_embedding_backward_split_indice_weights_codegen_cuda.cu", src_cu)
    src_cu = template.render(dense=True)
    write("gen_embedding_backward_dense_indice_weights_codegen_cuda.cu", src_cu)


def backward_dense() -> None:
    generate(
        optimizer="dense",
        dense=True,
        args=make_args(
            [
                (FLOAT, "unused"),
            ]
        ),
        has_cpu_support=True,
        has_gpu_support=True,
        has_vbe_support=False,
    )


def gen__init__py() -> None:
    template = env.get_template("__init__.template")
    src_py = template.render()
    write("__init__.py", src_py)


def emb_codegen(
    install_dir: Optional[str] = None, is_fbcode: Optional[bool] = None
) -> None:
    if install_dir is not None and len(install_dir) != 0:
        args.install_dir = install_dir
    if is_fbcode is not None:
        args.is_fbcode = is_fbcode
    backward_grad()

    # Generate forwards and specialized backwards
    backward_indices()
    backward_dense()
    # forward_quantized()
    forward_split()

    # Generate common device kernels for backwards
    backward_device_kernel()

    # Generate backwards and optimizers
    generate(**(adagrad()))
    generate(**(adam()))
    generate(**(lamb()))
    generate(**(lars_sgd()))
    generate(**(partial_rowwise_adam()))
    generate(**(partial_rowwise_lamb()))
    generate(**(rowwise_adagrad()))
    generate(**(approx_rowwise_adagrad()))
    generate(**(rowwise_adagrad_with_weight_decay()))
    generate(**(approx_rowwise_adagrad_with_weight_decay()))
    generate(**(rowwise_adagrad_with_counter()))
    generate(**(approx_rowwise_adagrad_with_counter()))
    generate(**(rowwise_weighted_adagrad()))
    generate(**(sgd()))
    generate(**(approx_sgd()))
    generate(**(none_optimizer()))

    # Generate index_select ops using TBE backend
    index_select()
    gen__init__py()


def main() -> None:
    emb_codegen()


if __name__ == "__main__":
    print(f"[EMBEDDING BACKWARD CODE GENERATOR] {sys.argv}")
    main()

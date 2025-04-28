#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import jinja2

try:
    from .scripts_argsparse import args
    from .torch_type_utils import TensorType
except:
    # pyre-ignore[21]
    from scripts_argsparse import args

    # pyre-ignore[21]
    from torch_type_utils import TensorType


################################################################################
# Instantiate Jinja Environment
################################################################################

if args.is_fbcode:
    # In fbcode, buck injects SRCDIR into the environment when executing a
    # custom_rule().  The templates will be visible there because they are
    # specified in the `srcs` field of the rule.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(os.path.abspath(os.environ["SRCDIR"]))
    )
else:
    # In OSS, because the generation script is held in `codegen/genscript`, we
    # explicitly point to the parent directory as the root directory of the
    # templates.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )


################################################################################
# Register Variables in Jinja Environment
################################################################################

# Upper Limit of "max_embedding_dim (max_D)":
# BT_block_size * sizeof(float) * 4 * kWarpSize * {{ kMaxVecsPerThread }}
# needs to be smaller than the allocated shared memory size (2/3 of 96 KB
# on V100 and 160 KB on A100.
# BT_block_size * 4 * 4 * 32 * (max_D // 128) <= 64 * 1024 (V100) or 96 * 1024 (A100)
# Since BT_block_size >= 1, max_D <= 16K (V100) or 24K (A100).
# Note that if we increase max_D, it will increase the compilation time significantly.
env.globals["max_embedding_dim"] = 2048

# Max embedding dimension for legacy embedding kernels. TBE v2 can support
# larger max embedding dimension.
env.globals["legacy_max_embedding_dim"] = 1024

# An optimization for ROCm
env.globals["items_per_warp"] = 128 if args.is_rocm is False else 256

# The fixed max vectors per thread for different kernels.  The numbers were
# derived from empirical studies
env.globals["fixed_max_vecs_per_thread"] = {"backward": 2, "backward_indice_weights": 6}

env.globals["dense"] = False
env.globals["is_rocm"] = args.is_rocm


################################################################################
# Helper functions in Jinja Environment
################################################################################


def prepare_string_for_formatting(blob: str, format_keywords: List[str]) -> str:
    """
    Replace curly brackets ('{' or '}') with escape characters ('{{' or '}}')
    to prepare the string to be formatted by `str.format()`. `str.format()`
    searches curly brackets to find keywords to format. It will run into an
    error if the string contains curly brackets.
    """
    blob = blob.replace("{", "{{").replace("}", "}}")
    for kw in format_keywords:
        blob = blob.replace("{{" + kw + "}}", "{" + kw + "}")
    return blob


def generate_optimized_grad_sum_loop_access(
    blob: str, other_formats: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate an optimized code for grad_sum accessing
    - The indices of `grad_sum` when `kUseVecBlocking` is true and false are
      different. When `kUseVecBlocking` is true, `d_vec` is the index.
      Otherwise, `vec` is the index.
    - When `kUseVecBlocking` is false, the number times that the for-loop is
      executed is known at compile time. Thus, we can add the `#pragma unroll`
      hint to tell the compiler to optimize the for-loop.
    """
    blob = prepare_string_for_formatting(blob, ["grad_vec"])

    smem_blob = blob.format(grad_vec="smem_grad_sum[d_vec]")
    reg_blob = blob.format(grad_vec="grad_sum[vec]")
    gen_blob = """
    if (kUseVecBlocking) {
        // max_vecs is not known at compile time
        for (int32_t vec = 0;
            vec < max_vecs &&
            (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
            {smem_blob}
        }
    }
    else {
        // kFixedMaxVecsPerThread is known at compile time
        #pragma unroll kFixedMaxVecsPerThread
        for (int32_t vec = 0;
            vec < kFixedMaxVecsPerThread
                && (kThreadGroupSize * vec + threadIdx.x) * VEC_WIDTH < D;
            ++vec) {
            const int32_t d_vec = vec * kThreadGroupSize + threadIdx.x;
            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
            {reg_blob}
        }
    }
    """
    gen_blob = prepare_string_for_formatting(gen_blob, ["smem_blob", "reg_blob"])
    gen_blob = gen_blob.format(smem_blob=smem_blob, reg_blob=reg_blob)
    if other_formats is not None:
        gen_blob = prepare_string_for_formatting(gen_blob, list(other_formats.keys()))
        gen_blob = gen_blob.format(**other_formats)
    return gen_blob


def get_max_vecs_template_configs(
    items_per_warp: int,
    fixed_max_vecs_per_thread: int,
    use_subwarp_shuffle: bool,
    use_vec_blocking: bool,
) -> List[Tuple[int, int, str]]:
    """
    Generate the template configs for each kFixedMaxVecsPerThread,
    kThreadGroupSize, and kUseVecBlocking
    """
    warp_size = items_per_warp // 4
    configs: List[Tuple[int, int, str]] = []

    if use_vec_blocking:
        # kFixedMaxVecsPerThread = fixed_max_vecs_per_thread
        # kThreadGroupSize = kWarpSize
        # kUseVecBlocking = true
        configs.append((fixed_max_vecs_per_thread, warp_size, "true"))

    # Generate the cases where an entire embedding row can fit in the
    # thread-local buffer (i.e., shared memory is not need for grad_sum)
    if use_subwarp_shuffle:
        # Generate configs for sub-warp templates
        group_size = 8  # Smallest group size that TBE supports
        while group_size < warp_size:
            # kFixedMaxVecsPerThread = 1
            # kThreadGroupSize = group_size
            # kUseVecBlocking = false
            configs.append((1, group_size, "false"))
            group_size *= 2

    # Generate configs for the full-warp templates
    for v in range(1, fixed_max_vecs_per_thread + 1):
        configs.append((v, warp_size, "false"))

    return configs


def dispatch_non_vec_blocking_kernel(
    items_per_warp: int,
    fixed_max_vecs_per_thread: int,
    use_subwarp_shuffle: bool,
) -> str:
    """
    Generate code for kernel dispatching for kernels that do not use vector
    blocking (i.e., an entire embedding row can fit in the allocated Vec4T
    buffer)
    """
    blob = ""
    for (
        kFixedMaxVecsPerThread,
        kThreadGroupSize,
        kUseVecBlocking,
    ) in get_max_vecs_template_configs(
        items_per_warp,
        fixed_max_vecs_per_thread,
        use_subwarp_shuffle,
        use_vec_blocking=False,
    ):
        formats = {
            "max_D_val": kFixedMaxVecsPerThread * kThreadGroupSize * 4,
            "kFixedMaxVecsPerThread": kFixedMaxVecsPerThread,
            "kThreadGroupSize": kThreadGroupSize,
            "kUseVecBlocking": kUseVecBlocking,
        }
        max_D_val = kFixedMaxVecsPerThread * kThreadGroupSize * 4
        d_blob = """if (MAX_D <= {max_D_val}) {                               \\
             [[ maybe_unused ]] const int max_vecs_per_thread =               \\
               {kFixedMaxVecsPerThread};                                      \\
             constexpr int kFixedMaxVecsPerThread = {kFixedMaxVecsPerThread}; \\
             [[ maybe_unused ]] constexpr int kThreadGroupSize =              \\
               {kThreadGroupSize};                                            \\
             [[ maybe_unused ]] constexpr bool kUseVecBlocking =              \\
               {kUseVecBlocking};                                             \\
             return __VA_ARGS__();                                            \\
           }                                                                  \\
        """
        d_blob = prepare_string_for_formatting(d_blob, list(formats.keys()))
        blob += d_blob.format(**formats)
    return blob


def dispatch_vec_blocking_kernel(
    items_per_warp: int,
    fixed_max_vecs_per_thread: int,
) -> str:
    """
    Generate code for kernel dispatching for kernels that use vector blocking
    (i.e., an entire embedding row cannot fit in the allocated Vec4T buffer)
    """
    formats = {
        "max_D_val": fixed_max_vecs_per_thread * items_per_warp,
        "items_per_warp": items_per_warp,
        "fixed_max_vecs_per_thread": fixed_max_vecs_per_thread,
    }
    blob = """if (MAX_D > {max_D_val}) {                                     \\
         [[ maybe_unused ]] const int max_vecs_per_thread =                  \\
           (MAX_D + {items_per_warp} - 1) / {items_per_warp};                \\
         constexpr int kFixedMaxVecsPerThread = {fixed_max_vecs_per_thread}; \\
         [[ maybe_unused ]] constexpr int kThreadGroupSize = kWarpSize;      \\
         [[ maybe_unused ]] constexpr bool kUseVecBlocking = true;           \\
         return __VA_ARGS__();                                               \\
       }                                                                     \\
    """
    blob = prepare_string_for_formatting(blob, list(formats.keys()))
    return blob.format(**formats)


def dispatch_optimal_kernel(
    items_per_warp: int,
    fixed_max_vecs_per_thread: int,
    use_subwarp_shuffle: bool,
) -> str:
    """
    Generate code for kernel dispatching for both kernels that use/do not use
    vector blocking
    """
    blob = dispatch_non_vec_blocking_kernel(
        items_per_warp,
        fixed_max_vecs_per_thread,
        use_subwarp_shuffle,
    )
    blob += dispatch_vec_blocking_kernel(
        items_per_warp,
        fixed_max_vecs_per_thread,
    )
    return blob


def is_valid_forward_config(
    nobag: bool,
    weighted: bool,
    vbe: bool,
    is_index_select: bool,
) -> bool:
    """
    Check if the given combination of configs is valid for forward
    - nobag does not have weighted or vbe supports
    - is_index_select is nobag
    """
    return (not nobag or (not weighted and not vbe)) and (
        nobag or (not is_index_select)
    )


def has_experimental_support(
    dense: bool, nobag: bool, vbe: bool, is_index_select: bool, ssd: bool
) -> bool:
    """
    Check if the given combination of configs has TBE v2 support
    - TBE v2 does not support dense, nobag, vbe, is_index_select, is_rocm, and ssd
    """
    return not dense and not nobag and not vbe and not is_index_select and not ssd


def is_valid_gwd_config(
    dense: bool,
    nobag: bool,
    vbe: bool,
    is_index_select: bool,
    has_global_weight_decay_support: bool,
    ssd: bool,
) -> bool:
    """
    Check if the given combination of configs is valid for global weight decay support
    - `has_global_weight_decay_support` is whether global weight decay is available for
    an optimizer, but not all configs of such optimizer offer global weight decay support
    - any updates to the configs need to be reflected in embedding_backward_split_host_template.cpp
    - global weight decay does not support dense, nobag, vbe, index_select
    """
    return (
        not dense
        and not nobag
        and not is_index_select
        and has_global_weight_decay_support
        and not ssd
    )


def compute_global_weight_decay(is_global_weight_decay_kernel: bool) -> str:
    """
    For global weight decay kernel, compute the global weight decay value
    and update prev_iter to be current iteration
    This is to used in both warp and cta kernels.
    """
    if is_global_weight_decay_kernel:
        return """
        const auto prev_iter = prev_iter_dev[linear_index];
        const auto global_weight_decay = prev_iter == 0 ? 1 : max(gwd_lower_bound, powf(weight_decay_base, max(iter - prev_iter - 1, 0.0f)));
        if (threadIdx.x == 0) {
            prev_iter_dev[linear_index] = iter;
        }
        """
    else:
        return ""


################################################################################
# Register Helper Functions in Jinja Environment
################################################################################

env.globals["generate_optimized_grad_sum_loop_access"] = (
    generate_optimized_grad_sum_loop_access
)
env.globals["get_max_vecs_template_configs"] = get_max_vecs_template_configs
env.globals["dispatch_optimal_kernel"] = dispatch_optimal_kernel
env.globals["dispatch_non_vec_blocking_kernel"] = dispatch_non_vec_blocking_kernel
env.globals["dispatch_vec_blocking_kernel"] = dispatch_vec_blocking_kernel
env.globals["is_valid_forward_config"] = is_valid_forward_config
env.globals["has_experimental_support"] = has_experimental_support
env.globals["is_valid_gwd_config"] = is_valid_gwd_config
env.globals["compute_global_weight_decay"] = compute_global_weight_decay

################################################################################
# Filter functions in Jinja Environment
################################################################################


# Format the macro call to generate pta::PackedTensorAccessors
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
                match = re.search("at::acc_type<([a-zA-Z_0-9]*), true>", args)
                assert match is not None and len(match.groups()) == 1
                new_type = match.group(1)
                args = re.sub("at::acc_type<[a-zA-Z_]*, true>", new_type, args)
                macro_name = "MAKE_PTA_ACC_WITH_NAME"
            else:
                macro_name = "MAKE_PTA_WITH_NAME"
            args = args.replace(", at::RestrictPtrTraits", "")
            new_str_list.append(
                f"{macro_name}({func_name}, {tensor}, {args}, {acc_nbits})"
            )
        else:
            new_str_list.append(pta_str)
    return new_str_list


def replace_pta_namespace(pta_str_list: List[str]) -> List[str]:
    return [
        pta_str.replace("at::PackedTensorAccessor", "pta::PackedTensorAccessor")
        for pta_str in pta_str_list
    ]


def replace_placeholder_types(
    # pyre-fixme[11]: Annotation `TensorType` is not defined as a type.
    arg_str_list: List[str],
    # pyre-fixme[11]: Annotation `TensorType` is not defined as a type.
    type_combo: Optional[Dict[str, TensorType]],
) -> List[str]:
    """
    Replace the placeholder types with the primitive types
    """
    new_str_list = []
    for arg_str in arg_str_list:
        if type_combo is not None:
            for ph_name, ph_ty in type_combo.items():
                str_ty = ph_name + "_ph_t"
                if str_ty in arg_str:
                    arg_str = arg_str.replace(str_ty, ph_ty.primitive_type)
                    break
        new_str_list.append(arg_str)
    return new_str_list


def to_upper_placeholder_types(arg_str_list: List[str]) -> List[str]:
    """
    Make the placeholder type names upper cases
    """
    new_str_list = []
    for arg_str in arg_str_list:
        new_str_list.append(arg_str.upper() + "_T")
    return new_str_list


################################################################################
# Register Filter Functions in Jinja Environment
################################################################################

env.filters["make_pta_acc_format"] = make_pta_acc_format
env.filters["replace_pta_namespace"] = replace_pta_namespace
env.filters["replace_placeholder_types"] = replace_placeholder_types
env.filters["to_upper_placeholder_types"] = to_upper_placeholder_types

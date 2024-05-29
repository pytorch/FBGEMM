#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[29]
# flake8: noqa F401


import argparse
import itertools
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jinja2

try:
    from .torch_type_utils import arg_type_to_tensor_type, ArgType, TensorType

except ImportError:
    # pyre-ignore[21]
    from torch_type_utils import arg_type_to_tensor_type, ArgType, TensorType


######################################################################
## Helper functions for the code generator script                   ##
######################################################################


def _arg_constructor(
    type: str, name: str, gpu: bool = True, precision: int = 32
) -> str:
    return (
        f"{name}.packed_accessor{precision}<{type}, 1, at::RestrictPtrTraits>()"
        if gpu
        else f"{name}.accessor<{type}, 1>()"
    )


def _arg(
    type: str,
    name: str,
    gpu: bool = True,
    precision: int = 32,
    pass_by_ref: bool = False,
) -> str:
    ref = "&" if pass_by_ref else ""
    return (
        f"at::PackedTensorAccessor{precision}<{type}, 1, at::RestrictPtrTraits>{ref} {name}"
        if gpu
        else f"at::TensorAccessor<{type}, 1>{ref} {name}"
    )


def demangle_name(name: str) -> str:
    return name.replace("_dev", "").replace("_uvm", "")


def acc_cache_tensor_arg_constructor(name: str, gpu: bool = True) -> str:
    return _arg_constructor(
        "at::acc_type<" + ("cache_t" if gpu else "scalar_t") + ", true>",
        name,
        gpu=gpu,
        precision=64,
    )


def acc_placeholder_tensor_arg_constructor(name: str, gpu: bool = True) -> str:
    return _arg_constructor(
        demangle_name(name) + "_ph_t",
        name,
        gpu=gpu,
        precision=64,
    )


def acc_cache_tensor_arg(name: str, gpu: bool = True, pass_by_ref: bool = False) -> str:
    return _arg(
        "at::acc_type<" + ("cache_t" if gpu else "scalar_t") + ", true>",
        name,
        gpu=gpu,
        precision=64,
        pass_by_ref=pass_by_ref,
    )


def acc_placeholder_tensor_arg(
    name: str, gpu: bool = True, pass_by_ref: bool = False
) -> str:
    return _arg(
        demangle_name(name) + "_ph_t",
        name,
        gpu=gpu,
        precision=64,
        pass_by_ref=pass_by_ref,
    )


def long_tensor_arg_constructor(name: str, gpu: bool = True) -> str:
    return _arg_constructor("int64_t", name, gpu=gpu)


def long_tensor_arg(name: str, gpu: bool = True, pass_by_ref: bool = False) -> str:
    return _arg("int64_t", name, gpu=gpu, pass_by_ref=pass_by_ref)


def int_tensor_arg_constructor(name: str, gpu: bool = True) -> str:
    return _arg_constructor("int32_t", name, gpu=gpu)


def int_tensor_arg(name: str, gpu: bool = True, pass_by_ref: bool = False) -> str:
    return _arg("int32_t", name, gpu=gpu, pass_by_ref=pass_by_ref)


def tensor_arg(name: str) -> str:
    return f"Tensor {name}"


def double_arg(name: str, default: float = 0.0) -> str:
    return f"double {name} = {default}"


def double_arg_no_default(name: str) -> str:
    return f"double {name}"


def float_arg(name: str, default: float = 0.0) -> str:
    return f"float {name} = {default}"


def float_arg_no_default(name: str) -> str:
    return f"float {name}"


def int64_arg(name: str, default: int = 0) -> str:
    return f"int64_t {name} = {default}"


def int64_arg_no_default(name: str) -> str:
    return f"int64_t {name}"


def int_arg(name: str, default: int = 0) -> str:
    return f"int {name} = {default}"


def sym_int_arg(name: str, default: int = 0) -> str:
    return f"c10::SymInt {name} = {default}"


def sym_int_arg_no_default(name: str) -> str:
    return f"c10::SymInt {name}"


def schema_sym_int_arg(name: str, default: int = 0) -> str:
    return f"SymInt {name} = {default}"


def schema_sym_int_arg_no_default(name: str) -> str:
    return f"SymInt {name}"


def make_kernel_arg(
    ty: ArgType, name: str, default: Union[int, float, None], pass_by_ref: bool = False
) -> str:
    return {
        ArgType.TENSOR: lambda x: acc_cache_tensor_arg(x, pass_by_ref=pass_by_ref),
        ArgType.INT_TENSOR: lambda x: int_tensor_arg(x, pass_by_ref=pass_by_ref),
        ArgType.LONG_TENSOR: lambda x: long_tensor_arg(x, pass_by_ref=pass_by_ref),
        ArgType.PLACEHOLDER_TENSOR: lambda x: acc_placeholder_tensor_arg(
            x, pass_by_ref=pass_by_ref
        ),
        ArgType.INT: (
            (lambda x: int64_arg(x, default=int(default)))
            if default is not None
            else int64_arg_no_default
        ),
        ArgType.SYM_INT: (
            (lambda x: sym_int_arg(x, default=int(default)))
            if default is not None
            else sym_int_arg_no_default
        ),
        ArgType.FLOAT: (
            (lambda x: float_arg(x, default=default))
            if default is not None
            else float_arg_no_default
        ),
    }[ty](name)


def make_kernel_arg_constructor(ty: ArgType, name: str) -> str:
    return {
        ArgType.TENSOR: acc_cache_tensor_arg_constructor,
        ArgType.INT_TENSOR: int_tensor_arg_constructor,
        ArgType.LONG_TENSOR: long_tensor_arg_constructor,
        ArgType.PLACEHOLDER_TENSOR: acc_placeholder_tensor_arg_constructor,
        ArgType.INT: lambda x: x,
        ArgType.FLOAT: lambda x: x,
        ArgType.SYM_INT: lambda x: x,
    }[ty](name)


def make_cpu_kernel_arg(ty: ArgType, name: str, default: Union[int, float]) -> str:
    return {
        ArgType.TENSOR: lambda x: acc_cache_tensor_arg(x, gpu=False),
        ArgType.INT_TENSOR: lambda x: int_tensor_arg(x, gpu=False),
        ArgType.LONG_TENSOR: lambda x: long_tensor_arg(x, gpu=False),
        ArgType.PLACEHOLDER_TENSOR: acc_cache_tensor_arg_constructor,
        ArgType.INT: lambda x: int64_arg(x, default=int(default)),
        ArgType.FLOAT: lambda x: float_arg(x, default=default),
        ArgType.SYM_INT: lambda x: sym_int_arg(x, default=int(default)),
    }[ty](name)


def make_cpu_kernel_arg_constructor(ty: ArgType, name: str) -> str:
    return {
        ArgType.TENSOR: lambda x: acc_cache_tensor_arg_constructor(x, gpu=False),
        ArgType.INT_TENSOR: lambda x: int_tensor_arg_constructor(x, gpu=False),
        ArgType.LONG_TENSOR: lambda x: long_tensor_arg_constructor(x, gpu=False),
        ArgType.PLACEHOLDER_TENSOR: lambda x: acc_cache_tensor_arg_constructor(
            x, gpu=False
        ),
        ArgType.INT: lambda x: x,
        ArgType.FLOAT: lambda x: x,
        ArgType.SYM_INT: lambda x: x,
    }[ty](name)


def make_function_arg(
    ty: ArgType, name: str, default: Optional[Union[int, float]]
) -> str:
    return {
        ArgType.TENSOR: tensor_arg,
        ArgType.INT_TENSOR: tensor_arg,
        ArgType.LONG_TENSOR: tensor_arg,
        ArgType.PLACEHOLDER_TENSOR: tensor_arg,
        ArgType.INT: (
            (lambda x: int64_arg(x, default=int(default)))
            if default is not None
            else int64_arg_no_default
        ),
        ArgType.FLOAT: (
            (lambda x: double_arg(x, default=default))
            if default is not None
            else double_arg_no_default
        ),
        ArgType.SYM_INT: (
            (lambda x: sym_int_arg(x, default=int(default)))
            if default is not None
            else sym_int_arg_no_default
        ),
    }[ty](name)


def make_function_schema_arg(ty: ArgType, name: str, default: Union[int, float]) -> str:
    return {
        ArgType.TENSOR: tensor_arg,
        ArgType.INT_TENSOR: tensor_arg,
        ArgType.LONG_TENSOR: tensor_arg,
        ArgType.PLACEHOLDER_TENSOR: tensor_arg,
        ArgType.INT: lambda x: int_arg(x, default=int(default)),
        ArgType.FLOAT: lambda x: float_arg(x, default=default),
        ArgType.SYM_INT: lambda x: schema_sym_int_arg(x, default=default),
    }[ty](name)


def make_ivalue_cast(ty: ArgType) -> str:
    return {
        ArgType.INT: "toInt",
        ArgType.FLOAT: "toDouble",
        ArgType.SYM_INT: "toSymInt",
    }[ty]


######################################################################
# Optimizer Args Set Item
######################################################################


@dataclass
class OptimizerArgsSetItem:
    ty: ArgType  # type
    name: str
    default: Union[float, ArgType] = 0  # DEFAULT_ARG_VAL
    ph_tys: Optional[List[ArgType]] = None  # placeholder types


# Alias b/c the name is too long
OptimItem = OptimizerArgsSetItem


######################################################################
# Optimizer Args
######################################################################


@dataclass
class OptimizerArgs:
    split_kernel_args: List[str]
    split_kernel_args_no_defaults: List[str]
    split_kernel_arg_constructors: List[str]
    split_cpu_kernel_args: List[str]
    split_cpu_kernel_arg_constructors: List[str]
    split_function_args: List[str]
    split_function_args_no_defaults: List[str]
    split_saved_tensors: List[str]
    split_tensors: List[str]
    split_tensor_types: Dict[str, str]
    saved_data: List[Tuple[str, str]]
    split_function_arg_names: List[str]
    split_function_schemas: List[str]
    split_variables: List[str]
    split_ref_kernel_args: List[str]
    placeholder_tensor_names: List[str]
    placeholder_type_combos: Union[List[Dict[str, TensorType]], List[None]]

    @staticmethod
    # pyre-ignore[3]
    def create(
        split_arg_spec: List[OptimItem],
        arg_spec: List[OptimItem],
    ):
        # Compute placeholder tensor combinations
        ph_tensor_names = [
            s.name for s in arg_spec if s.ty == ArgType.PLACEHOLDER_TENSOR
        ]
        ph_tensor_types = [
            # pyre-ignore[16]
            [arg_type_to_tensor_type[t] for t in s.ph_tys]
            for s in arg_spec
            if s.ty == ArgType.PLACEHOLDER_TENSOR
        ]
        if len(ph_tensor_names) > 0:
            ph_combos_list = itertools.product(*ph_tensor_types)
            ph_combos = [
                {k: ph for k, ph in zip(ph_tensor_names, combo)}
                for combo in ph_combos_list
            ]
        else:
            ph_combos = [None]

        return OptimizerArgs(
            # GPU kernel args
            split_kernel_args=[
                make_kernel_arg(s.ty, s.name, s.default) for s in split_arg_spec
            ],
            split_kernel_args_no_defaults=[
                make_kernel_arg(s.ty, s.name, None) for s in split_arg_spec
            ],
            split_kernel_arg_constructors=[
                make_kernel_arg_constructor(s.ty, s.name) for s in split_arg_spec
            ],
            # CPU kernel args
            split_cpu_kernel_args=[
                make_cpu_kernel_arg(s.ty, s.name, s.default) for s in split_arg_spec
            ],
            split_cpu_kernel_arg_constructors=[
                make_cpu_kernel_arg_constructor(s.ty, s.name) for s in split_arg_spec
            ],
            # Function args
            split_function_args=[
                make_function_arg(s.ty, s.name, s.default) for s in split_arg_spec
            ],
            split_function_args_no_defaults=[
                make_function_arg(s.ty, s.name, None) for s in split_arg_spec
            ],
            # Helper values
            split_tensors=[
                s.name
                for s in arg_spec
                if s.ty in (ArgType.TENSOR, ArgType.PLACEHOLDER_TENSOR)
            ],
            split_tensor_types={
                s.name: (
                    "at::acc_type<cache_t, true>"
                    if s.ty == ArgType.TENSOR
                    else (s.name + "_ph_t")
                )
                for s in arg_spec
                if s.ty in (ArgType.TENSOR, ArgType.PLACEHOLDER_TENSOR)
            },
            split_saved_tensors=[
                s.name
                for s in split_arg_spec
                if s.ty
                in (
                    ArgType.TENSOR,
                    ArgType.INT_TENSOR,
                    ArgType.LONG_TENSOR,
                    ArgType.PLACEHOLDER_TENSOR,
                )
            ],
            saved_data=[
                (s.name, make_ivalue_cast(s.ty))
                for s in arg_spec
                if s.ty not in (ArgType.TENSOR, ArgType.PLACEHOLDER_TENSOR)
            ],
            split_function_arg_names=[s.name for s in split_arg_spec],
            split_function_schemas=[
                make_function_schema_arg(s.ty, s.name, s.default)
                for s in split_arg_spec
            ],
            split_variables=["Variable()" for _ in split_arg_spec],
            split_ref_kernel_args=[
                make_kernel_arg(s.ty, s.name, s.default, pass_by_ref=True)
                for s in split_arg_spec
            ],
            placeholder_tensor_names=ph_tensor_names,
            placeholder_type_combos=ph_combos,
        )


######################################################################
# Optimizer Args Set
######################################################################


@dataclass
class OptimizerArgsSet:
    cpu: OptimizerArgs
    cuda: OptimizerArgs
    any: OptimizerArgs

    @staticmethod
    def create_optim_args(
        arg_spec: List[OptimItem], ext_fn: Callable[[OptimItem], List[OptimItem]]
    ) -> OptimizerArgs:
        split_arg_spec = []
        for s in arg_spec:
            if s.ty in (ArgType.FLOAT, ArgType.INT, ArgType.SYM_INT):
                split_arg_spec.append(OptimItem(s.ty, s.name, s.default))
            else:
                assert s.ty in (ArgType.TENSOR, ArgType.PLACEHOLDER_TENSOR)
                # Treat PLACEHOLDER_TENSOR as TENSOR for CPU
                split_arg_spec.extend(ext_fn(s))
        return OptimizerArgs.create(split_arg_spec, arg_spec)

    @staticmethod
    def extend_for_cpu(spec: OptimItem) -> List[OptimItem]:
        name = spec.name
        default = spec.default
        return [
            OptimItem(ArgType.TENSOR, f"{name}_host", default),
            OptimItem(ArgType.INT_TENSOR, f"{name}_placements", default),
            OptimItem(ArgType.LONG_TENSOR, f"{name}_offsets", default),
        ]

    @staticmethod
    def extend_for_cuda(spec: OptimItem) -> List[OptimItem]:
        name = spec.name
        default = spec.default
        ty = spec.ty
        ph_tys = spec.ph_tys
        return [
            OptimItem(ty, f"{name}_dev", default, ph_tys),
            OptimItem(ty, f"{name}_uvm", default, ph_tys),
            OptimItem(ArgType.INT_TENSOR, f"{name}_placements", default),
            OptimItem(ArgType.LONG_TENSOR, f"{name}_offsets", default),
        ]

    @staticmethod
    def extend_for_any(spec: OptimItem) -> List[OptimItem]:
        name = spec.name
        default = spec.default
        ty = spec.ty
        ph_tys = spec.ph_tys
        return [
            OptimItem(ArgType.TENSOR, f"{name}_host", default),
            OptimItem(ty, f"{name}_dev", default, ph_tys),
            OptimItem(ty, f"{name}_uvm", default, ph_tys),
            OptimItem(ArgType.INT_TENSOR, f"{name}_placements", default),
            OptimItem(ArgType.LONG_TENSOR, f"{name}_offsets", default),
        ]

    @staticmethod
    # pyre-ignore[3]
    def create(arg_spec: List[OptimItem]):
        return OptimizerArgsSet(
            *(
                OptimizerArgsSet.create_optim_args(arg_spec, ext_fn)
                for ext_fn in (
                    OptimizerArgsSet.extend_for_cpu,
                    OptimizerArgsSet.extend_for_cuda,
                    OptimizerArgsSet.extend_for_any,
                )
            )
        )

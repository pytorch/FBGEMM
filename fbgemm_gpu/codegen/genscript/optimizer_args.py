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
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jinja2

# pyre-ignore[5]
TENSOR, INT_TENSOR, LONG_TENSOR, INT, FLOAT = range(5)


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


def acc_cache_tensor_arg_constructor(name: str, gpu: bool = True) -> str:
    return _arg_constructor(
        "at::acc_type<" + ("cache_t" if gpu else "scalar_t") + ", true>",
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


def make_kernel_arg(
    ty: int, name: str, default: Union[int, float, None], pass_by_ref: bool = False
) -> str:
    return {
        TENSOR: lambda x: acc_cache_tensor_arg(x, pass_by_ref=pass_by_ref),
        INT_TENSOR: lambda x: int_tensor_arg(x, pass_by_ref=pass_by_ref),
        LONG_TENSOR: lambda x: long_tensor_arg(x, pass_by_ref=pass_by_ref),
        INT: (
            (lambda x: int64_arg(x, default=int(default)))
            if default is not None
            else int64_arg_no_default
        ),
        FLOAT: (
            (lambda x: float_arg(x, default=default))
            if default is not None
            else float_arg_no_default
        ),
    }[ty](name)


def make_kernel_arg_constructor(ty: int, name: str) -> str:
    return {
        TENSOR: acc_cache_tensor_arg_constructor,
        INT_TENSOR: int_tensor_arg_constructor,
        LONG_TENSOR: long_tensor_arg_constructor,
        INT: lambda x: x,
        FLOAT: lambda x: x,
    }[ty](name)


def make_cpu_kernel_arg(ty: int, name: str, default: Union[int, float]) -> str:
    return {
        TENSOR: lambda x: acc_cache_tensor_arg(x, gpu=False),
        INT_TENSOR: lambda x: int_tensor_arg(x, gpu=False),
        LONG_TENSOR: lambda x: long_tensor_arg(x, gpu=False),
        INT: lambda x: int64_arg(x, default=int(default)),
        FLOAT: lambda x: float_arg(x, default=default),
    }[ty](name)


def make_cpu_kernel_arg_constructor(ty: int, name: str) -> str:
    return {
        TENSOR: lambda x: acc_cache_tensor_arg_constructor(x, gpu=False),
        INT_TENSOR: lambda x: int_tensor_arg_constructor(x, gpu=False),
        LONG_TENSOR: lambda x: long_tensor_arg_constructor(x, gpu=False),
        INT: lambda x: x,
        FLOAT: lambda x: x,
    }[ty](name)


def make_function_arg(ty: int, name: str, default: Optional[Union[int, float]]) -> str:
    return {
        TENSOR: tensor_arg,
        INT_TENSOR: tensor_arg,
        LONG_TENSOR: tensor_arg,
        INT: (
            (lambda x: int64_arg(x, default=int(default)))
            if default is not None
            else int64_arg_no_default
        ),
        FLOAT: (
            (lambda x: double_arg(x, default=default))
            if default is not None
            else double_arg_no_default
        ),
    }[ty](name)


def make_function_schema_arg(ty: int, name: str, default: Union[int, float]) -> str:
    return {
        TENSOR: tensor_arg,
        INT_TENSOR: tensor_arg,
        LONG_TENSOR: tensor_arg,
        INT: lambda x: int_arg(x, default=int(default)),
        FLOAT: lambda x: float_arg(x, default=default),
    }[ty](name)


def make_ivalue_cast(ty: int) -> str:
    return {INT: "toInt", FLOAT: "toDouble"}[ty]


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
    saved_data: List[Tuple[str, str]]
    split_function_arg_names: List[str]
    split_function_schemas: List[str]
    split_variables: List[str]
    split_ref_kernel_args: List[str]

    @staticmethod
    # pyre-ignore[3]
    def create(
        split_arg_spec: List[Tuple[int, str, Union[int, float]]],
        augmented_arg_spec: List[Tuple[int, str, Union[int, float]]],
    ):
        return OptimizerArgs(
            split_kernel_args=[
                make_kernel_arg(ty, name, default)
                for (ty, name, default) in split_arg_spec
            ],
            split_kernel_args_no_defaults=[
                make_kernel_arg(ty, name, None) for (ty, name, _) in split_arg_spec
            ],
            split_kernel_arg_constructors=[
                make_kernel_arg_constructor(ty, name)
                for (ty, name, default) in split_arg_spec
            ],
            split_cpu_kernel_args=[
                make_cpu_kernel_arg(ty, name, default)
                for (ty, name, default) in split_arg_spec
            ],
            split_cpu_kernel_arg_constructors=[
                make_cpu_kernel_arg_constructor(ty, name)
                for (ty, name, default) in split_arg_spec
            ],
            split_function_args=[
                make_function_arg(ty, name, default)
                for (ty, name, default) in split_arg_spec
            ],
            split_function_args_no_defaults=[
                make_function_arg(ty, name, None)
                for (ty, name, default) in split_arg_spec
            ],
            split_tensors=[
                name for (ty, name, default) in augmented_arg_spec if ty == TENSOR
            ],
            split_saved_tensors=[
                name
                for (ty, name, default) in split_arg_spec
                if ty in (TENSOR, INT_TENSOR, LONG_TENSOR)
            ],
            saved_data=[
                (name, make_ivalue_cast(ty))
                for (ty, name, default) in augmented_arg_spec
                if ty != TENSOR
            ],
            split_function_arg_names=[name for (ty, name, default) in split_arg_spec],
            split_function_schemas=[
                make_function_schema_arg(ty, name, default)
                for (ty, name, default) in split_arg_spec
            ],
            split_variables=["Variable()" for _ in split_arg_spec],
            split_ref_kernel_args=[
                make_kernel_arg(ty, name, default, pass_by_ref=True)
                for (ty, name, default) in split_arg_spec
            ],
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
    def create_for_cpu(
        augmented_arg_spec: List[Tuple[int, str, Union[float, int]]]
    ) -> OptimizerArgs:
        split_arg_spec = []
        for ty, arg, default in augmented_arg_spec:
            if ty in (FLOAT, INT):
                split_arg_spec.append((ty, arg, default))
            else:
                assert ty == TENSOR
                split_arg_spec.extend(
                    [
                        (TENSOR, f"{arg}_host", default),
                        (INT_TENSOR, f"{arg}_placements", default),
                        (LONG_TENSOR, f"{arg}_offsets", default),
                    ]
                )
        return OptimizerArgs.create(split_arg_spec, augmented_arg_spec)

    @staticmethod
    def create_for_cuda(
        augmented_arg_spec: List[Tuple[int, str, Union[float, int]]]
    ) -> OptimizerArgs:
        split_arg_spec = []
        for ty, arg, default in augmented_arg_spec:
            if ty in (FLOAT, INT):
                split_arg_spec.append((ty, arg, default))
            else:
                assert ty == TENSOR
                split_arg_spec.extend(
                    [
                        (TENSOR, f"{arg}_dev", default),
                        (TENSOR, f"{arg}_uvm", default),
                        (INT_TENSOR, f"{arg}_placements", default),
                        (LONG_TENSOR, f"{arg}_offsets", default),
                    ]
                )
        return OptimizerArgs.create(split_arg_spec, augmented_arg_spec)

    @staticmethod
    def create_for_any(
        augmented_arg_spec: List[Tuple[int, str, Union[float, int]]]
    ) -> OptimizerArgs:
        split_arg_spec = []
        for ty, arg, default in augmented_arg_spec:
            if ty in (FLOAT, INT):
                split_arg_spec.append((ty, arg, default))
            else:
                assert ty == TENSOR
                split_arg_spec.extend(
                    [
                        (TENSOR, f"{arg}_host", default),
                        (TENSOR, f"{arg}_dev", default),
                        (TENSOR, f"{arg}_uvm", default),
                        (INT_TENSOR, f"{arg}_placements", default),
                        (LONG_TENSOR, f"{arg}_offsets", default),
                    ]
                )
        return OptimizerArgs.create(split_arg_spec, augmented_arg_spec)

    @staticmethod
    # pyre-ignore[3]
    def create(
        arg_spec: List[Union[Tuple[int, str], Tuple[int, str, Union[float, int]]]]
    ):
        DEFAULT_ARG_VAL = 0
        # pyre-ignore[9]
        augmented_arg_spec: List[Tuple[int, str, Union[float, int]]] = [
            item if len(item) == 3 else (*item, DEFAULT_ARG_VAL) for item in arg_spec
        ]

        return OptimizerArgsSet(
            OptimizerArgsSet.create_for_cpu(augmented_arg_spec),
            OptimizerArgsSet.create_for_cuda(augmented_arg_spec),
            OptimizerArgsSet.create_for_any(augmented_arg_spec),
        )

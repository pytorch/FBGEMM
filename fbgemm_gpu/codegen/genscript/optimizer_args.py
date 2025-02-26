#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[29]
# pyre-ignore-all-errors[53]
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
# Optimizer Args Set Item
######################################################################


@dataclass
class OptimizerArgsSetItem:
    # pyre-fixme[11]: Annotation `ArgType` is not defined as a type.
    ty: ArgType  # type
    name: str
    default: Union[float, ArgType] = 0  # DEFAULT_ARG_VAL
    ph_tys: Optional[List[ArgType]] = None  # placeholder types
    is_optional: bool = False  # optional variable


# Alias b/c the name is too long
OptimItem = OptimizerArgsSetItem


######################################################################
## Data Dict for the code generator script                   ##
######################################################################
# a dict of tensor name and annotation to mark whether the tensor is mutable.
# this is use to annotate the tensor in the defintion schema.
annotation_dict: Dict[str, str] = {
    "weights": "(a!)",
    "weights_host": "(a!)",
    "weights_dev": "(b!)",
    "weights_uvm": "(c!)",
    "weights_lxu_cache": "(d!)",
    "aux_tensor": "(e!)",
    "uvm_cache_stats": "(f!)",
    "momentum1": "(g!)",
    "momentum1_host": "(g!)",
    "momentum1_dev": "(h!)",
    "momentum1_uvm": "(i!)",
    "momentum2": "(j!)",
    "momentum2_host": "(j!)",
    "momentum2_dev": "(k!)",
    "momentum2_uvm": "(l!)",
    "prev_iter": "(m!)",
    "prev_iter_host": "(m!)",
    "prev_iter_dev": "(n!)",
    "prev_iter_uvm": "(o!)",
    "row_counter": "(p!)",
    "row_counter_host": "(p!)",
    "row_counter_dev": "(q!)",
    "row_counter_uvm": "(r!)",
    "optim_tensor": "(s!)",
    "delta_weights_host": "(t!)",
    "delta_weights_dev": "(u!)",
    "delta_weights_uvm": "(v!)",
}

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


def tensor_list_arg_no_default(name: str, pass_by_ref: bool) -> str:
    ref = "&" if pass_by_ref else ""
    return f"at::TensorList{ref} {name}"


def tensor_arg(name: str) -> str:
    return f"Tensor {name}"


def tensor_arg_annotate(name: str) -> str:
    annotate = annotation_dict[name] if name in annotation_dict else ""
    return f"Tensor{annotate} {name}"


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


def schema_tensor_list_arg_no_default(name: str) -> str:
    annotate = annotation_dict[name] if name in annotation_dict else ""
    return f"Tensor[]{annotate} {name}"


def bool_arg(name: str, default: bool = False) -> str:
    return f"bool {name} = {'true' if default else 'false'}"


def bool_arg_no_default(name: str) -> str:
    return f"bool {name}"


def schema_bool_arg(name: str, default: bool = False) -> str:
    return f"bool {name} = {default}"


def list_arg(ty: str) -> str:
    """
    Returns a C++ argument for a list of optimizer arguments the given type.

    Parameters:
        ty (str) - type of the list e.g., "int", "float", "tensor"
    Returns:
        C++ arguemnt for a list of the given type e.g., for a list of int returns "std::vector<int> optim_int",
    """
    return {
        "tensor": "std::vector<std::optional<at::Tensor>> optim_tensor",
        "int": "std::vector<int64_t> optim_int",
        "float": "std::vector<double> optim_float",
        "bool": "c10::List<bool> optim_bool",
    }[ty]


def schema_list_arg(ty: str) -> str:
    """
    Returns a C++ schema for a list of optimizer arguments the given type.

    Parameters:
        ty (str) - type of the list e.g., "int", "float", "tensor"
    Returns:
        C++ arguemnt for a list of the given type e.g., for a list of int returns "int[] optim_int",
    """
    return {
        "tensor": "Tensor?[] optim_tensor",
        "int": "int[] optim_int",
        "float": "float[] optim_float",
        "bool": "bool[] optim_bool",
    }[ty]


def optional_tensor_arg(name: str) -> str:
    return f"std::optional<Tensor> {name} = std::nullopt"


def optional_tensor_arg_no_default(name: str) -> str:
    return f"std::optional<Tensor> {name}"


def schema_optional_tensor_arg(name: str) -> str:
    return f"Tensor? {name} = None"


def optional_tensorlist_arg(name: str) -> str:
    return f"std::optional<at::TensorList> {name} = std::nullopt"


def optional_tensorlist_arg_no_default(name: str) -> str:
    return f"std::optional<at::TensorList> {name}"


def schema_optional_tensorlist_arg(name: str) -> str:
    return f"Tensor[]? {name} = None"


def make_kernel_arg(
    ty: ArgType,
    name: str,
    default: Union[int, float, None],
    pass_by_ref: bool = False,
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
        ArgType.BOOL: (
            (lambda x: bool_arg(x, default=bool(default)))
            if default is not None
            else bool_arg_no_default
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
        ArgType.BOOL: lambda x: x,
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
        ArgType.BOOL: lambda x: bool_arg(x, default=bool(default)),
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
        ArgType.BOOL: lambda x: x,
    }[ty](name)


def make_function_arg(
    ty: ArgType,
    name: str,
    default: Optional[Union[int, float]],
    is_optional: bool = False,
) -> str:
    return {
        ArgType.TENSOR: (
            (lambda x: tensor_arg(x))
            if not is_optional
            else (
                optional_tensor_arg
                if default is not None
                else optional_tensor_arg_no_default
            )
        ),
        ArgType.INT_TENSOR: (
            (lambda x: tensor_arg(x))
            if not is_optional
            else (
                optional_tensor_arg
                if default is not None
                else optional_tensor_arg_no_default
            )
        ),
        ArgType.LONG_TENSOR: (
            (lambda x: tensor_arg(x))
            if not is_optional
            else (
                optional_tensor_arg
                if default is not None
                else optional_tensor_arg_no_default
            )
        ),
        ArgType.PLACEHOLDER_TENSOR: (
            (lambda x: tensor_arg(x))
            if not is_optional
            else (
                optional_tensor_arg
                if default is not None
                else optional_tensor_arg_no_default
            )
        ),
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
        ArgType.BOOL: (
            (lambda x: bool_arg(x, default=bool(default)))
            if default is not None
            else bool_arg_no_default
        ),
    }[ty](name)


def make_function_schema_arg(ty: ArgType, name: str, default: Union[int, float]) -> str:
    return {
        ArgType.TENSOR: tensor_arg_annotate,
        ArgType.INT_TENSOR: tensor_arg,
        ArgType.LONG_TENSOR: tensor_arg,
        ArgType.PLACEHOLDER_TENSOR: tensor_arg,
        ArgType.INT: lambda x: int_arg(x, default=int(default)),
        ArgType.FLOAT: lambda x: float_arg(x, default=default),
        # pyre-fixme[6]: For 2nd argument expected `int` but got `Union[float, int]`.
        ArgType.SYM_INT: lambda x: schema_sym_int_arg(x, default=default),
        ArgType.BOOL: lambda x: schema_bool_arg(x, default=bool(default)),
    }[ty](name)


def _extend_tensor_str(name: str, is_cuda: bool, optional: bool) -> str:
    """
    Take a tensor name and extend for cpu or cuda

    Parameters:
    name (str)  - tensor name e.g., "momentum1"
    is_cuda (bool) - If True, extend for cuda tensors. Otherwise, extend for cpu tensors

    Returns:
    String of extended tensors
    """
    opt = "?" if optional else ""
    default = " = None" if optional else ""
    if is_cuda:
        return f"Tensor{opt} {name}_dev {default}, Tensor{opt} {name}_uvm {default}, Tensor{opt} {name}_placements {default}, Tensor{opt} {name}_offsets {default}"
    else:
        return f"Tensor{opt} {name}_host {default}, Tensor{opt} {name}_placements {default}, Tensor{opt} {name}_offsets {default}"


def extend_tensors_args_from_str(args_str: str, example_tensor: str) -> str:
    """
    Extend tensor name for cuda/cpu if tensor args exist.
    For example, if `args_str` contains 'Tensor x', it needs to be extended to
    'Tensor x_host' for cpu, and 'Tensor x_dev, Tensor x_uvm, ...' for cuda

    Parameters:
        args: str - function args e.g., "Tensor momentum1, float eps"
        example_tensor: str - a tensor name already extended
                e.g., "momentum1_dev" for cuda or "momentum1_host" for cpu

    Returns:
        function args where tensor args are extended
    """
    num_tensors = args_str.count("Tensor")
    if num_tensors > 0:
        is_cuda = "_dev" in example_tensor
        args = args_str.split(", ")
        extended_tensors_args = []
        for arg in args:
            ty = arg.split(" ")[0]
            name = arg.split(" ")[1]
            if ty == "Tensor":
                extended_tensors_args.append(_extend_tensor_str(name, is_cuda, False))
            elif ty == "Tensor?":
                extended_tensors_args.append(_extend_tensor_str(name, is_cuda, True))
            else:
                extended_tensors_args.append(arg)
        return ", ".join(extended_tensors_args)
    else:
        return args_str


def make_split_function_args_v1(args_str: str) -> str:
    """
    Create function args for V1 interface from the args_str

    Parameters:
    args: str - function args e.g., "Tensor momentum1_host, float eps"

    Returns:
    function args in string where
        int -> int64_t
        SymInt -> c10::SymInt
        float -> double
    """
    return (
        args_str.replace("int", "int64_t")
        .replace("SymInt", "c10::SymInt")
        .replace("float", "double")
        .replace("Tensor?", "std::optional<Tensor>")
        .replace("None", "std::nullopt")
        .replace("False", "false")
    )


def make_ivalue_cast(ty: ArgType) -> str:
    return {
        ArgType.INT: "toInt",
        ArgType.FLOAT: "toDouble",
        ArgType.SYM_INT: "toSymInt",
        ArgType.BOOL: "toBool",
    }[ty]


def reorder_args(split_arg_spec: List[OptimItem]) -> List[OptimItem]:
    """
    Reorder such that tensor arguments come first. This is used in backend, wrapper and kernels where tensors are no longer optional.
    We need to pass tensor arguments before other types which have default arguments.

    Parameters:
        split_arg_spec (List[OptimItem]): List of argument items

    Return:
        reordered of split_arg_spec
    """
    tensor_args = []
    non_tensor_args = []
    for s in split_arg_spec:
        if s.ty in (
            ArgType.TENSOR,
            ArgType.INT_TENSOR,
            ArgType.LONG_TENSOR,
            ArgType.PLACEHOLDER_TENSOR,
        ):
            tensor_args.append(s)
        else:
            non_tensor_args.append(s)

    return tensor_args + non_tensor_args


@dataclass
class PT2ArgsSet:
    split_function_args: List[str]
    split_function_arg_names: List[str]
    split_function_schemas: List[str]
    split_saved_tensorlist: List[str]
    split_saved_tensorlist_optional: List[str]
    split_saved_data: List[dict[str, str]]
    split_variables: List[str]
    split_unpacked_arg_names: List[str]
    split_args_dict: Dict[str, List[str]]

    @staticmethod
    # pyre-ignore[3]
    def create(
        arg_spec: List[OptimItem],
    ):
        """
        PT2ArgsSet.create() is a method that creates different formats given the optimization arguments
        to be used in TBE codegen PT2 templates.

        Mainly, PT2 unified interface packs tensors to tensor list
        due to limited number of arguments for registered torch ops
        e.g., instead of passing `momentum_host, `momentum_dev`, etc, we pass `momentum`

        Parameters:
        arg_spec: List[OptimItem] - list of argument specs

        Returns:
            PT2ArgsSet object with the following attributes:
            split_function_args: List[str] - List of function arguments used in unified lookup and autograd functions
                                            Tensors will be packed and pass as TensorList. Auxillary arguments will be packed in dict.
                                            e.g., ['at::TensorList momentum1', 'at::Dict<std:string, int> optim_int'].
            split_function_arg_names: List[str] - List of argument names used in unified lookup and autograd functions
                                            e.g., ['momentum1', 'optim_int', 'optim_float'].
            split_function_schemas: List[str] - List of arguments used in unified lookup and autograd functions in the schema format
                                            e.g., ['Tensor[] momentum1', 'float eps', 'float weight_decay'].
            split_saved_tensorlist: List[str] - List of tensor names that are packed into tensorlist and will be unpacked in
                                            PT2 autograd function. e.g., ['momentum1'].
            split_saved_tensorlist_optional: List[str] - List of tensor names that are packed into tensorlist but are optional
                                            and will be unpacked in PT2 autograd function e.g., ['row_counter'].
            split_saved_data: List[dict[str, str]] - List of non-tensor arguments that are saved for backward
            split_unpacked_arg_names: List[str] - List of argument names, unrolled from list
                                            e.g., ['momentum1', 'eps', 'weight_decay', 'iter'].
            split_args_dict: Dict[str, List[str]] - Dict of optim arguments' types containing the argument names of that type.
                                            e.g., if an optimizer only has an int argument called iter, the dict will look like:
                                            {'optim_tensor': [], 'optim_int': ['iter'], 'optim_float': [], 'optim_bool': []}
        """
        split_function_arg_names = []
        split_function_args = []
        split_function_schemas = []
        split_saved_tensorlist = []
        split_saved_tensorlist_optional = []
        split_saved_data = []
        split_variables = []
        split_unpacked_arg_names = []
        has_optim_tensor = False  # optim tensors here are optional tensor
        has_optim_int = False
        has_optim_float = False
        has_optim_bool = False
        split_args_dict = {
            "optim_tensor": [],
            "optim_int": [],
            "optim_float": [],
            "optim_bool": [],
        }
        # list of symint args to be appended after optim_xxx args
        # since they have default values
        symint_list: List[OptimItem] = []

        for s in arg_spec:
            if s.name == "learning_rate_tensor":
                split_function_arg_names.append(s.name)
                split_unpacked_arg_names.append(s.name)
                split_function_args.append(tensor_arg(s.name))
                split_function_schemas.append(tensor_arg(s.name))
                split_variables.append(f"ret.push_back(Variable()); // {s.name}")
            elif s.ty in (
                ArgType.TENSOR,
                ArgType.INT_TENSOR,
                ArgType.LONG_TENSOR,
                ArgType.PLACEHOLDER_TENSOR,
            ):
                name = s.name
                split_unpacked_arg_names.append(name)
                if s.is_optional:
                    split_saved_tensorlist_optional.append(name)
                    split_args_dict["optim_tensor"].append(s.name)
                    has_optim_tensor = True
                else:
                    split_function_args.append(
                        tensor_list_arg_no_default(name, pass_by_ref=False)
                    )
                    split_function_arg_names.append(name)
                    split_function_schemas.append(
                        schema_tensor_list_arg_no_default(name)
                    )
                    split_saved_tensorlist.append(name)
                    split_variables.append(
                        f"ret.push_back(Variable()); // {s.name}_dev or host"
                    )
                    split_variables.append(
                        f"ret.push_back(Variable()); // {s.name}_placements"
                    )
                    split_variables.append(
                        f"ret.push_back(Variable()); // {s.name}_offsets"
                    )
                    split_variables.append("if (" + name + "_host.numel() == 0) {")
                    split_variables.append(
                        f"ret.push_back(Variable()); // {s.name}_uvm"
                    )
                    split_variables.append("}")
            else:
                if s.ty == ArgType.INT:
                    # iter is passed in aux_int
                    if s.name != "iter":
                        split_args_dict["optim_int"].append(s.name)
                        split_saved_data.append(
                            (
                                s.name,
                                f'optim_int[{len(split_args_dict["optim_int"]) - 1}]',
                                make_ivalue_cast(s.ty),
                                "int64_t",
                            )
                        )
                        has_optim_int = True
                elif s.ty == ArgType.SYM_INT:
                    symint_list.append(s)
                    split_saved_data.append(
                        (
                            s.name,
                            "",
                            make_ivalue_cast(s.ty),
                            "c10::SymInt",
                        )
                    )
                elif s.ty == ArgType.FLOAT:
                    split_args_dict["optim_float"].append(s.name)
                    split_saved_data.append(
                        (
                            s.name,
                            f'optim_float[{len(split_args_dict["optim_float"])- 1}]',
                            make_ivalue_cast(s.ty),
                            "double",
                        )
                    )
                    has_optim_float = True
                elif s.ty == ArgType.BOOL:
                    split_args_dict["optim_bool"].append(s.name)
                    split_saved_data.append(
                        (
                            s.name,
                            f'optim_bool[{len(split_args_dict["optim_bool"])- 1}]',
                            make_ivalue_cast(s.ty),
                            "bool",
                        )
                    )
                    has_optim_bool = True
                else:
                    raise ValueError(f"Unsupported type {s.ty}")
                split_unpacked_arg_names.append(s.name)

        def append_lists(type_name: str) -> None:
            """
            Append the list as one argument to the list of function arguments, schemas, names and saved_variables.
            e.g., if type_name is "tensor", optim_tensor will be appended with the corresponding syntax.

            Parameters:
                type_name (str) - type name of the list to be appended

            Returns:
                None
            """
            split_function_args.append(list_arg(type_name))
            split_function_schemas.append(schema_list_arg(type_name))
            split_function_arg_names.append(f"optim_{type_name}")
            split_variables.append(f"ret.push_back(Variable()); // optim_{type_name}")

        if has_optim_tensor:
            append_lists("tensor")
        if has_optim_int:
            append_lists("int")
        if has_optim_float:
            append_lists("float")
        if has_optim_bool:
            append_lists("bool")
        for s in symint_list:
            split_function_arg_names.append(s.name)
            split_function_args.append(make_function_arg(s.ty, s.name, s.default))
            split_function_schemas.append(
                make_function_schema_arg(s.ty, s.name, s.default)
            )
            split_variables.append(f"ret.push_back(Variable()); // {s.name}")
        return PT2ArgsSet(
            split_function_args=split_function_args,
            split_function_arg_names=split_function_arg_names,
            split_function_schemas=split_function_schemas,
            split_saved_tensorlist=split_saved_tensorlist,
            split_saved_tensorlist_optional=split_saved_tensorlist_optional,
            split_saved_data=split_saved_data,
            split_variables=split_variables,
            split_unpacked_arg_names=split_unpacked_arg_names,
            split_args_dict=split_args_dict,
        )


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
    # pyre-fixme[11]: Annotation `TensorType` is not defined as a type.
    placeholder_type_combos: Union[List[Dict[str, TensorType]], List[None]]
    unified_pt2: PT2ArgsSet
    split_kernel_arg_names: List[str]
    split_function_args_autograd: List[str]
    split_function_arg_names_autograd: List[str]
    split_saved_tensors_optional: List[str]
    split_function_args_v1: Optional[str] = None
    split_function_schemas_v1: Optional[str] = None

    @staticmethod
    # pyre-ignore[3]
    def create(
        split_arg_spec: List[OptimItem],
        arg_spec: List[OptimItem],
        additional_spec: Optional[dict[str, Any]] = None,
    ):
        # Keep the argument order for forward/backward compatibility
        # Arg order: non-optional tensors, learning_rate_tensor, non-tensors, optional tensors
        # This is used in lookup and autograd functions
        frontend_split_arg_spec = split_arg_spec.copy()

        has_optional_tensors: bool = False
        # Create another spec for kernels where learning_rate is float
        # This is used in kernels
        kernel_split_arg_spec = split_arg_spec.copy()
        for i, s in enumerate(kernel_split_arg_spec):
            if s.name == "learning_rate_tensor":
                # pyre-ignore[6]
                kernel_split_arg_spec[i] = OptimItem(ArgType.FLOAT, "learning_rate")
            if s.is_optional:
                has_optional_tensors = True

        # Optim arg order: non-optional tensors, learning_rate_tensor, non-tensors, optional tensors
        # The optional tensors are converted to Tensor in autograd functions
        # Hence, need to reorganize such that the tensors come before non-tensors which have default values values
        # This is used in wrapper, backend and kernel functions
        if has_optional_tensors:
            # reordered args for split_arg_spec: non-optional tensors, learning_rate_tensor, optional tensors as tensors, non-tensors
            split_arg_spec = reorder_args(split_arg_spec)
            # reordered args for kernel_split_arg_spec: non-optional tensors, optional tensors as tensors, learning rate (float), non-tensors
            kernel_split_arg_spec = reorder_args(kernel_split_arg_spec)

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

        split_saved_tensors = [
            s.name
            for s in split_arg_spec
            if s.ty
            in (
                ArgType.TENSOR,
                ArgType.INT_TENSOR,
                ArgType.LONG_TENSOR,
                ArgType.PLACEHOLDER_TENSOR,
            )
        ]
        # Create empty tensors based on weights
        # weights name convention is different between v1 and pt2 unified interface (v2)
        # i.e., host_weights, dev_weights uvm_weights, weights_placements, weights_offsets in v1 and weights_{} in v2
        # This is only used in v1, so we fix the name based on v1
        create_empty_tensor = {
            "host": "host_weights.options()",
            "dev": "dev_weights.options()",
            "uvm": "uvm_weights.options()",
            "placements": "weights_placements.options()",
            "offsets": "weights_offsets.options()",
        }
        split_saved_tensors_optional = [
            (
                f"{s.name}.has_value() ? {s.name}.value() : at::empty("
                + "{0}, "
                + create_empty_tensor[s.name.rsplit("_", 1)[1]]
                + ")"
                if s.is_optional
                else s.name
            )
            for s in split_arg_spec
            if s.ty
            in (
                ArgType.TENSOR,
                ArgType.INT_TENSOR,
                ArgType.LONG_TENSOR,
                ArgType.PLACEHOLDER_TENSOR,
            )
        ]

        # Create function args and schemas for V1 interface for backward compatibility
        # V1 interface refers to separate CPU/CUDA lookup functions
        # e.g., split_embedding_codegen_lookup_{}_funtion and split_embedding_codegen_lookup_{}_funtion_cpu)
        split_function_args_v1 = None
        split_function_schemas_v1 = None
        if additional_spec is not None:
            if len(split_saved_tensors) > 0:
                extended_args_str = extend_tensors_args_from_str(
                    additional_spec["v1"], split_saved_tensors[0]
                )
            else:
                extended_args_str = additional_spec["v1"]
            split_function_args_v1 = make_split_function_args_v1(extended_args_str)
            split_function_schemas_v1 = extended_args_str

        # pyre-fixme[28]: Unexpected keyword argument `placeholder_type_combos`.
        return OptimizerArgs(
            # GPU kernel args
            split_kernel_args=[
                make_kernel_arg(s.ty, s.name, s.default) for s in kernel_split_arg_spec
            ],
            split_kernel_args_no_defaults=[
                make_kernel_arg(s.ty, s.name, None) for s in kernel_split_arg_spec
            ],
            split_kernel_arg_constructors=[
                make_kernel_arg_constructor(s.ty, s.name) for s in kernel_split_arg_spec
            ],
            # CPU kernel args
            split_cpu_kernel_args=[
                make_cpu_kernel_arg(s.ty, s.name, s.default)
                for s in kernel_split_arg_spec
            ],
            split_cpu_kernel_arg_constructors=[
                make_cpu_kernel_arg_constructor(s.ty, s.name)
                for s in kernel_split_arg_spec
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
                if (
                    s.ty in (ArgType.TENSOR, ArgType.PLACEHOLDER_TENSOR)
                    and s.name != "learning_rate_tensor"
                    and not s.is_optional
                )
            ],
            split_tensor_types={
                s.name: (
                    "at::acc_type<cache_t, true>"
                    if s.ty == ArgType.TENSOR
                    else (s.name + "_ph_t")
                )
                for s in arg_spec
                if (
                    s.ty in (ArgType.TENSOR, ArgType.PLACEHOLDER_TENSOR)
                    and s.name != "learning_rate_tensor"
                    and not s.is_optional
                )
            },
            split_saved_tensors=split_saved_tensors,
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
                for s in kernel_split_arg_spec
            ],
            placeholder_tensor_names=ph_tensor_names,
            placeholder_type_combos=ph_combos,
            unified_pt2=PT2ArgsSet.create(arg_spec),
            # learning rate remains float in kernels
            split_kernel_arg_names=[
                "learning_rate" if s.name == "learning_rate_tensor" else s.name
                for s in kernel_split_arg_spec
            ],
            split_function_args_autograd=[
                make_function_arg(s.ty, s.name, s.default, s.is_optional)
                for s in frontend_split_arg_spec
            ],
            split_function_arg_names_autograd=[s.name for s in frontend_split_arg_spec],
            split_saved_tensors_optional=split_saved_tensors_optional,
            split_function_args_v1=split_function_args_v1,
            split_function_schemas_v1=split_function_schemas_v1,
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
        arg_spec: List[OptimItem],
        ext_fn: Callable[[OptimItem], List[OptimItem]],
        additional_spec: Optional[dict[str, Any]] = None,
    ) -> OptimizerArgs:
        split_arg_spec = []
        for s in arg_spec:
            # no cpu/cuda extension for learning_rate
            if (
                s.ty in (ArgType.FLOAT, ArgType.INT, ArgType.SYM_INT, ArgType.BOOL)
                or s.name == "learning_rate_tensor"
            ):
                # pyre-fixme[19]: Expected 1 positional argument.
                split_arg_spec.append(OptimItem(s.ty, s.name, s.default))
            else:
                assert s.ty in (ArgType.TENSOR, ArgType.PLACEHOLDER_TENSOR)
                # Treat PLACEHOLDER_TENSOR as TENSOR for CPU
                split_arg_spec.extend(ext_fn(s))
        return OptimizerArgs.create(split_arg_spec, arg_spec, additional_spec)

    @staticmethod
    def extend_for_cpu(spec: OptimItem) -> List[OptimItem]:
        name = spec.name
        default = spec.default
        is_optional = spec.is_optional
        return [
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(ArgType.TENSOR, f"{name}_host", default, is_optional=is_optional),
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(
                ArgType.INT_TENSOR,
                f"{name}_placements",
                default,
                is_optional=is_optional,
            ),
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(
                ArgType.LONG_TENSOR, f"{name}_offsets", default, is_optional=is_optional
            ),
        ]

    @staticmethod
    def extend_for_cuda(spec: OptimItem) -> List[OptimItem]:
        name = spec.name
        default = spec.default
        ty = spec.ty
        ph_tys = spec.ph_tys
        is_optional = spec.is_optional
        return [
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(ty, f"{name}_dev", default, ph_tys, is_optional),
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(ty, f"{name}_uvm", default, ph_tys, is_optional),
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(
                ArgType.INT_TENSOR,
                f"{name}_placements",
                default,
                is_optional=is_optional,
            ),
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(
                ArgType.LONG_TENSOR, f"{name}_offsets", default, is_optional=is_optional
            ),
        ]

    @staticmethod
    def extend_for_any(spec: OptimItem) -> List[OptimItem]:
        name = spec.name
        default = spec.default
        ty = spec.ty
        ph_tys = spec.ph_tys
        is_optional = spec.is_optional
        return [
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(ArgType.TENSOR, f"{name}_host", default, is_optional=is_optional),
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(ty, f"{name}_dev", default, ph_tys, is_optional=is_optional),
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(ty, f"{name}_uvm", default, ph_tys, is_optional=is_optional),
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(
                ArgType.INT_TENSOR,
                f"{name}_placements",
                default,
                is_optional=is_optional,
            ),
            # pyre-fixme[19]: Expected 1 positional argument.
            OptimItem(
                ArgType.LONG_TENSOR, f"{name}_offsets", default, is_optional=is_optional
            ),
        ]

    @staticmethod
    # pyre-ignore[3]
    def create(
        arg_spec: List[OptimItem], additional_spec: Optional[dict[str, Any]] = None
    ):
        return OptimizerArgsSet(
            *(
                OptimizerArgsSet.create_optim_args(arg_spec, ext_fn, additional_spec)
                for ext_fn in (
                    OptimizerArgsSet.extend_for_cpu,
                    OptimizerArgsSet.extend_for_cuda,
                    OptimizerArgsSet.extend_for_any,
                )
            )
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import inspect
import typing
from typing import Iterable, List, Optional, Sequence, Union  # noqa: F401

import torch
from torch import device, dtype, Tensor, types

from torch._library.infer_schema import (
    derived_types,
    parse_return,
    supported_param,
    SUPPORTED_PARAM_TYPES,
    tuple_to_list,
)

# Temporary work around for `infer_schema`

# `get_supported_param_types` and `SUPPORTED_RETURN_TYPES` are modified from torch/_library/infer_schema.py
# as `torch.library.infer_schema` infers any `int` to be `SymInt` in the schema and does not
# support `str` as return type, which may not reflect the actual signature of the function.
# Other modifications are to address linter warning.
# The rest of the code is copied from `torch/_library/infer_schema.py`
# TO DO: clean up and remove this when we implement our own


def error_fn(what: str, sig: Optional[inspect.Signature] = None):
    raise ValueError(f"infer_schema(func): {what} " f"Got func with signature {sig})")


def convert_type_string(annotation_type: str):
    try:
        return eval(annotation_type)
    except Exception:
        error_fn(f"Unsupported type annotation {annotation_type}. It is not a type.")


# Modified support param types and return types from torch/_library/infer_schema.py
def get_supported_param_types():
    data = [
        # (python type, schema type, type[] variant, type?[] variant, type[]? variant
        (Tensor, "Tensor", True, True, False),
        (int, "int", True, False, True),
        (float, "float", True, False, True),
        (bool, "bool", True, False, True),
        (str, "str", False, False, False),
        (types.Number, "Scalar", True, False, False),
        (dtype, "ScalarType", False, False, False),
        (device, "Device", False, False, False),
    ]
    result = []
    for line in data:
        result.extend(derived_types(*line))
    return dict(result)


SUPPORTED_RETURN_TYPES = {
    Tensor: "Tensor",
    typing.List[Tensor]: "Tensor[]",
    int: "int",
    float: "float",
    bool: "bool",
    str: "str",
    types.Number: "Scalar",
}


def check_param_annotation(name: str, annotation: type, sig: inspect.Signature):
    if annotation is inspect.Parameter.empty:
        error_fn(f"Parameter {name} must have a type annotation.", sig)

    # The annotation might be converted to a string by annotation,
    # we convert it to the actual type.
    annotation_type = annotation
    if isinstance(annotation_type, str):
        annotation_type = convert_type_string(annotation_type)

    if annotation_type not in SUPPORTED_PARAM_TYPES.keys():
        if annotation_type.__origin__ is tuple:
            list_type = tuple_to_list(annotation_type)
            example_type_str = "\n\n"
            # Only suggest the list type if this type is supported.
            if list_type in SUPPORTED_PARAM_TYPES.keys():
                example_type_str = f"For example, {list_type}.\n\n"
            error_fn(
                f"Parameter {name} has unsupported type {annotation}. "
                f"We do not support Tuple inputs in schema. As a workaround, please try to use List instead. "
                f"{example_type_str}"
                f"The valid types are: {SUPPORTED_PARAM_TYPES.keys()}.",
                sig,
            )
        else:
            error_fn(
                f"Parameter {name} has unsupported type {annotation}. "
                f"The valid types are: {SUPPORTED_PARAM_TYPES.keys()}.",
                sig,
            )
    return annotation_type


def get_schema_type(
    schema_type: str,
    mutates_args: Union[str, Iterable[str]],
    name: str,
    sig: inspect.Signature,
    idx: int,
):
    if isinstance(mutates_args, str):
        if mutates_args != "unknown":
            raise ValueError(
                "mutates_args must either be a sequence of the names of "
                "the arguments that are mutated or the string 'unknown'. "
            )
        if schema_type.startswith("Tensor"):
            schema_type = f"Tensor(a{idx}!){schema_type[len('Tensor'):]}"
    elif name in mutates_args:
        if not schema_type.startswith("Tensor"):
            error_fn(
                f"Parameter {name} is in mutable_args but only Tensors or collections of Tensors can be mutated"
            )
        schema_type = f"Tensor(a{idx}!){schema_type[len('Tensor'):]}"
    return schema_type


def check_mutates_args(
    mutates_args: Union[str, Iterable[str]], sig: inspect.Signature, seen_args: set
):
    if mutates_args != "unknown":
        mutates_args_not_seen = set(mutates_args) - seen_args
        if len(mutates_args_not_seen) > 0:
            error_fn(
                f"{mutates_args_not_seen} in mutates_args were not found in "
                f"the custom op's signature. "
                f"mutates_args should contain the names of all args that the "
                f"custom op mutates, or just the string 'unknown' if you don't know.",
                sig,
            )


def get_return_annonation(
    return_annotation: type,
):
    if isinstance(return_annotation, str):
        return_annotation = convert_type_string(return_annotation)
    return parse_return(return_annotation, error_fn)


def infer_schema(
    prototype_function: typing.Callable,
    /,
    *,
    mutates_args,
    op_name: Optional[str] = None,
) -> str:
    r"""
    This is modified from torch._library.infer_schema.infer_schema.

    Parses the schema of a given function with type hints. The schema is inferred from the
    function's type hints, and can be used to define a new operator.

    We make the following assumptions:

    * None of the outputs alias any of the inputs or each other.
    * | String type annotations "device, dtype, Tensor, types" without library specification are
      | assumed to be torch.*. Similarly, string type annotations "Optional, List, Sequence, Union"
      | without library specification are assumed to be typing.*.
    * | Only the args listed in ``mutates_args`` are being mutated. If ``mutates_args`` is "unknown",
      | it assumes that all inputs to the operator are being mutates.

    Callers (e.g. the custom ops API) are responsible for checking these assumptions.

    Args:
        prototype_function: The function from which to infer a schema for from its type annotations.
        op_name (Optional[str]): The name of the operator in the schema. If ``name`` is None, then the
            name is not included in the inferred schema. Note that the input schema to
            ``torch.library.Library.define`` requires a operator name.
        mutates_args ("unknown" | Iterable[str]): The arguments that are mutated in the function.

    Returns:
        The inferred schema.

    Example:
        >>> def foo_impl(x: torch.Tensor) -> torch.Tensor:
        >>>     return x.sin()
        >>>
        >>> infer_schema(foo_impl, op_name="foo", mutates_args={})
        foo(Tensor x) -> Tensor
        >>>
        >>> infer_schema(foo_impl, mutates_args={})
        (Tensor x) -> Tensor
    """
    sig = inspect.signature(prototype_function)

    params = []
    seen_args = set()
    saw_kwarg_only_arg = False
    for idx, (name, param) in enumerate(sig.parameters.items()):
        if not supported_param(param):
            error_fn(
                "We do not support positional-only args, varargs, or varkwargs.", sig
            )

        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            # The first time we see a kwarg-only arg, add "*" to the schema.
            if not saw_kwarg_only_arg:
                params.append("*")
                saw_kwarg_only_arg = True

        annotation_type = check_param_annotation(name, param.annotation, sig)

        schema_type = SUPPORTED_PARAM_TYPES[annotation_type]
        schema_type = get_schema_type(schema_type, mutates_args, name, sig, idx)

        seen_args.add(name)
        if param.default is inspect.Parameter.empty:
            params.append(f"{schema_type} {name}")
        else:
            default_repr = None
            if param.default is None or isinstance(param.default, (int, float, bool)):
                default_repr = str(param.default)
            elif isinstance(param.default, (str, torch.device)):
                default_repr = f'"{param.default}"'
            elif isinstance(param.default, torch.dtype):
                dtype_repr = str(param.default)
                torch_dot = "torch."
                assert dtype_repr.startswith(torch_dot)
                default_repr = dtype_repr[len(torch_dot) :]
            else:
                error_fn(
                    f"Parameter {name} has an unsupported default value type {type(param.default)}. "
                    f"Please file an issue on GitHub so we can prioritize this.",
                    sig,
                )
            params.append(f"{schema_type} {name}={default_repr}")
    check_mutates_args(mutates_args, sig, seen_args)

    ret = get_return_annonation(sig.return_annotation)
    if op_name is not None:
        return f"{op_name}({', '.join(params)}) -> {ret}"
    return f"({', '.join(params)}) -> {ret}"

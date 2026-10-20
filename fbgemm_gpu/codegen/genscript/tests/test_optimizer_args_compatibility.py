#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check compatibility between separately packaged Python and C++ code.

Missing frontend scalars use typed defaults; newer backends default absent
scalar slots. Published lookup names, argument contracts, and return types
stay fixed. Reserved lists default to None so older frontends can omit them.
Lookup normalizes a used reserved list to an empty list and ignores unused lists.

These tests check codegen and schemas, not two released packages. Newer
frontends rely on older backends reading the known scalar prefix and accepting
reserved lists. Backends predating those parameters cannot accept them.
"""

import ast
import os
import re
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, List

from deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.optimizer_args import (
    make_lookup_function_args,
    OptimItem,
    PT2ArgsSet,
)
from deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.torch_type_utils import ArgType
from parameterized import parameterized


def resolve_generated_code_dir() -> Path:
    """Resolve Buck's cell-relative generated-output location."""
    configured = Path(os.environ["GENERATED_CODE_DIR"])
    if configured.is_absolute():
        return configured

    from_working_directory = (Path.cwd() / configured).resolve()
    if from_working_directory.exists():
        return from_working_directory

    buck_out = next(
        parent
        for parent in Path(__file__).absolute().parents
        if parent.name == "buck-out"
    )
    return (buck_out.parent / "fbcode" / configured).resolve()


GENERATED_CODE_DIR = resolve_generated_code_dir()


# Independent ABI fixtures pin published types, aliases, defaults, and order.
LOOKUP_PREFIX = (
    "Tensor placeholder_autograd_tensor",
    "Tensor[](a!) weights",
    "Tensor D_offsets",
    "SymInt total_D",
    "SymInt max_D",
    "Tensor hash_size_cumsum",
    "int total_hash_size_bits",
    "Tensor indices",
    "Tensor offsets",
    "int pooling_mode",
    "Tensor? indice_weights",
    "Tensor? feature_requires_grad",
    "int output_dtype",
    "Tensor?[](e!) aux_tensor",
    "int[] aux_int",
    "float[] aux_float",
    "bool[] aux_bool",
)
OPTIMIZER_ARGS = {
    "adagrad": "momentum1 learning_rate_tensor optim_float",
    "adam": "momentum1 momentum2 learning_rate_tensor optim_tensor optim_float optim_bool",
    "lamb": "momentum1 momentum2 learning_rate_tensor optim_float",
    "lars_sgd": "momentum1 learning_rate_tensor optim_float",
    "partial_rowwise_adam": "momentum1 momentum2 learning_rate_tensor optim_float",
    "partial_rowwise_lamb": "momentum1 momentum2 learning_rate_tensor optim_float",
    "rowwise_adagrad": "momentum1 learning_rate_tensor optim_int optim_float",
    "rowwise_adagrad_with_counter": (
        "momentum1 prev_iter row_counter learning_rate_tensor optim_int optim_float"
    ),
    "rowwise_rmsprop_ar": "momentum1 prev_iter learning_rate_tensor optim_float",
    "sgd": "learning_rate_tensor",
    "none": "optim_int total_unique_indices",
}
LIST_TYPES = {
    "optim_tensor": ("Tensor?[]", "std::vector<std::optional<at::Tensor>>"),
    "optim_int": ("int[]", "std::vector<int64_t>"),
    "optim_float": ("float[]", "std::vector<double>"),
    "optim_bool": ("bool[]", "c10::List<bool>"),
}
OPTIMIZER_DECLARATIONS = {
    "momentum1": "Tensor[](g!) momentum1",
    "momentum2": "Tensor[](j!) momentum2",
    "prev_iter": "Tensor[](m!) prev_iter",
    "row_counter": "Tensor[](p!) row_counter",
    "learning_rate_tensor": "Tensor learning_rate_tensor",
    "total_unique_indices": "SymInt total_unique_indices=0",
    **{name: f"{ty} {name}" for name, (ty, _) in LIST_TYPES.items()},
}


def lookup_cases() -> list[tuple[str, str]]:
    """Pin each published lookup name, argument contract, and return type."""
    cases = []
    for optimizer, arguments in OPTIMIZER_ARGS.items():
        argument_names = arguments.split()
        variants = (
            ["split", "ssd"]
            if optimizer in {"adam", "partial_rowwise_adam", "rowwise_adagrad"}
            else ["split"]
        )
        for variant in variants:
            tail = [
                "SymInt max_B=-1",
                "SymInt max_B_feature_rank=-1",
                "SymInt vbe_output_size=-1",
            ]
            if variant == "ssd":
                tail.append("Tensor[]? ssd_tensors=None")
            tail.append("Tensor? vbe_output=None")
            declarations = (
                list(LOOKUP_PREFIX)
                + [OPTIMIZER_DECLARATIONS[name] for name in argument_names]
                + tail
                + [
                    f"{ty}? {name}=None"
                    for name, (ty, _) in LIST_TYPES.items()
                    if name not in argument_names
                ]
            )
            cases.append(
                (
                    f"{variant}_{optimizer}",
                    f"{variant}_embedding_codegen_lookup_{optimizer}_function_pt2"
                    f"({','.join(declarations)}) -> Tensor",
                )
            )
    return cases


def read_lookup_schema(filename: str) -> str:
    """Read the complete dispatcher schema, ignoring delimiter whitespace."""
    source = (GENERATED_CODE_DIR / filename).read_text()
    registration = source[source.index("TORCH_LIBRARY_FRAGMENT") :]
    definition = registration[
        registration.index("m.def(") : registration.index("{PT2_COMPLIANT_TAG}")
    ]
    schema = " ".join("".join(re.findall(r'"([^"]*)"', definition)).split())
    return re.sub(r"\s*([,=])\s*", r"\1", schema).replace("( ", "(").replace(" )", ")")


def pack_args(
    filename: str,
    list_name: str,
    provided_values: dict[str, Any],
) -> tuple[list[Any], list[str]]:
    """Execute one typed packing block twice to verify warning deduplication."""
    source_path = GENERATED_CODE_DIR / filename
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    invoke = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "invoke"
    )

    packing_statement = next(
        statement
        for statement in invoke.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == list_name
    )

    lookup_args_filename = (
        "lookup_args_ssd.py" if "_ssd.py" in filename else "lookup_args.py"
    )
    lookup_args_path = GENERATED_CODE_DIR / lookup_args_filename
    lookup_args_tree = ast.parse(
        lookup_args_path.read_text(), filename=str(lookup_args_path)
    )
    helper_name = f"get_optimizer_{list_name.removeprefix('optim_')}_arg_or_default"
    helper = next(
        statement
        for statement in lookup_args_tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == helper_name
    )

    dictionary_name = f"dict_{list_name}"
    namespace: dict[str, Any] = {
        "Dict": Dict,
        "List": List,
        dictionary_name: provided_values,
        "warnings": warnings,
    }
    module = ast.Module(
        body=[helper, packing_statement, packing_statement], type_ignores=[]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        exec(compile(module, str(source_path), "exec"), namespace)

    return namespace[list_name], [str(warning.message) for warning in caught]


class OptimizerCompatibilityTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "sgd",
                [OptimItem(ArgType.TENSOR, "learning_rate_tensor")],
            ),
            (
                "rowwise_rmsprop_ar",
                [
                    OptimItem(ArgType.TENSOR, "momentum1"),
                    OptimItem(ArgType.TENSOR, "prev_iter"),
                    OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
                    OptimItem(ArgType.FLOAT, "eps"),
                ],
            ),
        ]
    )
    def test_reserved_lists(self, optimizer: str, spec: list[OptimItem]) -> None:
        base_args = PT2ArgsSet.create(spec)
        additions = [
            (
                "optim_tensor",
                OptimItem(ArgType.TENSOR, "row_counter", is_optional=True),
            ),
            ("optim_int", OptimItem(ArgType.INT, "new_int", 7)),
            ("optim_float", OptimItem(ArgType.FLOAT, "new_float", 0.25)),
            ("optim_bool", OptimItem(ArgType.BOOL, "new_bool", True)),
        ]
        reserved = {
            name: types
            for name, types in LIST_TYPES.items()
            if name not in base_args.split_function_arg_names
        }
        for added in [[], *[[item] for item in additions], additions]:
            with self.subTest(lists=[name for name, _ in added]):
                used = {name for name, _ in added}
                lookup = make_lookup_function_args(
                    PT2ArgsSet.create(spec + [item for _, item in added]), optimizer
                )
                self.assertEqual(
                    lookup,
                    {
                        "args": base_args.split_function_args,
                        "schemas": base_args.split_function_schemas,
                        "default_args": [
                            ("" if name in used else "[[maybe_unused]] ")
                            + f"std::optional<{cpp_type}> {name}_optional = std::nullopt"
                            for name, (_, cpp_type) in reserved.items()
                        ],
                        "default_schemas": [
                            f"{schema_type}? {name}=None"
                            for name, (schema_type, _) in reserved.items()
                        ],
                        "default_initializers": [
                            f"[[maybe_unused]] auto {name} = "
                            f"{name}_optional.value_or({cpp_type}{{}});"
                            for name, (_, cpp_type) in reserved.items()
                            if name in used
                        ],
                    },
                )

    @parameterized.expand(lookup_cases())
    def test_lookup_schema(self, name: str, expected: str) -> None:
        self.assertEqual(
            read_lookup_schema(f"gen_embedding_{name}_pt2_autograd.cpp"), expected
        )

    def test_scalar_defaults(self) -> None:
        args = PT2ArgsSet.create(
            [
                OptimItem(ArgType.INT, "new_int", 7),
                OptimItem(ArgType.FLOAT, "new_float", 0.25),
                OptimItem(ArgType.BOOL, "new_bool", True),
            ]
        )

        self.assertEqual(
            {
                "optim_int": {"new_int": 7},
                "optim_float": {"new_float": 0.25},
                "optim_bool": {"new_bool": True},
            },
            args.split_args_defaults,
        )
        reads = {name: read for name, read, _, _ in args.split_saved_data}
        self.assertEqual("optim_int.size() > 0 ? optim_int[0] : 7", reads["new_int"])
        self.assertEqual(
            "optim_float.size() > 0 ? optim_float[0] : 0.25",
            reads["new_float"],
        )
        self.assertEqual(
            "optim_bool.size() > 0 ? optim_bool[0] : true", reads["new_bool"]
        )

    def test_frontend_defaults(self) -> None:
        cases = [
            (
                "lookup_none.py",
                "optim_int",
                {},
                [0],
                "total_hash_size",
            ),
            (
                "lookup_adam.py",
                "optim_float",
                {"eps": 0.01, "beta1": 0.9, "beta2": 0.99},
                [0.01, 0.9, 0.99, 0.0],
                "weight_decay",
            ),
            (
                "lookup_adam.py",
                "optim_bool",
                {},
                [False],
                "use_rowwise_bias_correction",
            ),
        ]

        for filename, list_name, provided, expected, missing_name in cases:
            with self.subTest(list_name=list_name):
                packed, emitted_warnings = pack_args(filename, list_name, provided)

                self.assertEqual(expected, packed)
                self.assertEqual(1, len(emitted_warnings))
                self.assertIn(missing_name, emitted_warnings[0])
                self.assertIn("using its default value", emitted_warnings[0])

    def test_backend_defaults(self) -> None:
        adam = (
            GENERATED_CODE_DIR / "gen_embedding_split_adam_pt2_autograd.cpp"
        ).read_text()
        none = (
            GENERATED_CODE_DIR / "gen_embedding_split_none_pt2_autograd.cpp"
        ).read_text()

        self.assertIn(
            "auto total_hash_size = optim_int.size() > 0 ? optim_int[0] : 0;",
            none,
        )
        self.assertIn(
            "auto weight_decay = optim_float.size() > 3 ? optim_float[3] : 0.0;",
            adam,
        )
        self.assertIn(
            "bool use_rowwise_bias_correction = "
            "optim_bool.size() > 0 ? optim_bool[0] : false;",
            adam,
        )

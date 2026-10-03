#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verify packed optimizer defaults for newer-backend/older-frontend skew.

The Python frontend packs optimizer scalars into typed positional lists, while
the C++ backend unpacks those lists. The packages can be upgraded separately,
so a missing frontend value must occupy its declared slot with a typed default,
and a newer backend must use the same default when an older frontend sends a
shorter list.

These tests do not instantiate an older backend with a newer frontend. That
direction relies on scalar arguments remaining append-only: the older backend
reads its known prefix and ignores the suffix appended by the newer frontend.
"""

import ast
import os
import unittest
import warnings
from pathlib import Path
from typing import Any, List

from deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.optimizer_args import (
    OptimItem,
    PT2ArgsSet,
)
from deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.torch_type_utils import ArgType


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


def execute_generated_packing(
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

    start = next(
        index
        for index, statement in enumerate(invoke.body)
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == list_name
    )
    packing_statements = [invoke.body[start]]
    for statement in invoke.body[start + 1 :]:
        if not isinstance(statement, ast.If):
            break
        packing_statements.append(statement)

    dictionary_name = f"dict_{list_name}"
    namespace: dict[str, Any] = {
        "List": List,
        dictionary_name: provided_values,
        "warnings": warnings,
    }
    packing_module = ast.fix_missing_locations(
        ast.Module(body=packing_statements, type_ignores=[])
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        code = compile(packing_module, str(source_path), "exec")
        exec(code, namespace)
        exec(code, namespace)

    return namespace[list_name], [str(warning.message) for warning in caught]


class OptimizerArgumentCompatibilityTest(unittest.TestCase):
    def test_generator_declares_typed_defaults_for_short_frontend_lists(self) -> None:
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

    def test_generated_frontend_warns_and_packs_missing_typed_defaults(self) -> None:
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
                packed, emitted_warnings = execute_generated_packing(
                    filename, list_name, provided
                )

                self.assertEqual(expected, packed)
                self.assertEqual(1, len(emitted_warnings))
                self.assertIn(missing_name, emitted_warnings[0])
                self.assertIn("using its default value", emitted_warnings[0])

    def test_generated_backend_guards_reads_from_short_frontend_lists(self) -> None:
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


if __name__ == "__main__":
    unittest.main()

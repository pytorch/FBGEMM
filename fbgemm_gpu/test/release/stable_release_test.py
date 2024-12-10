#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import os
import unittest
from typing import Callable

import fbgemm_gpu
import fbgemm_gpu.permute_pooled_embedding_modules
import fbgemm_gpu.sparse_ops

import torch
from torch._C import FunctionSchema, parse_schema
from torch._utils_internal import get_file_path_2

from .utils import infer_schema

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    from test_utils import TestSuite  # pyre-fixme[21]

else:
    # pyre-fixme[21]
    from fbgemm_gpu.test.test_utils import TestSuite


def _check_schema_compatibility(
    schema: FunctionSchema,
    ref_schema_str: str,
) -> None:
    """
    Check if the schema is forward and backward compatible with the reference schema.
    This function will raise an Exception error if the schema is not compatible.

    Args:
        schema (FunctionSchema): The schema object to check.
        ref_schema_str (str): The reference schema in string format.
    Returns:
        None
    """
    assert isinstance(schema, FunctionSchema)
    ref_schema = parse_schema(ref_schema_str)
    # pyre-fixme[16]
    fwd_compatible = schema.check_forward_compatible_with(ref_schema)
    # pyre-fixme[16]
    bwd_compatible = schema.is_backward_compatible_with(ref_schema)
    msg = ""
    if not fwd_compatible:
        msg += f"Schema of {schema} is not forward compatible with {ref_schema}\n"
    # pyre-fixme[16]
    if not bwd_compatible:
        msg += f"Schema of {schema} is not backward compatible with {ref_schema}"
    assert fwd_compatible and bwd_compatible, msg


def check_schema_compatibility(
    op: Callable,
    ref_schema: str,
) -> None:
    """
    Check if the schema of the given operator is forward and backward compatible with the reference schema.
    This works with python functions whose schema do NOT have positional-only args, varargs, or varkwargs
    For ops registered via torch.ops.fbgemm and ops with *args and **kwargs, please use check_schema_compatibility_from_op_name.

    Args:
        op (Callable): The operator to check.
        ref_schema (str): The reference schema in string format.
    Returns:
        None
    """
    op_schema = infer_schema(op, mutates_args={})
    # pyre-fixme[16]
    op_name = op.__name__
    # Create schema string
    schema_str = f"{op_name}{op_schema}"
    # Create FunctionalSchema
    functional_schema = parse_schema(schema_str)

    # Get stable schema to compare against
    return _check_schema_compatibility(functional_schema, ref_schema)


def check_schema_compatibility_from_op_name(
    namespace: Callable,
    op_name: str,
    ref_schema_str: str,
) -> None:
    """
    Check if the schema of the given operator is forward and backward compatible with the reference schema.
    Use this function to check registered ops (via torch.ops.fbgemm).
    This function will raise an Exception error if the schema is not compatible.

    Args:
        namespace (Callable): The namespace of the operator e.g., torch.ops.fbgemm.
        op_name (str): The name of the operator.
        ref_schema_str (str): The reference schema in string format.
    Returns:
        None
    """
    op = getattr(namespace, op_name)
    schema = op._schemas[""]

    return _check_schema_compatibility(schema, ref_schema_str)


class StableRelease(TestSuite):  # pyre-ignore[11]
    def _test_stable_schema(self, version: str) -> None:
        """
        Test the schema compatibility of the operators against stable schema.
        This is to ensure that any changes to the ops' schema do not break compatibility of the stable versions.
        This test will fail if the current op schema is not forward or backward compatible with the stable schema.
        """

        majorversion = version.split(".")[0]
        filepath = get_file_path_2(
            "", os.path.dirname(__file__), f"stable_ops_v{majorversion}.json"
        )

        # Load stable ops from file into dict
        with open(filepath) as file:
            for release_info in [
                info
                for info in json.load(file)["releases"]
                if info["version"] == version
            ]:
                stable_op_dict = release_info["api"]

                # Get all op names
                stable_op_names = set(stable_op_dict.keys())

                # Check compatibility for all ops that are marked stable
                for full_op_name in stable_op_names:
                    # Test the schema given the op name
                    ref_schema_str = stable_op_dict[full_op_name]
                    op_name = full_op_name.split(".")[3]

                    check_schema_compatibility_from_op_name(
                        torch.ops.fbgemm, op_name, ref_schema_str
                    )

    def test_backwards_compatibility(self) -> None:
        """
        Test the schema compatibility of the operators against previous versions of the API.
        """
        for version in ["1.0.0", "1.1.0"]:
            try:
                self._test_stable_schema(version)
            except Exception as e:
                self.fail(f"Compatibility test failed for version {version}: {e}")

    def test_example_ops(self) -> None:
        """
        Test examples for schema compatibility.
        """

        # Load example ops to dict
        stable_dict_file = open(
            get_file_path_2("", os.path.dirname(__file__), "example.json")
        )
        op_dict = json.load(stable_dict_file)["data"]
        stable_dict_file.close()

        # Example op 1
        # Expect to pass
        check_schema_compatibility(
            fbgemm_gpu.sparse_ops.merge_pooled_embeddings,
            op_dict["merge_pooled_embeddings"],
        )

        # Example op 2
        # stable schema is: dummy_func(str var1, int var2) -> ()"
        def dummy_func(var1: str, var2: int, var3: torch.Tensor) -> None:
            pass

        # Expect to fail
        with self.assertRaises(AssertionError):  # pyre-fixme[16]
            check_schema_compatibility(
                dummy_func,
                op_dict["dummy_func"],
            )

        # Example op 3
        # stable schema is: dummy_func(str var1, int var2) -> ()"
        def dummy_func(var1: str, var2: int, var3: str = "default") -> None:
            pass

        # Expect to pass
        check_schema_compatibility(
            dummy_func,
            op_dict["dummy_func"],
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re
from typing import Callable, Dict

import torch


class TorchLibraryFragment:
    """
    A wrapper class around PyTorch library fragments, which are used to define
    and register PyTorch operators.  Handles duplicate operator definitions and
    registrations under the hood.
    """

    def __init__(self, namespace: str) -> None:
        """
        Constructs the TorchLibraryFragment class.

        Args:
            namespace: The namespace for the operators.

        Returns:
            None

        Example:
            lib = TorchLibrary("fbgemm")
        """
        self.namespace = namespace
        self.lib = torch.library.Library(namespace, "FRAGMENT")

    def define(self, schema: str) -> None:
        """
        Defines an operator schema.  This function handles the case where the
        opeator name has already been defined.

        Args:
            schema: The schema of the operator to be defined.  The operator name
            should NOT be prefixed with the operator namespace.

        Returns:
            None

        Example:
            lib = TorchLibrary("fbgemm")
            lib.define("sll_jagged_jagged_bmm(Tensor x, Tensor y, bool flag=True) -> Tensor")
        """
        pattern = re.compile(
            r"""
            (\w+)               # Match the function name (capturing group)
            \s*\(               # Match the opening parenthesis with optional whitespace
            ([^)]*)             # Match params list (capturing group)
            \s*\)               # Match the closing parenthesis with optional whitespace
            \s*->\s*.+          # Match '-> <Return Type>'
            """,
            re.VERBOSE,
        )

        match = pattern.search(schema.strip())
        if match:
            name = match.group(1)
            if f"{self.namespace}::{name}" not in torch.library._defs:
                self.lib.define(schema)
        else:
            raise ValueError(
                f"PyTorch operator schema appears to be ill-defined: '''{schema}'''"
            )

    # pyre-ignore[24]
    def register_dispatch(self, op_name: str, dispatch_key: str, fn: Callable) -> None:
        """
        Registers a single dispatch for an operator with the given name and dispatch key.

        Args:
            op_name: operator name
            dispatch_key: dispatch key that the function should be registered for (e.g., "CUDA")
            fn: a function that is the operator implementation for the input dispatch key

        Returns:
            None

        Example:
            lib = TorchLibrary("fbgemm")
            lib.define(...)
            lib.register_dispatch(lib, "jagged_dense_bmm", jagged_dense_bmm, "CUDA")
        """

        valid_backends = [
            "CUDA",
            "AutogradCUDA",
            "CPU",
            "AutogradCPU",
            "AutogradMeta",
            "Meta",
            "CompositeImplicitAutograd",
        ]
        assert dispatch_key in valid_backends

        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"{self.namespace}::{op_name}", dispatch_key
        ):
            if dispatch_key == "Meta":
                self.lib._register_fake(op_name, fn)
            else:
                self.lib.impl(op_name, fn, dispatch_key)

    # pyre-ignore[24]
    def register(self, op_name: str, functors: Dict[str, Callable]) -> None:
        """
        Registers a set of dispatches for a defined operator.

        Args:
            op_name: operator name
            functors: A dictionary of dispatch keys to dispatch implementations

        Returns:
            None

        Example:
            lib = TorchLibrary("fbgemm")
            lib.define(...)
            lib.register(lib, "jagged_dense_bmm", {"CUDA": jagged_dense_bmm, "Meta": jagged_dense_bmm_meta })
        """
        for dispatch, func in functors.items():
            self.register_dispatch(op_name, dispatch, func)

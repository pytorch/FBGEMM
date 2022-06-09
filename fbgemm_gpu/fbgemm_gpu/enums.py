#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import typing
from typing import Any, Callable, List, Tuple

# Create enums in given namespace with information from query_op
def create_enums(
    namespace: typing.Dict[str, Any],
    query_op: Callable[[], List[Tuple[str, List[Tuple[str, int]]]]],
) -> None:
    for enum_name, items in query_op():
        # Create matching python enumeration
        # pyre-fixme[19]: Expected 1 positional argument.
        new_enum = enum.Enum(enum_name, items)
        # and store it in the module
        namespace[enum_name] = new_enum

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os

from typing import Any, Callable, Dict

# pyre-ignore[5]
environment_variables: Dict[str, Callable[[], Any]] = {
    # Decide which rounding mode to use when doing quantization and dequantization to/from MX4
    # check https://fburl.com/code/rohboxgv for what's available
    "MX4_QUANT_ROUNDING_MODE": lambda: os.getenv("MX4_QUANT_ROUNDING_MODE", "nearest"),
}


# pyre-ignore[3]
def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# pyre-ignore[3]
def __dir__():
    return list(environment_variables.keys())

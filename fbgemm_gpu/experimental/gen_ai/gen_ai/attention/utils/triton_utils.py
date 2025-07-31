# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

import logging
import os

import torch

try:
    #  `fbgemm_gpu.experimental.gen_ai.attention.utils.version`.
    from .version import __version__  # noqa: F401
except ImportError:
    __version__ = "0.0.0"

logger = logging.getLogger("xformers")


def compute_once(func):
    value = None

    def func_wrapper():
        nonlocal value
        if value is None:
            value = func()
        return value

    return func_wrapper


@compute_once
def _is_triton_available():
    if os.environ.get("XFORMERS_ENABLE_TRITON", "0") == "1":
        return True
    if not torch.cuda.is_available():
        return False
    if os.environ.get("XFORMERS_FORCE_DISABLE_TRITON", "0") == "1":
        return False
    # We have many errors on V100 with recent triton versions
    # Let's just drop support for triton kernels below A100
    if torch.cuda.get_device_capability("cuda") < (8, 0):
        return False
    try:
        import triton  # noqa

        return True
    except (ImportError, AttributeError):
        logger.warning(
            "A matching Triton is not available, some optimizations will not be enabled",
            exc_info=True,
        )
        return False

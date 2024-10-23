# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os

import torch


def load_custom_library(lib_name: str) -> None:
    """
    Load a custom library implemented in C++. This
    helper function handles loading libraries both in
    fbcode and OSS.
    """
    try:
        # pyre-ignore[21]
        # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
        from fbgemm_gpu import open_source

        # pyre-ignore[21]
        # @manual=//deeplearning/fbgemm/fbgemm_gpu:test_utils
        from fbgemm_gpu.docs.version import __version__  # noqa: F401
    except Exception:
        open_source: bool = False

    # In open source, all custom ops are packaged into a single library
    # that we load.
    # pyre-ignore[16]
    if open_source:
        torch.ops.load_library(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "fbgemm_gpu_experimental_gen_ai_py.so",
            )
        )
        torch.classes.load_library(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "fbgemm_gpu_experimental_gen_ai_py.so",
            )
        )
    else:
        torch.ops.load_library(lib_name)

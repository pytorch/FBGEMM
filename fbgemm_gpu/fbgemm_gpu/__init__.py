#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re

import torch

# Based on the FBGEMM-PyTorch compatibility table at
# https://docs.pytorch.org/FBGEMM/general/Releases.html#fbgemm-releases-compatibility
_fbgemm_torch_compat_table = {
    "1.3": "2.8",
    "1.2": "2.7",
    "1.1": "2.6",
    "1.0": "2.5",
    "0.8": "2.4",
    "0.7": "2.3",
    "0.6": "2.2",
    "0.5": "2.1",
    "0.4": "2.0",
}


def _load_target_info(target: str) -> dict[str, str]:
    try:
        filepath = os.path.join(
            os.path.dirname(__file__), "docs", f"target.{target}.json.py"
        )
        with open(filepath, "r") as file:
            data = json.load(file)
    except Exception:
        data = {}

    return data


def _load_library(filename: str, version: str, no_throw: bool = False) -> None:
    """Load a shared library from the given filename."""

    # Check if the version of PyTorch is compatible with the version of FBGEMM
    # that we are trying to load, and print a loud warning if not.  This is
    # useful for the OSS build, where we have a single FBGEMM library that is
    # compatible with multiple versions of PyTorch.
    #
    # Based on: https://github.com/pytorch/ao/blob/main/torchao/__init__.py#L30

    keys = [
        key
        for key in _fbgemm_torch_compat_table.keys()
        if version.startswith(f"{key}.")
    ]

    if version == "INTERNAL" or "+git" in version:
        # if FBGEMM version has "+git", assume it's locally built and we don't know
        #   anything about the PyTorch version used to build it
        logging.info(
            "FBGEMM version is INTERNAL or local, ignoring version compatibility check with PyTorch"
        )

    elif re.match(r"^\d{4}\.\d{1,2}\.\d{1,2}.*$", version):
        # if FBGEMM version is a date, assume it's a nightly build and that we
        # know what we're doing
        logging.info(
            "FBGEMM version is a nightly version, ignoring version compatibility check with PyTorch"
        )

    elif not keys:
        logging.warning(
            f"""
            \033[33m
            _fbgemm_torch_compat_table has no entry for {version} of FBGEMM;
            cannot determine compatibility with PyTorch {torch.__version__}
            \033[0m
            """
        )

    elif str(torch.__version__) != _fbgemm_torch_compat_table[keys[0]]:
        logging.warning(
            f"""
            \033[31m
            FBGEMM_GPU version is {version}, which is not guaranteed to be
            compatible with PyTorch {torch.__version__}; library loading might
            crash!

            Please refer to
            https://docs.pytorch.org/FBGEMM/general/Releases.html#fbgemm-releases-compatibility
            for the FBGEMM-PyTorch compatibility table.
            \033[0m
            """
        )

    try:
        torch.ops.load_library(os.path.join(os.path.dirname(__file__), filename))
        logging.info(f"Successfully loaded: '{filename}'")

    except Exception as error:
        logging.error(f"Could not load the library '{filename}'!\n\n\n{error}\n\n\n")
        if not no_throw:
            raise error


# Since __init__.py is only used in OSS context, we define `open_source` here
# and use its existence to determine whether or not we are in OSS context
open_source: bool = True

# Trigger the manual addition of docstrings to pybind11-generated operators
import fbgemm_gpu.docs  # noqa: F401, E402


__targets_infos__ = {
    target: _load_target_info(target) for target in ["default", "genai", "hstu"]
}
__targets_infos__ = {k: v for (k, v) in __targets_infos__.items() if v}

try:
    __target__, __info__ = next(iter(__targets_infos__.items()))
    __variant__ = __info__["variant"]
    __version__ = __info__["version"]
except Exception:
    __variant__: str = "INTERNAL"
    __version__: str = "INTERNAL"
    __target__: str = "INTERNAL"

fbgemm_gpu_libraries = [
    "fbgemm_gpu_config",
    "fbgemm_gpu_tbe_utils",
    "fbgemm_gpu_tbe_index_select",
    "fbgemm_gpu_tbe_cache",
    "fbgemm_gpu_tbe_optimizers",
    "fbgemm_gpu_tbe_inference",
    "fbgemm_gpu_tbe_training_forward",
    "fbgemm_gpu_tbe_training_backward",
    "fbgemm_gpu_tbe_training_backward_pt2",
    "fbgemm_gpu_tbe_training_backward_dense",
    "fbgemm_gpu_tbe_training_backward_split_host",
    "fbgemm_gpu_tbe_training_backward_gwd",
    "fbgemm_gpu_tbe_training_backward_vbe",
    "fbgemm_gpu_py",
]

fbgemm_genai_libraries = [
    "experimental/gen_ai/fbgemm_gpu_experimental_gen_ai",
]

# NOTE: While FBGEMM_GPU GenAI is not available for ROCm yet, we would like to
# be able to install the existing CUDA variant of the package onto ROCm systems,
# so that we can at least use the Triton GEMM libraries from experimental/gemm.
# But loading fbgemm_gpu package will trigger load-checking the .SO file for the
# GenAI libraries, which will fail.  This workaround ignores check-loading the
# .SO file for the ROCm case, so that clients can import
# fbgemm_gpu.experimental.gemm without triggering an error.
if torch.cuda.is_available() and torch.version.hip:
    fbgemm_genai_libraries = []

libraries_to_load = {
    "default": fbgemm_gpu_libraries,
    "genai": fbgemm_genai_libraries,
}

for target, info in __targets_infos__.items():
    for library in libraries_to_load.get(target, []):
        # NOTE: In all cases, we want to throw an error if we cannot load the
        # library.  However, this appears to break the OSS documentation build,
        # where the Python documentation doesn't show up in the generated docs.
        #
        # To work around this problem, we introduce a fake build variant called
        # `docs` and we only throw a library load error when the variant is not
        # `docs`.  For more information, see:
        #
        #   https://github.com/pytorch/FBGEMM/pull/3477
        #   https://github.com/pytorch/FBGEMM/pull/3717
        _load_library(f"{library}.so", info["version"], info["variant"] == "docs")

try:
    # Trigger meta operator registrations
    from . import sparse_ops  # noqa: F401, E402
except Exception:
    pass

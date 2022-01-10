# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from skbuild import setup

cpu_only_build = False


cub_include_path = os.getenv("CUB_DIR", None)
if cub_include_path is None:
    print(
        "CUDA CUB directory environment variable not set.  Using default CUB location."
    )

# Get the long description from the relevant file
cur_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(cur_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

import torch

torch_root = os.path.dirname(torch.__file__)

# Handle command line args before passing to main setup() method.
if "--cpu_only" in sys.argv:
    cpu_only_build = True
    sys.argv.remove("--cpu_only")

setup(
    name="fbgemm_gpu",
    version="0.0.1",
    long_description=long_description,
    packages=["fbgemm_gpu"],
    cmake_args=[f"-DCMAKE_PREFIX_PATH={torch_root}"],
)

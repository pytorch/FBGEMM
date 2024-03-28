# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import jinja2

################################################################################
# Parse Arguments
################################################################################

parser = argparse.ArgumentParser()
# By default the source template files are in the same folder as this script:
# The install dir is by default the same as the current folder.
parser.add_argument(
    "--install_dir", default=".", help="Output directory for generated source files"
)
parser.add_argument("--opensource", action="store_false", dest="is_fbcode")
parser.add_argument("--is_rocm", action="store_true")

args: argparse.Namespace
_: List[str]
args, _ = parser.parse_known_args()


################################################################################
# Setup Jinja Environment
################################################################################

if args.is_fbcode:
    # In fbcode, buck injects SRCDIR into the environment when executing a
    # custom_rule().  The templates will be visible there because they are
    # specified in the `srcs` field of the rule.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(os.path.abspath(os.environ["SRCDIR"]))
    )
else:
    # In OSS, because the generation script is held in `codegen/genscript`, we
    # explicitly point to the parent directory as the root directory of the
    # templates.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )

# Upper Limit of "max_embedding_dim (max_D)":
# BT_block_size * sizeof(float) * 4 * kWarpSize * {{ kMaxVecsPerThread }}
# needs to be smaller than the allocated shared memory size (2/3 of 96 KB
# on V100 and 160 KB on A100.
# BT_block_size * 4 * 4 * 32 * (max_D // 128) <= 64 * 1024 (V100) or 96 * 1024 (A100)
# Since BT_block_size >= 1, max_D <= 16K (V100) or 24K (A100).
# Note that if we increase max_D, it will increase the compilation time significantly.
env.globals["max_embedding_dim"] = 1024
# An optimization for ROCm
# env.globals["items_per_warp"] = 128 if args.is_rocm is False else 256
env.globals["dense"] = False


def write(filename: str, s: str) -> None:
    # All generated files are written to the specified install directory.
    with open(os.path.join(args.install_dir, filename), "w") as f:
        f.write(s)

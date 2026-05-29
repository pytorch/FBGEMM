# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import argparse

################################################################################
# Parse Codegen Scripts' Arguments
################################################################################

parser = argparse.ArgumentParser()
# By default the source template files are in the same folder as this script:
# The install dir is by default the same as the current folder.
parser.add_argument(
    "--install_dir", default=".", help="Output directory for generated source files"
)
parser.add_argument("--opensource", action="store_false", dest="is_fbcode")
parser.add_argument("--is_rocm", action="store_true")
# CMake-derived wave-size set for the current ROCm build. CUDA builds ignore
# these (always wave32). Both unset on a ROCm build defaults to wave64-only
# to preserve pre-port behavior.
parser.add_argument("--has_wave32", action="store_true")
parser.add_argument("--has_wave64", action="store_true")

args: argparse.Namespace
_: list[str]
args, _ = parser.parse_known_args()

print(f"[ARGS PARSE] Parsed arguments: {args}")

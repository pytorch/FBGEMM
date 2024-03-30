# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# flake8: noqa F401

import argparse
from typing import List

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

args: argparse.Namespace
_: List[str]
args, _ = parser.parse_known_args()

print(f"[ARGS PARSE] Parsed arguments: {args}")

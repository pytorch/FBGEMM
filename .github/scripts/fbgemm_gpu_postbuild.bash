#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

echo "################################################################################"
echo "[CMAKE] Running post-build script ..."

# Print directory
pwd

# List all generated .SO files
find . -name '*.so'

# Remove errant RPATHs from the .SO
# https://github.com/pytorch/FBGEMM/issues/3098
# https://github.com/NixOS/patchelf/issues/453
find . -name '*.so' -print0 | xargs -0 patchelf --remove-rpath

echo "[CMAKE] Removed errant RPATHs"
echo "################################################################################"

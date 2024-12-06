#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

echo "################################################################################"
echo "[CMAKE] Running post-build script ..."

TARGET=$1
SET_RPATH_TO_ORIGIN=$2
echo "Target file: ${TARGET}"

# Set or remove RPATHs for the .SO
# https://github.com/pytorch/FBGEMM/issues/3098
# https://github.com/NixOS/patchelf/issues/453
if [ "${SET_RPATH_TO_ORIGIN}" != "" ]; then
    echo "Resetting RPATH to \$ORIGIN ..."
    patchelf --force-rpath --set-rpath "\$ORIGIN" "${TARGET}" || exit 1
else
    echo "Removing all RPATHs ..."
    patchelf --remove-rpath "${TARGET}" || exit 1
fi

readelf -d "${TARGET}" | grep -i rpath
echo "################################################################################"

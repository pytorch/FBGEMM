# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

## Workaround for nova workflow to build wheels from fbgemm_gpu folder
FBGEMM_DIR="/__w/FBGEMM/FBGEMM"
FBGEMM_REPO="${FBGEMM_DIR}/${REPOSITORY}"
export FBGEMM_REPO
working_dir=$(pwd)
BUILD_FROM_NOVA=1
export BUILD_FROM_NOVA

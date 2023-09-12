#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

echo "Current working directory: $(pwd)"
cd "${FBGEMM_REPO}" || echo "Failed to cd to ${FBGEMM_REPO}"
PRELUDE="${FBGEMM_REPO}/.github/scripts/setup_env.bash"
BUILD_ENV_NAME=base
GITHUB_ENV=TRUE
export GITHUB_ENV

# Install FBGEMM_GPU Nightly
echo "Current working directory: $(pwd)"
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. $PRELUDE; install_fbgemm_gpu_wheel "${BUILD_ENV_NAME}" fbgemm_gpu/dist/*.whl

# Test with PyTest
echo "Current working directory: $(pwd)"
CPU_GPU=${CU_VERSION}
if [ "${CU_VERSION}" != 'cpu' ]; then
    CPU_GPU=""
fi
python3 -c "import torch; print("cuda.is_available() ", torch.cuda.is_available()); print ("device_count() ",torch.cuda.device_count());"
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
cd "${FBGEMM_REPO}" || echo "Failed to cd to ${FBGEMM_REPO}"
. $PRELUDE; cd fbgemm_gpu/test || { echo "Failed to cd to fbgemm_gpu/test from $(pwd)"; exit 1; }; run_fbgemm_gpu_tests "${BUILD_ENV_NAME}" "${CPU_GPU}"

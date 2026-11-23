#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Workaround EACCES: permission denied error at checkout step
nova_exit_status=0
trap 'nova_exit_status=$?; set +e
  chown -R 1000:1000 /__w/FBGEMM/FBGEMM/ || echo "Unable to chown 1000:1000 from $USER, uid: $(id -u)"
  exit "$nova_exit_status"
' EXIT

echo "[NOVA] Current working directory: $(pwd)"
cd "${FBGEMM_REPO}" || { echo "[NOVA] Failed to cd to ${FBGEMM_REPO}"; exit 1; }
PRELUDE="${FBGEMM_REPO}/.github/scripts/setup_env.bash"
BUILD_ENV_NAME=${CONDA_ENV}
GITHUB_ENV=TRUE
export GITHUB_ENV
BUILD_FROM_NOVA=1
export BUILD_FROM_NOVA

# Install FBGEMM_GPU Nightly
echo "[NOVA] Current working directory: $(pwd)"

# Load the FBGEMM_GPU build scripts infrastructure
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. "${PRELUDE}" || exit $?

# Record time for each step
start_time=$(date +%s)

echo "################################################################################"
echo "Environment Variables:"
printenv
echo "################################################################################"

# Collect PyTorch environment information
collect_pytorch_env_info "${BUILD_ENV_NAME}"
end_time=$(date +%s)
runtime=$((end_time-start_time))
start_time=${end_time}
echo "[NOVA] Time taken to collect PyTorch environment information: ${runtime} seconds"

# Install the wheel
install_fbgemm_gpu_wheel "${BUILD_ENV_NAME}" fbgemm_gpu/dist/*.whl || exit $?
end_time=$(date +%s)
runtime=$((end_time-start_time))
start_time=${end_time}
echo "[NOVA] Time taken to install wheel: ${runtime} seconds"

# Test with PyTest
$CONDA_RUN python3 -c "import torch; print('cuda.is_available() ', torch.cuda.is_available()); print ('device_count() ',torch.cuda.device_count());"
cd "${FBGEMM_REPO}" || { echo "[NOVA] Failed to cd to ${FBGEMM_REPO} from $(pwd)"; exit 1; }
test_all_fbgemm_gpu_modules "${BUILD_ENV_NAME}" || exit $?
end_time=$(date +%s)
runtime=$((end_time-start_time))
start_time=${end_time}
echo "[NOVA] Time taken to test all unit tests: ${runtime} seconds  / $(display_time ${runtime})"

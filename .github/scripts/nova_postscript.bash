#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

echo "[NOVA] Current working directory: $(pwd)"
cd "${FBGEMM_REPO}" || echo "[NOVA] Failed to cd to ${FBGEMM_REPO}"
PRELUDE="${FBGEMM_REPO}/.github/scripts/setup_env.bash"
BUILD_ENV_NAME=${CONDA_ENV}
GITHUB_ENV=TRUE
export GITHUB_ENV

# Install FBGEMM_GPU Nightly
echo "[NOVA] Current working directory: $(pwd)"

# Load the FBGEMM_GPU build scripts infrastructure
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. "${PRELUDE}";

# Collect PyTorch environment information
collect_pytorch_env_info "${BUILD_ENV_NAME}"

# Install the wheel
install_fbgemm_gpu_wheel "${BUILD_ENV_NAME}" fbgemm_gpu/dist/*.whl

# Test with PyTest
echo "[NOVA] Current working directory: $(pwd)"
if [[ $CU_VERSION = cu* ]]; then
  echo "[NOVA] Testing the CUDA variant of FBGEMM_GPU ..."
  export fbgemm_variant="cuda"

elif [[ $CU_VERSION = rocm* ]]; then
  echo "[NOVA] Testing the ROCm variant of FBGEMM_GPU ..."
  export fbgemm_variant="rocm"

else
  echo "[NOVA] Testing the CPU variant of FBGEMM_GPU ..."
  export fbgemm_variant="cpu"
fi

$CONDA_RUN python3 -c "import torch; print('cuda.is_available() ', torch.cuda.is_available()); print ('device_count() ',torch.cuda.device_count());"
cd "${FBGEMM_REPO}" || { echo "[NOVA] Failed to cd to ${FBGEMM_REPO} from $(pwd)"; };
test_all_fbgemm_gpu_modules "${BUILD_ENV_NAME}" "${fbgemm_variant}"

# Workaround EACCES: permission denied error at checkout step
chown -R 1000:1000 /__w/FBGEMM/FBGEMM/ || echo "Unable to chown 1000:1000 from $USER, uid: $(id -u)"

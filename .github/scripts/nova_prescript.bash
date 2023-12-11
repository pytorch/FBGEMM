#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

export PATH="${PATH}:/usr/sbin:/sbin"

echo "[NOVA] Current working directory: $(pwd)"
cd "${FBGEMM_REPO}" || exit 1

PRELUDE="${FBGEMM_REPO}/.github/scripts/setup_env.bash"
BUILD_ENV_NAME=${CONDA_ENV}

# Load the FBGEMM_GPU build scripts infrastructure
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. "${PRELUDE}";

# Display System Info
print_system_info

# Display Conda information
print_conda_info

# Display GPU Info
print_gpu_info

# Install C/C++ Compilers
install_cxx_compiler "${BUILD_ENV_NAME}"

# Install Build Tools
install_build_tools "${BUILD_ENV_NAME}"

# Collect PyTorch environment information
collect_pytorch_env_info "${BUILD_ENV_NAME}"

if [[ $CU_VERSION = cu* ]]; then
  # Extract the CUDA version number from CU_VERSION
  cuda_version=$(echo "[NOVA] ${CU_VERSION}" | cut -c 3-)
  install_cudnn "${BUILD_ENV_NAME}" "$(pwd)/build_only/cudnn" "${cuda_version}"

  echo "[NOVA] -------- Finding NVML_LIB_PATH -----------"
  if [[ ${NVML_LIB_PATH} == "" ]]; then
    NVML_LIB_PATH=$(find "${CUDA_HOME}" -name libnvidia-ml.so) &&
    export NVML_LIB_PATH &&
    echo "[NOVA] looking in ${CUDA_HOME}" ||
    echo "[NOVA] libnvidia-ml.so not found in ${CUDA_HOME}";
  fi

  if [[ ${NVML_LIB_PATH} == "" ]]; then
    NVML_LIB_PATH=$(find "${CONDA_ENV}" -name libnvidia-ml.so) &&
    export NVML_LIB_PATH &&
    echo "[NOVA] looking in ${CONDA_ENV}" ||
    echo "[NOVA] libnvidia-ml.so not found in ${CONDA_ENV}";
  fi

  echo "[NOVA] NVML_LIB_PATH = ${NVML_LIB_PATH}"
  echo "[NOVA] ------------------------------------------"

  echo "[NOVA] Building the CUDA variant of FBGEMM_GPU ..."
  export fbgemm_variant="cuda"

elif [[ $CU_VERSION = rocm* ]]; then
  echo "[NOVA] Building the ROCm variant of FBGEMM_GPU ..."
  export fbgemm_variant="rocm"

else
  echo "[NOVA] Building the CPU variant of FBGEMM_GPU ..."
  export fbgemm_variant="cpu"
fi

# Install the necessary Python eggs for building
cd "${FBGEMM_REPO}/fbgemm_gpu" || exit 1
prepare_fbgemm_gpu_build "${BUILD_ENV_NAME}"

# Reset the BUILD_FROM_NOVA flag to run setup.py for the actual build
BUILD_FROM_NOVA=0
export BUILD_FROM_NOVA

# Build FBGEMM_GPU nightly by default
if [[ ${CHANNEL} == "" ]]; then
  export CHANNEL="nightly"
fi

# Build the wheel
build_fbgemm_gpu_package "${BUILD_ENV_NAME}" "${CHANNEL}" "${fbgemm_variant}"

# Temporary workaround - copy dist/ to root repo for smoke test
echo "[NOVA] Copying dist folder to root repo ..."
if print_exec cp -r "${FBGEMM_REPO}/fbgemm_gpu/dist" "${FBGEMM_REPO}"; then
  echo "[NOVA] dist folder has been copied to ${FBGEMM_REPO}"
  ls -al "${FBGEMM_REPO}/dist"
else
  echo "[NOVA] Failed to copy dist/ folder to ${FBGEMM_REPO}"
  exit 1
fi

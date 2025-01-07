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

# Record time for each step
start_time=$(date +%s)

# Display System Info
print_system_info
end_time=$(date +%s)
runtime=$((end_time-start_time))
start_time=${end_time}
echo "[NOVA] Time taken to display System Info: ${runtime} seconds"

# Display Conda information
print_conda_info
end_time=$(date +%s)
runtime=$((end_time-start_time))
start_time=${end_time}
echo "[NOVA] Time taken to display Conda information: ${runtime} seconds"

# Display GPU Info
print_gpu_info
end_time=$(date +%s)
runtime=$((end_time-start_time))
start_time=${end_time}
echo "[NOVA] Time taken to display GPU Info: ${runtime} seconds"

# Install Build Tools
install_build_tools "${BUILD_ENV_NAME}"
end_time=$(date +%s)
runtime=$((end_time-start_time))
start_time=${end_time}
echo "[NOVA] Time taken to install Build Tools: ${runtime} seconds"

# Collect PyTorch environment information
collect_pytorch_env_info "${BUILD_ENV_NAME}"
end_time=$(date +%s)
runtime=$((end_time-start_time))
start_time=${end_time}
echo "[NOVA] Time taken to collect PyTorch environment information: ${runtime} seconds"

if [[ $CU_VERSION = cu* ]]; then
  # shellcheck disable=SC2155
  env_prefix=$(env_name_or_prefix "${BUILD_ENV_NAME}")

  echo "[INSTALL] Set environment variables LD_LIBRARY_PATH ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} \
    LD_LIBRARY_PATH="/usr/local/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}" \
    CUDNN_INCLUDE_DIR="${CUDA_HOME}/include" \
    CUDNN_LIBRARY="${CUDA_HOME}/lib64"

  echo "[NOVA] -------- Finding libcuda.so -----------"
  LIBCUDA_PATH=$(find /usr/local -type f -name libcuda.so)
  print_exec ln "${LIBCUDA_PATH}" -s "/usr/local/lib/libcuda.so.1"

  echo "[NOVA] -------- Finding NVML_LIB_PATH -----------"
  if [[ ${NVML_LIB_PATH} == "" ]]; then
    NVML_LIB_PATH=$(find "${CUDA_HOME}" -name libnvidia-ml.so) &&
    ln "${NVML_LIB_PATH}" -s "/usr/local/lib/libnvidia-ml.so.1" &&
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
  end_time=$(date +%s)
  runtime=$((end_time-start_time))
  start_time=${end_time}
  echo "[NOVA] Time taken to find NVML_LIB_PATH: ${runtime} seconds"

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
end_time=$(date +%s)
runtime=$((end_time-start_time))
start_time=${end_time}
echo "[NOVA] Time taken to prepare the build : ${runtime} seconds / $(display_time ${runtime})"

# Reset the BUILD_FROM_NOVA flag to run setup.py for the actual build
BUILD_FROM_NOVA=0
export BUILD_FROM_NOVA

# Build FBGEMM_GPU nightly by default
if [[ ${CHANNEL} == "" ]]; then
  export CHANNEL="nightly"
fi

# Build the wheel
build_fbgemm_gpu_package "${BUILD_ENV_NAME}" "${CHANNEL}" "${fbgemm_variant}"
end_time=$(date +%s)
runtime=$((end_time-start_time))
start_time=${end_time}
echo "[NOVA] Time taken to build the package: ${runtime} seconds / $(display_time ${runtime})"

# Temporary workaround - copy dist/ to root repo for smoke test
echo "[NOVA] Copying dist folder to root repo ..."
if print_exec cp -r "${FBGEMM_REPO}/fbgemm_gpu/dist" "${FBGEMM_REPO}"; then
  echo "[NOVA] dist folder has been copied to ${FBGEMM_REPO}"
  ls -al "${FBGEMM_REPO}/dist"
else
  echo "[NOVA] Failed to copy dist/ folder to ${FBGEMM_REPO}"
  exit 1
fi

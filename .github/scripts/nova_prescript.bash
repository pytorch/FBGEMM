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
echo "--------------------------"
echo "----- conda env list -----"
conda env list
echo "--------------------------"
echo "PRELUDE = $PRELUDE"
export PATH="${PATH}:/usr/sbin:/sbin"
echo "CU_VERSION = ${CU_VERSION}"
echo "PYTHON_VERSION = ${PYTHON_VERSION}"
echo "python3 --version = $(python3 --version)"
echo "ARCH = ${ARCH}"
echo "---------------------------"
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. "${PRELUDE}";

## Display System Info
print_system_info

## Display GPU Info
print_gpu_info

## Install C/C++ Compilers
install_cxx_compiler "${BUILD_ENV_NAME}"

## Install Build Tools
install_build_tools "${BUILD_ENV_NAME}"

## Install cuDNN
CPU_GPU=${CU_VERSION}
if [ "${CU_VERSION}" != 'cpu' ]; then
    ## Nova $CU_VERSION is e.g., cu118
    cuda_version_num=$(echo "$CU_VERSION" | cut -c 3-)
    install_cudnn "${BUILD_ENV_NAME}" "$(pwd)/build_only/cudnn" "$cuda_version_num"
    echo "-------- Finding NVML_LIB_PATH -----------"
    echo "NVML_LIB_PATH = ${NVML_LIB_PATH}"
    echo "CONDA_ENV = ${CONDA_ENV}, CUDA_HOME = ${CUDA_HOME}"
    if [[ ${NVML_LIB_PATH} == "" ]]; then NVML_LIB_PATH=$(find "${CUDA_HOME}" -name libnvidia-ml.so) && export NVML_LIB_PATH && echo "looking in ${CUDA_HOME}" || echo "libnvidia-ml.so not found in ${CUDA_HOME}"; fi
    if [[ ${NVML_LIB_PATH} == "" ]]; then NVML_LIB_PATH=$(find "${CONDA_ENV}" -name libnvidia-ml.so) && export NVML_LIB_PATH && echo "looking in ${CONDA_ENV}" || echo "libnvidia-ml.so not found in ${CONDA_ENV}"; fi
    echo "NVML_LIB_PATH = ${NVML_LIB_PATH}"
    echo "------------------------------------------"
    CPU_GPU="cuda"
fi

cd "${FBGEMM_REPO}/fbgemm_gpu" || { echo "Failed to cd to fbgemm_gpu from $(pwd)"; }
prepare_fbgemm_gpu_build "${BUILD_ENV_NAME}"

# reset NOVA flag to run setup.py
BUILD_FROM_NOVA=0
export BUILD_FROM_NOVA

## Build FBGEMM_GPU Nightly
cd "${FBGEMM_REPO}/fbgemm_gpu" || echo "Failed to cd to ${FBGEMM_REPO}/fbgemm_gpu from $(pwd)"
if [[ ${CHANNEL} == "" ]]; then CHANNEL="nightly"; fi #set nightly by default
echo "----------------------------------------------"
echo "build_fbgemm_gpu_package ${BUILD_ENV_NAME} ${CHANNEL} ${CPU_GPU}"
build_fbgemm_gpu_package "${BUILD_ENV_NAME}" "${CHANNEL}" "${CPU_GPU}"
echo "----------------------------------------------"

## Temporary workaround - copy dist/ to root repo for smoke test
echo "Copying dist folder to root repo.."
(cp -r "${FBGEMM_REPO}/fbgemm_gpu/dist" "${FBGEMM_REPO}") && (echo "dist folder has been copied to ${FBGEMM_REPO}") || echo "Failed to copy dist/ folder to ${FBGEMM_REPO}"
echo "----------------------------------"
ls -al "${FBGEMM_REPO}/dist"
echo "----------------------------------"

#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

echo "Current working directory: $(pwd)"
cd "$FBGEMM_REPO" || echo "Failed to cd to $FBGEMM_REPO"
PRELUDE=.github/scripts/setup_env.bash
BUILD_ENV_NAME=base
echo "----- conda env list -----"
conda env list
echo "--------------------------"
export PATH="$PATH:/usr/sbin:/sbin"
echo "CU_VERSION = $CU_VERSION"
echo "PYTHON_VERSION = $PYTHON_VERSION"
echo "python3 --version = $(python3 --version)"
echo "ARCH = $ARCH"
echo "---------------------------"
## Display System Info
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. $PRELUDE; print_system_info
# shellcheck source=.github/scripts/setup_env.bash
## Display GPU Info
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. $PRELUDE; print_gpu_info
## Install C/C++ Compilers
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. $PRELUDE; install_cxx_compiler "$BUILD_ENV_NAME"
## Install Build Tools
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. $PRELUDE; install_build_tools "$BUILD_ENV_NAME"
## Install cuDNN
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
CPU_GPU=$CU_VERSION
if [ "$CU_VERSION" != 'cpu' ]; then
    cuda_version_num=$(echo "$CU_VERSION" | cut -c 3-)
    # shellcheck disable=SC1091
    # shellcheck source=.github/scripts/setup_env.bash
    . $PRELUDE; install_cudnn "$BUILD_ENV_NAME" "$(pwd)/build_only/cudnn" "$cuda_version_num"
    echo "-------- Finding NVML_LIB_PATH -----------"
    NVML_LIB_PATH=$(find "$FBGEMM_DIR" -name libnvidia-ml.so) || echo "libnvidia-ml.so not found in $FBGEMM_DIR!"
    echo "NVML_LIB_PATH = $NVML_LIB_PATH"
    [[ $NVML_LIB_PATH = "" ]] && NVML_LIB_PATH=$(find "$CUDA_HOME" -name libnvidia-ml.so) || echo "libnvidia-ml.so not found in $CUDA_HOME!"
    [[ $NVML_LIB_PATH = "" ]] && NVML_LIB_PATH=$(find "$CONDA_PREFIX" -name libnvidia-ml.so) || echo "libnvidia-ml.so not found in $CONDA_PREFIX!"
    export NVML_LIB_PATH
    echo "------------------------------------------"
    CPU_GPU="cuda"
fi
## Prepare FBGEMM_GPU Build
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. $PRELUDE; cd fbgemm_gpu || { echo "Failed to cd to fbgemm_gpu/test from $(pwd)"; exit 1; }; prepare_fbgemm_gpu_build "$BUILD_ENV_NAME"

# reset NOVA flag to run setup.py
BUILD_FROM_NOVA=0
export BUILD_FROM_NOVA

## Build FBGEMM_GPU Nightly
cd "$FBGEMM_REPO" || echo "Failed to cd to $FBGEMM_REPO from $(pwd)"
if [[ $CHANNEL == "" ]]; then CHANNEL="nightly"; fi
echo "----------------------------------------------"
echo ". $PRELUDE; build_fbgemm_gpu_package $BUILD_ENV_NAME $CHANNEL $CPU_GPU"
# shellcheck disable=SC1091
# shellcheck source=.github/scripts/setup_env.bash
. $PRELUDE; cd fbgemm_gpu || echo "Failed to cd to fbgemm_gpu from $(pwd)"; build_fbgemm_gpu_package "$BUILD_ENV_NAME" "$CHANNEL" "$CPU_GPU"
echo "----------------------------------------------"

# ## For debugging - preventing the container from terminating
# for ((i=1; i<=10; i++)); do sleep 30m; echo "30 mins hold"; done

## Temporary workaround - copy dist/ to root repo for smoke test
echo "Copying dist folder to root repo.."
(cp -r dist "$FBGEMM_REPO") && (echo "dist folder has been copied to $FBGEMM_REPO") || echo "Failed to copy dist/ folder to $FBGEMM_REPO"
echo "----------------------------------"
ls -al "$FBGEMM_REPO/dist"
echo "----------------------------------"

#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Set project directory
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Set Python and CUDA versions, along with Conda environment name
PYTHON_VERSION=${PYTHON_VERSION:-3.14}
PYTORCH_VERSION=${PYTORCH_VERSION:-nightly}
BUILD_CUDA_VERSION=${BUILD_CUDA_VERSION:-12.9.1}
BUILD_TARGET=${BUILD_TARGET:-default}

# Set build environment name
BUILD_ENV=${BUILD_ENV:-"build-py$PYTHON_VERSION-torch$PYTORCH_VERSION-cu$BUILD_CUDA_VERSION"}

echo "################################################################################"
echo "# FBGEMM_GPU OSS Build and Install (CUDA)"
echo "#"
echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
echo "################################################################################"
echo ""
echo "REPO_ROOT                 : $REPO_ROOT"
echo ""
echo "PYTHON_VERSION            : $PYTHON_VERSION"
echo "PYTORCH_VERSION           : $PYTORCH_VERSION"
echo "BUILD_CUDA_VERSION        : $BUILD_CUDA_VERSION"
echo "BUILD_TARGET              : $BUILD_TARGET"
echo ""
echo "Target Conda Enviroment   : $BUILD_ENV"
echo ""
echo "################################################################################"

# Load the build scripts
# shellcheck disable=SC1091
source "${REPO_ROOT}/.github/scripts/setup_env.bash"
echo "[FBGEMM_GPU CI] Loaded the build scripts ..."

# Install Conda thruogh Miniconda/Miniforge if not already installed
if command -v conda &> /dev/null; then
  echo "[FBGEMM_GPU CI] conda was found ..."
  conda info
else
  echo "[FBGEMM_GPU CI] conda was NOT found; will install conda ..."
  setup_miniconda "$HOME/miniconda"
fi

# Set up the Conda enviroment for building the package
echo "[FBGEMM_GPU CI] Setting up the conda environment ..."
integration_setup_conda_environment "$BUILD_ENV" gcc "$PYTHON_VERSION" "$PYTORCH_VERSION" "cuda/$BUILD_CUDA_VERSION"

# Build and install package into the Conda environment
echo "[FBGEMM_GPU CI] Building the package ..."
integration_fbgemm_gpu_build_and_install "$BUILD_ENV" "$BUILD_TARGET/cuda" "$REPO_ROOT"
echo "[FBGEMM_GPU CI] Package is now installed into $BUILD_ENV"

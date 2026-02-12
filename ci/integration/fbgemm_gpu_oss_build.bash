#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Helper Functions
################################################################################

# Generate the build environment name based on Python version, PyTorch version,
# and build variant. This function can be called from other scripts.
#
# Arguments:
#   PYTHON_VERSION        - Python version (e.g., 3.14)
#   PYTORCH_VERSION       - PyTorch version or git SHA (e.g., nightly, 2.1.0, abc123def)
#   BUILD_VARIANT         - Build variant: cuda, rocm, or cpu
#   BUILD_VARIANT_VERSION - Version for the variant (e.g., 12.9.1 for CUDA, 7.0 for ROCm, empty for cpu)
#
# Returns:
#   Prints the generated environment name to stdout
#
generate_build_env_name() {
  local python_version="${1:-3.14}"
  local pytorch_version="${2:-nightly}"
  local build_variant="${3:-cuda}"
  local build_variant_version="${4:-}"

  # Replace slashes with dashes in pytorch_version for valid env name
  echo "build-py${python_version}-torch-${pytorch_version//\//-}-${build_variant}${build_variant_version}"
}

# Check if a string looks like a git SHA (hexadecimal, 7-40 characters)
is_git_sha() {
    local version="$1"
    if [[ "$version" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

################################################################################
# Main Script
################################################################################

# Set project directory
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Set Python and CUDA versions, along with Conda environment name
PYTHON_VERSION=${PYTHON_VERSION:-3.14}
PYTORCH_VERSION=${PYTORCH_VERSION:-nightly}
BUILD_VARIANT=${BUILD_VARIANT:-cuda}
BUILD_CUDA_VERSION=${BUILD_CUDA_VERSION:-12.9.1}
BUILD_ROCM_VERSION=${BUILD_ROCM_VERSION:-7.0}
BUILD_TARGET=${BUILD_TARGET:-default}
BUILD_SCRIPTS_INIT=${BUILD_SCRIPTS_INIT:-"${REPO_ROOT}/.github/scripts/setup_env.bash"}
PYTORCH_BUILD_SCRIPT=${PYTORCH_BUILD_SCRIPT:-"$(pwd)/pytorch_oss_build.bash"}

# Compute BUILD_VARIANT_VERSION based on build variant
if [[ "$BUILD_VARIANT" == "rocm" ]]; then
  BUILD_VARIANT_VERSION=${BUILD_ROCM_VERSION}
elif [[ "$BUILD_VARIANT" == "cuda" ]]; then
  BUILD_VARIANT_VERSION=${BUILD_CUDA_VERSION}
elif [[ "$BUILD_VARIANT" == "cpu" ]]; then
  BUILD_VARIANT_VERSION=""
else
  echo "[FBGEMM_GPU CI] Invalid build variant: $BUILD_VARIANT"
  exit 1
fi

# Generate build environment name using the helper function
BUILD_ENV=${BUILD_ENV:-$(generate_build_env_name "$PYTHON_VERSION" "$PYTORCH_VERSION" "$BUILD_VARIANT" "$BUILD_VARIANT_VERSION")}

echo "################################################################################"
echo "# FBGEMM_GPU OSS Build and Install"
echo "#"
echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
echo "################################################################################"
echo ""
echo "REPO_ROOT                 : $REPO_ROOT"
echo ""
echo "PYTHON_VERSION            : $PYTHON_VERSION"
echo "PYTORCH_VERSION           : $PYTORCH_VERSION"
echo ""
echo "BUILD_VARIANT             : $BUILD_VARIANT"
echo "BUILD_TARGET              : $BUILD_TARGET"
if [[ "$BUILD_VARIANT" == "rocm" ]]; then
echo "BUILD_ROCM_VERSION        : $BUILD_ROCM_VERSION"
elif [[ "$BUILD_VARIANT" == "cuda" ]]; then
echo "BUILD_CUDA_VERSION        : $BUILD_CUDA_VERSION"
fi
echo ""
echo "Target Conda Enviroment   : $BUILD_ENV"
echo ""
echo "################################################################################"

# Load the build scripts
# shellcheck disable=SC1091,SC1090
source "${BUILD_SCRIPTS_INIT}" || exit 1
echo "[FBGEMM_GPU CI] Loaded the build scripts from: $BUILD_SCRIPTS_INIT ..."

# Source the PyTorch build script
# shellcheck disable=SC1091,SC1090
source "${PYTORCH_BUILD_SCRIPT}" || exit 1
echo "[FBGEMM_GPU CI] Loaded the PyTorch build script from: $PYTORCH_BUILD_SCRIPT ..."

# Install Conda thruogh Miniconda/Miniforge if not already installed
if command -v conda &> /dev/null; then
  echo "[FBGEMM_GPU CI] conda was found ..."
  conda info || exit 1
else
  echo "[FBGEMM_GPU CI] conda was NOT found; will install conda ..."
  setup_miniconda "$HOME/miniconda" || exit 1
fi

# Set up the Conda enviroment for building the package
# If PYTORCH_VERSION looks like a git SHA, build PyTorch from source
# Otherwise, install pre-built PyTorch from PyPI/Conda
echo "[FBGEMM_GPU CI] Setting up the conda environment ..."
if is_git_sha "$PYTORCH_VERSION"; then
  echo "[FBGEMM_GPU CI] Detected git SHA in PYTORCH_VERSION ($PYTORCH_VERSION), will build PyTorch from source ..."

  # Step 1: Set up the base environment (without PyTorch)
  integration_setup_conda_environment_base "$BUILD_ENV" gcc "$PYTHON_VERSION" "$BUILD_VARIANT" "$BUILD_VARIANT_VERSION" || exit 1

  # Step 2: Build PyTorch from source at the specified git SHA
  # Use a unique clone directory based on the SHA to support parallel builds
  PYTORCH_CLONE_DIR="${PYTORCH_CLONE_DIR:-/tmp/pytorch-${PYTORCH_VERSION}}"
  build_pytorch_from_source "$BUILD_ENV" "$PYTORCH_VERSION" "$BUILD_VARIANT" "https://github.com/pytorch/pytorch" "$PYTORCH_CLONE_DIR" || exit 1

else
  echo "[FBGEMM_GPU CI] Using pre-built PyTorch version: $PYTORCH_VERSION ..."
  integration_setup_conda_environment "$BUILD_ENV" gcc "$PYTHON_VERSION" "$PYTORCH_VERSION" "$BUILD_VARIANT/$BUILD_VARIANT_VERSION" || exit 1
fi

# Build and install package into the Conda environment
echo "[FBGEMM_GPU CI] Building the package ..."
integration_fbgemm_gpu_build_and_install "$BUILD_ENV" "$BUILD_TARGET/$BUILD_VARIANT" "$REPO_ROOT" || exit 1

# Run checks and update the Conda environment to support FBGEMM testing
echo "[FBGEMM_GPU CI] Updating the conda environment to support testing ..."
fbgemm_gpu_testing_setup "$BUILD_ENV" || exit 1

echo "[FBGEMM_GPU CI] Exporting the build environment name ..."
export CURRENT_FBGEMM_BUILD_ENV="$BUILD_ENV"

echo "[FBGEMM_GPU CI] Package is now installed into $BUILD_ENV"
echo ""
echo ""
echo "################################################################################"
echo "#"
echo "# FBGEMM_GPU package build and install has successfully completed!"
echo "#"
echo "################################################################################"

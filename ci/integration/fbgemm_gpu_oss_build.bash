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
#   BUILD_VARIANT_VERSION - Version for the variant (e.g., 12.9.1 for CUDA)
#
# Returns:
#   Prints the generated environment name to stdout
generate_build_env_name() {
  local python_version="${1:-3.14}"
  local pytorch_version="${2:-nightly}"
  local build_variant="${3:-cuda}"
  local build_variant_version="${4:-}"

  # For wheel file paths, extract basename and sanitize for env name
  if [[ "$pytorch_version" == *.whl ]]; then
    pytorch_version=$(basename "$pytorch_version" .whl)
    pytorch_version="${pytorch_version//+/_}"
  fi

  echo "build-py${python_version}-torch-${pytorch_version//\//-}-${build_variant}${build_variant_version}"
}

# Check if a string looks like a git SHA (hexadecimal, 7-40 characters)
is_git_sha() {
  [[ "$1" =~ ^[0-9a-fA-F]{7,40}$ ]]
}

################################################################################
# Configuration
################################################################################

REPO_ROOT="$(git rev-parse --show-toplevel)"

PYTHON_VERSION="${PYTHON_VERSION:-3.14}"
PYTORCH_VERSION="${PYTORCH_VERSION:-nightly}"
PYTORCH_VERSION="${PYTORCH_VERSION/#\~/$HOME}"  # Expand ~ to $HOME
BUILD_VARIANT="${BUILD_VARIANT:-cuda}"
BUILD_CUDA_VERSION="${BUILD_CUDA_VERSION:-12.9.1}"
BUILD_ROCM_VERSION="${BUILD_ROCM_VERSION:-7.0}"
BUILD_TARGET="${BUILD_TARGET:-default}"
BUILD_SCRIPTS_INIT="${BUILD_SCRIPTS_INIT:-${REPO_ROOT}/.github/scripts/setup_env.bash}"
PYTORCH_BUILD_SCRIPT="${PYTORCH_BUILD_SCRIPT:-$(pwd)/pytorch_oss_build.bash}"

# Compute BUILD_VARIANT_VERSION based on build variant
case "$BUILD_VARIANT" in
  cuda) BUILD_VARIANT_VERSION="$BUILD_CUDA_VERSION" ;;
  rocm) BUILD_VARIANT_VERSION="$BUILD_ROCM_VERSION" ;;
  cpu)  BUILD_VARIANT_VERSION="" ;;
  *)    echo "[FBGEMM_GPU CI] Invalid build variant: $BUILD_VARIANT"; exit 1 ;;
esac

BUILD_ENV="${BUILD_ENV:-$(generate_build_env_name "$PYTHON_VERSION" "$PYTORCH_VERSION" "$BUILD_VARIANT" "$BUILD_VARIANT_VERSION")}"

################################################################################
# Print Configuration
################################################################################

cat <<EOF
################################################################################
# FBGEMM_GPU OSS Build and Install
#
# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}
################################################################################

REPO_ROOT             : $REPO_ROOT
PYTHON_VERSION        : $PYTHON_VERSION
PYTORCH_VERSION       : $PYTORCH_VERSION
BUILD_VARIANT         : $BUILD_VARIANT
BUILD_TARGET          : $BUILD_TARGET
EOF

[[ "$BUILD_VARIANT" == "cuda" ]] && echo "BUILD_CUDA_VERSION    : $BUILD_CUDA_VERSION"
[[ "$BUILD_VARIANT" == "rocm" ]] && echo "BUILD_ROCM_VERSION    : $BUILD_ROCM_VERSION"

cat <<EOF

Target Conda Environment: $BUILD_ENV

################################################################################
EOF

################################################################################
# Load Build Scripts
################################################################################

# shellcheck disable=SC1091,SC1090
source "$BUILD_SCRIPTS_INIT" || exit 1
echo "[FBGEMM_GPU CI] Loaded build scripts from: $BUILD_SCRIPTS_INIT"

# shellcheck disable=SC1091,SC1090
source "$PYTORCH_BUILD_SCRIPT" || exit 1
echo "[FBGEMM_GPU CI] Loaded PyTorch build script from: $PYTORCH_BUILD_SCRIPT"

################################################################################
# Setup Conda
################################################################################

if command -v conda &>/dev/null; then
  echo "[FBGEMM_GPU CI] conda found"
  conda info || exit 1
else
  echo "[FBGEMM_GPU CI] conda not found; installing..."
  setup_miniconda "$HOME/miniconda" || exit 1
fi

################################################################################
# Setup Conda Environment and Install PyTorch
################################################################################

echo "[FBGEMM_GPU CI] Setting up conda environment..."

if [[ -f "$PYTORCH_VERSION" && "$PYTORCH_VERSION" == *.whl ]]; then
  # Install PyTorch from wheel file
  echo "[FBGEMM_GPU CI] Installing PyTorch from wheel: $PYTORCH_VERSION"
  integration_setup_conda_environment "$BUILD_ENV" gcc "$PYTHON_VERSION" nightly "$BUILD_VARIANT/$BUILD_VARIANT_VERSION" || exit 1
  install_pytorch_wheel_only "$BUILD_ENV" "$PYTORCH_VERSION" || exit 1

elif is_git_sha "$PYTORCH_VERSION"; then
  # Build PyTorch from source at specified git SHA
  echo "[FBGEMM_GPU CI] Building PyTorch from source at: $PYTORCH_VERSION"
  integration_setup_conda_environment_base "$BUILD_ENV" gcc "$PYTHON_VERSION" "$BUILD_VARIANT" "$BUILD_VARIANT_VERSION" || exit 1
  PYTORCH_CLONE_DIR="${PYTORCH_CLONE_DIR:-/tmp/pytorch-${PYTORCH_VERSION}}"
  build_pytorch_from_source "$BUILD_ENV" "$PYTORCH_VERSION" "$BUILD_VARIANT" "https://github.com/pytorch/pytorch" "$PYTORCH_CLONE_DIR" || exit 1

else
  # Install pre-built PyTorch
  echo "[FBGEMM_GPU CI] Installing pre-built PyTorch: $PYTORCH_VERSION"
  integration_setup_conda_environment "$BUILD_ENV" gcc "$PYTHON_VERSION" "$PYTORCH_VERSION" "$BUILD_VARIANT/$BUILD_VARIANT_VERSION" || exit 1
fi

################################################################################
# Build and Install FBGEMM_GPU
################################################################################

echo "[FBGEMM_GPU CI] Building FBGEMM_GPU..."
integration_fbgemm_gpu_build_and_install "$BUILD_ENV" "$BUILD_TARGET/$BUILD_VARIANT" "$REPO_ROOT" || exit 1

echo "[FBGEMM_GPU CI] Setting up test environment..."
fbgemm_gpu_testing_setup "$BUILD_ENV" || exit 1

export CURRENT_FBGEMM_BUILD_ENV="$BUILD_ENV"

cat <<EOF

################################################################################
#
# FBGEMM_GPU package build and install completed successfully!
# Environment: $BUILD_ENV
#
################################################################################
EOF

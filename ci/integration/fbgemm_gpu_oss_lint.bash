#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Set project directory
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"

# Set Python and CUDA versions, along with Conda environment name
PYTHON_VERSION=${PYTHON_VERSION:-3.14}
BUILD_SCRIPTS_INIT=${BUILD_SCRIPTS_INIT:-"${REPO_ROOT}/.github/scripts/setup_env.bash"}

# Set build environment name
BUILD_ENV=${BUILD_ENV:-"lint-py${PYTHON_VERSION}"}

echo "################################################################################"
echo "# FBGEMM_GPU OSS Lint"
echo "#"
echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
echo "################################################################################"
echo ""
echo "REPO_ROOT                 : $REPO_ROOT"
echo ""
echo "PYTHON_VERSION            : $PYTHON_VERSION"
echo ""
echo "Target Conda Enviroment   : $BUILD_ENV"
echo ""
echo "################################################################################"

pushd "$REPO_ROOT" || exit 1

# Load the build scripts
# shellcheck disable=SC1091,SC1090
source "${BUILD_SCRIPTS_INIT}" || exit 1
echo "[FBGEMM_GPU CI] Loaded the build scripts from: $BUILD_SCRIPTS_INIT ..."

# Install Conda thruogh Miniconda/Miniforge if not already installed
if command -v conda &> /dev/null; then
  echo "[FBGEMM_GPU CI] conda was found ..."
  conda info || exit 1
else
  echo "[FBGEMM_GPU CI] conda was NOT found; will install conda ..."
  setup_miniconda "$HOME/miniconda" || exit 1
fi

# Set up the Conda enviroment for building the package
echo "[FBGEMM_GPU CI] Setting up the conda environment ..."
create_conda_environment "$BUILD_ENV" "$PYTHON_VERSION" || exit 1

# Build and install package into the Conda environment
echo "[FBGEMM_GPU CI] Installing lint tools ..."
install_lint_tools "$BUILD_ENV"

echo "[FBGEMM_GPU CI] Linting the Codebase with flake8 ..."
lint_fbgemm_gpu_flake8 "$BUILD_ENV"

echo "[FBGEMM_GPU CI] Linting the Codebase with ufmt ..."
lint_fbgemm_gpu_ufmt "$BUILD_ENV"

echo "[FBGEMM_GPU CI] Check Meta Copyright Headers ..."
lint_fbgemm_gpu_copyright "$BUILD_ENV"

popd || exit 1

echo ""
echo "[FBGEMM_GPU CI] Lint completed; environment: $BUILD_ENV"

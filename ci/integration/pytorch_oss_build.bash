#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# PyTorch OSS Build Script
#
# This script builds PyTorch from source at a specified git SHA and installs
# it into a Conda environment. It reuses the FBGEMM build infrastructure for
# proper CUDA/ROCm setup.
#
# The script is organized into two parts:
#   Part 1: Environment setup (main script) - Sets up Conda environment with
#           compilers, CUDA/ROCm, etc.
#   Part 2: Build PyTorch (build_pytorch_from_source function) - Clones and
#           builds PyTorch. Can be called independently with a pre-configured
#           environment.
#
# Usage:
#   ./pytorch_oss_build.bash <GIT_SHA>
#
# Example:
#   ./pytorch_oss_build.bash abc123def456
#
# Environment Variables:
#   PYTHON_VERSION      - Python version to use (default: 3.12)
#   BUILD_VARIANT       - Build variant: cuda, rocm, or cpu (default: cuda)
#   BUILD_CUDA_VERSION  - CUDA version for CUDA builds (default: 12.9.1)
#   BUILD_ROCM_VERSION  - ROCm version for ROCm builds (default: 7.0)
#   PYTORCH_REPO_URL    - PyTorch repository URL (default: https://github.com/pytorch/pytorch)
#   PYTORCH_CLONE_DIR   - Directory to clone PyTorch into (default: /tmp/pytorch)
#   BUILD_ENV           - Conda environment name (default: auto-generated)
#
################################################################################

# Only set these if running as main script (not being sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -eo pipefail
fi

################################################################################
# Helper Functions
################################################################################

__pytorch_log_info() {
    echo "[PYTORCH BUILD] $*"
}

__pytorch_log_error() {
    echo "[PYTORCH BUILD] ERROR: $*" >&2
}

################################################################################
# Part 2: Build PyTorch from Source
#
# This function can be called independently with a pre-configured Conda
# environment. It clones PyTorch at the specified SHA and builds it.
#
# Arguments:
#   ENV_NAME         - Name of the pre-configured Conda environment
#   GIT_SHA          - Git SHA to build PyTorch from
#   BUILD_VARIANT    - Build variant: cuda, rocm, or cpu
#   PYTORCH_REPO_URL - (optional) PyTorch repository URL
#   PYTORCH_CLONE_DIR - (optional) Directory to clone PyTorch into
#
################################################################################

build_pytorch_from_source() {
    local env_name="$1"
    local git_sha="$2"
    local build_variant="$3"
    local repo_url="${4:-https://github.com/pytorch/pytorch}"
    local clone_dir="${5:-/tmp/pytorch}"

    if [ "$build_variant" == "" ]; then
        echo "Usage: ${FUNCNAME[0]} ENV_NAME GIT_SHA BUILD_VARIANT [PYTORCH_REPO_URL] [PYTORCH_CLONE_DIR]"
        echo "Example(s):"
        echo "    ${FUNCNAME[0]} build_env abc123def456 cuda"
        echo "    ${FUNCNAME[0]} build_env abc123def456 rocm https://github.com/pytorch/pytorch /tmp/pytorch"
        return 1
    else
        echo "################################################################################"
        echo "# Build PyTorch from Source"
        echo "#"
        echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
        echo "################################################################################"
        echo ""
    fi

    __pytorch_log_info "Building PyTorch from source ..."
    __pytorch_log_info "  Environment: ${env_name}"
    __pytorch_log_info "  Git SHA: ${git_sha}"
    __pytorch_log_info "  Build Variant: ${build_variant}"
    __pytorch_log_info "  Repo URL: ${repo_url}"
    __pytorch_log_info "  Clone Dir: ${clone_dir}"

    # shellcheck disable=SC2155
    local env_prefix=$(env_name_or_prefix "${env_name}")

    ############################################################################
    # Step 1: Clone PyTorch Repository (Shallow Clone at Specific SHA)
    ############################################################################

    __pytorch_log_info "Cloning PyTorch repository at SHA: $git_sha ..."
    if [ -d "$clone_dir" ]; then
        __pytorch_log_info "Removing existing PyTorch directory: $clone_dir"
        rm -rf "$clone_dir"
    fi

    # Create the directory and initialize an empty git repo
    mkdir -p "$clone_dir"
    cd "$clone_dir"
    git init

    # Add the remote
    git remote add origin "$repo_url"

    # Fetch only the specific commit with depth=1 (shallow fetch)
    __pytorch_log_info "Fetching commit $git_sha (shallow fetch) ..."
    (exec_with_retries 3 git fetch --depth=1 origin "$git_sha") || return 1

    # Checkout the fetched commit
    __pytorch_log_info "Checking out git SHA: $git_sha"
    git checkout FETCH_HEAD || return 1

    # Initialize and update submodules (shallow)
    __pytorch_log_info "Initializing and updating submodules (shallow) ..."
    git submodule sync || return 1
    (exec_with_retries 3 git submodule update --init --recursive --depth=1) || return 1

    ############################################################################
    # Step 2: Install Build Dependencies
    ############################################################################

    __pytorch_log_info "Installing build dependencies ..."

    # Install common dependencies from PyTorch's pyproject.toml
    # shellcheck disable=SC2086
    (exec_with_retries 3 conda run ${env_prefix} pip install -r requirements.txt) || true

    # Install MKL for x86_64
    if [[ "$(uname -m)" == "x86_64" ]]; then
        __pytorch_log_info "Installing MKL for x86_64 ..."
        # shellcheck disable=SC2086
        (exec_with_retries 3 conda run ${env_prefix} pip install mkl-static mkl-include) || return 1
    fi

    # Install numpy (required for PyTorch build)
    __pytorch_log_info "Installing numpy ..."
    # shellcheck disable=SC2086
    (exec_with_retries 3 conda install ${env_prefix} -c conda-forge --override-channels -y \
        numpy) || return 1

    ############################################################################
    # Step 3: Configure Build Environment
    ############################################################################

    __pytorch_log_info "Configuring build environment ..."

    # Set CMAKE_PREFIX_PATH
    # shellcheck disable=SC2155,SC2086
    local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)

    # shellcheck disable=SC2086
    print_exec conda env config vars set ${env_prefix} CMAKE_PREFIX_PATH="${conda_prefix}"

    # Configure build variant environment variables
    if [[ "$build_variant" == "rocm" ]]; then
        __pytorch_log_info "Configuring ROCm build environment ..."
        # shellcheck disable=SC2086
        print_exec conda env config vars set ${env_prefix} USE_ROCM=1
        # shellcheck disable=SC2086
        print_exec conda env config vars set ${env_prefix} USE_CUDA=0

        # Run AMD build script
        __pytorch_log_info "Running AMD build preprocessing script ..."
        # shellcheck disable=SC2086
        (conda run ${env_prefix} python tools/amd_build/build_amd.py) || return 1

    elif [[ "$build_variant" == "cuda" ]]; then
        __pytorch_log_info "Configuring CUDA build environment ..."
        # shellcheck disable=SC2086
        print_exec conda env config vars set ${env_prefix} USE_CUDA=1
        # shellcheck disable=SC2086
        print_exec conda env config vars set ${env_prefix} USE_ROCM=0

        # Set CUDA_HOME to the conda prefix where CUDA is installed
        # shellcheck disable=SC2086
        print_exec conda env config vars set ${env_prefix} CUDA_HOME="${conda_prefix}"

    elif [[ "$build_variant" == "cpu" ]]; then
        __pytorch_log_info "Configuring CPU build environment ..."
        # shellcheck disable=SC2086
        print_exec conda env config vars set ${env_prefix} USE_CUDA=0
        # shellcheck disable=SC2086
        print_exec conda env config vars set ${env_prefix} USE_ROCM=0
    fi

    ############################################################################
    # Step 4: Build and Install PyTorch
    ############################################################################

    __pytorch_log_info "Building and installing PyTorch (this may take a while) ..."

    cd "$clone_dir"

    # Build PyTorch
    # shellcheck disable=SC2086
    (exec_with_retries 3 conda run ${env_prefix} python -m pip install --no-build-isolation -v -e .) || return 1

    ############################################################################
    # Step 5: Verify Installation
    ############################################################################

    __pytorch_log_info "Verifying PyTorch installation ..."

    # Check that PyTorch is importable
    (test_python_import_package "${env_name}" torch) || return 1
    (test_python_import_package "${env_name}" torch.distributed) || return 1

    # Print installed version
    # shellcheck disable=SC2086,SC2155
    local installed_pytorch_version=$(conda run ${env_prefix} python -c "import torch; print(torch.__version__)")
    __pytorch_log_info "Installed PyTorch version: ${installed_pytorch_version}"

    # Print device availability
    if [[ "$build_variant" == "cuda" ]]; then
        # shellcheck disable=SC2086
        local cuda_available
        cuda_available=$(conda run ${env_prefix} python -c "import torch; print(torch.cuda.is_available())")
        __pytorch_log_info "CUDA available: ${cuda_available}"
        # shellcheck disable=SC2086
        local cuda_version
        cuda_version=$(conda run ${env_prefix} python -c "import torch; print(torch.version.cuda)")
        __pytorch_log_info "CUDA version: ${cuda_version}"
    elif [[ "$build_variant" == "rocm" ]]; then
        # shellcheck disable=SC2086
        local hip_available
        hip_available=$(conda run ${env_prefix} python -c "import torch; print(torch.cuda.is_available())")
        __pytorch_log_info "HIP/ROCm available: ${hip_available}"
        # shellcheck disable=SC2086
        local hip_version
        hip_version=$(conda run ${env_prefix} python -c "import torch; print(torch.version.hip)")
        __pytorch_log_info "HIP version: ${hip_version}"
    fi

    __pytorch_log_info "Successfully built and installed PyTorch from source"
    __pytorch_log_info "PyTorch source directory: $clone_dir"
}

################################################################################
# Part 1: Main Script (Environment Setup + Build)
#
# This is the main entry point when running the script directly.
# It sets up the Conda environment and then calls build_pytorch_from_source.
################################################################################

# Only run main if this script is being executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then

    # Get the script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

    # Set default values for configuration
    PYTHON_VERSION=${PYTHON_VERSION:-3.12}
    BUILD_VARIANT=${BUILD_VARIANT:-cuda}
    BUILD_CUDA_VERSION=${BUILD_CUDA_VERSION:-12.9.1}
    BUILD_ROCM_VERSION=${BUILD_ROCM_VERSION:-7.0}
    PYTORCH_REPO_URL=${PYTORCH_REPO_URL:-https://github.com/pytorch/pytorch}
    PYTORCH_CLONE_DIR=${PYTORCH_CLONE_DIR:-/tmp/pytorch}
    BUILD_SCRIPTS_INIT=${BUILD_SCRIPTS_INIT:-"${REPO_ROOT}/.github/scripts/setup_env.bash"}

    print_usage() {
        echo "Usage: $0 <GIT_SHA>"
        echo ""
        echo "Build PyTorch from source at a specified git SHA."
        echo ""
        echo "Arguments:"
        echo "  GIT_SHA    The git commit SHA to build PyTorch from"
        echo ""
        echo "Environment Variables:"
        echo "  PYTHON_VERSION      - Python version to use (default: 3.12)"
        echo "  BUILD_VARIANT       - Build variant: cuda, rocm, or cpu (default: cuda)"
        echo "  BUILD_CUDA_VERSION  - CUDA version for CUDA builds (default: 12.9.1)"
        echo "  BUILD_ROCM_VERSION  - ROCm version for ROCm builds (default: 7.0)"
        echo "  PYTORCH_REPO_URL    - PyTorch repository URL (default: https://github.com/pytorch/pytorch)"
        echo "  PYTORCH_CLONE_DIR   - Directory to clone PyTorch into (default: /tmp/pytorch)"
        echo ""
        echo "Example:"
        echo "  $0 abc123def456"
        echo "  BUILD_VARIANT=rocm $0 abc123def456"
    }

    # Check for required argument
    if [ $# -lt 1 ]; then
        __pytorch_log_error "Missing required argument: GIT_SHA"
        print_usage
        exit 1
    fi

    PYTORCH_GIT_SHA="$1"

    # Set build environment name based on variant
    if [[ "$BUILD_VARIANT" == "rocm" ]]; then
        BUILD_VARIANT_VERSION=${BUILD_ROCM_VERSION}
    elif [[ "$BUILD_VARIANT" == "cuda" ]]; then
        BUILD_VARIANT_VERSION=${BUILD_CUDA_VERSION}
    elif [[ "$BUILD_VARIANT" == "cpu" ]]; then
        BUILD_VARIANT_VERSION=none
    else
        __pytorch_log_error "Invalid build variant: $BUILD_VARIANT"
        exit 1
    fi

    # Generate environment name if not provided
    BUILD_ENV=${BUILD_ENV:-"pytorch-py${PYTHON_VERSION}-${BUILD_VARIANT}${BUILD_VARIANT_VERSION}-${PYTORCH_GIT_SHA:0:8}"}

    echo "################################################################################"
    echo "# PyTorch OSS Build from Source"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${0} ${*}"
    echo "################################################################################"
    echo ""
    echo "PYTORCH_GIT_SHA           : $PYTORCH_GIT_SHA"
    echo "PYTORCH_REPO_URL          : $PYTORCH_REPO_URL"
    echo "PYTORCH_CLONE_DIR         : $PYTORCH_CLONE_DIR"
    echo ""
    echo "PYTHON_VERSION            : $PYTHON_VERSION"
    echo "BUILD_VARIANT             : $BUILD_VARIANT"
    if [[ "$BUILD_VARIANT" == "rocm" ]]; then
    echo "BUILD_ROCM_VERSION        : $BUILD_ROCM_VERSION"
    elif [[ "$BUILD_VARIANT" == "cuda" ]]; then
    echo "BUILD_CUDA_VERSION        : $BUILD_CUDA_VERSION"
    fi
    echo ""
    echo "Target Conda Environment  : $BUILD_ENV"
    echo ""
    echo "################################################################################"

    # Load the build scripts
    # shellcheck disable=SC1091,SC1090
    source "${BUILD_SCRIPTS_INIT}" || exit 1
    __pytorch_log_info "Loaded the build scripts from: $BUILD_SCRIPTS_INIT"

    ############################################################################
    # Step 1: Set up Conda
    ############################################################################

    if command -v conda &> /dev/null; then
        __pytorch_log_info "conda was found ..."
        conda info || exit 1
    else
        __pytorch_log_info "conda was NOT found; will install conda ..."
        setup_miniconda "$HOME/miniconda" || exit 1
    fi

    ############################################################################
    # Step 2: Set up Conda Environment with CUDA/ROCm
    ############################################################################

    __pytorch_log_info "Setting up Conda environment with ${BUILD_VARIANT} support ..."

    integration_setup_conda_environment_base \
        "$BUILD_ENV" \
        "gcc" \
        "$PYTHON_VERSION" \
        "$BUILD_VARIANT" \
        "$BUILD_VARIANT_VERSION" \
        || exit 1

    ############################################################################
    # Step 3: Build PyTorch from source (calls Part 2 function)
    ############################################################################

    build_pytorch_from_source \
        "$BUILD_ENV" \
        "$PYTORCH_GIT_SHA" \
        "$BUILD_VARIANT" \
        "$PYTORCH_REPO_URL" \
        "$PYTORCH_CLONE_DIR" \
        || exit 1

    ############################################################################
    # Step 4: Export Environment
    ############################################################################

    __pytorch_log_info "Exporting the build environment name ..."
    export CURRENT_PYTORCH_BUILD_ENV="$BUILD_ENV"

    echo ""
    echo "################################################################################"
    echo "#"
    echo "# PyTorch build and install has successfully completed!"
    echo "#"
    echo "# To activate the environment, run:"
    echo "#     conda activate $BUILD_ENV"
    echo "#"
    echo "# PyTorch source directory: $PYTORCH_CLONE_DIR"
    echo "#"
    echo "################################################################################"
fi

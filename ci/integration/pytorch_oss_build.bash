#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# PyTorch OSS Build Script
#
# Builds PyTorch from source at a specified git SHA and installs it into a
# Conda environment. Reuses FBGEMM build infrastructure for CUDA/ROCm setup.
#
# Usage:
#   ./pytorch_oss_build.bash <GIT_SHA>
#
# Environment Variables:
#   PYTHON_VERSION      - Python version (default: 3.12)
#   BUILD_VARIANT       - cuda, rocm, or cpu (default: cuda)
#   BUILD_CUDA_VERSION  - CUDA version (default: 12.9.1)
#   BUILD_ROCM_VERSION  - ROCm version (default: 7.0)
#   PYTORCH_REPO_URL    - Repository URL (default: https://github.com/pytorch/pytorch)
#   PYTORCH_CLONE_DIR   - Clone directory (default: /tmp/pytorch)
#   BUILD_ENV           - Conda environment name (default: auto-generated)
################################################################################

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && set -eo pipefail

################################################################################
# Helper Functions
################################################################################

__pytorch_log_info()  { echo "[PYTORCH BUILD] $*"; }
__pytorch_log_error() { echo "[PYTORCH BUILD] ERROR: $*" >&2; }

# Configure CUDA environment variables for PyTorch build.
# For newer conda CUDA (12.6+), headers are in targets/<arch>-linux/include/
__configure_pytorch_cuda_build() {
  local env_name="$1" conda_prefix="$2"
  local machine_name_lc
  machine_name_lc=$(uname -m | tr '[:upper:]' '[:lower:]')

  export PYTORCH_CUDA_TOOLKIT_ROOT="${conda_prefix}/targets/${machine_name_lc}-linux"
  export PYTORCH_CUDA_NVCC_EXECUTABLE="${conda_prefix}/bin/nvcc"

  __pytorch_log_info "CUDA config: TOOLKIT_ROOT=${PYTORCH_CUDA_TOOLKIT_ROOT}, NVCC=${PYTORCH_CUDA_NVCC_EXECUTABLE}"
}

################################################################################
# Install PyTorch Wheel
################################################################################

install_pytorch_wheel_only() {
  local env_name="$1" wheel_path="$2"

  if [[ -z "$wheel_path" ]]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME WHEEL_PATH"
    return 1
  fi

  cat <<EOF
################################################################################
# Install PyTorch Wheel
# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}
################################################################################
EOF

  __pytorch_log_info "Installing wheel: $wheel_path into env: $env_name"

  [[ ! -f "$wheel_path" ]] && { __pytorch_log_error "Wheel not found: $wheel_path"; return 1; }

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Uninstall existing PyTorch, install new wheel (both without deps)
  # shellcheck disable=SC2086
  conda run --no-capture-output ${env_prefix} pip uninstall -y --no-deps torch 2>/dev/null || true
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} pip install --no-deps "$wheel_path") || return 1

  # Verify installation
  (test_python_import_package "${env_name}" torch) || return 1

  # shellcheck disable=SC2086,SC2155
  local version=$(conda run --no-capture-output ${env_prefix} python -c "import torch; print(torch.__version__)")
  __pytorch_log_info "Installed PyTorch version: $version"
}

################################################################################
# Build PyTorch from Source
#
# Arguments:
#   ENV_NAME          - Pre-configured Conda environment
#   GIT_SHA           - Git SHA to build
#   BUILD_VARIANT     - cuda, rocm, or cpu
#   PYTORCH_REPO_URL  - (optional) Repository URL
#   PYTORCH_CLONE_DIR - (optional) Clone directory
################################################################################

build_pytorch_from_source() {
  local env_name="$1" git_sha="$2" build_variant="$3"
  local repo_url="${4:-https://github.com/pytorch/pytorch}"
  local clone_dir="${5:-/tmp/pytorch}"

  if [[ -z "$build_variant" ]]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME GIT_SHA BUILD_VARIANT [REPO_URL] [CLONE_DIR]"
    return 1
  fi

  cat <<EOF
################################################################################
# Build PyTorch from Source
# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}
################################################################################

Environment:    $env_name
Git SHA:        $git_sha
Build Variant:  $build_variant
Repo URL:       $repo_url
Clone Dir:      $clone_dir
EOF

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Clone repository (shallow at specific SHA)
  __pytorch_log_info "Cloning PyTorch at SHA: $git_sha"
  rm -rf "$clone_dir"
  mkdir -p "$clone_dir" && cd "$clone_dir"
  git init && git remote add origin "$repo_url"
  (exec_with_retries 3 git fetch --depth=1 origin "$git_sha") || return 1
  git checkout FETCH_HEAD || return 1
  git submodule sync || return 1
  (exec_with_retries 3 git submodule update --init --recursive --depth=1) || return 1

  # Install build dependencies
  __pytorch_log_info "Installing build dependencies..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} pip install -r requirements.txt) || true

  if [[ "$(uname -m)" == "x86_64" ]]; then
    # shellcheck disable=SC2086
    (exec_with_retries 3 conda run --no-capture-output ${env_prefix} pip install mkl-static mkl-include) || return 1
  fi

  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge --override-channels -y numpy numactl) || return 1

  # Configure build environment
  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run --no-capture-output ${env_prefix} printenv CONDA_PREFIX)

  # Compiler/linker flags for large CUDA builds
  # -mcmodel=medium: Handle large symbol tables
  # --no-relax: Prevent relocation overflow errors
  local compiler_flags="-mcmodel=medium"
  local linker_flags="-Wl,--no-relax -L${conda_prefix}/lib -Wl,-rpath,${conda_prefix}/lib -Wl,-rpath-link,${conda_prefix}/lib"

  __pytorch_log_info "Building PyTorch (this may take a while)..."
  cd "$clone_dir"

  # Common build environment
  local -a build_env=(
    "CMAKE_PREFIX_PATH=${conda_prefix}"
    "LDFLAGS=${linker_flags}"
    "CMAKE_SHARED_LINKER_FLAGS=${linker_flags}"
    "CMAKE_EXE_LINKER_FLAGS=${linker_flags}"
  )

  case "$build_variant" in
    cuda)
      __configure_pytorch_cuda_build "${env_name}" "${conda_prefix}"
      build_env+=(
        "LD_LIBRARY_PATH=${conda_prefix}/lib:${LD_LIBRARY_PATH:-}"
        "CUDA_TOOLKIT_ROOT=${PYTORCH_CUDA_TOOLKIT_ROOT}"
        "CUDA_NVCC_EXECUTABLE=${PYTORCH_CUDA_NVCC_EXECUTABLE}"
        "CMAKE_C_FLAGS=${compiler_flags}"
        "CMAKE_CXX_FLAGS=${compiler_flags}"
        "CFLAGS=${compiler_flags}"
        "CXXFLAGS=${compiler_flags}"
        "MAX_JOBS=8"
        "USE_CUDA=1"
        "USE_ROCM=0"
      )
      ;;
    rocm)
      __pytorch_log_info "Running AMD build preprocessing..."
      # shellcheck disable=SC2086
      conda run --no-capture-output ${env_prefix} python tools/amd_build/build_amd.py || return 1
      build_env+=("USE_ROCM=1" "USE_CUDA=0")
      ;;
    *)
      build_env+=("USE_CUDA=0" "USE_ROCM=0")
      ;;
  esac

  # Build and install
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} \
    env "${build_env[@]}" python -m pip install --no-build-isolation -v -e .) || return 1

  # Verify installation
  __pytorch_log_info "Verifying installation..."
  (test_python_import_package "${env_name}" torch) || return 1
  (test_python_import_package "${env_name}" torch.distributed) || return 1

  # shellcheck disable=SC2086,SC2155
  local version=$(conda run --no-capture-output ${env_prefix} python -c "import torch; print(torch.__version__)")
  __pytorch_log_info "Installed PyTorch version: $version"

  # Check device availability
  if [[ "$build_variant" == "cuda" ]]; then
    # shellcheck disable=SC2086,SC2155
    local cuda_avail=$(conda run --no-capture-output ${env_prefix} python -c "import torch; print(torch.cuda.is_available())")
    # shellcheck disable=SC2086,SC2155
    local cuda_ver=$(conda run --no-capture-output ${env_prefix} python -c "import torch; print(torch.version.cuda)")
    __pytorch_log_info "CUDA available: $cuda_avail, version: $cuda_ver"
  elif [[ "$build_variant" == "rocm" ]]; then
    # shellcheck disable=SC2086,SC2155
    local hip_avail=$(conda run --no-capture-output ${env_prefix} python -c "import torch; print(torch.cuda.is_available())")
    # shellcheck disable=SC2086,SC2155
    local hip_ver=$(conda run --no-capture-output ${env_prefix} python -c "import torch; print(torch.version.hip)")
    __pytorch_log_info "HIP available: $hip_avail, version: $hip_ver"
  fi

  __pytorch_log_info "Successfully built PyTorch from source at: $clone_dir"
}

################################################################################
# Main Script (when executed directly)
################################################################################

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

  # Configuration
  PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
  BUILD_VARIANT="${BUILD_VARIANT:-cuda}"
  BUILD_CUDA_VERSION="${BUILD_CUDA_VERSION:-12.9.1}"
  BUILD_ROCM_VERSION="${BUILD_ROCM_VERSION:-7.0}"
  PYTORCH_REPO_URL="${PYTORCH_REPO_URL:-https://github.com/pytorch/pytorch}"
  PYTORCH_CLONE_DIR="${PYTORCH_CLONE_DIR:-/tmp/pytorch}"
  BUILD_SCRIPTS_INIT="${BUILD_SCRIPTS_INIT:-${REPO_ROOT}/.github/scripts/setup_env.bash}"

  # NCCL v2.28.9+ requires glibc 2.18+ for pthread_setattr_default_np
  export GLIBC_VERSION="${GLIBC_VERSION:-2.28}"

  if [[ $# -lt 1 ]]; then
    cat <<EOF
Usage: $0 <GIT_SHA>

Build PyTorch from source at a specified git SHA.

Environment Variables:
  PYTHON_VERSION, BUILD_VARIANT, BUILD_CUDA_VERSION, BUILD_ROCM_VERSION,
  PYTORCH_REPO_URL, PYTORCH_CLONE_DIR, GLIBC_VERSION

Example:
  $0 abc123def456
  BUILD_VARIANT=rocm $0 abc123def456
EOF
    exit 1
  fi

  PYTORCH_GIT_SHA="$1"

  # Set variant version
  case "$BUILD_VARIANT" in
    cuda) BUILD_VARIANT_VERSION="$BUILD_CUDA_VERSION" ;;
    rocm) BUILD_VARIANT_VERSION="$BUILD_ROCM_VERSION" ;;
    cpu)  BUILD_VARIANT_VERSION="none" ;;
    *)    __pytorch_log_error "Invalid BUILD_VARIANT: $BUILD_VARIANT"; exit 1 ;;
  esac

  BUILD_ENV="${BUILD_ENV:-pytorch-py${PYTHON_VERSION}-${BUILD_VARIANT}${BUILD_VARIANT_VERSION}-${PYTORCH_GIT_SHA:0:8}}"

  cat <<EOF
################################################################################
# PyTorch OSS Build from Source
# [$(date --utc +%FT%T.%3NZ)] + ${0} ${*}
################################################################################

Git SHA:        $PYTORCH_GIT_SHA
Repo URL:       $PYTORCH_REPO_URL
Clone Dir:      $PYTORCH_CLONE_DIR
Python:         $PYTHON_VERSION
Build Variant:  $BUILD_VARIANT ${BUILD_VARIANT_VERSION}
GLIBC:          $GLIBC_VERSION
Environment:    $BUILD_ENV

################################################################################
EOF

  # Load build scripts
  # shellcheck disable=SC1091,SC1090
  source "${BUILD_SCRIPTS_INIT}" || exit 1
  __pytorch_log_info "Loaded build scripts from: $BUILD_SCRIPTS_INIT"

  # Setup conda
  if command -v conda &>/dev/null; then
    __pytorch_log_info "conda found"
    conda info || exit 1
  else
    __pytorch_log_info "Installing conda..."
    setup_miniconda "$HOME/miniconda" || exit 1
  fi

  # Setup environment and build
  __pytorch_log_info "Setting up Conda environment with ${BUILD_VARIANT} support..."
  integration_setup_conda_environment_base "$BUILD_ENV" gcc "$PYTHON_VERSION" "$BUILD_VARIANT" "$BUILD_VARIANT_VERSION" || exit 1

  build_pytorch_from_source "$BUILD_ENV" "$PYTORCH_GIT_SHA" "$BUILD_VARIANT" "$PYTORCH_REPO_URL" "$PYTORCH_CLONE_DIR" || exit 1

  export CURRENT_PYTORCH_BUILD_ENV="$BUILD_ENV"

  cat <<EOF

################################################################################
# PyTorch build completed successfully!
#
# Activate: conda activate $BUILD_ENV
# Source:   $PYTORCH_CLONE_DIR
################################################################################
EOF
fi

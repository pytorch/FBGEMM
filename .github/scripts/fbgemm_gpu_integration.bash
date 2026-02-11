#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pip.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pytorch.bash"

################################################################################
# MSLK Integration Scripts
################################################################################

integration_setup_conda_environment_base () {
  # Sets up a Conda build environment with compilers, build tools, and
  # CUDA/ROCm support, but WITHOUT installing PyTorch. This is useful for
  # building PyTorch from source.
  #
  # Arguments:
  #   ENV_NAME         - Name of the Conda environment to create
  #   COMPILER         - Compiler to use (gcc or clang)
  #   PYTHON_VERSION   - Python version to install (e.g., 3.12)
  #   VARIANT_TYPE     - Build variant type: cuda, rocm, or cpu
  #   VARIANT_VERSION  - Version of the variant (e.g., 12.9.1 for CUDA, 7.0 for ROCm, "none" for CPU)
  #
  local env_name="$1"
  local compiler="$2"
  local python_version="$3"
  local variant_type="$4"
  local variant_version="$5"
  if [ "$variant_type" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME COMPILER PYTHON_VERSION VARIANT_TYPE VARIANT_VERSION"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env gcc 3.12 cuda 12.9.1   # Setup environment for GCC + Python 3.12 + CUDA 12.9.1"
    echo "    ${FUNCNAME[0]} build_env gcc 3.12 rocm 7.0      # Setup environment for GCC + Python 3.12 + ROCm 7.0"
    echo "    ${FUNCNAME[0]} build_env gcc 3.12 cpu none      # Setup environment for GCC + Python 3.12 (CPU only)"
    return 1
  else
    echo "################################################################################"
    echo "# Setup Conda Build Environment (Without PyTorch)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  echo "[SETUP] Creating the Conda build environment: ${env_name} ..."
  create_conda_environment  "${env_name}" "${python_version}"           || return 1

  echo "[SETUP] Build variant: ${variant_type} / ${variant_version}"

  # Set the BUILD_*_VERSION environment variables
  if [ "$variant_type" == "cuda" ]; then
    print_exec conda env config vars set -n "${env_name}" BUILD_CUDA_VERSION="${variant_version}"
  elif [[ "$variant_type" == "rocm" ]]; then
    print_exec conda env config vars set -n "${env_name}" BUILD_ROCM_VERSION="${variant_version}"
  fi

  # Install C++ compiler and build tools
  if [ "$compiler" == "gcc" ] || [ "$compiler" == "clang" ]; then
    install_cxx_compiler  "${env_name}" "${compiler}"   || return 1
  fi
  install_build_tools     "${env_name}"                 || return 1

  # Install CUDA tools and runtime
  if [ "$variant_type" == "cuda" ]; then
    install_cuda  "${env_name}" "${variant_version}"                                            || return 1
    install_cudnn "${env_name}" "${HOME}/cudnn-${variant_version}" "${variant_version}"         || return 1
  # Install ROCm tools and runtime
  elif [[ "$variant_type" == "rocm" ]] && ! [[ "$(hostname)" =~ ^.*facebook.com$ ]]; then
    install_rocm_ubuntu     "${env_name}" "${variant_version}"                                  || return 1
  fi

  echo "[SETUP] Successfully created the Conda build environment (without PyTorch): ${env_name}"
  export env_name="${env_name}"
}

integration_setup_conda_environment () {
  # Sets up a Conda build environment with compilers, build tools, CUDA/ROCm
  # support, AND installs PyTorch. This leverages integration_setup_conda_environment_base
  # for the base environment setup.
  #
  # Arguments:
  #   ENV_NAME                       - Name of the Conda environment to create
  #   COMPILER                       - Compiler to use (gcc or clang)
  #   PYTHON_VERSION                 - Python version to install (e.g., 3.12)
  #   PYTORCH_CHANNEL_VERSION        - PyTorch channel/version (e.g., nightly, test/2.1.0)
  #   PYTORCH_VARIANT_TYPE_VERSION   - Variant type/version (e.g., cuda/12.9.1, rocm/7.0)
  #   PYTORCH_INSTALLER              - Installer to use: pip or conda (default: pip)
  #
  local env_name="$1"
  local compiler="$2"
  local python_version="$3"
  local pytorch_channel_version="$4"
  local pytorch_variant_type_version="$5"
  local pytorch_installer="$6"
  if [ "$pytorch_variant_type_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME COMPILER PYTHON_VERSION PYTORCH_CHANNEL[/VERSION] PYTORCH_VARIANT_TYPE/PYTORCH_VARIANT_VERSION [PYTORCH_INSTALLER]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env clang 3.14 test/1.0.0 cuda/12.8.1        # Setup environment with pytorch-test 1.0.0 for Clang + Python 3.14 + CUDA 12.8.1"
    echo "    ${FUNCNAME[0]} build_env gcc 3.12 nightly rocm/7.0 pip            # Setup environment with pytorch-nightly for GCC + Python 3.12 + ROCm 7.0"
    return 1
  else
    echo "################################################################################"
    echo "# Setup FBGEMM-GPU Build Environment (All Steps)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # Parse the variant type and version from the combined argument
  IFS='/' read -r pytorch_variant_type pytorch_variant_version <<< "${pytorch_variant_type_version}"
  echo "[INTEGRATION] Will install PyTorch (channel/version, variant): (${pytorch_channel_version}, ${pytorch_variant_type}/${pytorch_variant_version})"

  # Interpret "genai" as "cuda" for base environment setup
  local base_variant_type="${pytorch_variant_type}"
  if [ "$pytorch_variant_type" == "genai" ]; then
    echo "[INTEGRATION] Interpreting 'genai' variant as 'cuda' for base environment setup"
    base_variant_type="cuda"
  fi

  # Set up the base environment (without PyTorch) using the shared function
  integration_setup_conda_environment_base \
    "${env_name}" \
    "${compiler}" \
    "${python_version}" \
    "${base_variant_type}" \
    "${pytorch_variant_version}" \
    || return 1

  # Install PyTorch
  echo "[INTEGRATION] Installing PyTorch ..."
  if [ "$pytorch_installer" == "conda" ]; then
    install_pytorch_conda     "${env_name}" "${pytorch_channel_version}" "${pytorch_variant_type}" "${pytorch_variant_version}" || return 1
  else
    install_pytorch_pip       "${env_name}" "${pytorch_channel_version}" "${pytorch_variant_type}"/"${pytorch_variant_version}" || return 1
  fi

  echo "[INTEGRATION] Successfully created the Conda build environment: ${env_name}"
}

integration_fbgemm_gpu_build_and_install () {
  local env_name="$1"
  local build_target_variant="$2"
  local repo="$3"
  if [ "$build_target_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME BUILD_TARGET_VARIANT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env genai/cuda   # Build and install FBGEMM-GenAI for CUDA (All Steps)"
    return 1
  else
    echo "################################################################################"
    echo "# FBGEMM build + install Combo Step"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  if [ "$repo" == "" ]; then
    echo "[INTEGRATION]: repo argument not provided, defaulting to ~/FBGEMM"
    repo=~/FBGEMM
  fi

  # Assume we are starting from the repository root directory
  print_exec pushd "${repo}/fbgemm_gpu"                                       || return 1

  # Remove previous build artifacts
  print_exec rm -rf "${repo}/dist/"                                           || return 1
  prepare_fbgemm_gpu_build    "${env_name}"                                   || return 1
  build_fbgemm_gpu_package    "${env_name}" nightly "${build_target_variant}" || return 1

  print_exec popd                                                             || return 1

  # Move to another directory, to avoid Python package import confusion, since
  # there exists a fbgemm_gpu/ subdirectory in the FBGEMM repo
  print_exec mkdir -p _tmp_dir_fbgemm_gpu         || return 1
  print_exec pushd _tmp_dir_fbgemm_gpu            || return 1

  # Uninstall the FBGEMM_GPU package (if installed)
  uninstall_pip_wheel "${env_name}" "fbgemm_gpu-" || return 1

  # Install the FBGEMM_GPU package and test the package import
  # shellcheck disable=SC2086
  install_fbgemm_gpu_wheel "${env_name}" ${repo}/fbgemm_gpu/dist/*.whl || return 1

  # Return to the repo root directory
  print_exec popd                                 || return 1

  echo "[INTEGRATION] Successfully built and installed FBGEMM_GPU in the Conda environment: ${env_name}"
}

integration_fbgemm_gpu_build_and_install_and_test () {
  local env_name="$1"
  local build_target_variant="$2"
  local repo="$3"

  if [ "$repo" == "" ]; then
    echo "[INTEGRATION]: repo argument not provided, defaulting to ~/FBGEMM"
    repo=~/FBGEMM
  fi

  integration_fbgemm_gpu_build_and_install "${env_name}" "${build_target_variant}" "${repo}"  || return 1
  test_all_fbgemm_gpu_modules "${env_name}" "${repo}"                                         || return 1
}

integration_fbgemm_gpu_install_matrix_run () {
  local variant_type="$1"
  local pytorch_channel_version="$2"
  local fbgemm_gpu_channel_version="$3"
  local repo="$4"
  if [ "$fbgemm_gpu_channel_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTORCH_CHANNEL[/VERSION] FBGEMM_GPU_CHANNEL[/VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} test_env cpu test/2.2.0 test/0.8.0     # Run tests against all Python versions with PyTorch test/2.2.0 and FBGEMM_GPU test/0.8.0 (CPU-only)"
    echo "    ${FUNCNAME[0]} test_env cuda test/2.3.0 test/0.8.0    # Run tests against all Python versions with PyTorch test/2.3.0 and FBGEMM_GPU test/0.8.0 (all CUDA versions)"
    return 1
  else
    echo "################################################################################"
    echo "# Run Bulk FBGEMM-GPU Testing from PIP Package Installation"
    echo "#   (Environment Setup + Download + Install + Test)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  if [ "$repo" == "" ]; then
    echo "[INTEGRATION]: repo argument not provided, defaulting to ~/FBGEMM"
    repo=~/FBGEMM
  fi

  __single_run () {
    local py_version="$1"
    local variant_version="$2"
    local repo="$3"

    local env_name="test_py${py_version}_pytorch_${pytorch_channel_version}_fbgemm_${fbgemm_gpu_channel_version}_${variant_type}/${variant_version}"
    local env_name="${env_name//\//_}"
    integration_setup_conda_environment "${env_name}" gcc "${py_version}" "${pytorch_channel_version}" "${variant_type}"/"${variant_version}"   || return 1
    install_fbgemm_gpu_pip              "${env_name}" "${fbgemm_gpu_channel_version}" "${variant_type}/${variant_version}"                      || return 1
    test_all_fbgemm_gpu_modules         "${env_name}" "${repo}"
    local retcode=$?

    echo "################################################################################"
    echo "# RUN SUMMARY"
    echo "#"
    echo "# Conda Environment       : ${env_name}"
    echo "# Python Version          : ${py_version}"
    echo "# PyTorch Version         : ${pytorch_channel_version}"
    echo "# FBGEMM_GPU Version      : ${fbgemm_gpu_channel_version}"
    echo "# Variant type / Version  : ${variant_type}/${variant_version}"
    echo "#"
    echo "# Run Result              : $([ $retcode -eq 0 ] && echo "PASSED" || echo "FAILED")"
    echo "################################################################################"

    if [ $retcode -eq 0 ]; then
      # Clean out environment only if there were no errors
      conda remove -n "$env_name" -y --all
    fi

    cd - || return 1
    return $retcode
  }

  local python_versions=(
    3.10
    3.11
    3.12
    3.13
    3.14
  )

  if [ "$variant_type" == "cuda" ]; then
    local variant_versions=(
      12.6.3
      12.8.1
      12.9.1
      13.0.2
    )
  elif [ "$variant_type" == "genai" ]; then
    local variant_versions=(
      12.6.3
      12.8.1
      13.0.2
    )
  elif [ "$variant_type" == "rocm" ]; then
    local variant_versions=(
      7.0
      7.1
    )
  elif [ "$variant_type" == "cpu" ]; then
    local variant_versions=(
      "none"
    )
  else
    echo "[INTEGRATION] Invalid variant type: ${variant_type}"
    return 1
  fi

  for py_ver in "${python_versions[@]}"; do
    for var_ver in "${variant_versions[@]}"; do
      __single_run "${py_ver}" "${var_ver}" "${repo}" || return 1
    done
  done
}

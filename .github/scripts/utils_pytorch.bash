#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_conda.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pip.bash"

################################################################################
# PyTorch Setup Functions
################################################################################

install_pytorch_conda () {
  local env_name="$1"
  local pytorch_version="$2"
  local pytorch_variant_type="$3"
  if [ "$pytorch_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTORCH_VERSION [CPU]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 1.11.0       # Install a specific version"
    echo "    ${FUNCNAME[0]} build_env release      # Install the latest release"
    echo "    ${FUNCNAME[0]} build_env test         # Install the latest pre-release"
    echo "    ${FUNCNAME[0]} build_env nightly      # Install the latest nightly"
    return 1
  else
    echo "################################################################################"
    echo "# Install PyTorch (Conda)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # Install the cpuonly package if needed
  if [ "$pytorch_variant_type" == "cpu" ]; then
    local pytorch_package="cpuonly pytorch"
  else
    pytorch_variant_type="cuda"
    local pytorch_package="pytorch"
  fi

  # Set package name and installation channel
  if [ "$pytorch_version" == "nightly" ] || [ "$pytorch_version" == "test" ]; then
    local pytorch_channel="pytorch-${pytorch_version}"
  elif [ "$pytorch_version" == "release" ]; then
    local pytorch_channel="pytorch"
  else
    local pytorch_package="${pytorch_package}==${pytorch_version}"
    local pytorch_channel="pytorch"
  fi

  # Clean up packages before installation
  conda_cleanup

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Install PyTorch packages
  # NOTE: Installation of large package might fail due to corrupt package download
  # Use --force-reinstall to address this on retries - https://datascience.stackexchange.com/questions/41732/conda-verification-failed
  echo "[INSTALL] Attempting to install '${pytorch_package}' (${pytorch_version}, variant = ${pytorch_variant_type}) through Conda using channel '${pytorch_channel}' ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install --force-reinstall ${env_prefix} -y ${pytorch_package} -c "${pytorch_channel}") || return 1

  # Check that PyTorch is importable
  (test_python_import_package "${env_name}" torch.distributed) || return 1

  # Print out the actual installed PyTorch version
  # shellcheck disable=SC2086,SC2155
  local installed_pytorch_version=$(conda run ${env_prefix} python -c "import torch; print(torch.__version__)")
  echo "[CHECK] NOTE: The installed version is: ${installed_pytorch_version}"

  # Run check for GPU variant
  if [ "$pytorch_variant_type" == "cuda" ]; then
    # Ensure that the PyTorch build is the GPU variant (i.e. contains cuDNN reference)
    # This test usually applies to the PyTorch nightly builds
    # shellcheck disable=SC2086
    if conda list ${env_prefix} pytorch | grep cudnn; then
      echo "[CHECK] The installed PyTorch ${pytorch_version} contains references to cuDNN"
    else
      echo "[CHECK] The installed PyTorch ${pytorch_version} appears to be the CPU-only version as it is missing references to cuDNN!"
      echo "[CHECK] This can happen if the variant of PyTorch (e.g. GPU, nightly) for the MAJOR.MINOR version of CUDA presently installed on the system has not been published yet."
      echo "[CHECK] Please verify in Conda using the logged timestamp, the installed CUDA version, and the version of PyTorch that was attempted for installation:"
      echo "[CHECK]     * https://anaconda.org/pytorch-nightly/pytorch/files"
      echo "[CHECK]     * https://anaconda.org/pytorch-test/pytorch/files"
      echo "[CHECK]     * https://anaconda.org/pytorch/pytorch/files"
      return 1
    fi

    # Ensure that the PyTorch-CUDA headers are properly installed
    (test_filepath "${env_name}" cuda_cmake_macros.h) || return 1
  fi

  echo "[INSTALL] Successfully installed PyTorch through Conda"
}

install_pytorch_pip () {
  local env_name="$1"
  local pytorch_version="$2"
  local pytorch_variant_type="$3"
  local pytorch_variant_version="$4"
  if [ "$pytorch_variant_type" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTORCH_VERSION PYTORCH_VARIANT_TYPE [PYTORCH_VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 1.11.0 cpu         # Install the CPU variant for a specific version"
    echo "    ${FUNCNAME[0]} build_env release cpu        # Install the CPU variant, latest release version"
    echo "    ${FUNCNAME[0]} build_env test cuda 12.1.0   # Install the CUDA 12.1 variant, latest test version"
    echo "    ${FUNCNAME[0]} build_env nightly rocm 5.3   # Install the ROCM 5.3 variant, latest nightly version"
    return 1
  else
    echo "################################################################################"
    echo "# Install PyTorch (PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Install the package from PyTorch PIP (not PyPI)
  install_from_pytorch_pip "${env_name}" torch "${pytorch_version}" "${pytorch_variant_type}" "${pytorch_variant_version}" || return 1

  # Check that PyTorch is importable
  (test_python_import_package "${env_name}" torch.distributed) || return 1

  # Print out the actual installed PyTorch version
  # shellcheck disable=SC2086,SC2155
  local installed_pytorch_version=$(conda run ${env_prefix} python -c "import torch; print(torch.__version__)")
  echo "[CHECK] NOTE: The installed version is: ${installed_pytorch_version}"

  if [ "$pytorch_variant_type" == "cuda" ]; then
    # Ensure that the PyTorch-CUDA headers are properly installed
    (test_filepath "${env_name}" cuda_cmake_macros.h) || return 1
  fi

  echo "[INSTALL] Successfully installed PyTorch through PyTorch PIP"
}


################################################################################
# PyTorch Diagnose Functions
################################################################################

collect_pytorch_env_info () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env         # Collect PyTorch environment information from Conda environment build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Collect PyTorch Environment Information (for Reporting Issues)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # This is the script required for collecting info and reporting to https://github.com/pytorch/pytorch/issues/new
  echo "[INFO] Downloading the PyTorch environment info collection script ..."
  print_exec wget -q "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py"

  echo "[INFO] Collecting PyTorch environment info (will be needed for reporting issues to PyTorch) ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} python collect_env.py) || return 1
}

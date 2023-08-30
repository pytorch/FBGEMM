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
    echo "    ${FUNCNAME[0]} build_env latest       # Install the latest stable release"
    echo "    ${FUNCNAME[0]} build_env test         # Install the pre-release"
    echo "    ${FUNCNAME[0]} build_env nightly cpu  # Install the CPU variant of the nightly"
    return 1
  else
    echo "################################################################################"
    echo "# Install PyTorch (Conda)"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

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
  elif [ "$pytorch_version" == "latest" ]; then
    local pytorch_channel="pytorch"
  else
    local pytorch_package="${pytorch_package}==${pytorch_version}"
    local pytorch_channel="pytorch"
  fi

  # Clean up packages before installation
  conda_cleanup

  # Install PyTorch packages
  # NOTE: Installation of large package might fail due to corrupt package download
  # Use --force-reinstall to address this on retries - https://datascience.stackexchange.com/questions/41732/conda-verification-failed
  echo "[INSTALL] Attempting to install '${pytorch_package}' (${pytorch_version}, variant = ${pytorch_variant_type}) through Conda using channel '${pytorch_channel}' ..."
  # shellcheck disable=SC2086
  (exec_with_retries conda install --force-reinstall -n "${env_name}" -y ${pytorch_package} -c "${pytorch_channel}") || return 1

  # Check that PyTorch is importable
  (test_python_import "${env_name}" torch.distributed) || return 1

  # Print out the actual installed PyTorch version
  installed_pytorch_version=$(conda run -n "${env_name}" python -c "import torch; print(torch.__version__)")
  echo "[CHECK] NOTE: The installed version is: ${installed_pytorch_version}"

  # Run check for GPU variant
  if [ "$pytorch_variant_type" == "cuda" ]; then
    # Ensure that the PyTorch build is the GPU variant (i.e. contains cuDNN reference)
    # This test usually applies to the PyTorch nightly builds
    if conda list -n "${env_name}" pytorch | grep cudnn; then
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
    echo "    ${FUNCNAME[0]} build_env 1.11.0 cpu         # Install the CPU variant a specific version"
    echo "    ${FUNCNAME[0]} build_env latest cpu         # Install the CPU variant of the latest stable version"
    echo "    ${FUNCNAME[0]} build_env test cuda 11.7.1   # Install the variant for CUDA 11.7"
    echo "    ${FUNCNAME[0]} build_env nightly rocm 5.3   # Install the variant for ROCM 5.3"
    return 1
  else
    echo "################################################################################"
    echo "# Install PyTorch (PIP)"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # Set the package variant
  if [ "$pytorch_variant_type" == "cuda" ]; then
    # Extract the CUDA version or default to 11.7.1
    local cuda_version="${pytorch_variant_version:-11.7.1}"
    # shellcheck disable=SC2206
    local cuda_version_arr=(${cuda_version//./ })
    # Convert, i.e. cuda 11.7.1 => cu117
    local pytorch_variant="cu${cuda_version_arr[0]}${cuda_version_arr[1]}"
  elif [ "$pytorch_variant_type" == "rocm" ]; then
    # Extract the ROCM version or default to 5.3
    local rocm_version="${pytorch_variant_version:-5.3}"
    # shellcheck disable=SC2206
    local rocm_version_arr=(${rocm_version//./ })
    # Convert, i.e. rocm 5.5.1 => rocm5.5
    local pytorch_variant="rocm${rocm_version_arr[0]}.${rocm_version_arr[1]}"
  else
    local pytorch_variant_type="cpu"
    local pytorch_variant="cpu"
  fi
  echo "[INSTALL] Extracted PyTorch variant: ${pytorch_variant}"

  # Set the package name and installation channel
  if [ "$pytorch_version" == "nightly" ] || [ "$pytorch_version" == "test" ]; then
    local pytorch_package="--pre torch"
    local pytorch_channel="https://download.pytorch.org/whl/${pytorch_version}/${pytorch_variant}/"
  elif [ "$pytorch_version" == "latest" ]; then
    local pytorch_package="torch"
    local pytorch_channel="https://download.pytorch.org/whl/${pytorch_variant}/"
  else
    local pytorch_package="torch==${pytorch_version}+${pytorch_variant}"
    local pytorch_channel="https://download.pytorch.org/whl/${pytorch_variant}/"
  fi

  echo "[INSTALL] Attempting to install PyTorch ${pytorch_version}+${pytorch_variant} through PIP using channel ${pytorch_channel} ..."
  # shellcheck disable=SC2086
  (exec_with_retries conda run -n "${env_name}" pip install ${pytorch_package} --extra-index-url ${pytorch_channel}) || return 1

  # Check that PyTorch is importable
  (test_python_import "${env_name}" torch.distributed) || return 1

  # Print out the actual installed PyTorch version
  installed_pytorch_version=$(conda run -n "${env_name}" python -c "import torch; print(torch.__version__)")
  echo "[CHECK] NOTE: The installed version is: ${installed_pytorch_version}"

  if [ "$pytorch_variant_type" != "cpu" ]; then
    # Ensure that the PyTorch build is of the correct variant
    # This test usually applies to the PyTorch nightly builds
    if conda run -n "${env_name}" pip list torch | grep torch | grep "${pytorch_variant}"; then
      echo "[CHECK] The installed PyTorch ${pytorch_version} is the correct variant (${pytorch_variant})"
    else
      echo "[CHECK] The installed PyTorch ${pytorch_version} appears to be an incorrect variant as it is missing references to ${pytorch_variant}!"
      echo "[CHECK] This can happen if the variant of PyTorch (e.g. GPU, nightly) for the MAJOR.MINOR version of CUDA or ROCm presently installed on the system is not available."
      return 1
    fi
  fi

  if [ "$pytorch_variant_type" == "cuda" ]; then
    # Ensure that the PyTorch-CUDA headers are properly installed
    (test_filepath "${env_name}" cuda_cmake_macros.h) || return 1
  fi

  echo "[INSTALL] Successfully installed PyTorch through PIP"
}

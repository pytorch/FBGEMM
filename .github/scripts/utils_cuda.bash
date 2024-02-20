#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# CUDA Setup Functions
################################################################################

install_cuda () {
  local env_name="$1"
  local cuda_version="$2"
  if [ "$cuda_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME CUDA_VERSION"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 11.7.1"
    return 1
  else
    echo "################################################################################"
    echo "# Install CUDA"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # Check CUDA version formatting
  # shellcheck disable=SC2206
  local cuda_version_arr=(${cuda_version//./ })
  if [ ${#cuda_version_arr[@]} -lt 3 ]; then
    echo "[ERROR] CUDA minor version number must be specified (i.e. X.Y.Z)"
    return 1
  fi

  # Clean up packages before installation
  conda_cleanup

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Install CUDA packages
  echo "[INSTALL] Installing CUDA ${cuda_version} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install --force-reinstall ${env_prefix} -y cuda -c "nvidia/label/cuda-${cuda_version}") || return 1

  # Ensure that nvcc is properly installed
  (test_binpath "${env_name}" nvcc) || return 1

  # Ensure that the CUDA headers are properly installed
  (test_filepath "${env_name}" cuda_runtime.h) || return 1

  # Ensure that the libraries are properly installed
  (test_filepath "${env_name}" libnvToolsExt.so) || return 1
  (test_filepath "${env_name}" libnvidia-ml.so) || return 1

  echo "[INSTALL] Set environment variable NVML_LIB_PATH ..."
  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
  # shellcheck disable=SC2155
  local nvml_lib_path=$(find "${conda_prefix}" -name libnvidia-ml.so)
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} NVML_LIB_PATH="${nvml_lib_path}"

  # https://stackoverflow.com/questions/27686382/how-can-i-dump-all-nvcc-preprocessor-defines
  echo "[INFO] Printing out all preprocessor defines in nvcc ..."
  # shellcheck disable=SC2086
  print_exec "conda run ${env_prefix} nvcc --compiler-options -dM -E -x cu - < /dev/null"

  # Print nvcc version
  # shellcheck disable=SC2086
  print_exec conda run ${env_prefix} nvcc --version

  if which nvidia-smi; then
    # If nvidia-smi is installed on a machine without GPUs, this will return error
    (print_exec nvidia-smi) || true
  else
    echo "[CHECK] nvidia-smi not found"
  fi

  echo "[INSTALL] Successfully installed CUDA ${cuda_version}"
}

install_cudnn () {
  local env_name="$1"
  local install_path="$2"
  local cuda_version="$3"
  if [ "$cuda_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME INSTALL_PATH CUDA_VERSION"
    echo "Example:"
    echo "    ${FUNCNAME[0]} build_env \$(pwd)/cudnn_install 11.7"
    return 1
  else
    echo "################################################################################"
    echo "# Install cuDNN"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # Install cuDNN manually
  # Based on install script in https://github.com/pytorch/builder/blob/main/common/install_cuda.sh
  declare -A cudnn_packages=(
    ["115"]="https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-${PLATFORM_NAME_LC}-8.3.2.44_cuda11.5-archive.tar.xz"
    ["116"]="https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-${PLATFORM_NAME_LC}-8.3.2.44_cuda11.5-archive.tar.xz"
    ["117"]="https://ossci-linux.s3.amazonaws.com/cudnn-${PLATFORM_NAME_LC}-8.5.0.96_cuda11-archive.tar.xz"
    ["118"]="https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-${PLATFORM_NAME_LC}-8.7.0.84_cuda11-archive.tar.xz"
    ["121"]="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.8.1.3_cuda12-archive.tar.xz"
  )

  # Split version string by dot into array, i.e. 11.7.1 => [11, 7, 1]
  # shellcheck disable=SC2206
  local cuda_version_arr=(${cuda_version//./ })
  # Fetch the major and minor version to concat
  local cuda_concat_version="${cuda_version_arr[0]}${cuda_version_arr[1]}"

  # Get the URL
  local cudnn_url="${cudnn_packages[$cuda_concat_version]}"
  if [ "$cudnn_url" == "" ]; then
    # Default to cuDNN for 11.8 if no CUDA version fits
    echo "[INSTALL] Defaulting to cuDNN for CUDA 11.8"
    cudnn_url="${cudnn_packages[118]}"
  fi

  # Clear the install path
  rm -rf "$install_path"
  mkdir -p "$install_path"

  # Create temporary directory
  # shellcheck disable=SC2155
  local tmp_dir=$(mktemp -d)
  cd "$tmp_dir" || return 1

  # Download cuDNN
  echo "[INSTALL] Downloading cuDNN to ${tmp_dir} ..."
  (exec_with_retries 3 wget -q "$cudnn_url" -O cudnn.tar.xz) || return 1

  # Unpack the tarball
  echo "[INSTALL] Unpacking cuDNN ..."
  tar -xvf cudnn.tar.xz

  # Copy the includes and libs over to the install path
  echo "[INSTALL] Moving cuDNN files to ${install_path} ..."
  rm -rf "${install_path:?}/include"
  rm -rf "${install_path:?}/lib"
  mv cudnn-linux-*/include "$install_path"
  mv cudnn-linux-*/lib "$install_path"

  # Delete the temporary directory
  cd - || return 1
  rm -rf "$tmp_dir"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Export the environment variables to the Conda environment
  echo "[INSTALL] Set environment variables CUDNN_INCLUDE_DIR and CUDNN_LIBRARY ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} CUDNN_INCLUDE_DIR="${install_path}/include" CUDNN_LIBRARY="${install_path}/lib"

  echo "[INSTALL] Successfully installed cuDNN (for CUDA ${cuda_version})"
}

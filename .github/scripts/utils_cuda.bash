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

__set_cuda_symlinks_envvars () {
  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
  local new_cuda_home="${conda_prefix}/targets/${MACHINE_NAME_LC}-linux"

  if  [[ "$BUILD_CUDA_VERSION" =~ ^12.6.*$ ]] ||
      [[ "$BUILD_CUDA_VERSION" =~ ^12.8.*$ ]]; then
    # CUDA 12.6 installation has a very different package layout than previous
    # CUDA versions - notably, NVTX has been moved elsewhere, which causes
    # PyTorch CMake scripts to complain.
    echo "[INSTALL] Fixing file placements for CUDA ${BUILD_CUDA_VERSION}+ ..."

    echo "[INSTALL] Creating symlinks: libnvToolsExt.so"
    print_exec ln -sf "${conda_prefix}/lib/libnvToolsExt.so.1" "${conda_prefix}/lib/libnvToolsExt.so"
    print_exec ln -sf "${new_cuda_home}/lib/libnvToolsExt.so.1" "${new_cuda_home}/lib/libnvToolsExt.so"

    echo "[INSTALL] Copying nvtx3 headers ..."
    # shellcheck disable=SC2086
    print_exec cp -r ${conda_prefix}/nsight-compute*/host/*/nvtx/include/nvtx3/* ${conda_prefix}/include/
    # shellcheck disable=SC2086
    print_exec cp -r ${conda_prefix}/nsight-compute*/host/*/nvtx/include/nvtx3/* ${new_cuda_home}/include/
  fi

  echo "[INSTALL] Appending libcuda.so path to LD_LIBRARY_PATH ..."
  # shellcheck disable=SC2155
  local libcuda_path=$(find "${conda_prefix}" -type f -name libcuda.so)
  append_to_library_path "${env_name}" "$(dirname "$libcuda_path")"

  # The symlink appears to be missing when we attempt to run FBGEMM_GPU on the
  # `ubuntu-latest` runners on GitHub, so we have to manually add this in.
  if [ "$ADD_LIBCUDA_SYMLINK" == "1" ]; then
    echo "[INSTALL] Setting up symlink to libcuda.so.1"
    print_exec ln "${libcuda_path}" -s "$(dirname "$libcuda_path")/libcuda.so.1"
  fi

  echo "[INSTALL] Setting environment variable NVML_LIB_PATH ..."
  # shellcheck disable=SC2155
  local libnvml_path=$(find "${conda_prefix}" -name libnvidia-ml.so | head -n1)
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} NVML_LIB_PATH="${libnvml_path}"

  if [ "$ADD_LIBCUDA_SYMLINK" == "1" ]; then
    echo "[INSTALL] Setting up symlink to libnvidia-ml.so.1"
    print_exec ln "${libnvml_path}" -s "${conda_prefix}/lib/libnvidia-ml.so.1"
  fi

  echo "[INSTALL] Setting environment variable CUDA_INCLUDE_DIRS ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} CUDA_INCLUDE_DIRS=\""${conda_prefix}/include/:${new_cuda_home}/include/"\"

  # Ensure that the CUDA headers are properly installed
  (test_filepath "${env_name}" cuda_runtime.h) || return 1
  # Ensure that the libraries are properly installed
  (test_filepath "${env_name}" libcuda.so) || return 1
  (test_filepath "${env_name}" libnvToolsExt.so) || return 1
  (test_filepath "${env_name}" libnvidia-ml.so) || return 1

  # Ensure that nvcc is properly installed
  (test_binpath "${env_name}" nvcc) || return 1
}

__set_nvcc_prepend_flags () {
  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)

  # If clang is available, but CUDA was installed through conda-forge, the
  # cc/c++ symlinks will be reset to gcc/g++, so fix this first
  # shellcheck disable=SC2155,SC2086
  if conda run ${env_prefix} clang --version; then
    echo "[INSTALL] Resetting compiler symlinks to clang ..."
    set_clang_symlinks "${env_name}"
  fi

  # The NVCC activation scripts append `-ccbin=${CXX}`` to NVCC_PREPEND_FLAGS,
  # which overrides whatever `-ccbin` flag we set manually, so remove this
  # unwanted hook
  print_exec ls -la "${conda_prefix}/etc/conda/activate.d"
  if  [[ "$BUILD_CUDA_VERSION" =~ ^12.6.*$ ]] ||
      [[ "$BUILD_CUDA_VERSION" =~ ^12.8.*$ ]]; then
    echo "[INSTALL] Removing the -ccbin=CXX hook from NVCC activation scripts ..."
    print_exec sed -i '/-ccbin=/d' "${conda_prefix}/etc/conda/activate.d/*cuda-nvcc_activate.sh"
  fi

  local nvcc_prepend_flags=(
    # Allow for the use of newer compilers than what the current CUDA SDK
    # supports
    -allow-unsupported-compiler
  )

  if print_exec "conda run ${env_prefix} c++ --version | grep -i clang"; then
    # Explicitly set whatever $CONDA_PREFIX/bin/c++ points to as the the host
    # compiler, but set GNU libstdc++ (as opposed to Clang libc++) as the
    # standard library.
    #
    # NOTE: NVCC_PREPEND_FLAGS is set here to allow the nvcc install check to
    # pass.  It will be overridden to include more compilation flags during the
    # FBGEMM_GPU build stage.
    #
    # NOTE: There appears to be no ROCm equivalent for NVCC_PREPEND_FLAGS:
    #   https://github.com/ROCm/HIP/issues/931
    #
    echo "[BUILD] Setting Clang as the NVCC host compiler: ${cxx_path}"

    # shellcheck disable=SC2155,SC2086
    local cxx_path=$(conda run ${env_prefix} which c++)

    nvcc_prepend_flags+=(
      -Xcompiler -stdlib=libstdc++
      -ccbin "${cxx_path}"
    )
  fi

  echo "[BUILD] Setting prepend flags for NVCC ..."
  # shellcheck disable=SC2086,SC2145
  print_exec conda env config vars set ${env_prefix} NVCC_PREPEND_FLAGS=\""${nvcc_prepend_flags[@]}"\"
  # shellcheck disable=SC2086
  print_exec conda run ${env_prefix} printenv NVCC_PREPEND_FLAGS
}

__print_cuda_info () {
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
}

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
  echo "[INSTALL] Installing CUDA ${cuda_version} ..."

  # NOTE: Currently, CUDA 12.6 cannot be installed using the nvidia/label/cuda-*
  # conda channels, because we run into the following error:
  #
  #   LibMambaUnsatisfiableError: Encountered problems while solving:
  #     - nothing provides __win needed by cuda-12.6.3-0
  #
  # For now, we only use conda-forge for installing 12.6, but it is likely that
  # in the future, we will be using conda-forge for installing all CUDA versions
  # (except for versions 11.8 and below, which are only available through
  # nvidia/label/cuda-*)
  if  [[ "$cuda_version" =~ ^12.6.*$ ]] ||
      [[ "$cuda_version" =~ ^12.8.*$ ]]; then
    # shellcheck disable=SC2086
    (exec_with_retries 3 conda install --force-reinstall ${env_prefix} -c conda-forge --override-channels -y \
      cuda=${cuda_version}) || return 1
  else
    # shellcheck disable=SC2086
    (exec_with_retries 3 conda install --force-reinstall ${env_prefix} -c "nvidia/label/cuda-${cuda_version}" -y \
      cuda) || return 1
  fi

  # Set the symlinks and environment variables not covered by conda install
  __set_cuda_symlinks_envvars

  # Set the NVCC prepend flags depending on gcc or clang
  __set_nvcc_prepend_flags

  # Print debug info
  __print_cuda_info

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
    ["121"]="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/${PLATFORM_NAME_LC}/cudnn-${PLATFORM_NAME_LC}-8.9.2.26_cuda12-archive.tar.xz"
    ["124"]="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/${PLATFORM_NAME_LC}/cudnn-${PLATFORM_NAME_LC}-8.9.2.26_cuda12-archive.tar.xz"
    ["126"]="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/${PLATFORM_NAME_LC}/cudnn-${PLATFORM_NAME_LC}-9.5.1.17_cuda12-archive.tar.xz"
  )

  # Split version string by dot into array, i.e. 11.7.1 => [11, 7, 1]
  # shellcheck disable=SC2206
  local cuda_version_arr=(${cuda_version//./ })
  # Fetch the major and minor version to concat
  local cuda_concat_version="${cuda_version_arr[0]}${cuda_version_arr[1]}"
  echo "[INSTALL] cuda_concat_version is determined to be: ${cuda_concat_version}"

  # Get the URL
  local cudnn_url="${cudnn_packages[$cuda_concat_version]}"
  if [ "$cudnn_url" == "" ]; then
    # Default to cuDNN for 11.8 if no CUDA version fits
    echo "[INSTALL] Could not find cuDNN URL for the given cuda_concat_version ${cuda_concat_version}; defaulting to cuDNN for CUDA 11.8"
    cudnn_url="${cudnn_packages[118]}"
  fi

  # Clear the install path
  print_exec rm -rf "$install_path"
  print_exec mkdir -p "$install_path"

  # Create temporary directory
  # shellcheck disable=SC2155
  local tmp_dir=$(mktemp -d)
  cd "$tmp_dir" || return 1

  # Download cuDNN
  echo "[INSTALL] Downloading cuDNN to ${tmp_dir} ..."
  (exec_with_retries 3 wget -q "$cudnn_url" -O cudnn.tar.xz) || return 1

  # Unpack the tarball
  echo "[INSTALL] Unpacking cuDNN ..."
  print_exec tar -xvf cudnn.tar.xz

  # Copy the includes and libs over to the install path
  echo "[INSTALL] Moving cuDNN files to ${install_path} ..."
  print_exec rm -rf "${install_path:?}/include"
  print_exec rm -rf "${install_path:?}/lib"
  print_exec mv cudnn-linux-*/include "$install_path"
  print_exec mv cudnn-linux-*/lib "$install_path"

  # Delete the temporary directory
  cd - || return 1
  print_exec rm -rf "$tmp_dir"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Export the environment variables to the Conda environment
  echo "[INSTALL] Set environment variables CUDNN_INCLUDE_DIR and CUDNN_LIBRARY ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} CUDNN_INCLUDE_DIR="${install_path}/include" CUDNN_LIBRARY="${install_path}/lib"

  echo "[INSTALL] Successfully installed cuDNN (for CUDA ${cuda_version})"
}

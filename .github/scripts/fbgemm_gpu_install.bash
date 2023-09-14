#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# FBGEMM_GPU Install Functions
################################################################################

__fbgemm_gpu_post_install_checks () {
  echo "[INSTALL] Checking imports and symbols ..."
  (test_python_import_package "${env_name}" fbgemm_gpu) || return 1
  (test_python_import_package "${env_name}" fbgemm_gpu.split_embedding_codegen_lookup_invokers) || return 1
  (test_python_import_symbol "${env_name}" fbgemm_gpu __version__) || return 1

  echo "[CHECK] Printing out the FBGEMM-GPU version ..."
  installed_fbgemm_gpu_version=$(conda run -n "${env_name}" python -c "import fbgemm_gpu; print(fbgemm_gpu.__version__)")
  echo "[CHECK] The installed version is: ${installed_fbgemm_gpu_version}"
}

install_fbgemm_gpu_wheel () {
  local env_name="$1"
  local wheel_path="$2"
  if [ "$wheel_path" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME WHEEL_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu.whl     # Install the package (wheel)"
    return 1
  else
    echo "################################################################################"
    echo "# Install FBGEMM-GPU from Wheel"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "[INSTALL] Printing out FBGEMM-GPU wheel SHA: ${wheel_path}"
  print_exec sha1sum "${wheel_path}"
  print_exec sha256sum "${wheel_path}"
  print_exec md5sum "${wheel_path}"

  echo "[INSTALL] Installing FBGEMM-GPU wheel: ${wheel_path} ..."
  (exec_with_retries conda run -n "${env_name}" python -m pip install "${wheel_path}") || return 1

  __fbgemm_gpu_post_install_checks || return 1

  echo "[INSTALL] FBGEMM-GPU installation through wheel completed ..."
}

install_fbgemm_gpu_pip () {
  local env_name="$1"
  local fbgemm_gpu_version="$2"
  local fbgemm_gpu_variant_type="$3"
  local fbgemm_gpu_variant_version="$4"
  if [ "$fbgemm_gpu_variant_type" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME FBGEMM_GPU_VERSION FBGEMM_GPU_VARIANT_TYPE [FBGEMM_GPU_VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 0.5.0rc2 cuda 12.1.1        # Install a specific version of the package (PyPI)"
    return 1
  else
    echo "################################################################################"
    echo "# Install FBGEMM-GPU Package from PIP"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # Set the package variant
  if [ "$fbgemm_gpu_variant_type" == "cuda" ]; then
    # Extract the CUDA version or default to 11.8.0
    local cuda_version="${fbgemm_gpu_variant_version:-11.8.0}"
    # shellcheck disable=SC2206
    local cuda_version_arr=(${cuda_version//./ })
    # Convert, i.e. cuda 11.7.1 => cu117
    local fbgemm_gpu_variant="cu${cuda_version_arr[0]}${cuda_version_arr[1]}"
  elif [ "$fbgemm_gpu_variant_type" == "rocm" ]; then
    # Extract the ROCM version or default to 5.5.1
    local rocm_version="${fbgemm_gpu_variant_version:-5.5.1}"
    # shellcheck disable=SC2206
    local rocm_version_arr=(${rocm_version//./ })
    # Convert, i.e. rocm 5.5.1 => rocm5.5
    local fbgemm_gpu_variant="rocm${rocm_version_arr[0]}.${rocm_version_arr[1]}"
  else
    local fbgemm_gpu_variant_type="cpu"
    local fbgemm_gpu_variant="cpu"
  fi
  echo "[INSTALL] Extracted FBGEMM-GPU variant: ${fbgemm_gpu_variant}"

  # Set the package name and installation channel
#   if [ "$fbgemm_gpu_version" == "nightly" ] || [ "$fbgemm_gpu_version" == "test" ]; then
#     local fbgemm_gpu_package="--pre fbgemm-gpu"
#     local fbgemm_gpu_channel="https://download.pytorch.org/whl/${fbgemm_gpu_version}/${fbgemm_gpu_variant}/"
#   elif [ "$fbgemm_gpu_version" == "latest" ]; then
#     local fbgemm_gpu_package="fbgemm-gpu"
#     local fbgemm_gpu_channel="https://download.pytorch.org/whl/${fbgemm_gpu_variant}/"
#   else
#     local fbgemm_gpu_package="fbgemm-gpu==${fbgemm_gpu_version}+${fbgemm_gpu_variant}"
#     local fbgemm_gpu_channel="https://download.pytorch.org/whl/${fbgemm_gpu_variant}/"
#   fi

  if [ "$fbgemm_gpu_variant_type" == "cuda" ]; then
    if [ "$fbgemm_gpu_version" == "nightly" ]; then
      local fbgemm_gpu_package="fbgemm-gpu-nightly"
    elif [ "$fbgemm_gpu_version" == "latest" ]; then
      local fbgemm_gpu_package="fbgemm-gpu"
    else
      local fbgemm_gpu_package="fbgemm-gpu==${fbgemm_gpu_version}"
    fi

  elif [ "$fbgemm_gpu_variant_type" == "rocm" ]; then
    echo "ROCm is currently not supported in PyPI!"
    return 1

  else
    if [ "$fbgemm_gpu_version" == "nightly" ]; then
      local fbgemm_gpu_package="fbgemm-gpu-nightly-cpu"
    elif [ "$fbgemm_gpu_version" == "latest" ]; then
      local fbgemm_gpu_package="fbgemm-gpu-cpu"
    else
      local fbgemm_gpu_package="fbgemm-gpu-cpu==${fbgemm_gpu_version}"
    fi
  fi

  echo "[INSTALL] Attempting to install FBGEMM-GPU ${fbgemm_gpu_version}+${fbgemm_gpu_variant} through PIP ..."
  # shellcheck disable=SC2086
  (exec_with_retries conda run -n "${env_name}" pip install ${fbgemm_gpu_package}) || return 1

  __fbgemm_gpu_post_install_checks || return 1

  echo "[INSTALL] FBGEMM-GPU installation through PIP completed ..."
}

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

################################################################################
# FBGEMM_GPU Install Functions
################################################################################

__fbgemm_gpu_post_install_checks () {
  local env_name="$1"
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Checking imports and symbols ..."
  (test_python_import_package "${env_name}" fbgemm_gpu) || return 1
  (test_python_import_package "${env_name}" fbgemm_gpu.split_embedding_codegen_lookup_invokers) || return 1
  (test_python_import_symbol "${env_name}" fbgemm_gpu __version__) || return 1

  echo "[CHECK] Printing out the FBGEMM-GPU version ..."
  # shellcheck disable=SC2086,SC2155
  local installed_fbgemm_gpu_version=$(conda run ${env_prefix} python -c "import fbgemm_gpu; print(fbgemm_gpu.__version__)")
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
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  echo "[INSTALL] Printing out FBGEMM-GPU wheel SHA: ${wheel_path}"
  print_exec sha1sum "${wheel_path}"
  print_exec sha256sum "${wheel_path}"
  print_exec md5sum "${wheel_path}"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Installing FBGEMM-GPU wheel: ${wheel_path} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} python -m pip install "${wheel_path}") || return 1

  __fbgemm_gpu_post_install_checks "${env_name}" || return 1

  echo "[INSTALL] FBGEMM-GPU installation through wheel completed ..."
}

install_fbgemm_gpu_pip () {
  local env_name="$1"
  local fbgemm_gpu_channel_version="$2"
  local fbgemm_gpu_variant_type_version="$3"
  if [ "$fbgemm_gpu_variant_type_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME FBGEMM_GPU_CHANNEL[/VERSION] FBGEMM_GPU_VARIANT_TYPE[/VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 0.5.0 cpu                  # Install the CPU variant, specific version from release channel"
    echo "    ${FUNCNAME[0]} build_env release cuda 12.1.1        # Install the CUDA variant, latest version from release channel"
    echo "    ${FUNCNAME[0]} build_env test/0.6.0rc0 cuda 12.1.0  # Install the CUDA 12.1 variant, specific version from test channel"
    echo "    ${FUNCNAME[0]} build_env nightly rocm 5.3           # Install the ROCM 5.3 variant, latest version from nightly channel"
    return 1
  else
    echo "################################################################################"
    echo "# Install FBGEMM-GPU Package from PIP"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # Install the package from PyTorch PIP (not PyPI)
  # The package's canonical name is 'fbgemm-gpu' (hyphen, not underscore)
  install_from_pytorch_pip "${env_name}" fbgemm_gpu "${fbgemm_gpu_channel_version}" "${fbgemm_gpu_variant_type_version}" || return 1

  # Run post-installation checks
  __fbgemm_gpu_post_install_checks "${env_name}" || return 1

  echo "[INSTALL] Successfully installed FBGEMM-GPU through PyTorch PIP"
}

#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_system.bash"

################################################################################
# ROCm Setup Functions
################################################################################

install_rocm_amdsmi_ubuntu () {
  # Manually install AMD SMI to work around missing package error
  # https://github.com/pytorch/pytorch/pull/119182/
  local env_name="$1"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Installing roctracer-dev and amd-smi-lib ..."
  (apt-get install -y --allow-unauthenticated \
      roctracer-dev \
      amd-smi-lib) || return 1

  echo "[INSTALL] Installing amd-smi ..."
  cd /opt/rocm/share/amd_smi || return 1
  # shellcheck disable=SC2086
  (conda run ${env_prefix} pip install .) || return 1

  echo "[INSTALL] Checking Python imports for amd-smi ..."
  (test_python_import_package "${env_name}" amdsmi) || return 1
}

install_rocm_ubuntu () {
  # shellcheck disable=SC2034
  local env_name="$1"
  local rocm_version="$2"
  if [ "$rocm_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME ROC_VERSION"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 5.4.3"
    return 1
  else
    echo "################################################################################"
    echo "# Install ROCm (Ubuntu)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # Based on instructions found in https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.3/page/How_to_Install_ROCm.html

  # Disable CLI prompts during package installation
  export DEBIAN_FRONTEND=noninteractive

  echo "[INSTALL] Loading OS release info to fetch VERSION_CODENAME ..."
  # shellcheck disable=SC1091
  . /etc/os-release

  # Split version string by dot into array, i.e. 5.4.3 => [5, 4, 3]
  # shellcheck disable=SC2206,SC2155
  local rocm_version_arr=(${rocm_version//./ })
  # Materialize the long version string, i.e. 5.3 => 50500, 5.4.3 => 50403
  # shellcheck disable=SC2155
  local long_version="${rocm_version_arr[0]}$(printf %02d "${rocm_version_arr[1]}")$(printf %02d "${rocm_version_arr[2]}")"
  # Materialize the full deb package name
  local package_name="amdgpu-install_${rocm_version_arr[0]}.${rocm_version_arr[1]}.${long_version}-1_all.deb"
  # Materialize the download URL
  local rocm_download_url="https://repo.radeon.com/amdgpu-install/${rocm_version}/ubuntu/${VERSION_CODENAME}/${package_name}"

  echo "[INSTALL] Downloading the ROCm installer script ..."
  print_exec wget -q "${rocm_download_url}" -O "${package_name}"

  echo "[INSTALL] Installing the ROCm installer script ..."
  (install_system_packages "./${package_name}") || return 1

  # Skip installation of kernel driver when run in Docker mode with --no-dkms
  echo "[INSTALL] Installing ROCm ..."
  (exec_with_retries 3 amdgpu-install -y --usecase=hiplibsdk,rocm,rocmdevtools --no-dkms) || return 1

  echo "[INSTALL] Installing HIP-relevant packages ..."
  (install_system_packages hipify-clang miopen-hip miopen-hip-dev) || return 1

  # There is no need to install these packages for ROCm
  # install_system_packages mesa-common-dev clang comgr libopenblas-dev jp intel-mkl-full locales libnuma-dev

  # Fix known issue of amdsmi error  https://github.com/pytorch/pytorch/pull/119182/
  install_rocm_amdsmi_ubuntu "$env_name" || return 1

  echo "[INSTALL] Cleaning up ..."
  print_exec rm -f "${package_name}"

  echo "[INFO] Check ROCM GPU info ..."
  # If rocm-smi is installed on a machine without GPUs, this will return error
  (print_exec rocminfo) || true
  print_exec rocm-smi

  echo "[INSTALL] Successfully installed ROCm ${rocm_version}"
}

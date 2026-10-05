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

rocm_install_dir () {
  # Print the location of the ROCm installation: ROCM_PATH if set, else
  # /opt/rocm if it exists, else as reported by hipconfig.  hipconfig comes last
  # because PyTorch's ROCm wheels can install a hipconfig of their own, for a
  # ROCm runtime that is not a full ROCm installation.
  if [ -n "${ROCM_PATH:-}" ]; then
    echo "${ROCM_PATH}"
  elif [ -d /opt/rocm ]; then
    echo /opt/rocm
  elif command -v hipconfig > /dev/null 2>&1; then
    hipconfig --rocmpath
  else
    echo "[ROCM] Unable to locate ROCm: ROCM_PATH is not set, /opt/rocm does not exist, and hipconfig is not on PATH" >&2
    return 1
  fi
}

install_rocm_pip () {
  # Install ROCm into the Conda env from the AMD wheel index.  The version is
  # the CI matrix value (for example 7.14 or 10.0), not a fixed release.
  local env_name="$1"
  local rocm_version="$2"
  if [ "$rocm_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME ROCM_VERSION"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 10.0"
    return 1
  else
    echo "################################################################################"
    echo "# Install ROCm (PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # 7.14 is published on the multi-arch index. 10.x is published on whl-next.
  # The version itself still comes from the CI matrix.
  local rocm_index="https://stable.repo.amd.com/rocm/core/whl-next/"
  case "${rocm_version}" in
    7.14*)
      rocm_index="https://repo.amd.com/rocm/whl-multi-arch/"
      ;;
  esac

  echo "[INSTALL] Installing ROCm ${rocm_version} from ${rocm_index} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} python -m pip install \
    --index-url "${rocm_index}" \
    "rocm[devel]==${rocm_version}") || return 1

  echo "[INSTALL] Locating the ROCm wheel install ..."
  local rocm_dir
  # shellcheck disable=SC2086
  rocm_dir=$(conda run ${env_prefix} python -c 'import importlib.util; from pathlib import Path; spec = importlib.util.find_spec("_rocm_sdk_core");
if spec is None or not spec.origin:
    raise SystemExit("ROCm wheel did not install _rocm_sdk_core")
print(Path(spec.origin).parent.resolve())' | tail -n 1) || return 1
  if [ -z "${rocm_dir}" ] || [ ! -d "${rocm_dir}" ]; then
    echo "[INSTALL] Unable to locate the ROCm ${rocm_version} wheel install" >&2
    return 1
  fi

  echo "[INSTALL] Setting ROCM_PATH=${rocm_dir}"
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} ROCM_PATH="${rocm_dir}"
  export ROCM_PATH="${rocm_dir}"
  # Later job steps start a new shell, so publish the path for those too.
  if [ -n "${GITHUB_ENV:-}" ]; then
    echo "ROCM_PATH=${rocm_dir}" >> "${GITHUB_ENV}"
  fi

  echo "[INSTALL] Successfully installed ROCm ${rocm_version}"
}

install_rocm_amdsmi_ubuntu () {
  # Manually install AMD SMI to work around missing package error
  # https://github.com/pytorch/pytorch/pull/119182/
  local env_name="$1"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")
  local rocm_dir
  rocm_dir=$(rocm_install_dir) || return 1

  echo "[INSTALL] Installing libdw ..."
  (apt-get install -y --allow-unauthenticated libdw-dev) || return 1

  if [ -d "${rocm_dir}/share/amd_smi" ]; then
    echo "[INSTALL] Installing amd-smi from ${rocm_dir} ..."
    # shellcheck disable=SC2086
    (conda run ${env_prefix} python -m pip install "${rocm_dir}/share/amd_smi") || return 1
  else
    echo "[INSTALL] ${rocm_dir}/share/amd_smi is not present; expecting amd-smi from the ROCm wheel"
  fi

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

  echo "[INSTALL] Installing roctracer-dev and amd-smi-lib ..."
  (apt-get install -y --allow-unauthenticated roctracer-dev amd-smi-lib) || return 1

  # There is no need to install these packages for ROCm
  # install_system_packages mesa-common-dev clang comgr libopenblas-dev jp intel-mkl-full locales libnuma-dev

  # Fix known issue of amdsmi error  https://github.com/pytorch/pytorch/pull/119182/
  install_rocm_amdsmi_ubuntu "$env_name" || return 1

  echo "[INSTALL] Cleaning up ..."
  print_exec rm -f "${package_name}"

  echo "[INFO] Printing AMD-SMI utilities info ..."
  # If amd-smi is installed on a machine without GPUs, this will return error
  (print_exec amd-smi) || true
  (print_exec hipcc -v) || true

  echo "[INSTALL] Successfully installed ROCm ${rocm_version}"
}

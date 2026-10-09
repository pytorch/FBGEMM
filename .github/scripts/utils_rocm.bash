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
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pip.bash"

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
  # Install the ROCm devel extra from the same PyTorch wheel index as torch.
  # nightly + rocm/7.14 resolves to
  # https://download.pytorch.org/whl/nightly/rocm7.14/. The matrix version
  # selects that index. The package pin is the rocm version torch already
  # installed, so the devel tree matches torch's runtime patch.
  local env_name="$1"
  local rocm_version="$2"
  local pytorch_channel_version="${3:-nightly}"
  if [ "$rocm_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME ROCM_VERSION [PYTORCH_CHANNEL[/VERSION]]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 7.14 nightly"
    echo "    ${FUNCNAME[0]} build_env 10.0 nightly"
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

  # Same index construction as install_pytorch_pip. Do not use the resulting
  # pip_package: that omits the [devel] extra and would append +rocm<version>.
  __prepare_pip_arguments "rocm" "${pytorch_channel_version}" "rocm/${rocm_version}" || return 1
  local pip_pre=""
  if [ "${package_channel}" != "release" ]; then
    pip_pre="--pre"
  fi

  echo "[INSTALL] Installing ROCm ${rocm_installed} devel from ${pip_channel} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} python -m pip install \
    ${pip_pre} \
    --index-url "${pip_channel}" \
    "rocm[devel]") || return 1

  # Headers and lib/cmake are packed in rocm-sdk-devel and only appear after
  # that tree is expanded. rocm-sdk path --root expands it and prints the root
  # CMake uses to find HIP.
  echo "[INSTALL] Expanding the ROCm devel tree ..."
  local rocm_dir
  # shellcheck disable=SC2086
  rocm_dir=$(conda run ${env_prefix} rocm-sdk path --root | tail -n 1) || return 1
  if [ -z "${rocm_dir}" ] || [ ! -d "${rocm_dir}/lib/cmake/hip" ]; then
    echo "[INSTALL] ROCm devel tree is missing lib/cmake/hip: ${rocm_dir:-<empty>}" >&2
    return 1
  fi
  if [ ! -e "${rocm_dir}/bin/hipcc" ] || [ ! -e "${rocm_dir}/bin/amd-smi" ]; then
    echo "[INSTALL] ROCm devel tree is missing hipcc or amd-smi: ${rocm_dir}/bin" >&2
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
  # The wheel's libamd_smi is versioned, and later steps load it by the
  # unversioned name. amd-smi and hipcc are not on PATH until published.
  publish_rocm_library_path "${env_name}" "${rocm_dir}" || return 1
  publish_rocm_bin_path "${rocm_dir}" || return 1

  echo "[INSTALL] Successfully installed ROCm ${rocm_installed}"
}

ensure_libamd_smi_soname () {
  # ROCm wheels ship libamd_smi.so.<version>. amdsmi looks up libamd_smi.so.
  local rocm_dir="$1"
  local lib_dir="${rocm_dir}/lib"
  local soname="${lib_dir}/libamd_smi.so"
  if [ -e "${soname}" ]; then
    return 0
  fi
  if [ ! -d "${lib_dir}" ]; then
    echo "[INSTALL] ROCm lib directory does not exist: ${lib_dir}" >&2
    return 1
  fi

  local versioned
  versioned=$(find "${lib_dir}" -maxdepth 1 -name 'libamd_smi.so.*' -printf '%p\n' | sort -V | tail -n 1)
  if [ -z "${versioned}" ]; then
    echo "[INSTALL] libamd_smi.so was not found under ${lib_dir}" >&2
    return 1
  fi

  echo "[INSTALL] Linking ${soname} -> $(basename "${versioned}")"
  ln -s "$(basename "${versioned}")" "${soname}"
}

publish_rocm_library_path () {
  local env_name="$1"
  local rocm_dir="$2"
  local lib_dir="${rocm_dir}/lib"

  ensure_libamd_smi_soname "${rocm_dir}" || return 1

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")
  # shellcheck disable=SC2155,SC2086
  local current_library_path=$(conda run ${env_prefix} printenv LD_LIBRARY_PATH 2>/dev/null || true)
  case ":${current_library_path}:" in
    *":${lib_dir}:"*) ;;
    *)
      (append_to_library_path "${env_name}" "${lib_dir}") || return 1
      ;;
  esac

  case ":${LD_LIBRARY_PATH:-}:" in
    *":${lib_dir}:"*) ;;
    *)
      export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      ;;
  esac
  if [ -n "${GITHUB_ENV:-}" ]; then
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> "${GITHUB_ENV}"
  fi
}

publish_rocm_bin_path () {
  # amd-smi resolves ../libexec relative to the script path, so the real bin
  # directory has to come first when the devel tree only symlinks bin/amd-smi.
  local rocm_dir="$1"
  local bin_dir="${rocm_dir}/bin"
  if [ ! -e "${bin_dir}/amd-smi" ]; then
    echo "[INSTALL] amd-smi is not at ${bin_dir}; leaving PATH unchanged"
    return 0
  fi
  local real_bin
  real_bin=$(dirname "$(readlink -f "${bin_dir}/amd-smi")")

  if [ "${real_bin}" != "${bin_dir}" ]; then
    publish_one_bin_path "${bin_dir}" || return 1
  fi
  publish_one_bin_path "${real_bin}" || return 1
}

publish_one_bin_path () {
  local bin_dir="$1"
  case ":${PATH}:" in
    *":${bin_dir}:"*) ;;
    *)
      export PATH="${bin_dir}:${PATH}"
      ;;
  esac
  # GITHUB_PATH is prepended for later steps. Repeating a directory is harmless.
  if [ -n "${GITHUB_PATH:-}" ]; then
    echo "${bin_dir}" >> "${GITHUB_PATH}"
  fi
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

  # The pip-installed amdsmi package looks for libamd_smi.so next to itself
  # and under ${ROCM_PATH}/lib. The wheel only provides the versioned library.
  publish_rocm_library_path "${env_name}" "${rocm_dir}" || return 1
  publish_rocm_bin_path "${rocm_dir}" || return 1

  local amd_smi_src="${rocm_dir}/share/amd_smi"
  if [ -d "${amd_smi_src}" ]; then
    echo "[INSTALL] Installing amd-smi from ${amd_smi_src} ..."
    # shellcheck disable=SC2086
    (conda run ${env_prefix} python -m pip install "${amd_smi_src}") || return 1
  else
    echo "[INSTALL] ${amd_smi_src} is not present; expecting amd-smi from the ROCm wheel"
  fi

  local amdsmi_dir
  # shellcheck disable=SC2086
  amdsmi_dir=$(conda run ${env_prefix} python -c 'import importlib.util; from pathlib import Path; spec = importlib.util.find_spec("amdsmi");
if spec is None or not spec.origin:
    raise SystemExit("amdsmi was not installed")
print(Path(spec.origin).parent.resolve())' | tail -n 1) || return 1
  if [ ! -e "${amdsmi_dir}/libamd_smi.so" ]; then
    echo "[INSTALL] Linking ${amdsmi_dir}/libamd_smi.so -> ${rocm_dir}/lib/libamd_smi.so"
    ln -s "${rocm_dir}/lib/libamd_smi.so" "${amdsmi_dir}/libamd_smi.so" || return 1
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

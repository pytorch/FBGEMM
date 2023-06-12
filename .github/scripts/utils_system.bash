#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# System Functions
################################################################################

install_system_packages () {
  if [ $# -le 0 ]; then
    echo "Usage: ${FUNCNAME[0]} PACKAGE_NAME ... "
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} miopen-hip miopen-hip-dev"
    return 1
  fi

  if which sudo; then
    local update_cmd=(sudo)
    local install_cmd=(sudo)
  else
    local update_cmd=()
    local install_cmd=()
  fi

  if which apt-get; then
    update_cmd+=(apt update -y)
    install_cmd+=(apt install -y "$@")
  elif which yum; then
    update_cmd+=(yum update -y)
    install_cmd+=(yum install -y "$@")
  else
    echo "[INSTALL] Could not find a system package installer to install packages!"
    return 1
  fi

  echo "[INSTALL] Updating system repositories ..."
  # shellcheck disable=SC2068
  exec_with_retries ${update_cmd[@]}

  # shellcheck disable=SC2145
  echo "[INSTALL] Installing system package(s): $@ ..."
  # shellcheck disable=SC2068
  exec_with_retries ${install_cmd[@]}
}

free_disk_space () {
  echo "################################################################################"
  echo "# Free Disk Space"
  echo "#"
  echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
  echo "################################################################################"
  echo ""

  sudo rm -rf \
    /usr/local/android \
    /usr/share/dotnet \
    /usr/local/share/boost \
    /opt/ghc \
    /usr/local/share/chrom* \
    /usr/share/swift \
    /usr/local/julia* \
    /usr/local/lib/android

  echo "[CLEANUP] Freed up some disk space"
}


################################################################################
# Info Functions
################################################################################

print_gpu_info () {
  echo "################################################################################"
  echo "[INFO] Printing general display info ..."
  install_system_packages lshw
  print_exec sudo lshw -C display

  echo "################################################################################"
  echo "[INFO] Printing NVIDIA GPU info ..."

  (lspci -v | grep -e 'controller.*NVIDIA') || true

  if [[ "${ENFORCE_NVIDIA_GPU}" ]]; then
    # Ensure that nvidia-smi is available and returns GPU entries
    if ! nvidia-smi; then
      echo "[CHECK] NVIDIA driver is required, but does not appear to have been installed.  This will cause FBGEMM_GPU installation to fail!"
      return 1
    fi
  else
    if which nvidia-smi; then
      # If nvidia-smi is installed on a machine without GPUs, this will return error
      (print_exec nvidia-smi) || true
    else
      echo "[CHECK] nvidia-smi not found"
    fi
  fi

  echo "################################################################################"
  echo "[INFO] Printing AMD GPU info ..."

  (lspci -v | grep -e 'Display controller: Advanced') || true

  if [[ "${ENFORCE_AMD_GPU}" ]]; then
    # Ensure that rocm-smi is available and returns GPU entries
    if ! rocm-smi; then
      echo "[CHECK] AMD driver is required, but does not appear to have been installed.  This will cause FBGEMM_GPU installation to fail!"
      return 1
    fi
  else
    if which rocm-smi; then
      # If rocm-smi is installed on a machine without GPUs, this will return error
      (print_exec rocm-smi) || true
    else
      echo "[CHECK] rocm-smi not found"
    fi
  fi
}

__print_system_info_linux () {
  echo "################################################################################"
  echo "[INFO] Print ldd version ..."
  print_exec ldd --version

  echo "################################################################################"
  echo "[INFO] Print CPU info ..."
  print_exec nproc
  print_exec cat /proc/cpuinfo

  echo "################################################################################"
  echo "[INFO] Print PCI info ..."
  print_exec lspci -v

  echo "################################################################################"
  echo "[INFO] Print Linux distribution info ..."
  print_exec uname -a
  print_exec uname -m
  print_exec cat /proc/version
  print_exec cat /etc/os-release
}

__print_system_info_macos () {
  echo "################################################################################"
  echo "[INFO] Print CPU info ..."
  sysctl -a | grep machdep.cpu

  echo "################################################################################"
  echo "[INFO] Print MacOS version info ..."
  print_exec uname -a
  print_exec sw_vers
}

print_system_info () {
  echo "################################################################################"
  echo "# Print System Info"
  echo "#"
  echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
  echo "################################################################################"
  echo ""

  echo "################################################################################"
  echo "[INFO] Printing environment variables ..."
  print_exec printenv

  if [[ $OSTYPE == 'darwin'* ]]; then
    __print_system_info_macos
  else
    __print_system_info_linux
  fi
}

print_ec2_info () {
  echo "################################################################################"
  echo "# Print EC2 Instance Info"
  echo "#"
  echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
  echo "################################################################################"
  echo ""

  get_ec2_metadata() {
    # Pulled from instance metadata endpoint for EC2
    # see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
    local category=$1
    curl -fsSL "http://169.254.169.254/latest/meta-data/${category}"
  }

  echo "ami-id: $(get_ec2_metadata ami-id)"
  echo "instance-id: $(get_ec2_metadata instance-id)"
  echo "instance-type: $(get_ec2_metadata instance-type)"
}

print_glibc_info () {
  local library_path="$1"
  if [ "$library_path" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} LIBRARY_PATH"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} /usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    return 1
  fi

  if [ -f "${library_path}" ]; then
    echo "[CHECK] Listing out the GLIBC versions referenced by: ${library_path}"
    print_exec "objdump -TC ${library_path} | grep GLIBC_ | sed 's/.*GLIBC_\([.0-9]*\).*/GLIBC_\1/g' | sort -Vu | cat"
    echo ""

    echo "[CHECK] Listing out the GLIBCXX versions referenced by: ${library_path}"
    print_exec "objdump -TC ${library_path} | grep GLIBCXX_ | sed 's/.*GLIBCXX_\([.0-9]*\).*/GLIBCXX_\1/g' | sort -Vu | cat"
    echo ""

  else
    echo "[CHECK] No file at path: ${library_path}"
    return 1
  fi
}

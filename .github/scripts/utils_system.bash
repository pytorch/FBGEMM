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

  test_network_connection || return 1

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
  (exec_with_retries 3 ${update_cmd[@]}) || return 1

  # shellcheck disable=SC2145
  echo "[INSTALL] Installing system package(s): $@ ..."
  # shellcheck disable=SC2068
  (exec_with_retries 3 ${install_cmd[@]}) || return 1
}

free_disk_space () {
  echo "################################################################################"
  echo "# Free Disk Space"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
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

free_disk_space_on_host () {
  echo "################################################################################"
  echo "# Free Disk Space On CI Host"
  echo "################################################################################"

  # NOTE: This is meant to be run from ** inside ** containers hosted on
  # non-PyTorch-infra GitHub runners, where the hosts might be close to full
  # disk from serving many CI jobs.  When the container is set up properly, we
  # can escape the container using nsenter to run commands on the host.
  #
  # On average, we see roughly 3GB of disk freed when running this cleanup,
  # which appears to be sufficient to avoid the somewhat-frequent out-of-disk
  # errors that we were previously running into.
  #
  # Frees up disk space on the ubuntu-latest host machine based on recommendations:
  # https://github.com/orgs/community/discussions/25678
  # https://github.com/apache/flink/blob/02d30ace69dc18555a5085eccf70ee884e73a16e/tools/azure-pipelines/free_disk_space.sh
  #
  # Escape the docker container to run the free disk operation on the host:
  # https://stackoverflow.com/questions/66160057/how-to-run-a-command-in-host-before-entering-docker-container-in-github-ci
  # https://stackoverflow.com/questions/32163955/how-to-run-shell-script-on-host-from-docker-container/63140387#63140387

  nsenter -t 1 -m -u -n -i bash -c "
    echo 'Listing 100 largest packages';
    dpkg-query -Wf '\${Installed-Size}\t\${Package}\n' | sort -n | tail -n 100;
    df -h;

    echo 'Removing large packages';
    sudo apt-get remove -y '^ghc-8.*';
    sudo apt-get remove -y '^dotnet-.*';
    sudo apt-get remove -y '^llvm-.*';
    sudo apt-get remove -y 'php.*';
    sudo apt-get remove -y azure-cli google-cloud-sdk hhvm google-chrome-stable firefox powershell mono-devel;
    sudo apt-get autoremove -y;
    sudo apt-get clean;
    df -h;

    echo 'Removing large directories';
    rm -rf /usr/local/android;
    rm -rf /usr/share/dotnet;
    rm -rf /usr/local/share/boost;
    rm -rf /opt/ghc;
    rm -rf /usr/local/share/chrom*;
    rm -rf /usr/share/swift;
    rm -rf /usr/local/julia*;
    rm -rf /usr/local/lib/android;
    rm -rf /opt/hostedtoolcache;
    df -h;
  "
}


################################################################################
# Info Functions
################################################################################

print_gpu_info () {
  if [[ "${BUILD_FROM_NOVA}" != '1' ]]; then
    echo "################################################################################"
    echo "[INFO] Printing general display info ..."
    install_system_packages lshw
    print_exec sudo lshw -C display
  fi

  echo "################################################################################"
  echo "[INFO] Printing NVIDIA GPU info ..."

  (lspci -v | grep -e 'controller.*NVIDIA') || true

  if [[ "${ENFORCE_CUDA_DEVICE}" == '1' ]]; then
    # Ensure that nvidia-smi is available and returns GPU entries
    if ! nvidia-smi; then
      echo "[CHECK] NVIDIA drivers and CUDA device are required for this workflow, but does not appear to be installed or available!"
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

  if [[ "${ENFORCE_ROCM_DEVICE}" ]]; then
    # Ensure that rocm-smi is available and returns GPU entries
    if ! rocm-smi; then
      echo "[CHECK] ROCm drivers and ROCm device(s) are required for this workflow, but does not appear to be installed or available!"
      return 1
    fi
  else
    local smi_programs=( rocminfo rocm-smi )

    for smi_program in "${smi_programs[@]}"; do
      # shellcheck disable=SC2086
      if which $smi_program; then
        # If the program is installed on a machine without GPUs, invoking it will return error
        # shellcheck disable=SC2086
        (print_exec $smi_program) || true
      else
        echo "[CHECK] $smi_program not found"
      fi
    done
  fi
}

__print_system_info_linux () {
  echo "################################################################################"
  echo "[INFO] Print ldd version ..."
  print_exec ldd --version

  echo "################################################################################"
  echo "[INFO] Print CPU info ..."
  print_exec nproc
  print_exec lscpu
  print_exec cat /proc/cpuinfo


  if [[ "${BUILD_FROM_NOVA}" != '1' ]]; then
    echo "################################################################################"
    echo "[INFO] Print PCI info ..."
    print_exec lspci -v
  fi

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
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
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
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  get_ec2_metadata() {
    # Pulled from instance metadata endpoint for EC2
    # see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
    local category=$1
    curl -H "X-aws-ec2-metadata-token: $(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 30")" -fsSL "http://169.254.169.254/latest/meta-data/${category}"
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

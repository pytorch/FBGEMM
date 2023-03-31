#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


################################################################################
# Command Execution Functions
################################################################################

print_exec () {
  echo "+ $*"
  echo ""
  if "$@"; then
    local retcode=0
  else
    local retcode=$?
  fi
  echo ""
  return $retcode
}

exec_with_retries () {
  local max=5
  local delay=2
  local retcode=0

  for i in $(seq 1 ${max}); do
    echo "[EXEC] [ATTEMPT ${i}/${max}]    + $*"

    if "$@"; then
      retcode=0
      break
    else
      retcode=$?
      echo "[EXEC] [ATTEMPT ${i}/${max}] Command attempt failed."
      echo ""
      sleep $delay
    fi
  done

  if [ $retcode -ne 0 ]; then
    echo "[EXEC] The command has failed after ${max} attempts; aborting."
  fi

  return $retcode
}


################################################################################
# Assert Functions
################################################################################

test_python_import () {
  local env_name="$1"
  local python_import="$2"
  if [ "$python_import" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTHON_IMPORT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env numpy"
    return 1
  fi

  if conda run -n "${env_name}" python -c "import ${python_import}"; then
    echo "[CHECK] Python package ${python_import} found."
  else
    echo "[CHECK] Python package ${python_import} not found!"
    return 1
  fi
}

test_binpath () {
  local env_name="$1"
  local bin_name="$2"
  if [ "$bin_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME BIN_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env nvcc"
    return 1
  fi

  if conda run -n "${env_name}" which "${bin_name}"; then
    echo "[CHECK] Binary ${bin_name} found in PATH"
  else
    echo "[CHECK] Binary ${bin_name} not found in PATH!"
    return 1
  fi
}

test_filepath () {
  local env_name="$1"
  local file_name="$2"
  if [ "$file_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME FILE_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env cuda_runtime.h"
    return 1
  fi

  # shellcheck disable=SC2155
  local conda_prefix=$(conda run -n "${env_name}" printenv CONDA_PREFIX)
  # shellcheck disable=SC2155
  local file_path=$(find "${conda_prefix}" -type f -name "${file_name}")
  # shellcheck disable=SC2155
  local link_path=$(find "${conda_prefix}" -type l -name "${file_name}")
  if [ "${file_path}" != "" ]; then
    echo "[CHECK] ${file_name} found in CONDA_PREFIX PATH (file): ${file_path}"
  elif [ "${link_path}" != "" ]; then
    echo "[CHECK] ${file_name} found in CONDA_PREFIX PATH (symbolic link): ${link_path}"
  else
    echo "[CHECK] ${file_name} not found in CONDA_PREFIX PATH!"
    return 1
  fi
}

test_env_var () {
  local env_name="$1"
  local env_key="$2"
  if [ "$env_key" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME ENV_KEY"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env CUDNN_INCLUDE_DIR"
    return 1
  fi

  if conda run -n "${env_name}" printenv "${env_key}"; then
    echo "[CHECK] Environment variable ${env_key} is defined in the Conda environment"
  else
    echo "[CHECK] Environment variable ${env_key} is not defined in the Conda environment!"
    return 1
  fi
}

test_library_symbol () {
  local lib_path="$1"
  local lib_symbol="$2"
  if [ "$lib_symbol" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} LIB_PATH FULL_NAMESPACE_PATH_LIB_SYMBOL"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} fbgemm_gpu_py.so fbgemm_gpu::merge_pooled_embeddings"
    return 1
  fi

  # Add space and '(' to the grep string to get the full method path
  symbol_entries=$(nm -gDC "${lib_path}" | grep " ${lib_symbol}(")
  if [ "${symbol_entries}" != "" ]; then
    echo "[CHECK] Found symbol in ${lib_path}: ${lib_symbol}"
  else
    echo "[CHECK] Symbol NOT found in ${lib_path}: ${lib_symbol}"
    return 1
  fi
}


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

run_python_test () {
  local env_name="$1"
  local python_test_file="$2"
  if [ "$python_test_file" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTHON_TEST_FILE"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env quantize_ops_test.py"
    return 1
  else
    echo "################################################################################"
    echo "# [$(date --utc +%FT%T.%3NZ)] Run Python Test Suite:"
    echo "#   ${python_test_file}"
    echo "################################################################################"
  fi

  if print_exec conda run -n "${env_name}" python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning "${python_test_file}"; then
    echo "[TEST] Python test suite PASSED: ${python_test_file}"
    echo ""
  else
    echo "[TEST] Python test suite FAILED: ${python_test_file}"
    echo ""
    return 1
  fi
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
  echo "[INFO] Check GPU info ..."
  install_system_packages lshw
  print_exec sudo lshw -C display

  echo "################################################################################"
  echo "[INFO] Check NVIDIA GPU info ..."

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
  echo "[INFO] Check ldd version ..."
  print_exec ldd --version

  echo "################################################################################"
  echo "[INFO] Check CPU info ..."
  print_exec nproc
  print_exec cat /proc/cpuinfo

  echo "################################################################################"
  echo "[INFO] Check Linux distribution info ..."
  print_exec uname -a
  print_exec cat /proc/version
  print_exec cat /etc/os-release
}

__print_system_info_macos () {
  echo "################################################################################"
  echo "[INFO] Check CPU info ..."
  sysctl -a | grep machdep.cpu

  echo "################################################################################"
  echo "[INFO] Check MacOS version info ..."
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
    objdump -TC "${library_path}" | grep GLIBC_ | sed 's/.*GLIBC_\([.0-9]*\).*/GLIBC_\1/g' | sort -Vu | cat
    echo ""

    echo "[CHECK] Listing out the GLIBCXX versions referenced by: ${library_path}"
    objdump -TC "${library_path}" | grep GLIBCXX_ | sed 's/.*GLIBCXX_\([.0-9]*\).*/GLIBCXX_\1/g' | sort -Vu | cat
    echo ""

  else
    echo "[CHECK] No file at path: ${library_path}"
    return 1
  fi
}


################################################################################
# Bazel Setup Functions
################################################################################

setup_bazel () {
  local bazel_version="${1:-6.1.1}"
  echo "################################################################################"
  echo "# Setup Bazel"
  echo "#"
  echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
  echo "################################################################################"
  echo ""

  if [[ $OSTYPE == 'darwin'* ]]; then
    # shellcheck disable=SC2155
    local bazel_variant="darwin-$(uname -m)"
  else
    local bazel_variant="linux-x86_64"
  fi

  echo "[SETUP] Downloading installer Bazel ${bazel_version} (${bazel_variant}) ..."
  print_exec wget -q "https://github.com/bazelbuild/bazel/releases/download/${bazel_version}/bazel-${bazel_version}-installer-${bazel_variant}.sh" -O install-bazel.sh

  echo "[SETUP] Installing Bazel ..."
  print_exec bash install-bazel.sh
  print_exec rm -f install-bazel.sh

  print_exec bazel --version
  echo "[SETUP] Successfully set up Bazel"
}


################################################################################
# Miniconda Setup Functions
################################################################################

__conda_cleanup () {
  echo "[SETUP] Cleaning up Conda packages ..."
  (print_exec conda clean --packages --tarball -y) || return 1
  (print_exec conda clean --all -y) || return 1
}

setup_miniconda () {
  local miniconda_prefix="$1"
  if [ "$miniconda_prefix" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} MINICONDA_PREFIX_PATH"
    echo "Example:"
    echo "    setup_miniconda /home/user/tmp/miniconda"
    return 1
  else
    echo "################################################################################"
    echo "# Setup Miniconda"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # Download and install Miniconda if doesn't exist
  if [ ! -f "${miniconda_prefix}/bin/conda" ]; then
    print_exec mkdir -p "$miniconda_prefix"

    echo "[SETUP] Downloading the Miniconda installer ..."
    (exec_with_retries wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh) || return 1

    echo "[SETUP] Installing Miniconda ..."
    print_exec bash miniconda.sh -b -p "$miniconda_prefix" -u
    print_exec rm -f miniconda.sh
  fi

  echo "[SETUP] Reloading the bash configuration ..."
  print_exec "${miniconda_prefix}/bin/conda" init bash
  print_exec . ~/.bashrc

  echo "[SETUP] Updating Miniconda base packages ..."
  (exec_with_retries conda update -n base -c defaults --update-deps -y conda) || return 1

  # Clean up packages
  __conda_cleanup

  # Print Conda info
  print_exec conda info

  # These variables will be exported outside
  echo "[SETUP] Exporting Miniconda variables ..."
  export PATH="${miniconda_prefix}/bin:${PATH}"
  export CONDA="${miniconda_prefix}"

  if [ -f "${GITHUB_PATH}" ]; then
    echo "[SETUP] Saving Miniconda variables to ${GITHUB_PATH} ..."
    echo "${miniconda_prefix}/bin" >> "${GITHUB_PATH}"
    echo "CONDA=${miniconda_prefix}" >> "${GITHUB_PATH}"
  fi

  echo "[SETUP] Successfully set up Miniconda at ${miniconda_prefix}"
}

create_conda_environment () {
  local env_name="$1"
  local python_version="$2"
  if [ "$python_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTHON_VERSION"
    echo "Example:"
    echo "    ${FUNCNAME[0]} build_env 3.10"
    return 1
  else
    echo "################################################################################"
    echo "# Create Conda Environment"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # The `-y` flag removes any existing Conda environment with the same name
  echo "[SETUP] Creating new Conda environment (Python ${python_version}) ..."
  (exec_with_retries conda create -y --name "${env_name}" python="${python_version}") || return 1

  echo "[SETUP] Upgrading PIP to latest ..."
  (exec_with_retries conda run -n "${env_name}" pip install --upgrade pip) || return 1

  # The pyOpenSSL and cryptography packages versions need to line up for PyPI publishing to work
  # https://stackoverflow.com/questions/74981558/error-updating-python3-pip-attributeerror-module-lib-has-no-attribute-openss
  echo "[SETUP] Upgrading pyOpenSSL ..."
  (exec_with_retries conda run -n "${env_name}" python -m pip install "pyOpenSSL>22.1.0") || return 1

  # This test fails with load errors if the pyOpenSSL and cryptography package versions don't align
  echo "[SETUP] Testing pyOpenSSL import ..."
  (test_python_import "${env_name}" OpenSSL) || return 1

  echo "[SETUP] Installed Python version: $(conda run -n "${env_name}" python --version)"
  echo "[SETUP] Successfully created Conda environment: ${env_name}"
}


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
    pytorch_package="${pytorch_package}==${pytorch_version}"
    local pytorch_channel="pytorch"
  fi

  # Clean up packages before installation
  __conda_cleanup

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
  env_name="$1"
  pytorch_version="$2"
  pytorch_variant_type="$3"
  pytorch_variant_version="$4"
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
    cuda_version="${pytorch_variant_version:-11.7.1}"
    # shellcheck disable=SC2206
    cuda_version_arr=(${cuda_version//./ })
    # Convert, i.e. cuda 11.7.1 => cu117
    pytorch_variant="cu${cuda_version_arr[0]}${cuda_version_arr[1]}"
  elif [ "$pytorch_variant_type" == "rocm" ]; then
    # Extract the ROCM version or default to 5.3
    rocm_version="${pytorch_variant_version:-5.3}"
    pytorch_variant="rocm${rocm_version}"
  else
    pytorch_variant_type="cpu"
    pytorch_variant="cpu"
  fi
  echo "[INSTALL] Extracted PyTorch variant: ${pytorch_variant}"

  # Set the package name and installation channel
  if [ "$pytorch_version" == "nightly" ] || [ "$pytorch_version" == "test" ]; then
    pytorch_package="--pre torch"
    pytorch_channel="https://download.pytorch.org/whl/${pytorch_version}/${pytorch_variant}/"
  elif [ "$pytorch_version" == "latest" ]; then
    pytorch_package="torch"
    pytorch_channel="https://download.pytorch.org/whl/${pytorch_variant}/"
  else
    pytorch_package="torch==${pytorch_version}+${pytorch_variant}"
    pytorch_channel="https://download.pytorch.org/whl/${pytorch_variant}/"
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


################################################################################
# CUDA Setup Functions
################################################################################

install_nvidia_drivers_centos () {
  echo "################################################################################"
  echo "# Install NVIDIA Drivers"
  echo "#"
  echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
  echo "################################################################################"
  echo ""

  echo "[SETUP] Adding NVIDIA repos to yum ..."
  print_exec sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
  print_exec sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
  print_exec sudo yum clean expire-cache

  echo "[SETUP] Installing NVIDIA drivers ..."
  install_system_packages nvidia-driver-latest-dkms
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
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # Check CUDA version formatting
  # shellcheck disable=SC2206
  local cuda_version_arr=(${cuda_version//./ })
  if [ ${#cuda_version_arr[@]} -lt 3 ]; then
    echo "[ERROR] CUDA minor version number must be specified (i.e. X.Y.Z)"
    return 1
  fi

  # Clean up packages before installation
  __conda_cleanup

  # Install CUDA packages
  echo "[INSTALL] Installing CUDA ${cuda_version} ..."
  (exec_with_retries conda install --force-reinstall -n "${env_name}" -y cuda -c "nvidia/label/cuda-${cuda_version}") || return 1

  # Ensure that nvcc is properly installed
  (test_binpath "${env_name}" nvcc) || return 1

  # Ensure that the CUDA headers are properly installed
  (test_filepath "${env_name}" cuda_runtime.h) || return 1

  # Ensure that the libraries are properly installed
  (test_filepath "${env_name}" libnvToolsExt.so) || return 1
  (test_filepath "${env_name}" libnvidia-ml.so) || return 1

  echo "[INSTALL] Set environment variable NVML_LIB_PATH ..."
  # shellcheck disable=SC2155
  local conda_prefix=$(conda run -n "${env_name}" printenv CONDA_PREFIX)
  # shellcheck disable=SC2155
  local nvml_lib_path=$(find "${conda_prefix}" -name libnvidia-ml.so)
  print_exec conda env config vars set -n "${env_name}" NVML_LIB_PATH="${nvml_lib_path}"

  # Print nvcc version
  print_exec conda run -n "${env_name}" nvcc --version
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
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # Install cuDNN manually
  # Based on install script in https://github.com/pytorch/builder/blob/main/common/install_cuda.sh
  local cudnn_packages=(
    ["115"]="https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz"
    ["116"]="https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz"
    ["117"]="https://ossci-linux.s3.amazonaws.com/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz"
    ["118"]="https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz"
  )

  # Split version string by dot into array, i.e. 11.7.1 => [11, 7, 1]
  # shellcheck disable=SC2206
  local cuda_version_arr=(${cuda_version//./ })
  # Fetch the major and minor version to concat
  local cuda_concat_version="${cuda_version_arr[0]}${cuda_version_arr[1]}"

  # Get the URL
  local cudnn_url="${cudnn_packages[cuda_concat_version]}"
  if [ "$cudnn_url" == "" ]; then
    # Default to cuDNN for 11.7 if no CUDA version fits
    echo "[INSTALL] Defaulting to cuDNN for CUDA 11.7"
    cudnn_url="${cudnn_packages[117]}"
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
  (exec_with_retries wget -q "$cudnn_url" -O cudnn.tar.xz) || return 1

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

  # Export the environment variables to the Conda environment
  echo "[INSTALL] Set environment variables CUDNN_INCLUDE_DIR and CUDNN_LIBRARY ..."
  print_exec conda env config vars set -n "${env_name}" CUDNN_INCLUDE_DIR="${install_path}/include" CUDNN_LIBRARY="${install_path}/lib"

  echo "[INSTALL] Successfully installed cuDNN (for CUDA ${cuda_version})"
}

################################################################################
# ROCm Setup Functions
################################################################################

install_rocm_ubuntu () {
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
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

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
  install_system_packages "./${package_name}"

  # Skip installation of kernel driver when run in Docker mode with --no-dkms
  echo "[INSTALL] Installing ROCm ..."
  (exec_with_retries amdgpu-install -y --usecase=hiplibsdk,rocm --no-dkms) || return 1

  echo "[INSTALL] Installing HIP-relevant packages ..."
  install_system_packages hipify-clang miopen-hip miopen-hip-dev

  # There is no need to install these packages for ROCm
  # install_system_packages mesa-common-dev clang comgr libopenblas-dev jp intel-mkl-full locales libnuma-dev

  echo "[INSTALL] Cleaning up ..."
  print_exec rm -f "${package_name}"

  echo "[INFO] Check ROCM GPU info ..."
  print_exec rocm-smi

  echo "[INSTALL] Successfully installed ROCm ${rocm_version}"
}


################################################################################
# Build Tools Setup Functions
################################################################################

install_cxx_compiler () {
  local env_name="$1"
  local use_system_package_manager="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [USE_YUM]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env     # Install C/C++ compilers through Conda"
    echo "    ${FUNCNAME[0]} build_env 1   # Install C/C++ compilers through the system package manager"
    return 1
  else
    echo "################################################################################"
    echo "# Install C/C++ Compilers"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  if [ "$use_system_package_manager" != "" ]; then
    echo "[INSTALL] Installing C/C++ compilers through the system package manager ..."
    install_system_packages gcc gcc-c++

  else
    # Install gxx_linux-64 from conda-forge instead of from anaconda channel.
    # sysroot_linux-64 needs to be installed alongside this:
    #
    #   https://root-forum.cern.ch/t/error-timespec-get-has-not-been-declared-with-conda-root-package/45712/6
    #   https://github.com/conda-forge/conda-forge.github.io/issues/1625
    #   https://conda-forge.org/docs/maintainer/knowledge_base.html#using-centos-7
    #   https://github.com/conda/conda-build/issues/4371
    #
    # NOTE: We install g++ 10.x instead of 11.x becaue 11.x builds binaries that
    # reference GLIBCXX_3.4.29, which may not be available on systems with older
    # versions of libstdc++.so.6 such as CentOS Stream 8 and Ubuntu 20.04
    echo "[INSTALL] Installing C/C++ compilers through Conda ..."
    (exec_with_retries conda install -n "${env_name}" -y gxx_linux-64=10.4.0 sysroot_linux-64=2.17 -c conda-forge) || return 1

    # The compilers are visible in the PATH as `x86_64-conda-linux-gnu-cc` and
    # `x86_64-conda-linux-gnu-c++`, so symlinks will need to be created
    echo "[INSTALL] Setting the C/C++ compiler symlinks ..."
    # shellcheck disable=SC2155
    local cc_path=$(conda run -n "${env_name}" printenv CC)
    # shellcheck disable=SC2155
    local cxx_path=$(conda run -n "${env_name}" printenv CXX)

    print_exec ln -s "${cc_path}" "$(dirname "$cc_path")/cc"
    print_exec ln -s "${cc_path}" "$(dirname "$cc_path")/gcc"
    print_exec ln -s "${cxx_path}" "$(dirname "$cxx_path")/c++"
    print_exec ln -s "${cxx_path}" "$(dirname "$cxx_path")/g++"
  fi

  # Check C/C++ compilers are visible
  (test_binpath "${env_name}" cc) || return 1
  (test_binpath "${env_name}" gcc) || return 1
  (test_binpath "${env_name}" c++) || return 1
  (test_binpath "${env_name}" g++) || return 1

  # Print out the C++ version
  print_exec conda run -n "${env_name}" c++ --version

  # https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
  echo "[INSTALL] Printing the default version of the C++ standard used by the compiler ..."
  print_exec conda run -n "${env_name}" c++ -x c++ /dev/null -E -dM | grep __cplusplus

  # https://stackoverflow.com/questions/4991707/how-to-find-my-current-compilers-standard-like-if-it-is-c90-etc
  echo "[INSTALL] Printing the default version of the C standard used by the compiler ..."
  print_exec conda run -n "${env_name}" cc -dM -E - < /dev/null | grep __STDC_VERSION__

  echo "[INSTALL] Successfully installed C/C++ compilers"
}

install_build_tools () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Install Build Tools"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "[INSTALL] Installing build tools ..."
  (exec_with_retries conda install -n "${env_name}" -y \
    click \
    cmake \
    hypothesis \
    jinja2 \
    ninja \
    numpy \
    scikit-build \
    wheel) || return 1

  # Check binaries are visible in the PAATH
  (test_binpath "${env_name}" cmake) || return 1
  (test_binpath "${env_name}" ninja) || return 1

  # Check Python packages are importable
  local import_tests=( click hypothesis jinja2 numpy skbuild wheel )
  for p in "${import_tests[@]}"; do
    (test_python_import "${env_name}" "${p}") || return 1
  done

  echo "[INSTALL] Successfully installed all the build tools"
}

install_docs_tools () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Install Documentation Tools"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "[INSTALL] Installing docs tools ..."
  (exec_with_retries conda install -n "${env_name}" -c conda-forge -y \
    doxygen) || return 1

  # Check binaries are visible in the PAATH
  (test_binpath "${env_name}" doxygen) || return 1

  echo "[INSTALL] Successfully installed all the build tools"
}

################################################################################
# Combination Functions
################################################################################

create_conda_pytorch_environment () {
  env_name="$1"
  python_version="$2"
  pytorch_channel_name="$3"
  cuda_version="$4"
  if [ "$python_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTHON_VERSION PYTORCH_CHANNEL_NAME CUDA_VERSION"
    echo "Example:"
    echo "    ${FUNCNAME[0]} build_env 3.10 pytorch-nightly 11.7.1"
    return 1
  fi

  # Create the Conda environment
  create_conda_environment "${env_name}" "${python_version}"

  # Convert the channels to versions
  if [ "${pytorch_channel_name}" == "pytorch-nightly" ]; then
    pytorch_version="nightly"
  elif [ "${pytorch_channel_name}" == "pytorch-test" ]; then
    pytorch_version="test"
  else
    pytorch_version="latest"
  fi

  if [ "${cuda_version}" == "" ]; then
    # Install the CPU variant of PyTorch
    install_pytorch_conda "${env_name}" "${pytorch_version}" cpu
  else
    # Install CUDA and the GPU variant of PyTorch
    install_cuda "${env_name}" "${cuda_version}"
    install_pytorch_conda "${env_name}" "${pytorch_version}"
  fi
}


################################################################################
# FBGEMM_GPU Build Functions
################################################################################

prepare_fbgemm_gpu_build () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Prepare FBGEMM-GPU Build"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  if [[ "${GITHUB_WORKSPACE}" ]]; then
    # https://github.com/actions/checkout/issues/841
    git config --global --add safe.directory "${GITHUB_WORKSPACE}"
  fi

  echo "[BUILD] Running git submodules update ..."
  git submodule sync
  git submodule update --init --recursive

  echo "[BUILD] Installing other build dependencies ..."
  (exec_with_retries conda run -n "${env_name}" python -m pip install -r requirements.txt) || return 1

  (test_python_import "${env_name}" numpy) || return 1
  (test_python_import "${env_name}" skbuild) || return 1

  echo "[BUILD] Successfully ran git submodules update"
}

__configure_fbgemm_gpu_build_cpu () {
  # Update the package name and build args depending on if CUDA is specified
  echo "[BUILD] Setting CPU-only build args ..."
  build_args=(--cpu_only)
}

__configure_fbgemm_gpu_build_rocm () {
  local fbgemm_variant_targets="$1"

  # Fetch available ROCm architectures on the machine
  if [ "$fbgemm_variant_targets" != "" ]; then
    echo "[BUILD] ROCm targets have been manually provided: ${fbgemm_variant_targets}"
    local arch_list="${fbgemm_variant_targets}"
  else
    if which rocminfo; then
      # shellcheck disable=SC2155
      local arch_list=$(rocminfo | grep -o -m 1 'gfx.*')
      echo "[BUILD] Architectures list from rocminfo: ${arch_list}"

      if [ "$arch_list" == "" ]; then
        # By default, build for MI250 only to save time
        local arch_list=gfx90a
      fi
    else
      echo "[BUILD] rocminfo not found in PATH!"
    fi
  fi

  echo "[BUILD] Setting the following ROCm targets: ${arch_list}"
  print_exec conda env config vars set -n "${env_name}" PYTORCH_ROCM_ARCH="${arch_list}"

  echo "[BUILD] Setting ROCm build args ..."
  build_args=()
}

__configure_fbgemm_gpu_build_cuda () {
  local fbgemm_variant_targets="$1"

  # Check nvcc is visible
  (test_binpath "${env_name}" nvcc) || return 1

  # Check that cuDNN environment variables are available
  (test_env_var "${env_name}" CUDNN_INCLUDE_DIR) || return 1
  (test_env_var "${env_name}" CUDNN_LIBRARY) || return 1
  (test_env_var "${env_name}" NVML_LIB_PATH) || return 1

  local arch_list="${fbgemm_variant_targets:-7.0;8.0}"
  echo "[BUILD] Setting the following CUDA targets: ${arch_list}"

  # Build only CUDA 7.0 and 8.0 (i.e. V100 and A100) because of 100 MB binary size limits from PyPI.
  echo "[BUILD] Setting CUDA build args ..."
  # shellcheck disable=SC2155
  local nvml_lib_path=$(conda run -n "${env_name}" printenv NVML_LIB_PATH)
  build_args=(
    --nvml_lib_path="${nvml_lib_path}"
    -DTORCH_CUDA_ARCH_LIST="'${arch_list}'"
  )
}

__configure_fbgemm_gpu_build () {
  local fbgemm_variant="$1"
  local fbgemm_variant_targets="$2"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} FBGEMM_VARIANT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} cpu                          # CPU-only variant"
    echo "    ${FUNCNAME[0]} cuda                         # CUDA variant for default target(s)"
    echo "    ${FUNCNAME[0]} cuda '7.0;8.0'               # CUDA variant for custom target(s)"
    echo "    ${FUNCNAME[0]} rocm                         # ROCm variant for default target(s)"
    echo "    ${FUNCNAME[0]} rocm 'gfx906;gfx908;gfx90a'  # ROCm variant for custom target(s)"
    return 1
  else
    echo "################################################################################"
    echo "# Configure FBGEMM-GPU Build"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  if [ "$fbgemm_variant" == "cpu" ]; then
    echo "[BUILD] Configuring build as CPU variant ..."
    __configure_fbgemm_gpu_build_cpu

  elif [ "$fbgemm_variant" == "rocm" ]; then
    echo "[BUILD] Configuring build as ROCm variant ..."
    __configure_fbgemm_gpu_build_rocm "${fbgemm_variant_targets}"

  else
    echo "[BUILD] Configuring build as CUDA variant (this is the default behavior) ..."
    __configure_fbgemm_gpu_build_cuda "${fbgemm_variant_targets}"
  fi

  # shellcheck disable=SC2145
  echo "[BUILD] FBGEMM_GPU build arguments have been set:  ${build_args[@]}"
}

__build_fbgemm_gpu_common_pre_steps () {
  # Private function that uses variables instantiated by its caller

  # Check C/C++ compilers are visible (the build scripts look specifically for `gcc`)
  (test_binpath "${env_name}" cc) || return 1
  (test_binpath "${env_name}" gcc) || return 1
  (test_binpath "${env_name}" c++) || return 1
  (test_binpath "${env_name}" g++) || return 1

  if [ "$fbgemm_variant" == "cpu" ]; then
    package_name="${package_name}-cpu"
  elif [ "$fbgemm_variant" == "rocm" ]; then
    package_name="${package_name}-rocm"
  else
    # Set to the default variant
    fbgemm_variant="cuda"
  fi

  # Extract the Python tag
  # shellcheck disable=SC2207
  python_version=($(conda run -n "${env_name}" python --version))
  # shellcheck disable=SC2206
  python_version_arr=(${python_version[1]//./ })
  python_tag="py${python_version_arr[0]}${python_version_arr[1]}"
  echo "[BUILD] Extracted Python tag: ${python_tag}"

  echo "[BUILD] Running pre-build cleanups ..."
  print_exec rm -rf dist
  print_exec conda run -n "${env_name}" python setup.py clean

  echo "[BUILD] Printing git status ..."
  print_exec git status
  print_exec git diff
}

run_fbgemm_gpu_postbuild_checks () {
  local fbgemm_variant="$1"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} FBGEMM_VARIANT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} cpu"
    echo "    ${FUNCNAME[0]} cuda"
    echo "    ${FUNCNAME[0]} rocm"
    return 1
  fi

  # Find the .SO file
  # shellcheck disable=SC2155
  local fbgemm_gpu_so_files=$(find . -name fbgemm_gpu_py.so)
  readarray -t fbgemm_gpu_so_files <<<"$fbgemm_gpu_so_files"
  if [ "${#fbgemm_gpu_so_files[@]}" -le 0 ]; then
    echo "[CHECK] .SO library fbgemm_gpu_py.so is missing from the build path!"
    return 1
  fi

  # Prepare a sample set of symbols whose existence in the built library should be checked
  # This is by no means an exhaustive set, and should be updated accordingly
  local lib_symbols_to_check=(
    fbgemm_gpu::asynchronous_inclusive_cumsum_cpu
    fbgemm_gpu::jagged_2d_to_dense
  )

  # Add more symbols to check for if it's a non-CPU variant
  if [ "${fbgemm_variant}" == "cuda" ]; then
    lib_symbols_to_check+=(
      fbgemm_gpu::asynchronous_inclusive_cumsum_gpu
      fbgemm_gpu::merge_pooled_embeddings
    )
  elif [ "${fbgemm_variant}" == "rocm" ]; then
    # merge_pooled_embeddings is missing in ROCm builds bc it requires NVML
    lib_symbols_to_check+=(
      fbgemm_gpu::asynchronous_inclusive_cumsum_gpu
      fbgemm_gpu::merge_pooled_embeddings
    )
  fi

  for library in "${fbgemm_gpu_so_files[@]}"; do
    echo "[CHECK] Listing out the GLIBCXX versions referenced by the library: ${library}"
    print_glibc_info "${library}"

    echo "[CHECK] Verifying sample subset of symbols in the library ..."
    for symbol in "${lib_symbols_to_check[@]}"; do
      (test_library_symbol "${library}" "${symbol}") || return 1
    done

    echo ""
  done
}

build_fbgemm_gpu_package () {
  env_name="$1"
  package_name="$2"
  fbgemm_variant="$3"
  fbgemm_variant_targets="$4"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME VARIANT [TARGETS]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly cpu                           # CPU-only variant"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly cuda                          # CUDA variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly cuda '7.0;8.0'                # CUDA variant for custom target(s)"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly rocm                          # ROCm variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly rocm 'gfx906;gfx908;gfx90a'   # ROCm variant for custom target(s)"
    return 1
  fi

  # Set up and configure the build
  __build_fbgemm_gpu_common_pre_steps || return 1
  __configure_fbgemm_gpu_build "${fbgemm_variant}" "${fbgemm_variant_targets}" || return 1

  echo "################################################################################"
  echo "# Build FBGEMM-GPU Package (Wheel)"
  echo "#"
  echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
  echo "################################################################################"
  echo ""

  # manylinux1_x86_64 is specified for PyPI upload
  # Distribute Python extensions as wheels on Linux
  echo "[BUILD] Building FBGEMM-GPU wheel (VARIANT=${fbgemm_variant}) ..."
  print_exec conda run -n "${env_name}" \
    python setup.py bdist_wheel \
      --package_name="${package_name}" \
      --python-tag="${python_tag}" \
      --plat-name=manylinux1_x86_64 \
      "${build_args[@]}"

  # Run checks on the built libraries
  (run_fbgemm_gpu_postbuild_checks "${fbgemm_variant}") || return 1

  echo "[BUILD] Enumerating the built wheels ..."
  print_exec ls -lth dist/*.whl

  echo "[BUILD] Enumerating the wheel SHAs ..."
  print_exec sha1sum dist/*.whl

  echo "[BUILD] FBGEMM-GPU build wheel completed"
}

build_fbgemm_gpu_install () {
  env_name="$1"
  fbgemm_variant="$2"
  fbgemm_variant_targets="$3"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME VARIANT [TARGETS]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env cpu                          # CPU-only variant"
    echo "    ${FUNCNAME[0]} build_env cuda                         # CUDA variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env cuda '7.0;8.0'               # CUDA variant for custom target(s)"
    echo "    ${FUNCNAME[0]} build_env rocm                         # ROCm variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env rocm 'gfx906;gfx908;gfx90a'  # ROCm variant for custom target(s)"
    return 1
  fi

  # Set up and configure the build
  __build_fbgemm_gpu_common_pre_steps || return 1
  __configure_fbgemm_gpu_build "${fbgemm_variant}" "${fbgemm_variant_targets}" || return 1

  echo "################################################################################"
  echo "# Build + Install FBGEMM-GPU Package"
  echo "#"
  echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
  echo "################################################################################"
  echo ""

  # Parallelism may need to be limited to prevent the build from being
  # canceled for going over ulimits
  echo "[BUILD] Building + installing FBGEMM-GPU (VARIANT=${fbgemm_variant}) ..."
  print_exec conda run -n "${env_name}" \
    python setup.py install "${build_args[@]}"

  # Run checks on the built libraries
  (run_fbgemm_gpu_postbuild_checks "${fbgemm_variant}") || return 1

  echo "[INSTALL] Checking imports ..."
  # Exit this directory to prevent import clashing, since there is an
  # fbgemm_gpu/ subdirectory present
  cd - || return 1
  (test_python_import "${env_name}" fbgemm_gpu) || return 1

  echo "[BUILD] FBGEMM-GPU build + install completed"
}

build_fbgemm_gpu_develop () {
  env_name="$1"
  fbgemm_variant="$2"
  fbgemm_variant_targets="$3"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME VARIANT [TARGETS]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env cpu                          # CPU-only variant"
    echo "    ${FUNCNAME[0]} build_env cuda                         # CUDA variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env cuda '7.0;8.0'               # CUDA variant for custom target(s)"
    echo "    ${FUNCNAME[0]} build_env rocm                         # ROCm variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env rocm 'gfx906;gfx908;gfx90a'  # ROCm variant for custom target(s)"
    return 1
  fi

  # Set up and configure the build
  __build_fbgemm_gpu_common_pre_steps || return 1
  __configure_fbgemm_gpu_build "${fbgemm_variant}" "${fbgemm_variant_targets}" || return 1

  echo "################################################################################"
  echo "# Build + Install FBGEMM-GPU Package"
  echo "#"
  echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
  echo "################################################################################"
  echo ""

  # Parallelism may need to be limited to prevent the build from being
  # canceled for going over ulimits
  echo "[BUILD] Building (develop) FBGEMM-GPU (VARIANT=${fbgemm_variant}) ..."
  print_exec conda run -n "${env_name}" \
    python setup.py build develop "${build_args[@]}"

  # Run checks on the built libraries
  (run_fbgemm_gpu_postbuild_checks "${fbgemm_variant}") || return 1

  echo "[BUILD] FBGEMM-GPU build + develop completed"
}

build_fbgemm_gpu_docs () {
  env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env      # Build the docs"
    return 1
  else
    echo "################################################################################"
    echo "# Build FBGEMM-GPU Documentation"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "[BUILD] Installing docs-build dependencies ..."
  (exec_with_retries conda run -n "${env_name}" python -m pip install -r requirements.txt) || return 1

  echo "[BUILD] Running Doxygen build ..."
  (exec_with_retries conda run -n "${env_name}" doxygen Doxyfile.in) || return 1

  echo "[BUILD] Building HTML pages ..."
  (exec_with_retries conda run -n "${env_name}" make html) || return 1

  echo "[INSTALL] FBGEMM-GPU documentation build completed"
}

install_fbgemm_gpu_package () {
  local env_name="$1"
  local package_name="$2"
  if [ "$package_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME WHEEL_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu.whl     # Install the package (wheel)"
    return 1
  else
    echo "################################################################################"
    echo "# Install FBGEMM-GPU Package (Wheel)"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "[INSTALL] Printing out FBGEMM-GPU wheel SHA: ${package_name}"
  print_exec sha1sum "${package_name}"

  echo "[INSTALL] Installing FBGEMM-GPU wheel: ${package_name} ..."
  (exec_with_retries conda run -n "${env_name}" python -m pip install "${package_name}") || return 1

  echo "[INSTALL] Checking imports ..."
  (test_python_import "${env_name}" fbgemm_gpu) || return 1
  (test_python_import "${env_name}" fbgemm_gpu.split_embedding_codegen_lookup_invokers) || return 1

  echo "[INSTALL] Wheel installation completed ..."
}


################################################################################
# FBGEMM_GPU Test Functions
################################################################################

run_fbgemm_gpu_tests () {
  local env_name="$1"
  local fbgemm_variant="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [FBGEMM_VARIANT]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env        # Run all tests applicable to CUDA"
    echo "    ${FUNCNAME[0]} build_env cpu    # Run all tests applicable to CPU"
    echo "    ${FUNCNAME[0]} build_env rocm   # Run all tests applicable to ROCm"
    return 1
  else
    echo "################################################################################"
    echo "# Run FBGEMM-GPU Tests"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # Enable ROCM testing if specified
  if [ "$fbgemm_variant" == "rocm" ]; then
    echo "[TEST] Set environment variable FBGEMM_TEST_WITH_ROCM to enable ROCm tests ..."
    print_exec conda env config vars set -n "${env_name}" FBGEMM_TEST_WITH_ROCM=1
  fi

  # These are either non-tests or currently-broken tests in both FBGEMM_GPU and FBGEMM_GPU-CPU
  local files_to_skip=(
    test_utils.py
    split_table_batched_embeddings_test.py
    ssd_split_table_batched_embeddings_test.py
  )

  if [ "$fbgemm_variant" == "cpu" ]; then
    # These are tests that are currently broken in FBGEMM_GPU-CPU
    local ignored_tests=(
      uvm_test.py
    )
  elif [ "$fbgemm_variant" == "rocm" ]; then
    # https://github.com/pytorch/FBGEMM/issues/1559
    local ignored_tests=(
      batched_unary_embeddings_test.py
    )
  else
    local ignored_tests=()
  fi

  echo "[TEST] Installing pytest ..."
  print_exec conda install -n "${env_name}" -y pytest

  echo "[TEST] Checking imports ..."
  (test_python_import "${env_name}" fbgemm_gpu) || return 1
  (test_python_import "${env_name}" fbgemm_gpu.split_embedding_codegen_lookup_invokers) || return 1

  echo "[TEST] Enumerating test files ..."
  print_exec ls -lth ./*.py

  # NOTE: Tests running on single CPU core with a less powerful testing GPU in
  # GHA can take up to 5 hours.
  for test_file in *.py; do
    if echo "${files_to_skip[@]}" | grep "${test_file}"; then
      echo "[TEST] Skipping test file known to be broken: ${test_file}"
    elif echo "${ignored_tests[@]}" | grep "${test_file}"; then
      echo "[TEST] Skipping test file: ${test_file}"
    elif run_python_test "${env_name}" "${test_file}"; then
      echo ""
    else
      return 1
    fi
  done
}


################################################################################
# FBGEMM_GPU Publish Functions
################################################################################

publish_to_pypi () {
  local env_name="$1"
  local package_name="$2"
  local pypi_token="$3"
  if [ "$pypi_token" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME PYPI_TOKEN"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly-*.whl MY_TOKEN"
    echo ""
    echo "PYPI_TOKEN is missing!"
    return 1
  else
    echo "################################################################################"
    echo "# Publish to PyPI"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "[INSTALL] Installing twine ..."
  print_exec conda install -n "${env_name}" -y twine
  (test_python_import "${env_name}" twine) || return 1
  (test_python_import "${env_name}" OpenSSL) || return 1

  echo "[PUBLISH] Uploading package(s) to PyPI: ${package_name} ..."
  conda run -n "${env_name}" \
    python -m twine upload \
      --username __token__ \
      --password "${pypi_token}" \
      --skip-existing \
      --verbose \
      "${package_name}"

  echo "[PUBLISH] Successfully published package(s) to PyPI: ${package_name}"
  echo "[PUBLISH] NOTE: The publish command is a successful no-op if the wheel version already existed in PyPI; please double check!"
}

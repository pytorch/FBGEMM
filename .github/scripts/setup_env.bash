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
  "$@"
  echo ""
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

  if conda run -n "${env_name}" python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning "${python_test_file}"; then
    echo "[TEST] Python test suite PASSED: ${python_test_file}"
  else
    echo "[TEST] Python test suite FAILED: ${python_test_file}"
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
    fi
  fi
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


################################################################################
# Environment Setup and Install Functions
################################################################################

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
    print_exec wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

    echo "[SETUP] Installing Miniconda ..."
    print_exec bash miniconda.sh -b -p "$miniconda_prefix" -u
    print_exec rm -f miniconda.sh
  fi

  echo "[SETUP] Reloading the bash configuration ..."
  print_exec "${miniconda_prefix}/bin/conda" init bash
  print_exec . ~/.bashrc

  echo "[SETUP] Updating Miniconda base packages ..."
  (exec_with_retries conda update -n base -c defaults -y conda) || return 1

  # Print Conda info
  print_exec conda info

  # These variables will be exported outside
  export PATH="${miniconda_prefix}/bin:${PATH}"
  export CONDA="${miniconda_prefix}"

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

install_pytorch_conda () {
  local env_name="$1"
  local pytorch_version="$2"
  local pytorch_cpu="$3"
  if [ "$pytorch_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTORCH_VERSION [CPU]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 1.11.0      # Install a specific version"
    echo "    ${FUNCNAME[0]} build_env latest      # Install the latest stable release"
    echo "    ${FUNCNAME[0]} build_env test        # Install the pre-release"
    echo "    ${FUNCNAME[0]} build_env nightly 1   # Install the CPU variant of the nightly"
    return 1
  else
    echo "################################################################################"
    echo "# Install PyTorch (Conda)"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # Install cpuonly if needed
  if [ "$pytorch_cpu" != "" ]; then
    pytorch_cpu=1
    local pytorch_package="cpuonly pytorch"
  else
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

  # Install PyTorch packages
  echo "[INSTALL] Attempting to install '${pytorch_package}' (${pytorch_version}, CPU=${pytorch_cpu:-0}) through Conda using channel '${pytorch_channel}' ..."
  # shellcheck disable=SC2086
  (exec_with_retries conda install -n "${env_name}" -y ${pytorch_package} -c "${pytorch_channel}") || return 1

  # Run check for GPU variant
  if [ "$pytorch_cpu" == "" ]; then
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

  # Check that PyTorch is importable
  (test_python_import "${env_name}" torch.distributed) || return 1

  # Print out the actual installed PyTorch version
  installed_pytorch_version=$(conda run -n "${env_name}" python -c "import torch; print(torch.__version__)")
  echo "[INSTALL] Installed PyTorch through Conda"
  echo "[INSTALL] NOTE: The installed version is: ${installed_pytorch_version}"
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

  if [ "$pytorch_variant_type" != "cpu" ]; then
    if [ "$pytorch_variant_type" == "cuda" ]; then
      # Ensure that the PyTorch-CUDA headers are properly installed
      (test_filepath "${env_name}" cuda_cmake_macros.h) || return 1
    fi

    # Ensure that the PyTorch build is of the correct variant
    # This test usually applies to the PyTorch nightly builds
    if conda run -n build_binary pip list torch | grep torch | grep "${pytorch_variant}"; then
      echo "[CHECK] The installed PyTorch ${pytorch_version} is the correct variant (${pytorch_variant})"
    else
      echo "[CHECK] The installed PyTorch ${pytorch_version} appears to be an incorrect variant as it is missing references to ${pytorch_variant}!"
      echo "[CHECK] This can happen if the variant of PyTorch (e.g. GPU, nightly) for the MAJOR.MINOR version of CUDA presently installed on the system has not been published yet."
      return 1
    fi
  fi

  # Check that PyTorch is importable
  (test_python_import "${env_name}" torch.distributed) || return 1

  # Print out the actual installed PyTorch version
  installed_pytorch_version=$(conda run -n "${env_name}" python -c "import torch; print(torch.__version__)")
  echo "[INSTALL] Installed PyTorch through PIP"
  echo "[INSTALL] NOTE: The installed version is: ${installed_pytorch_version}"
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

  # Install CUDA packages
  echo "[INSTALL] Installing CUDA ${cuda_version} ..."
  (exec_with_retries conda install -n "${env_name}" -y cuda -c "nvidia/label/cuda-${cuda_version}") || return 1

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
  install_system_packages mesa-common-dev clang comgr libopenblas-dev jp intel-mkl-full locales libnuma-dev
  install_system_packages hipify-clang miopen-hip miopen-hip-dev

  echo "[INSTALL] Cleaning up ..."
  print_exec rm -f "${package_name}"

  echo "[INSTALL] Successfully installed ROCm ${rocm_version}"
}

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
    # Install gxx_linux-64 from main instead of cxx-compiler from conda-forge, as
    # the latter breaks builds:
    #   https://root-forum.cern.ch/t/error-timespec-get-has-not-been-declared-with-conda-root-package/45712/6
    #
    # NOTE: Install g++ 9.x instead of 11.x becaue 11.x builds libraries with
    # references to GLIBCXX_3.4.29, which is not available on systems with older
    # versions of libstdc++.so.6 such as CentOS Stream 8 and Ubuntu 20.04
    echo "[INSTALL] Installing C/C++ compilers through Conda ..."
    (exec_with_retries conda install -n "${env_name}" -y gxx_linux-64=9.3.0) || return 1

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
    install_pytorch_conda "${env_name}" "${pytorch_version}" 1
  else
    # Install CUDA and the GPU variant of PyTorch
    install_cuda "${env_name}" "${cuda_version}"
    install_pytorch_conda "${env_name}" "${pytorch_version}"
  fi
}


################################################################################
# Build Functions
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

  echo "[BUILD] Running git submodules update ..."
  git submodule sync
  git submodule update --init --recursive

  echo "[BUILD] Installing other build dependencies ..."
  (exec_with_retries conda run -n "${env_name}" python -m pip install -r requirements.txt) || return 1

  (test_python_import "${env_name}" numpy) || return 1
  (test_python_import "${env_name}" skbuild) || return 1

  echo "[BUILD] Successfully ran git submodules update"
}

__build_fbgemm_gpu_common_pre_steps () {
  # Private function that uses variables instantiated by its caller

  # Check C/C++ compilers are visible (the build scripts look specifically for `gcc`)
  (test_binpath "${env_name}" cc) || return 1
  (test_binpath "${env_name}" gcc) || return 1
  (test_binpath "${env_name}" c++) || return 1
  (test_binpath "${env_name}" g++) || return 1

  if [ "$fbgemm_variant" == "cpu" ]; then
    # Update the package name and build args depending on if CUDA is specified
    echo "[BUILD] Applying CPU-only build args ..."
    build_args=(--cpu_only)
    package_name="${package_name}-cpu"

  elif [ "$fbgemm_variant" == "rocm" ]; then
    (test_env_var "${env_name}" PYTORCH_ROCM_ARCH) || return 1

    echo "[BUILD] Applying ROCm build args ..."
    build_args=()
    package_name="${package_name}-rocm"

  else
    # Set to the default variant
    fbgemm_variant="gpu"

    # Check nvcc is visible
    (test_binpath "${env_name}" nvcc) || return 1

    # Check that cuDNN environment variables are available
    (test_env_var "${env_name}" CUDNN_INCLUDE_DIR) || return 1
    (test_env_var "${env_name}" CUDNN_LIBRARY) || return 1
    (test_env_var "${env_name}" NVML_LIB_PATH) || return 1

    # Build only CUDA 7.0 and 8.0 (i.e. V100 and A100) because of 100 MB binary size limits from PyPI.
    echo "[BUILD] Applying GPU build args ..."
    # shellcheck disable=SC2155
    local nvml_lib_path=$(conda run -n "${env_name}" printenv NVML_LIB_PATH)
    build_args=(
      --nvml_lib_path="${nvml_lib_path}"
      -DTORCH_CUDA_ARCH_LIST='7.0;8.0'
    )
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

check_fbgemm_gpu_build () {
  local fbgemm_variant="$1"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} FBGEMM_VARIANT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} cpu"
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
  if [ "${fbgemm_variant}" != "cpu" ]; then
    lib_symbols_to_check+=(
      fbgemm_gpu::asynchronous_inclusive_cumsum_gpu
      fbgemm_gpu::merge_pooled_embeddings
    )
  fi

  for library in "${fbgemm_gpu_so_files[@]}"; do
    echo "[CHECK] Listing out the GLIBCXX versions referenced by the library: ${library}"
    objdump -TC "${library}" | grep GLIBCXX | sed 's/.*GLIBCXX_\([.0-9]*\).*/GLIBCXX_\1/g' | sort -Vu | cat

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
  if [ "$package_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME [CPU_ONLY]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly       # Build the full wheel package"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly cpu   # Build the CPU-only variant of the wheel package"
    return 1
  else
    echo "################################################################################"
    echo "# Build FBGEMM-GPU Package (Wheel)"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # Run all the common FBGEMM-GPU build pre-steps (set up variables)
  __build_fbgemm_gpu_common_pre_steps || return 1

  # manylinux1_x86_64 is specified for PyPI upload
  # Distribute Python extensions as wheels on Linux
  echo "[BUILD] Building FBGEMM-GPU (VARIANT=${fbgemm_variant}) wheel ..."
  print_exec conda run -n "${env_name}" \
    python setup.py bdist_wheel \
      --package_name="${package_name}" \
      --python-tag="${python_tag}" \
      --plat-name=manylinux1_x86_64 \
      "${build_args[@]}"

  # Run checks on the built libraries
  (check_fbgemm_gpu_build "${fbgemm_variant}") || return 1

  echo "[BUILD] Enumerating the built wheels ..."
  print_exec ls -lth dist/*.whl

  echo "[BUILD] Enumerating the wheel SHAs ..."
  print_exec sha1sum dist/*.whl

  echo "[BUILD] FBGEMM-GPU build wheel completed"
}

build_fbgemm_gpu_install () {
  env_name="$1"
  fbgemm_variant="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [CPU_ONLY]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env      # Build + install the package"
    echo "    ${FUNCNAME[0]} build_env cpu  # Build + Install the CPU-only variant of the package"
    return 1
  else
    echo "################################################################################"
    echo "# Build + Install FBGEMM-GPU Package"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # Run all the common FBGEMM-GPU build pre-steps (set up variables)
  __build_fbgemm_gpu_common_pre_steps

  # Parallelism may need to be limited to prevent the build from being
  # canceled for going over ulimits
  echo "[BUILD] Building and installing FBGEMM-GPU (VARIANT=${fbgemm_variant}) ..."
  print_exec conda run -n "${env_name}" \
    python setup.py install "${build_args[@]}"

  # Run checks on the built libraries
  (check_fbgemm_gpu_build "${fbgemm_variant}") || return 1

  echo "[BUILD] FBGEMM-GPU build + install completed"
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
# Test Functions
################################################################################

run_fbgemm_gpu_tests () {
  local env_name="$1"
  local fbgemm_variant="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [FBGEMM_VARIANT]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env        # Run all tests applicable to GPU (Nvidia)"
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
    local ignored_tests=()
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
# Publish Functions
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

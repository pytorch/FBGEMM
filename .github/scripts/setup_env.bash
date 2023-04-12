#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_system.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/fbgemm_gpu_build.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/fbgemm_gpu_docs.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/fbgemm_gpu_lint.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/fbgemm_gpu_test.bash"


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

  echo "[SETUP] Listing existing Conda environments ..."
  print_exec conda info --envs

  # Occasionally, we run into `CondaValueError: Value error: prefix already exists`
  # We resolve this by pre-deleting the directory, if it exists:
  # https://stackoverflow.com/questions/40180652/condavalueerror-value-error-prefix-already-exists
  echo "[SETUP] Deleting the prefix directory if it exists ..."
  # shellcheck disable=SC2155
  local conda_prefix=$(conda run -n base printenv CONDA_PREFIX)
  print_exec rm -rf "${conda_prefix}/envs/${env_name}"

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

  # https://stackoverflow.com/questions/27686382/how-can-i-dump-all-nvcc-preprocessor-defines
  echo "[INFO] Printing out all preprocessor defines in nvcc ..."
  print_exec conda run -n "${env_name}" nvcc --compiler-options -dM -E -x cu - < /dev/null

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

  # https://stackoverflow.com/questions/2224334/gcc-dump-preprocessor-defines
  echo "[INFO] Printing out all preprocessor defines in the C compiler ..."
  print_exec conda run -n "${env_name}" cc -dM -E -

  # https://stackoverflow.com/questions/2224334/gcc-dump-preprocessor-defines
  echo "[INFO] Printing out all preprocessor defines in the C++ compiler ..."
  print_exec conda run -n "${env_name}" c++ -dM -E -x c++ -

  # Print out the C++ version
  print_exec conda run -n "${env_name}" c++ --version

  # https://stackoverflow.com/questions/4991707/how-to-find-my-current-compilers-standard-like-if-it-is-c90-etc
  echo "[INFO] Printing the default version of the C standard used by the compiler ..."
  print_exec conda run -n "${env_name}" cc -dM -E - | grep __STDC_VERSION__

  # https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
  echo "[INFO] Printing the default version of the C++ standard used by the compiler ..."
  print_exec conda run -n "${env_name}" c++ -dM -E -x c++ - | grep __cplusplus

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

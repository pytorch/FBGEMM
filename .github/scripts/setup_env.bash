#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


################################################################################
# Utility Functions
################################################################################

print_exec () {
  echo "+ $*"
  echo ""
  "$@"
}

test_python_import () {
  env_name="$1"
  python_import="$2"
  if [ "$python_import" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTHON_IMPORT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env numpy"
    return 1
  fi

  if conda run -n "${env_name}" python -c "import ${python_import}"; then
    echo "[CHECK] Python package ${python_import} found"
  else
    echo "[CHECK] Python package ${python_import} not found!"
    return 1
  fi
}

test_binpath () {
  env_name="$1"
  bin_name="$2"
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
  env_name="$1"
  file_name="$2"
  if [ "$file_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME FILE_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env cuda_runtime.h"
    return 1
  fi

  conda_prefix=$(conda run -n "${env_name}" printenv CONDA_PREFIX)
  file_path=$(find "${conda_prefix}" -type f -name "${file_name}")
  link_path=$(find "${conda_prefix}" -type l -name "${file_name}")
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
  env_name="$1"
  env_key="$2"
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

print_system_info () {
  echo "################################################################################"
  echo "# Print System Info"
  echo "#"
  echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
  echo "################################################################################"
  echo ""

  echo "[INFO] Check ldd version"
  print_exec ldd --version

  echo "[INFO] Check CPU info"
  print_exec nproc
  print_exec cat /proc/cpuinfo

  echo "[INFO] Check Linux distribution info"
  print_exec uname -a
  print_exec cat /proc/version
  print_exec cat /etc/os-release

  echo "[INFO] Check GPU info"
  if which apt-get; then
    print_exec sudo apt-get install -y lshw
  else
    print_exec sudo yum install -y lshw
  fi

  print_exec sudo lshw -C display
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
    category=$1
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
  miniconda_prefix="$1"
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
  print_exec conda update -n base -c defaults -y conda

  # Print Conda info
  print_exec conda info

  # These variables will be exported outside
  export PATH="${miniconda_prefix}/bin:${PATH}"
  export CONDA="${miniconda_prefix}"

  echo "[SETUP] Successfully set up Miniconda at ${miniconda_prefix}"
}

create_conda_environment () {
  env_name="$1"
  python_version="$2"
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

  # -y removes any existing conda environment with the same name
  echo "[SETUP] Creating new Conda environment (Python ${python_version}) ..."
  print_exec conda create -y --name "${env_name}" python="${python_version}"

  echo "[SETUP] Installed Python version: $(conda run -n "${env_name}" python --version)"
  echo "[SETUP] Successfully created Conda environment: ${env_name}"
}

install_pytorch_conda () {
  env_name="$1"
  pytorch_version="$2"
  pytorch_cpu="$3"
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
    pytorch_package="cpuonly pytorch"
  else
    pytorch_package="pytorch"
  fi

  # Set package name and installation channel
  if [ "$pytorch_version" == "nightly" ] || [ "$pytorch_version" == "test" ]; then
    pytorch_channel="pytorch-${pytorch_version}"
  elif [ "$pytorch_version" == "latest" ]; then
    pytorch_channel="pytorch"
  else
    pytorch_package="${pytorch_package}==${pytorch_version}"
    pytorch_channel="pytorch"
  fi

  # Install PyTorch packages
  echo "[INSTALL] Attempting to install '${pytorch_package}' (${pytorch_version}, CPU=${pytorch_cpu:-0}) through Conda using channel '${pytorch_channel}' ..."
  # shellcheck disable=SC2086
  print_exec conda install -n "${env_name}" -y ${pytorch_package} -c "${pytorch_channel}"

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
    test_filepath "${env_name}" cuda_cmake_macros.h || return 1
  fi

  # Check that PyTorch is importable
  test_python_import "${env_name}" torch.distributed || return 1

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
  print_exec conda run -n "${env_name}" pip install ${pytorch_package} --extra-index-url ${pytorch_channel}

  if [ "$pytorch_variant_type" != "cpu" ]; then
    if [ "$pytorch_variant_type" == "cuda" ]; then
      # Ensure that the PyTorch-CUDA headers are properly installed
      test_filepath "${env_name}" cuda_cmake_macros.h || return 1
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
  test_python_import "${env_name}" torch.distributed || return 1

  # Print out the actual installed PyTorch version
  installed_pytorch_version=$(conda run -n "${env_name}" python -c "import torch; print(torch.__version__)")
  echo "[INSTALL] Installed PyTorch through PIP"
  echo "[INSTALL] NOTE: The installed version is: ${installed_pytorch_version}"
}

install_cuda () {
  env_name="$1"
  cuda_version="$2"

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
  cuda_version_arr=(${cuda_version//./ })
  if [ ${#cuda_version_arr[@]} -lt 3 ]; then
    echo "[ERROR] CUDA minor version number must be specified (i.e. X.Y.Z)"
    return 1
  fi

  # Install CUDA packages
  echo "[INSTALL] Installing CUDA ${cuda_version} ..."
  print_exec conda install -n "${env_name}" -y cuda -c "nvidia/label/cuda-${cuda_version}"

  # Ensure that nvcc is properly installed
  test_binpath "${env_name}" nvcc || return 1

  # Ensure that the CUDA headers are properly installed
  test_filepath "${env_name}" cuda_runtime.h || return 1

  # Ensure that the libraries are properly installed
  test_filepath "${env_name}" libnvToolsExt.so || return 1

  # LIBNVTOOLSEXT
  # CUDA_TOOLKIT_ROOT_DIR
  # print_exec conda env config vars set -n "${env_name}" CUDNN_INCLUDE_DIR="${install_path}/include" CUDNN_LIBRARY="${install_path}/lib"

  # Print nvcc version
  print_exec conda run -n "${env_name}" nvcc --version
  echo "[INSTALL] Successfully installed CUDA ${cuda_version}"
}

install_cxx_compiler () {
  env_name="$1"
  use_yum="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [USE_YUM]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env     # Install C/C++ compilers through Conda"
    echo "    ${FUNCNAME[0]} build_env 1   # Install C/C++ compilers through yum"
    return 1
  else
    echo "################################################################################"
    echo "# Install C/C++ Compilers"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  if [ "$use_yum" != "" ]; then
    echo "[INSTALL] Installing C/C++ compilers through Yum ..."
    print_exec sudo yum update -y
    print_exec sudo yum install -y gcc gcc-c++
  else
    # Install gxx_linux-64 from main instead of cxx-compiler from conda-forge, as
    # the latter breaks builds:
    #   https://root-forum.cern.ch/t/error-timespec-get-has-not-been-declared-with-conda-root-package/45712/6
    echo "[INSTALL] Installing C/C++ compilers through Conda ..."
    print_exec conda install -n "${env_name}" -y gxx_linux-64

    # The compilers are visible in the PATH as `x86_64-conda-linux-gnu-cc` and
    # `x86_64-conda-linux-gnu-c++`, so symlinks will need to be created
    echo "[INSTALL] Setting the C/C++ compiler symlinks ..."
    cc_path=$(conda run -n "${env_name}" printenv CC)
    cxx_path=$(conda run -n "${env_name}" printenv CXX)

    print_exec ln -s "${cc_path}" "$(dirname "$cc_path")/cc"
    print_exec ln -s "${cc_path}" "$(dirname "$cc_path")/gcc"
    print_exec ln -s "${cxx_path}" "$(dirname "$cxx_path")/c++"
    print_exec ln -s "${cxx_path}" "$(dirname "$cxx_path")/g++"
  fi

  # Check C/C++ compilers are visible
  test_binpath "${env_name}" cc || return 1
  test_binpath "${env_name}" gcc || return 1
  test_binpath "${env_name}" c++ || return 1
  test_binpath "${env_name}" g++ || return 1

  # Print out the C++ version
  print_exec conda run -n "${env_name}" c++ --version
  echo "[INSTALL] Successfully installed C/C++ compilers"
}

install_build_tools () {
  env_name="$1"
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
  print_exec conda install -n "${env_name}" -y \
    cmake \
    hypothesis \
    jinja2 \
    ninja \
    numpy \
    scikit-build \
    wheel

  # Check binaries are visible in the PAATH
  test_binpath "${env_name}" cmake || return 1
  test_binpath "${env_name}" ninja || return 1

  # Check Python packages are importable
  test_python_import "${env_name}" hypothesis || return 1
  test_python_import "${env_name}" jinja2 || return 1
  test_python_import "${env_name}" numpy || return 1
  test_python_import "${env_name}" skbuild || return 1
  test_python_import "${env_name}" wheel || return 1

  echo "[INSTALL] Successfully installed all the build tools"
}

install_cudnn () {
  env_name="$1"
  install_path="$2"
  cuda_version="$3"
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
  cudnn_packages=(
    ["115"]="https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz"
    ["116"]="https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.2/local_installers/11.5/cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive.tar.xz"
    ["117"]="https://ossci-linux.s3.amazonaws.com/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz"
    ["118"]="https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz"
  )

  # Split version string by dot into array, i.e. 11.7.1 => [11, 7, 1]
  # shellcheck disable=SC2206
  cuda_version_arr=(${cuda_version//./ })
  # Fetch the major and minor version to concat
  cuda_concat_version="${cuda_version_arr[0]}${cuda_version_arr[1]}"

  # Get the URL
  cudnn_url="${cudnn_packages[cuda_concat_version]}"
  if [ "$cudnn_url" == "" ]; then
    # Default to cuDNN for 11.7 if no CUDA version fits
    echo "[INSTALL] Defaulting to cuDNN for CUDA 11.7"
    cudnn_url="${cudnn_packages[117]}"
  fi

  # Clear the install path
  rm -rf "$install_path"
  mkdir -p "$install_path"

  # Create temporary directory
  tmp_dir=$(mktemp -d)
  cd "$tmp_dir" || return 1

  # Download cuDNN
  echo "[INSTALL] Downloading cuDNN to ${tmp_dir} ..."
  print_exec wget -q "$cudnn_url" -O cudnn.tar.xz

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
  env_name="$1"
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
  print_exec conda run -n "${env_name}" python -m pip install -r requirements.txt

  test_python_import "${env_name}" numpy || return 1
  test_python_import "${env_name}" skbuild || return 1

  echo "[BUILD] Successfully ran git submodules update"
}

__build_fbgemm_gpu_common_pre_steps () {
  # Private function that uses variables instantiated by its caller

  # Check C/C++ compilers are visible (the build scripts look specifically for `gcc`)
  test_binpath "${env_name}" cc || return 1
  test_binpath "${env_name}" gcc || return 1
  test_binpath "${env_name}" c++ || return 1
  test_binpath "${env_name}" g++ || return 1

  if [ "$cpu_only" != "" ]; then
    # Update the package name and build args depending on if CUDA is specified
    echo "[BUILD] Applying CPU-only build args ..."
    cpu_only=1
    build_args="--cpu_only"
    package_name="${package_name}-cpu"
  else
    # Check nvcc is visible
    test_binpath "${env_name}" nvcc || return 1

    # Check that cuDNN environment variables are available
    test_env_var "${env_name}" CUDNN_INCLUDE_DIR || return 1
    test_env_var "${env_name}" CUDNN_LIBRARY || return 1

    # Build only CUDA 7.0 and 8.0 (i.e. V100 and A100) because of 100 MB binary size limits from PyPI.
    echo "[BUILD] Applying GPU build args ..."
    build_args="-DTORCH_CUDA_ARCH_LIST='7.0;8.0'"
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
}

build_fbgemm_gpu_package () {
  env_name="$1"
  package_name="$2"
  cpu_only="$3"
  if [ "$package_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME [CPU_ONLY]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly    # Build the full wheel package"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly 1  # Build the CPU-only variant of the wheel package"
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
  __build_fbgemm_gpu_common_pre_steps

  # manylinux1_x86_64 is specified for PyPI upload
  # Distribute Python extensions as wheels on Linux
  echo "[BUILD] Building FBGEMM-GPU (CPU=${cpu_only:-0}) wheel ..."
  print_exec conda run -n "${env_name}" \
    python setup.py bdist_wheel \
      --package_name="${package_name}" \
      --python-tag="${python_tag}" \
      --plat-name=manylinux1_x86_64 \
      "${build_args}"

  echo "[BUILD] Enumerating the built wheels ..."
  print_exec ls -lth dist/*.whl

  echo "[BUILD] FBGEMM-GPU build wheel completed"
}

build_fbgemm_gpu_install () {
  env_name="$1"
  cpu_only="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [CPU_ONLY]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env      # Build + install the package"
    echo "    ${FUNCNAME[0]} build_env 1    # Build + Install the CPU-only variant of the package"
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
  echo "[BUILD] Building and installing FBGEMM-GPU (CPU=${cpu_only:-0}) ..."
  print_exec conda run -n "${env_name}" \
    python setup.py install "${build_args}"

  echo "[BUILD] FBGEMM-GPU build + install completed"
}

################################################################################
# Publish Functions
################################################################################

publish_to_pypi () {
  env_name="$1"
  package_name="$2"
  pypi_token="$3"
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
  test_python_import "${env_name}" twine || return 1

  echo "[PUBLISH] Uploading package(s) to PyPI: ${package_name} ..."
  conda run -n "${env_name}" \
    python -m twine upload \
      --username __token__ \
      --password "${pypi_token}" \
      --skip-existing \
      --verbose \
      "${package_name}"

  echo "[PUBLISH] Successfully published package(s) to PyPI: ${package_name}"
}

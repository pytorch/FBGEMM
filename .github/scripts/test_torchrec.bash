#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Exit on failure
set -e

# shellcheck source=/dev/null
. "$(dirname "$(realpath -s "$0")")/setup_env.bash"

create_conda_pytorch_environment () {
  local env_name="$1"
  local python_version="$2"
  local pytorch_channel_name="$3"
  local cuda_version="$4"
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

verbose=0
torchrec_package_name=""
python_version=""
cuda_version="x"
fbgemm_wheel_path="x"
miniconda_prefix="${HOME}/miniconda"

usage () {
  echo "Usage: bash test_torchrec.bash -o PACKAGE_NAME -p PYTHON_VERSION -P PYTORCH_CHANNEL_NAME -c CUDA_VERSION -w FBGEMM_WHEEL_PATH [-m MINICONDA_PREFIX] [-v] [-h]"
  echo "-v                  : verbose"
  echo "-h                  : help"
  echo "PACKAGE_NAME        : output package name of TorchRec (e.g., torchrec_nightly)"
  echo "                      Note: TorchRec is sensitive to its package name"
  echo "                      e.g., torchrec needs fbgemm-gpu while torchrec_nightly needs fbgemm-gpu-nightly"
  echo "PYTHON_VERSION      : Python version (e.g., 3.8, 3.9, 3.10)"
  echo "PYTORCH_CHANNEL_NAME: PyTorch's channel name (e.g., pytorch-nightly, pytorch-test (=pre-release), pytorch (=stable release))"
  echo "CUDA_VERSION        : PyTorch's CUDA version (e.g., 11.6, 11.7)"
  echo "FBGEMM_WHEEL_PATH   : path to FBGEMM_GPU's wheel file"
  echo "MINICONDA_PREFIX    : path to install Miniconda (default: \$HOME/miniconda)"
  echo "Example: Python 3.10 + PyTorch nightly (CUDA 11.7), install miniconda at \$HOME/miniconda, using dist/fbgemm_gpu_nightly.whl"
  echo "       bash test_torchrec.bash -v -o torchrec_nightly -p 3.10 -P pytorch-nightly -c 11.7 -w dist/fbgemm_gpu_nightly.whl"
}

while getopts vho:p:P:c:m:w: flag
do
    case "$flag" in
        v) verbose="1";;
        o) torchrec_package_name="${OPTARG}";;
        p) python_version="${OPTARG}";;
        P) pytorch_channel_name="${OPTARG}";;
        c) cuda_version="${OPTARG}";;
        m) miniconda_prefix="${OPTARG}";;
        w) fbgemm_wheel_path="${OPTARG}";;
        h) usage
           exit 0;;
        *) usage
           exit 1;;
    esac
done

if [ "$torchrec_package_name" == "" ] || [ "$python_version" == "" ] || [ "$cuda_version" == "x" ] || [ "$miniconda_prefix" == "" ] || [ "$pytorch_channel_name" == "" ] || [ "$fbgemm_wheel_path" == "" ]; then
  usage
  exit 1
fi
python_tag="${python_version//\./}"

if [ "$verbose" == "1" ]; then
  # Print each line verbosely
  set -x -e
fi

################################################################################
echo "## 0. Minimal check"
################################################################################

if [ ! -d "torchrec" ]; then
  echo "Error: this script must be executed in torchrec/"
  exit 1
fi

################################################################################
echo "## 1. Set up Miniconda"
################################################################################

setup_miniconda "$miniconda_prefix"

################################################################################
echo "## 2. Create test_binary environment"
################################################################################

create_conda_pytorch_environment test_binary "$python_version" "$pytorch_channel_name" "$cuda_version"

# Comment out FBGEMM_GPU since we will install it from "$fbgemm_wheel_path"
sed -i 's/fbgemm-gpu/#fbgemm-gpu/g' requirements.txt
conda run -n test_binary python -m pip install -r requirements.txt
# Install FBGEMM_GPU from a local wheel file.
conda run -n test_binary python -m pip install "$fbgemm_wheel_path"
conda run -n test_binary python -c "import fbgemm_gpu"

################################################################################
echo "## 3. Build TorchRec"
################################################################################

rm -rf dist
conda run -n test_binary python setup.py bdist_wheel --package_name "${torchrec_package_name}" --python-tag="py${python_tag}"

################################################################################
echo "## 4. Import TorchRec"
################################################################################

conda run -n test_binary python -m pip install dist/"${torchrec_package_name}"*.whl
conda run -n test_binary python -c "import torchrec"

echo "Test succeeded"

#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Exit on failure
set -e

# shellcheck source=/dev/null
. "$(dirname "$(realpath -s "$0")")/setup_env.bash"

verbose=0
package_name=""
python_version=""
pytorch_channel_name=""
cuda_version="x"
miniconda_prefix="${HOME}/miniconda"

usage () {
  echo "Usage: bash build_wheel.bash -o PACKAGE_NAME -p PYTHON_VERSION -P PYTORCH_CHANNEL_NAME -c CUDA_VERSION [-m MINICONDA_PREFIX] [-v] [-h]"
  echo "-v                  : verbose"
  echo "-h                  : help"
  echo "PACKAGE_NAME        : output package name (e.g., fbgemm_gpu_nightly)"
  echo "PYTHON_VERSION      : Python version (e.g., 3.8, 3.9, 3.10)"
  echo "PYTORCH_CHANNEL_NAME: PyTorch's channel name (e.g., pytorch-nightly, pytorch-test (=pre-release), pytorch (=stable release))"
  echo "CUDA_VERSION        : PyTorch's CUDA version (e.g., 11.6, 11.7)"
  echo "MINICONDA_PREFIX    : path to install Miniconda (default: \$HOME/miniconda)"
  echo "Example 1: Python 3.10 + PyTorch nightly (CUDA 11.7), install miniconda at /home/user/tmp/miniconda"
  echo "       bash build_wheel.bash -v -P pytorch-nightly -p 3.10 -c 11.7 -m /home/user/tmp/miniconda"
  echo "Example 2: Python 3.10 + PyTorch stable (CPU), install miniconda at \$HOME/miniconda"
  echo "       bash build_wheel.bash -v -P pytorch -p 3.10 -c \"\""
}

while getopts vfho:p:P:c:m: flag
do
    case "$flag" in
        v) verbose="1";;
        o) package_name="${OPTARG}";;
        p) python_version="${OPTARG}";;
        P) pytorch_channel_name="${OPTARG}";;
        c) cuda_version="${OPTARG}";;
        m) miniconda_prefix="${OPTARG}";;
        h) usage
           exit 0;;
        *) usage
           exit 1;;
    esac
done

if [ "$python_version" == "" ] || [ "$cuda_version" == "x" ] || [ "$miniconda_prefix" == "" ] || [ "$pytorch_channel_name" == "" ] || [ "$package_name" == "" ]; then
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

if [ ! -d "fbgemm_gpu" ]; then
  echo "Error: this script must be executed in FBGEMM/"
  exit 1
elif [ "$(which gcc 2>/dev/null)" == "" ]; then
  echo "Error: GCC is needed to compile FBGEMM"
  exit 1
fi

################################################################################
echo "## 1. Set up Miniconda"
################################################################################

setup_miniconda "$miniconda_prefix"

################################################################################
echo "## 2. Create build_binary environment"
################################################################################

create_conda_pytorch_environment build_binary "$python_version" "$pytorch_channel_name" "$cuda_version"

cd fbgemm_gpu

# cuDNN is needed to "build" FBGEMM
install_cudnn build_binary "$miniconda_prefix/build_only/cudnn" "$cuda_version"

conda run -n build_binary python -m pip install -r requirements.txt

# TODO: Do we need these checks?
ldd --version
conda info
conda run -n build_binary python --version
gcc --version
conda run -n build_binary python -c "import torch.distributed"
conda run -n build_binary python -c "import skbuild"
conda run -n build_binary python -c "import numpy"
cd ../

################################################################################
echo "## 3. Build FBGEMM_GPU"
################################################################################

cd fbgemm_gpu
rm -rf dist _skbuild
if [ "$cuda_version" == "" ]; then
  # CPU version
  build_arg="--cpu_only"
  package_name="${package_name}_cpu"
else
  # GPU version
  # We build only CUDA 7.0 and 8.0 (i.e., for v100 and a100) because of 100 MB binary size limit from PYPI website.
  build_arg="-DTORCH_CUDA_ARCH_LIST=7.0;8.0"
fi

# manylinux1_x86_64 is specified for pypi upload: distribute python extensions as wheels on Linux
conda run -n build_binary python setup.py bdist_wheel --package_name="${package_name}" --python-tag="py${python_tag}" "${build_arg}" --plat-name=manylinux1_x86_64
cd ../

# Usage:
#     pip install $(ls fbgemm_gpu/dist/${package_name}-*.whl)
#     python -c "import fbgemm_gpu"

wheel_name="$(ls fbgemm_gpu/dist/"${package_name}"-*.whl)"
echo "Successfully built $wheel_name"

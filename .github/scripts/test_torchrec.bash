#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Exit on failure
set -e

verbose=0
torchrec_package_name=""
python_version=""
cuda_version="x"
miniconda_prefix="${HOME}/miniconda"

usage () {
  # shellcheck disable=SC2086
  echo "Usage: bash $(basename ${BASH_SOURCE[0]}) -o PACKAGE_NAME -p PYTHON_VERSION -P PYTORCH_CHANNEL_NAME -c CUDA_VERSION [-m MINICONDA_PREFIX] [-v] [-h]"
  echo "-v                  : verbose"
  echo "-h                  : help"
  echo "PACKAGE_NAME        : output package name of TorchRec (e.g., torchrec_nightly)"
  echo "                      Note: TorchRec is sensitive to its package name"
  echo "                      e.g., torchrec needs fbgemm-gpu while torchrec_nightly needs fbgemm-gpu-nightly"
  echo "PYTHON_VERSION      : Python version (e.g., 3.10)"
  echo "PYTORCH_CHANNEL_NAME: PyTorch's channel name (e.g., pytorch-nightly, pytorch-test (=pre-release), pytorch (=stable release))"
  echo "CUDA_VERSION        : PyTorch's CUDA version (e.g., 12.4)"
  echo "MINICONDA_PREFIX    : path to install Miniconda (default: \$HOME/miniconda)"
  echo "Example: Python 3.10 + PyTorch nightly (CUDA 12.4), install miniconda at \$HOME/miniconda, using dist/fbgemm_gpu_nightly.whl"
  # shellcheck disable=SC2086
  echo "       bash $(basename ${BASH_SOURCE[0]}) -v -o torchrec_nightly -p 3.10 -P pytorch-nightly -c 11.7 -w dist/fbgemm_gpu_nightly.whl"
}

while getopts vho:p:P:c:m:b: flag
do
    case "$flag" in
        v) verbose="1";;
        o) torchrec_package_name="${OPTARG}";;
        p) python_version="${OPTARG}";;
        P) pytorch_channel_name="${OPTARG}";;
        c) cuda_version="${OPTARG}";;
        m) miniconda_prefix="${OPTARG}";;
        b) build_env="${OPTARG}";;
        h) usage
           exit 0;;
        *) usage
           exit 1;;
    esac
done

if [ "$torchrec_package_name" == "" ] || [ "$python_version" == "" ] || [ "$cuda_version" == "x" ] || [ "$miniconda_prefix" == "" ] || [ "$pytorch_channel_name" == "" ] || [ "$build_env" == "" ]; then
  usage
  exit 1
fi

env_name=$build_env
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

# Install PyTorch
conda run -n "$env_name" pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
conda run -n "$env_name" python -c "import torch"

# Import torch.distributed
conda run -n "$env_name" python -c "import torch.distributed"

# Import fbgemm_gpu
conda run -n "$env_name" pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cpu
conda run -n "$env_name" python -c "import fbgemm_gpu"

################################################################################
echo "## 1. Install TorchRec Requirements"
################################################################################
# Comment out FBGEMM_GPU since we should pre-install it from the downloaded wheel file
sed -i 's/fbgemm-gpu/#fbgemm-gpu/g' requirements.txt
conda run -n "$env_name" python -m pip install -r requirements.txt


################################################################################
echo "## 2. Build TorchRec"
################################################################################

rm -rf dist
conda run -n "$env_name" python setup.py bdist_wheel --python-tag="py${python_tag}"

################################################################################
echo "## 3. Import TorchRec"
################################################################################

conda run -n "$env_name" python -c "import torchrec"

echo "Test succeeded"

################################################################################
echo "## 4. Run TorchRec tests"
################################################################################

conda install -n "$env_name" -y pytest
# Read the list of tests to skip from a file, ignoring empty lines and comments
skip_expression=$(awk '!/^($|#)/ {printf " and not %s", $0}' ./.github/scripts/tests_to_skip.txt)
# Check if skip_expression is effectively empty
if [ -z "$skip_expression" ]; then
  skip_expression=""
else
  skip_expression=${skip_expression:5}  # Remove the leading " and "
fi
conda run -n "$env_name" \
  python -m pytest torchrec -v -s -W ignore::pytest.PytestCollectionWarning --continue-on-collection-errors \
  --ignore-glob=**/test_utils/ -k "$skip_expression"

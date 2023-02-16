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
python_version=""
cuda_version="x"
fbgemm_wheel_path="x"
miniconda_prefix="${HOME}/miniconda"

usage () {
  echo "Usage: bash test_wheel.bash -p PYTHON_VERSION -P PYTORCH_CHANNEL_NAME -c CUDA_VERSION -w FBGEMM_WHEEL_PATH [-m MINICONDA_PREFIX] [-v] [-h]"
  echo "-v                  : verbose"
  echo "-h                  : help"
  echo "PYTHON_VERSION      : Python version (e.g., 3.8, 3.9, 3.10)"
  echo "PYTORCH_CHANNEL_NAME: PyTorch's channel name (e.g., pytorch-nightly, pytorch-test (=pre-release), pytorch (=stable release))"
  echo "CUDA_VERSION        : PyTorch's CUDA version (e.g., 11.6, 11.7)"
  echo "FBGEMM_WHEEL_PATH   : path to FBGEMM_GPU's wheel file"
  echo "MINICONDA_PREFIX    : path to install Miniconda (default: \$HOME/miniconda)"
  echo "Example 1: Python 3.10 + PyTorch nightly (CUDA 11.7), install miniconda at /home/user/tmp/miniconda, using dist/fbgemm_gpu.whl"
  echo "       bash test_wheel.bash -v -p 3.10 -P pytorch-nightly -c 11.7 -m /home/user/tmp/miniconda -w dist/fbgemm_gpu.whl"
  echo "Example 2: Python 3.10 + PyTorch stable (CPU), install miniconda at \$HOME/miniconda, using /tmp/fbgemm_gpu_cpu.whl"
  echo "       bash test_wheel.bash -v -p 3.10 -P pytorch -c \"\" -w /tmp/fbgemm_gpu_cpu.whl"
}

while getopts vhp:P:c:m:w: flag
do
    case "$flag" in
        v) verbose="1";;
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

if [ "$python_version" == "" ] || [ "$cuda_version" == "x" ] || [ "$miniconda_prefix" == "" ] || [ "$pytorch_channel_name" == "" ] || [ "$fbgemm_wheel_path" == "" ]; then
  usage
  exit 1
fi

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
fi

################################################################################
echo "## 1. Set up Miniconda"
################################################################################

setup_miniconda "$miniconda_prefix"

################################################################################
echo "## 2. Create test_binary environment"
################################################################################

create_conda_pytorch_environment test_binary "$python_version" "$pytorch_channel_name" "$cuda_version"
conda install -n test_binary -y pytest

cd fbgemm_gpu
conda run -n test_binary python -m pip install -r requirements.txt
cd ../

################################################################################
echo "## 3. Install and test FBGEMM_GPU"
################################################################################

conda run -n test_binary python -m pip install "$fbgemm_wheel_path"
conda run -n test_binary python -c "import fbgemm_gpu"

if [ "$cuda_version" == "" ]; then
  # CPU version: unfortunately, not all tests are properly excluded,
  # so we cherry-pick what we can run.
  conda run -n test_binary python fbgemm_gpu/test/batched_unary_embeddings_test.py -v
  conda run -n test_binary python fbgemm_gpu/test/input_combine_test.py -v
  conda run -n test_binary python fbgemm_gpu/test/layout_transform_ops_test.py -v
  conda run -n test_binary python fbgemm_gpu/test/merge_pooled_embeddings_test.py -v
  conda run -n test_binary python fbgemm_gpu/test/permute_pooled_embedding_modules_test.py -v
  conda run -n test_binary python fbgemm_gpu/test/quantize_ops_test.py -v
  conda run -n test_binary python fbgemm_gpu/test/sparse_ops_test.py -v
else
  # GPU version
  # Don't run it in the fbgemm_gpu directory; fbgemm_gpu has a fbgemm_gpu directory,
  # which confuses "import" in Python.
  # conda run -n test_binary python -m pytest fbgemm_gpu -v -s -W ignore::pytest.PytestCollectionWarning --continue-on-collection-errors
  conda run -n test_binary python -m pytest fbgemm_gpu -v -s -W ignore::pytest.PytestCollectionWarning --continue-on-collection-errors --ignore-glob=**/ssd_split_table_batched_embeddings_test.py --ignore-glob=**/split_table_batched_embeddings_test.py
fi

echo "Test succeeded"

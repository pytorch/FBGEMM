#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Exit on failure
set -e

verbose=0
python_version=""

usage () {
  # shellcheck disable=SC2086
  echo "Usage: bash $(basename ${BASH_SOURCE[0]}) -p PYTHON_VERSION -b BUILD_ENV [-v] [-h]"
  echo "-v                  : verbose"
  echo "-h                  : help"
  echo "PYTHON_VERSION      : Python version (e.g., 3.10)"
  echo "BUILD_ENV           : build environment name (e.g., build_env)"
  echo "Example: Run torchrec tests with Python 3.10 and dist/*.wfl fbgemm wheel"
  # shellcheck disable=SC2086
  echo "       bash $(basename ${BASH_SOURCE[0]}) -v -o torchrec_nightly -p 3.10 -P pytorch-nightly -c 11.7 -w dist/fbgemm_gpu_nightly.whl"
}

while getopts vho:p:P:b: flag
do
    case "$flag" in
        v) verbose="1";;
        p) python_version="${OPTARG}";;
        b) build_env="${OPTARG}";;
        h) usage
           exit 0;;
        *) usage
           exit 1;;
    esac
done

if [ "$python_version" == "" ] || [ "$build_env" == "" ]; then
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

echo "Importing FBGEMM-GPU..."
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

# Add test_dlrm_inference_package to skip expression
skip_expression="${skip_expression} and not test_dlrm_inference_package"

conda run -n "$env_name" \
  python -m pytest torchrec -v -s -W ignore::pytest.PytestCollectionWarning --continue-on-collection-errors \
  --ignore-glob=**/test_utils/ -k "$skip_expression"

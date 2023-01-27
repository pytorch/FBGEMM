#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eux

FBGEMM_REPO_DIR=${1:-/workspace/FBGEMM}

git config --global --add safe.directory "$FBGEMM_REPO_DIR"
git config --global --add safe.directory "$FBGEMM_REPO_DIR/third_party/asmjit"
git config --global --add safe.directory "$FBGEMM_REPO_DIR/third_party/cpuinfo"
git config --global --add safe.directory "$FBGEMM_REPO_DIR/third_party/googletest"
git config --global --add safe.directory "$FBGEMM_REPO_DIR/third_party/hipify_torch"

# Install dependencies
apt-get update --allow-insecure-repositories && \
  apt-get install -y --allow-unauthenticated \
  git \
  jq \
  sshfs \
  sshpass \
  unzip

apt-get install -y locales
locale-gen en_US.UTF-8

pip3 install click
pip3 install jinja2
pip3 install ninja
# scikit-build >=0.16.5 needs a newer CMake
pip3 install --upgrade cmake
pip3 install scikit-build
pip3 install --upgrade hypothesis
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.3/

pip3 list

# Build fbgemm_gpu
cd "$FBGEMM_REPO_DIR/fbgemm_gpu"
MAX_JOBS="$(nproc)"
export MAX_JOBS
export PYTORCH_ROCM_ARCH="gfx908"
python setup.py build develop

export FBGEMM_TEST_WITH_ROCM=1

# Test fbgemm_gpu
cd test

python batched_unary_embeddings_test.py --verbose
python input_combine_test.py --verbose
python jagged_tensor_ops_test.py --verbose
python layout_transform_ops_test.py --verbose
python merge_pooled_embeddings_test.py --verbose
python metric_ops_test.py --verbose
python permute_pooled_embedding_modules_test.py --verbose
python quantize_ops_test.py --verbose
python sparse_ops_test.py --verbose
python split_embedding_inference_converter_test.py --verbose
# test_nbit_forward_fused_pooled_emb_quant is failing. It's skipped in the test code
python split_table_batched_embeddings_test.py --verbose
python uvm_test.py --verbose

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

## Workaround for nova workflow to build wheels from fbgemm_gpu folder
FBGEMM_DIR="/__w/FBGEMM/FBGEMM"
export FBGEMM_REPO="${FBGEMM_DIR}/${REPOSITORY}"
working_dir=$(pwd)
export BUILD_FROM_NOVA=1
<<<<<<< HEAD
if [[ "$CONDA_ENV" != "" ]]; then export CONDA_RUN="conda run --no-capture-output -p ${CONDA_ENV}"; fi
if [[ "$CU_VERSION" == "cu118" ]]; then export TORCH_CUDA_ARCH_LIST='7.0;8.0'; fi
if [[ "$CU_VERSION" == "cu121" ]]; then export TORCH_CUDA_ARCH_LIST='7.0;8.0;9.0'; fi
if [[ "$working_dir" == "$FBGEMM_REPO" ]]; then cd fbgemm_gpu || echo "Failed to cd fbgemm_gpu from $(pwd)"; fi
echo $TORCH_CUDA_ARCH_LIST
echo $CONDA_RUN

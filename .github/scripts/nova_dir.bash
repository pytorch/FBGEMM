# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

## Workaround for Nova Workflow to look for setup.py in fbgemm_gpu rather than root repo
FBGEMM_DIR="/__w/FBGEMM/FBGEMM"
export FBGEMM_REPO="${FBGEMM_DIR}/${REPOSITORY}"
working_dir=$(pwd)
if [[ "$working_dir" == "$FBGEMM_REPO" ]]; then cd fbgemm_gpu || echo "Failed to cd fbgemm_gpu from $(pwd)"; fi

## Build clean/wheel will be done in pre-script. Set flag such that setup.py will skip these steps in Nova workflow
export BUILD_FROM_NOVA=1

## Overwrite existing ENV VAR in Nova
if [[ "$CONDA_ENV" != "" ]]; then export CONDA_RUN="conda run --no-capture-output -p ${CONDA_ENV}" && echo "$CONDA_RUN"; fi

if  [[ "$CU_VERSION" == "cu121" ]] ||
    [[ "$CU_VERSION" == "cu124" ]] ||
    [[ "$CU_VERSION" == "cu126" ]] ||
    [[ "$CU_VERSION" == "cu128" ]]; then
    export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0;9.0a"
    echo "Set TORCH_CUDA_ARCH_LIST to: ${TORCH_CUDA_ARCH_LIST}"

elif [[ "$CU_VERSION" == "cu118" ]]; then
    export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"
    echo "Set TORCH_CUDA_ARCH_LIST to: ${TORCH_CUDA_ARCH_LIST}"

elif [[ "$CU_VERSION" == "cu"* ]]; then
    echo "################################################################################"
    echo "[NOVA] Currently building the CUDA variant, but the supplied CU_VERSION is"
    echo "unknown or not supported in FBGEMM_GPU: ${CU_VERSION}"
    echo ""
    echo "Will default to the TORCH_CUDA_ARCH_LIST supplied by the environment!!!"
    echo "################################################################################"
fi

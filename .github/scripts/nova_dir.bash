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

if [[ "$CU_VERSION" == "cu"* ]]; then
    echo "Current TORCH_CUDA_ARCH_LIST value: ${TORCH_CUDA_ARCH_LIST}"
elif [[ "$CU_VERSION" == "rocm"* ]]; then
    echo "Current PYTORCH_ROCM_ARCH value: ${PYTORCH_ROCM_ARCH}"
fi

## Overwrite existing ENV VAR in Nova
if [[ "$CONDA_ENV" != "" ]]; then export CONDA_RUN="conda run --no-capture-output -p ${CONDA_ENV}" && echo "$CONDA_RUN"; fi

if [[ "$CU_VERSION" == "cu130" ]] ||
     [[ "$CU_VERSION" == "cu129" ]] ||
     [[ "$CU_VERSION" == "cu128" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;9.0a;10.0a;12.0a"
    echo "[NOVA] Set TORCH_CUDA_ARCH_LIST to: ${TORCH_CUDA_ARCH_LIST}"

elif [[ "$CU_VERSION" == "cu126" ]] ||
     [[ "$CU_VERSION" == "cu124" ]] ||
     [[ "$CU_VERSION" == "cu121" ]] ||
     [[ "$CU_VERSION" == "cu118" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;9.0a"
    echo "[NOVA] Set TORCH_CUDA_ARCH_LIST to: ${TORCH_CUDA_ARCH_LIST}"

elif [[ "$CU_VERSION" == "cu"* ]]; then
    echo "################################################################################"
    echo "[NOVA] Currently building the CUDA variant, but the supplied CU_VERSION is"
    echo "[NOVA] unknown or not supported in FBGEMM_GPU: ${CU_VERSION}"
    echo ""
    echo "[NOVA] Will default to the TORCH_CUDA_ARCH_LIST supplied by the environment!!!"
    echo "################################################################################"


elif [[ "$CU_VERSION" == "rocm7."* ]] ||
     [[ "$CU_VERSION" == "rocm10."* ]]; then
    export PYTORCH_ROCM_ARCH="gfx908,gfx90a,gfx942,gfx950"
    echo "[NOVA] Set PYTORCH_ROCM_ARCH to: ${PYTORCH_ROCM_ARCH}"

elif [[ "$CU_VERSION" == "rocm6.4"* ]] ||
     [[ "$CU_VERSION" == "rocm6.3"* ]] ||
     [[ "$CU_VERSION" == "rocm6.2"* ]]; then
    export PYTORCH_ROCM_ARCH="gfx908,gfx90a,gfx942"
    echo "[NOVA] Set PYTORCH_ROCM_ARCH to: ${PYTORCH_ROCM_ARCH}"

elif [[ "$CU_VERSION" == "rocm"* ]]; then
    echo "################################################################################"
    echo "[NOVA] Currently building the ROCm variant, but the supplied CU_VERSION is"
    echo "[NOVA] unknown or not supported in FBGEMM_GPU: ${CU_VERSION}"
    echo ""
    echo "[NOVA] Will default to the PYTORCH_ROCM_ARCH supplied by the environment!!!"
    echo "################################################################################"
fi

## Cap CUDA build parallelism by container memory, not just cores.
# nvcc peaks at several GiB per translation unit, so the core count alone
# overcommits a memory-limited container. 8GiB per TU, minus headroom for
# everything else in the container, is empirical.
if [[ "$CU_VERSION" == "cu"* ]] && [[ -n "${TORCH_CI_MAX_MEMORY:-}" ]]; then
    mem_gib=$(((TORCH_CI_MAX_MEMORY / 1073741824) - 8))
    max_jobs=$((mem_gib / 8))
    cores=$(nproc)
    if [[ "$max_jobs" -lt 1 ]]; then max_jobs=1; fi
    if [[ "$max_jobs" -gt "$cores" ]]; then max_jobs="$cores"; fi
    export BUILD_PARALLELISM="$max_jobs"
    echo "[NOVA] Set BUILD_PARALLELISM to ${BUILD_PARALLELISM} (${mem_gib}GiB usable / 8GiB per CUDA TU, ${cores} cores)"
fi

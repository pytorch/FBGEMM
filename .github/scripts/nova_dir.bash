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


elif [[ "$CU_VERSION" == "rocm7.0"* ]]; then
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

## Optional cgroup memory trace: set FBGEMM_MEMORY_TRACE=1 on the job.
#
# Builds that fit a 64GiB EC2 box at -j 16 are being OOM-killed in a 226Gi
# container at -j 27, so the limit is being hit by something other than the
# compilers' own anonymous memory. This samples the cgroup's own accounting so
# that can be read off directly rather than inferred: anon is what the
# compilers hold, file/file_dirty/file_writeback is page cache the build
# generates by writing object files, and memory.events counts how many times
# the limit was actually reached.
if [[ -n "${FBGEMM_MEMORY_TRACE:-}" ]] && [[ -r /sys/fs/cgroup/memory.current ]]; then
    (
        while true; do
            awk -v ts="$(date -u +%H:%M:%S)" \
                -v cur="$(cat /sys/fs/cgroup/memory.current)" \
                -v max="$(cat /sys/fs/cgroup/memory.max)" \
                -v ev="$(tr '\n' ' ' < /sys/fs/cgroup/memory.events)" '
                { stat[$1] = $2 }
                END {
                    g = 1073741824
                    printf "[mem] %s max=%.0fG current=%.1fG anon=%.1fG file=%.1fG dirty=%.2fG writeback=%.2fG inactive_file=%.1fG slab=%.1fG | %s\n",
                        ts, max/g, cur/g, stat["anon"]/g, stat["file"]/g,
                        stat["file_dirty"]/g, stat["file_writeback"]/g,
                        stat["inactive_file"]/g, stat["slab"]/g, ev
                }' /sys/fs/cgroup/memory.stat
            sleep 10
        done
    ) &
    echo "[NOVA] cgroup memory trace started (pid $!)"
fi

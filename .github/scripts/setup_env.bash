#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Define the list of script files to source
utils_scripts=(
    utils_base.bash
    utils_system.bash
    utils_build.bash
    utils_conda.bash
    utils_cuda.bash
    utils_pip.bash
    utils_rocm.bash
    utils_pytorch.bash
    utils_triton.bash
    utils_torchrec.bash
    fbgemm_build.bash
    fbgemm_gpu_build.bash
    fbgemm_gpu_docs.bash
    fbgemm_gpu_install.bash
    fbgemm_gpu_lint.bash
    fbgemm_gpu_test.bash
    fbgemm_gpu_benchmarks.bash
    fbgemm_gpu_integration.bash
)

# Loop and source each script
for script in "${utils_scripts[@]}"; do
    # shellcheck disable=SC1091,SC2128,SC1090
    . "$( dirname -- "$BASH_SOURCE"; )/${script}"
done

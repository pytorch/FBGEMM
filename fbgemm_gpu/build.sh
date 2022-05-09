#!/bin/bash

export MAX_JOBS=96
gpu_arch="$(/opt/rocm/bin/rocminfo | grep -o -m 1 'gfx.*')"
export PYTORCH_ROCM_ARCH=$gpu_arch
git clean -dfx
python setup.py build develop 2>&1 | tee build.log
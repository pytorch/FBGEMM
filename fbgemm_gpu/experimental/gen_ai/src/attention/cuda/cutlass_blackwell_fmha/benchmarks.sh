#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

options=(fwd_bf16 bwd_bf16 fwd_fp8 bwd_fp8)
batch_sizes=(32 16 8 4 2 1)
seq_lengths=(512 1024 2048 4096 8192 16384)
for option in "${options[@]}"; do
rm ${option}.txt
for i in "${!batch_sizes[@]}"; do
  batch=${batch_sizes[i]}
  seq=${seq_lengths[i]}
  echo "Running with batch size $batch and sequence length $seq"

  ~/fbsource/fbcode/triton/scripts/denoise.sh buck run @mode/opt -c fbcode.enable_gpu_sections=true -c fbcode.platform010_cuda_version=12.8 -c fbcode.nvcc_arch=b200a \
    ai_acceleration/kernels/attentions/cutlass_blackwell_fmha:blackwell_fmha_${option} -- \
    --b=$batch --h=16 --d=128 --k=$seq >> ${option}.txt
done
done


for option in "${options[@]}"; do
for i in "${!batch_sizes[@]}"; do
  batch=${batch_sizes[i]}
  seq=${seq_lengths[i]}
  echo "Running --mask=causal with batch size $batch and sequence length $seq"

  ~/fbsource/fbcode/triton/scripts/denoise.sh buck run @mode/opt -c fbcode.enable_gpu_sections=true -c fbcode.platform010_cuda_version=12.8 -c fbcode.nvcc_arch=b200a \
    ai_acceleration/kernels/attentions/cutlass_blackwell_fmha:blackwell_fmha_${option} -- \
    --b=$batch --h=16 --d=128 --mask=causal --k=$seq >> ${option}.txt
done
done

rm results.txt
for option in "${options[@]}"; do
  if [[ $option == *bwd* ]]; then
    echo "${option} bwd tma" >> results.txt
    grep "tma" ${option}.txt | awk '{gsub("TFLOPS/s", ""); print $NF}' >> results.txt
  else
    echo "${option} individual" >> results.txt
    grep "individual" ${option}.txt | awk '{gsub("TFLOPS/s", ""); print $NF}' >> results.txt
    echo "${option} persistent" >> results.txt
    grep "persistent" ${option}.txt | awk '{gsub("TFLOPS/s", ""); print $NF}' >> results.txt
  fi
done

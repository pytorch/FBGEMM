#!/bin/bash

common_opts="--bag-size 55 \
        --batch-size 65536 \
        --num-embeddings 19300000 \
        --num-tables 1 \
        --iters 5"

# Run on GPU and get PyTorch-level performance
for D in 64 128 192 256 512; do
  for fp in "fp32" "fp16"; do
    for alpha in 1 1.15; do
      echo "D = ${D}, FP = ${fp}, alpha = ${alpha}"
      python3.6 bench/split_table_batched_embeddings_benchmark.py device \
        $common_opts \
        --embedding-dim $D \
        --alpha ${alpha} \
        --weights-precision $fp
    done
  done
done 2>&1 | tee log_fbgemm_gpu_m1.log

# Run on GPU and get rocprof-level performance
for D in 64 128 192 256 512; do
  for fp in "fp32" "fp16"; do
    for alpha in 1 1.15; do
      rm -rf rocprof
      rm -rf rocprof_tmp
      echo "D = ${D}, FP = ${fp}, alpha = ${alpha}"
      outf="rocprof_fbgemm_gpu_D_${D}_${fp}_alpha_${alpha}.csv"
      rocprof --timestamp on -o $outf -d rocprof -t rocprof_tmp \
      python3.6 bench/split_table_batched_embeddings_benchmark.py device \
        $common_opts \
        --embedding-dim $D \
        --alpha ${alpha} \
        --weights-precision $fp
    done
  done
done

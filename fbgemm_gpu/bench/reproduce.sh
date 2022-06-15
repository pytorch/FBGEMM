#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#You need to change the TARGET_BIN
TARGET_BIN="buck run @mode/opt -c python.package_style=inplace -j 40 //deeplearning/fbgemm/fbgemm_gpu:split_table_batched_embeddings_benchmark"

NUMA_CMD="numactl --cpunodebind=0 --membind=0"
CUDA_CMD="CUDA_VISIBLE_DEVICES=1"

COMMON_PART="${CUDA_CMD} ${NUMA_CMD} ${TARGET_BIN} -- nbit-uvm --alpha 1.0 --reuse 0.1 --weights-precision int4 --uvm-num-embeddings 5000000 --num-embeddings 5000000 --batch-size 2048 --bag-size 256 --uvm-bag-size 256 --num-tables 10 --uvm-tables 10 --embedding-dim 248 "

for f in 0.305 0.405 0.505
do
    CMD1="${COMMON_PART} --memory-fraction ${f}"
    CMD2="${COMMON_PART} --memory-fraction ${f} --set-preferred-location False"
    echo ${CMD1}
    eval ${CMD1}
    echo ${CMD2}
    eval ${CMD2}
done

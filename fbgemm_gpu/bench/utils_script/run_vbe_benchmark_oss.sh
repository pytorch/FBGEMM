#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# OSS version of run_vbe_benchmark.sh
# Runs the VBE benchmark Python script directly instead of building with buck2.
#
# Requires one of these environment variables:
#   FBGEMM_REPO_ROOT   - root of the fbgemm repository checkout
#   FBGEMM_BENCH_SCRIPT - direct path to split_table_batched_embeddings_benchmark.py
#
# Usage:
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_vbe_benchmark_oss.sh [options]
#
# Examples:
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_vbe_benchmark_oss.sh -e
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_vbe_benchmark_oss.sh -t 3 --batch-size-list 2048,4096,2048 -e
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_vbe_benchmark_oss.sh -n --ncu-filter "forward"

set -euo pipefail

if [[ -n "${FBGEMM_BENCH_SCRIPT:-}" ]]; then
    BENCH_SCRIPT="${FBGEMM_BENCH_SCRIPT}"
elif [[ -n "${FBGEMM_REPO_ROOT:-}" ]]; then
    BENCH_SCRIPT="${FBGEMM_REPO_ROOT}/fbgemm_gpu/bench/tbe/split_table_batched_embeddings_benchmark.py"
else
    echo "Error: Set FBGEMM_BENCH_SCRIPT or FBGEMM_REPO_ROOT to locate the benchmark script"
    echo "  FBGEMM_BENCH_SCRIPT: direct path to split_table_batched_embeddings_benchmark.py"
    echo "  FBGEMM_REPO_ROOT:   root of the fbgemm repository checkout"
    exit 1
fi

if [[ ! -f "$BENCH_SCRIPT" ]]; then
    echo "Error: Benchmark script not found at ${BENCH_SCRIPT}"
    exit 1
fi

with_ncu=False
ncu_bin=/usr/local/cuda/bin/ncu
log_suffix=""
export_trace=""
num_tables=3
weights_precision="fp32"
output_dtype="fp32"
iters=100
ncu_filter=""
batch_size_list="2048,4096,1024"
embedding_dim_list="128,128,64"
bag_size_list="20,20,20"
num_embeddings_list="10000000,5000000,1000000"
alpha_list=""
merge_output=""

_print_help() {
  echo "Usage: FBGEMM_REPO_ROOT=<path> $0 [options]"
  echo ""
  echo "OSS VBE (Variable Batch-size Embedding) Benchmark"
  echo ""
  echo "Environment variables (one required):"
  echo "  FBGEMM_REPO_ROOT          root of the fbgemm repository checkout"
  echo "  FBGEMM_BENCH_SCRIPT       direct path to split_table_batched_embeddings_benchmark.py"
  echo ""
  echo "VBE parameters (comma-separated, one value per table):"
  echo "  -t|--num-tables N         Number of tables (default: 3)"
  echo '  --batch-size-list LIST    Batch sizes per table (default: "2048,4096,1024")'
  echo '  --embedding-dim-list LIST Embedding dims per table (default: "128,128,64")'
  echo '  --bag-size-list LIST      Bag sizes per table (default: "20,20,20")'
  echo '  --num-embeddings-list LIST Num embeddings per table (default: "10000000,5000000,1000000")'
  echo '  --alpha-list LIST         ZipF alpha per table (default: "None"=uniform)'
  echo ""
  echo "Precision:"
  echo "  -w|--weights-precision P  Weight precision (default: fp32)"
  echo "  -o|--output-dtype P       Output dtype (default: fp32)"
  echo ""
  echo "Benchmarking:"
  echo "  -i|--iter N               Number of iterations (default: 100)"
  echo "  -e|--export-trace         Export kineto trace"
  echo "  --merge-output            Enable merged output mode"
  echo "  -n|--ncu                  Run with NCU profiler (sets iters=2)"
  echo '  --ncu-filter FILTER       NCU kernel filter regex'
  echo "  --log-suffix SUFFIX       Append suffix to log file name"
  exit 0
}

while true; do
  case "${1-}" in
    -h | --help ) _print_help ;;
    -n | --ncu ) with_ncu=True; shift ;;
    -c | --arch ) shift 2 ;;
    -cu | --cuda ) shift 2 ;;
    --buck-config ) shift 2 ;;
    -t | --num-tables ) num_tables="${2}"; shift 2 ;;
    -w | --weights-precision ) weights_precision="${2}"; shift 2 ;;
    -o | --output-dtype ) output_dtype="${2}"; shift 2 ;;
    -i | --iter ) iters="${2}"; shift 2 ;;
    -e | --export-trace ) export_trace="true"; shift ;;
    --batch-size-list ) batch_size_list="${2}"; shift 2 ;;
    --embedding-dim-list ) embedding_dim_list="${2}"; shift 2 ;;
    --bag-size-list ) bag_size_list="${2}"; shift 2 ;;
    --num-embeddings-list ) num_embeddings_list="${2}"; shift 2 ;;
    --alpha-list ) alpha_list="${2}"; shift 2 ;;
    --merge-output ) merge_output="--merge-output"; shift ;;
    --log-suffix ) log_suffix="_${2}"; shift 2 ;;
    --ncu-filter ) ncu_filter="${2}"; shift 2 ;;
    * ) break ;;
  esac
done

if [[ $with_ncu == "True" ]]; then
  iters=2
  ncu="_ncu"
else
  ncu=""
fi

echo "Benchmark script: $BENCH_SCRIPT"

warmup_opts=""
if [[ $with_ncu != "True" ]]; then
  warmup_opts="--bench-warmup-iterations 10"
fi

run_vbe() {
  local a="$1"
  local log_name="${num_tables}_${weights_precision}_${output_dtype}_${a}_B_${batch_size_list}_L_${bag_size_list}${ncu}"
  log_name="${log_name//,/-}"

  local trace_opts=""
  if [[ "$export_trace" == "true" ]]; then
    trace_opts="--bench-export-trace --bench-trace-url={emb_op_type}_${log_name}.json"
  fi

  echo "---------------------------------"
  echo "VBE benchmark:"
  echo "  num_tables=${num_tables}, batch_size_list=${batch_size_list}"
  echo "  embedding_dim_list=${embedding_dim_list}, bag_size_list=${bag_size_list}"
  echo "  num_embeddings_list=${num_embeddings_list}, alpha_list=${a}"
  echo "  weights_precision=${weights_precision}, output_dtype=${output_dtype}"
  echo "  iters=${iters}"
  echo "---------------------------------"

  if [[ $with_ncu == "True" ]]; then
    ncu_cmd="$ncu_bin \
      --set full \
      -o ncu_vbe_${log_name}${log_suffix} \
      -k regex:\".$ncu_filter.\" \
      --target-processes all -f"
    echo "$ncu_cmd"
  else
    ncu_cmd=""
  fi

  # shellcheck disable=SC2086
  CUDA_INJECTION64_PATH=none \
    timeout 10m \
    $ncu_cmd \
    python3 "$BENCH_SCRIPT" \
    vbe \
    --num-tables ${num_tables} \
    --batch-size-list ${batch_size_list} \
    --embedding-dim-list ${embedding_dim_list} \
    --bag-size-list ${bag_size_list} \
    --num-embeddings-list ${num_embeddings_list} \
    --alpha-list ${a} \
    --bench-iterations ${iters} \
    --emb-weights-dtype=${weights_precision} \
    --emb-output-dtype=${output_dtype} \
    ${trace_opts} \
    ${warmup_opts} \
    ${merge_output} \
    2>&1 | tee "log_vbe_${log_name}.log"
}

if [[ -z "$alpha_list" ]]; then
  run_vbe "1"
  run_vbe "1.15"
else
  run_vbe "${alpha_list}"
fi

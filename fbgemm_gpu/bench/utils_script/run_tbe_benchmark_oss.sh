#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# OSS version of run_tbe_benchmark.sh
# Runs the TBE benchmark Python script directly instead of building with buck2.
#
# Requires one of these environment variables:
#   FBGEMM_REPO_ROOT   - root of the fbgemm repository checkout
#   FBGEMM_BENCH_SCRIPT - direct path to split_table_batched_embeddings_benchmark.py
#
# Usage:
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_tbe_benchmark_oss.sh [options]
#
# Examples:
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_tbe_benchmark_oss.sh -b 2048 -t 10 -d "128 256 512"
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_tbe_benchmark_oss.sh -b 4096 -t 1 -d "8 128 256" -e
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_tbe_benchmark_oss.sh -n --ncu-filter "forward"

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
bench=device
log_suffix=""
opts=""
export_trace=""
batch_size=131072
num_tables=1
weights_precision="fp32"
output_dtype="fp32"
a=1.15
L=55
iters=100
ncu_filter=""
dims="4 8 12 20 32 64 96 160 192 200 240 256 320 384 512 640 800 1024"
v2=""

_print_help() {
  echo "Usage: FBGEMM_REPO_ROOT=<path> $0 [options]"
  echo ""
  echo "OSS version of run_tbe_benchmark.sh (runs Python script directly)"
  echo ""
  echo "Environment variables (one required):"
  echo "  FBGEMM_REPO_ROOT          root of the fbgemm repository checkout"
  echo "  FBGEMM_BENCH_SCRIPT       direct path to split_table_batched_embeddings_benchmark.py"
  echo ""
  echo "Options:"
  echo "  -h|--help                 print this help message"
  echo "  -n|--ncu                  run with NCU profiler"
  echo "  -df|--defuse-bwd          defuse backward"
  echo "  -a|--alpha                set alpha value"
  echo "  -w|--weights-precision    set weight precision (e.g., fp32, fp16)"
  echo "  -o|--output-dtype         set output dtype (e.g., fp32, fp16)"
  echo "  -t|--num-tables           set number of tables"
  echo "  -b|--batch-size           set batch size"
  echo '  -d|--dims                 set embedding dims (e.g., -d "4 8 12 20")'
  echo "  -l|--bag-size             set bag size (pooling factor)"
  echo "  -e|--export-trace         export trace"
  echo "  -v2|--v2                  enable TBE forward v2 kernel"
  echo "  -i|--iter                 set number of iterations (NCU always uses 2)"
  echo "  -bench|--bench            select benchmark (e.g., device, device_with_spec)"
  echo "  --log-suffix SUFFIX       append suffix to log file name"
  echo '  --ncu-filter FILTER       NCU kernel filter regex'
  exit 0
}

while true; do
  case "${1-}" in
    -h | --help ) _print_help ;;
    -n | --ncu ) with_ncu=True; shift ;;
    -df | --defuse-bwd ) opts="$opts --defuse-bwd"; shift ;;
    -c | --arch ) shift 2 ;;
    -cu | --cuda ) shift 2 ;;
    --buck-config ) shift 2 ;;
    -a | --alpha ) a="${2}"; shift 2 ;;
    -w | --weights-precision ) weights_precision="${2}"; shift 2 ;;
    -o | --output-dtype ) output_dtype="${2}"; shift 2 ;;
    -t | --num-tables ) num_tables="${2}"; shift 2 ;;
    -b | --batch-size ) batch_size="${2}"; shift 2 ;;
    -d | --dims ) dims="${2}"; shift 2 ;;
    -l | --bag-size ) L="${2}"; shift 2 ;;
    -e | --export-trace ) export_trace="--export-trace"; shift ;;
    -v2 | --v2 ) v2="v2"; shift ;;
    -i | --iter ) iters="${2}"; shift 2 ;;
    -bench | --bench ) bench="${2}"; shift 2 ;;
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
  opts="$opts \
    --flush-gpu-cache-size-mb 40 \
    --warmup-runs 10"
fi

if [[ "${v2:-}" != "" ]]; then
  echo "Using TBE v2"
  export FBGEMM_NO_JK=1
  export FBGEMM_TBE_V2=1
fi

echo "Benchmark script: $BENCH_SCRIPT"
echo "Embedding dims to bench: ${dims}"
echo "---------------------------------"

log_name="${num_tables}_${weights_precision}_${output_dtype}_${a}_B_${batch_size}_L_${L}${ncu}"
# shellcheck disable=SC2086  # Intentional: dims is space-separated and needs word splitting
for D in $dims; do
  if [[ $with_ncu == "True" ]]; then
    ncu_cmd="$ncu_bin \
      --set full \
      -o ncu_tbe_bench_${bench}_L_${L}_D_${D}_B_${batch_size}${log_suffix} \
      -k regex:\".$ncu_filter.\" \
      --target-processes all -f"
    echo "$ncu_cmd"
  else
    ncu_cmd=""
  fi

  echo "Running ${bench} ${v2:-} ${export_trace} ${ncu} : D = ${D}, batch_size = ${batch_size}, iters = ${iters}, alpha = ${a}, num_tables = ${num_tables}, weights_precision = ${weights_precision}, output_dtype = ${output_dtype}"

  # shellcheck disable=SC2086
  CUDA_INJECTION64_PATH=none \
    timeout 10m \
    $ncu_cmd \
    python3 "$BENCH_SCRIPT" \
    $bench \
    --batch-size ${batch_size} \
    --embedding-dim $D \
    --iters $iters \
    --alpha ${a} \
    --bag-size ${L} \
    --weights-precision=${weights_precision} \
    --output-dtype=${output_dtype} \
    --num-embeddings 10000000 \
    --num-tables ${num_tables} \
    ${export_trace} \
    --trace-url="{tbe_type}_tbe_{phase}_${log_name}_D_${D}.json" \
    $opts
done 2>&1 | tee "log_tbe_${log_name}.log"

if [[ $with_ncu == "True" ]]; then
  echo "NCU profiling complete"
fi

unset FBGEMM_NO_JK
unset FBGEMM_TBE_V2

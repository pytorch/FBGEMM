#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# OSS version of run_tbe_benchmark_sweep.sh
# Uses python3 directly instead of buck2-built binary.
#
# Requires one of these environment variables:
#   FBGEMM_REPO_ROOT   - root of the fbgemm repository checkout
#   FBGEMM_BENCH_SCRIPT - direct path to split_table_batched_embeddings_benchmark.py
#
# Usage:
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_tbe_benchmark_sweep_oss.sh [--run-5|--run-10] [--export]
#
# Examples:
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_tbe_benchmark_sweep_oss.sh --run-10 --export
#   FBGEMM_REPO_ROOT=~/fbgemm bash run_tbe_benchmark_sweep_oss.sh --run-5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/run_tbe_benchmark_oss.sh"
VBE_BENCHMARK_SCRIPT="${SCRIPT_DIR}/run_vbe_benchmark_oss.sh"

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

batch_sizes=(2048 4096 131072)
num_tables=(1 10)
dims="8 128 256 512 1024"

run_mode=""
export_flag=""
export_trace=""

_print_help() {
  echo "Usage: $0 [--run-5|--run-10] [options]"
  echo ""
  echo "OSS version of run_tbe_benchmark_sweep.sh"
  echo ""
  echo "Modes:"
  echo "  --run-5      Run 5 representative benchmarks"
  echo "  --run-10     Run 10 representative benchmarks"
  echo ""
  echo "Default sweep and VBE benchmarks always run."
  echo ""
  echo "Options:"
  echo "  -h|--help     Print this help message"
  echo "  --export      Export traces (--run-5/--run-10 only)"
  exit 0
}

while true; do
  case "${1-}" in
    -h | --help ) _print_help ;;
    --run-5 ) run_mode="5"; shift ;;
    --run-10 ) run_mode="10"; shift ;;
    --export ) export_flag="true"; export_trace="true"; shift ;;
    -c | --arch ) shift 2 ;;
    -cu | --cuda ) shift 2 ;;
    --buck-config ) shift 2 ;;
    * ) break ;;
  esac
done

extra_args=("$@")

# ============================================================
# Representative benchmarks (from production TBE configs)
# All use: batch_size=2048, FP16 weights, FP32 output,
#          sum pooling, bag_size=20, iters=100, warmup_runs=10
# ============================================================

_export_opts() {
  if [[ "$export_trace" == "true" ]]; then
    echo "--export-trace --trace-url=bench_${1}.json"
  fi
}

# Benchmark 103: T=2, dim=8
run_benchmark_103() {
  # shellcheck disable=SC2046
  python3 "$BENCH_SCRIPT" device-with-spec \
    --batch-size 2048 \
    --num-embeddings-list 20072405,633 \
    --embedding-dim-list 8,8 \
    --bag-size-list 20,20 \
    --weights-precision fp16 \
    --output-dtype fp32 \
    --pooling sum \
    --iters 100 \
    --warmup-runs 10 \
    $(_export_opts 103)
}

# Benchmark 50: T=2, dim=128
run_benchmark_50() {
  # shellcheck disable=SC2046
  python3 "$BENCH_SCRIPT" device-with-spec \
    --batch-size 2048 \
    --num-embeddings-list 2,11899818 \
    --embedding-dim-list 128,128 \
    --bag-size-list 20,20 \
    --weights-precision fp16 \
    --output-dtype fp32 \
    --pooling sum \
    --iters 100 \
    --warmup-runs 10 \
    $(_export_opts 50)
}

# Benchmark 61: T=6, dim=128
run_benchmark_61() {
  # shellcheck disable=SC2046
  python3 "$BENCH_SCRIPT" device-with-spec \
    --batch-size 2048 \
    --num-embeddings-list 3519,7529582,155,5797,2737037,66450 \
    --embedding-dim-list 128,128,128,128,128,128 \
    --bag-size-list 20,20,20,20,20,20 \
    --weights-precision fp16 \
    --output-dtype fp32 \
    --pooling sum \
    --iters 100 \
    --warmup-runs 10 \
    $(_export_opts 61)
}

# Benchmark 0: T=10, dim=128
run_benchmark_0() {
  # shellcheck disable=SC2046
  python3 "$BENCH_SCRIPT" device-with-spec \
    --batch-size 2048 \
    --num-embeddings-list 2,284285,4,30000000,4348658,6784913,642110,236,278,3618081 \
    --embedding-dim-list 128,128,128,128,128,128,128,128,128,128 \
    --bag-size-list 20,20,20,20,20,20,20,20,20,20 \
    --weights-precision fp16 \
    --output-dtype fp32 \
    --pooling sum \
    --iters 100 \
    --warmup-runs 10 \
    $(_export_opts 0)
}

# Benchmark 5: T=10, dims=20/128
run_benchmark_5() {
  # shellcheck disable=SC2046
  python3 "$BENCH_SCRIPT" device-with-spec \
    --batch-size 2048 \
    --num-embeddings-list 426463,8171523,5587,2,30000000,1943,107153,424,6442728,4062956 \
    --embedding-dim-list 128,128,128,20,128,128,128,128,128,128 \
    --bag-size-list 20,20,20,20,20,20,20,20,20,20 \
    --weights-precision fp16 \
    --output-dtype fp32 \
    --pooling sum \
    --iters 100 \
    --warmup-runs 10 \
    $(_export_opts 5)
}

# Benchmark 13: T=12, dims=24/128
run_benchmark_13() {
  # shellcheck disable=SC2046
  python3 "$BENCH_SCRIPT" device-with-spec \
    --batch-size 2048 \
    --num-embeddings-list 2,14451,714,5,30000000,6784941,4364852,257,2,6171979,758,257 \
    --embedding-dim-list 128,128,128,24,128,128,128,128,128,128,24,24 \
    --bag-size-list 20,20,20,20,20,20,20,20,20,20,20,20 \
    --weights-precision fp16 \
    --output-dtype fp32 \
    --pooling sum \
    --iters 100 \
    --warmup-runs 10 \
    $(_export_opts 13)
}

# Benchmark 2: T=14, dims=12/24/128
run_benchmark_2() {
  # shellcheck disable=SC2046
  python3 "$BENCH_SCRIPT" device-with-spec \
    --batch-size 2048 \
    --num-embeddings-list 227,2,1605631,31,2,642,2,30000000,6783150,478,237,4539974,5755862,690 \
    --embedding-dim-list 128,128,128,128,128,24,24,128,128,128,128,128,128,12 \
    --bag-size-list 20,20,20,20,20,20,20,20,20,20,20,20,20,20 \
    --weights-precision fp16 \
    --output-dtype fp32 \
    --pooling sum \
    --iters 100 \
    --warmup-runs 10 \
    $(_export_opts 2)
}

# Benchmark 1: T=18, dims=20/128
run_benchmark_1() {
  # shellcheck disable=SC2046
  python3 "$BENCH_SCRIPT" device-with-spec \
    --batch-size 2048 \
    --num-embeddings-list 661,48,5801,653,30,224,612,710,30000000,6784921,375,11,2658,5376992,296,236,760,763 \
    --embedding-dim-list 128,128,128,128,128,20,20,20,128,128,128,128,128,128,20,20,20,20 \
    --bag-size-list 20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20 \
    --weights-precision fp16 \
    --output-dtype fp32 \
    --pooling sum \
    --iters 100 \
    --warmup-runs 10 \
    $(_export_opts 1)
}

# Benchmark 3: T=20, dims=20/24/128
run_benchmark_3() {
  # shellcheck disable=SC2046
  python3 "$BENCH_SCRIPT" device-with-spec \
    --batch-size 2048 \
    --num-embeddings-list 4,2,301571,26077834,2,2,4,6,752,5287864,2307981,518,613,7677271,3489261,313,3,484,773,504 \
    --embedding-dim-list 128,128,128,128,128,24,20,20,20,128,128,128,128,128,128,24,24,20,20,20 \
    --bag-size-list 20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20 \
    --weights-precision fp16 \
    --output-dtype fp32 \
    --pooling sum \
    --iters 100 \
    --warmup-runs 10 \
    $(_export_opts 3)
}

# Benchmark 27: T=22, dims=20/24/128
run_benchmark_27() {
  # shellcheck disable=SC2046
  python3 "$BENCH_SCRIPT" device-with-spec \
    --batch-size 2048 \
    --num-embeddings-list 2,2,61787,144013,4,2,685,603,692,30000000,1776198,6784906,11,729,16,6561149,3086,420,507,3,22,613 \
    --embedding-dim-list 128,128,128,128,128,24,20,20,20,128,128,128,128,128,128,128,128,24,24,20,20,20 \
    --bag-size-list 20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20 \
    --weights-precision fp16 \
    --output-dtype fp32 \
    --pooling sum \
    --iters 100 \
    --warmup-runs 10 \
    $(_export_opts 27)
}

# ============================================================
# Run representative benchmarks
# ============================================================

if [[ "$run_mode" == "5" ]]; then
  echo "Running 5 representative benchmarks"
  echo "=========================================="
  run_benchmark_103
  run_benchmark_0
  run_benchmark_2
  run_benchmark_1
  run_benchmark_27

elif [[ "$run_mode" == "10" ]]; then
  echo "Running 10 representative benchmarks"
  echo "=========================================="
  run_benchmark_103
  run_benchmark_50
  run_benchmark_61
  run_benchmark_0
  run_benchmark_5
  run_benchmark_13
  run_benchmark_2
  run_benchmark_1
  run_benchmark_3
  run_benchmark_27

fi

# ============================================================
# Default sweep over batch size, num tables, dims
# ============================================================

echo ""
echo "Running default sweep"
echo "Embedding dims: ${dims}"

for B in "${batch_sizes[@]}"; do
  for T in "${num_tables[@]}"; do
    echo "=========================================="
    echo "Running: B=${B}, T=${T}, D=[${dims}]"
    echo "=========================================="
    cmd_args=(-b "${B}" -t "${T}" -d "${dims}")
    if [[ "$export_flag" == "true" ]]; then
      cmd_args+=(-e)
    fi
    cmd_args+=("${extra_args[@]}")
    bash "${BENCHMARK_SCRIPT}" "${cmd_args[@]}"
  done
done

# ============================================================
# VBE benchmarks
# ============================================================

echo ""
echo "Running VBE benchmarks"
echo "=========================================="
vbe_args=("${extra_args[@]}")
if [[ "$export_flag" == "true" ]]; then
  vbe_args+=(-e)
fi
bash "${VBE_BENCHMARK_SCRIPT}" "${vbe_args[@]}"

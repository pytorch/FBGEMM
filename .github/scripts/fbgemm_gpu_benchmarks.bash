#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# FBGEMM_GPU Test Helper Functions
################################################################################

run_tbe_microbench () {
  local env_name="$1"

  __single_run() {
    local cache_type="$1"
    local embedding_location="$2"

    echo "################################################################################"
    echo "# Running Benchmark: (${cache_type}, ${embedding_location})"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""

    # shellcheck disable=SC2155
    local env_prefix=$(env_name_or_prefix "${env_name}")

    if [ "$embedding_location" == "hbm" ]; then
      local managed="device"
    elif [ "$embedding_location" == "uvm" ]; then
      local managed="managed"
    fi

    # Old TBE benchmark script
    # shellcheck disable=SC2086
    print_exec conda run --no-capture-output ${env_prefix} python tbe/split_table_batched_embeddings_benchmark.py device \
      --batch-size 13107 \
      --embedding-dim 256 \
      --iters 400 \
      --warmup-runs 50 \
      --alpha 1.15 \
      --bag-size 55 \
      --weights-precision fp16 \
      --cache-precision "${cache_type}" \
      --output-dtype bf16 \
      --managed="${managed}" \
      --num-embeddings 10000000 \
      --num-tables 1 \
      --row-wise

    # New TBE benchmark script
    #
    # Invoke `python tbe/tbe_training_benchmark.py device --help` for
    # documentation on all available flags
    # shellcheck disable=SC2086
    print_exec conda run --no-capture-output ${env_prefix} python tbe/tbe_training_benchmark.py device \
      --bench-iterations 400 \
      --bench-warmup-iterations 50 \
      --bench-num-requests 10 \
      --tbe-batch-size 13107 \
      --tbe-embedding-dim 256 \
      --tbe-pooling-size 55 \
      --tbe-num-embeddings 10000000 \
      --tbe-num-tables 1 \
      --emb-weights-dtype fp16 \
      --emb-cache-dtype "${cache_type}" \
      --emb-output-dtype bf16 \
      --emb-location "${managed}" \
      --row-wise
  }

  pushd fbgemm_gpu/bench || return 1

  local cache_types=(
    # fp16
    fp32
  )

  local embedding_locations=(
    # uvm
    hbm
  )

  for cache_type in "${cache_types[@]}"; do
    for embedding_location in "${embedding_locations[@]}"; do
      __single_run "${cache_type}" "${embedding_location}" || return 1
      echo ""
      echo ""
    done
  done

  popd || return 1
}

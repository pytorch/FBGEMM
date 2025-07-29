#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
. "$( dirname -- "$BASH_SOURCE"; )/fbgemm_gpu_test.bash"

################################################################################
# FBGEMM_GPU Test Helper Functions
################################################################################

setup_fbgemm_gpu_bench () {
  env_name="$1"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # shellcheck disable=SC2086
  fbgemm_build_variant=$(conda run ${env_prefix} python -c "import fbgemm_gpu; print(fbgemm_gpu.__variant__)")
  echo "[BENCH] Determined FBGEMM_GPU variant from installation: ${fbgemm_build_variant}"

  if [ "$fbgemm_build_variant" == "rocm" ]; then
    echo "[BENCH] Configuring for ROCm-based benchmarking ..."
    __configure_fbgemm_gpu_test_rocm
  fi

  if [[ $MACHINE_NAME == 'aarch64' ]]; then
    # NOTE: Setting KMP_DUPLICATE_LIB_OK silences the error about multiple
    # OpenMP being linked when FBGEMM_GPU is compiled under Clang on aarch64
    # machines:
    #   https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
    echo "[TEST] Platform is aarch64; will set KMP_DUPLICATE_LIB_OK ..."
    # shellcheck disable=SC2086
    print_exec conda env config vars set ${env_prefix} KMP_DUPLICATE_LIB_OK=1
  fi

  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} TORCH_SHOW_CPP_STACKTRACES=1
}

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

    # Invoke `python tbe/tbe_training_benchmark.py device --help` for
    # documentation on all available flags
    # shellcheck disable=SC2086
    print_exec conda run --no-capture-output ${env_prefix} python tbe/tbe_training_benchmark.py device \
        --tbe-batch-size 13107 \
        --tbe-embedding-dim 256 \
        --tbe-pooling-size 55 \
        --tbe-num-embeddings 10000000 \
        --tbe-num-tables 1 \
        --tbe-indices-zipf 1.0 1.15 \
        --emb-weights-dtype fp16 \
        --emb-cache-dtype "${cache_type}" \
        --emb-output-dtype fp16 \
        --emb-location "${embedding_location}" \
        --emb-pooling-mode sum \
        --row-wise \
        --bench-iterations 100 \
        --bench-warmup-iterations 50 \
        --bench-export-trace --bench-trace-url "test_${cache_type}_${embedding_location}.json"
  }

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  pushd fbgemm_gpu/bench || return 1

  local cache_types=(
    fp16
    fp32
  )

  if  [ "${BUILD_VARIANT}" == "cpu" ]; then
    local embedding_locations=(
      host
    )
  else
    # shellcheck disable=SC2086,SC2155
    local torch_cuda_available=$(conda run ${env_prefix} python -c "import torch; print(int(torch.cuda.is_available()))")

    if [ "${torch_cuda_available}" == "1" ]; then
      local embedding_locations=(
        managed_caching
        managed
        device
      )
    else
      local embedding_locations=(
        host
      )
      echo "[BENCH] GPU devices not visible from torch; falling back to host-only embedding location ..."
    fi
  fi

  for cache_type in "${cache_types[@]}"; do
    for embedding_location in "${embedding_locations[@]}"; do
      __single_run "${cache_type}" "${embedding_location}" || return 1
      echo ""
      echo ""
    done
  done

  popd || return 1
}

#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# FBGEMM_GPU Test Helper Functions
################################################################################

run_python_test () {
  local env_name="$1"
  local python_test_file="$2"
  if [ "$python_test_file" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTHON_TEST_FILE"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env quantize_ops_test.py"
    return 1
  else
    echo "################################################################################"
    echo "# [$(date --utc +%FT%T.%3NZ)] Run Python Test Suite:"
    echo "#   ${python_test_file}"
    echo "################################################################################"
  fi

  if print_exec conda run -n "${env_name}" python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning "${python_test_file}"; then
    echo "[TEST] Python test suite PASSED: ${python_test_file}"
    echo ""
  else
    echo "[TEST] Python test suite FAILED: ${python_test_file}"
    echo ""
    return 1
  fi
}


################################################################################
# FBGEMM_GPU Test Functions
################################################################################

run_fbgemm_gpu_tests () {
  local env_name="$1"
  local fbgemm_variant="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [FBGEMM_VARIANT]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env        # Run all tests applicable to CUDA"
    echo "    ${FUNCNAME[0]} build_env cpu    # Run all tests applicable to CPU"
    echo "    ${FUNCNAME[0]} build_env rocm   # Run all tests applicable to ROCm"
    return 1
  else
    echo "################################################################################"
    echo "# Run FBGEMM-GPU Tests"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # Enable ROCM testing if specified
  if [ "$fbgemm_variant" == "rocm" ]; then
    echo "[TEST] Set environment variable FBGEMM_TEST_WITH_ROCM to enable ROCm tests ..."
    print_exec conda env config vars set -n "${env_name}" FBGEMM_TEST_WITH_ROCM=1
  fi

  # These are either non-tests or currently-broken tests in both FBGEMM_GPU and FBGEMM_GPU-CPU
  local files_to_skip=(
    test_utils.py
    split_table_batched_embeddings_test.py
    ssd_split_table_batched_embeddings_test.py
  )

  if [ "$fbgemm_variant" == "cpu" ]; then
    # These are tests that are currently broken in FBGEMM_GPU-CPU
    local ignored_tests=(
      uvm_test.py
    )
  elif [ "$fbgemm_variant" == "rocm" ]; then
    # https://github.com/pytorch/FBGEMM/issues/1559
    local ignored_tests=(
      batched_unary_embeddings_test.py
    )
  else
    local ignored_tests=()
  fi

  echo "[TEST] Installing pytest ..."
  print_exec conda install -n "${env_name}" -y pytest

  echo "[TEST] Checking imports ..."
  (test_python_import "${env_name}" fbgemm_gpu) || return 1
  (test_python_import "${env_name}" fbgemm_gpu.split_embedding_codegen_lookup_invokers) || return 1

  echo "[TEST] Enumerating test files ..."
  print_exec ls -lth ./*.py

  # NOTE: Tests running on single CPU core with a less powerful testing GPU in
  # GHA can take up to 5 hours.
  for test_file in *.py; do
    if echo "${files_to_skip[@]}" | grep "${test_file}"; then
      echo "[TEST] Skipping test file known to be broken: ${test_file}"
    elif echo "${ignored_tests[@]}" | grep "${test_file}"; then
      echo "[TEST] Skipping test file: ${test_file}"
    elif run_python_test "${env_name}" "${test_file}"; then
      echo ""
    else
      return 1
    fi
  done
}

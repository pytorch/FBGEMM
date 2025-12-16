#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pytorch.bash"

################################################################################
# FBGEMM_GPU Test Helper Functions
################################################################################

install_fbgemm_gpu_deps () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Install FBGEMM-GPU PIP dependencies"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[BUILD] Installing PIP dependencies ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} python -m pip install -r requirements.txt) || return 1

  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" einops) || return 1
  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" numpy) || return 1
}

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

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")
  # shellcheck disable=SC2155
  local start=$(date +%s)

  # shellcheck disable=SC2086
  if print_exec conda run --no-capture-output ${env_prefix} python -m pytest "${pytest_args[@]}" --cache-clear  "${python_test_file}"; then
    echo "[TEST] Python test suite PASSED: ${python_test_file}"
    local test_time=$(($(date +%s)-start))
    echo "[TEST] Python test time for ${python_test_file}: ${test_time} seconds"
    echo ""
    echo ""
    echo ""
    return 0
  fi

  echo "[TEST] Some tests FAILED.  Re-attempting only FAILED tests: ${python_test_file}"
  echo ""
  echo ""

  # NOTE: Running large test suites may result in OOM error that will cause the
  # process to be prematurely killed.  To work around this, when we re-run test
  # suites, we only run tests that have failed in the previous round.  This is
  # enabled by using the pytest cache and the --lf flag.

  # shellcheck disable=SC2086
  if exec_with_retries 2 conda run --no-capture-output ${env_prefix} python -m pytest "${pytest_args[@]}" --lf --last-failed-no-failures none "${python_test_file}"; then
    echo "[TEST] Python test suite PASSED with retries: ${python_test_file}"
    local test_time=$(($(date +%s)-start))
    echo "[TEST] Python test time with retries for ${python_test_file}: ${test_time} seconds"
    echo ""
    echo ""
    echo ""
  else
    echo "[TEST] Python test suite FAILED for some or all tests despite multiple retries: ${python_test_file}"
    echo ""
    echo ""
    echo ""
    return 1
  fi
}

__configure_fbgemm_gpu_test_cpu () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")
  echo "[TEST] Set environment variables for CPU-only testing ..."

  # Prevent automatically running CUDA-enabled tests on a GPU-capable machine
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} CUDA_VISIBLE_DEVICES=-1

  export ignored_tests=(
    # These tests have non-CPU operators referenced in @given
    ./uvm/copy_test.py
    ./uvm/uvm_test.py
  )
}

__configure_fbgemm_gpu_test_cuda () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")
  echo "[TEST] Set environment variables for CUDA testing ..."

  # Disabled by default; enable for debugging
  # shellcheck disable=SC2086
  # print_exec conda env config vars set ${env_prefix} CUDA_LAUNCH_BLOCKING=1

  # Remove CUDA device specificity when running CUDA tests
  # shellcheck disable=SC2086
  print_exec conda env config vars unset ${env_prefix} CUDA_VISIBLE_DEVICES

  export ignored_tests=(
    ./moe/layers_test.py  # Not a python unittest file
    ./attention/blackwell_fmha_test.py
  )
}

__configure_fbgemm_gpu_test_rocm () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")
  echo "[TEST] Set environment variables for ROCm testing ..."

  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} FBGEMM_TEST_WITH_ROCM=1
  # Disabled by default; enable for debugging
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} HIP_LAUNCH_BLOCKING=1
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} FBGEMM_TBE_ROCM_INFERENCE_PACKED_BAGS=1

  # AMD GPUs need to be explicitly made visible to PyTorch for use
  # shellcheck disable=SC2155,SC2126
  local num_gpus=$(rocm-smi --showproductname | grep GUID | wc -l)
  # shellcheck disable=SC2155
  local gpu_indices=$(seq 0 $((num_gpus - 1)) | paste -sd, -)
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} HIP_VISIBLE_DEVICES="${gpu_indices}"

  # Starting from MI250 AMD GPUs support per process XNACK mode change
  # shellcheck disable=SC2155
  local rocm_version=$(awk -F'[.-]' '{print $1 * 10000 + $2 * 100 + $3}' /opt/rocm/.info/version-dev)
  if [ "$rocm_version" -ge 50700 ]; then
    # shellcheck disable=SC2086
    print_exec conda env config vars set ${env_prefix} HSA_XNACK=1
  fi

  # https://github.com/pytorch/FBGEMM/issues/1559
  export ignored_tests=(
    ./batched_unary_embeddings_test.py
    ./sll/triton_sll_test.py
    ./gather_scatter/gather_scatter_test.py
    ./moe/layers_test.py  # Not a python unittest file
    ./attention/blackwell_fmha_test.py
  )
}

__set_feature_flags () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # NOTE: The full list of feature flags is defined (without the `FBGEMM_`
  # prefix) in:
  #   fbgemm_gpu/include/config/feature_gates.h
  local feature_flags=(
    FBGEMM_TBE_ENSEMBLE_ROWWISE_ADAGRAD
  )

  echo "[TEST] Setting feature flags ..."
  for flag in "${feature_flags[@]}"; do
    # shellcheck disable=SC2086
    print_exec conda env config vars set ${env_prefix} ${flag}=1
  done
}

__setup_fbgemm_gpu_test () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Configure the environment for ignored test suites for each FBGEMM_GPU
  # variant
  if [ "$fbgemm_build_variant" == "cpu" ]; then
    echo "[TEST] Configuring for CPU-based testing ..."
    __configure_fbgemm_gpu_test_cpu

  elif [ "$fbgemm_build_variant" == "rocm" ]; then
    echo "[TEST] Configuring for ROCm-based testing ..."
    __configure_fbgemm_gpu_test_rocm

  else
    echo "[TEST] FBGEMM_GPU variant is ${fbgemm_build_variant}; configuring for CUDA-based testing ..."
    __configure_fbgemm_gpu_test_cuda
  fi

  if [ "$fbgemm_build_target" == "hstu" ]; then
    ignored_tests+=(
      ./tma_error_test.py
    )
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

  echo "[TEST] Installing PyTest ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge --override-channels -y \
    pytest \
    expecttest) || return 1

  echo "[TEST] Checking imports ..."
  (test_python_import_package "${env_name}" fbgemm_gpu) || return 1

  # Set the feature flags to enable experimental features as needed
  __set_feature_flags

  # Configure the PyTest args
  pytest_args=(
    -v
    -rsx
    -s
    -W ignore::pytest.PytestCollectionWarning
  )

  # shellcheck disable=SC2145
  echo "[TEST] PyTest args:  ${pytest_args[@]}"
}

################################################################################
# FBGEMM_GPU Test Functions
################################################################################

__run_fbgemm_gpu_tests_in_directory () {
  echo "################################################################################"
  # shellcheck disable=SC2154
  echo "# Run FBGEMM-GPU Tests: ${pwd}"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[TEST] Enumerating ALL test files ..."
  # shellcheck disable=SC2155
  local all_test_files=$(find . -type f -name '*_test.py' -print | sort)
  for f in $all_test_files; do echo "$f"; done
  echo ""

  echo "[TEST] Enumerating IGNORED test files ..."
  shopt -s nullglob  # Avoid printing pattern if no matches
  for pattern in "${ignored_tests[@]}"; do
    for f in $pattern; do   # unquoted $pattern to trigger glob expansion
      if [[ -e "$f" ]]; then
        echo "$f"
      fi
    done
  done
  echo ""

  for test_file in $all_test_files; do
    # Check if test_file matches any ignored glob pattern
    skip=false
    for pattern in "${ignored_tests[@]}"; do
      # shellcheck disable=SC2053
      if [[ "$test_file" == $pattern ]]; then
        skip=true
        break
      fi
    done

    if [[ "$skip" == true ]]; then
      echo "[TEST] Skipping test file: ${test_file}"
      echo ""
      continue

    elif run_python_test "${env_name}" "${test_file}"; then
      echo ""

    else
      return 1
    fi
  done
}

__determine_test_directories () {
  target_directories=()

  if [ "$fbgemm_build_target" == "genai" ]; then
    target_directories+=(
      fbgemm_gpu/experimental/gen_ai/test
    )

    if [ "$fbgemm_build_variant" == "cuda" ]; then
      target_directories+=(
        fbgemm_gpu/experimental/example/test
        fbgemm_gpu/experimental/gemm/test
      )
    fi

  elif [ "$fbgemm_build_target" == "hstu" ]; then
    target_directories+=(
      fbgemm_gpu/experimental/hstu/test
    )

  else
    target_directories+=(
      fbgemm_gpu/test
    )
  fi

  echo "[TEST] Determined the test directories:"
  for test_dir in "${target_directories[@]}"; do
    echo "$test_dir"
  done
  echo ""
}

test_all_fbgemm_gpu_modules () {
  env_name="$1"
  local repo="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env        # Test all FBGEMM_GPU modules applicable to to the installed variant"
    return 1
  else
    echo "################################################################################"
    echo "# Test All FBGEMM-GPU Modules"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  if [ "$repo" == "" ]; then
    echo "[TEST]: repo argument not provided, defaulting to current directory"
    repo=$(pwd)
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Move to another directory, to avoid Python package import confusion, since
  # there exists a fbgemm_gpu/ subdirectory in the MSLK repo
  print_exec mkdir -p _tmp_dir_fbgemm_gpu   || return 1
  print_exec pushd _tmp_dir_fbgemm_gpu      || return 1

  # Determine the FBGEMM build target and variant
  # shellcheck disable=SC2086
  fbgemm_build_target=$(conda run ${env_prefix} python -c "import fbgemm_gpu; print(fbgemm_gpu.__target__)")
  # shellcheck disable=SC2086
  fbgemm_build_variant=$(conda run ${env_prefix} python -c "import fbgemm_gpu; print(fbgemm_gpu.__variant__)")

  echo "[TEST] Determined FBGEMM_GPU (target : variant) from installation: (${fbgemm_build_target} : ${fbgemm_build_variant})"
  echo "[TEST] Will be running tests specific to this target and variant ..."

  # Set the ignored tests and PyTest args
  __setup_fbgemm_gpu_test           || return 1

  # Verify that the GPUs are visible
  __verify_pytorch_gpu_integration  || return 1

  # Go to the repo root directory
  print_exec pushd "${repo}"        || return 1

  # Determine the test directories to include for testing
  __determine_test_directories      || return 1

  # Iterate through the test directories and run bulk tests
  for test_dir in "${target_directories[@]}"; do
    print_exec pushd "${test_dir}"                      || return 1
    __run_fbgemm_gpu_tests_in_directory "${env_name}"   || return 1
    print_exec popd                                     || return 1
  done

  echo "[TEST] Successfully executed all FBGEMM_GPU tests"
}

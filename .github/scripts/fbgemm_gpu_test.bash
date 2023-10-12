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

  # shellcheck disable=SC2086
  if exec_with_retries 2 conda run --no-capture-output ${env_prefix} python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning "${python_test_file}"; then
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
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Enable ROCM testing if specified
  if [ "$fbgemm_variant" == "rocm" ]; then
    echo "[TEST] Set environment variable FBGEMM_TEST_WITH_ROCM to enable ROCm tests ..."
    # shellcheck disable=SC2086
    print_exec conda env config vars set ${env_prefix} FBGEMM_TEST_WITH_ROCM=1
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
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -y pytest expecttest) || return 1

  echo "[TEST] Checking imports ..."
  (test_python_import_package "${env_name}" fbgemm_gpu) || return 1
  (test_python_import_package "${env_name}" fbgemm_gpu.split_embedding_codegen_lookup_invokers) || return 1

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


################################################################################
# FBGEMM_GPU Test Bulk-Combination Functions
################################################################################

test_setup_conda_environment () {
  local env_name="$1"
  local python_version="$2"
  local pytorch_installer="$3"
  local pytorch_version="$4"
  local pytorch_variant_type="$5"
  local pytorch_variant_version="$6"
  if [ "$pytorch_variant_type" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTHON_VERSION PYTORCH_INSTALLER PYTORCH_VERSION PYTORCH_VARIANT_TYPE [PYTORCH_VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 3.8 pip test cuda 11.8.0       # Setup environment with pytorch-test for Python 3.8 + CUDA 11.8.0"
    return 1
  else
    echo "################################################################################"
    echo "# Setup FBGEMM-GPU Build Container (All Steps)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  if [ "$env_name" == "" ]; then
    local env_name="test_py${python_version}_${pytorch_installer}_pytorch_${pytorch_version}_${pytorch_variant_type}"
    if [ "$pytorch_variant_version" != "" ]; then
      local env_name="${env_name}_${pytorch_variant_version}"
    fi
  fi

  echo "Creating the Build Environment: ${env_name} ..."
  create_conda_environment    "${env_name}" "${python_version}" || return 1

  # Set up the build tools and/or GPU runtimes
  if [ "$pytorch_variant_type" == "cuda" ]; then
    install_cxx_compiler      "${env_name}"                                                                         || return 1
    install_build_tools       "${env_name}"                                                                         || return 1
    install_cuda              "${env_name}" "${pytorch_variant_version}"                                            || return 1
    install_cudnn             "${env_name}" "${HOME}/cudnn-${pytorch_variant_version}" "${pytorch_variant_version}" || return 1

  elif [ "$pytorch_variant_type" == "rocm" ]; then
    install_rocm_ubuntu       "${env_name}" "${pytorch_variant_version}"  || return 1
    install_build_tools       "${env_name}"                               || return 1
    return 1

  else
    install_cxx_compiler      "${env_name}" || return 1
    install_build_tools       "${env_name}" || return 1
  fi

  # Install PyTorch
  if [ "$pytorch_installer" == "conda" ]; then
    install_pytorch_conda     "${env_name}" "${pytorch_version}" "${pytorch_variant_type}" "${pytorch_variant_version}" || return 1
  else
    install_pytorch_pip       "${env_name}" "${pytorch_version}" "${pytorch_variant_type}" "${pytorch_variant_version}" || return 1
  fi
}

test_fbgemm_gpu_build_and_install () {
  local env_name="$1"
  local pytorch_variant_type="$2"
  if [ "$pytorch_variant_type" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTORCH_VARIANT_TYPE"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env cuda   # Build and install FBGEMM_GPU for CUDA (All Steps)"
    return 1
  else
    echo "################################################################################"
    echo "# Setup FBGEMM-GPU Build Container (All Steps)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # Assume we are starting from the repository root directory
  cd fbgemm_gpu                                                               || return 1
  prepare_fbgemm_gpu_build    "${env_name}"                                   || return 1
  build_fbgemm_gpu_package    "${env_name}" release "${pytorch_variant_type}" || return 1

  # shellcheck disable=SC2164
  cd -
  install_fbgemm_gpu_wheel    "${env_name}" fbgemm_gpu/dist/*.whl             || return 1

  cd fbgemm_gpu/test                        || return 1
  run_fbgemm_gpu_tests        "${env_name}" || return 1
  # shellcheck disable=SC2164
  cd -
}

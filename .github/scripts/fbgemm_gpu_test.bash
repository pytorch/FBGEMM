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
  if print_exec conda run --no-capture-output ${env_prefix} python -m pytest "${pytest_args[@]}" --cache-clear  "${python_test_file}"; then
    echo "[TEST] Python test suite PASSED: ${python_test_file}"
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
  ignored_tests=(
    ./ssd_split_table_batched_embeddings_test.py
    # These tests have non-CPU operators referenced in @given
    ./uvm/copy_test.py
    ./uvm/uvm_test.py
  )
}

__configure_fbgemm_gpu_test_cuda () {
  ignored_tests=(
    ./ssd_split_table_batched_embeddings_test.py
  )
}

__configure_fbgemm_gpu_test_rocm () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[TEST] Set environment variables for ROCm testing ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} FBGEMM_TEST_WITH_ROCM=1
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} HIP_LAUNCH_BLOCKING=1

  # Starting from MI250 AMD GPUs support per process XNACK mode change
  # shellcheck disable=SC2155
  local rocm_version=$(awk -F'[.-]' '{print $1 * 10000 + $2 * 100 + $3}' /opt/rocm/.info/version-dev)
  if [ "$rocm_version" -ge 50700 ]; then
    # shellcheck disable=SC2086
    print_exec conda env config vars set ${env_prefix} HSA_XNACK=1
  fi

  ignored_tests=(
    ./ssd_split_table_batched_embeddings_test.py
    # https://github.com/pytorch/FBGEMM/issues/1559
    ./batched_unary_embeddings_test.py
  )
}

__setup_fbgemm_gpu_test () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Configure the environment for ignored test suites for each FBGEMM_GPU
  # variant
  if [ "$fbgemm_variant" == "cpu" ]; then
    echo "[TEST] Configuring for CPU-based testing ..."
    __configure_fbgemm_gpu_test_cpu

  elif [ "$fbgemm_variant" == "rocm" ]; then
    echo "[TEST] Configuring for ROCm-based testing ..."
    __configure_fbgemm_gpu_test_rocm

  else
    echo "[TEST] Configuring for CUDA-based testing ..."
    __configure_fbgemm_gpu_test_cuda
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

  echo "[TEST] Installing PyTest ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -y pytest expecttest) || return 1

  echo "[TEST] Checking imports ..."
  (test_python_import_package "${env_name}" fbgemm_gpu) || return 1
  (test_python_import_package "${env_name}" fbgemm_gpu.split_embedding_codegen_lookup_invokers) || return 1

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

run_fbgemm_gpu_tests () {
  env_name="$1"
  fbgemm_variant="$2"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [FBGEMM_VARIANT]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env cpu    # Run all tests applicable to CPU"
    echo "    ${FUNCNAME[0]} build_env cuda   # Run all tests applicable to CUDA"
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
  __setup_fbgemm_gpu_test

  echo "[TEST] Enumerating ALL test files ..."
  # shellcheck disable=SC2155
  local all_test_files=$(find . -type f -name '*_test.py' -print | sort)
  for f in $all_test_files; do echo "$f"; done
  echo ""

  echo "[TEST] Enumerating IGNORED test files ..."
  for f in $ignored_tests; do echo "$f"; done
  echo ""

  # NOTE: Tests running on single CPU core with a less powerful testing GPU in
  # GHA can take up to 5 hours.
  for test_file in $all_test_files; do
    if echo "${ignored_tests[@]}" | grep "${test_file}"; then
      echo "[TEST] Skipping test file: ${test_file}"
      echo ""
    elif run_python_test "${env_name}" "${test_file}"; then
      echo ""
    else
      return 1
    fi
  done
}

test_all_fbgemm_gpu_modules () {
  local env_name="$1"
  local fbgemm_variant="$2"

  local target_directories=(
    fbgemm_gpu/test
    fbgemm_gpu/experimental/example/test
  )

  for test_dir in "${target_directories[@]}"; do
    cd "${test_dir}"                                        || return 1
    run_fbgemm_gpu_tests "${env_name}" "${fbgemm_variant}"  || return 1
    cd -                                                    || return 1
  done
}


################################################################################
# FBGEMM_GPU Test Bulk-Combination Functions
################################################################################

test_setup_conda_environment () {
  local env_name="$1"
  local compiler="$2"
  local python_version="$3"
  local pytorch_installer="$4"
  local pytorch_channel_version="$5"
  local pytorch_variant_type="$6"
  local pytorch_variant_version="$7"
  if [ "$pytorch_variant_type" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME COMPILER PYTHON_VERSION PYTORCH_INSTALLER PYTORCH_CHANNEL[/VERSION] PYTORCH_VARIANT_TYPE [PYTORCH_VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env clang 3.12 pip test/0.6.0 cuda 12.1.0       # Setup environment with pytorch-test 0.6.0 for Clang + Python 3.12 + CUDA 12.1.0"
    return 1
  else
    echo "################################################################################"
    echo "# Setup FBGEMM-GPU Build Environment (All Steps)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  if [ "$env_name" == "" ]; then
    local env_name="test_py${python_version}_${pytorch_installer}_pytorch_${pytorch_channel_version}_${pytorch_variant_type}"
    if [ "$pytorch_variant_version" != "" ]; then
      local env_name="${env_name}_${pytorch_variant_version}"
    fi
  fi

  echo "Creating the Build Environment: ${env_name} ..."
  create_conda_environment  "${env_name}" "${python_version}"           || return 1

  # Install C++ compiler and build tools (all FBGEMM_GPU variants)
  if [ "$compiler" == "gcc" ] || [ "$compiler" == "clang" ]; then
    install_cxx_compiler  "${env_name}" "${compiler}"   || return 1
  fi
  install_build_tools     "${env_name}"                 || return 1

  # Install CUDA tools and runtime
  if [ "$pytorch_variant_type" == "cuda" ]; then
    install_cuda  "${env_name}" "${pytorch_variant_version}"                                            || return 1
    install_cudnn "${env_name}" "${HOME}/cudnn-${pytorch_variant_version}" "${pytorch_variant_version}" || return 1
  # Install ROCm tools and runtime
  elif [ "$pytorch_variant_type" == "rocm" ]; then
    install_rocm_ubuntu     "${env_name}" "${pytorch_variant_version}"  || return 1
  fi

  # Install PyTorch
  if [ "$pytorch_installer" == "conda" ]; then
    install_pytorch_conda     "${env_name}" "${pytorch_channel_version}" "${pytorch_variant_type}" "${pytorch_variant_version}" || return 1
  else
    install_pytorch_pip       "${env_name}" "${pytorch_channel_version}" "${pytorch_variant_type}"/"${pytorch_variant_version}" || return 1
  fi

  export env_name="${env_name}"
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
    echo "# Test FBGEMM_GPU build + installation  (All Steps)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # Assume we are starting from the repository root directory
  cd ~/FBGEMM/fbgemm_gpu                                                      || return 1
  prepare_fbgemm_gpu_build    "${env_name}"                                   || return 1
  build_fbgemm_gpu_package    "${env_name}" release "${pytorch_variant_type}" || return 1

  cd ~/FBGEMM/                                                                || return 1
  install_fbgemm_gpu_wheel    "${env_name}" fbgemm_gpu/dist/*.whl             || return 1

  cd ~/FBGEMM/                                                                || return 1
  test_all_fbgemm_gpu_modules "${env_name}" "${pytorch_variant_type}"         || return 1
}

test_fbgemm_gpu_setup_and_pip_install () {
  local variant_type="$1"
  local pytorch_channel_version="$2"
  local fbgemm_gpu_channel_version="$3"
  if [ "$fbgemm_gpu_channel_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTORCH_CHANNEL[/VERSION] FBGEMM_GPU_CHANNEL[/VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} test_env cpu test/2.2.0 test/0.6.0     # Run tests against all Python versions with PyTorch test/2.2.0 and FBGEMM_GPU test/0.6.0 (CPU-only)"
    echo "    ${FUNCNAME[0]} test_env cuda test/2.3.0 test/0.7.0    # Run tests against all Python versions with PyTorch test/2.3.0 and FBGEMM_GPU test/0.7.0 (all CUDA versions)"
    return 1
  else
    echo "################################################################################"
    echo "# Run Bulk FBGEMM-GPU Testing from PIP Package Installation"
    echo "#   (Environment Setup + Download + Install + Test)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  __single_run () {
    local py_version="$1"
    local variant_version="$2"

    local env_name="test_py${py_version}_pytorch_${pytorch_channel_version}_fbgemm_${fbgemm_gpu_channel_version}_${variant_type}/${variant_version}"
    local env_name="${env_name//\//_}"
    test_setup_conda_environment  "${env_name}" 'no-compiler' "${py_version}" pip "${pytorch_channel_version}" "${variant_type}" "${variant_version}"   || return 1
    install_fbgemm_gpu_pip        "${env_name}" "${fbgemm_gpu_channel_version}" "${variant_type}/${variant_version}"                                    || return 1
    cd ~/FBGEMM                                                                                                                                         || return 1

    test_all_fbgemm_gpu_modules "${env_name}" "${variant_type}";
    local retcode=$?

    echo "################################################################################"
    echo "# RUN SUMMARY"
    echo "#"
    echo "# Conda Environment       : ${env_name}"
    echo "# Python Version          : ${py_version}"
    echo "# PyTorch Version         : ${pytorch_channel_version}"
    echo "# FBGEMM_GPU Version      : ${fbgemm_gpu_channel_version}"
    echo "# Variant type / Version  : ${variant_type}/${variant_version}"
    echo "#"
    echo "# Run Result              : $([ $retcode -eq 0 ] && echo "PASSED" || echo "FAILED")"
    echo "################################################################################"

    cd - || return 1
    return $retcode
  }

  local python_versions=(
    3.8
    3.9
    3.10
    3.11
    3.12
  )

  if [ "$variant_type" == "cuda" ]; then
    local variant_versions=(
      11.8.0
      12.1.1
    )
  elif [ "$variant_type" == "rocm" ]; then
    local variant_versions=(
      6.0.2
    )
  elif [ "$variant_type" == "cpu" ]; then
    local variant_versions=(
      "none"
    )
  else
    echo "[TEST] Invalid variant type: ${variant_type}"
    return 1
  fi

  for py_ver in "${python_versions[@]}"; do
    for var_ver in "${variant_versions[@]}"; do
      __single_run "${py_ver}" "${var_ver}" || return 1
    done
  done
}

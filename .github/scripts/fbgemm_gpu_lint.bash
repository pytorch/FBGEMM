#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# Lint Tools Setup Functions
################################################################################

install_lint_tools () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Install Lint Tools"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "[INSTALL] Installing lint tools ..."
  (exec_with_retries conda install -n "${env_name}" -c conda-forge -y \
    click \
    flake8 \
    ufmt) || return 1

  # Check binaries are visible in the PAATH
  (test_binpath "${env_name}" flake8) || return 1
  (test_binpath "${env_name}" ufmt) || return 1

  # Check Python packages are importable
  local import_tests=( click )
  for p in "${import_tests[@]}"; do
    (test_python_import "${env_name}" "${p}") || return 1
  done

  echo "[INSTALL] Successfully installed all the lint tools"
}

################################################################################
# FBGEMM_GPU Lint Functions
################################################################################

lint_fbgemm_gpu_flake8 () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Run FBGEMM_GPU Lint: flake8"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "::add-matcher::fbgemm_gpu/test/lint/flake8_problem_matcher.json"

  # E501 = line too long
  # W503 = line break before binary operator (deprecated)
  # E203 = whitespace before ":"
  (print_exec conda run -n "${env_name}" flake8 --ignore=E501,W503,E203 .) || return 1

  echo "[TEST] Finished running flake8 lint checks"
}

lint_fbgemm_gpu_ufmt () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Run FBGEMM_GPU Lint: ufmt"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  local lint_paths=(
    fbgemm_gpu/fbgemm_gpu
    fbgemm_gpu/test
    fbgemm_gpu/bench
  )

  for p in "${lint_paths[@]}"; do
    (print_exec conda run -n "${env_name}" ufmt diff "${p}") || return 1
  done

  echo "[TEST] Finished running ufmt lint checks"
}

lint_fbgemm_gpu_copyright () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Run FBGEMM_GPU Lint: Meta Copyright Headers"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  local lint_paths=(
    fbgemm_gpu/fbgemm_gpu
    fbgemm_gpu/test
    fbgemm_gpu/bench
  )

  for p in "${lint_paths[@]}"; do
    (print_exec conda run -n "${env_name}" python fbgemm_gpu/test/lint/check_meta_header.py --path="${p}" --fixit=False) || return 1
  done

  echo "[TEST] Finished running Meta Copyright Header checks"
}

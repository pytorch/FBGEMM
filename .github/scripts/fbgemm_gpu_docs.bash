#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# Docs Tools Setup Functions
################################################################################

install_docs_tools () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Install Documentation Tools"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Installing documentation tools ..."

  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge --override-channels -y \
    doxygen \
    graphviz \
    make) || return 1

  # Check binaries are visible in the PATH
  (test_binpath "${env_name}" dot) || return 1
  (test_binpath "${env_name}" doxygen) || return 1
  (test_binpath "${env_name}" make) || return 1

  echo "[BUILD] Installing docs-build dependencies ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} python -m pip install -r requirements.txt) || return 1

  echo "[INSTALL] Successfully installed all the docs tools"
}


################################################################################
# FBGEMM_GPU Docs Functions
################################################################################

build_fbgemm_gpu_docs () {
  env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env      # Build the docs"
    return 1
  else
    echo "################################################################################"
    echo "# Build FBGEMM-GPU Documentation"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[DOCS] Running the first-pass build (i.e. documentation linting) ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} SPHINX_LINT=1

  # shellcheck disable=SC2086
  if print_exec conda run ${env_prefix} make clean doxygen; then
    echo "[DOCS] Doxygen build passed"
  else
    echo "[DOCS] Doxygen build failed!"
    return 1
  fi

  # Run the first build pass with linting enabled.  The purpose of this pass
  # is only to perform the lint checks, as the generated output will be broken
  # when linting is enabled.
    # shellcheck disable=SC2086
  if print_exec conda run ${env_prefix} make html; then
    echo "[DOCS] Docs linting passed"
  else
    echo "[DOCS] Docs linting failed; showing build output ..."
    # Show the buidl logs on error
    cat build/html/output.txt || true
    return 1
  fi

  echo "[DOCS] Running the second-pass documentation build ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars unset ${env_prefix} SPHINX_LINT

  # Run the second build pass with linting disabled.  The generated output will
  # then be used for publication.
  # shellcheck disable=SC2086
  (print_exec conda run ${env_prefix} make html) || return 1

  echo "[DOCS] FBGEMM-GPU documentation build completed"
}

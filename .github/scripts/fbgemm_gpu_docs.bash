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

  echo "[INSTALL] Installing Doxygen ..."

  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge -y \
    doxygen \
    make) || return 1

  # Check binaries are visible in the PATH
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

  echo "[BUILD] Running Doxygen build ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} doxygen Doxyfile.in) || return 1

  echo "[BUILD] Building HTML pages ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} make html) || return 1

  echo "[INSTALL] FBGEMM-GPU documentation build completed"
}

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
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "[INSTALL] Installing docs tools ..."
  (exec_with_retries conda install -n "${env_name}" -c conda-forge -y \
    doxygen) || return 1

  # Check binaries are visible in the PAATH
  (test_binpath "${env_name}" doxygen) || return 1

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
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "[BUILD] Installing docs-build dependencies ..."
  (exec_with_retries conda run -n "${env_name}" python -m pip install -r requirements.txt) || return 1

  echo "[BUILD] Running Doxygen build ..."
  (exec_with_retries conda run -n "${env_name}" doxygen Doxyfile.in) || return 1

  echo "[BUILD] Building HTML pages ..."
  (exec_with_retries conda run -n "${env_name}" make html) || return 1

  echo "[INSTALL] FBGEMM-GPU documentation build completed"
}

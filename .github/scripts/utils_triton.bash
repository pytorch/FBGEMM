#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pip.bash"

################################################################################
# Triton Setup Functions
################################################################################

install_triton_git_repo () {
  local env_name="$1"
  local triton_version="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [TRITON_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env          # Install the repo-default version of Triton"
    echo "    ${FUNCNAME[0]} build_env 2.1      # Install a designated version of Triton"
    return 1
  else
    echo "################################################################################"
    echo "# Build + Install Triton (git repo)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[BUILD] Checking out triton ..."
  cd ~/                                               || return 1
  git clone https://github.com/triton-lang/triton.git || return 1
  cd triton/python                                    || return 1

  if [ "$triton_version" != "" ]; then
    (print_exec git reset --hard "${triton_version}") || return 1
  fi

  echo "[BUILD] Installing Triton from git repo ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} python -m pip install -e .) || return 1

  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" triton) || return 1

  cd - || return 1
  echo "[INSTALL] Successfully installed Triton ${triton_version} from git repo"
}

install_triton_pip () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Install PyTorch (PyTorch PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  echo "[BUILD] Installing Triton from PIP ..."
  # NOTE: Install pytorch-triton from nightly with commit SHA that follows the
  # version tracked in FB internal (/third-party/tp2/triton/2.0.x/METADATA.bzl)
  # as close as possible.
  #
  # https://download.pytorch.org/whl/nightly/pytorch-triton/
  # https://download.pytorch.org/whl/nightly/pytorch-triton-rocm/
  #
  # This needs to be manually updated to follow PyTorch's triton pinning
  # updates:
  #
  # https://github.com/pytorch/pytorch/blob/main/.ci/docker/ci_commit_pins/triton.txt
  # https://github.com/pytorch/pytorch/pull/126098
  #
  # shellcheck disable=SC2086
  install_from_pytorch_pip "${env_name}" pytorch-triton nightly/3.0.0+dedb7bdf33 || return 1

  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" triton) || return 1

  echo "[INSTALL] Successfully installed PyTorch through PyTorch PIP"
}

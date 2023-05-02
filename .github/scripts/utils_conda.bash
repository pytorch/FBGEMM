#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# Miniconda Setup Functions
################################################################################

conda_cleanup () {
  echo "[SETUP] Cleaning up Conda packages ..."
  (print_exec conda clean --packages --tarball -y) || return 1
  (print_exec conda clean --all -y) || return 1
}

setup_miniconda () {
  local miniconda_prefix="$1"
  if [ "$miniconda_prefix" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} MINICONDA_PREFIX_PATH"
    echo "Example:"
    echo "    setup_miniconda /home/user/tmp/miniconda"
    return 1
  else
    echo "################################################################################"
    echo "# Setup Miniconda"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  # Download and install Miniconda if doesn't exist
  if [ ! -f "${miniconda_prefix}/bin/conda" ]; then
    print_exec mkdir -p "$miniconda_prefix"

    echo "[SETUP] Downloading the Miniconda installer ..."
    (exec_with_retries wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-${PLATFORM_NAME}.sh" -O miniconda.sh) || return 1

    echo "[SETUP] Installing Miniconda ..."
    print_exec bash miniconda.sh -b -p "$miniconda_prefix" -u
    print_exec rm -f miniconda.sh
  fi

  echo "[SETUP] Reloading the bash configuration ..."
  print_exec "${miniconda_prefix}/bin/conda" init bash
  print_exec . ~/.bashrc

  echo "[SETUP] Updating Miniconda base packages ..."
  (exec_with_retries conda update -n base -c defaults --update-deps -y conda) || return 1

  # Clean up packages
  conda_cleanup

  # Print Conda info
  print_exec conda info

  # These variables will be exported outside
  echo "[SETUP] Exporting Miniconda variables ..."
  export PATH="${miniconda_prefix}/bin:${PATH}"
  export CONDA="${miniconda_prefix}"

  if [ -f "${GITHUB_PATH}" ]; then
    echo "[SETUP] Saving Miniconda variables to ${GITHUB_PATH} ..."
    echo "${miniconda_prefix}/bin" >> "${GITHUB_PATH}"
    echo "CONDA=${miniconda_prefix}" >> "${GITHUB_PATH}"
  fi

  echo "[SETUP] Successfully set up Miniconda at ${miniconda_prefix}"
}

create_conda_environment () {
  local env_name="$1"
  local python_version="$2"
  if [ "$python_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTHON_VERSION"
    echo "Example:"
    echo "    ${FUNCNAME[0]} build_env 3.10"
    return 1
  else
    echo "################################################################################"
    echo "# Create Conda Environment"
    echo "#"
    echo "# [TIMESTAMP] $(date --utc +%FT%T.%3NZ)"
    echo "################################################################################"
    echo ""
  fi

  echo "[SETUP] Listing existing Conda environments ..."
  print_exec conda info --envs

  # Occasionally, we run into `CondaValueError: Value error: prefix already exists`
  # We resolve this by pre-deleting the directory, if it exists:
  # https://stackoverflow.com/questions/40180652/condavalueerror-value-error-prefix-already-exists
  echo "[SETUP] Deleting the prefix directory if it exists ..."
  # shellcheck disable=SC2155
  local conda_prefix=$(conda run -n base printenv CONDA_PREFIX)
  print_exec rm -rf "${conda_prefix}/envs/${env_name}"

  # The `-y` flag removes any existing Conda environment with the same name
  echo "[SETUP] Creating new Conda environment (Python ${python_version}) ..."
  (exec_with_retries conda create -y --name "${env_name}" python="${python_version}") || return 1

  echo "[SETUP] Upgrading PIP to latest ..."
  (exec_with_retries conda run -n "${env_name}" pip install --upgrade pip) || return 1

  # The pyOpenSSL and cryptography packages versions need to line up for PyPI publishing to work
  # https://stackoverflow.com/questions/74981558/error-updating-python3-pip-attributeerror-module-lib-has-no-attribute-openss
  echo "[SETUP] Upgrading pyOpenSSL ..."
  (exec_with_retries conda run -n "${env_name}" python -m pip install "pyOpenSSL>22.1.0") || return 1

  # This test fails with load errors if the pyOpenSSL and cryptography package versions don't align
  echo "[SETUP] Testing pyOpenSSL import ..."
  (test_python_import "${env_name}" OpenSSL) || return 1

  echo "[SETUP] Installed Python version: $(conda run -n "${env_name}" python --version)"
  echo "[SETUP] Successfully created Conda environment: ${env_name}"
}

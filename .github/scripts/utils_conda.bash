#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
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
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  if [ -f "${miniconda_prefix}/bin/conda" ]; then
    echo "[SETUP] A Miniconda installation appears to already exist in ${miniconda_prefix} ..."
    echo "[SETUP] Clearing out directory: ${miniconda_prefix} ..."
    print_exec rm -rf "${miniconda_prefix}"
  fi

  print_exec mkdir -p "$miniconda_prefix"

  echo "[SETUP] Downloading the Miniconda installer ..."
  (exec_with_retries 3 wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-${PLATFORM_NAME}.sh" -O miniconda.sh) || return 1

  echo "[SETUP] Installing Miniconda ..."
  print_exec bash miniconda.sh -b -p "$miniconda_prefix" -u
  print_exec rm -f miniconda.sh

  echo "[SETUP] Reloading the bash configuration ..."
  print_exec "${miniconda_prefix}/bin/conda" init bash
  print_exec . ~/.bashrc

  echo "[SETUP] Updating Miniconda base packages ..."
  (exec_with_retries 3 conda update -n base -c defaults --update-deps -y conda) || return 1

  # https://medium.com/data-tyro/resolving-the-conda-libmamba-issue-and-environment-activation-trouble-9f911a6106a4
  # https://www.reddit.com/r/learnpython/comments/160kjz9/how_do_i_get_anaconda_to_work_the_way_i_want_it_to/
  echo "[SETUP] Installing libmamba-solver (required since Anaconda 2024.02-1) ..."
  (exec_with_retries 3 conda install -n base conda-libmamba-solver --solver classic) || return 1

  # https://stackoverflow.com/questions/77617946/solve-conda-libmamba-solver-libarchive-so-19-error-after-updating-conda-to-23
  echo "[SETUP] Installing libarchive ..."
  (exec_with_retries 3 conda install -n base -c main libarchive --force-reinstall) || return 1

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

__handle_pyopenssl_version_issue () {
  # NOTE: pyOpenSSL needs to be above a certain version for PyPI publishing to
  # work correctly.
  local env_name="$1"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # The pyOpenSSL and cryptography packages versions need to line up for PyPI publishing to work
  # https://stackoverflow.com/questions/74981558/error-updating-python3-pip-attributeerror-module-lib-has-no-attribute-openss
  echo "[SETUP] Upgrading pyOpenSSL ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} python -m pip install "pyOpenSSL>22.1.0") || return 1

  # This test fails with load errors if the pyOpenSSL and cryptography package versions don't align
  echo "[SETUP] Testing pyOpenSSL import ..."
  (test_python_import_package "${env_name}" OpenSSL) || return 1

}

__handle_libcrypt_header_issue () {
  # NOTE: <crypt.h> appears to be missing, especially in Python 3.8 and under,
  # which results in runtime errors when `torch.compile()` is called.
  local env_name="$1"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # https://git.sr.ht/~andir/nixpkgs/commit/4ace88d63b14ef62f24d26c984775edc2ab1737c
  echo "[SETUP] Installing libxcrypt ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge -y libxcrypt) || return 1

  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
  # shellcheck disable=SC2207,SC2086
  local python_version=($(conda run --no-capture-output ${env_prefix} python --version))
  # shellcheck disable=SC2206
  local python_version_arr=(${python_version[1]//./ })

  # Copy the header file from include/ to include/python3.X/
  #   https://github.com/stanford-futuredata/ColBERT/issues/309
  echo "[SETUP] Copying <crypt.h> over ..."
  # shellcheck disable=SC2206
  local dst_file="${conda_prefix}/include/python${python_version_arr[0]}.${python_version_arr[1]}/crypt.h"
  print_exec cp "${conda_prefix}/include/crypt.h" "${dst_file}"
}

create_conda_environment () {
  local env_name="$1"
  # shellcheck disable=SC2178
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
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  echo "[SETUP] Listing existing Conda environments ..."
  print_exec conda info --envs

  # Occasionally, we run into `CondaValueError: Value error: prefix already exists`
  # We resolve this by pre-deleting the directory, if it exists:
  # https://stackoverflow.com/questions/40180652/condavalueerror-value-error-prefix-already-exists
  echo "[SETUP] Deleting the prefix directory if it exists ..."
  # shellcheck disable=SC2155
  local conda_prefix=$(conda run -n base printenv CONDA_PREFIX)
  print_exec rm -rf "${conda_prefix}/envs/${env_name}"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # The `-y` flag removes any existing Conda environment with the same name
  echo "[SETUP] Creating new Conda environment (Python ${python_version}) ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda create -y ${env_prefix} python="${python_version}") || return 1

  echo "[SETUP] Upgrading PIP to latest ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} pip install --upgrade pip) || return 1

  # Handle pyOpenSSL version issue
  __handle_pyopenssl_version_issue "${env_name}"

  # Handle missing <crypt.h> issue
  __handle_libcrypt_header_issue "${env_name}"

  # shellcheck disable=SC2086
  echo "[SETUP] Installed Python version: $(conda run ${env_prefix} python --version)"
  echo "[SETUP] Successfully created Conda environment: ${env_name}"
}

print_conda_info () {
  echo "################################################################################"
  echo "# Print Conda Environment Info"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""
  print_exec conda info
  echo ""
  print_exec conda info --envs
  echo ""
  # shellcheck disable=SC2153
  echo "PYTHON_VERSION:     ${PYTHON_VERSION}"
  echo "python3 --version:  $(python3 --version)"
}

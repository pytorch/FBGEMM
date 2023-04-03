#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


################################################################################
# Command Execution Functions
################################################################################

print_exec () {
  echo "+ $*"
  echo ""
  if "$@"; then
    local retcode=0
  else
    local retcode=$?
  fi
  echo ""
  return $retcode
}

exec_with_retries () {
  local max=5
  local delay=2
  local retcode=0

  for i in $(seq 1 ${max}); do
    echo "[EXEC] [ATTEMPT ${i}/${max}]    + $*"

    if "$@"; then
      retcode=0
      break
    else
      retcode=$?
      echo "[EXEC] [ATTEMPT ${i}/${max}] Command attempt failed."
      echo ""
      sleep $delay
    fi
  done

  if [ $retcode -ne 0 ]; then
    echo "[EXEC] The command has failed after ${max} attempts; aborting."
  fi

  return $retcode
}


################################################################################
# Assert Functions
################################################################################

test_python_import () {
  local env_name="$1"
  local python_import="$2"
  if [ "$python_import" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTHON_IMPORT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env numpy"
    return 1
  fi

  if conda run -n "${env_name}" python -c "import ${python_import}"; then
    echo "[CHECK] Python package ${python_import} found."
  else
    echo "[CHECK] Python package ${python_import} not found!"
    return 1
  fi
}

test_binpath () {
  local env_name="$1"
  local bin_name="$2"
  if [ "$bin_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME BIN_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env nvcc"
    return 1
  fi

  if conda run -n "${env_name}" which "${bin_name}"; then
    echo "[CHECK] Binary ${bin_name} found in PATH"
  else
    echo "[CHECK] Binary ${bin_name} not found in PATH!"
    return 1
  fi
}

test_filepath () {
  local env_name="$1"
  local file_name="$2"
  if [ "$file_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME FILE_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env cuda_runtime.h"
    return 1
  fi

  # shellcheck disable=SC2155
  local conda_prefix=$(conda run -n "${env_name}" printenv CONDA_PREFIX)
  # shellcheck disable=SC2155
  local file_path=$(find "${conda_prefix}" -type f -name "${file_name}")
  # shellcheck disable=SC2155
  local link_path=$(find "${conda_prefix}" -type l -name "${file_name}")
  if [ "${file_path}" != "" ]; then
    echo "[CHECK] ${file_name} found in CONDA_PREFIX PATH (file): ${file_path}"
  elif [ "${link_path}" != "" ]; then
    echo "[CHECK] ${file_name} found in CONDA_PREFIX PATH (symbolic link): ${link_path}"
  else
    echo "[CHECK] ${file_name} not found in CONDA_PREFIX PATH!"
    return 1
  fi
}

test_env_var () {
  local env_name="$1"
  local env_key="$2"
  if [ "$env_key" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME ENV_KEY"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env CUDNN_INCLUDE_DIR"
    return 1
  fi

  if conda run -n "${env_name}" printenv "${env_key}"; then
    echo "[CHECK] Environment variable ${env_key} is defined in the Conda environment"
  else
    echo "[CHECK] Environment variable ${env_key} is not defined in the Conda environment!"
    return 1
  fi
}

test_library_symbol () {
  local lib_path="$1"
  local lib_symbol="$2"
  if [ "$lib_symbol" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} LIB_PATH FULL_NAMESPACE_PATH_LIB_SYMBOL"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} fbgemm_gpu_py.so fbgemm_gpu::merge_pooled_embeddings"
    return 1
  fi

  # Add space and '(' to the grep string to get the full method path
  symbol_entries=$(nm -gDC "${lib_path}" | grep " ${lib_symbol}(")
  if [ "${symbol_entries}" != "" ]; then
    echo "[CHECK] Found symbol in ${lib_path}: ${lib_symbol}"
  else
    echo "[CHECK] Symbol NOT found in ${lib_path}: ${lib_symbol}"
    return 1
  fi
}

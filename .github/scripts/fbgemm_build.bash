#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# FBGEMM Build Auxiliary Functions
################################################################################

__configure_fbgemm_build_gcc () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # shellcheck disable=SC2155,SC2086
  local python_path=$(conda run ${env_prefix} which python)

  # shellcheck disable=SC2206
  build_args=(
    -DUSE_SANITIZER=address
    -DFBGEMM_LIBRARY_TYPE=${fbgemm_library_type}
    -DPYTHON_EXECUTABLE=${python_path}
  )
}

__configure_fbgemm_build_clang () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # shellcheck disable=SC2155,SC2086
  local python_path=$(conda run ${env_prefix} which python)
  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)

  # shellcheck disable=SC2206
  build_args=(
    -DUSE_SANITIZER=address
    -DFBGEMM_LIBRARY_TYPE=${fbgemm_library_type}
    -DPYTHON_EXECUTABLE=${python_path}
    -DOpenMP_C_LIB_NAMES=libomp
    -DOpenMP_C_FLAGS=\"-fopenmp=libomp -I ${conda_prefix}/include\"
    -DOpenMP_CXX_LIB_NAMES=libomp
    -DOpenMP_CXX_FLAGS=\"-fopenmp=libomp -I ${conda_prefix}/include\"
    -DOpenMP_libomp_LIBRARY=${conda_prefix}/lib/libomp.so
  )
}

__configure_fbgemm_build () {
  if [ "$fbgemm_compiler" == "clang" ]; then
    echo "[BUILD] Configuring for building using Clang ..."
    __configure_fbgemm_build_clang

  else
    echo "[BUILD] Configuring for building using GCC ..."
    __configure_fbgemm_build_gcc
  fi

  # shellcheck disable=SC2145
  echo "[BUILD] FBGEMM build arguments have been set:  ${build_args[@]}"
}

################################################################################
# FBGEMM_GPU Build Functions
################################################################################

build_fbgemm_library () {
  env_name="$1"
  local build_dir="$2"
  fbgemm_library_type="$3"
  fbgemm_compiler="$4"
  if [ "$fbgemm_compiler" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME BUILD_DIR LIBRARY_TYPE COMPILER"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env shared clang   # Build shared library using Clang"
    echo "    ${FUNCNAME[0]} build_env static gcc     # Build static library using GCC"
    return 1
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "################################################################################"
  echo "# Build FBGEMM Library"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  # Set up and configure the build
  __configure_fbgemm_build || return 1

  mkdir "$build_dir" || return 1
  cd "$build_dir" || return 1

  echo "[BUILD] Running CMake ..."
  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} \
    cmake "${build_args[@]}" ..

  echo "[BUILD] Running the build ..."
  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} \
    make -j VERBOSE=1

  cd - || return 1
}

################################################################################
# FBGEMM_GPU Test Functions
################################################################################

test_fbgemm_library () {
  local env_name="$1"
  local build_dir="$2"
  if [ "$build_dir" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME BUILD_DIR"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env build    # Run tests"
    return 1
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  cd "$build_dir" || return 1

  echo "[BUILD] Running FBGEMM tests ..."
  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} \
    ctest --rerun-failed --output-on-failure

  cd - || return 1
}

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

__configure_fbgemm_build () {
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

  if print_exec "conda run ${env_prefix} c++ --version | grep -i clang"; then
    echo "[BUILD] Host compiler is Clang; adding extra compiler flags ..."

    # shellcheck disable=SC2155,SC2086
    local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
    # shellcheck disable=SC2155,SC2086
    local cc_path=$(conda run ${env_prefix} which cc)
    # shellcheck disable=SC2155,SC2086
    local cxx_path=$(conda run ${env_prefix} which c++)

    # shellcheck disable=SC2206
    build_args+=(
      -DCMAKE_C_COMPILER="${cc_path}"
      -DCMAKE_CXX_COMPILER="${cxx_path}"
      -DCMAKE_C_FLAGS=\"-fopenmp=libomp -stdlib=libc++ -I ${conda_prefix}/include\"
      -DCMAKE_CXX_FLAGS=\"-fopenmp=libomp -stdlib=libc++ -I ${conda_prefix}/include\"
    )
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
  if [ "$fbgemm_library_type" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME BUILD_DIR LIBRARY_TYPE COMPILER"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env shared   # Build shared library"
    echo "    ${FUNCNAME[0]} build_env static   # Build static library"
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

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

__configure_fbgemm_build_cmake () {
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

__build_fbgemm_library_cmake () {
  # Set up and configure the build
  __configure_fbgemm_build_cmake || return 1

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  mkdir "$build_dir" || return 1
  cd "$build_dir" || return 1

  echo "[BUILD] Running CMake ..."
  # shellcheck disable=SC2086
  (print_exec conda run --no-capture-output ${env_prefix} \
    cmake "${build_args[@]}" ..) || return 1

  echo "[BUILD] Running the build ..."
  # shellcheck disable=SC2086
  (print_exec conda run --no-capture-output ${env_prefix} \
    make -j VERBOSE=1) || return 1

  cd - || return 1
}

__build_fbgemm_library_bazel () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # shellcheck disable=SC2155,SC2086
  local cc_path=$(conda run ${env_prefix} which cc)
  # shellcheck disable=SC2155,SC2086
  local cxx_path=$(conda run ${env_prefix} which c++)

  echo "[BUILD] Running Bazel ..."
  # Prefix CC and CXX directly before the invocation to force bazel to build
  # using the specified compiler
  #
  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} \
    CC="${cc_path}" CXX="${cxx_path}" bazel build -s :*
}

build_fbgemm_library () {
  env_name="$1"
  build_system="$2"
  build_dir="$3"
  fbgemm_library_type="$4"
  if [ "$build_system" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME BUILD_SYSTEM [BUILD_DIR] [LIBRARY_TYPE]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env bazel                    # Build library using Bazel (static)"
    echo "    ${FUNCNAME[0]} build_env cmake build_dir shared   # Build library using CMake (shared)"
    echo "    ${FUNCNAME[0]} build_env cmake build_dir static   # Build library using CMake (static)"
    return 1
  fi

  echo "################################################################################"
  echo "# Build FBGEMM Library"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  if [ "$build_system" == "cmake" ]; then
    __build_fbgemm_library_cmake
  elif [ "$build_system" == "bazel" ]; then
    __build_fbgemm_library_bazel
  else
    echo "[BUILD] Unknown build system; select either cmake or bazel!"
    return 1
  fi
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

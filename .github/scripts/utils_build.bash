#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# Bazel Setup Functions
################################################################################

setup_bazel () {
  local bazel_version="${1:-6.1.1}"
  echo "################################################################################"
  echo "# Setup Bazel"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  test_network_connection || return 1

  local bazel_variant="$PLATFORM_NAME_LC"
  echo "[SETUP] Downloading installer Bazel ${bazel_version} (${bazel_variant}) ..."
  print_exec wget -q "https://github.com/bazelbuild/bazel/releases/download/${bazel_version}/bazel-${bazel_version}-installer-${bazel_variant}.sh" -O install-bazel.sh

  echo "[SETUP] Installing Bazel ..."
  print_exec bash install-bazel.sh
  print_exec rm -f install-bazel.sh

  print_exec bazel --version
  echo "[SETUP] Successfully set up Bazel"
}


################################################################################
# Build Tools Setup Functions
################################################################################

install_cxx_compiler () {
  local env_name="$1"
  local use_system_package_manager="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [USE_YUM]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env     # Install C/C++ compilers through Conda"
    echo "    ${FUNCNAME[0]} build_env 1   # Install C/C++ compilers through the system package manager"
    return 1
  else
    echo "################################################################################"
    echo "# Install C/C++ Compilers"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  if [ "$use_system_package_manager" != "" ]; then
    echo "[INSTALL] Installing C/C++ compilers through the system package manager ..."
    install_system_packages gcc gcc-c++

  else
    # Install gxx_linux-<arch> from conda-forge instead of from anaconda channel.
    # sysroot_linux-<arch> needs to be installed alongside this:
    #
    #   https://root-forum.cern.ch/t/error-timespec-get-has-not-been-declared-with-conda-root-package/45712/6
    #   https://github.com/conda-forge/conda-forge.github.io/issues/1625
    #   https://conda-forge.org/docs/maintainer/knowledge_base.html#using-centos-7
    #   https://github.com/conda/conda-build/issues/4371
    #
    # NOTE: We install g++ 10.x instead of 11.x becaue 11.x builds binaries that
    # reference GLIBCXX_3.4.29, which may not be available on systems with older
    # versions of libstdc++.so.6 such as CentOS Stream 8 and Ubuntu 20.04
    local archname=""
    if [ "$MACHINE_NAME_LC" = "x86_64" ]; then
      archname="64"
    elif [ "$MACHINE_NAME_LC" = "aarch64" ] || [ "$MACHINE_NAME_LC" = "arm64" ]; then
      archname="aarch64"
    else
      archname="$MACHINE_NAME_LC"
    fi

    # shellcheck disable=SC2155
    local env_prefix=$(env_name_or_prefix "${env_name}")

    echo "[INSTALL] Installing C/C++ compilers through Conda (architecture = ${archname}) ..."
    # shellcheck disable=SC2086
    (exec_with_retries 3 conda install ${env_prefix} -y "gxx_linux-${archname}"=10.4.0 "sysroot_linux-${archname}"=2.17 -c conda-forge) || return 1

    # The compilers are visible in the PATH as `x86_64-conda-linux-gnu-cc` and
    # `x86_64-conda-linux-gnu-c++`, so symlinks will need to be created
    echo "[INSTALL] Setting the C/C++ compiler symlinks ..."
    # shellcheck disable=SC2155,SC2086
    local cc_path=$(conda run ${env_prefix} printenv CC)
    # shellcheck disable=SC2155,SC2086
    local cxx_path=$(conda run ${env_prefix} printenv CXX)

    print_exec ln -s "${cc_path}" "$(dirname "$cc_path")/cc"
    print_exec ln -s "${cc_path}" "$(dirname "$cc_path")/gcc"
    print_exec ln -s "${cxx_path}" "$(dirname "$cxx_path")/c++"
    print_exec ln -s "${cxx_path}" "$(dirname "$cxx_path")/g++"
  fi

  # Check C/C++ compilers are visible
  (test_binpath "${env_name}" cc) || return 1
  (test_binpath "${env_name}" gcc) || return 1
  (test_binpath "${env_name}" c++) || return 1
  (test_binpath "${env_name}" g++) || return 1

  # https://stackoverflow.com/questions/2224334/gcc-dump-preprocessor-defines
  echo "[INFO] Printing out all preprocessor defines in the C compiler ..."
  # shellcheck disable=SC2086
  print_exec conda run ${env_prefix} cc -dM -E -

  # https://stackoverflow.com/questions/2224334/gcc-dump-preprocessor-defines
  echo "[INFO] Printing out all preprocessor defines in the C++ compiler ..."
  # shellcheck disable=SC2086
  print_exec conda run ${env_prefix} c++ -dM -E -x c++ -

  # Print out the C++ version
  # shellcheck disable=SC2086
  print_exec conda run ${env_prefix} c++ --version

  # https://stackoverflow.com/questions/4991707/how-to-find-my-current-compilers-standard-like-if-it-is-c90-etc
  echo "[INFO] Printing the default version of the C standard used by the compiler ..."
  print_exec "conda run ${env_prefix} cc -dM -E - | grep __STDC_VERSION__"

  # https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
  echo "[INFO] Printing the default version of the C++ standard used by the compiler ..."
  print_exec "conda run ${env_prefix} c++ -dM -E -x c++ - | grep __cplusplus"

  echo "[INSTALL] Successfully installed C/C++ compilers"
}

install_build_tools () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Install Build Tools"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Installing build tools ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -y \
    click \
    cmake \
    hypothesis \
    jinja2 \
    ninja \
    numpy \
    scikit-build \
    wheel) || return 1

  # Check binaries are visible in the PAATH
  (test_binpath "${env_name}" cmake) || return 1
  (test_binpath "${env_name}" ninja) || return 1

  # Check Python packages are importable
  local import_tests=( click hypothesis jinja2 numpy skbuild wheel )
  for p in "${import_tests[@]}"; do
    (test_python_import_package "${env_name}" "${p}") || return 1
  done

  echo "[INSTALL] Successfully installed all the build tools"
}

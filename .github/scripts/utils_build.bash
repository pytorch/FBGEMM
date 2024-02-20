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

__extract_archname () {
  export archname=""
  if [ "$MACHINE_NAME_LC" = "x86_64" ]; then
    export archname="64"
  elif [ "$MACHINE_NAME_LC" = "aarch64" ] || [ "$MACHINE_NAME_LC" = "arm64" ]; then
    export archname="aarch64"
  else
    export archname="$MACHINE_NAME_LC"
  fi
}

__conda_install_glibc () {
  # sysroot_linux-<arch> needs to be installed alongside the C/C++ compiler for GLIBC:
  #   https://root-forum.cern.ch/t/error-timespec-get-has-not-been-declared-with-conda-root-package/45712/6
  #   https://github.com/conda-forge/conda-forge.github.io/issues/1625
  #   https://conda-forge.org/docs/maintainer/knowledge_base.html#using-centos-7
  #   https://github.com/conda/conda-build/issues/4371

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Installing GLIBC (architecture = ${archname}) ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge -y "sysroot_linux-${archname}"=2.17) || return 1
}

__conda_install_gcc () {
  # Install gxx_linux-<arch> from conda-forge instead of from anaconda channel.
  #
  # NOTE: We install g++ 10.x instead of 11.x becaue 11.x builds binaries that
  # reference GLIBCXX_3.4.29, which may not be available on systems with older
  # versions of libstdc++.so.6 such as CentOS Stream 8 and Ubuntu 20.04

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Installing GCC through Conda (architecture = ${archname}) ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge -y "gxx_linux-${archname}"=10.4.0 ) || return 1

  # The compilers are visible in the PATH as `x86_64-conda-linux-gnu-cc` and
  # `x86_64-conda-linux-gnu-c++`, so symlinks will need to be created
  echo "[INSTALL] Setting the C/C++ compiler symlinks ..."
  # shellcheck disable=SC2155,SC2086
  local cc_path=$(conda run ${env_prefix} printenv CC)
  # shellcheck disable=SC2155,SC2086
  local cxx_path=$(conda run ${env_prefix} printenv CXX)

  print_exec ln -sf "${cc_path}" "$(dirname "$cc_path")/cc"
  print_exec ln -sf "${cc_path}" "$(dirname "$cc_path")/gcc"
  print_exec ln -sf "${cxx_path}" "$(dirname "$cxx_path")/c++"
  print_exec ln -sf "${cxx_path}" "$(dirname "$cxx_path")/g++"
}

__conda_install_clang () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # shellcheck disable=SC2155
  local llvm_version=15.0.7

  echo "[INSTALL] Installing Clang and relevant libraries through Conda ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge -y \
    clangxx=${llvm_version} \
    libcxx \
    llvm-openmp=${llvm_version} \
    compiler-rt=${llvm_version}) || return 1

  # libcxx from conda-forge is very outdated for linux-aarch64
  # echo "[INSTALL] Installing LLVM libcxx from Anaconda channel..."
  # (exec_with_retries 3 conda install ${env_prefix} -c anaconda -y libcxx) || return 1

  # The compilers are visible in the PATH as `clang` and `clang++`, so symlinks
  # will need to be created
  echo "[INSTALL] Setting the C/C++ compiler symlinks ..."
  # shellcheck disable=SC2155,SC2086
  local cc_path=$(conda run ${env_prefix} which clang)
  # shellcheck disable=SC2155,SC2086
  local cxx_path=$(conda run ${env_prefix} which clang++)

  print_exec ln -sf "${cc_path}" "$(dirname "$cc_path")/cc"
  print_exec ln -sf "${cc_path}" "$(dirname "$cc_path")/gcc"
  print_exec ln -sf "${cxx_path}" "$(dirname "$cxx_path")/c++"
  print_exec ln -sf "${cxx_path}" "$(dirname "$cxx_path")/g++"

  echo "[INSTALL] Updating LD_LIBRARY_PATH ..."
  # shellcheck disable=SC2155,SC2086
  local ld_library_path=$(conda run ${env_prefix} printenv LD_LIBRARY_PATH)
  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} LD_LIBRARY_PATH="${ld_library_path}:${conda_prefix}/lib"

  echo "[BUILD] Setting Clang (should already be symlinked as c++) as the host compiler for NVCC: ${cxx_path}"
  # When NVCC is used, set Clang to be the host compiler, but set GNU libstdc++
  # (not Clang libc++) as the standard library
  #
  # NOTE: There appears to be no ROCm equivalent for NVCC_PREPEND_FLAGS:
  #   https://github.com/ROCm/HIP/issues/931
  #
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} NVCC_PREPEND_FLAGS=\"-std=c++17 -Xcompiler -std=c++17 -Xcompiler -stdlib=libstdc++ -ccbin ${cxx_path} -allow-unsupported-compiler\"
}

__compiler_post_install_checks () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

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
  print_exec "conda run ${env_prefix} cc -dM -E - < /dev/null | grep __STDC_VERSION__"

  # https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
  echo "[INFO] Printing the default version of the C++ standard used by the compiler ..."
  print_exec "conda run ${env_prefix} c++ -dM -E -x c++ - < /dev/null | grep __cplusplus"
}

install_cxx_compiler () {
  env_name="$1"
  local compiler="$2"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME [USE_YUM]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env clang  # Install C/C++ compilers (clang)"
    echo "    ${FUNCNAME[0]} build_env gcc    # Install C/C++ compilers (gcc)"
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

  # Extract the archname
  __extract_archname

  # Install GLIBC
  __conda_install_glibc

  # Install GCC and libstdc++
  # NOTE: We unconditionally install libstdc++ here because CUDA only supports
  # libstdc++, even if host compiler is set to Clang:
  #   https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy
  #   https://forums.developer.nvidia.com/t/cuda-issues-with-clang-compiler/177589/8
  __conda_install_gcc

  # Install the C/C++ compiler
  if [ "$compiler" == "clang" ]; then
    # Existing symlinks to cc / c++ / gcc / g++ will be overridden
    __conda_install_clang
  fi

  # Run post-install checks
  __compiler_post_install_checks
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
  # NOTE: Only the openblas package will install cblas.h directly into
  # $CONDA_PREFIX/include directory
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge -y \
    click \
    cmake \
    hypothesis \
    jinja2 \
    make \
    ninja \
    numpy \
    openblas \
    scikit-build \
    wheel) || return 1

  # Check binaries are visible in the PAATH
  (test_binpath "${env_name}" make) || return 1
  (test_binpath "${env_name}" cmake) || return 1
  (test_binpath "${env_name}" ninja) || return 1

  # Check Python packages are importable
  local import_tests=( click hypothesis jinja2 numpy skbuild wheel )
  for p in "${import_tests[@]}"; do
    (test_python_import_package "${env_name}" "${p}") || return 1
  done

  echo "[INSTALL] Successfully installed all the build tools"
}

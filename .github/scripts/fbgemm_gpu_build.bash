#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# FBGEMM_GPU Build Auxiliary Functions
################################################################################

prepare_fbgemm_gpu_build () {
  local env_name="$1"
  if [ "$env_name" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env"
    return 1
  else
    echo "################################################################################"
    echo "# Prepare FBGEMM-GPU Build"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  if [[ "${GITHUB_WORKSPACE}" ]]; then
    # https://github.com/actions/checkout/issues/841
    git config --global --add safe.directory "${GITHUB_WORKSPACE}"
  fi

  echo "[BUILD] Running git submodules update ..."
  git submodule sync
  git submodule update --init --recursive

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[BUILD] Installing other build dependencies ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} python -m pip install -r requirements.txt) || return 1

  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" numpy) || return 1
  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" skbuild) || return 1

  echo "[BUILD] Successfully ran git submodules update"
}

__configure_fbgemm_gpu_build_cpu () {
  # Update the package name and build args depending on if CUDA is specified
  echo "[BUILD] Setting CPU-only build args ..."
  build_args=(
    --package_variant=cpu
  )
}

__configure_fbgemm_gpu_build_rocm () {
  local fbgemm_variant_targets="$1"

  # Fetch available ROCm architectures on the machine
  if [ "$fbgemm_variant_targets" != "" ]; then
    echo "[BUILD] ROCm targets have been manually provided: ${fbgemm_variant_targets}"
    local arch_list="${fbgemm_variant_targets}"
  else
    if which rocminfo; then
      # shellcheck disable=SC2155
      local arch_list=$(rocminfo | grep -o -m 1 'gfx.*')
      echo "[BUILD] Architectures list from rocminfo: ${arch_list}"

      if [ "$arch_list" == "" ]; then
        # It is possible to build FBGEMM_GPU-ROCm on a machine without AMD
        # cards, in which case the arch_list will be empty.
        echo "[BUILD] rocminfo did not return anything valid!"

        # By default, we build just for MI100 and MI250 to save time.  This list
        # needs to be updated if the CI ROCm machines have different hardware.
        # Architecture mapping can be found at: https://wiki.gentoo.org/wiki/ROCm
        local arch_list="gfx908,gfx90a"
      fi
    else
      echo "[BUILD] rocminfo not found in PATH!"
    fi
  fi

  echo "[BUILD] Setting the following ROCm targets: ${arch_list}"
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} PYTORCH_ROCM_ARCH="${arch_list}"

  echo "[BUILD] Setting ROCm build args ..."
  build_args=(
    --package_variant=rocm
    # HIP_ROOT_DIR now required for HIP to be correctly detected by CMake
    -DHIP_ROOT_DIR=/opt/rocm
    # Enable device-side assertions in HIP
    # https://stackoverflow.com/questions/44284275/passing-compiler-options-in-cmake-command-line
    -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA"
    -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA"
  )
}

__configure_fbgemm_gpu_build_cuda () {
  local fbgemm_variant_targets="$1"

  # Check nvcc is visible
  (test_binpath "${env_name}" nvcc) || return 1

  # Check that cuDNN environment variables are available
  (test_env_var "${env_name}" CUDNN_INCLUDE_DIR) || return 1
  (test_env_var "${env_name}" CUDNN_LIBRARY) || return 1
  (test_env_var "${env_name}" NVML_LIB_PATH) || return 1

  if [ "$fbgemm_variant_targets" != "" ]; then
    echo "[BUILD] Using the user-supplied CUDA targets ..."
    local arch_list="${fbgemm_variant_targets}"

  elif [ "$TORCH_CUDA_ARCH_LIST" != "" ]; then
    echo "[BUILD] Using the environment-supplied TORCH_CUDA_ARCH_LIST as the CUDA targets ..."
    local arch_list="${TORCH_CUDA_ARCH_LIST}"

  else
    echo "[BUILD] Using the default CUDA targets ..."
    # For cuda version 12.1, enable sm 9.0
    cuda_version_nvcc=$(conda run -n "${env_name}" nvcc --version)
    echo "$cuda_version_nvcc"
    if [[ $cuda_version_nvcc == *"V12.1"* ]]; then
      local arch_list="7.0;8.0;9.0"
    else
      local arch_list="7.0;8.0"
    fi
  fi

  # Unset the environment-supplied TORCH_CUDA_ARCH_LIST because it will take
  # precedence over cmake -DTORCH_CUDA_ARCH_LIST
  unset TORCH_CUDA_ARCH_LIST

  echo "[BUILD] Setting the following CUDA targets: ${arch_list}"

  # Build only CUDA 7.0 and 8.0 (i.e. V100 and A100) because of 100 MB binary size limits from PyPI.
  echo "[BUILD] Setting CUDA build args ..."
  # shellcheck disable=SC2155,SC2086
  local nvml_lib_path=$(conda run --no-capture-output ${env_prefix} printenv NVML_LIB_PATH)
  build_args=(
    --package_variant=cuda
    --nvml_lib_path="${nvml_lib_path}"
    # Pass to PyTorch CMake
    -DTORCH_CUDA_ARCH_LIST="'${arch_list}'"
  )
}

__configure_fbgemm_gpu_build () {
  local fbgemm_variant="$1"
  local fbgemm_variant_targets="$2"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} FBGEMM_VARIANT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} cpu                          # CPU-only variant"
    echo "    ${FUNCNAME[0]} cuda                         # CUDA variant for default target(s)"
    echo "    ${FUNCNAME[0]} cuda '7.0;8.0'               # CUDA variant for custom target(s)"
    echo "    ${FUNCNAME[0]} rocm                         # ROCm variant for default target(s)"
    echo "    ${FUNCNAME[0]} rocm 'gfx906;gfx908;gfx90a'  # ROCm variant for custom target(s)"
    return 1
  else
    echo "################################################################################"
    echo "# Configure FBGEMM-GPU Build"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  if [ "$fbgemm_variant" == "cpu" ]; then
    echo "[BUILD] Configuring build as CPU variant ..."
    __configure_fbgemm_gpu_build_cpu

  elif [ "$fbgemm_variant" == "rocm" ]; then
    echo "[BUILD] Configuring build as ROCm variant ..."
    __configure_fbgemm_gpu_build_rocm "${fbgemm_variant_targets}"

  else
    echo "[BUILD] Configuring build as CUDA variant (this is the default behavior) ..."
    __configure_fbgemm_gpu_build_cuda "${fbgemm_variant_targets}"
  fi

  # shellcheck disable=SC2145
  echo "[BUILD] FBGEMM_GPU build arguments have been set:  ${build_args[@]}"
}

__build_fbgemm_gpu_common_pre_steps () {
  # Private function that uses variables instantiated by its caller

  # Check C/C++ compilers are visible (the build scripts look specifically for `gcc`)
  (test_binpath "${env_name}" cc) || return 1
  (test_binpath "${env_name}" gcc) || return 1
  (test_binpath "${env_name}" c++) || return 1
  (test_binpath "${env_name}" g++) || return 1

  # Determine the package name based on release type and variant
  package_name="fbgemm_gpu"
  if [ "$fbgemm_release_type" != "release" ]; then
    package_name="${package_name}_${fbgemm_release_type}"
  fi
  if [ "$fbgemm_variant" == "cpu" ]; then
    package_name="${package_name}-cpu"
  elif [ "$fbgemm_variant" == "rocm" ]; then
    package_name="${package_name}-rocm"
  else
    # Set to the default variant
    fbgemm_variant="cuda"
  fi
  echo "[BUILD] Determined Python package name to use: ${package_name}"

  # Extract the Python tag
  # shellcheck disable=SC2207,SC2086
  python_version=($(conda run --no-capture-output ${env_prefix} python --version))
  # shellcheck disable=SC2206
  python_version_arr=(${python_version[1]//./ })
  python_tag="py${python_version_arr[0]}${python_version_arr[1]}"
  echo "[BUILD] Extracted Python tag: ${python_tag}"

  echo "[BUILD] Running pre-build cleanups ..."
  print_exec rm -rf dist
  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} python setup.py clean

  echo "[BUILD] Printing git status ..."
  print_exec git status
  print_exec git diff
}

run_fbgemm_gpu_postbuild_checks () {
  local fbgemm_variant="$1"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} FBGEMM_VARIANT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} cpu"
    echo "    ${FUNCNAME[0]} cuda"
    echo "    ${FUNCNAME[0]} rocm"
    return 1
  fi

  # Find the .SO file
  # shellcheck disable=SC2155
  local fbgemm_gpu_so_files=$(find . -name fbgemm_gpu_py.so)
  readarray -t fbgemm_gpu_so_files <<<"$fbgemm_gpu_so_files"
  if [ "${#fbgemm_gpu_so_files[@]}" -le 0 ]; then
    echo "[CHECK] .SO library fbgemm_gpu_py.so is missing from the build path!"
    return 1
  fi

  # Prepare a sample set of symbols whose existence in the built library should be checked
  # This is by no means an exhaustive set, and should be updated accordingly
  local lib_symbols_to_check=(
    fbgemm_gpu::asynchronous_inclusive_cumsum_cpu
    fbgemm_gpu::jagged_2d_to_dense
  )

  # Add more symbols to check for if it's a non-CPU variant
  if [ "${fbgemm_variant}" == "cuda" ]; then
    lib_symbols_to_check+=(
      fbgemm_gpu::asynchronous_inclusive_cumsum_gpu
      fbgemm_gpu::merge_pooled_embeddings
    )
  elif [ "${fbgemm_variant}" == "rocm" ]; then
    # merge_pooled_embeddings is missing in ROCm builds bc it requires NVML
    lib_symbols_to_check+=(
      fbgemm_gpu::asynchronous_inclusive_cumsum_gpu
      fbgemm_gpu::merge_pooled_embeddings
    )
  fi

  # Print info for only the first instance of the .SO file, since the build makes multiple copies
  local library="${fbgemm_gpu_so_files[0]}"

  echo "[CHECK] Listing out library size: ${library}"
  print_exec "du -h --block-size=1M ${library}"

  echo "[CHECK] Listing out the GLIBCXX versions referenced by the library: ${library}"
  print_glibc_info "${library}"

  echo "[CHECK] Listing out undefined symbols in the library: ${library}"
  print_exec "nm -gDCu ${library} | sort"

  echo "[CHECK] Listing out external shared libraries required by the library: ${library}"
  print_exec ldd "${library}"

  echo "[CHECK] Verifying sample subset of symbols in the library ..."
  for symbol in "${lib_symbols_to_check[@]}"; do
    (test_library_symbol "${library}" "${symbol}") || return 1
  done
}

################################################################################
# FBGEMM_GPU Build Functions
################################################################################

build_fbgemm_gpu_package () {
  env_name="$1"
  fbgemm_release_type="$2"
  fbgemm_variant="$3"
  fbgemm_variant_targets="$4"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME RELEASE_TYPE VARIANT [VARIANT_TARGETS]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env nightly cpu                           # Nightly CPU-only variant"
    echo "    ${FUNCNAME[0]} build_env nightly cuda                          # Nightly CUDA variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env nightly cuda '7.0;8.0'                # Nightly CUDA variant for custom target(s)"
    echo "    ${FUNCNAME[0]} build_env release rocm                          # Release ROCm variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env release rocm 'gfx906;gfx908;gfx90a'   # Release ROCm variant for custom target(s)"
    return 1
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Set up and configure the build
  __build_fbgemm_gpu_common_pre_steps || return 1
  __configure_fbgemm_gpu_build "${fbgemm_variant}" "${fbgemm_variant_targets}" || return 1

  echo "################################################################################"
  echo "# Build FBGEMM-GPU Package (Wheel)"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  # manylinux2014 is specified, bc manylinux1 does not support aarch64
  # See https://github.com/pypa/manylinux
  local plat_name="manylinux2014_${MACHINE_NAME}"

  echo "[BUILD] Checking arch_list = ${arch_list}"
  echo "[BUILD] Checking build_args:"
  echo "${build_args[@]}"

  # shellcheck disable=SC2155
  local core=$(lscpu | grep "Core(s)" | awk '{print $NF}') && echo "core = ${core}" || echo "core not found"
  # shellcheck disable=SC2155
  local sockets=$(lscpu | grep "Socket(s)" | awk '{print $NF}') && echo "sockets = ${sockets}" || echo "sockets not found"
  local re='^[0-9]+$'
  local run_multicore=""
  if [[ $core =~ $re && $sockets =~ $re ]] ; then
    local n_core=$((core * sockets))
    local run_multicore=" -j ${n_core}"
  fi

  # Distribute Python extensions as wheels on Linux
  echo "[BUILD] Building FBGEMM-GPU wheel (VARIANT=${fbgemm_variant}) ..."
  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} \
    python setup.py "${run_multicore}" bdist_wheel \
      --package_name="${package_name}" \
      --python-tag="${python_tag}" \
      --plat-name="${plat_name}" \
      --verbose \
      "${build_args[@]}"

  # Run checks on the built libraries
  (run_fbgemm_gpu_postbuild_checks "${fbgemm_variant}") || return 1

  echo "[BUILD] Enumerating the built wheels ..."
  print_exec ls -lth dist/*.whl

  echo "[BUILD] Enumerating the wheel SHAs ..."
  print_exec sha1sum dist/*.whl
  print_exec sha256sum dist/*.whl
  print_exec md5sum dist/*.whl

  echo "[BUILD] FBGEMM-GPU build wheel completed"
}

build_fbgemm_gpu_install () {
  env_name="$1"
  fbgemm_variant="$2"
  fbgemm_variant_targets="$3"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME VARIANT [TARGETS]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env cpu                          # CPU-only variant"
    echo "    ${FUNCNAME[0]} build_env cuda                         # CUDA variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env cuda '7.0;8.0'               # CUDA variant for custom target(s)"
    echo "    ${FUNCNAME[0]} build_env rocm                         # ROCm variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env rocm 'gfx906;gfx908;gfx90a'  # ROCm variant for custom target(s)"
    return 1
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Set up and configure the build
  __build_fbgemm_gpu_common_pre_steps || return 1
  __configure_fbgemm_gpu_build "${fbgemm_variant}" "${fbgemm_variant_targets}" || return 1

  echo "################################################################################"
  echo "# Build + Install FBGEMM-GPU Package"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  # Parallelism may need to be limited to prevent the build from being
  # canceled for going over ulimits
  echo "[BUILD] Building + installing FBGEMM-GPU (VARIANT=${fbgemm_variant}) ..."
  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} \
    python setup.py install "${build_args[@]}"

  # Run checks on the built libraries
  (run_fbgemm_gpu_postbuild_checks "${fbgemm_variant}") || return 1

  echo "[INSTALL] Checking imports ..."
  # Exit this directory to prevent import clashing, since there is an
  # fbgemm_gpu/ subdirectory present
  cd - || return 1
  (test_python_import_package "${env_name}" fbgemm_gpu) || return 1
  (test_python_import_symbol "${env_name}" fbgemm_gpu __version__) || return 1
  cd - || return 1

  echo "[BUILD] FBGEMM-GPU build + install completed"
}

build_fbgemm_gpu_develop () {
  env_name="$1"
  fbgemm_variant="$2"
  fbgemm_variant_targets="$3"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME VARIANT [TARGETS]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env cpu                          # CPU-only variant"
    echo "    ${FUNCNAME[0]} build_env cuda                         # CUDA variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env cuda '7.0;8.0'               # CUDA variant for custom target(s)"
    echo "    ${FUNCNAME[0]} build_env rocm                         # ROCm variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env rocm 'gfx906;gfx908;gfx90a'  # ROCm variant for custom target(s)"
    return 1
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Set up and configure the build
  __build_fbgemm_gpu_common_pre_steps || return 1
  __configure_fbgemm_gpu_build "${fbgemm_variant}" "${fbgemm_variant_targets}" || return 1

  echo "################################################################################"
  echo "# Build + Install FBGEMM-GPU Package (Develop)"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  # Parallelism may need to be limited to prevent the build from being
  # canceled for going over ulimits
  echo "[BUILD] Building (develop) FBGEMM-GPU (VARIANT=${fbgemm_variant}) ..."
  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} \
    python setup.py build develop "${build_args[@]}"

  # Run checks on the built libraries
  (run_fbgemm_gpu_postbuild_checks "${fbgemm_variant}") || return 1

  echo "[BUILD] FBGEMM-GPU build + develop completed"
}

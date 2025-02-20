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
  (exec_with_retries 3 git submodule sync) || return 1
  (exec_with_retries 3 git submodule update --init --recursive) || return 1

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[BUILD] Installing other build dependencies ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run --no-capture-output ${env_prefix} python -m pip install -r requirements.txt) || return 1

  (install_triton_pip "${env_name}") || return 1

  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" numpy) || return 1
  # shellcheck disable=SC2086
  (test_python_import_package "${env_name}" skbuild) || return 1

  echo "[BUILD] Successfully ran git submodules update"
}

__configure_fbgemm_gpu_build_clang () {
  echo "[BUILD] Clang is available; configuring for Clang-based build ..."
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
  # shellcheck disable=SC2206
  build_args+=(
    --cxxprefix=${conda_prefix}
  )
}

__configure_fbgemm_gpu_build_nvcc () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[BUILD] Looking up CUDA version ..."
  # shellcheck disable=SC2155,SC2086
  local cxx_path=$(conda run ${env_prefix} which c++)
  # shellcheck disable=SC2155,SC2086
  local cuda_version=$(conda run ${env_prefix} nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
  # shellcheck disable=SC2206
  local cuda_version_arr=(${cuda_version//./ })

  # Only NVCC 12+ supports C++20
  if [[ ${cuda_version_arr[0]} -lt 12 ]]; then
    local cppstd_ver=17
  else
    local cppstd_ver=20
  fi

  if print_exec "conda run ${env_prefix} c++ --version | grep -i clang"; then
    local nvcc_prepend_flags="-std=c++${cppstd_ver} -Xcompiler -std=c++${cppstd_ver} -Xcompiler -stdlib=libstdc++ -ccbin ${cxx_path} -allow-unsupported-compiler"
  else
    # NOTE: The `-stdlib=libstdc++` flag doesn't exist for GCC
    local nvcc_prepend_flags="-std=c++${cppstd_ver} -Xcompiler -std=c++${cppstd_ver} -ccbin ${cxx_path} -allow-unsupported-compiler"
  fi

  # Explicitly set whatever $CONDA_PREFIX/bin/c++ points to as the the host
  # compiler, but set GNU libstdc++ (as opposed to Clang libc++) as the standard
  # library
  #
  # NOTE: There appears to be no ROCm equivalent for NVCC_PREPEND_FLAGS:
  #   https://github.com/ROCm/HIP/issues/931
  #
  echo "[BUILD] Setting NVCC flags ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} NVCC_PREPEND_FLAGS=\"${nvcc_prepend_flags}\"
  # shellcheck disable=SC2086
  print_exec conda run ${env_prefix} printenv NVCC_PREPEND_FLAGS

  echo "[BUILD] Setting CUDA build args ..."
  # shellcheck disable=SC2206
  build_args+=(
    # Override CMake configuration
    -DCMAKE_CXX_STANDARD="${cppstd_ver}"
  )
}

__configure_fbgemm_gpu_cuda_home () {
  if  [[ "$BUILD_CUDA_VERSION" =~ ^12.6.*$ ]] ||
      [[ "$BUILD_CUDA_VERSION" =~ ^12.8.*$ ]]; then
    # shellcheck disable=SC2155,SC2086
    local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
    local new_cuda_home="${conda_prefix}/targets/${MACHINE_NAME_LC}-linux"

    # shellcheck disable=SC2206
    build_args+=(
      # NOTE: The legacy find_package(CUDA) uses CUDA_TOOLKIT_ROOT_DIR
      # while the newer and recomended find_package(CUDAToolkit)
      # uses CUDAToolkit_ROOT.
      #
      # https://github.com/conda-forge/cuda-feedstock/issues/59
      -DCUDA_TOOLKIT_ROOT_DIR="${new_cuda_home}"
      -DCUDAToolkit_ROOT="${new_cuda_home}"
    )
  fi
}

__configure_fbgemm_gpu_build_cpu () {
  # Update the package name and build args depending on if CUDA is specified
  echo "[BUILD] Setting CPU-only build args ..."
  build_args=(
    --package_variant=cpu
  )
}

__configure_fbgemm_gpu_build_docs () {
  # Update the package name and build args depending on if CUDA is specified
  echo "[BUILD] Setting CPU-only (docs) build args ..."
  build_args=(
    --package_variant=docs
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
        #
        # Architecture mapping can be found at:
        #   https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
        local arch_list="gfx908,gfx90a,gfx942"
      fi
    else
      echo "[BUILD] rocminfo not found in PATH!"
    fi
  fi

  echo "[BUILD] Setting the following ROCm targets: ${arch_list}"
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} PYTORCH_ROCM_ARCH="${arch_list}"

  echo "[BUILD] Setting HIPCC verbose mode ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} HIPCC_VERBOSE=1

  # For more info on rocmcc flags:
  #   https://rocm.docs.amd.com/en/docs-6.1.1/reference/rocmcc.html
  echo "[BUILD] Setting ROCm build args ..."
  build_args=(
    --package_variant=rocm
    # HIP_ROOT_DIR now required for HIP to be correctly detected by CMake
    -DHIP_ROOT_DIR=/opt/rocm
    # ROCm CMake complains about missing AMDGPU_TARGETS, so we explicitly set this
    -DAMDGPU_TARGETS="${arch_list}"
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
    # To keep binary sizes to minimum, build only against the CUDA architectures
    # that the latest PyTorch supports:
    #   7.0 (V100), 8.0 (A100), and 9.0,9.0a (H100)
    cuda_version_nvcc=$(conda run -n "${env_name}" nvcc --version)
    echo "[BUILD] Using the default architectures for CUDA $cuda_version_nvcc ..."

    if  [[ $cuda_version_nvcc == *"V12.1"* ]] ||
        [[ $cuda_version_nvcc == *"V12.4"* ]] ||
        [[ $cuda_version_nvcc == *"V12.6"* ]] ||
        [[ $cuda_version_nvcc == *"V12.8"* ]]; then
      # sm_90 and sm_90a are only available for CUDA 12.1+
      # NOTE: CUTLASS kernels for Hopper require sm_90a to be enabled
      # See:
      #   https://github.com/NVIDIA/nvbench/discussions/129
      #   https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt#L187
      #   https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp#L224
      local arch_list="7.0;8.0;9.0;9.0a"
    else
      local arch_list="7.0;8.0"
    fi
  fi
  echo "[BUILD] Setting the following CUDA targets: ${arch_list}"

  # Unset the environment-supplied TORCH_CUDA_ARCH_LIST because it will take
  # precedence over cmake -DTORCH_CUDA_ARCH_LIST
  unset TORCH_CUDA_ARCH_LIST

  echo "[BUILD] Looking up NVML filepath ..."
  # shellcheck disable=SC2155,SC2086
  local nvml_lib_path=$(conda run --no-capture-output ${env_prefix} printenv NVML_LIB_PATH)

  echo "[BUILD] Looking up NCCL filepath ..."
  # shellcheck disable=SC2155,SC2086
  local conda_prefix=$(conda run ${env_prefix} printenv CONDA_PREFIX)
  # shellcheck disable=SC2155,SC2086
  local nccl_lib_path=$(conda run ${env_prefix} find ${conda_prefix} -name "libnccl.so*")

  echo "[BUILD] Setting NVCC verbose mode ..."
  # shellcheck disable=SC2086
  print_exec conda env config vars set ${env_prefix} NVCC_VERBOSE=1

  echo "[BUILD] Setting CUDA build args ..."
  build_args=(
    --package_variant=cuda
    --nvml_lib_path="${nvml_lib_path}"
    --nccl_lib_path="${nccl_lib_path}"
    # Pass to PyTorch CMake
    -DTORCH_CUDA_ARCH_LIST="'${arch_list}'"
  )

  # Explicitly set CUDA_HOME (for CUDA 12.6+)
  __configure_fbgemm_gpu_cuda_home

  # Set NVCC flags
  __configure_fbgemm_gpu_build_nvcc
}

__configure_fbgemm_gpu_build_genai () {
  local fbgemm_variant_targets="$1"

  __configure_fbgemm_gpu_build_cuda "$fbgemm_variant_targets" || return 1

  # Replace the package_variant flag, since GenAI is also a CUDA-type build
  for i in "${!build_args[@]}"; do
    build_args[i]="${build_args[i]/--package_variant=cuda/--package_variant=genai}"
  done
}

# shellcheck disable=SC2120
__configure_fbgemm_gpu_build () {
  echo "################################################################################"
  echo "# Configure FBGEMM-GPU Build"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  if [ "$fbgemm_variant" == "cpu" ]; then
    echo "[BUILD] Configuring build as CPU variant ..."
    __configure_fbgemm_gpu_build_cpu

  elif [ "$fbgemm_variant" == "docs" ]; then
    echo "[BUILD] Configuring build as CPU (docs) variant ..."
    __configure_fbgemm_gpu_build_docs

  elif [ "$fbgemm_variant" == "rocm" ]; then
    echo "[BUILD] Configuring build as ROCm variant ..."
    __configure_fbgemm_gpu_build_rocm "${fbgemm_variant_targets}"

  elif [ "$fbgemm_variant" == "genai" ]; then
    echo "[BUILD] Configuring build as GenAI variant ..."
    __configure_fbgemm_gpu_build_genai "${fbgemm_variant_targets}"

  else
    echo "[BUILD] Configuring build as CUDA variant (this is the default behavior) ..."
    __configure_fbgemm_gpu_build_cuda "${fbgemm_variant_targets}"
  fi

  # shellcheck disable=SC2086
  print_exec conda run ${env_prefix} c++ --version

  # Set other compiler flags as needed
  if print_exec "conda run ${env_prefix} c++ --version | grep -i clang"; then
    __configure_fbgemm_gpu_build_clang
  fi

  # Set verbosity
  build_args+=(
    --verbose
  )

  # Set debugging options
  if [ "$fbgemm_release_channel" != "release" ] || [ "$BUILD_DEBUG" -eq 1 ]; then
    build_args+=(
      --debug
    )
  fi

  # shellcheck disable=SC2145
  echo "[BUILD] FBGEMM_GPU build arguments have been set:  ${build_args[@]}"
}

__build_fbgemm_gpu_set_python_tag () {
  # shellcheck disable=SC2207,SC2086
  local python_version=($(conda run --no-capture-output ${env_prefix} python --version))

  # shellcheck disable=SC2206
  local python_version_arr=(${python_version[1]//./ })

  # Set the python tag (e.g. Python 3.13 --> py313)
  export python_tag="py${python_version_arr[0]}${python_version_arr[1]}"
  echo "[BUILD] Extracted and set Python tag: ${python_tag}"
}

__build_fbgemm_gpu_set_python_plat_name () {
  if [[ $KERN_NAME == 'Darwin' ]]; then
    # This follows PyTorch package naming conventions
    # See https://pypi.org/project/torch/#files
    if [[ $MACHINE_NAME == 'arm64' ]]; then
      export python_plat_name="macosx_11_0_${MACHINE_NAME}"
    else
      export python_plat_name="macosx_10_9_${MACHINE_NAME}"
    fi

  elif [[ $KERN_NAME == 'Linux' ]]; then
    # NOTE: manylinux2014 is the minimum platform tag specified, bc
    # manylinux1 does not support aarch64; see https://github.com/pypa/manylinux
    #
    # As of 2024-12, upstream torch has switched to manylinux_2_28:
    #   https://dev-discuss.pytorch.org/t/pytorch-linux-wheels-switching-to-new-wheel-build-platform-manylinux-2-28-on-november-12-2024/2581
    #   https://github.com/pytorch/pytorch/pull/143423
    export python_plat_name="manylinux_2_28_${MACHINE_NAME}"

  else
    echo "[BUILD] Unsupported OS platform: ${KERN_NAME}"
    return 1
  fi

  echo "[BUILD] Extracted and set Python platform name: ${python_plat_name}"
}

__build_fbgemm_gpu_set_run_multicore () {
  # shellcheck disable=SC2155
  local core=$(lscpu | grep "Core(s)" | awk '{print $NF}') && echo "core = ${core}" || echo "core not found"
  # shellcheck disable=SC2155
  local sockets=$(lscpu | grep "Socket(s)" | awk '{print $NF}') && echo "sockets = ${sockets}" || echo "sockets not found"
  local re='^[0-9]+$'

  export run_multicore=""
  if [[ $core =~ $re && $sockets =~ $re ]] ; then
    local n_core=$((core * sockets))
    export run_multicore="-j ${n_core}"
  fi

  echo "[BUILD] Set multicore run option for setup.py: ${run_multicore}"
}

__build_fbgemm_gpu_common_pre_steps () {
  # Private function that uses variables instantiated by its caller

  # Check C/C++ compilers are visible (the build scripts look specifically for `gcc`)
  (test_binpath "${env_name}" cc) || return 1
  (test_binpath "${env_name}" gcc) || return 1
  (test_binpath "${env_name}" c++) || return 1
  (test_binpath "${env_name}" g++) || return 1

  # Set the default the FBGEMM_GPU variant to be CUDA
  if  [ "$fbgemm_variant" != "cpu" ] &&
      [ "$fbgemm_variant" != "docs" ] &&
      [ "$fbgemm_variant" != "rocm" ] &&
      [ "$fbgemm_variant" != "genai" ]; then
    echo "################################################################################"
    echo "[BUILD] Unknown FBGEMM_GPU variant: $fbgemm_variant"
    echo "[BUILD] Defaulting to CUDA"
    echo "################################################################################"
    export fbgemm_variant="cuda"
  fi

  # Extract and set the Python tag
  __build_fbgemm_gpu_set_python_tag

  # Extract and set the platform name
  __build_fbgemm_gpu_set_python_plat_name

  # Set multicore run option for setup.py if the number of cores on the machine
  # permit for this
  __build_fbgemm_gpu_set_run_multicore

  # Check LD_LIBRARY_PATH for numpy
  echo "[CHECK] LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"

  echo "[BUILD] Running pre-build cleanups ..."
  print_exec rm -rf dist

  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} python setup.py clean

  echo "[BUILD] Printing git status ..."
  print_exec git status
  print_exec git diff
}

__print_library_infos () {
  # shellcheck disable=SC2035,SC2061,SC2062,SC2155,SC2178
  local fbgemm_gpu_so_files=$(find . -name *.so | grep .*cmake-build/.*)
  readarray -t fbgemm_gpu_so_files <<<"$fbgemm_gpu_so_files"

  for library in "${fbgemm_gpu_so_files[@]}"; do
    echo "################################################################################"
    echo "[CHECK] BUILT LIBRARY: ${library}"

    echo "[CHECK] Listing out library size:"
    print_exec "du -h --block-size=1M ${library}"

    print_glibc_info "${library}"

    # shellcheck disable=SC2155
    local symbols_file=$(mktemp --suffix ".symbols.txt")
    print_exec "nm -gDC ${library} > ${symbols_file}"
    # shellcheck disable=SC2086
    echo "[CHECK] Total Number of symbols: $(wc -l ${symbols_file} | awk '{print $1}')"
    # shellcheck disable=SC2086
    echo "[CHECK] Number of fbgemm symbols: $(grep -c fbgemm ${symbols_file})"

    # shellcheck disable=SC2155
    local usymbols_file=$(mktemp --suffix ".usymbols.txt")
    print_exec "nm -gDCu ${library} > ${usymbols_file}"
    # shellcheck disable=SC2086
    echo "[CHECK] Listing out undefined symbols ($(wc -l ${usymbols_file} | awk '{print $1}') total):"
    cat "${usymbols_file}" | sort

    echo "[CHECK] Listing out external shared libraries linked:"
    print_exec ldd "${library}"

    echo "[CHECK] Displaying ELF information:"
    print_exec readelf -d "${library}"
    echo "################################################################################"
    echo ""
    echo ""
  done
}

__verify_library_symbols () {
  __test_one_symbol () {
    local symbol="$1"
    if [ "$symbol" == "" ]; then
      echo "Usage: ${FUNCNAME[0]} SYMBOL"
      echo "Example(s):"
      echo "    ${FUNCNAME[0]} fbgemm_gpu::asynchronous_inclusive_cumsum_cpu"
      return 1
    fi

    # shellcheck disable=SC2035,SC2061,SC2062,SC2155,SC2178
    local fbgemm_gpu_so_files=$(find . -name *.so | grep .*cmake-build/.*)
    readarray -t fbgemm_gpu_so_files <<<"$fbgemm_gpu_so_files"

    # Iterate through the built .SO files to check for the symbol's existence
    for library in "${fbgemm_gpu_so_files[@]}"; do
      if test_library_symbol "${library}" "${symbol}"; then
        return 0
      fi
    done

    return 1
  }

  # Prepare a sample set of symbols whose existence in the built library should be checked
  # This is by no means an exhaustive set, and should be updated accordingly
  if  [ "${fbgemm_variant}" == "cpu" ] ||
      [ "${fbgemm_variant}" == "docs" ]; then
    local lib_symbols_to_check=(
      fbgemm_gpu::asynchronous_inclusive_cumsum_cpu
      fbgemm_gpu::jagged_2d_to_dense
    )
  elif [ "${fbgemm_variant}" == "cuda" ]; then
    local lib_symbols_to_check=(
      fbgemm_gpu::asynchronous_inclusive_cumsum_cpu
      fbgemm_gpu::jagged_2d_to_dense
      fbgemm_gpu::asynchronous_inclusive_cumsum_gpu
      fbgemm_gpu::merge_pooled_embeddings
    )
  elif [ "${fbgemm_variant}" == "rocm" ]; then
    local lib_symbols_to_check=(
      fbgemm_gpu::asynchronous_inclusive_cumsum_cpu
      fbgemm_gpu::jagged_2d_to_dense
      fbgemm_gpu::asynchronous_inclusive_cumsum_gpu
      fbgemm_gpu::merge_pooled_embeddings
    )
  elif [ "${fbgemm_variant}" == "genai" ]; then
    local lib_symbols_to_check=(
      fbgemm_gpu::car_init
      fbgemm_gpu::per_tensor_quantize_i8
    )
  fi

  echo "[CHECK] Verifying sample subset of symbols in the built libraries ..."
  for symbol in "${lib_symbols_to_check[@]}"; do
    (__test_one_symbol "${symbol}") || return 1
  done
}

run_fbgemm_gpu_postbuild_checks () {
  fbgemm_variant="$1"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} FBGEMM_VARIANT"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} cpu"
    echo "    ${FUNCNAME[0]} docs"
    echo "    ${FUNCNAME[0]} cuda"
    echo "    ${FUNCNAME[0]} rocm"
    echo "    ${FUNCNAME[0]} genai"
    return 1
  fi

  # Find the .SO file
  # shellcheck disable=SC2035,SC2061,SC2062,SC2155,SC2178
  local fbgemm_gpu_so_files=$(find . -name *.so | grep .*cmake-build/.*)
  readarray -t fbgemm_gpu_so_files <<<"$fbgemm_gpu_so_files"
  if [ "${#fbgemm_gpu_so_files[@]}" -le 0 ]; then
    echo "[CHECK] .SO library is missing from the build path!"
    return 1
  fi

  __print_library_infos     || return 1
  __verify_library_symbols  || return 1
}

run_fbgemm_gpu_audit_wheel () {
  fbgemm_wheel="$1"
  if [ "$fbgemm_wheel" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} FBGEMM_WHEEL_PATH"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} dist/fbgemm_gpu_nightly_cpu-2024.12.20-cp39-cp39-manylinux_2_28_x86_64.whl"
    return 1
  fi

  echo "################################################################################"
  echo "[BUILD] Wheel Audit: ${fbgemm_wheel}"
  echo ""

  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} auditwheel show "${fbgemm_wheel}"
  echo ""
  echo "################################################################################"
}

################################################################################
# FBGEMM_GPU Build Functions
################################################################################

build_fbgemm_gpu_package () {
  env_name="$1"
  fbgemm_release_channel="$2"
  fbgemm_variant="$3"
  fbgemm_variant_targets="$4"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME RELEASE_CHANNEL VARIANT [VARIANT_TARGETS]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env release cpu                      # CPU-only variant"
    echo "    ${FUNCNAME[0]} build_env release docs                     # CPU-only (docs) variant"
    echo "    ${FUNCNAME[0]} build_env nightly cuda                     # CUDA variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env test cuda '7.0;8.0'              # CUDA variant for custom target(s)"
    echo "    ${FUNCNAME[0]} build_env test rocm                        # ROCm variant for default target(s)"
    echo "    ${FUNCNAME[0]} build_env test rocm 'gfx906;gfx908;gfx90a' # ROCm variant for custom target(s)"
    return 1
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Set up and configure the build
  __build_fbgemm_gpu_common_pre_steps || return 1
  __configure_fbgemm_gpu_build        || return 1

  echo "################################################################################"
  echo "# Build FBGEMM-GPU Package (Wheel)"
  echo "#"
  echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
  echo "################################################################################"
  echo ""

  # Set packaging options
  build_args+=(
    --package_channel="${fbgemm_release_channel}"
    --python-tag="${python_tag}"
    --plat-name="${python_plat_name}"
  )

  # Prepend build options correctly for `python -m build`
  # https://build.pypa.io/en/stable/index.html
  # https://gregoryszorc.com/blog/2023/10/30/my-user-experience-porting-off-setup.py/
  for i in "${!build_args[@]}"; do
    build_args[i]="--config-setting=--build-option=${build_args[i]}"
  done

  # Build the wheel.  Invoke using `python -m build`
  #   https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html
  echo "[BUILD] Building FBGEMM-GPU wheel (VARIANT=${fbgemm_variant}) ..."
  # shellcheck disable=SC2086
  print_exec conda run --no-capture-output ${env_prefix} \
    python -m build --wheel --no-isolation \
      "${build_args[@]}" || return 1

  # Run checks on the built libraries
  (run_fbgemm_gpu_postbuild_checks "${fbgemm_variant}") || return 1

  for wheelfile in dist/*.whl; do
    run_fbgemm_gpu_audit_wheel "${wheelfile}"
  done

  echo "[BUILD] Enumerating the built wheels ..."
  print_exec ls -lth dist/*.whl || return 1

  echo "[BUILD] Enumerating the wheel SHAs ..."
  print_exec sha1sum dist/*.whl || return 1
  print_exec sha256sum dist/*.whl || return 1
  print_exec md5sum dist/*.whl || return 1

  echo "[BUILD] FBGEMM-GPU build + package completed"
}

build_fbgemm_gpu_install () {
  env_name="$1"
  fbgemm_variant="$2"
  fbgemm_variant_targets="$3"
  if [ "$fbgemm_variant" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME VARIANT [TARGETS]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env cpu                          # CPU-only variant"
    echo "    ${FUNCNAME[0]} build_env docs                         # CPU-only (docs) variant"
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
  __configure_fbgemm_gpu_build        || return 1

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
    python setup.py "${run_multicore}" install \
      "${build_args[@]}" || return 1

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

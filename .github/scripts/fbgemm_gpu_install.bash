#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pip.bash"

################################################################################
# FBGEMM_GPU Install Functions
################################################################################

__install_print_dependencies_info () {
  # shellcheck disable=SC2086,SC2155
  local installed_pytorch_version=$(conda run ${env_prefix} python -c "import torch; print(torch.__version__)")
  # shellcheck disable=SC2086,SC2155
  local installed_cuda_version=$(conda run ${env_prefix} python -c "import torch; print(torch.version.cuda)")
  echo "################################################################################"
  echo "[CHECK] !!!!    INFO    !!!!"
  echo "[CHECK] The installed version of PyTorch is: ${installed_pytorch_version}"
  echo "[CHECK] CUDA version reported by PyTorch is: ${installed_cuda_version}"
  echo "[CHECK]"
  echo "[CHECK] NOTE: If the PyTorch package channel is different from the FBGEMM_GPU"
  echo "[CHECK]       package channel; the package may be broken at runtime!!!"
  echo "################################################################################"
  echo ""
}

__install_fetch_version_and_variant_info () {
  echo "[INSTALL] Checking imports and symbols ..."
  (test_python_import_package "${env_name}" fbgemm_gpu) || return 1
  (test_python_import_symbol "${env_name}" fbgemm_gpu __version__) || return 1
  (test_python_import_symbol "${env_name}" fbgemm_gpu __variant__) || return 1

  echo "[CHECK] Printing out the FBGEMM-GPU version ..."
  # shellcheck disable=SC2086,SC2155
  installed_fbgemm_gpu_version=$(conda run ${env_prefix} python -c "import fbgemm_gpu; print(fbgemm_gpu.__version__)")
  # shellcheck disable=SC2086,SC2155
  installed_fbgemm_gpu_variant=$(conda run ${env_prefix} python -c "import fbgemm_gpu; print(fbgemm_gpu.__variant__)")
  echo "################################################################################"
  echo "[CHECK] The installed VERSION of FBGEMM_GPU is: ${installed_fbgemm_gpu_version}"
  echo "[CHECK] The installed VARIANT of FBGEMM_GPU is: ${installed_fbgemm_gpu_variant}"
  echo "################################################################################"
  echo ""
}

__install_check_subpackages () {
  # shellcheck disable=SC2086,SC2155
  local fbgemm_gpu_packages=$(conda run ${env_prefix} python -c "import fbgemm_gpu; print(dir(fbgemm_gpu))")

  if [ "$installed_fbgemm_gpu_variant" == "cuda" ] || [ "$installed_fbgemm_gpu_variant" == "genai" ]; then
    # shellcheck disable=SC2086,SC2155
    local experimental_packages=$(conda run ${env_prefix} python -c "import fbgemm_gpu.experimental; print(dir(fbgemm_gpu.experimental))")
  fi

  echo "################################################################################"
  echo "[CHECK] FBGEMM_GPU Experimental Packages"
  echo "[CHECK] fbgemm_gpu: ${fbgemm_gpu_packages}"
  echo "[CHECK] fbgemm_gpu.experimental: ${experimental_packages}"
  echo "################################################################################"
  echo ""


  echo "[INSTALL] Check for installation of Python sources ..."
  local subpackages=(
    "fbgemm_gpu.config"
    "fbgemm_gpu.docs"
    "fbgemm_gpu.quantize"
    "fbgemm_gpu.tbe.cache"
  )

  if [ "$installed_fbgemm_gpu_variant" != "genai" ]; then
    subpackages+=(
      "fbgemm_gpu.split_embedding_codegen_lookup_invokers"
      "fbgemm_gpu.tbe.ssd"
      "fbgemm_gpu.tbe.utils"
    )
  fi

  for package in "${subpackages[@]}"; do
    (test_python_import_package "${env_name}" "${package}") || return 1
  done
}

__install_check_operator_registrations () {
  echo "[INSTALL] Check for operator registrations ..."
  if [ "$installed_fbgemm_gpu_variant" == "genai" ]; then
    local test_operators=(
      "torch.ops.fbgemm.nccl_init"
      "torch.ops.fbgemm.gqa_attn_splitk"
      "torch.ops.fbgemm.rope_qkv_decoding"
    )
  else
    local test_operators=(
      "torch.ops.fbgemm.asynchronous_inclusive_cumsum"
      "torch.ops.fbgemm.split_embedding_codegen_lookup_sgd_function_pt2"
    )
  fi

  for operator in "${test_operators[@]}"; do
    # shellcheck disable=SC2086
    if conda run ${env_prefix} python -c "import torch; import fbgemm_gpu; print($operator)"; then
      echo "[CHECK] FBGEMM_GPU operator appears to be correctly registered: $operator"
    else
      echo "################################################################################"
      echo "[CHECK] FBGEMM_GPU operator hasn't been registered on torch.ops.load():"
      echo "[CHECK]"
      echo "[CHECK] $operator"
      echo "[CHECK]"
      echo "[CHECK] Please check that all operators defined with m.def() have an appropriate"
      echo "[CHECK] m.impl() defined, AND that the definition sources are included in the "
      echo "[CHECK] CMake build configuration!"
      echo "################################################################################"
      echo ""
      return 1
    fi
  done
}

__fbgemm_gpu_post_install_checks () {
  local env_name="$1"
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Print PyTorch and CUDA versions for sanity check
  __install_print_dependencies_info         || return 1

  # Fetch the version and variant info from the package
  __install_fetch_version_and_variant_info  || return 1

  # Check FBGEMM_GPU subpackages are installed correctly
  __install_check_subpackages               || return 1

  # Check operator registrations are working
  __install_check_operator_registrations    || return 1
}

install_fbgemm_gpu_wheel () {
  local env_name="$1"
  local wheel_path="$2"
  if [ "$wheel_path" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME WHEEL_NAME"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu.whl     # Install the package (wheel)"
    return 1
  else
    echo "################################################################################"
    echo "# Install FBGEMM-GPU from Wheel"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  echo "[INSTALL] Printing out FBGEMM-GPU wheel SHA: ${wheel_path}"
  print_exec sha1sum "${wheel_path}"
  print_exec sha256sum "${wheel_path}"
  print_exec md5sum "${wheel_path}"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Installing FBGEMM-GPU wheel: ${wheel_path} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} python -m pip install "${wheel_path}") || return 1

  __fbgemm_gpu_post_install_checks "${env_name}" || return 1

  echo "[INSTALL] FBGEMM-GPU installation through wheel completed ..."
}

install_fbgemm_gpu_pip () {
  local env_name="$1"
  local fbgemm_gpu_channel_version="$2"
  local fbgemm_gpu_variant_type_version="$3"
  if [ "$fbgemm_gpu_variant_type_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME FBGEMM_GPU_CHANNEL[/VERSION] FBGEMM_GPU_VARIANT_TYPE[/VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env 0.8.0 cpu                  # Install the CPU variant, specific version from release channel"
    echo "    ${FUNCNAME[0]} build_env release cuda/12.6.3        # Install the CUDA 12.3 variant, latest version from release channel"
    echo "    ${FUNCNAME[0]} build_env test/0.8.0 cuda/12.6.3     # Install the CUDA 12.3 variant, specific version from test channel"
    echo "    ${FUNCNAME[0]} build_env nightly rocm/6.2           # Install the ROCM 6.2 variant, latest version from nightly channel"
    return 1
  else
    echo "################################################################################"
    echo "# Install FBGEMM-GPU Package from PIP"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # Install the package from PyTorch PIP (not PyPI)
  # The package's canonical name is 'fbgemm-gpu' (hyphen, not underscore)
  install_from_pytorch_pip "${env_name}" fbgemm_gpu "${fbgemm_gpu_channel_version}" "${fbgemm_gpu_variant_type_version}" || return 1

  # Run post-installation checks
  __fbgemm_gpu_post_install_checks "${env_name}" || return 1

  echo "[INSTALL] Successfully installed FBGEMM-GPU through PyTorch PIP"
}

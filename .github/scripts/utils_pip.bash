#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"

################################################################################
# PyTorch PIP Install Functions
################################################################################

__extract_pip_arguments () {
  export env_name="$1"
  export package_name_raw="$2"
  export package_version="$3"
  export package_variant_type="$4"
  export package_variant_version="$5"
  if [ "$package_variant_type" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME PACKAGE_VERSION PACKAGE_VARIANT_TYPE [PACKAGE_VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env torch 1.11.0 cpu             # Install the CPU variant a specific version"
    echo "    ${FUNCNAME[0]} build_env torch release cpu            # Install the CPU variant of the latest stable version"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu test cuda 12.1.0  # Install the variant for CUDA 12.1"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu nightly rocm 5.3  # Install the variant for ROCM 5.3"
    return 1
  else
    echo "################################################################################"
    echo "# Install ${package_name_raw} (PyTorch PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # Replace underscores with hyphens to materialize the canonical name of the package
  # shellcheck disable=SC2155
  export package_name=$(echo "${package_name_raw}" | tr '_' '-')

  # Set the package variant
  if [ "$package_variant_type" == "cuda" ]; then
    # Extract the CUDA version or default to 11.8.0
    local cuda_version="${package_variant_version:-11.8.0}"
    # shellcheck disable=SC2206
    local cuda_version_arr=(${cuda_version//./ })
    # Convert, i.e. cuda 12.1.0 => cu121
    export package_variant="cu${cuda_version_arr[0]}${cuda_version_arr[1]}"
  elif [ "$package_variant_type" == "rocm" ]; then
    # Extract the ROCM version or default to 5.5.1
    local rocm_version="${package_variant_version:-5.5.1}"
    # shellcheck disable=SC2206
    local rocm_version_arr=(${rocm_version//./ })
    # Convert, i.e. rocm 5.5.1 => rocm5.5
    export package_variant="rocm${rocm_version_arr[0]}.${rocm_version_arr[1]}"
  else
    export package_variant_type="cpu"
    export package_variant="cpu"
  fi
  echo "[INSTALL] Extracted package variant: ${package_variant}"

  # Set the package name and installation channel
  if [ "$package_version" == "nightly" ] || [ "$package_version" == "test" ]; then
    export pip_package="--pre ${package_name}"
    export pip_channel="https://download.pytorch.org/whl/${package_version}/${package_variant}/"
  elif [ "$package_version" == "release" ]; then
    export pip_package="${package_name}"
    export pip_channel="https://download.pytorch.org/whl/${package_variant}/"
  else
    export pip_package="${package_name}==${package_version}+${package_variant}"
    export pip_channel="https://download.pytorch.org/whl/${package_variant}/"
  fi
}

install_from_pytorch_pip () {
  local env_name="$1"
  local package_name_raw="$2"
  local package_version="$3"
  local package_variant_type="$4"
  local package_variant_version="$5"
  if [ "$package_variant_type" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME PACKAGE_VERSION PACKAGE_VARIANT_TYPE [PACKAGE_VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env torch 1.11.0 cpu             # Install the CPU variant for a specific version"
    echo "    ${FUNCNAME[0]} build_env torch release cpu            # Install the CPU variant, latest release version"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu test cuda 12.1.0  # Install the CUDA 12.1 variant, latest test version"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu nightly rocm 5.3  # Install the ROCM 5.3 variant, latest nightly version"
    return 1
  else
    echo "################################################################################"
    echo "# Install ${package_name_raw} (PyTorch PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  __extract_pip_arguments "$env_name" "$package_name_raw" "$package_version" "$package_variant_type" "$package_variant_version"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Attempting to install [${package_name}, ${package_version}+${package_variant}] from PyTorch PIP using channel ${pip_channel} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} pip install ${pip_package} --extra-index-url ${pip_channel}) || return 1

  # Check only applies to non-CPU variants
  if [ "$package_variant_type" != "cpu" ]; then
    # Ensure that the package build is of the correct variant
    # This test usually applies to the nightly builds
    # shellcheck disable=SC2086
    if conda run ${env_prefix} pip list "${package_name}" | grep "${package_name}" | grep "${package_variant}"; then
      echo "[CHECK] The installed package [${package_name}, ${package_version}] is the correct variant (${package_variant})"
    else
      echo "[CHECK] The installed package [${package_name}, ${package_version}] appears to be an incorrect variant as it is missing references to ${package_variant}!"
      echo "[CHECK] This can happen if the variant of the package (e.g. GPU, nightly) for the MAJOR.MINOR version of CUDA or ROCm presently installed on the system is not available."
      return 1
    fi
  fi
}

################################################################################
# PyTorch PIP Download Functions
################################################################################

download_from_pytorch_pip () {
  local env_name="$1"
  local package_name_raw="$2"
  local package_version="$3"
  local package_variant_type="$4"
  local package_variant_version="$5"
  if [ "$package_variant_type" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME PACKAGE_VERSION PACKAGE_VARIANT_TYPE [PACKAGE_VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env torch 1.11.0 cpu             # Download the CPU variant for a specific version"
    echo "    ${FUNCNAME[0]} build_env torch release cpu            # Download the CPU variant, latest stable version"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu test cuda 12.1.0  # Download the CUDA 12.1 variant, latest test version"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu nightly rocm 5.3  # Download the ROCM 5.3 variant, latest nightly version"
    return 1
  else
    echo "################################################################################"
    echo "# Download ${package_name_raw} (PyTorch PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  __extract_pip_arguments "$env_name" "$package_name_raw" "$package_version" "$package_variant_type" "$package_variant_version"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[DOWNLOAD] Removing previously downloaded wheels from current directory ..."
  # shellcheck disable=SC2035
  rm -rf *.whl || return 1

  echo "[DOWNLOAD] Attempting to download wheel [${package_name}, ${package_version}+${package_variant}] from PyTorch PIP using channel ${pip_channel} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} pip download ${pip_package} --extra-index-url ${pip_channel}) || return 1

  # Ensure that the package build is of the correct variant
  # This test usually applies to the nightly builds
  # shellcheck disable=SC2010
  if ls -la . | grep "${package_name}-"; then
    echo "[CHECK] Successfully downloaded the wheel."
  else
    echo "[CHECK] The wheel was not found!"
    return 1
  fi
}

################################################################################
# PyPI Publish Functions
################################################################################

publish_to_pypi () {
  local env_name="$1"
  local package_name="$2"
  local pypi_token="$3"
  if [ "$pypi_token" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME PYPI_TOKEN"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu_nightly-*.whl MY_TOKEN"
    echo ""
    echo "PYPI_TOKEN is missing!"
    return 1
  else
    echo "################################################################################"
    echo "# Publish to PyPI"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Installing twine ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -y twine) || return 1
  (test_python_import_package "${env_name}" twine) || return 1
  (test_python_import_package "${env_name}" OpenSSL) || return 1

  echo "[PUBLISH] Uploading package(s) to PyPI: ${package_name} ..."
  # shellcheck disable=SC2086
  conda run ${env_prefix} \
    python -m twine upload \
      --username __token__ \
      --password "${pypi_token}" \
      --skip-existing \
      --verbose \
      "${package_name}"

  echo "[PUBLISH] Successfully published package(s) to PyPI: ${package_name}"
  echo "[PUBLISH] NOTE: The publish command is a successful no-op if the wheel version already existed in PyPI; please double check!"
}

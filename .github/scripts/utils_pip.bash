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

__export_package_channel_info () {
  local package_channel_version="$1"

  # Extract the package channel and version from the tuple-string
  if [ "$package_channel_version" == "nightly" ] || [ "$package_channel_version" == "test" ] || [ "$package_channel_version" == "release" ]; then
    export package_channel="$package_channel_version"
    export package_version=""
  else
    # shellcheck disable=SC2207
    local package_channel_version_arr=($(echo "${package_channel_version}" | tr '/' '\n'))
    if [ ${#package_channel_version_arr[@]} -lt 2 ]; then
      export package_channel="release"
      export package_version="${package_channel_version_arr[0]}"
    else
      export package_channel="${package_channel_version_arr[0]}"
      export package_version="${package_channel_version_arr[1]}"
    fi
  fi
  if [ "$package_channel" != "nightly" ] && [ "$package_channel" != "test" ] && [ "$package_channel" != "release" ]; then
    echo "[INSTALL] Invalid PyTorch PIP package channel: ${package_channel}"
    return 1
  fi
  echo "[INSTALL] Extracted package (channel, version): (${package_channel}, ${package_version:-LATEST})"
}

__export_package_variant_info () {
  local package_variant_type_version="$1"

  local FALLBACK_VERSION_CUDA="12.4.1"
  local FALLBACK_VERSION_ROCM="6.2.4"

  if [ "$package_variant_type_version" == "cuda" ]; then
    # If "cuda", default to latest CUDA
    local variant_type="cu"
    local variant_version="$FALLBACK_VERSION_CUDA"

  elif [ "$package_variant_type_version" == "rocm" ]; then
    # If "rocm", default to latest ROCm
    local variant_type="rocm"
    local variant_version="$FALLBACK_VERSION_ROCM"

  elif [ "$package_variant_type_version" == "cpu" ]; then
    # If "cpu", default to latest cpu
    local variant_type="cpu"
    local variant_version=""

  else
    # Split along '/', e.g. cuda/12.4.0
    # shellcheck disable=SC2207
    local package_variant_type_version_arr=($(echo "${package_variant_type_version}" | tr '/' '\n'))
    local variant_type="${package_variant_type_version_arr[0]}"
    local variant_version="${package_variant_type_version_arr[1]}"

    if [ "$variant_type" == "cuda" ]; then
      # Extract the CUDA version or set to default
      local cuda_version="${variant_version:-${FALLBACK_VERSION_CUDA}}"
      # shellcheck disable=SC2206
      local cuda_version_arr=(${cuda_version//./ })
      # Convert to package variant label,
      # e.g. cuda 12.4.0 => cu124
      local variant_type="cu"
      local variant_version="${cuda_version_arr[0]}${cuda_version_arr[1]}"

    elif [ "$variant_type" == "rocm" ]; then
      # Extract the ROCM version or set to default
      local rocm_version="${variant_version:-${FALLBACK_VERSION_ROCM}}"
      # shellcheck disable=SC2206
      local rocm_version_arr=(${rocm_version//./ })
      # Convert to package variant label,
      # e.g. rocm 6.2.4 => rocm6.2.4
      #
      # NOTE: Unlike CUDA-based releases, which ignores the minor patch version,
      # ROCm-based releases may use the full version string.
      # See https://download.pytorch.org/whl/nightly/torch/ for examples.
      local variant_type="rocm"
      if [ "${rocm_version_arr[2]}" == "" ]; then
        local variant_version="${rocm_version_arr[0]}.${rocm_version_arr[1]}"
      else
        local variant_version="${rocm_version_arr[0]}.${rocm_version_arr[1]}.${rocm_version_arr[2]}"
      fi

    else
      echo "[INSTALL] Package variant type '$variant_type' is neither CUDA nor ROCm variant, falling back to cpu"
      local variant_type="cpu"
      local variant_version=""
    fi
  fi

  # Export the extracted information
  export package_variant_type="${variant_type}"
  export package_variant="${variant_type}${variant_version}"
  echo "[INSTALL] Extracted package variant: ${package_variant}"
}

__export_pip_arguments () {
  local include_variant="$1"

  # shellcheck disable=SC2155
  local postfix=$([ "$include_variant" = true ] && echo "${package_variant}/" || echo "")

  # Extract the PIP channel
  if [ "$package_channel" == "release" ]; then
    export pip_channel="https://download.pytorch.org/whl/${postfix}"
  else
    echo "[INSTALL] Using a non-RELEASE channel: ${package_channel} ..."
    export pip_channel="https://download.pytorch.org/whl/${package_channel}/${postfix}"
  fi
  echo "[INSTALL] Extracted the full PIP channel: ${pip_channel}"

  # Extract the full PIP package
  # If the channel is non-release, then prepend with `--pre``
  if [ "$package_channel" != "release" ]; then
    export pip_package="--pre ${package_name}"
  else
    export pip_package="${package_name}"
  fi

  # shellcheck disable=SC2155
  local postfix=$([ "$include_variant" = true ] && echo "+${package_variant}" || echo "")

  # If a specific version is specified, then append with `==<version>`
  if [ "$package_version" != "" ]; then
    export pip_package="${pip_package}==${package_version}${postfix}"
  fi
  echo "[INSTALL] Extracted the full PIP package: ${pip_package}"
}

__prepare_pip_arguments () {
  local package_name_raw="$1"
  local package_channel_version="$2"
  local package_variant_type_version="$3"
  if [ "$package_channel_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} PACKAGE_NAME PACKAGE_CHANNEL[/VERSION] [PACKAGE_VARIANT_TYPE[/VARIANT_VERSION]]"
    return 1
  else
    echo "################################################################################"
    echo "# Prepare PIP Arguments (PyTorch PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # Replace underscores with hyphens to materialize the canonical name of the
  # package, and export variable to environment
  # shellcheck disable=SC2155
  export package_name=$(echo "${package_name_raw}" | tr '_' '-')

  # Extract the package channel and package version from the tuple-string, and
  # export variables to environment
  __export_package_channel_info "$package_channel_version"

  if [ "$package_variant_type_version" != "" ]; then
    # Extract the package variant type and variant version from the tuple-string,
    # and export variables to environment
    __export_package_variant_info "${package_variant_type_version}"
  fi

  # With all package_* variables exported, extract the arguments for PIP, and
  # export variabels to environment
  __export_pip_arguments "$([ "$package_variant_type_version" != "" ] && echo "true" || echo "false")"
}

__check_package_variant () {
  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  # Check applies to installation of packages with variants, and only to non-CPU variants
  if [ "$package_variant_type_version" != "" ] && [ "$package_variant_type" != "cpu" ]; then
    # Ensure that the package build is of the correct variant
    # This test usually applies to the nightly builds
    # shellcheck disable=SC2086
    if conda run ${env_prefix} pip list | grep "${package_name_raw}" | grep "${package_variant}"; then
      local check_passed=1
    elif conda run ${env_prefix} pip list | grep "${package_name}" | grep "${package_variant}"; then
      local check_passed=1
    else
      local check_passed=0
    fi

    if [ $check_passed -eq 1 ]; then
      echo "[CHECK] The installed package [${package_name}, ${package_channel}/${package_version:-LATEST}] is the correct variant (${package_variant})"
      return 0
    else
      echo "[CHECK] The installed package [${package_name}, ${package_channel}/${package_version:-LATEST}] appears to be an incorrect variant as it is missing references to ${package_variant}!"
      echo "[CHECK] This can happen if the variant of the package (e.g. GPU, nightly) for the MAJOR.MINOR version of CUDA or ROCm presently installed on the system is not available."
      return 1
    fi
  fi
}

install_from_pytorch_pip () {
  env_name="$1"
  package_name_raw="$2"
  package_channel_version="$3"
  package_variant_type_version="$4"
  if [ "$package_channel_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME PACKAGE_CHANNEL[/VERSION] [PACKAGE_VARIANT_TYPE[/VARIANT_VERSION]]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env torch 1.11.0 cpu                       # Install the CPU variant, specific version from release channel"
    echo "    ${FUNCNAME[0]} build_env torch release cpu                      # Install the CPU variant, latest version from release channel"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu test/0.8.0 cuda/12.4.0      # Install the CUDA 12.4 variant, specific version from test channel"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu nightly rocm/6.2            # Install the ROCM 6.2 variant, latest version from nightly channel"
    echo "    ${FUNCNAME[0]} build_env pytorch_triton 1.11.0                  # Install specific version from release channel"
    echo "    ${FUNCNAME[0]} build_env pytorch_triton release                 # Install latest version from release channel"
    echo "    ${FUNCNAME[0]} build_env pytorch_triton test/0.8.0              # Install specific version from test channel"
    echo "    ${FUNCNAME[0]} build_env pytorch_triton_rocm nightly            # Install latest version from nightly channel"
    return 1
  else
    echo "################################################################################"
    echo "# Install Package From PyTorch PIP: ${package_name_raw}"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  test_network_connection || return 1

  __prepare_pip_arguments "$package_name_raw" "$package_channel_version" "$package_variant_type_version"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Attempting to install [${package_name}, ${package_version:-LATEST}] from PyTorch PIP using channel ${pip_channel} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} pip install ${pip_package} --index-url ${pip_channel}) || return 1

  # Ensure that the correct package variant has been installed
  __check_package_variant || return 1
}

################################################################################
# PyTorch PIP Download Functions
################################################################################

download_from_pytorch_pip () {
  local env_name="$1"
  local package_name_raw="$2"
  local package_channel_version="$3"
  local package_variant_type_version="$4"
  if [ "$package_variant_type_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME PACKAGE_CHANNEL[/VERSION] PACKAGE_VARIANT_TYPE[/VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env torch 1.11.0 cpu                       # Download the CPU variant, specific version from release channel"
    echo "    ${FUNCNAME[0]} build_env torch release cpu                      # Download the CPU variant, latest version from release channel"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu test/0.8.0 cuda/12.4.0      # Download the CUDA 12.4 variant, specific version from test channel"
    echo "    ${FUNCNAME[0]} build_env fbgemm_gpu nightly rocm/6.2            # Download the ROCM 6.2 variant, latest version from nightly channel"
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

  __prepare_pip_arguments "$package_name_raw" "$package_channel_version" "$package_variant_type_version"

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[DOWNLOAD] Removing previously downloaded wheels from current directory ..."
  # shellcheck disable=SC2035
  rm -rf *.whl || return 1

  echo "[DOWNLOAD] Attempting to download wheel [${package_name}, ${package_version:-LATEST}] from PyTorch PIP using channel ${pip_channel} ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda run ${env_prefix} pip download ${pip_package} --index-url ${pip_channel}) || return 1

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
  local pypi_token="$2"
  local package_filepath="$3"
  if [ "$pypi_token" == "" ] || [ "$package_filepath" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PACKAGE_NAME PYPI_TOKEN"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env MY_TOKEN fbgemm_gpu_nightly-*.whl"
    echo ""
    echo "Either PYPI_TOKEN and/or package filepath is missing!"
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
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge --override-channels -y \
    twine) || return 1
  (test_python_import_package "${env_name}" twine) || return 1
  (test_python_import_package "${env_name}" OpenSSL) || return 1

  echo "[PUBLISH] Uploading package(s) to PyPI: ${package_filepath} ..."
  # shellcheck disable=SC2086
  conda run ${env_prefix} \
    python -m twine upload \
      --username __token__ \
      --password "${pypi_token}" \
      --skip-existing \
      --verbose \
      "${package_filepath}"

  echo "[PUBLISH] Successfully published package(s) to PyPI: ${package_filepath}"
  echo "[PUBLISH] NOTE: The publish command is a successful no-op if the wheel version already existed in PyPI; please double check!"
}

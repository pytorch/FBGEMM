#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_base.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_conda.bash"
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pip.bash"

################################################################################
# TorchRec Setup Functions
################################################################################

install_torchrec_pip () {
  local env_name="$1"
  local torchrec_channel_version="$2"
  local torchrec_variant_type_version="$3"
  if [ "$torchrec_variant_type_version" == "" ]; then
    echo "Usage: ${FUNCNAME[0]} ENV_NAME PYTORCH_CHANNEL[/VERSION] PYTORCH_VARIANT_TYPE[/VARIANT_VERSION]"
    echo "Example(s):"
    echo "    ${FUNCNAME[0]} build_env test/2.1.0rc0 cpu                      # Install the CPU variant for a specific version"
    echo "    ${FUNCNAME[0]} build_env release cpu                            # Install the CPU variant, latest release version"
    echo "    ${FUNCNAME[0]} build_env nightly/0.9.0.dev20240716 cuda/12.4.0  # Install the CUDA 12.4 variant, nightly version"
    return 1
  else
    echo "################################################################################"
    echo "# Install TorchRec (PIP)"
    echo "#"
    echo "# [$(date --utc +%FT%T.%3NZ)] + ${FUNCNAME[0]} ${*}"
    echo "################################################################################"
    echo ""
  fi

  # shellcheck disable=SC2155
  local env_prefix=$(env_name_or_prefix "${env_name}")

  echo "[INSTALL] Installing TorchRec dependencies ..."
  # shellcheck disable=SC2086
  (exec_with_retries 3 conda install ${env_prefix} -c conda-forge --override-channels -y \
    iopath \
    lightning-utilities \
    pyre-extensions) || return 1

  # Install the package from TorchRec PIP (not PyPI)
  install_from_pytorch_pip "${env_name}" torchrec "${torchrec_channel_version}" "${torchrec_variant_type_version}" || return 1

  # Exit this directory to prevent import clashing, since there might be an
  # fbgemm_gpu/ subdirectory present
  cd - || return 1

  # Check that TorchRec is importable
  (test_python_import_package "${env_name}" torchrec) || return 1

  # Print out the actual installed TorchRec version
  # shellcheck disable=SC2086,SC2155
  local installed_torchrec_version=$(conda run ${env_prefix} python -c "import torchrec; print(torchrec.__version__)")
  echo "[CHECK] NOTE: The installed version is: ${installed_torchrec_version}"

  cd - || return 1
  echo "[INSTALL] Successfully installed TorchRec through PyTorch PIP"
}

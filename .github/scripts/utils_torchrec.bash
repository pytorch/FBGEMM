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
# shellcheck disable=SC1091,SC2128
. "$( dirname -- "$BASH_SOURCE"; )/utils_pytorch.bash"

################################################################################
# TorchRec Setup Functions
################################################################################

install_torchrec_cpu_pytorch () {
  local env_name="$1"
  local channel_version="$2"
  local python_version="$3"
  local host_arch="$4"
  local torch_version=""
  local env_prefix
  env_prefix=$(env_name_or_prefix "$env_name") || return 1

  if [[ -z "$channel_version" && "$host_arch" == "x86" &&
        "$python_version" =~ ^3\.(11|12|13|14)$ ]]; then
    torch_version="2.16.0.dev20261109+cpu"
    local python_tag="cp${python_version//./}"
    local wheel_url="https://download.pytorch.org/whl/nightly/cpu/torch-2.16.0.dev20261109%2Bcpu-${python_tag}-${python_tag}-manylinux_2_28_x86_64.whl"
    # shellcheck disable=SC2086
    conda run ${env_prefix} python -m pip install "$wheel_url" || return 1
    # shellcheck disable=SC2086
    conda run ${env_prefix} python -c 'import torch; assert callable(torch.jit.script) and callable(torch.jit.trace)' || return 1
  else
    install_pytorch_pip "$env_name" "${channel_version:-nightly}" cpu || return 1
  fi

  local selected_version
  # shellcheck disable=SC2086
  selected_version=$(conda run ${env_prefix} python -c 'from importlib.metadata import version; print(version("torch"))') || return 1
  if [[ -n "$torch_version" && "$selected_version" != "$torch_version" ]]; then
    echo "Error: TorchRec CPU CI did not install the selected PyTorch version." >&2
    return 1
  fi
  PIP_CONSTRAINT=$(mktemp "${RUNNER_TEMP:-/tmp}/torchrec_cpu_pytorch.XXXXXX") || return 1
  printf 'torch==%s\n' "$selected_version" > "$PIP_CONSTRAINT" || return 1
  export PIP_CONSTRAINT
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf 'PIP_CONSTRAINT=%s\n' "$PIP_CONSTRAINT" >> "$GITHUB_ENV" || return 1
  fi
}

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

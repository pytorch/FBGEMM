#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# GPU Detection Functions
################################################################################

# Detect the GPU vendor based on available system management tools.
#
# Returns:
#   "nvidia" - if nvidia-smi command is detected
#   "amd"    - if rocm-smi command is detected
#   ""       - if neither is detected
#
# Usage:
#   source gpu.bash
#   vendor=$(detect_gpu_vendor)
#   if [[ "$vendor" == "nvidia" ]]; then
#       echo "NVIDIA GPU detected"
#   elif [[ "$vendor" == "amd" ]]; then
#       echo "AMD GPU detected"
#   else
#       echo "No GPU detected"
#   fi
#
detect_gpu_vendor() {
    if command -v nvidia-smi &> /dev/null; then
        echo "nvidia"
    elif command -v amd-smi &> /dev/null; then
        echo "amd"
    else
        echo ""
    fi
}

# Detect the GPU model of the first NVIDIA GPU.
#
# This function queries nvidia-smi for the GPU name and extracts the model.
# It supports custom mappings via the GPU_MODEL_MAP associative array.
# The returned model name is always lowercased.
#
# Returns:
#   Lowercased GPU model (e.g., "h100", "a100", "v100")
#   "" - if nvidia-smi is not available or no GPU is detected
#
# Usage:
#   source gpu.bash
#   model=$(detect_nvidia_gpu_model)
#   echo "GPU model: $model"  # e.g., "h100"
#
#   # Add custom mapping before calling:
#   GPU_MODEL_MAP["CustomGPU"]="a100"
#   model=$(detect_nvidia_gpu_model)
#
detect_nvidia_gpu_model() {
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi not found; cannot detect NVIDIA GPU model" >&2
        return 1
    fi

    # Associative array for custom GPU model mappings.
    # Keys should be the model name (after stripping "NVIDIA " prefix).
    # Values should be the desired lowercased model name.
    #
    # Example:
    #   GPU_MODEL_MAP["PG509-210"]="a100"
    #
    declare -A GPU_MODEL_MAP=(
        ["PG509-210"]="a100"
    )

    # Get the raw GPU name from nvidia-smi (first GPU only)
    local raw_name
    raw_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

    if [[ -z "$raw_name" ]]; then
        echo ""
        return
    fi

    # Trim whitespace and strip "NVIDIA " prefix
    local model
    model=$(echo "$raw_name" | xargs | sed 's/^NVIDIA //')

    # Check if there's a custom mapping for this model
    if [[ -n "${GPU_MODEL_MAP[$model]+x}" ]]; then
        echo "${GPU_MODEL_MAP[$model]}"
        return
    fi

    # Extract just the base model (e.g., "A100-SXM4-80GB" -> "A100")
    local base_model="$model"
    if [[ "$model" =~ ^([A-Za-z]+[0-9]+) ]]; then
        base_model="${BASH_REMATCH[1]}"
    fi

    # Check if there's a custom mapping for the base model
    if [[ -n "${GPU_MODEL_MAP[$base_model]+x}" ]]; then
        echo "${GPU_MODEL_MAP[$base_model]}"
        return
    fi

    # Lowercase the base model
    echo "${base_model,,}"
}

# Detect the GPU model of the first AMD GPU.
#
# This function queries amd-smi for the GFX Version and maps it to a known
# GPU model name using the AMD_GFX_MODEL_MAP associative array.
# The returned model name is always lowercased.
#
# Returns:
#   Lowercased GPU model (e.g., "mi300", "mi350", "mi250")
#   "" - if amd-smi is not available or no GPU is detected
#
# Usage:
#   source gpu.bash
#   model=$(detect_amd_gpu_model)
#   echo "GPU model: $model"  # e.g., "mi350"
#
detect_amd_gpu_model() {
    # Check if amd-smi is available
    if ! command -v amd-smi &> /dev/null; then
        echo "amd-smi not found; cannot detect AMD GPU model" >&2
        return 1
    fi

    # Associative array mapping GFX versions to GPU model names.
    # Keys are the GFX versions (from "TARGET_GRAPHICS_VERSION" field in amd-smi static --asic).
    # Values are the desired lowercased model names.
    #
    # Target architecture, card model, and ROCm compatibility tables can be found
    # in the following links:
    #   https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
    #   https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
    #   https://www.coelacanth-dream.com/posts/2019/12/30/did-rid-product-matome-p2/
    #
    # To find the GFX version for a new GPU, run:
    #   amd-smi static --asic | grep "TARGET_GRAPHICS_VERSION"
    #
    declare -A AMD_GFX_MODEL_MAP=(
        # MI350 series (CDNA 4)
        ["gfx950"]="mi350"
        # MI300 series (CDNA 3)
        ["gfx942"]="mi300"
        # MI200 series (CDNA 2)
        ["gfx90a"]="mi250"
        # MI100 series (CDNA 1)
        ["gfx908"]="mi100"
        # MI50/MI60 (Vega)
        ["gfx906"]="mi50"
    )

    # Get the GFX Version from amd-smi (first GPU only)
    # amd-smi static --asic outputs something like:
    #   GPU: 0
    #       ASIC:
    #           TARGET_GRAPHICS_VERSION: gfx950
    local gfx_version
    gfx_version=$(amd-smi static --asic 2>/dev/null | grep -m1 "TARGET_GRAPHICS_VERSION:" | sed 's/.*TARGET_GRAPHICS_VERSION:[[:space:]]*//' | xargs)

    if [[ -z "$gfx_version" ]]; then
        echo "Could not detect AMD GPU GFX version" >&2
        return 1
    fi

    # Lowercase the GFX version for consistent lookup
    gfx_version="${gfx_version,,}"

    # Look up the GFX version in the map
    if [[ -n "${AMD_GFX_MODEL_MAP[$gfx_version]+x}" ]]; then
        echo "${AMD_GFX_MODEL_MAP[$gfx_version]}"
        return 0
    fi

    # GFX version not found in map
    echo "Unknown AMD GPU GFX version: $gfx_version" >&2
    return 1
}

# Resolve GPU vendor and model from an optional vendor/model argument, falling
# back to auto-detection when no argument is provided.
#
# On success, this function sets two global variables:
#   GPU_VENDOR  - "nvidia" or "amd"
#   GPU_MODEL   - lowercased model name (e.g., "h100", "mi350")
#
# If a vendor/model argument is supplied (e.g., "nvidia/h100"), the vendor is
# validated and both values are accepted directly.  Otherwise, detect_gpu_vendor
# and the vendor-specific model detection functions are called.
#
# Arguments:
#   $1 (optional) - GPU specification in "vendor/model" format
#
# Returns:
#   0 on success (GPU_VENDOR and GPU_MODEL are set)
#   1 on failure (with an error message on stderr)
#
# Usage:
#   source gpu_detect.bash
#
#   # With an explicit argument
#   resolve_gpu "nvidia/h100"
#
#   # With auto-detection
#   resolve_gpu
#
#   echo "Vendor: $GPU_VENDOR  Model: $GPU_MODEL"
#
resolve_gpu() {
    local gpu_spec="${1:-}"

    if [ -n "${gpu_spec}" ] && [[ "${gpu_spec}" =~ ^[^/]+/[^/]+$ ]]; then
        GPU_VENDOR=$(echo "${gpu_spec}" | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')
        GPU_MODEL=$(echo "${gpu_spec}" | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')

        case "${GPU_VENDOR}" in
            nvidia|amd) ;;
            *)
                echo "Error: Unsupported vendor '${GPU_VENDOR}'. Supported vendors: nvidia, amd" >&2
                return 1
                ;;
        esac
    else
        echo "No GPU specification provided. Auto-detecting GPU..."

        GPU_VENDOR=$(detect_gpu_vendor)
        if [ -z "${GPU_VENDOR}" ]; then
            echo "Error: Could not detect GPU vendor. Please specify <vendor>/<model> manually." >&2
            return 1
        fi

        case "${GPU_VENDOR}" in
            nvidia) GPU_MODEL=$(detect_nvidia_gpu_model) ;;
            amd)    GPU_MODEL=$(detect_amd_gpu_model) ;;
        esac

        if [ -z "${GPU_MODEL}" ]; then
            echo "Error: Could not detect GPU model. Please specify <vendor>/<model> manually." >&2
            return 1
        fi

        echo "Detected GPU: ${GPU_VENDOR}/${GPU_MODEL}"
    fi
}

################################################################################
# GPU Availability Functions
################################################################################

# Detect the total number of GPUs on the system.
#
# Uses GPU_VENDOR if set, otherwise calls detect_gpu_vendor to determine
# which GPU management tool to query.
#
# Returns:
#   The number of GPUs (printed to stdout)
#
# Usage:
#   num_gpus=$(detect_gpu_count)
#
detect_gpu_count() {
    local vendor="${GPU_VENDOR:-$(detect_gpu_vendor)}"

    if [[ "${vendor}" == "nvidia" ]]; then
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
    elif [[ "${vendor}" == "amd" ]]; then
        amd-smi list 2>/dev/null | grep -c "^GPU"
    else
        echo 1
    fi
}

# Check if a specific GPU is busy (has running processes or high utilization).
#
# For NVIDIA GPUs, checks if any compute processes are running on the GPU.
# For AMD GPUs, checks if GPU utilization exceeds a threshold (default 5%).
#
# Uses GPU_VENDOR if set, otherwise calls detect_gpu_vendor to determine
# which GPU management tool to query.
#
# Arguments:
#   $1 - GPU device index (e.g., 0, 1, 2)
#   $2 (optional) - Utilization threshold for AMD GPUs (default: 5)
#
# Returns:
#   0 if the GPU is busy
#   1 if the GPU is free
#
# Usage:
#   if gpu_is_busy 0; then
#       echo "GPU 0 is busy"
#   else
#       echo "GPU 0 is free"
#   fi
#
gpu_is_busy() {
    local gpu_id="$1"
    local util_threshold="${2:-5}"
    local vendor="${GPU_VENDOR:-$(detect_gpu_vendor)}"

    if [[ "${vendor}" == "nvidia" ]]; then
        local procs
        procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "${gpu_id}" 2>/dev/null | grep -v "^$" || true)
        if [[ -n "${procs}" ]]; then
            return 0
        fi
    elif [[ "${vendor}" == "amd" ]]; then
        local util
        util=$(amd-smi metric -g "${gpu_id}" --usage 2>/dev/null | grep "GFX_ACTIVITY:" | awk '{print $(NF-1)}' || echo "0")
        if [[ "${util}" -gt "${util_threshold}" ]]; then
            return 0
        fi
    fi
    return 1
}

# Discover which GPUs are available (not busy) on the system.
#
# Probes each GPU and populates two global arrays:
#   AVAILABLE_GPUS  - indices of free GPUs
#   EXCLUDED_GPUS   - indices of busy GPUs
#
# Requires GPU_VENDOR to be set (call resolve_gpu first).
#
# Arguments:
#   $1 (optional) - Utilization threshold for AMD GPUs (default: 5)
#
# Usage:
#   resolve_gpu
#   discover_available_gpus
#   echo "Available: ${AVAILABLE_GPUS[*]}"
#   echo "Excluded:  ${EXCLUDED_GPUS[*]}"
#
discover_available_gpus() {
    local util_threshold="${1:-5}"
    local vendor="${GPU_VENDOR:-$(detect_gpu_vendor)}"
    local num_gpus
    num_gpus=$(detect_gpu_count)

    AVAILABLE_GPUS=()
    EXCLUDED_GPUS=()

    for ((i = 0; i < num_gpus; i++)); do
        if gpu_is_busy "$i" "${util_threshold}"; then
            EXCLUDED_GPUS+=("$i")
            if [[ "${vendor}" == "nvidia" ]]; then
                local procs
                procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$i" 2>/dev/null || true)
                echo "GPU $i: EXCLUDED (processes: ${procs})"
            else
                echo "GPU $i: EXCLUDED (busy)"
            fi
        else
            AVAILABLE_GPUS+=("$i")
        fi
    done

    if [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; then
        echo "WARNING: All GPUs are busy. Falling back to GPU 0."
        AVAILABLE_GPUS=(0)
    fi
}

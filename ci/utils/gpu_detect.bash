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
    elif command -v rocm-smi &> /dev/null; then
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
    if [[ -n "${GPU_MODEL_MAP[$model]}" ]]; then
        echo "${GPU_MODEL_MAP[$model]}"
        return
    fi

    # Extract just the base model (e.g., "A100-SXM4-80GB" -> "A100")
    local base_model="$model"
    if [[ "$model" =~ ^([A-Za-z]+[0-9]+) ]]; then
        base_model="${BASH_REMATCH[1]}"
    fi

    # Check if there's a custom mapping for the base model
    if [[ -n "${GPU_MODEL_MAP[$base_model]}" ]]; then
        echo "${GPU_MODEL_MAP[$base_model]}"
        return
    fi

    # Lowercase the base model
    echo "${base_model,,}"
}

# Detect the GPU model of the first AMD GPU.
#
# This function queries rocm-smi for the GPU name and extracts the model.
# It supports custom mappings via the AMD_GPU_MODEL_MAP associative array.
# The returned model name is always lowercased.
#
# Returns:
#   Lowercased GPU model (e.g., "mi300", "mi250", "mi100")
#   "" - if rocm-smi is not available or no GPU is detected
#
# Usage:
#   source gpu.bash
#   model=$(detect_amd_gpu_model)
#   echo "GPU model: $model"  # e.g., "mi300"
#
#   # Add custom mapping before calling:
#   AMD_GPU_MODEL_MAP["CustomGPU"]="mi300"
#   model=$(detect_amd_gpu_model)
#
detect_amd_gpu_model() {
    # Check if rocm-smi is available
    if ! command -v rocm-smi &> /dev/null; then
        echo "rocm-smi not found; cannot detect AMD GPU model" >&2
        return 1
    fi

    # Associative array for custom AMD GPU model mappings.
    # Keys should be the model name (after stripping common prefixes).
    # Values should be the desired lowercased model name.
    #
    # Example:
    #   AMD_GPU_MODEL_MAP["Instinct MI300X OAM"]="mi300"
    #
    declare -A AMD_GPU_MODEL_MAP=(
        # Add custom mappings here as needed
    )

    # Get the raw GPU name from rocm-smi (first GPU only)
    # rocm-smi --showproductname outputs something like:
    #   GPU[0]          : Card Series:       AMD Instinct MI300X OAM
    local raw_name
    raw_name=$(rocm-smi --showproductname 2>/dev/null | grep -m1 "Card Series:" | sed 's/.*Card Series:[[:space:]]*//' | xargs)

    # If showproductname doesn't work, try alternative methods
    if [[ -z "$raw_name" ]]; then
        # Try --showname which outputs: GPU[0] : Name : <name>
        raw_name=$(rocm-smi --showname 2>/dev/null | grep -m1 "Name" | sed 's/.*Name[[:space:]]*:[[:space:]]*//' | xargs)
    fi

    if [[ -z "$raw_name" ]]; then
        echo ""
        return
    fi

    # Strip common prefixes like "AMD ", "AMD Instinct "
    local model
    model=$(echo "$raw_name" | sed -e 's/^AMD Instinct //' -e 's/^AMD //' -e 's/^Instinct //')

    # Check if there's a custom mapping for this model
    if [[ -n "${AMD_GPU_MODEL_MAP[$model]}" ]]; then
        echo "${AMD_GPU_MODEL_MAP[$model]}"
        return
    fi

    # Extract just the base model without trailing letters
    # e.g., "MI300X OAM" -> "MI300", "MI250X" -> "MI250"
    # This strips trailing letters like X, A after the number
    local base_model="$model"
    if [[ "$model" =~ ^([A-Za-z]+[0-9]+) ]]; then
        base_model="${BASH_REMATCH[1]}"
    fi

    # Check if there's a custom mapping for the base model
    if [[ -n "${AMD_GPU_MODEL_MAP[$base_model]}" ]]; then
        echo "${AMD_GPU_MODEL_MAP[$base_model]}"
        return
    fi

    # Lowercase the base model
    echo "${base_model,,}"
}

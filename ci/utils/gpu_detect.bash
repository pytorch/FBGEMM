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
# This function queries rocm-smi for the GFX Version and maps it to a known
# GPU model name using the AMD_GFX_MODEL_MAP associative array.
# The returned model name is always lowercased.
#
# Returns:
#   Lowercased GPU model (e.g., "mi300", "mi350", "mi250")
#   "" - if rocm-smi is not available or no GPU is detected
#
# Usage:
#   source gpu.bash
#   model=$(detect_amd_gpu_model)
#   echo "GPU model: $model"  # e.g., "mi350"
#
detect_amd_gpu_model() {
    # Check if rocm-smi is available
    if ! command -v rocm-smi &> /dev/null; then
        echo "rocm-smi not found; cannot detect AMD GPU model" >&2
        return 1
    fi

    # Associative array mapping GFX versions to GPU model names.
    # Keys are the GFX versions (from "GFX Version" field in rocm-smi --showproductname).
    # Values are the desired lowercased model names.
    #
    # Target architecture, card model, and ROCm compatibility tables can be found
    # in the following links:
    #   https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
    #   https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
    #   https://www.coelacanth-dream.com/posts/2019/12/30/did-rid-product-matome-p2/
    #
    # To find the GFX version for a new GPU, run:
    #   rocm-smi --showproductname | grep "GFX Version"
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

    # Get the GFX Version from rocm-smi (first GPU only)
    # rocm-smi --showproductname outputs something like:
    #   GPU[0]          : GFX Version:       gfx950
    local gfx_version
    gfx_version=$(rocm-smi --showproductname 2>/dev/null | grep -m1 "GFX Version:" | sed 's/.*GFX Version:[[:space:]]*//' | xargs)

    if [[ -z "$gfx_version" ]]; then
        echo "Could not detect AMD GPU GFX version" >&2
        return 1
    fi

    # Lowercase the GFX version for consistent lookup
    gfx_version="${gfx_version,,}"

    # Look up the GFX version in the map
    if [[ -n "${AMD_GFX_MODEL_MAP[$gfx_version]}" ]]; then
        echo "${AMD_GFX_MODEL_MAP[$gfx_version]}"
        return 0
    fi

    # GFX version not found in map
    echo "Unknown AMD GPU GFX version: $gfx_version" >&2
    return 1
}

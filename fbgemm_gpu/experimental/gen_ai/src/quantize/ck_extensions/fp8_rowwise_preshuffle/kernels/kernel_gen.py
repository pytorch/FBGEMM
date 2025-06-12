# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re


def load_file_to_list(filename):
    try:
        with open(filename, "r") as file:
            lines = [line.strip() for line in file]
        return lines[1:]
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return []


def parse_lines(lines):
    parsed_lines = []
    for line in lines:
        # Skip commented out lines.
        if line[0] == "#":
            continue
        entries = line.replace(" ", "")
        # Convert entry string to a list.
        entries = re.split(r",(?![^<>]*>)", entries)
        # Create configuration map.
        config_map = {
            "BLOCK_SIZE": entries[0],
            "MBLOCK": entries[1],
            "NBLOCK": entries[2],
            "KBLOCK": entries[3],
            "WAVE_TILE_M": entries[6],
            "WAVE_TILE_N": entries[7],
            "WAVE_MAP_M": entries[8],
            "WAVE_MAP_N": entries[9],
            "ABLOCK_TRANSFER": entries[10],
            "BBLOCK_TRANSFER": entries[17],
            "CSHUFFLE_MX_PER_WAVE_PERSHUFFLE": entries[24],
            "CSHUFFLE_NX_PER_WAVE_PERSHUFFLE": entries[25],
            "CBLOCK_TRANSFER": entries[26],
            "CBLOCK_SPV": entries[27],
            "LOOP_SCHEDULE": "ck::" + entries[28],
            "PIPELINE_VERSION": entries[29],
        }

        def sequence_to_name(s):
            # Replace sequences with encoding strings.
            pattern = r"S<(\d+),(\d+),(\d+)>"
            replacement = r"\1x\2x\3"
            s = re.sub(pattern, replacement, s)
            pattern = r"S<(\d+),(\d+),(\d+),(\d+)>"
            replacement = r"\1x\2x\3x\4"
            return re.sub(pattern, replacement, s)

        # Construct kernel name for this configuration.
        kernel_name = (
            f"fp8_rowwise_preshuffle_"
            f"{config_map['BLOCK_SIZE']}x{config_map['MBLOCK']}x{config_map['NBLOCK']}x{config_map['KBLOCK']}_"
            f"{config_map['WAVE_TILE_M']}x{config_map['WAVE_TILE_N']}_"
            f"{config_map['WAVE_MAP_M']}x{config_map['WAVE_MAP_N']}_"
            f"{sequence_to_name(config_map['ABLOCK_TRANSFER'])}_"
            f"{sequence_to_name(config_map['BBLOCK_TRANSFER'])}_"
            f"{sequence_to_name(config_map['CBLOCK_TRANSFER'])}_"
            f"{sequence_to_name(config_map['CBLOCK_SPV'])}_"
            f"{config_map['CSHUFFLE_MX_PER_WAVE_PERSHUFFLE']}x{config_map['CSHUFFLE_NX_PER_WAVE_PERSHUFFLE']}_"
            f"{'intrawave' if 'Intrawave' in config_map['LOOP_SCHEDULE'] else 'interwave'}_"
        )

        # Extend unspecified pipeline version.
        if config_map["PIPELINE_VERSION"] == "BlkGemmPipeVer":
            for version in ["v1", "v2"]:
                config_map["PIPELINE_VERSION"] = (
                    "ck::BlockGemmPipelineVersion::" + version
                )
                config_map["KERNEL_NAME"] = kernel_name + version
                parsed_lines.append(config_map.copy())
        else:
            config_map["PIPELINE_VERSION"] = "ck::" + config_map["PIPELINE_VERSION"]
            config_map["KERNEL_NAME"] = (
                kernel_name + config_map["PIPELINE_VERSION"].split("::")[-1]
            )
            parsed_lines.append(config_map)

    return parsed_lines


def generate_kernels(kernel_configs):
    kernel_template = """/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_preshuffle_common.h"

at::Tensor
{KERNEL_NAME}(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {{

  return f8f8bf16_rowwise_preshuffle_wrapper<
      {BLOCK_SIZE},
      {MBLOCK},
      {NBLOCK},
      {KBLOCK},
      {WAVE_TILE_M},
      {WAVE_TILE_N},
      {WAVE_MAP_M},
      {WAVE_MAP_N},
      {ABLOCK_TRANSFER},
      {BBLOCK_TRANSFER},
      {CBLOCK_TRANSFER},
      {CBLOCK_SPV},
      {CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
      {CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
      {LOOP_SCHEDULE},
      {PIPELINE_VERSION}>(XQ, WQ, x_scale, w_scale, Y, 1);
}}
"""
    for kernel_config in kernel_configs:
        # Format template for passed kernel configuration.
        generated_kernel = kernel_template.format(
            KERNEL_NAME=kernel_config["KERNEL_NAME"],
            BLOCK_SIZE=kernel_config["BLOCK_SIZE"],
            MBLOCK=kernel_config["MBLOCK"],
            NBLOCK=kernel_config["NBLOCK"],
            KBLOCK=kernel_config["KBLOCK"],
            WAVE_TILE_M=kernel_config["WAVE_TILE_M"],
            WAVE_TILE_N=kernel_config["WAVE_TILE_N"],
            WAVE_MAP_M=kernel_config["WAVE_MAP_M"],
            WAVE_MAP_N=kernel_config["WAVE_MAP_N"],
            ABLOCK_TRANSFER=kernel_config["ABLOCK_TRANSFER"],
            BBLOCK_TRANSFER=kernel_config["BBLOCK_TRANSFER"],
            CBLOCK_TRANSFER=kernel_config["CBLOCK_TRANSFER"],
            CBLOCK_SPV=kernel_config["CBLOCK_SPV"],
            CSHUFFLE_MX_PER_WAVE_PERSHUFFLE=kernel_config[
                "CSHUFFLE_MX_PER_WAVE_PERSHUFFLE"
            ],
            CSHUFFLE_NX_PER_WAVE_PERSHUFFLE=kernel_config[
                "CSHUFFLE_NX_PER_WAVE_PERSHUFFLE"
            ],
            LOOP_SCHEDULE=kernel_config["LOOP_SCHEDULE"],
            PIPELINE_VERSION=kernel_config["PIPELINE_VERSION"],
        )
        kernel_file_name = kernel_config["KERNEL_NAME"] + ".hip"
        with open(kernel_file_name, "w") as kernel_file:
            kernel_file.write(generated_kernel)
        print(f"Generated kernel: {kernel_file_name}")


def generate_manifest(kernel_configs):
    # Define templates for manifest.
    manifest_body = """/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>

"""
    kernel_stub_template = """at::Tensor
{KERNEL_NAME}(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

"""
    # Iterate over kernel configurations and populate manifest.
    for kernel_config in kernel_configs:
        manifest_body += kernel_stub_template.format(
            KERNEL_NAME=kernel_config["KERNEL_NAME"]
        )

    # Write out manifest file.
    manifest_file_name = "fp8_rowwise_preshuffle_kernel_manifest.h"
    with open(manifest_file_name, "w") as manifest_file:
        manifest_file.write(manifest_body)
    print(f"Generated manifest: {manifest_file_name}")


if __name__ == "__main__":
    # Load a file of kernel templates and generate individual kernels.
    filename = "kernel_templates.txt"
    lines = load_file_to_list(filename)
    kernel_configs = parse_lines(lines)
    generate_kernels(kernel_configs)
    generate_manifest(kernel_configs)
    # Run clang format to make sure all generated code looks nice.
    os.system("clang-format -i ./*.hip")
    os.system("clang-format -i ./*.h")

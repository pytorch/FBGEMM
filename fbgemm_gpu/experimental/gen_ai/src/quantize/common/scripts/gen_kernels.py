# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess

COPYRIGHT = """/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
 """

KERNEL_ID_TEMPLATES = {
    "bf16bf16bf16_grouped": "bf16bf16bf16_grouped_{tM}_{tN}_{tK}_{cM}_{cN}_{cK}_{pong[0]}",
    "f8f8bf16_rowwise": "f8f8bf16_rowwise_{tM}_{tN}_{tK}_{cM}_{cN}_{cK}_{arch}_{pong[0]}_{coop[0]}",
}

bf16bf16bf16_grouped_decl_template = """
at::Tensor {kernel_id}(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);

at::Tensor {kernel_id}(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes);
    """

f8f8bf16_rowwise_decl_template = """
at::Tensor {kernel_id}(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt);
    """

DECL_TEMPLATES = {
    "bf16bf16bf16_grouped": bf16bf16bf16_grouped_decl_template,
    "f8f8bf16_rowwise": f8f8bf16_rowwise_decl_template,
}


bf16bf16bf16_grouped_file_template = """
at::Tensor {kernel_id}(
    at::Tensor X, // BF16
    at::Tensor W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes) {{
  return bf16bf16bf16_grouped_impl<at::Tensor, {tM}, {tN}, {tK}, {cM}, {cN}, {cK}, {pong}>(
      X, W, output, zero_start_index_M, M_sizes);
}}

at::Tensor {kernel_id}(
    at::TensorList X, // BF16
    at::TensorList W, // BF16
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes) {{
  return bf16bf16bf16_grouped_impl<at::TensorList, {tM}, {tN}, {tK}, {cM}, {cN}, {cK}, {pong}>(
      X, W, output, zero_start_index_M, M_sizes);
}}
"""

f8f8bf16_rowwise_file_template = """
at::Tensor {kernel_id}(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    bool use_fast_accum = true,
    std::optional<at::Tensor> bias = std::nullopt,
    std::optional<at::Tensor> output = std::nullopt) {{
  // Dispatch this kernel to the correct underlying implementation.
  return f8f8bf16_rowwise_wrapper<{tM}, {tN}, {tK}, {cM}, {cN}, {cK}, {arch}, {pong}, {coop}>(
      XQ, WQ, x_scale, w_scale, use_fast_accum, bias, output);
}}
"""

FILE_TEMPLATES = {
    "bf16bf16bf16_grouped": bf16bf16bf16_grouped_file_template,
    "f8f8bf16_rowwise": f8f8bf16_rowwise_file_template,
}

bf16bf16bf16_grouped_kernel_map_template = """
template <typename InputType>
using Kernel_bf16bf16bf16_grouped = at::Tensor (*)(
    InputType,
    InputType,
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>);

template <typename InputType>
const std::unordered_map<std::string, Kernel_bf16bf16bf16_grouped<InputType>>&
get_bf16bf16bf16_grouped_kernels() {{
  static const std::unordered_map<std::string, Kernel_bf16bf16bf16_grouped<InputType>> kernels = {{
    {body}
  }};
  return kernels;
}}
"""

f8f8bf16_rowwise_kernel_map_template = """
using Kernel_f8f8bf16_rowwise = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    bool,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>);

const std::unordered_map<std::string, Kernel_f8f8bf16_rowwise>&
get_f8f8bf16_rowwise_kernels(int arch) {{
  static const std::unordered_map<std::string, Kernel_f8f8bf16_rowwise> kernelsSM90 = {{
    {bodySM90}
  }};
  static const std::unordered_map<std::string, Kernel_f8f8bf16_rowwise> kernelsSM100 = {{
    {bodySM100}
  }};
  if (arch == 10) {{
    return kernelsSM100;
  }} else {{
    return kernelsSM90;
  }}
}}
"""

ARCH_MAP_TEMPLATES = {"f8f8bf16_rowwise": f8f8bf16_rowwise_kernel_map_template}
MAP_TEMPLATES = {"bf16bf16bf16_grouped": bf16bf16bf16_grouped_kernel_map_template}


def gen_kernel_map_body(kernel_confs, arch=None):
    return "\n".join(
        [
            f'{{"{kernel_conf['kernel_id']}", {kernel_conf['kernel_id']}}},'
            for kernel_conf in kernel_confs
            if arch is None or kernel_conf["arch"] == arch
        ]
    )


def gen_kernel_map(kernel_name, kernel_confs):
    if kernel_name in MAP_TEMPLATES:
        body = gen_kernel_map_body(kernel_confs)
        return MAP_TEMPLATES[kernel_name].format(body=body)

    # ARCH_MAP_TEMPLATES
    bodySM90 = gen_kernel_map_body(kernel_confs, 9)
    bodySM100 = gen_kernel_map_body(kernel_confs, 10)
    return ARCH_MAP_TEMPLATES[kernel_name].format(
        bodySM90=bodySM90, bodySM100=bodySM100
    )


def get_kernel_confs_nv(kernel_name):
    # Change these as needed to explore different kernels configurations.
    tiles = [
        (M, N, K)
        for M in (64, 128, 256)
        for N in (
            16,
            32,
            64,
            128,
            256,
        )
        for K in (128,)
    ]
    clusters = [(1, 1, 1), (2, 1, 1), (4, 1, 1)]
    schedules = [("false", "false"), ("true", "false"), ("false", "true")]
    # SM90 and SM100
    archs = [9, 10]

    # Some kernels may not support all parameters (e.g. only 1 type of schedule), filter them out to prevent duplicates.
    generated = set()

    kernel_confs = []
    for arch in archs:
        for tM, tN, tK in tiles:
            for cM, cN, cK in clusters:
                for pong, coop in schedules:
                    # Co-operative requires tM >= 128
                    if tM < 128 and (
                        coop == "true"
                        # This kernel only supports pong OR coop, and not regular warp persistent
                        or (kernel_name == "bf16bf16bf16_grouped" and pong == "false")
                    ):
                        continue

                    # This tile size is generally bad
                    if tM == 256 and tN == 256:
                        continue

                    # To compile less kernels skip pong & coop for smaller tiles as they don't reach the compute roofline
                    if (pong == "true" or coop == "true") and not (
                        tM >= 128 and tN >= 128
                    ):
                        continue

                    # SM100 specific
                    if arch == 10:
                        # M cluster == 1 requires specific M tile size
                        if cM == 1 and not (tM == 64 or tM == 128):
                            continue

                        # M cluter > 1 requires N tile >= 32
                        if cM > 1 and tN < 32:
                            continue

                    kernel_conf = {
                        "arch": arch,
                        "tM": tM,
                        "tN": tN,
                        "tK": tK,
                        "cM": cM,
                        "cN": cN,
                        "cK": cK,
                        "pong": pong,
                        "coop": coop,
                    }
                    kernel_id = gen_kernel_id(kernel_name, kernel_conf)
                    if kernel_id not in generated:
                        generated.add(kernel_id)

                        kernel_conf["kernel_id"] = kernel_id
                        kernel_confs.append(kernel_conf)

    return kernel_confs


def gen_kernel_id(kernel_name, kernel_conf):
    template = KERNEL_ID_TEMPLATES[kernel_name]
    return template.format(**kernel_conf)


def gen_kernel_file(kernel_name, kernel_conf):
    template = FILE_TEMPLATES[kernel_name]
    formatted = template.format(**kernel_conf)

    return f"""{COPYRIGHT}
#include "{kernel_name}_common.cuh"

namespace fbgemm_gpu {{

{formatted}

}} // namespace fbgemm_gpu
"""


def gen_kernel_files(kernel_name, kernel_confs, output_dir):
    for kernel_conf in kernel_confs:
        kernel_id = kernel_conf["kernel_id"]
        file_path = os.path.join(output_dir, f"{kernel_id}.cu")
        with open(file_path, "w") as f:
            f.write(gen_kernel_file(kernel_name, kernel_conf))


def gen_kernel_manifest_decl(kernel_name, kernel_conf):
    template = DECL_TEMPLATES[kernel_name]
    return template.format(**kernel_conf)


def gen_kernel_manifest(kernel_name, kernel_confs, output_dir):
    body = "\n".join(
        gen_kernel_manifest_decl(kernel_name, kernel_conf)
        for kernel_conf in kernel_confs
    )
    kernel_map = gen_kernel_map(kernel_name, kernel_confs)

    manifest_content = f"""{COPYRIGHT}
#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {{

{body}

{kernel_map}

}} // namespace fbgemm_gpu
"""

    manifest_path = os.path.join(output_dir, f"{kernel_name}_manifest.cuh")
    with open(manifest_path, "w") as f:
        f.write(manifest_content)


def main():
    parser = argparse.ArgumentParser(description="Generate kernel files and manifest.")
    parser.add_argument(
        "--kernel_name",
        type=str,
        required=True,
        help="Name of the kernel to generate, e.g. bf16bf16bf16_grouped.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to place generated kernels. If unset, will default to the current working directory.",
    )
    args = parser.parse_args()

    # Determine the output directory
    output_dir = args.output_dir if args.output_dir is not None else os.getcwd()
    print(f"Will place generated files in {output_dir}")

    kernel_confs = get_kernel_confs_nv(args.kernel_name)
    print(f"Will generate {len(kernel_confs)}  kernels")
    gen_kernel_files(args.kernel_name, kernel_confs, output_dir)
    gen_kernel_manifest(args.kernel_name, kernel_confs, output_dir)

    # Format the generated files
    command = f"clang-format -i {output_dir}/*.{{cu,cuh}}"
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()

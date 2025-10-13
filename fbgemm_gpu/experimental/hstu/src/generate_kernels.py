#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.

import itertools
import os


DISABLE_BACKWARD = os.getenv("HSTU_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_DETERMINISTIC = os.getenv("HSTU_DISABLE_DETERMINISTIC", "TRUE") == "TRUE"
DISABLE_BF16 = os.getenv("HSTU_DISABLE_BF16", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("HSTU_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = os.getenv("HSTU_DISABLE_FP8", "FALSE") == "TRUE"
USE_E5M2_BWD = os.getenv("HSTU_USE_E5M2_BWD", "FALSE") == "TRUE"
DISABLE_HDIM32 = os.getenv("HSTU_DISABLE_HDIM32", "FALSE") == "TRUE"
DISABLE_HDIM64 = os.getenv("HSTU_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("HSTU_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("HSTU_DISABLE_HDIM256", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("HSTU_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_CAUSAL = os.getenv("HSTU_DISABLE_CAUSAL", "FALSE") == "TRUE"
DISABLE_CONTEXT = os.getenv("HSTU_DISABLE_CONTEXT", "FALSE") == "TRUE"
DISABLE_TARGET = os.getenv("HSTU_DISABLE_TARGET", "FALSE") == "TRUE"
DISABLE_ARBITRARY = os.getenv("HSTU_DISABLE_ARBITRARY", "FALSE") == "TRUE"
ARBITRARY_NFUNC = int(os.getenv("HSTU_ARBITRARY_NFUNC", "1"))
DISABLE_RAB = os.getenv("HSTU_DISABLE_RAB", "FALSE") == "TRUE"
DISABLE_DRAB = os.getenv("HSTU_DISABLE_DRAB", "FALSE") == "TRUE"
DISABLE_86OR89 = os.getenv("HSTU_DISABLE_86OR89", "FALSE") == "TRUE"

def generate_kernels_ampere(install_dir: str):
    """
    Generate HSTU kernels for Ampere architecture.
    """

    if DISABLE_BF16 and DISABLE_FP16:
        raise ValueError("At least one of DISABLE_BF16 or DISABLE_FP16 must be False")
    if DISABLE_HDIM32 and DISABLE_HDIM64 and DISABLE_HDIM128 and DISABLE_HDIM256:
        raise ValueError("At least one of DISABLE_HDIM32, DISABLE_HDIM64, DISABLE_HDIM128, or DISABLE_HDIM256 must be False")
    if DISABLE_RAB and not DISABLE_DRAB:
        raise ValueError("Cannot support drab without rab")
    if DISABLE_CAUSAL and not DISABLE_TARGET:
        raise ValueError("Cannot support target without causal")
    if not DISABLE_ARBITRARY and ARBITRARY_NFUNC % 2 == 0:
        raise ValueError("ARBITRARY_NFUNC must be odd")

    ARCH_SM = ["80"] + (["89"] if not DISABLE_86OR89 else [])
    DTYPE_16 = (["bf16"] if not DISABLE_BF16 else []) + (["fp16"] if not DISABLE_FP16 else [])
    HEAD_DIMENSIONS = (
        []
        + ([32] if not DISABLE_HDIM32 else [])
        + ([64] if not DISABLE_HDIM64 else [])
        + ([128] if not DISABLE_HDIM128 else [])
        + ([256] if not DISABLE_HDIM256 else [])
    )
    RAB = [""] + (["_rab"] if not DISABLE_RAB else [])
    RAB_DRAB = [""] + ((["_rab_drab", "_rab"]) if not DISABLE_DRAB else ["_rab"] if not DISABLE_RAB else [])
    MASK = [""]
    if not DISABLE_LOCAL:
        MASK += ["_local"]
    if not DISABLE_CAUSAL:
        CAUSAL_MASK = ["_causal"]
        CONTEXT_MASK = [""] + (["_context"] if not DISABLE_CONTEXT else [])
        TARGET_MASK = [""] + (["_target"] if not DISABLE_TARGET else [])
        MASK += [f"{c}{x}{t}" for c, x, t in itertools.product(CAUSAL_MASK, CONTEXT_MASK, TARGET_MASK)]
    if not DISABLE_ARBITRARY:
        MASK += ["_arbitrary"]
    BWD_DETERMINISTIC = ["false"] + (["true"] if not DISABLE_DETERMINISTIC else [])

    dtype_to_str = {
        "bf16": "cutlass::bfloat16_t",
        "fp16": "cutlass::half_t",
    }

    os.makedirs(install_dir, exist_ok=True)

    ampere_fwd_file_head = """
/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Splitting different head dimensions, data types and masks to different files to speed up
// compilation. This file is auto-generated. See generate_kernels.py

#include "hstu_fwd.h"

template void run_hstu_fwd_8x<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>
                             (Hstu_fwd_params& params, cudaStream_t stream);

    """
    for hdim, dtype, rab, mask, arch_sm in itertools.product(
        HEAD_DIMENSIONS, DTYPE_16, RAB, MASK, ARCH_SM
    ):
        file_name = f"{install_dir}/hstu_fwd_hdim{hdim}_{dtype}{rab}{mask}_fn{ARBITRARY_NFUNC}_sm{arch_sm}.cu" if "arbitrary" in mask else f"{install_dir}/hstu_fwd_hdim{hdim}_{dtype}{rab}{mask}_sm{arch_sm}.cu"
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(
                    ampere_fwd_file_head.format(
                        arch_sm,
                        dtype_to_str[dtype],
                        hdim,
                        "true" if "_rab" in rab else "false",
                        "true" if "local" in mask else "false",
                        "true" if "causal" in mask else "false",
                        "true" if "context" in mask else "false",
                        "true" if "target" in mask else "false",
                        "true" if "arbitrary" in mask else "false",
                        str(ARBITRARY_NFUNC) if "arbitrary" in mask else "0",
                    )
                )

    ampere_bwd_file_head = """
/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Splitting different head dimensions, data types and masks to different files to speed up
// compilation. This file is auto-generated. See generate_kernels.py

#include "hstu_bwd.h"

template void run_hstu_bwd_80<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>
                             (Hstu_bwd_params& params, cudaStream_t stream);

    """
    for hdim, dtype, rab_drab, mask, bwd_deterministic, arch_sm in itertools.product(
        HEAD_DIMENSIONS, DTYPE_16, RAB_DRAB, MASK, BWD_DETERMINISTIC, ARCH_SM
    ):
        file_name = f"{install_dir}/hstu_bwd_hdim{hdim}_{dtype}{rab_drab}{mask}_fn{ARBITRARY_NFUNC}_{bwd_deterministic}_sm{arch_sm}.cu" if "arbitrary" in mask else f"{install_dir}/hstu_bwd_hdim{hdim}_{dtype}{rab_drab}{mask}_{bwd_deterministic}_sm{arch_sm}.cu"
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(
                    ampere_bwd_file_head.format(
                        arch_sm,
                        dtype_to_str[dtype],
                        hdim,
                        "true" if "_rab" in rab_drab else "false",
                        "true" if "drab" in rab_drab else "false",
                        "true" if "local" in mask else "false",
                        "true" if "causal" in mask else "false",
                        "true" if "context" in mask else "false",
                        "true" if "target" in mask else "false",
                        "true" if "arbitrary" in mask else "false",
                        str(ARBITRARY_NFUNC) if "arbitrary" in mask else "0",
                        bwd_deterministic,
                    )
                )


def generate_kernels_hopper(install_dir: str):
    """
    Generate HSTU kernels for Hopper architecture.
    """

    if DISABLE_BF16 and DISABLE_FP16 and DISABLE_FP8:
        raise ValueError("At least one of DISABLE_BF16, DISABLE_FP16, or DISABLE_FP8 must be False")
    if DISABLE_FP8 and USE_E5M2_BWD:
        raise ValueError("Cannot support e5m2 bwd with fp8 disabled")
    if DISABLE_HDIM32 and DISABLE_HDIM64 and DISABLE_HDIM128 and DISABLE_HDIM256:
        raise ValueError("At least one of DISABLE_HDIM32, DISABLE_HDIM64, DISABLE_HDIM128, or DISABLE_HDIM256 must be False")
    if DISABLE_BACKWARD and not DISABLE_DRAB:
        raise ValueError("Cannot support drab without backward")
    if DISABLE_RAB and not DISABLE_DRAB:
        raise ValueError("Cannot support drab without rab")
    if DISABLE_CAUSAL and not DISABLE_TARGET:
        raise ValueError("Cannot support target without causal")
    if DISABLE_CAUSAL and not DISABLE_CONTEXT:
        raise ValueError("Cannot support context without causal")
    if not DISABLE_ARBITRARY and ARBITRARY_NFUNC % 2 == 0:
        raise ValueError("ARBITRARY_NFUNC must be odd")

    DTYPE_16 = (["bf16"] if not DISABLE_BF16 else []) + (["fp16"] if not DISABLE_FP16 else [])
    HEAD_DIMENSIONS = (
        []
        + ([32] if not DISABLE_HDIM32 else [])
        + ([64] if not DISABLE_HDIM64 else [])
        + ([128] if not DISABLE_HDIM128 else [])
        + ([256] if not DISABLE_HDIM256 else [])
    )
    RAB = [""] + (["_rab"] if not DISABLE_RAB else [])
    RAB_DRAB = [""] + ((["_rab_drab", "_rab"]) if not DISABLE_DRAB else ["_rab"] if not DISABLE_RAB else [])
    MASK = [""]
    if not DISABLE_LOCAL:
        MASK += ["_local"]
    if not DISABLE_CAUSAL:
        CAUSAL_MASK = ["_causal"]
        CONTEXT_MASK = [""] + (["_context"] if not DISABLE_CONTEXT else [])
        TARGET_MASK = [""] + (["_target"] if not DISABLE_TARGET else [])
        MASK += [f"{c}{x}{t}" for c, x, t in itertools.product(CAUSAL_MASK, CONTEXT_MASK, TARGET_MASK)]
    if not DISABLE_ARBITRARY:
        MASK += ["_arbitrary"]

    dtype_to_str = {
        "bf16": "cutlass::bfloat16_t",
        "fp16": "cutlass::half_t",
    }

    os.makedirs(install_dir, exist_ok=True)

    hopper_fwd_file_head = """
/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Splitting different head dimensions, data types and masks to different files to speed up
// compilation. This file is auto-generated. See generate_kernels.py

#include "hstu_fwd_launch_template.h"

template void run_hstu_fwd_<90, {}, {}, {}, {}, {}, {}, {}, {}, {}>
                           (Hstu_fwd_params& params, cudaStream_t stream);

    """
    for hdim, dtype, rab, mask in itertools.product(
        HEAD_DIMENSIONS, DTYPE_16, RAB, MASK
    ):
        file_name = f"{install_dir}/hstu_fwd_hdim{hdim}_{dtype}{rab}{mask}_fn{ARBITRARY_NFUNC}_sm90.cu" if "arbitrary" in mask else f"{install_dir}/hstu_fwd_hdim{hdim}_{dtype}{rab}{mask}_sm90.cu"
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(
                    hopper_fwd_file_head.format(
                        dtype_to_str[dtype],
                        hdim,
                        "true" if "_rab" in rab else "false",
                        "true" if "local" in mask else "false",
                        "true" if "causal" in mask else "false",
                        "true" if "context" in mask else "false",
                        "true" if "target" in mask else "false",
                        "true" if "arbitrary" in mask else "false",
                        str(ARBITRARY_NFUNC) if "arbitrary" in mask else "0",
                    )
                )

    if not DISABLE_FP8:
        for hdim, rab, mask in itertools.product(HEAD_DIMENSIONS, RAB, MASK):
            if hdim == 32:
                continue
            file_name = f"{install_dir}/hstu_fwd_hdim{hdim}_e4m3{rab}{mask}_fn{ARBITRARY_NFUNC}_sm90.cu" if "arbitrary" in mask else f"{install_dir}/hstu_fwd_hdim{hdim}_e4m3{rab}{mask}_sm90.cu"
            if not os.path.exists(file_name):
                with open(file_name, "w") as f:
                    f.write(
                        hopper_fwd_file_head.format(
                            "cutlass::float_e4m3_t",
                            hdim,
                            "true" if "_rab" in rab else "false",
                            "true" if "local" in mask else "false",
                            "true" if "causal" in mask else "false",
                            "true" if "context" in mask else "false",
                            "true" if "target" in mask else "false",
                            "true" if "arbitrary" in mask else "false",
                            str(ARBITRARY_NFUNC) if "arbitrary" in mask else "0",
                        )
                    )

    hopper_bwd_file_head = """
/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Splitting different head dimensions, data types and masks to different files to speed up
// compilation. This file is auto-generated. See generate_kernels.py

#include "hstu_bwd_launch_template.h"

template void run_hstu_bwd_<90, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>
                           (Hstu_bwd_params& params, cudaStream_t stream);

    """
    for hdim, dtype, rab_drab, mask in itertools.product(
        HEAD_DIMENSIONS, DTYPE_16, RAB_DRAB, MASK
    ):
        file_name = f"{install_dir}/hstu_bwd_hdim{hdim}_{dtype}{rab_drab}{mask}_fn{ARBITRARY_NFUNC}_sm90.cu" if "arbitrary" in mask else f"{install_dir}/hstu_bwd_hdim{hdim}_{dtype}{rab_drab}{mask}_sm90.cu"
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(
                    hopper_bwd_file_head.format(
                        dtype_to_str[dtype],
                        hdim,
                        "true" if "_rab" in rab_drab else "false",
                        "true" if "drab" in rab_drab else "false",
                        "true" if "local" in mask else "false",
                        "true" if "causal" in mask else "false",
                        "true" if "context" in mask else "false",
                        "true" if "target" in mask else "false",
                        "true" if "arbitrary" in mask else "false",
                        str(ARBITRARY_NFUNC) if "arbitrary" in mask else "0",
                    )
                )

    if not DISABLE_FP8 and not USE_E5M2_BWD:
        for hdim, rab_drab, mask in itertools.product(HEAD_DIMENSIONS, RAB_DRAB, MASK):
            if hdim == 32:
                continue
            file_name = f"{install_dir}/hstu_bwd_hdim{hdim}_e4m3{rab_drab}{mask}_fn{ARBITRARY_NFUNC}_sm90.cu" if "arbitrary" in mask else f"{install_dir}/hstu_bwd_hdim{hdim}_e4m3{rab_drab}{mask}_sm90.cu"
            if not os.path.exists(file_name):
                with open(file_name, "w") as f:
                    f.write(
                        hopper_bwd_file_head.format(
                            "cutlass::float_e4m3_t",
                            hdim,
                            "true" if "_rab" in rab_drab else "false",
                            "true" if "drab" in rab_drab else "false",
                            "true" if "local" in mask else "false",
                            "true" if "causal" in mask else "false",
                            "true" if "context" in mask else "false",
                            "true" if "target" in mask else "false",
                            "true" if "arbitrary" in mask else "false",
                            str(ARBITRARY_NFUNC) if "arbitrary" in mask else "0",
                        )
                    )

    if not DISABLE_FP8 and USE_E5M2_BWD:
        for hdim, rab_drab, mask in itertools.product(HEAD_DIMENSIONS, RAB_DRAB, MASK):
            if hdim == 32:
                continue
            file_name = f"{install_dir}/hstu_bwd_hdim{hdim}_e5m2{rab_drab}{mask}_fn{ARBITRARY_NFUNC}_sm90.cu" if "arbitrary" in mask else f"{install_dir}/hstu_bwd_hdim{hdim}_e5m2{rab_drab}{mask}_sm90.cu"
            if not os.path.exists(file_name):
                with open(file_name, "w") as f:
                    f.write(
                        hopper_bwd_file_head.format(
                            "cutlass::float_e5m2_t",
                            hdim,
                            "true" if "_rab" in rab_drab else "false",
                            "true" if "drab" in rab_drab else "false",
                            "true" if "local" in mask else "false",
                            "true" if "causal" in mask else "false",
                            "true" if "context" in mask else "false",
                            "true" if "target" in mask else "false",
                            "true" if "arbitrary" in mask else "false",
                            str(ARBITRARY_NFUNC) if "arbitrary" in mask else "0",
                        )
                    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch-list",
        type=str,
        default="8.0 9.0",
        help="Comma-separated list of CUDA architectures to generate kernels for",
    )
    parser.add_argument(
        "--install_dir",
        type=str,
        default=None,
        help="Output directory for generated source files",
    )
    args = parser.parse_args()

    if "8.0" in args.arch_list:
        # In OSS, the generated files will be written to hstu_ampere/instantiations
        generate_kernels_ampere(args.install_dir or "hstu_ampere/instantiations")

    if "9.0" in args.arch_list:
        # In OSS, the generated files will be written to hstu_hopper/instantiations
        generate_kernels_hopper(args.install_dir or "hstu_hopper/instantiations")


if __name__ == "__main__":
    main()

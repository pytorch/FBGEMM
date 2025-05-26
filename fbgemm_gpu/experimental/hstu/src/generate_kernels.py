# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import itertools


def generate_kernels_ampere():
    DTYPE_16 = ["bf16", "fp16"]
    HEAD_DIMENSIONS = [32, 64, 128, 256]
    RAB = ["", "_rab"]
    RAB_DRAB = ["", "_rab", "_rab_drab"]
    MASK = ["", "_local", "_local_deltaq", "_causal", "_causal_context", "_causal_target", "_causal_context_target", "_causal_deltaq"]

    dtype_to_str = {
        "bf16": "cutlass::bfloat16_t",
        "fp16": "cutlass::half_t",
    }

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

template void run_hstu_fwd_80<{}, {}, {}, {}, {}, {}, {}, {}>
                             (Hstu_fwd_params& params, cudaStream_t stream);

    """
    for hdim, dtype, rab, mask in itertools.product(HEAD_DIMENSIONS, DTYPE_16, RAB, MASK):
        file_name = f"hstu_ampere/instantiations/hstu_fwd_hdim{hdim}_{dtype}{rab}{mask}_sm80.cu"
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(ampere_fwd_file_head.format(dtype_to_str[dtype],
                                                    hdim,
                                                    "true" if "_rab" in rab else "false",
                                                    "true" if "local" in mask else "false",
                                                    "true" if "causal" in mask else "false",
                                                    "true" if "context" in mask else "false",
                                                    "true" if "target" in mask else "false",
                                                    "true" if "deltaq" in mask else "false"))

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

template void run_hstu_bwd_80<{}, {}, {}, {}, {}, {}, {}, {}, {}>
                             (Hstu_bwd_params& params, cudaStream_t stream);

    """
    for hdim, dtype, rab_drab, mask in itertools.product(HEAD_DIMENSIONS, DTYPE_16, RAB_DRAB, MASK):
        file_name = f"hstu_ampere/instantiations/hstu_bwd_hdim{hdim}_{dtype}{rab_drab}{mask}_sm80.cu"
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(ampere_bwd_file_head.format(dtype_to_str[dtype],
                                                    hdim,
                                                    "true" if "_rab" in rab_drab else "false",
                                                    "true" if "drab" in rab_drab else "false",
                                                    "true" if "local" in mask else "false",
                                                    "true" if "causal" in mask else "false",
                                                    "true" if "context" in mask else "false",
                                                    "true" if "target" in mask else "false",
                                                    "true" if "deltaq" in mask else "false"))


def generate_kernels_hopper():
    DTYPE_16 = ["bf16", "fp16"]
    HEAD_DIMENSIONS = [32, 64, 128, 256]
    RAB = ["", "_rab"]
    RAB_DRAB = ["", "_rab", "_rab_drab"]
    MASK = ["", "_local", "_local_deltaq", "_causal", "_causal_context", "_causal_target", "_causal_context_target", "_causal_deltaq"]
    FP8_MASK = ["", "_local", "_causal"]

    dtype_to_str = {
        "bf16": "cutlass::bfloat16_t",
        "fp16": "cutlass::half_t",
    }

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

template void run_hstu_fwd_<90, {}, {}, {}, {}, {}, {}, {}, {}>
                           (Hstu_fwd_params& params, cudaStream_t stream);

    """
    for hdim, dtype, rab, mask in itertools.product(HEAD_DIMENSIONS, DTYPE_16, RAB, MASK):
        file_name = f"hstu_hopper/instantiations/hstu_fwd_hdim{hdim}_{dtype}{rab}{mask}_sm90.cu"
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(hopper_fwd_file_head.format(dtype_to_str[dtype],
                                                    hdim,
                                                    "true" if "_rab" in rab else "false",
                                                    "true" if "local" in mask else "false",
                                                    "true" if "causal" in mask else "false",
                                                    "true" if "context" in mask else "false",
                                                    "true" if "target" in mask else "false",
                                                    "true" if "deltaq" in mask else "false"))

    for hdim, rab, mask in itertools.product(HEAD_DIMENSIONS, RAB, FP8_MASK):
        if hdim == 32:
            continue
        file_name = f"hstu_hopper/instantiations/hstu_fwd_hdim{hdim}_e4m3{rab}{mask}_sm90.cu"
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(hopper_fwd_file_head.format("cutlass::float_e4m3_t",
                                                    hdim,
                                                    "true" if "_rab" in rab else "false",
                                                    "true" if "local" in mask else "false",
                                                    "true" if "causal" in mask else "false",
                                                    "false",   # context
                                                    "false",   # target
                                                    "false"))  # deltaq

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

template void run_hstu_bwd_<90, {}, {}, {}, {}, {}, {}, {}, {}, {}>
                           (Hstu_bwd_params& params, cudaStream_t stream);

    """
    for hdim, dtype, rab_drab, mask in itertools.product(HEAD_DIMENSIONS, DTYPE_16, RAB_DRAB, MASK):
        file_name = f"hstu_hopper/instantiations/hstu_bwd_hdim{hdim}_{dtype}{rab_drab}{mask}_sm90.cu"
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(hopper_bwd_file_head.format(dtype_to_str[dtype],
                                                    hdim,
                                                    "true" if "_rab" in rab_drab else "false",
                                                    "true" if "drab" in rab_drab else "false",
                                                    "true" if "local" in mask else "false",
                                                    "true" if "causal" in mask else "false",
                                                    "true" if "context" in mask else "false",
                                                    "true" if "target" in mask else "false",
                                                    "true" if "deltaq" in mask else "false"))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch-list", type=str, default="8.0 9.0", help="Comma-separated list of CUDA architectures to generate kernels for")
    args = parser.parse_args()

    if "8.0" in args.arch_list:
        generate_kernels_ampere()

    if "9.0" in args.arch_list:
        generate_kernels_hopper()

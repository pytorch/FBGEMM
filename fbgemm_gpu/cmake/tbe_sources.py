# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Optimizer Group Definitions
################################################################################

COMMON_OPTIMIZERS = [
    "adagrad",
    "rowwise_adagrad",
    "sgd",
]

# To be populated in the subsequent diffs
CPU_ONLY_OPTIMIZERS = []

GPU_ONLY_OPTIMIZERS = [
    "adam",
    "lamb",
    "lars_sgd",
    "partial_rowwise_adam",
    "partial_rowwise_lamb",
    "none",
    "rowwise_adagrad_with_counter",
]

DEPRECATED_OPTIMIZERS = [
    "approx_sgd",
    "approx_rowwise_adagrad",
    "approx_rowwise_adagrad_with_counter",
    "approx_rowwise_adagrad_with_weight_decay",
    "rowwise_adagrad_with_weight_decay",
    "rowwise_weighted_adagrad",
]

# Optimizers with the global_weight_decay support
GWD_OPTIMIZERS = [
    "rowwise_adagrad",
]

SSD_OPTIMIZERS = [
    "rowwise_adagrad",
]

ALL_OPTIMIZERS = (
    COMMON_OPTIMIZERS
    + CPU_ONLY_OPTIMIZERS
    + GPU_ONLY_OPTIMIZERS
    + DEPRECATED_OPTIMIZERS
)

CPU_OPTIMIZERS = COMMON_OPTIMIZERS + CPU_ONLY_OPTIMIZERS

GPU_OPTIMIZERS = COMMON_OPTIMIZERS + GPU_ONLY_OPTIMIZERS

# Optimizers with the VBE support
VBE_OPTIMIZERS = [
    "rowwise_adagrad",
    "rowwise_adagrad_with_counter",
    "sgd",
    "dense",
    "adam",
]

# Individual optimizers (not fused with SplitTBE backward)
DEFUSED_OPTIMIZERS = [
    "rowwise_adagrad",
]

WEIGHT_OPTIONS = [
    "weighted",
    "unweighted_nobag",
    "unweighted",
]

PARTIAL_WEIGHT_OPTIONS = [
    "weighted",
    "unweighted",
]

DENSE_OPTIONS = [
    "split",
    "dense",
    "ssd",
]

################################################################################
# C++ Inference Code
################################################################################

static_cpu_files_inference = [
    "codegen/inference/embedding_forward_quantized_host_cpu.cpp",
]

static_gpu_files_inference = [
    "codegen/inference/embedding_forward_quantized_host.cpp",
    "codegen/inference/embedding_forward_quantized_split_lookup.cu",
]

gen_cpu_files_inference = [
    "gen_embedding_forward_quantized_unweighted_codegen_cpu.cpp",
    "gen_embedding_forward_quantized_weighted_codegen_cpu.cpp",
]

gen_gpu_files_inference = [
    "gen_embedding_forward_quantized_split_nbit_kernel_{}_{}_codegen_cuda.cu".format(
        wdesc,
        etype,
    )
    for wdesc in WEIGHT_OPTIONS
    for etype in [
        "fp32",
        "fp16",
        "fp8",
        "int8",
        "int4",
        "int2",
    ]
] + [
    fstring.format(wdesc)
    for wdesc in WEIGHT_OPTIONS
    for fstring in [
        "gen_embedding_forward_quantized_split_nbit_host_{}_codegen_cuda.cu",
    ]
]

################################################################################
# C++ Training Code - Fused Optimizer
################################################################################

gen_fused_optim_header_files = (
    [
        "gen_embedding_optimizer_{}_{}_device_kernel.cuh".format(
            optimizer,
            "ssd" if ssd else "split",
        )
        for ssd in [
            True,
            False,
        ]
        for optimizer in (SSD_OPTIMIZERS if ssd else GPU_OPTIMIZERS)
    ]
    + [
        "gen_embedding_backward_split_{}{}{}_device_kernel.cuh".format(
            "weighted" if weighted else "unweighted",
            "_nobag" if nobag else "",
            "_vbe" if vbe else "",
        )
        for nobag in [
            True,
            False,
        ]
        for weighted in (
            [
                True,
                False,
            ]
            if not nobag
            else [False]
        )
        for vbe in (
            [
                True,
                False,
            ]
            if not nobag
            else [False]
        )
    ]
    + [
        "gen_embedding_backward_split_{}{}_device_kernel_hip.hip".format(
            "weighted" if weighted else "unweighted",
            "_nobag" if nobag else "",
        )
        for nobag in [
            True,
            False,
        ]
        for weighted in (
            [
                True,
                False,
            ]
            if not nobag
            else [False]
        )
    ]
    + [
        "gen_embedding_backward_split_common_device_kernel.cuh",
    ]
    + [
        "pt2_arg_utils.h",
    ]
)

gen_defused_optim_templates = [
    "codegen/training/optimizer/embedding_optimizer_split_device_kernel_template.cuh",
    "codegen/training/optimizer/embedding_optimizer_split_host_template.cpp",
    "codegen/training/optimizer/embedding_optimizer_split_kernel_template.cu",
    "codegen/training/optimizer/embedding_optimizer_split_template.cu",
    "codegen/training/python/split_embedding_optimizer_codegen.template",
    "codegen/training/python/optimizer_args.py",
]

gen_defused_optim_header_files = [
    "gen_embedding_optimizer_{}_split_device_kernel.cuh".format(optimizer)
    for optimizer in DEFUSED_OPTIMIZERS
]

gen_defused_optim_src_files = [
    fstring.format(optimizer)
    for fstring in [
        "gen_embedding_optimizer_{}_split.cpp",
        "gen_embedding_optimizer_{}_split_cuda.cu",
        "gen_embedding_optimizer_{}_split_kernel.cu",
    ]
    for optimizer in DEFUSED_OPTIMIZERS
]

gen_py_files_defused_optim = [
    "split_embedding_optimizer_{}.py".format(optimizer)
    for optimizer in DEFUSED_OPTIMIZERS
] + ["optimizer_args.py"]

################################################################################
# C++ Training Code - Forward Split
################################################################################

gen_cpu_files_forward_split = ["gen_embedding_forward_split_pt2_cpu_wrapper.cpp"]

gen_gpu_files_forward_split = (
    [
        fstring.format(wdesc)
        for wdesc in WEIGHT_OPTIONS
        for fstring in [
            "gen_embedding_forward_split_{}_kernel.cu",
            "gen_embedding_forward_dense_{}_kernel.cu",
            "gen_embedding_forward_ssd_{}_kernel.cu",
        ]
    ]
    + [
        fstring.format(desc, wdesc)
        for desc in DENSE_OPTIONS
        for wdesc in PARTIAL_WEIGHT_OPTIONS
        for fstring in [
            "gen_embedding_forward_{}_{}_codegen_cuda.cu",
            "gen_embedding_forward_{}_{}_codegen_meta.cpp",
        ]
    ]
    + [
        fstring.format(wdesc)
        for wdesc in PARTIAL_WEIGHT_OPTIONS
        for fstring in [
            "gen_embedding_forward_split_{}_vbe_codegen_cuda.cu",
            "gen_embedding_forward_split_{}_vbe_codegen_meta.cpp",
            "gen_embedding_forward_split_{}_vbe_kernel.cu",
            "gen_embedding_forward_split_{}_v2_kernel.cu",
            "gen_embedding_forward_split_{}_gwd_codegen_cuda.cu",
            "gen_embedding_forward_split_{}_vbe_gwd_codegen_cuda.cu",
            "gen_embedding_forward_split_{}_gwd_kernel.cu",
            "gen_embedding_forward_dense_{}_vbe_codegen_cuda.cu",
            "gen_embedding_forward_dense_{}_vbe_kernel.cu",
            "gen_embedding_forward_split_{}_vbe_gwd_kernel.cu",
            "gen_embedding_forward_ssd_{}_vbe_codegen_cuda.cu",
            "gen_embedding_forward_ssd_{}_vbe_codegen_meta.cpp",
            "gen_embedding_forward_ssd_{}_vbe_kernel.cu",
        ]
    ]
    + [
        "gen_embedding_forward_{}_unweighted_nobag_kernel_small.cu".format(desc)
        for desc in DENSE_OPTIONS
    ]
    + [
        "gen_embedding_forward_split_pt2_cuda_wrapper.cpp",
        "gen_embedding_forward_ssd_pt2_cuda_wrapper.cpp",
    ]
)

################################################################################
# C++ Training Code - Index Select
################################################################################

gen_gpu_files_index_select = [
    "gen_batch_index_select_dim0_forward_codegen_cuda.cu",
    "gen_batch_index_select_dim0_forward_kernel.cu",
    "gen_batch_index_select_dim0_forward_kernel_small.cu",
    "gen_batch_index_select_dim0_backward_codegen_cuda.cu",
    "gen_batch_index_select_dim0_backward_kernel_cta.cu",
    "gen_batch_index_select_dim0_backward_kernel_warp.cu",
    "gen_embedding_backward_split_grad_index_select.cu",
]

gen_index_select_header_files = [
    "gen_embedding_backward_split_common_device_kernel.cuh",
    "gen_embedding_backward_split_batch_index_select_device_kernel.cuh",
]

static_cpu_files_index_select = [
    "codegen/training/index_select/batch_index_select_dim0_cpu_host.cpp",
    "codegen/training/index_select/batch_index_select_dim0_ops.cpp",
]

static_gpu_files_index_select = [
    "codegen/training/index_select/batch_index_select_dim0_host.cpp",
]

################################################################################
# C++ Training Code - Backward Split
################################################################################

static_cpu_files_training = [
    "codegen/training/backward/embedding_backward_dense_host_cpu.cpp",
]

static_cpu_files_common = [
    "codegen/utils/embedding_bounds_check_host_cpu.cpp",
    "codegen/training/forward/embedding_forward_split_cpu.cpp",
    "codegen/training/pt2/pt2_autograd_utils.cpp",
]

static_gpu_files_common = [
    "codegen/utils/embedding_bounds_check_v1.cu",
    "codegen/utils/embedding_bounds_check_v2.cu",
    "codegen/utils/embedding_bounds_check_host.cpp",
]

gen_cpu_files_training = (
    [
        "gen_embedding_backward_dense_split_cpu.cpp",
    ]
    + [
        "gen_embedding_backward_split_{}_cpu.cpp".format(optimizer)
        for optimizer in ALL_OPTIMIZERS
    ]
    + [
        "gen_embedding_backward_{}_split_cpu.cpp".format(optimizer)
        for optimizer in CPU_OPTIMIZERS
    ]
)

gen_cpu_files_training_pt2 = (
    [
        "gen_embedding_split_{}_pt2_autograd.cpp".format(optimizer)
        for optimizer in ALL_OPTIMIZERS
    ]
    + [
        "gen_embedding_ssd_{}_pt2_autograd.cpp".format(optimizer)
        for optimizer in SSD_OPTIMIZERS
    ]
    + [
        "gen_embedding_backward_split_{}_pt2_cpu_wrapper.cpp".format(optimizer)
        for optimizer in ALL_OPTIMIZERS
    ]
)

gen_gpu_files_training_pt2 = [
    "gen_embedding_backward_split_{}_pt2_cuda_wrapper.cpp".format(optimizer)
    for optimizer in ALL_OPTIMIZERS
] + [
    "gen_embedding_backward_ssd_{}_pt2_cuda_wrapper.cpp".format(optimizer)
    for optimizer in SSD_OPTIMIZERS
]

gen_gpu_files_training_dense = [
    # Dense host and kernel, and forward-quantized host src files
    fstring.format(wdesc)
    for wdesc in WEIGHT_OPTIONS
    for fstring in [
        "gen_embedding_backward_dense_split_{}_cuda.cu",
        "gen_embedding_backward_dense_split_{}_meta.cpp",
        "gen_embedding_backward_dense_split_{}_kernel_cta.cu",
        "gen_embedding_backward_dense_split_{}_kernel_warp.cu",
    ]
] + [
    "gen_embedding_backward_split_dense.cpp",
]

gen_gpu_files_training_split_host = (
    [
        "gen_embedding_backward_split_{}.cpp".format(optimizer)
        for optimizer in ALL_OPTIMIZERS
    ]
    + [
        "gen_embedding_backward_ssd_{}.cpp".format(optimizer)
        for optimizer in SSD_OPTIMIZERS
    ]
    + [
        "gen_embedding_backward_{}_split_{}_meta.cpp".format(optimizer, wdesc)
        for optimizer in GPU_OPTIMIZERS
        for wdesc in [
            "weighted",
            "unweighted",
        ]
    ]
)

gen_gpu_files_training_gwd = [
    fstring.format(optimizer, wdesc)
    for optimizer in GWD_OPTIMIZERS
    for wdesc in PARTIAL_WEIGHT_OPTIONS
    for fstring in [
        "gen_embedding_backward_{}_split_{}_gwd_cuda.cu",
        "gen_embedding_backward_{}_split_{}_gwd_kernel_cta.cu",
        "gen_embedding_backward_{}_split_{}_gwd_kernel_warp.cu",
    ]
] + [
    fstring.format(optimizer, wdesc)
    for optimizer in VBE_OPTIMIZERS
    for wdesc in PARTIAL_WEIGHT_OPTIONS
    for fstring in (
        [
            "gen_embedding_backward_{}_split_{}_vbe_gwd_cuda.cu",
            "gen_embedding_backward_{}_split_{}_vbe_gwd_kernel_cta.cu",
            "gen_embedding_backward_{}_split_{}_vbe_gwd_kernel_warp.cu",
        ]
        if optimizer in GWD_OPTIMIZERS
        else []
    )
]

gen_gpu_files_training_vbe = [
    fstring.format(optimizer, wdesc)
    for optimizer in VBE_OPTIMIZERS
    for wdesc in PARTIAL_WEIGHT_OPTIONS
    for fstring in [
        "gen_embedding_backward_{}_split_{}_vbe_meta.cpp",
    ]
    + (
        [
            "gen_embedding_backward_{}_ssd_{}_vbe_meta.cpp",
        ]
        if optimizer in SSD_OPTIMIZERS
        else []
    )
] + [
    fstring.format(optimizer, wdesc)
    for optimizer in VBE_OPTIMIZERS
    for wdesc in PARTIAL_WEIGHT_OPTIONS
    for fstring in [
        "gen_embedding_backward_{}_split_{}_vbe_cuda.cu",
        "gen_embedding_backward_{}_split_{}_vbe_kernel_cta.cu",
        "gen_embedding_backward_{}_split_{}_vbe_kernel_warp.cu",
    ]
    + (
        [
            "gen_embedding_backward_{}_ssd_{}_vbe_cuda.cu",
            "gen_embedding_backward_{}_ssd_{}_vbe_kernel_cta.cu",
            "gen_embedding_backward_{}_ssd_{}_vbe_kernel_warp.cu",
        ]
        if optimizer in SSD_OPTIMIZERS
        else []
    )
]

gen_gpu_files_training = (
    [
        "gen_embedding_backward_split_grad_embedding_ops.cu",
    ]
    + [
        # Backward-split positional weights and forward src files
        fstring.format(desc)
        for desc in DENSE_OPTIONS
        for fstring in [
            "gen_embedding_backward_{}_indice_weights_codegen_cuda.cu",
        ]
    ]
    + [
        fstring.format(
            optimizer,
            "ssd" if ssd else "split",
            wdesc,
        )
        for ssd in [
            True,
            False,
        ]
        for optimizer in (SSD_OPTIMIZERS if ssd else GPU_OPTIMIZERS)
        for wdesc in WEIGHT_OPTIONS
        for fstring in [
            "gen_embedding_backward_{}_{}_{}_cuda.cu",
            "gen_embedding_backward_{}_{}_{}_kernel_cta.cu",
            "gen_embedding_backward_{}_{}_{}_kernel_warp.cu",
        ]
    ]
)

gen_hip_files_training = [
    "gen_embedding_backward_split_{}{}_device_kernel_hip.hip".format(
        "weighted" if weighted else "unweighted",
        "_nobag" if nobag else "",
    )
    for nobag in [
        True,
        False,
    ]
    for weighted in (
        [
            True,
            False,
        ]
        if not nobag
        else [False]
    )
]

################################################################################
# Python Training Code
################################################################################

gen_py_files_training = (
    [
        fstring.format(optimizer)
        for optimizer in COMMON_OPTIMIZERS + CPU_ONLY_OPTIMIZERS + GPU_ONLY_OPTIMIZERS
        for fstring in [
            "lookup_{}.py",
        ]
    ]
    + [
        fstring.format(optimizer)
        for optimizer in SSD_OPTIMIZERS
        for fstring in [
            "lookup_{}_ssd.py",
        ]
    ]
    + [
        "__init__.py",
        "lookup_args.py",
        "lookup_args_ssd.py",
    ]
)

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import shutil
import sysconfig
import sys
import re
import tempfile

from codegen.embedding_backward_code_generator import emb_codegen
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
sys.path.append("..")
from third_party.hipify_torch.hipify import hipify_python

cpu_only_build = False
cur_dir = os.path.dirname(os.path.realpath(__file__))

cub_include_path = os.getenv("CUB_DIR", None)
if cub_include_path is None:
    print(
        "CUDA CUB directory environment variable not set.  Using default CUB location."
    )
build_codegen_path = "build/codegen"
py_path = "python"

is_rocm_pytorch = False
maj_ver, min_ver, _ = torch.__version__.split('.')
if int(maj_ver) > 1 or (int(maj_ver) == 1 and int(min_ver) >= 5):
    from torch.utils.cpp_extension import ROCM_HOME
    is_rocm_pytorch = True \
            if ((torch.version.hip is not None) and (ROCM_HOME is not None)) \
            else False

# Get the long description from the relevant file
cur_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(cur_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
extra_compile_args += ["-mavx2", "-mf16c", "-mfma"]
if not is_rocm_pytorch:
    extra_compile_args += ["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl"]

OPTIMIZERS = [
    "adagrad",
    "adam",
    "approx_rowwise_adagrad",
    "approx_sgd",
    "lamb",
    "lars_sgd",
    "partial_rowwise_adam",
    "partial_rowwise_lamb",
    "rowwise_adagrad",
    "sgd",
    "rowwise_weighted_adagrad"
]

cpp_asmjit_files = glob.glob("../third_party/asmjit/src/asmjit/*/*.cpp")

cpp_fbgemm_files = [
    "../src/EmbeddingSpMDMAvx2.cc",
    "../src/EmbeddingSpMDM.cc",
    "../src/EmbeddingSpMDMNBit.cc",
    "../src/QuantUtils.cc",
    "../src/QuantUtilsAvx2.cc",
    "../src/RefImplementations.cc",
    "../src/RowWiseSparseAdagradFused.cc",
    "../src/SparseAdagrad.cc",
    "../src/Utils.cc",
]

if not is_rocm_pytorch:
    cpp_fbgemm_files += ["../src/EmbeddingSpMDMAvx512.cc"]

cpp_cpu_output_files = (
    [
        "gen_embedding_forward_quantized_unweighted_codegen_cpu.cpp",
        "gen_embedding_forward_quantized_weighted_codegen_cpu.cpp",
        "gen_embedding_backward_dense_split_cpu.cpp",
    ]
    + [
        "gen_embedding_backward_split_{}_cpu.cpp".format(optimizer)
        for optimizer in OPTIMIZERS
    ]
    + [
        "gen_embedding_backward_{}_split_cpu.cpp".format(optimizer)
        for optimizer in OPTIMIZERS
    ]
)

cpp_cuda_output_files = (
    [
        "gen_embedding_forward_dense_weighted_codegen_cuda.cu",
        "gen_embedding_forward_dense_unweighted_codegen_cuda.cu",
        "gen_embedding_forward_quantized_split_unweighted_codegen_cuda.cu",
        "gen_embedding_forward_quantized_split_weighted_codegen_cuda.cu",
        "gen_embedding_forward_split_weighted_codegen_cuda.cu",
        "gen_embedding_forward_split_unweighted_codegen_cuda.cu",
        "gen_embedding_backward_split_indice_weights_codegen_cuda.cu",
        "gen_embedding_backward_dense_indice_weights_codegen_cuda.cu",
        "gen_embedding_backward_dense_split_unweighted_cuda.cu",
        "gen_embedding_backward_dense_split_weighted_cuda.cu",
    ]
    + [
        "gen_embedding_backward_{}_split_{}_cuda.cu".format(optimizer, weighted)
        for optimizer in OPTIMIZERS
        for weighted in [
            "weighted",
            "unweighted",
        ]
    ]
    + [
        "gen_embedding_backward_split_{}.cpp".format(optimizer)
        for optimizer in OPTIMIZERS
    ]
)

py_output_files = ["lookup_{}.py".format(optimizer) for optimizer in OPTIMIZERS]


def generate_jinja_files():
    abs_build_path = os.path.join(cur_dir, build_codegen_path)
    if not os.path.exists(abs_build_path):
        os.makedirs(abs_build_path)
    emb_codegen(install_dir=abs_build_path, is_fbcode=False)

    dst_python_path = os.path.join(cur_dir, py_path)
    if not os.path.exists(dst_python_path):
        os.makedirs(dst_python_path)
    for filename in py_output_files:
        shutil.copy2(os.path.join(abs_build_path, filename), dst_python_path)
    shutil.copy2(os.path.join(cur_dir, "codegen", "lookup_args.py"), dst_python_path)


class FBGEMM_GPU_BuildExtension(BuildExtension.with_options(no_python_abi_suffix=True)):
    def build_extension(self, ext):
        if not is_rocm_pytorch:
            generate_jinja_files()
        else:
            with hipify_python.GeneratedFileCleaner(keep_intermediates=True) as clean_ctx:
                hipify_python.hipify(
                        project_directory=cur_dir,
                        output_directory=cur_dir,
                        includes="codegen/*",
                        show_detailed=True,
                        is_pytorch_extension=True,
                        clean_ctx=clean_ctx)

            def replace_pattern(hip_file, pattern_map):
                patterns = {}
                for regexp in pattern_map:
                    patterns[regexp] = re.compile(regexp.format(exclude=""))
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
                   with open(hip_file) as src_file:
                        for line in src_file:
                            for regexp in pattern_map:
                                pattern = pattern_map[regexp]
                                exclude = pattern[0]
                                replacement = pattern[1]
                                in_regexp = regexp.format(exclude="")
                                if len(pattern_map[regexp]) == 4:
                                    all_ori = pattern[2]
                                    all_new = pattern[3]
                                else:
                                    all_ori = None
                                    all_new = None
                                if re.search(in_regexp, line) and \
                                        (exclude is None or not re.search(regexp.format(exclude=exclude), line)):
                                    ori = line
                                    if all_ori is not None and all_ori in line:
                                        line = line.replace(all_ori, all_new)
                                    else:
                                        line = patterns[regexp].sub(replacement, line)

                            tmp_file.write(line)

                shutil.copystat(hip_file, tmp_file.name)
                shutil.move(tmp_file.name, hip_file)

            def post_hipify(hip_file):
                replace_pattern(hip_file, {"(#include.*\"codegen.*){exclude}[.]cuh": ["_hip", "\\1_hip.cuh"],
                    "{exclude}cub(::DeviceRunLengthEncode)": ["hip", "hipcub\\1"],
                    "(#include.*[<\"].*){exclude}cub(.*)[.]cuh": ["hip", "\\1hipcub\\2.hpp"],
                    "(#include.*[<\"]fbgemm_gpu.*)({exclude}[.]cuh)": ["_hip", "\\1_hip\\2", "cuda", "hip"],
                    "cudaCpuDeviceId": [None, "hipCpuDeviceId"],
                    "split_embeddings_utils[.]cuh": [None, "split_embeddings_utils_hip.cuh"]})

            abs_build_path = os.path.join(cur_dir, build_codegen_path)
            for f in cpp_cuda_output_files:
                if f.endswith(".cu"):
                    hip_f = os.path.join(abs_build_path, f.replace("cuda", "hip").replace(".cu", ".hip"))
                    post_hipify(hip_f)

            for s in ["codegen", "src"]:
                for f in os.listdir(s):
                    if f.endswith(".hip") or f.endswith("hip.cuh"):
                        hip_f = os.path.join(s, f)
                        post_hipify(hip_f)

            os.system("hipify-perl src/split_embeddings_utils.cuh > src/split_embeddings_utils_hip.cuh")
            post_hipify("src/split_embeddings_utils_hip.cuh")

        super().build_extension(ext)

if is_rocm_pytorch:
    generate_jinja_files()
    rocm_include_dirs = ["/opt/rocm/include/hiprand", "/opt/rocm/include/rocrand"]
    libraries = []
else:
    rocm_include_dirs = []
    libraries = ["nvidia-ml"]

include_dirs = [ cur_dir,
                 os.path.join(cur_dir, "include"),
                 os.path.join(cur_dir, "include/fbgemm_gpu"),
               ] + rocm_include_dirs

if cub_include_path is not None:
    include_dirs += [cub_include_path]

# Handle command line args before passing to main setup() method.
if "--cpu_only" in sys.argv:
    cpu_only_build = True
    sys.argv.remove("--cpu_only")

setup(
    name="fbgemm_gpu",
    install_requires=[
        "torch",
        "Jinja2",
        "click",
        "hypothesis",
    ],
    version="0.0.1",
    long_description=long_description,
    ext_modules=[
        CUDAExtension(
            name="fbgemm_gpu_py",
            sources=[
                os.path.join(cur_dir, build_codegen_path, "{}".format(f))
                for f in cpp_cuda_output_files + cpp_cpu_output_files
            ]
            + cpp_asmjit_files
            + cpp_fbgemm_files
            + [
                os.path.join(cur_dir, "codegen/embedding_forward_split_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_forward_quantized_host_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_forward_quantized_host.cpp"),
                os.path.join(cur_dir, "codegen/embedding_backward_dense_host_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_backward_dense_host.cpp"),
                os.path.join(cur_dir, "codegen/embedding_bounds_check_host.cpp"),
                os.path.join(cur_dir, "codegen/embedding_bounds_check_host_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_bounds_check.cu"),
                os.path.join(cur_dir, "src/split_embeddings_cache_cuda.cu"),
                os.path.join(cur_dir, "src/split_table_batched_embeddings.cpp"),
                os.path.join(cur_dir, "src/cumem_utils.cu"),
                os.path.join(cur_dir, "src/cumem_utils_host.cpp"),
                os.path.join(cur_dir, "src/quantize_ops_cpu.cpp"),
                os.path.join(cur_dir, "src/quantize_ops_gpu.cpp"),
                os.path.join(cur_dir, "src/quantize_ops.cu"),
                os.path.join(cur_dir, "src/cpu_utils.cpp"),
                os.path.join(cur_dir, "src/sparse_ops_cpu.cpp"),
                os.path.join(cur_dir, "src/sparse_ops_gpu.cpp"),
                os.path.join(cur_dir, "src/sparse_ops.cu"),
                os.path.join(cur_dir, "src/merge_pooled_embeddings_gpu.cpp"),
                os.path.join(cur_dir, "src/permute_pooled_embedding_ops.cu"),
                os.path.join(cur_dir, "src/permute_pooled_embedding_ops_gpu.cpp"),
                os.path.join(cur_dir, "src/layout_transform_ops_cpu.cpp"),
                os.path.join(cur_dir, "src/layout_transform_ops_gpu.cpp"),
                os.path.join(cur_dir, "src/layout_transform_ops.cu"),
                os.path.join(cur_dir, "src/jagged_tensor_ops.cu"),
                os.path.join(cur_dir, "src/histogram_binning_calibration_ops.cu"),
                os.path.join(cur_dir, "src/split_embeddings_utils.cu"),
            ],
            include_dirs=[
                cur_dir,
                os.path.join(cur_dir, "include"),
                os.path.join(cur_dir, "../include"),
                os.path.join(cur_dir, "../src"),
                os.path.join(cur_dir, "../third_party/asmjit/src"),
                os.path.join(cur_dir, "../third_party/asmjit/src/core"),
                os.path.join(cur_dir, "../third_party/asmjit/src/x86"),
                os.path.join(cur_dir, "../third_party/cpuinfo/include"),
            ] + include_dirs,
            extra_compile_args={"cxx": extra_compile_args + ["-DFBGEMM_GPU_WITH_CUDA"],
                                "nvcc": ["-U__CUDA_NO_HALF_CONVERSIONS__"]},
            libraries=libraries,
        ) if not cpu_only_build else
        CppExtension(
            name="fbgemm_gpu_py",
            sources=[
                os.path.join(cur_dir, build_codegen_path, "{}".format(f))
                for f in cpp_cpu_output_files
            ]
            + cpp_asmjit_files
            + cpp_fbgemm_files
            + [
                os.path.join(cur_dir, "codegen/embedding_forward_split_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_forward_quantized_host_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_backward_dense_host_cpu.cpp"),
            ],
            include_dirs=[
                cur_dir,
                os.path.join(cur_dir, "include"),
                os.path.join(cur_dir, "../include"),
                os.path.join(cur_dir, "../src"),
                os.path.join(cur_dir, "../third_party/asmjit/src"),
                os.path.join(cur_dir, "../third_party/asmjit/src/core"),
                os.path.join(cur_dir, "../third_party/asmjit/src/x86"),
                os.path.join(cur_dir, "../third_party/cpuinfo/include"),
            ],
            extra_compile_args={"cxx": extra_compile_args},
        )
    ],
    cmdclass={"build_ext": FBGEMM_GPU_BuildExtension},
)

# Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

import sys
import warnings
import os
import copy
import re
import shutil
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
import itertools
import glob

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from src.generate_kernels import generate_kernels_ampere, generate_kernels_hopper

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "fbgemm_gpu_hstu"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("HSTU_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("HSTU_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("HSTU_FORCE_CXX11_ABI", "FALSE") == "TRUE"

DISABLE_BACKWARD = os.getenv("HSTU_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_DETERMINISTIC = os.getenv("HSTU_DISABLE_DETERMINISTIC", "TRUE") == "TRUE"
DISABLE_LOCAL = os.getenv("HSTU_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_CAUSAL = os.getenv("HSTU_DISABLE_CAUSAL", "FALSE") == "TRUE"
DISABLE_CONTEXT = os.getenv("HSTU_DISABLE_CONTEXT", "FALSE") == "TRUE"
DISABLE_TARGET = os.getenv("HSTU_DISABLE_TARGET", "FALSE") == "TRUE"
DISABLE_ARBITRARY = os.getenv("HSTU_DISABLE_ARBITRARY", "FALSE") == "TRUE"
ARBITRARY_NFUNC = int(os.getenv("HSTU_ARBITRARY_NFUNC", "0"))
DISABLE_RAB = os.getenv("HSTU_DISABLE_RAB", "FALSE") == "TRUE"
DISABLE_DRAB = os.getenv("HSTU_DISABLE_DRAB", "FALSE") == "TRUE"
DISABLE_BF16 = os.getenv("HSTU_DISABLE_BF16", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("HSTU_DISABLE_FP16", "TRUE") == "TRUE"
DISABLE_FP8 = os.getenv("HSTU_DISABLE_FP8", "FALSE") == "TRUE"
USE_E5M2_BWD = os.getenv("HSTU_USE_E5M2_BWD", "FALSE") == "TRUE"
DISABLE_HDIM32 = os.getenv("HSTU_DISABLE_HDIM32", "FALSE") == "TRUE"
DISABLE_HDIM64 = os.getenv("HSTU_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("HSTU_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("HSTU_DISABLE_HDIM256", "FALSE") == "TRUE"
DISABLE_86OR89 = os.getenv("HSTU_DISABLE_86OR89", "TRUE") == "TRUE"
arch_list = os.getenv("HSTU_ARCH_LIST", "8.0 9.0")

ONLY_COMPILE_SO = os.getenv("HSTU_ONLY_COMPILE_SO", "FALSE") == "TRUE"

if ONLY_COMPILE_SO:
    CUDA_HOME = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if CUDA_HOME is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        CUDA_HOME = os.path.dirname(os.path.dirname(nvcc_path))

    COMMON_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
        '--expt-relaxed-constexpr',
        '--compiler-options', "'-fPIC'"
    ]

    # a navie/minial way to build a cuda so, should only work under unix and may have differences compared to torch's BuildExtension
    def CUDAExtension(name, sources, *args, **kwargs):
        library_dirs = kwargs.get('library_dirs', [])
        library_dirs.append(os.path.join(CUDA_HOME, 'lib64'))
        kwargs['library_dirs'] = library_dirs
        libraries = kwargs.get('libraries', [])
        kwargs['libraries'] = libraries

        include_dirs = kwargs.get('include_dirs', [])
        include_dirs.append(os.path.join(CUDA_HOME, 'include'))
        kwargs['include_dirs'] = include_dirs

        kwargs['language'] = 'c++'
        return Extension(name, sources, *args, **kwargs)

    class BuildExtension(build_ext):
        def build_extensions(self):
            self.compiler.src_extensions += ['.cu', '.cuh']
            original_compile = self.compiler._compile

            def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
                # Copy before we make any modifications.
                cflags = copy.deepcopy(extra_postargs)
                try:
                    original_compiler = self.compiler.compiler_so
                    if src.endswith(".cu"):
                        nvcc = [os.path.join(CUDA_HOME, 'bin', 'nvcc')]
                        self.compiler.set_executable('compiler_so', nvcc)
                        if isinstance(cflags, dict):
                            cflags = COMMON_NVCC_FLAGS + cflags['nvcc']
                    elif isinstance(cflags, dict):
                        cflags = cflags['cxx']
                    cflags += ["-std=c++17"]

                    original_compile(obj, src, ext, cc_args, cflags, pp_opts)
                finally:
                    # Put the original compiler back in place.
                    self.compiler.set_executable(
                        'compiler_so', original_compiler)

            self.compiler._compile = unix_wrap_single_compile
            super().build_extensions()
else:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return ["--threads", nvcc_threads]


cmdclass = {}
ext_modules = []

cmdclass = []
install_requires = []

if not SKIP_CUDA_BUILD:
    if not ONLY_COMPILE_SO:
        print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("--hstu")
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.3"):
        raise RuntimeError("HSTU is only supported on CUDA 12.3 and above")

    cc_flag = []
    cc_flag.append("-gencode")
    if "9.0" in arch_list:
        cc_flag.append("arch=compute_90a,code=sm_90a")
    if "8.0" in arch_list:
        cc_flag.append("arch=compute_80,code=sm_80")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(this_dir).parent.parent.parent
    cutlass_dir = repo_dir / "external" / "cutlass"

    feature_args = (
        []
        + (["-DHSTU_DISABLE_BACKWARD"] if DISABLE_BACKWARD else [])
        + (["-DHSTU_DISABLE_DETERMINISTIC"] if DISABLE_DETERMINISTIC else [])
        + (["-DHSTU_DISABLE_LOCAL"] if DISABLE_LOCAL else [])
        + (["-DHSTU_DISABLE_CAUSAL"] if DISABLE_CAUSAL else [])
        + (["-DHSTU_DISABLE_CONTEXT"] if DISABLE_CONTEXT else [])
        + (["-DHSTU_DISABLE_TARGET"] if DISABLE_TARGET else [])
        + (["-DHSTU_DISABLE_ARBITRARY"] if DISABLE_ARBITRARY else [])
        + (["-DHSTU_ARBITRARY_NFUNC=" + str(ARBITRARY_NFUNC)])
        + (["-DHSTU_DISABLE_RAB"] if DISABLE_RAB else [])
        + (["-DHSTU_DISABLE_DRAB"] if DISABLE_DRAB else [])
        + (["-DHSTU_DISABLE_BF16"] if DISABLE_BF16 else [])
        + (["-DHSTU_DISABLE_FP16"] if DISABLE_FP16 else [])
        + (["-DHSTU_DISABLE_FP8"] if DISABLE_FP8 else [])
        + (["-DHSTU_USE_E5M2_BWD"] if USE_E5M2_BWD else [])
        + (["-DHSTU_DISABLE_HDIM32"] if DISABLE_HDIM32 else [])
        + (["-DHSTU_DISABLE_HDIM64"] if DISABLE_HDIM64 else [])
        + (["-DHSTU_DISABLE_HDIM128"] if DISABLE_HDIM128 else [])
        + (["-DHSTU_DISABLE_HDIM256"] if DISABLE_HDIM256 else [])
        + (["-DHSTU_DISABLE_86OR89"] if DISABLE_86OR89 else [])
    )

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
    if "8.0" not in arch_list and "9.0" not in arch_list:
        raise ValueError("At least one of 8.0 or 9.0 must be in arch_list")

    torch_cpp_sources = []
    subprocess.run(["rm", "-rf", "src/hstu_ampere/instantiations/*"])
    subprocess.run(["rm", "-rf", "src/hstu_hopper/instantiations/*"])
    if "8.0" in arch_list:
        torch_cpp_sources.append("src/hstu_ampere/hstu_ops_gpu.cpp")
        generate_kernels_ampere("src/hstu_ampere/instantiations")
    if "9.0" in arch_list:
        torch_cpp_sources.append("src/hstu_hopper/hstu_ops_gpu.cpp")
        generate_kernels_hopper("src/hstu_hopper/instantiations")
    cuda_sources = (glob.glob("src/hstu_ampere/instantiations/*.cu") if "8.0" in arch_list else []) + (glob.glob("src/hstu_hopper/instantiations/*.cu") if "9.0" in arch_list else [])

    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",  # printing out number of registers
        # "--resource-usage",  # printing out number of registers
        # "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-lineinfo",
    ]
    if get_platform() == "win_amd64":
        nvcc_flags.extend(
            [
                "-D_USE_MATH_DEFINES",  # for M_LN2
                "-Xcompiler=/Zc:__cplusplus",  # sets __cplusplus correctly, CUTLASS_CONSTEXPR_IF_CXX17 needed for cutlass::gcd
            ]
        )
    include_dirs = [
        cutlass_dir / "include",
    ]
    if "8.0" in arch_list:
        include_dirs.append(Path(this_dir) / "src" / "hstu_ampere")
    if "9.0" in arch_list:
        include_dirs.append(Path(this_dir) / "src" / "hstu_hopper")

    sources = None
    if ONLY_COMPILE_SO:
        sources = cuda_sources
    else:
        sources = torch_cpp_sources + cuda_sources

    ext_modules.append(
        CUDAExtension(
            name="hstu.fbgemm_gpu_experimental_hstu",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + feature_args,
                "nvcc": nvcc_threads_args() + nvcc_flags + cc_flag + feature_args,
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (
                1024**3
            )  # free memory in GB
            max_num_jobs_memory = int(
                free_memory_gb / 9
            )  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

    def build_extensions(self):
        super().build_extensions()
        import sysconfig
        for ext in self.extensions:
            ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
            if ext_suffix and ext_suffix != '.so':
                ext_path = self.get_ext_fullpath(ext.name)
                simple_path = ext_path.replace(ext_suffix, '.so')
                if os.path.exists(ext_path) and ext_path != simple_path:
                    shutil.copy2(ext_path, simple_path)

setup(
    name=PACKAGE_NAME,
    version="0.1.0" + '+cu' + str(bare_metal_version),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    author="NVIDIA-DevTech",
    py_modules=["cuda_hstu_attention"],
    description="HSTU Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.8",
    install_requires=[],
)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import random
import re
import subprocess
import sys

from datetime import date
from typing import List, Optional

import setuptools_git_versioning as gitversion
import torch
from skbuild import setup


def generate_package_version(package_name: str):
    print("[SETUP.PY] Generating the package version ...")

    if "nightly" in package_name:
        # Use date stamp for nightly versions
        print("[SETUP.PY] Package is for NIGHTLY; using timestamp for the versioning")
        today = date.today()
        version = f"{today.year}.{today.month}.{today.day}"

    elif "test" in package_name:
        # Use date stamp for nightly versions
        print("[SETUP.PY] Package is for TEST: using random number for the versioning")
        version = (f"0.0.{random.randint(0, 1000)}",)

    else:
        # Use git tag / branch / commit info to generate a PEP-440-compliant version string
        print("[SETUP.PY] Package is for RELEASE: using git info for the versioning")
        print(
            f"[SETUP.PY] TAG: {gitversion.get_tag()}, BRANCH: {gitversion.get_branch()}, SHA: {gitversion.get_sha()}"
        )
        # Remove the local version identifier, if any (e.g. 0.4.0rc0.post0+git.6a63116c.dirty => 0.4.0rc0.post0)
        # Then remove post0 (keep postN for N > 0) (e.g. 0.4.0rc0.post0 => 0.4.0rc0)
        version = re.sub(".post0$", "", gitversion.version_from_git().split("+")[0])

    print(f"[SETUP.PY] Setting the package version: {version}")
    return version


def get_cxx11_abi():
    try:
        import torch

        value = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except ImportError:
        value = 0
    return "-DGLIBCXX_USE_CXX11_ABI=" + str(value)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fbgemm_gpu setup")
    parser.add_argument(
        "--cpu_only",
        dest="cpu_only",
        action="store_true",
        help="build for cpu-only (no GPU support)",
    )
    parser.add_argument(
        "--package_name",
        type=str,
        default="fbgemm_gpu",
        help="the name of this output wheel",
    )
    parser.add_argument(
        "--nvml_lib_path",
        type=str,
        default=None,
        help="Certain operations require the nvml lib (libnvidia-ml.so). If you installed"
        " this in a custom location (through cudatoolkit-dev), provide the path here.",
    )
    return parser.parse_known_args(argv)


def nvcc_ok(cuda_home: str, major: int, minor: int) -> bool:
    if not cuda_home:
        return False

    nvcc_path = f"{cuda_home}/bin/nvcc"
    if not os.path.exists(nvcc_path):
        return False

    try:
        # Extract version from version string - inspired my NVIDIA/apex
        output = subprocess.check_output([nvcc_path, "-V"], text=True)
        fragments = output.split()
        version = fragments[fragments.index("release") + 1]
        version_fragments = version.split(".")
        major_nvcc = int(version_fragments[0])
        minor_nvcc = int(version_fragments[1].split(",")[0])
        result = major == major_nvcc and minor == minor_nvcc
    except BaseException:
        result = False

    return result


def find_cuda(major: int, minor: int) -> Optional[str]:
    cuda_home = os.environ.get("CUDA_BIN_PATH")
    if nvcc_ok(cuda_home, major, minor):
        return cuda_home

    cuda_nvcc = os.environ.get("CUDACXX")

    if cuda_nvcc and os.path.exists(cuda_nvcc):
        cuda_home = os.path.dirname(os.path.dirname(cuda_nvcc))
        if nvcc_ok(cuda_home, major, minor):
            return cuda_home

    # Search standard installation location with version first
    cuda_home = f"/usr/local/cuda-{major}.{minor}"
    if nvcc_ok(cuda_home, major, minor):
        return cuda_home

    cuda_home = "/usr/local/cuda"
    if nvcc_ok(cuda_home, major, minor):
        return cuda_home

    try:
        # Try to find nvcc with which
        with open(os.devnull, "w") as devnull:
            nvcc = (
                subprocess.check_output(["which", "nvcc"], stderr=devnull)
                .decode()
                .rstrip("\r\n")
            )
            cuda_home = os.path.dirname(os.path.dirname(nvcc))

    except Exception:
        cuda_home = None

    if nvcc_ok(cuda_home, major, minor):
        return cuda_home

    return None


def main(argv: List[str]) -> None:
    # Handle command line args before passing to main setup() method.
    args, unknown = parse_args(argv)
    print("args: ", args)
    if len(unknown) != 0 and (len(unknown) != 1 or unknown[0] != "clean"):
        print("unknown: ", unknown)

    if not args.cpu_only:
        cub_include_path = os.getenv("CUB_DIR", None)
        if cub_include_path is None:
            print(
                "CUDA CUB directory environment variable not set.  Using default CUB location."
            )
            if torch.version.cuda is not None:
                cuda_version = torch.version.cuda.split(".")
                cuda_home = find_cuda(int(cuda_version[0]), int(cuda_version[1]))
            else:
                cuda_home = False

            if cuda_home:
                print(f"Using CUDA = {cuda_home}")
                os.environ["CUDA_BIN_PATH"] = cuda_home
                os.environ["CUDACXX"] = f"{cuda_home}/bin/nvcc"

    # Get the long description from the relevant file
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(cur_dir, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    torch_root = os.path.dirname(torch.__file__)

    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(os.cpu_count() // 2)

    cmake_args = [f"-DCMAKE_PREFIX_PATH={torch_root}"]
    cmake_args.append(get_cxx11_abi())
    if args.cpu_only:
        cmake_args.append("-DFBGEMM_CPU_ONLY=ON")
    if args.nvml_lib_path:
        cmake_args.append(f"-DNVML_LIB_PATH={args.nvml_lib_path}")

    package_version = generate_package_version(args.package_name)

    # Repair command line args for setup.
    sys.argv = [sys.argv[0]] + unknown

    setup(
        # Metadata
        name=args.package_name,
        version=package_version,
        author="FBGEMM Team",
        author_email="packages@pytorch.org",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/pytorch/fbgemm",
        license="BSD-3",
        keywords=[
            "PyTorch",
            "Recommendation Models",
            "High Performance Computing",
            "GPU",
            "CUDA",
        ],
        packages=["fbgemm_gpu"],
        cmake_args=cmake_args,
        # PyPI package information.
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])

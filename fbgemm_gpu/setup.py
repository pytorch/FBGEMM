# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import subprocess
import sys
from typing import Optional

import torch
from skbuild import setup
from skbuild.constants import CMAKE_INSTALL_DIR, skbuild_plat_name


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


cpu_only_build = False

plat_name = skbuild_plat_name()
print("plat_name:", plat_name)
print("CMAKE_INSTALL_DIR:", CMAKE_INSTALL_DIR())



# Handle command line args before passing to main setup() method.
if "--cpu_only" in sys.argv:
    cpu_only_build = True
    sys.argv.remove("--cpu_only")

if not cpu_only_build:
    cub_include_path = os.getenv("CUB_DIR", None)
    if cub_include_path is None:
        print(
            "CUDA CUB directory environment variable not set.  Using default CUB location."
        )

        cuda_version = torch.version.cuda.split(".")

        cuda_home = find_cuda(int(cuda_version[0]), int(cuda_version[1]))

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
if cpu_only_build:
    cmake_args.append("-DFBGEMM_CPU_ONLY=ON")

setup(
    name="fbgemm_gpu",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["fbgemm_gpu"],
    cmake_args=cmake_args,
)

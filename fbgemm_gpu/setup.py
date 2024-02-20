# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @licenselint-loose-mode

import argparse
import os
import random
import re
import subprocess
import sys
import textwrap

from datetime import date
from typing import List, Optional

import setuptools
import setuptools_git_versioning as gitversion
import torch
from setuptools.command.install import install as PipInstall
from skbuild import setup
from tabulate import tabulate


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fbgemm_gpu setup")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose logs during the build.",
    )
    parser.add_argument(
        "--package_variant",
        type=str,
        choices=["cpu", "cuda", "rocm"],
        default="cuda",
        help="The FBGEMM_GPU variant to build.",
    )
    parser.add_argument(
        "--package_name",
        type=str,
        default="fbgemm_gpu",
        help="The candidate name of the output wheel.",
    )
    parser.add_argument(
        "--nvml_lib_path",
        type=str,
        default=None,
        help="Certain operations require the nvml lib (libnvidia-ml.so). If you installed"
        " this in a custom location (through cudatoolkit-dev), provide the path here.",
    )
    parser.add_argument(
        "--cxxprefix",
        type=str,
        default=None,
        # nargs=2,
        # metavar=("name", "path"),
        help="Explicit compiler path.",
    )

    setup_py_args, other_args = parser.parse_known_args(argv)

    print(f"[SETUP.PY] Parsed setup.py arguments: {setup_py_args}")
    print(f"[SETUP.PY] Other arguments: {other_args}")
    return setup_py_args, other_args


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


def set_cuda_environment_variables() -> None:
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


def cmake_environment_variables(
    setup_py_args: argparse.Namespace, other_args: List[str]
) -> None:
    def _get_cxx11_abi():
        try:
            value = int(torch._C._GLIBCXX_USE_CXX11_ABI)
        except ImportError:
            value = 0
        return "-DGLIBCXX_USE_CXX11_ABI=" + str(value)

    torch_root = os.path.dirname(torch.__file__)
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(os.cpu_count() // 2)

    cmake_args = [
        f"-DCMAKE_PREFIX_PATH={torch_root}",
        _get_cxx11_abi(),
    ]

    if setup_py_args.verbose:
        print("[SETUP.PY] Building in VERBOSE mode ...")
        cmake_args.append("-DCMAKE_VERBOSE_MAKEFILE=1")

    if setup_py_args.package_variant == "cpu":
        print("[SETUP.PY] Building the CPU-ONLY variant of FBGEMM_GPU ...")
        cmake_args.append("-DFBGEMM_CPU_ONLY=ON")

    if setup_py_args.nvml_lib_path:
        cmake_args.append(f"-DNVML_LIB_PATH={setup_py_args.nvml_lib_path}")

    if setup_py_args.cxxprefix:
        print("[SETUP.PY] Setting CMake flags ...")
        path = setup_py_args.cxxprefix
        cmake_args.extend(
            [
                f"-DCMAKE_C_COMPILER={path}/bin/cc",
                f"-DCMAKE_CXX_COMPILER={path}/bin/c++",
                f"-DCMAKE_C_FLAGS='-fopenmp=libgomp -stdlib=libstdc++ -I{path}/include'",
                f"-DCMAKE_CXX_FLAGS='-fopenmp=libgomp -stdlib=libstdc++ -I{path}/include'",
            ]
        )

    # Pass CMake args attached to the setup.py call over to the CMake invocation
    for arg in other_args:
        if arg.startswith("-D"):
            cmake_args.append(arg)

    print(f"[SETUP.PY] Passing CMake arguments: {cmake_args}")
    return cmake_args


class FbgemmGpuInstaller(PipInstall):
    """FBGEMM_GPU PIP Installer"""

    @classmethod
    def extract_package_name(cls, args_package_name: str) -> str:
        package_name: str = ""

        if "BUILD_FROM_NOVA" in os.environ:
            nova_flag = os.getenv("BUILD_FROM_NOVA")
            print(f"[SETUP.PY] BUILD_FROM_NOVA={nova_flag}")

            # The package name is the same for all build variants in Nova
            package_name = "fbgemm_gpu"

            if str(nova_flag) != "0":
                # Skip build clean and build wheel steps in Nova workflow since
                # they are done in pre-script
                print("[SETUP.PY] Build from Nova detected... exiting.")
                sys.exit(0)

        else:
            package_name = args_package_name

        print(f"[SETUP.PY] Extracted the package name: '{package_name}'")
        return package_name

    @classmethod
    def extract_variant_version(cls, variant: str) -> str:
        variant_version: str = ""

        if variant == "cpu":
            variant_version = "+cpu"

        elif variant == "cuda":
            set_cuda_environment_variables()
            if torch.version.cuda is not None:
                cuda_version = torch.version.cuda.split(".")
                variant_version = f"+cu{cuda_version[0]}{cuda_version[1]}"
            else:
                sys.exit(
                    "[SETUP.PY] Installed PyTorch variant is not CUDA; cannot determine the CUDA version!"
                )

        elif variant == "rocm":
            if torch.version.hip is not None:
                rocm_version = torch.version.hip.split(".")
                variant_version = f"+rocm{rocm_version[0]}.{rocm_version[1]}"
            else:
                sys.exit(
                    "[SETUP.PY] Installed PyTorch variant is not ROCm; cannot determine the ROCm version!"
                )
        else:
            sys.exit(
                f"[SETUP.PY] Unrecognized build variant variant '{variant}'; cannot proceed with FBGEMM_GPU build!"
            )

        if "BUILD_FROM_NOVA" not in os.environ:
            # If not building from Nova, use the fbgemm_gpu-<variant>
            # PyPI does not accept version+xx in the name convention.
            print("[SETUP.PY] Not building FBGEMM_GPU from Nova.")
            variant_version = ""

        print(f"[SETUP.PY] Extracted the package variant+version: '{variant_version}'")
        return variant_version

    @classmethod
    def generate_package_version(cls, package_name: str, variant_version: str):
        print("[SETUP.PY] Generating the package version ...")

        if "nightly" in package_name:
            # Use date stamp for nightly versions
            print(
                "[SETUP.PY] Package is for NIGHTLY; using timestamp for the versioning"
            )
            today = date.today()
            version = f"{today.year}.{today.month}.{today.day}"

        elif "test" in package_name and "BUILD_FROM_NOVA" not in os.environ:
            # Use random numbering for test versions
            print(
                "[SETUP.PY] Package is for TEST: using random number for the versioning"
            )
            version = (f"0.0.{random.randint(0, 1000)}",)

        else:
            # Use git tag / branch / commit info to generate a PEP-440-compliant version string
            print(
                "[SETUP.PY] Package is for RELEASE: using git info for the versioning"
            )
            print(
                f"[SETUP.PY] TAG: {gitversion.get_tag()}, BRANCH: {gitversion.get_branch()}, SHA: {gitversion.get_sha()}"
            )
            # Remove the local version identifier, if any (e.g. 0.4.0rc0.post0+git.6a63116c.dirty => 0.4.0rc0.post0)
            # Then remove post0 (keep postN for N > 0) (e.g. 0.4.0rc0.post0 => 0.4.0rc0)
            version = re.sub(".post0$", "", gitversion.version_from_git().split("+")[0])
        version = str(version) + variant_version
        print(f"[SETUP.PY] Setting the full package version string: {version}")
        return version

    @classmethod
    def generate_version_file(cls, package_version: str) -> None:
        with open("fbgemm_gpu/docs/version.py", "w") as file:
            print(
                f"[SETUP.PY] Generating version file at: {os.path.realpath(file.name)}"
            )
            text = textwrap.dedent(
                f"""
                #!/usr/bin/env python3
                # Copyright (c) Meta Platforms, Inc. and affiliates.
                # All rights reserved.
                #
                # This source code is licensed under the BSD-style license found in the
                # LICENSE file in the root directory of this source tree.

                __version__: str = "{package_version}"
                """
            )
            file.write(text)

    @classmethod
    def description(cls) -> str:
        # Get the long description from the relevant file
        current_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(current_dir, "README.md"), encoding="utf-8") as f:
            return f.read()

    def print_versions(self) -> None:
        pytorch_version = (
            subprocess.run(
                ["python", "-c", "import torch; print(torch.__version__)"],
                stdout=subprocess.PIPE,
            )
            .stdout.decode("utf-8")
            .strip()
        )

        cuda_version_declared = (
            subprocess.run(
                ["python", "-c", "import torch; print(torch.version.cuda)"],
                stdout=subprocess.PIPE,
            )
            .stdout.decode("utf-8")
            .strip()
        )

        table = [
            ["", "Version"],
            ["PyTorch", pytorch_version],
        ]

        if cuda_version_declared != "None":
            cuda_version = cuda_version_declared.split(".")
            cuda_home = find_cuda(int(cuda_version[0]), int(cuda_version[1]))

            actual_cuda_version = (
                subprocess.run(
                    [f"{cuda_home}/bin/nvcc", "--version"],
                    stdout=subprocess.PIPE,
                )
                .stdout.decode("utf-8")
                .strip()
            )

            table.extend(
                [
                    ["CUDA (Declared by PyTorch)", cuda_version_declared],
                    ["CUDA (Actual)", actual_cuda_version],
                ]
            )

        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

    def run(self):
        PipInstall.run(self)
        self.print_versions()


def main(argv: List[str]) -> None:
    # Handle command line args before passing to main setup() method.
    setup_py_args, other_args = parse_args(argv)

    # Repair command line args for setup.
    sys.argv = [sys.argv[0]] + other_args

    # Extract the package name
    package_name = FbgemmGpuInstaller.extract_package_name(setup_py_args.package_name)

    # Extract the variant version, e.g. cpu, cu121, rocm5.6
    variant_version = FbgemmGpuInstaller.extract_variant_version(
        setup_py_args.package_variant
    )

    # Generate the full package version string
    package_version = FbgemmGpuInstaller.generate_package_version(
        setup_py_args.package_name, variant_version
    )

    # Generate the version file
    FbgemmGpuInstaller.generate_version_file(package_version)

    setup(
        name=package_name,
        version=package_version,
        author="FBGEMM Team",
        author_email="packages@pytorch.org",
        long_description=FbgemmGpuInstaller.description(),
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
        packages=setuptools.find_packages(),
        install_requires=[
            # Only specify numpy, as specifying torch will auto-install the
            # release version of torch, which is not what we want for the
            # nightly and test packages
            "numpy",
        ],
        cmake_args=cmake_environment_variables(setup_py_args, other_args),
        cmdclass={
            "install": FbgemmGpuInstaller,
        },
        # PyPI package information
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ]
        + [
            f"Programming Language :: Python :: {x}"
            for x in ["3", "3.8", "3.9", "3.10", "3.11", "3.12"]
        ],
    )


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])

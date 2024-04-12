# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @licenselint-loose-mode

import argparse
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

import setuptools
import setuptools_git_versioning as gitversion
import torch
from setuptools.command.install import install as PipInstall
from skbuild import setup
from tabulate import tabulate


@dataclass(frozen=True)
class FbgemmGpuBuild:
    args: argparse.Namespace
    other_args: List[str]

    """FBGEMM_GPU Package Build Configuration"""

    @classmethod
    def from_args(cls, argv: List[str]):
        parser = argparse.ArgumentParser(description="fbgemm_gpu setup")
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print verbose logs during the build.",
        )
        parser.add_argument(
            "--dryrun",
            action="store_true",
            help="Print build information only.",
        )
        parser.add_argument(
            "--package_variant",
            type=str,
            choices=["cpu", "cuda", "rocm"],
            default="cuda",
            help="The FBGEMM_GPU variant to build.",
        )
        parser.add_argument(
            "--package_channel",
            type=str,
            default="nightly",
            choices=["nightly", "test", "release"],
            help="The target package release channel that the output wheel is intended for.",
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
            help="Explicit compiler path.",
        )

        setup_py_args, other_args = parser.parse_known_args(argv)
        print(f"[SETUP.PY] Parsed setup.py arguments: {setup_py_args}")
        print(f"[SETUP.PY] Other arguments: {other_args}")
        return FbgemmGpuBuild(setup_py_args, other_args)

    def nova_flag(self) -> Optional[bool]:
        if "BUILD_FROM_NOVA" in os.environ:
            if str(os.getenv("BUILD_FROM_NOVA")) == "0":
                return 0
            else:
                return 1
        else:
            return None

    def package_name(self) -> str:
        pkg_name: str = "fbgemm_gpu"

        if self.nova_flag() == 1:
            # When running in Nova workflow context, the actual package build is
            # run in the Nova CI's "pre-script" step, as denoted by the
            # `BUILD_FROM_NOVA` flag.  As such, we skip building in the clean
            # and build wheel steps.
            print(
                "[SETUP.PY] Running under Nova workflow context (clean or build wheel step) ... exiting"
            )
            sys.exit(0)

        elif self.nova_flag() == 0:
            # The package name is the same for all build variants in Nova
            pass

        else:
            # If running outside of Nova workflow context, append the channel
            # and variant to the package name as needed
            if self.args.package_channel != "release":
                pkg_name += f"_{self.args.package_channel}"

            if self.args.package_variant != "cuda":
                pkg_name += f"-{self.args.package_variant}"

        print(f"[SETUP.PY] Determined the Python package name: '{pkg_name}'")
        return pkg_name

    def variant_version(self) -> str:
        pkg_vver: str = ""

        if "egg_info" in self.other_args:
            # If build is invoked through `python -m build` instead of
            # `python setup.py`, this script is invoked twice, once as
            # `setup.py egg_info`, and once as `setup.py bdist_wheel`.
            # Ignore determining the variant_version for the first case.
            print(
                "[SETUP.PY] Script was invoked as `setup.py egg_info`, ignoring variant_version"
            )
            return pkg_vver

        elif self.nova_flag() is None:
            # If not running in a Nova workflow, then use the
            # `fbgemm_gpu-<variant>` naming convention for the package, since
            # PyPI does not accept version+xx in the naming convention.
            print(
                "[SETUP.PY] Not running under Nova workflow context; ignoring variant_version"
            )
            return pkg_vver

        if self.args.package_variant == "cuda":
            CudaUtils.set_cuda_environment_variables()
            if torch.version.cuda is not None:
                cuda_version = torch.version.cuda.split(".")
                pkg_vver = f"+cu{cuda_version[0]}{cuda_version[1]}"
            else:
                sys.exit(
                    "[SETUP.PY] The installed PyTorch variant is not CUDA; cannot determine the CUDA version!"
                )

        elif self.args.package_variant == "rocm":
            if torch.version.hip is not None:
                rocm_version = torch.version.hip.split(".")
                pkg_vver = f"+rocm{rocm_version[0]}.{rocm_version[1]}"
            else:
                sys.exit(
                    "[SETUP.PY] The installed PyTorch variant is not ROCm; cannot determine the ROCm version!"
                )

        else:
            pkg_vver = "+cpu"

        print(f"[SETUP.PY] Extracted the package variant+version: '{pkg_vver}'")
        return pkg_vver

    def package_version(self):
        pkg_vver = self.variant_version()

        print("[SETUP.PY] Extracting the package version ...")
        print(
            f"[SETUP.PY] TAG: {gitversion.get_tag()}, BRANCH: {gitversion.get_branch()}, SHA: {gitversion.get_sha()}"
        )

        if self.args.package_channel == "nightly":
            # Use date stamp for nightly versions
            print(
                "[SETUP.PY] Package is for NIGHTLY; using timestamp for the versioning"
            )
            today = date.today()
            pkg_version = f"{today.year}.{today.month}.{today.day}"

        elif self.nova_flag() is not None:
            # For Nova workflow contexts, we want to strip out the `rcN` suffix
            # from the git-tagged version strings, regardless of test or release
            # channels.  This is done to comply with PyTorch PIP package naming
            # convensions

            # Remove -rcN, .rcN, or rcN (e.g. 0.4.0-rc0 => 0.4.0)
            pkg_version = re.sub(
                r"(\.|\-)*rc\d+$",
                "",
                # Remove postN (e.g. 0.4.0rc0.post0 => 0.4.0rc0)
                re.sub(
                    r"\.post\d+$",
                    "",
                    # Remove the local version identifier, if any (e.g. 0.4.0rc0.post0+git.6a63116c.dirty => 0.4.0rc0.post0)
                    gitversion.version_from_git().split("+")[0],
                ),
            )

        else:
            # For non-Nova workflow contexts, i.e. PyPI, we want to maintain the
            # `rcN` suffix in the version string

            # Remove post0 (keep postN for N > 0) (e.g. 0.4.0rc0.post0 => 0.4.0rc0)
            pkg_version = re.sub(
                r"\.post0$",
                "",
                # Remove the local version identifier, if any (e.g. 0.4.0rc0.post0+git.6a63116c.dirty => 0.4.0rc0.post0)
                gitversion.version_from_git().split("+")[0],
            )

        full_version_string = f"{pkg_version}{pkg_vver}"
        print(
            f"[SETUP.PY] Setting the full package version string: {full_version_string}"
        )
        return full_version_string

    def cmake_args(self) -> None:
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

        if self.args.verbose:
            print("[SETUP.PY] Building in VERBOSE mode ...")
            cmake_args.extend(
                ["-DCMAKE_VERBOSE_MAKEFILE=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE"]
            )

        if self.args.package_variant == "cpu":
            print("[SETUP.PY] Building the CPU-ONLY variant of FBGEMM_GPU ...")
            cmake_args.append("-DFBGEMM_CPU_ONLY=ON")

        if self.args.nvml_lib_path:
            cmake_args.append(f"-DNVML_LIB_PATH={self.args.nvml_lib_path}")

        if self.args.cxxprefix:
            print("[SETUP.PY] Setting CMake flags ...")
            path = self.args.cxxprefix
            cmake_args.extend(
                [
                    f"-DCMAKE_C_COMPILER={path}/bin/cc",
                    f"-DCMAKE_CXX_COMPILER={path}/bin/c++",
                    f"-DCMAKE_C_FLAGS='-fopenmp=libgomp -stdlib=libstdc++ -I{path}/include'",
                    f"-DCMAKE_CXX_FLAGS='-fopenmp=libgomp -stdlib=libstdc++ -I{path}/include'",
                ]
            )

        # Pass CMake args attached to the setup.py call over to the CMake invocation
        for arg in self.other_args:
            if arg.startswith("-D"):
                cmake_args.append(arg)

        print(f"[SETUP.PY] Passing CMake arguments: {cmake_args}")
        return cmake_args


class CudaUtils:
    """CUDA Utilities"""

    @classmethod
    def nvcc_ok(cls, cuda_home: str, major: int, minor: int) -> bool:
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

    @classmethod
    def find_cuda(cls, major: int, minor: int) -> Optional[str]:
        cuda_home = os.environ.get("CUDA_BIN_PATH")
        if cls.nvcc_ok(cuda_home, major, minor):
            return cuda_home

        cuda_nvcc = os.environ.get("CUDACXX")

        if cuda_nvcc and os.path.exists(cuda_nvcc):
            cuda_home = os.path.dirname(os.path.dirname(cuda_nvcc))
            if cls.nvcc_ok(cuda_home, major, minor):
                return cuda_home

        # Search standard installation location with version first
        cuda_home = f"/usr/local/cuda-{major}.{minor}"
        if cls.nvcc_ok(cuda_home, major, minor):
            return cuda_home

        cuda_home = "/usr/local/cuda"
        if cls.nvcc_ok(cuda_home, major, minor):
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

        if cls.nvcc_ok(cuda_home, major, minor):
            return cuda_home

        return None

    @classmethod
    def set_cuda_environment_variables(cls) -> None:
        cub_include_path = os.getenv("CUB_DIR", None)
        if cub_include_path is None:
            print(
                "[SETUP.PY] CUDA CUB directory environment variable not set.  Using default CUB location."
            )
            if torch.version.cuda is not None:
                cuda_version = torch.version.cuda.split(".")
                cuda_home = cls.find_cuda(int(cuda_version[0]), int(cuda_version[1]))
            else:
                cuda_home = False

            if cuda_home:
                print(f"[SETUP.PY] Using CUDA = {cuda_home}")
                os.environ["CUDA_BIN_PATH"] = cuda_home
                os.environ["CUDACXX"] = f"{cuda_home}/bin/nvcc"


class FbgemmGpuInstall(PipInstall):
    """FBGEMM_GPU PIP Install Routines"""

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
            cuda_home = CudaUtils.find_cuda(int(cuda_version[0]), int(cuda_version[1]))

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
    build = FbgemmGpuBuild.from_args(argv)
    # Repair command line args for setup.
    sys.argv = [sys.argv[0]] + build.other_args

    # Extract the package name
    package_name = build.package_name()
    # Generate the full package version string
    package_version = build.package_version()

    if build.args.dryrun:
        print(
            f"[SETUP.PY] Extracted package name and version: ({package_name} : {package_version})"
        )
        print("")
        sys.exit(0)

    # Generate the version file
    FbgemmGpuInstall.generate_version_file(package_version)

    setup(
        name=package_name,
        version=package_version,
        author="FBGEMM Team",
        author_email="packages@pytorch.org",
        long_description=FbgemmGpuInstall.description(),
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
        cmake_args=build.cmake_args(),
        cmdclass={
            "install": FbgemmGpuInstall,
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
    print(f"[SETUP.PY] ARGV: {sys.argv}")
    main(sys.argv[1:])

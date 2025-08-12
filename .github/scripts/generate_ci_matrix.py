#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List


TARGET_DEFAULT = "default"
TARGET_GENAI = "genai"
TARGET_HSTU = "hstu"
ALL_TARGETS = [TARGET_DEFAULT, TARGET_GENAI, TARGET_HSTU]

VARIANT_CPU = "cpu"
VARIANT_CUDA = "cuda"
VARIANT_ROCM = "rocm"

JOBTYPE_BUILD = "build"
JOBTYPE_TEST = "test"

REPO_OWNER_PYTORCH = "pytorch"
REPO_OWNER_FACEBOOKRESEARCH = "facebookresearch"
ALL_REPO_OWNERS = [REPO_OWNER_PYTORCH, REPO_OWNER_FACEBOOKRESEARCH]


@dataclass(frozen=True)
class BuildConfigScheme:
    """
    Immutable generator that produces matrix entries for a given build target,
    variant, and job type.
    """

    target: str
    variant: str
    jobtype: str
    repo_owner: str

    @classmethod
    def _create_parser(cls) -> argparse.ArgumentParser:
        """
        Create and return the argument parser for this class.
        """
        parser = argparse.ArgumentParser(
            description="Generate GitHub CI build matrix based on target, variant, and job type."
        )
        parser.add_argument(
            "--targets",
            required=True,
            type=str,
            help=f"Comma-separated build targets: {', '.join(ALL_TARGETS)}",
        )
        parser.add_argument(
            "--variant",
            required=True,
            choices=[VARIANT_CPU, VARIANT_CUDA, VARIANT_ROCM],
            help="Build variant: cpu, cuda, or rocm",
        )
        parser.add_argument(
            "--jobtype",
            required=True,
            choices=[JOBTYPE_BUILD, JOBTYPE_TEST],
            help="Job type",
        )
        parser.add_argument(
            "--repo-owner",
            required=False,
            choices=[REPO_OWNER_PYTORCH, REPO_OWNER_FACEBOOKRESEARCH],
            default=REPO_OWNER_PYTORCH,
            help=f"Repository owner: {', '.join(ALL_REPO_OWNERS)}",
        )
        return parser

    @classmethod
    def from_args(cls) -> List["BuildConfigScheme"]:
        """
        Construct a BuildConfigScheme from command-line arguments.
        Parses args if not provided.
        """
        parser = cls._create_parser()
        args = parser.parse_args()

        targets = [t.strip() for t in args.targets.strip().split(",") if t.strip()]

        if not targets:
            raise ValueError("Target string cannot be empty")

        # Validate each target
        for t in targets:
            if t not in ALL_TARGETS:
                raise argparse.ArgumentTypeError(
                    f"Invalid target: '{t}'. Allowed: {ALL_TARGETS}"
                )

        return [
            cls(
                target=t,
                variant=args.variant,
                jobtype=args.jobtype,
                repo_owner=args.repo_owner,
            ).validated()
            for t in targets
        ]

    def _dict_cartesian_product(
        self, table: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute the Cartesian product of lists in a dictionary, e.g.:

        { "x": [1, 2], "y": [3, 4] }
        -> [
            { "x": 1, "y": 3 },
            { "x": 1, "y": 4 },
            { "x": 2, "y": 3 },
            { "x": 2, "y": 4 }
        ]

        :param table: Dictionary where each value is a list
        :return: List of dictionaries, each representing one combination
        """
        keys = table.keys()
        # Create list of value lists, ordered by keys
        value_lists = [table[key] for key in keys]

        # Compute Cartesian product of the value lists
        product_combinations = itertools.product(*value_lists)

        # Reconstruct dicts
        return [dict(zip(keys, combination)) for combination in product_combinations]

    def validated(self) -> "BuildConfigScheme":
        """
        Validate the build config scheme.
        """
        if self.target not in ALL_TARGETS:
            raise ValueError(f"Invalid target: {self.target}")
        if self.variant not in [VARIANT_CPU, VARIANT_CUDA, VARIANT_ROCM]:
            raise ValueError(f"Invalid variant: {self.variant}")
        if self.jobtype not in [JOBTYPE_BUILD, JOBTYPE_TEST]:
            raise ValueError(f"Invalid job type: {self.jobtype}")

        if self.target == TARGET_GENAI and self.variant not in [
            VARIANT_CUDA,
            VARIANT_ROCM,
        ]:
            raise ValueError("GenAI target must be CUDA or ROCM")
        if self.target == TARGET_HSTU and self.variant != VARIANT_CUDA:
            raise ValueError("HSTU target must be CUDA")

        return self

    def python_versions(self) -> List[str]:
        if self.repo_owner != REPO_OWNER_PYTORCH:
            return ["3.13"]
        if self.target == TARGET_HSTU:
            # FBGEMM HSTU is expensive, so conserve CI resources
            return ["3.13"]
        if self.variant == VARIANT_ROCM:
            return ["3.13"]
        return ["3.9", "3.10", "3.11", "3.12", "3.13"]

    def compilers(self) -> List[str]:
        if self.repo_owner != REPO_OWNER_PYTORCH:
            return ["gcc"]
        if self.target == TARGET_HSTU:
            return ["gcc"]
        else:
            return ["gcc", "clang"]

    def cuda_versions(self) -> List[str]:
        if self.repo_owner != REPO_OWNER_PYTORCH:
            return ["12.9.1"]
        if self.target == TARGET_HSTU:
            # FBGEMM HSTU is expensive, so conserve CI resources
            return ["12.9.1"]
        else:
            # GenAI is unable to support 11.8.0 anymore as of https://github.com/pytorch/FBGEMM/pull/4138
            return ["12.6.3", "12.8.1", "12.9.1"]

    def rocm_versions(self) -> List[str]:
        return ["6.3", "6.4"]

    def host_machines(self) -> List[Dict[str, str]]:
        if self.repo_owner != REPO_OWNER_PYTORCH:
            if self.jobtype == JOBTYPE_BUILD:
                return [{"arch": "x86", "instance": "32-core-ubuntu"}]
            else:
                return [{"arch": "x86", "instance": "ubuntu-latest"}]

        if self.variant == VARIANT_CPU:
            return [
                {"arch": "x86", "instance": "linux.4xlarge"},
                {"arch": "arm", "instance": "linux.arm64.2xlarge"},
            ]

        elif self.variant == VARIANT_CUDA:
            # TODO: Enable when A100 machine queues are reasonably small enough for doing per-PR CI
            # https://hud.pytorch.org/metrics
            # { arch: x86, instance: "linux.gcp.a100" },
            if self.jobtype == JOBTYPE_BUILD:
                table = {
                    TARGET_DEFAULT: [{"arch": "x86", "instance": "linux.24xlarge"}],
                    TARGET_GENAI: [
                        {"arch": "x86", "instance": "linux.12xlarge.memory"}
                    ],
                    TARGET_HSTU: [{"arch": "x86", "instance": "linux.24xlarge.memory"}],
                }
                return table[self.target]
            else:
                return [{"arch": "x86", "instance": "linux.g5.4xlarge.nvidia.gpu"}]

        elif self.variant == VARIANT_ROCM:
            return [{"arch": "x86", "instance": "linux.rocm.gpu.2"}]

        else:
            return []

    def generate(self) -> List[Dict[str, Any]]:
        # Build a table of dimensions to values for each dimension
        table: Dict[str, List[Any]] = {
            "compiler": self.compilers(),
            "python-version": self.python_versions(),
            "host-machine": self.host_machines(),
            "build-target": [self.target],
        }

        if self.variant == VARIANT_CUDA:
            table |= {"cuda-version": self.cuda_versions()}

        if self.variant == VARIANT_ROCM:
            table |= {"rocm-version": self.rocm_versions()}

        # Generate the Cartesian product matrix from the table
        return self._dict_cartesian_product(table)


def main():
    try:
        configs = BuildConfigScheme.from_args()

        matrix = []
        for c in configs:
            matrix.extend(c.generate())

        print(json.dumps({"include": matrix}))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

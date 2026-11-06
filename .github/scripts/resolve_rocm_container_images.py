#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Add the container image for its ROCm version to each row of a ROCm CI matrix.

The matrix is read from the MATRIX environment variable in `{"include": [...]}`
form, and each row specifies its `rocm-version`, or "latest" for the newest
ROCm version.  The images are taken from ROCM_CONTAINER_IMAGES in
generate_ci_matrix.py when known; otherwise ROCM_BASE_IMAGE is used so versions
sourced from pytorch/test-infra do not require a hardcoded map update.
"""

import json
import os
import sys

from generate_ci_matrix import ROCM_BASE_IMAGE, ROCM_CONTAINER_IMAGES


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(map(int, version.split(".")))


def main() -> None:
    matrix = json.loads(os.environ["MATRIX"])
    known = list(ROCM_CONTAINER_IMAGES)
    matrix_versions = [
        row["rocm-version"]
        for row in matrix["include"]
        if row.get("rocm-version") and row["rocm-version"] != "latest"
    ]
    latest = sorted(set(known) | set(matrix_versions), key=_version_key)[-1]

    for row in matrix["include"]:
        if "rocm-version" not in row:
            print(
                "Error: matrix row is missing required 'rocm-version': "
                f"{json.dumps(row, sort_keys=True)}",
                file=sys.stderr,
            )
            sys.exit(1)

        if row["rocm-version"] == "latest":
            row["rocm-version"] = latest

        row["container-image"] = ROCM_CONTAINER_IMAGES.get(
            row["rocm-version"], ROCM_BASE_IMAGE
        )

    print(json.dumps(matrix))


if __name__ == "__main__":
    main()

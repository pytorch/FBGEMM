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
generate_ci_matrix.py, so that callers of the ROCm workflows only need to
specify the ROCm version.
"""

import json
import os
import sys

from generate_ci_matrix import ROCM_CONTAINER_IMAGES


def main() -> None:
    matrix = json.loads(os.environ["MATRIX"])
    latest = list(ROCM_CONTAINER_IMAGES)[-1]

    for row in matrix["include"]:
        if row["rocm-version"] == "latest":
            row["rocm-version"] = latest

        if row["rocm-version"] not in ROCM_CONTAINER_IMAGES:
            print(
                f"Error: no container image for ROCm {row['rocm-version']}; "
                f"known versions: {', '.join(ROCM_CONTAINER_IMAGES)}",
                file=sys.stderr,
            )
            sys.exit(1)

        row["container-image"] = ROCM_CONTAINER_IMAGES[row["rocm-version"]]

    print(json.dumps(matrix))


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tags used for emitting CUTLASS C++ kernels
"""

import enum
from enum import auto as enum_auto

########################################################################################
# Common utilities
########################################################################################


def CudaToolkitVersionSatisfies(semantic_ver_string, major, minor, patch=0):

    # by default, use the latest CUDA Toolkit version
    cuda_version = [11, 0, 132]

    # Update cuda_version based on parsed string
    if semantic_ver_string != "":
        for i, x in enumerate([int(x) for x in semantic_ver_string.split(".")]):
            if i < len(cuda_version):
                cuda_version[i] = x
            else:
                cuda_version.append(x)
    return cuda_version >= [major, minor, patch]


########################################################################################
# Tags used for emitting CUTLASS C++ kernels
########################################################################################


class FusionKind(enum.Enum):
    NoneScaling = enum_auto()
    TensorWise = enum_auto()


FusionKindNames = {
    FusionKind.NoneScaling: "none",
    FusionKind.TensorWise: "tensorwise",
}
########################################################################################

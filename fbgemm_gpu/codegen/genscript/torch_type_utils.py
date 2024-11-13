# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

######################################################################
# PyTorch Type Utils
######################################################################

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict


class ArgType(IntEnum):
    TENSOR = 0
    INT_TENSOR = 1
    LONG_TENSOR = 2
    FLOAT_TENSOR = 3
    HALF_TENSOR = 4
    BFLOAT16_TENSOR = 5
    PLACEHOLDER_TENSOR = 6
    INT = 7
    FLOAT = 8
    SYM_INT = 9
    BOOL = 10


@dataclass
class TensorType:
    # Primitive type
    primitive_type: str
    # PyTorch Scalar type
    scalar_type: str


arg_type_to_tensor_type: Dict[ArgType, TensorType] = {
    ArgType.FLOAT_TENSOR: TensorType("float", "at::ScalarType::Float"),
    ArgType.HALF_TENSOR: TensorType("at::Half", "at::ScalarType::Half"),
    ArgType.BFLOAT16_TENSOR: TensorType("at::BFloat16", "at::ScalarType::BFloat16"),
}

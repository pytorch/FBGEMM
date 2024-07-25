#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from enum import IntEnum


class RoundingMode(IntEnum):
    """Rounding options for quantization."""

    nearest = 0
    floor = 1
    even = 2
    stochastic = 3
    ceil = 4


def get_mx4_exp_bias(ebits):
    """Helper function to get the proper exponent bias for specified mx4 format.

    Args:
        ebits: The number of exponent bits in quantized format.

    Returns:
        The exponent bias for the specified mx4 format.
    """
    if ebits == 2:
        return 1
    elif ebits == 3:
        return 3
    else:
        raise NotImplementedError(f"MX4 with ebits={ebits} not supported.")


def get_mx4_lookup_table(ebits, mbits):
    """Helper function to get the proper lookup table for specified mx4 format.

    Args:
        ebits: The number of exponent bits in quantized format.
        mbits: The number of mantissa bits in quantized format.

    Returns:
        The lookup table for the specified mx4 format.
    """
    if ebits == 2 and mbits == 1:
        return [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]
    elif ebits == 3 and mbits == 0:
        return [
            0,
            0.25,
            0.5,
            1,
            2,
            4,
            8,
            16,
            -0,
            -0.25,
            -0.5,
            -1,
            -2,
            -4,
            -8,
            -16,
        ]
    else:
        raise NotImplementedError(
            f"MX4 with ebits={ebits} and mbits={mbits} not supported."
        )

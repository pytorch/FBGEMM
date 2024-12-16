#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from enum import IntEnum

import torch


# We keep LUTs persistent to minimize the number of device copies required.
E2M1_LUT = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
E3M0_LUT = torch.tensor(
    [0, 0.25, 0.5, 1, 2, 4, 8, 16, -0, -0.25, -0.5, -1, -2, -4, -8, -16],
    dtype=torch.float32,
)


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


def get_mx4_lookup_table(ebits, mbits, device):
    """Helper function to get the proper lookup table for specified mx4 format.

    Args:
        ebits: The number of exponent bits in quantized format.
        mbits: The number of mantissa bits in quantized format.
        device: The device that the LUT should be copied to.

    Returns:
        The lookup table for the specified mx4 format.
    """
    global E2M1_LUT, E3M0_LUT
    if ebits == 2 and mbits == 1:
        # Update state of LUT to minimize copies.
        if E2M1_LUT.device != device:
            E2M1_LUT = E2M1_LUT.to(device)
        return E2M1_LUT
    elif ebits == 3 and mbits == 0:
        # Update state of LUT to minimize copies.
        if E3M0_LUT.device != device:
            E3M0_LUT = E3M0_LUT.to(device)
        return E3M0_LUT
    else:
        raise NotImplementedError(
            f"MX4 with ebits={ebits} and mbits={mbits} not supported."
        )

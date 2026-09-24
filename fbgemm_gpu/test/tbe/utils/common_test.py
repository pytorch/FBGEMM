#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

import torch
from fbgemm_gpu.tbe.utils.common import get_device, to_device


class TBEDeviceUtilsTest(unittest.TestCase):
    def test_get_device_selects_xpu(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.mtia, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(torch.xpu, "current_device", return_value=2),
        ):
            self.assertEqual(get_device(), 2)

    def test_get_device_preserves_accelerator_priority(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.mtia, "is_available", return_value=True),
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=1),
        ):
            self.assertEqual(get_device(), 1)

        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.mtia, "is_available", return_value=True),
            patch.object(torch.xpu, "is_available", return_value=True),
            patch.object(torch.mtia, "current_device", return_value=3),
        ):
            self.assertEqual(get_device(), 3)

    def test_to_device_selects_xpu(self) -> None:
        deviceable = MagicMock()
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.mtia, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=True),
        ):
            result = to_device(deviceable, use_cpu=False)

        deviceable.to.assert_called_once_with(device="xpu")
        self.assertIs(result, deviceable.to.return_value)

    def test_to_device_preserves_existing_paths(self) -> None:
        cpu_deviceable = MagicMock()
        self.assertIs(
            to_device(cpu_deviceable, use_cpu=True),
            cpu_deviceable.cpu.return_value,
        )
        cpu_deviceable.cpu.assert_called_once_with()

        cuda_deviceable = MagicMock()
        with patch.object(torch.cuda, "is_available", return_value=True):
            self.assertIs(
                to_device(cuda_deviceable, use_cpu=False),
                cuda_deviceable.cuda.return_value,
            )
        cuda_deviceable.cuda.assert_called_once_with()

        mtia_deviceable = MagicMock()
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.mtia, "is_available", return_value=True),
            patch.object(torch.xpu, "is_available", return_value=True),
        ):
            self.assertIs(
                to_device(mtia_deviceable, use_cpu=False),
                mtia_deviceable.to.return_value,
            )
        mtia_deviceable.to.assert_called_once_with(device="mtia")

        fallback_deviceable = MagicMock()
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.mtia, "is_available", return_value=False),
            patch.object(torch.xpu, "is_available", return_value=False),
        ):
            self.assertIs(
                to_device(fallback_deviceable, use_cpu=False),
                fallback_deviceable.to.return_value,
            )
        fallback_deviceable.to.assert_called_once_with(device="mtia")


if __name__ == "__main__":
    unittest.main()

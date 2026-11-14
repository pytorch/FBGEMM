# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
from fbgemm_gpu.utils.torchscript import has_torchscript


class TorchScriptCapabilityTest(unittest.TestCase):
    def test_available_compiler(self) -> None:
        with patch.object(torch.jit, "script", lambda value: value, create=True):
            self.assertTrue(has_torchscript())

    def test_missing_compiler(self) -> None:
        for initially_available in (False, True):
            with (
                self.subTest(initially_available=initially_available),
                patch.dict(torch.jit.__dict__),
            ):
                if initially_available:
                    torch.jit.__dict__["script"] = lambda value: value
                else:
                    torch.jit.__dict__.pop("script", None)

                with patch.dict(torch.jit.__dict__):
                    torch.jit.__dict__.pop("script", None)
                    self.assertFalse(has_torchscript())

                self.assertEqual(has_torchscript(), initially_available)

    def test_noncallable_compiler(self) -> None:
        with patch.object(torch.jit, "script", None, create=True):
            self.assertFalse(has_torchscript())

    def test_failing_compiler_remains_visible(self) -> None:
        def failing_compiler(value: object) -> None:
            raise RuntimeError("compiler failure")

        with patch.object(torch.jit, "script", failing_compiler, create=True):
            self.assertTrue(has_torchscript())
            with self.assertRaisesRegex(RuntimeError, "compiler failure"):
                torch.jit.script(object())

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

import torch
from fbgemm_gpu.split_embedding_configs import (
    _nfp8_dtype_for_device_index,
    nfp8_dtype,
    sparse_type_int_to_dtype,
    SparseType,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    _legacy_package_nfp8_dtype,
    _table_embedding_dtype,
)


def _empty_tensor_with_sparse_type_dtype(ty: int) -> torch.Tensor:
    return torch.empty(0, dtype=sparse_type_int_to_dtype(ty))


def _rocm_device(index: int) -> torch.device:
    # lint-fixme: TorchDeviceCuda, TorchFunctionCallCudaDevice
    # ROCm devices use PyTorch's CUDA device namespace.
    return torch.device("cuda", index)


class NFP8DtypeTest(unittest.TestCase):
    def setUp(self) -> None:
        _nfp8_dtype_for_device_index.cache_clear()

    def tearDown(self) -> None:
        _nfp8_dtype_for_device_index.cache_clear()

    def test_sparse_type_int_to_dtype_is_torchscript_compatible(self) -> None:
        scripted = torch.jit.script(_empty_tensor_with_sparse_type_dtype)

        self.assertIs(scripted(9).dtype, nfp8_dtype())

    def test_nfp8_dtype_uses_target_device_architecture(self) -> None:
        for device_index, (arch, expected_dtype) in enumerate(
            (
                ("gfx90a:sramecc+", torch.float8_e4m3fnuz),
                ("gfx942:sramecc+", torch.float8_e4m3fnuz),
                ("gfx950:sramecc+", torch.float8_e4m3fn),
            )
        ):
            target_device = _rocm_device(device_index)
            with (
                self.subTest(arch=arch),
                patch.object(torch.version, "hip", "6.3"),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "current_device") as current_device,
                patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=MagicMock(gcnArchName=arch),
                ) as get_properties,
            ):
                self.assertIs(nfp8_dtype(target_device), expected_dtype)
                self.assertIs(nfp8_dtype(target_device), expected_dtype)
                current_device.assert_not_called()
                get_properties.assert_called_once_with(device_index)

    def test_nfp8_dtype_skips_device_query_without_rocm_gpu(self) -> None:
        for hip_version, cuda_available, device, expected_dtype in (
            (
                None,
                True,
                _rocm_device(0),
                torch.float8_e4m3fn,
            ),
            ("6.3", False, None, torch.float8_e4m3fn),
        ):
            with (
                self.subTest(
                    hip_version=hip_version,
                    cuda_available=cuda_available,
                    device=device,
                ),
                patch.object(torch.version, "hip", hip_version),
                patch.object(torch.cuda, "is_available", return_value=cuda_available),
                patch.object(torch.cuda, "get_device_properties") as get_properties,
            ):
                self.assertIs(nfp8_dtype(device), expected_dtype)
                get_properties.assert_not_called()

    def test_device_less_nfp8_callers_use_current_device_architecture(self) -> None:
        with (
            patch.object(torch.version, "hip", "6.3"),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(
                torch.cuda, "current_device", return_value=1
            ) as current_device,
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=MagicMock(gcnArchName="gfx90a:sramecc+"),
            ) as get_properties,
        ):
            self.assertIs(nfp8_dtype(), torch.float8_e4m3fnuz)
            self.assertIs(nfp8_dtype(torch.device("cpu")), torch.float8_e4m3fnuz)
            self.assertIs(nfp8_dtype(torch.device("meta")), torch.float8_e4m3fnuz)
            self.assertIs(SparseType.NFP8.as_dtype(), torch.float8_e4m3fnuz)
            self.assertIs(sparse_type_int_to_dtype(9), torch.float8_e4m3fnuz)
            current_device.assert_called()
            get_properties.assert_called_once_with(1)

    def test_table_embedding_dtype_uses_legacy_fallback_when_symbol_missing(
        self,
    ) -> None:
        with (
            patch(
                "fbgemm_gpu.split_table_batched_embeddings_ops_training."
                "split_embedding_configs",
                object(),
            ),
            patch.object(torch.version, "hip", "6.3"),
        ):
            with self.assertRaisesRegex(RuntimeError, "architecture-aware"):
                _table_embedding_dtype(
                    SparseType.NFP8,
                    torch.device("cpu"),
                )

    def test_legacy_package_nfp8_dtype_fails_closed_on_rocm(self) -> None:
        with patch.object(torch.version, "hip", "6.3"):
            with self.assertRaisesRegex(RuntimeError, "architecture-aware"):
                _legacy_package_nfp8_dtype(torch.device("cpu"))

    def test_table_embedding_dtype_uses_target_device_for_nfp8(self) -> None:
        target_device = _rocm_device(1)
        with patch(
            "fbgemm_gpu.split_table_batched_embeddings_ops_training."
            "split_embedding_configs.nfp8_dtype",
            return_value=torch.float8_e4m3fnuz,
        ) as resolve_dtype:
            self.assertIs(
                _table_embedding_dtype(SparseType.NFP8, target_device),
                torch.float8_e4m3fnuz,
            )
            resolve_dtype.assert_called_once_with(target_device)

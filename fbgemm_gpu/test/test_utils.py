# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import math
import os
import struct
import subprocess
import unittest
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import fbgemm_gpu

import hypothesis.strategies as st
import numpy as np
import torch


TEST_WITH_ROCM: bool = os.getenv("FBGEMM_TEST_WITH_ROCM", "0") == "1"
# Eigen/Python round 0.5 away from 0, Numpy rounds to even
round_to_nearest: Callable[[np.ndarray], np.ndarray] = np.vectorize(round)


def bytes_to_floats(byte_matrix: np.ndarray) -> np.ndarray:
    floats = np.empty([np.shape(byte_matrix)[0], 1], dtype=np.float32)
    for i, byte_values in enumerate(byte_matrix):
        (floats[i],) = struct.unpack("f", bytearray(byte_values))
    return floats


def floats_to_bytes(floats: np.ndarray) -> np.ndarray:
    byte_matrix = np.empty([np.shape(floats)[0], 4], dtype=np.uint8)
    for i, value in enumerate(floats):
        assert isinstance(value, np.float32), (value, floats)
        as_bytes = struct.pack("f", value)
        # In Python3 bytes will be a list of int, in Python2 a list of string
        if isinstance(as_bytes[0], int):
            byte_matrix[i] = list(as_bytes)
        else:
            byte_matrix[i] = list(map(ord, as_bytes))
    return byte_matrix


def bytes_to_half_floats(byte_matrix: np.ndarray) -> np.ndarray:
    floats = np.empty([np.shape(byte_matrix)[0], 1], dtype=np.float16)
    for i, byte_values in enumerate(byte_matrix):
        (floats[i],) = np.frombuffer(
            memoryview(byte_values).tobytes(), dtype=np.float16
        )
    return floats


def half_floats_to_bytes(floats: np.ndarray) -> np.ndarray:
    byte_matrix = np.empty([np.shape(floats)[0], 2], dtype=np.uint8)
    for i, value in enumerate(floats):
        assert isinstance(value, np.float16), (value, floats)
        byte_matrix[i] = np.frombuffer(
            memoryview(value.tobytes()).tobytes(), dtype=np.uint8
        )
    return byte_matrix


def fused_rowwise_8bit_quantize_reference(data: np.ndarray) -> np.ndarray:
    minimum = np.min(data, axis=-1, keepdims=True)
    maximum = np.max(data, axis=-1, keepdims=True)
    span = maximum - minimum
    bias = minimum
    scale = span / 255.0
    inverse_scale = 255.0 / (span + 1e-8)
    quantized_data = round_to_nearest((data - bias) * inverse_scale)
    scale_bytes = floats_to_bytes(scale.reshape(-1))
    scale_bytes = scale_bytes.reshape(data.shape[:-1] + (scale_bytes.shape[-1],))
    bias_bytes = floats_to_bytes(bias.reshape(-1))
    bias_bytes = bias_bytes.reshape(data.shape[:-1] + (bias_bytes.shape[-1],))
    return np.concatenate([quantized_data, scale_bytes, bias_bytes], axis=-1)


def fused_rowwise_8bit_dequantize_reference(fused_quantized: np.ndarray) -> np.ndarray:
    scale = bytes_to_floats(fused_quantized[..., -8:-4].astype(np.uint8).reshape(-1, 4))
    scale = scale.reshape(fused_quantized.shape[:-1] + (scale.shape[-1],))
    bias = bytes_to_floats(fused_quantized[..., -4:].astype(np.uint8).reshape(-1, 4))
    bias = bias.reshape(fused_quantized.shape[:-1] + (bias.shape[-1],))
    quantized_data = fused_quantized[..., :-8]
    return quantized_data * scale + bias


def fused_rowwise_8bit_dequantize_reference_half(
    fused_quantized: np.ndarray,
) -> np.ndarray:
    scale = bytes_to_half_floats(
        fused_quantized[..., -8:-4].astype(np.uint8).reshape(-1, 4)
    )
    scale = scale.reshape(fused_quantized.shape[:-1] + (scale.shape[-1],))
    bias = bytes_to_half_floats(
        fused_quantized[..., -4:].astype(np.uint8).reshape(-1, 4)
    )
    bias = bias.reshape(fused_quantized.shape[:-1] + (bias.shape[-1],))
    quantized_data = fused_quantized[..., :-8]
    return quantized_data * scale + bias


def fused_rowwise_nbit_quantize_reference(data: np.ndarray, bit: int) -> np.ndarray:
    minimum = np.min(data, axis=1).astype(np.float16).astype(np.float32)
    maximum = np.max(data, axis=1)
    span = maximum - minimum
    qmax = (1 << bit) - 1
    scale = (span / qmax).astype(np.float16).astype(np.float32)
    bias = np.zeros(data.shape[0])
    quantized_data = np.zeros(data.shape).astype(np.uint8)

    for i in range(data.shape[0]):
        bias[i] = minimum[i]
        inverse_scale = 1.0 if scale[i] == 0.0 else 1 / scale[i]
        if scale[i] == 0.0 or math.isinf(inverse_scale):
            scale[i] = 1.0
            inverse_scale = 1.0
        quantized_data[i] = np.clip(
            np.round((data[i, :] - minimum[i]) * inverse_scale), 0, qmax
        )

    # pack
    assert 8 % bit == 0
    num_elem_per_byte = 8 // bit
    packed_dim = (data.shape[1] + num_elem_per_byte - 1) // num_elem_per_byte
    packed_data = np.zeros([data.shape[0], packed_dim]).astype(np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j % num_elem_per_byte == 0:
                packed_data[i, j // num_elem_per_byte] = quantized_data[i, j]
            else:
                packed_data[i, j // num_elem_per_byte] += quantized_data[i, j] << (
                    (j % num_elem_per_byte) * bit
                )

    scale_bytes = half_floats_to_bytes(scale.astype(np.float16))
    bias_bytes = half_floats_to_bytes(bias.astype(np.float16))
    return np.concatenate([packed_data, scale_bytes, bias_bytes], axis=1)


def fused_rowwise_nbit_quantize_dequantize_reference(
    data: np.ndarray, bit: int
) -> np.ndarray:
    fused_quantized = fused_rowwise_nbit_quantize_reference(data, bit)
    scale = bytes_to_half_floats(fused_quantized[:, -4:-2].astype(np.uint8)).astype(
        np.float32
    )
    bias = bytes_to_half_floats(fused_quantized[:, -2:].astype(np.uint8)).astype(
        np.float32
    )
    quantized_data = fused_quantized[:, :-4]

    # unpack
    packed_dim = fused_quantized.shape[1] - 4
    assert 8 % bit == 0
    num_elem_per_byte = 8 // bit
    assert packed_dim == ((data.shape[1] + num_elem_per_byte - 1) // num_elem_per_byte)
    unpacked_data = np.zeros(data.shape).astype(np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            unpacked_data[i, j] = (
                quantized_data[i, j // num_elem_per_byte]
                >> ((j % num_elem_per_byte) * bit)
            ) & ((1 << bit) - 1)

    return scale * unpacked_data + bias


# Used for `@unittest.skipIf`
gpu_unavailable: Tuple[bool, str] = (
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    "CUDA is not available or no GPUs detected",
)
# Used for `if` statements inside tests
gpu_available: bool = not gpu_unavailable[0]

# Used for `@unittest.skipIf` for tests that pass in internal CI, but fail on the GitHub runners
running_on_github: Tuple[bool, str] = (
    os.getenv("GITHUB_ENV") is not None,
    "Test is currently known to fail or hang when run in the GitHub runners",
)

# Used for `@unittest.skipIf` for tests that currently fail on ARM platform
on_arm_platform: Tuple[bool, str] = (
    subprocess.run(["uname", "-m"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
    == "aarch64",
    "Test is currently known to fail when running on ARM platform",
)


def cpu_and_maybe_gpu() -> st.SearchStrategy[List[torch.device]]:
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    # st.sampled_from is not guaranteed to test all the values passed to it.
    # However, Hypothesis, by default, generates 100 test cases from the specified strategies.
    # If st.sampled_from contains >100 items or if it's used in conjunction with other strategies
    # then it may not test all values; however, for smaller tests it may work fine.
    # This is still a stopgap solution until we figure out a way to parameterize UnitTestCase.
    return st.sampled_from(
        [torch.device("cpu")] + ([torch.device("cuda")] if gpu_available else [])
    )


def has_optests() -> bool:
    return (
        torch.__version__ >= "2.2.*"
        and hasattr(torch.library, "impl_abstract")
        and not hasattr(fbgemm_gpu, "open_source")
    )


class optests:
    # Usage examples:
    #
    # @generate_opcheck_tests
    # class MyOpTest(unittest.TestCase):
    #     ...
    #
    # @generate_opcheck_tests(additional_decorators={})
    # class MyOpTest(unittest.TestCase):
    #     ...
    #
    @staticmethod
    # pyre-ignore[3]
    def generate_opcheck_tests(
        test_class: Optional[unittest.TestCase] = None,
        *,
        fast: bool = False,
        # pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
        additional_decorators: Optional[Dict[str, Callable]] = None,
    ):
        if additional_decorators is None:
            additional_decorators = {}

        def decorator(test_class: unittest.TestCase) -> unittest.TestCase:
            if not has_optests():
                return test_class
            import torch.testing._internal.optests as optests
            from torch._utils_internal import get_file_path_2

            filename = inspect.getfile(test_class)
            failures_dict_name = "failures_dict.json"
            if fast:
                failures_dict_name = "failures_dict_fast.json"
            failures_dict_path = get_file_path_2(
                "", os.path.dirname(filename), failures_dict_name
            )
            tests_to_run = [
                "test_schema",
                "test_autograd_registration",
                "test_faketensor",
            ]
            if not fast:
                tests_to_run.extend(
                    [
                        "test_aot_dispatch_dynamic",
                    ]
                )
            optests.generate_opcheck_tests(
                test_class,
                ["fb", "fbgemm"],
                failures_dict_path,
                # pyre-ignore[6]
                additional_decorators,
                tests_to_run,
            )
            return test_class

        if test_class is None:
            return decorator
        else:
            return decorator(test_class)

    @staticmethod
    def is_inside_opcheck_mode() -> bool:
        if not has_optests():
            return False

        import torch.testing._internal.optests as optests

        return optests.is_inside_opcheck_mode()

    @staticmethod
    # pyre-ignore[3]
    def dontGenerateOpCheckTests(reason: str):
        if not has_optests():
            return lambda fun: fun
        import torch.testing._internal.optests as optests

        return optests.dontGenerateOpCheckTests(reason)


# Version of torch.autograd.gradcheck that works with generate_opcheck_tests.
# The problem with just torch.autograd.gradcheck is that it results in
# very slow tests when composed with generate_opcheck_tests.
def gradcheck(
    # pyre-ignore[24]: Generic type `Callable` expects 2 type parameters.
    f: Callable,
    # pyre-ignore[2]
    inputs: Union[torch.Tensor, Tuple[Any, ...]],
    *args: Any,
    **kwargs: Any,
) -> None:
    if optests.is_inside_opcheck_mode():
        if isinstance(inputs, torch.Tensor):
            f(inputs)
        else:
            f(*inputs)
        return
    torch.autograd.gradcheck(f, inputs, *args, **kwargs)


def cpu_only() -> st.SearchStrategy[List[torch.device]]:
    return st.sampled_from([torch.device("cpu")])


# pyre-fixme[3]: Return annotation cannot be `Any`.
def skipIfRocm(reason: str = "Test currently doesn't work on the ROCm stack") -> Any:
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def skipIfRocmDecorator(fn: Callable) -> Any:
        @wraps(fn)
        # pyre-fixme[3]: Return annotation cannot be `Any`.
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if TEST_WITH_ROCM:
                raise unittest.SkipTest(reason)
            else:
                fn(*args, **kwargs)

        return wrapper

    return skipIfRocmDecorator


def symint_vector_unsupported() -> Tuple[bool, str]:
    major, minor = torch.__version__.split(".")[0:2]
    return (
        int(major) < 2 or (int(major) == 2 and int(minor) < 1),
        """
        dynamic shape support for this op needs to be on PyTorch 2.1 or
        newer with https://github.com/pytorch/pytorch/pull/101056
        """,
    )

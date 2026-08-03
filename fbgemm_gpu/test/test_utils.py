# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import os
import random
import subprocess
import unittest
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import Any

import fbgemm_gpu
import hypothesis.strategies as st
import torch
from hypothesis import HealthCheck, settings

_suppressed_health_checks: list[HealthCheck] = [
    HealthCheck.filter_too_much,
    HealthCheck.data_too_large,
] + (
    [HealthCheck.differing_executors]
    if getattr(HealthCheck, "differing_executors", False)
    else []
)

settings.register_profile(
    "derandomize", derandomize=True, suppress_health_check=_suppressed_health_checks
)
settings.load_profile("derandomize")


TEST_WITH_ROCM: bool = os.getenv("FBGEMM_TEST_WITH_ROCM", "0") == "1" or (
    torch.version.hip is not None and torch.cuda.is_available()
)

# Skip pt2 compliant tag test for certain operators
# TODO: remove this once the operators are pt2 compliant
additional_decorators: dict[str, list[Callable[..., Any]]] = {
    # vbe_generate_metadata_cpu return different values from vbe_generate_metadata_meta
    # this fails fake_tensor test as the test expects them to be the same
    # fake_tensor test is added in failures_dict but failing fake_tensor test still cause pt2_compliant tag test to fail
    "test_pt2_compliant_tag_fbgemm_split_embedding_codegen_lookup_rowwise_adagrad_function_pt2": [
        unittest.skip("Operator failed on pt2 compliant tag"),
    ],
    # learning rate tensor needs to be on CPU to avoid D->H sync point since it will be used as float in the kernel
    # this fails fake_tensor test as the test expects all tensors to be on the same device
    "test_pt2_compliant_tag_fbgemm_split_embedding_codegen_lookup_rowwise_adagrad_function": [
        unittest.skip(
            "Operator failed on FakeTensor test since learning rate tensor is always on CPU regardless of other tensors"
        ),
    ],
}

# Used for `@unittest.skipIf`
gpu_unavailable: tuple[bool, str] = (
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    "CUDA is not available or no GPUs detected",
)
# Used for `if` statements inside tests
gpu_available: bool = not gpu_unavailable[0]

is_nvidia_device: bool = gpu_available and torch.version.cuda is not None

# Used for `@unittest.skipIf` for tests that pass in internal CI, but fail on the GitHub runners
running_on_github: tuple[bool, str] = (
    os.getenv("GITHUB_ENV") is not None,
    "Test is currently known to fail or hang when run in the GitHub runners",
)

running_in_oss: tuple[bool, str] = (
    # pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
    getattr(fbgemm_gpu, "open_source", False),
    "Test is currently known to fail in OSS mode",
)


def seed_all(seed: int = 0) -> None:
    """Seed all RNGs used by FBGEMM tests for deterministic behavior.

    Many legacy tests draw split points / weights / indices from global RNGs
    inside the test body (not via Hypothesis), which makes Hypothesis-recorded
    failures non-reproducible and the tests flaky. Call this in ``setUp`` (or at
    the top of a test) to make those draws deterministic.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


running_on_rocm: tuple[bool, str] = (
    TEST_WITH_ROCM,
    "Test currently doesn't work on the ROCm stack",
)

# Tests with this marker generally fails with `free(): corrupted unsorted chunks`
# errors when fbgemm_gpu is compiled under Clang
on_oss_clang: tuple[bool, str] = (
    (
        hasattr(fbgemm_gpu, "open_source")
        and os.system("c++ --version | grep -i clang") == 0
    ),
    "Test is currently known to fail when fbgemm_gpu is built by Clang in OSS",
)

# Used for `@unittest.skipIf` for tests that currently fail on ARM platform
on_arm_platform: tuple[bool, str] = (
    subprocess.run(["uname", "-m"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
    == "aarch64",
    "Test is currently known to fail when running on ARM platform",
)


def symint_vector_unsupported() -> tuple[bool, str]:
    major, minor = torch.__version__.split(".")[0:2]
    return (
        int(major) < 2 or (int(major) == 2 and int(minor) < 1),
        """
        dynamic shape support for this op needs to be on PyTorch 2.1 or
        newer with https://github.com/pytorch/pytorch/pull/101056
        """,
    )


def cpu_and_maybe_gpu() -> st.SearchStrategy[list[torch.device]]:
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    # st.sampled_from is not guaranteed to test all the values passed to it.
    # However, Hypothesis, by default, generates 100 test cases from the specified strategies.
    # If st.sampled_from contains >100 items or if it's used in conjunction with other strategies
    # then it may not test all values; however, for smaller tests it may work fine.
    # This is still a stopgap solution until we figure out a way to parameterize UnitTestCase.
    # lint-fixme: TorchDeviceCuda, TorchFunctionCallCudaDevice
    # CUDA specifically required: GPU device strategy for FBGEMM tests
    gpu_devices = [torch.device("cuda")] if gpu_available else []
    # pyrefly: ignore [bad-return]
    return st.sampled_from([torch.device("cpu")] + gpu_devices)


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
    def generate_opcheck_tests(
        test_class: unittest.TestCase | None = None,
        *,
        fast: bool = False,
        additional_decorators: dict[str, Callable[..., Any]] | None = None,
    ) -> unittest.TestCase | Callable[[unittest.TestCase], unittest.TestCase]:
        if additional_decorators is None:
            additional_decorators = {}

        def decorator(test_class: unittest.TestCase) -> unittest.TestCase:
            if hasattr(fbgemm_gpu, "open_source"):
                return test_class
            import torch.testing._internal.optests as optests
            from torch._utils_internal import (  # @manual=//caffe2:utils_internal
                get_file_path_2,
            )

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
        if hasattr(fbgemm_gpu, "open_source"):
            return False
        import torch.testing._internal.optests as optests

        return optests.is_inside_opcheck_mode()

    @staticmethod
    def dontGenerateOpCheckTests(
        reason: str,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if hasattr(fbgemm_gpu, "open_source"):
            return lambda fun: fun
        import torch.testing._internal.optests as optests

        return optests.dontGenerateOpCheckTests(reason)


class TestSuite(unittest.TestCase):
    @contextmanager
    def assertNotRaised(
        self, exc_type: type[BaseException]
    ) -> Generator[None, None, None]:
        try:
            yield None
        except exc_type as e:
            raise self.failureException(e)


# Version of torch.autograd.gradcheck that works with generate_opcheck_tests.
# The problem with just torch.autograd.gradcheck is that it results in
# very slow tests when composed with generate_opcheck_tests.
def gradcheck(
    f: Callable[..., Any],
    inputs: torch.Tensor | tuple[Any, ...],
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


def cpu_only() -> st.SearchStrategy[list[torch.device]]:
    # pyrefly: ignore [bad-return]
    return st.sampled_from([torch.device("cpu")])


def use_cpu_strategy() -> st.SearchStrategy[bool]:
    return (
        st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        # fmt: off
        else st.just(False)
        if (gpu_available and TEST_WITH_ROCM)
        else st.just(True)
        # fmt: on
    )


def gpu_memory_lt_gb(x: int) -> tuple[bool, str]:
    assert x > 0, "GB value must be positive"
    return (
        torch.cuda.is_available()
        and (torch.cuda.get_device_properties(0).total_memory / (1024**3)) < x,
        "GPU memory < 40GB",
    )


def skipIfRocm(
    reason: str = "Test currently doesn't work on the ROCm stack",
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def decorator(fn: Callable[..., None]) -> Callable[..., None]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            if TEST_WITH_ROCM:
                raise unittest.SkipTest(reason)
            else:
                fn(*args, **kwargs)

        return wrapper

    return decorator


def skipIfNotRocm(
    reason: str = "Test currently doesn work only on the ROCm stack",
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def decorator(fn: Callable[..., None]) -> Callable[..., None]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            if TEST_WITH_ROCM:
                fn(*args, **kwargs)
            else:
                raise unittest.SkipTest(reason)

        return wrapper

    return decorator

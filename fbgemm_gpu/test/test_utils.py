# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import os
import subprocess
import unittest
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import fbgemm_gpu
import hypothesis.strategies as st
import torch
from hypothesis import settings

settings.register_profile("derandomize", derandomize=True)
settings.load_profile("derandomize")


TEST_WITH_ROCM: bool = os.getenv("FBGEMM_TEST_WITH_ROCM", "0") == "1"

# Skip pt2 compliant tag test for certain operators
# TODO: remove this once the operators are pt2 compliant
# pyre-ignore
additional_decorators: Dict[str, List[Callable]] = {
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
gpu_unavailable: Tuple[bool, str] = (
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    "CUDA is not available or no GPUs detected",
)
# Used for `if` statements inside tests
gpu_available: bool = not gpu_unavailable[0]

running_on_sm70: Tuple[bool, str] = (
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    "Skip test if SM70, since the code is hardcoded to sm80+ support",
)

# Used for `@unittest.skipIf` for tests that pass in internal CI, but fail on the GitHub runners
running_on_github: Tuple[bool, str] = (
    os.getenv("GITHUB_ENV") is not None,
    "Test is currently known to fail or hang when run in the GitHub runners",
)

running_on_rocm: Tuple[bool, str] = (
    TEST_WITH_ROCM,
    "Test currently doesn't work on the ROCm stack",
)

# Tests with this marker generally fails with `free(): corrupted unsorted chunks`
# errors when fbgemm_gpu is compiled under Clang
on_oss_clang: Tuple[bool, str] = (
    (
        hasattr(fbgemm_gpu, "open_source")
        and os.system("c++ --version | grep -i clang") == 0
    ),
    "Test is currently known to fail when fbgemm_gpu is built by Clang in OSS",
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


class TestSuite(unittest.TestCase):
    @contextmanager
    # pyre-ignore[2]
    def assertNotRaised(self, exc_type) -> None:
        try:
            # pyre-ignore[7]
            yield None
        except exc_type as e:
            raise self.failureException(e)


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


def use_cpu_strategy() -> st.SearchStrategy[bool]:
    return (
        st.booleans()
        if (gpu_available and not TEST_WITH_ROCM)
        # fmt: off
        else st.just(False) if (gpu_available and TEST_WITH_ROCM) else st.just(True)
        # fmt: on
    )


# pyre-fixme[3]: Return annotation cannot be `Any`.
def skipIfRocm(reason: str = "Test currently doesn't work on the ROCm stack") -> Any:
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def decorator(fn: Callable) -> Any:
        @wraps(fn)
        # pyre-fixme[3]: Return annotation cannot be `Any`.
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if TEST_WITH_ROCM:
                raise unittest.SkipTest(reason)
            else:
                fn(*args, **kwargs)

        return wrapper

    return decorator


# pyre-fixme[3]: Return annotation cannot be `Any`.
def skipIfNotRocm(reason: str = "Test currently doesn work only on the ROCm stack") -> Any:
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def decorator(fn: Callable) -> Any:
        @wraps(fn)
        # pyre-fixme[3]: Return annotation cannot be `Any`.
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if TEST_WITH_ROCM:
                fn(*args, **kwargs)
            else:
                raise unittest.SkipTest(reason)

        return wrapper

    return decorator


# pyre-fixme[3]: Return annotation cannot be `Any`.
def skipIfRocmLessThan(min_version: int) -> Any:
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def decorator(testfn: Callable) -> Any:
        @wraps(testfn)
        # pyre-fixme[3]: Return annotation cannot be `Any`.
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ROCM_VERSION_FILEPATH = "/opt/rocm/.info/version-dev"
            if TEST_WITH_ROCM:
                # Fail if ROCm version file is missing.
                if not os.path.isfile(ROCM_VERSION_FILEPATH):
                    raise AssertionError(
                        f"ROCm version file {ROCM_VERSION_FILEPATH} is missing!"
                    )

                # Parse the version number from the file.
                with open(ROCM_VERSION_FILEPATH, "r") as file:
                    version = file.read().strip()
                version = version.replace("-", "").split(".")
                version = (
                    int(version[0]) * 10000 + int(version[1]) * 100 + int(version[2])
                )

                # Fail if ROCm version is less than the minimum version.
                if version < min_version:
                    raise unittest.SkipTest(
                        f"Skip the test since the ROCm version is less than {min_version}"
                    )
                else:
                    testfn(*args, **kwargs)

            else:
                testfn(*args, **kwargs)

        return wrapper

    return decorator


def symint_vector_unsupported() -> Tuple[bool, str]:
    major, minor = torch.__version__.split(".")[0:2]
    return (
        int(major) < 2 or (int(major) == 2 and int(minor) < 1),
        """
        Dynamic shape support for this operator needs to be on PyTorch 2.1 or
        newer with https://github.com/pytorch/pytorch/pull/101056
        """,
    )

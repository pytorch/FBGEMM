#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest
from typing import Tuple, Type

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import cpu_and_maybe_gpu
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import cpu_and_maybe_gpu


class CumSumTest(unittest.TestCase):
    @given(
        n=st.integers(min_value=0, max_value=10),
        index_types=st.sampled_from(
            [
                (torch.int64, np.int64),
                (torch.int32, np.int32),
                (torch.float32, np.float32),
            ]
        ),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_cumsum(
        self,
        n: int,
        index_types: Tuple[Type[object], Type[object]],
        device: torch.device,
    ) -> None:
        (pt_index_dtype, np_index_dtype) = index_types

        # The CPU variants of asynchronous_*_cumsum support floats, since some
        # downstream tests appear to be relying on this behavior.  As such, the
        # test is disabled for GPU + float test cases.
        if device == torch.device("cuda") and pt_index_dtype is torch.float32:
            return

        # pyre-ignore-errors[16]
        x = torch.randint(low=0, high=100, size=(n,)).type(pt_index_dtype).to(device)
        ze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(x)
        zi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(x)
        zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)

        torch.testing.assert_close(
            torch.from_numpy(np.cumsum(x.cpu().numpy()).astype(np_index_dtype)),
            zi.cpu(),
        )
        torch.testing.assert_close(
            torch.from_numpy(
                (np.cumsum([0] + x.cpu().numpy().tolist())[:-1]).astype(np_index_dtype)
            ),
            ze.cpu(),
        )
        torch.testing.assert_close(
            torch.from_numpy(
                (np.cumsum([0] + x.cpu().numpy().tolist())).astype(np_index_dtype)
            ),
            zc.cpu(),
        )

        # meta tests
        # pyre-ignore-errors[16]
        mx = torch.randint(low=0, high=100, size=(n,)).type(pt_index_dtype).to("meta")

        mze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(mx)
        self.assertEqual(ze.size(), mze.size())

        mzi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(mx)
        self.assertEqual(zi.size(), mzi.size())

        mzc = torch.ops.fbgemm.asynchronous_complete_cumsum(mx)
        self.assertEqual(zc.size(), mzc.size())

    @given(
        n=st.integers(min_value=0, max_value=60),
        b=st.integers(min_value=0, max_value=10),
        index_types=st.sampled_from(
            [
                (torch.int64, np.int64),
                (torch.int32, np.int32),
                (torch.float32, np.float32),
            ]
        ),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_asynchronous_complete_cumsum_2d(
        self,
        n: int,
        b: int,
        index_types: Tuple[Type[object], Type[object]],
        device: torch.device,
    ) -> None:
        (pt_index_dtype, np_index_dtype) = index_types

        # The CPU variants of asynchronous_*_cumsum support floats, since some
        # downstream tests appear to be relying on this behavior.  As such, the
        # test is disabled for GPU + float test cases.
        if device == torch.device("cuda") and pt_index_dtype is torch.float32:
            return

        # pyre-ignore-errors[16]
        x = torch.randint(low=0, high=100, size=(b, n)).type(pt_index_dtype).to(device)

        zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        zeros = torch.zeros(b, 1)
        torch.testing.assert_close(
            torch.from_numpy(
                np.cumsum(torch.concat([zeros, x.cpu()], dim=1).numpy(), axis=1).astype(
                    np_index_dtype
                )
            ),
            zc.cpu(),
        )


extend_test_class(CumSumTest)

if __name__ == "__main__":
    unittest.main()

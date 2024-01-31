#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import unittest

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_available


class CumSumTest(unittest.TestCase):
    @given(
        n=st.integers(min_value=0, max_value=10),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_cumsum(self, n: int, long_index: bool) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        np_index_dtype = np.int64 if long_index else np.int32

        # cpu tests
        x = torch.randint(low=0, high=100, size=(n,)).type(index_dtype)
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
        mx = torch.randint(low=0, high=100, size=(n,)).type(index_dtype).to("meta")
        mze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(mx)
        self.assertEqual(ze.size(), mze.size())
        # mzi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(mx)
        # self.assertEqual(zi.size(), mzi.size())
        mzc = torch.ops.fbgemm.asynchronous_complete_cumsum(mx)
        self.assertEqual(zc.size(), mzc.size())

        if gpu_available:
            x = x.cuda()
            ze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(x)
            zi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(x)
            zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
            torch.testing.assert_close(
                torch.from_numpy(np.cumsum(x.cpu().numpy()).astype(np_index_dtype)),
                zi.cpu(),
            )
            torch.testing.assert_close(
                torch.from_numpy(
                    (np.cumsum([0] + x.cpu().numpy().tolist())[:-1]).astype(
                        np_index_dtype
                    )
                ),
                ze.cpu(),
            )
            torch.testing.assert_close(
                torch.from_numpy(
                    (np.cumsum([0] + x.cpu().numpy().tolist())).astype(np_index_dtype)
                ),
                zc.cpu(),
            )

    @given(
        n=st.integers(min_value=0, max_value=60),
        b=st.integers(min_value=0, max_value=10),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_asynchronous_complete_cumsum_2d(
        self, n: int, b: int, long_index: bool
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32

        def test_asynchronous_complete_cumsum_2d_helper(x: torch.Tensor) -> None:
            np_index_dtype = np.int64 if long_index else np.int32
            zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
            zeros = torch.zeros(b, 1)
            torch.testing.assert_close(
                torch.from_numpy(
                    np.cumsum(
                        torch.concat([zeros, x.cpu()], dim=1).numpy(), axis=1
                    ).astype(np_index_dtype)
                ),
                zc.cpu(),
            )

        x = torch.randint(low=0, high=100, size=(b, n)).type(index_dtype)
        # cpu test
        test_asynchronous_complete_cumsum_2d_helper(x)
        if gpu_available:
            # gpu test
            test_asynchronous_complete_cumsum_2d_helper(x.cuda())


extend_test_class(CumSumTest)

if __name__ == "__main__":
    unittest.main()

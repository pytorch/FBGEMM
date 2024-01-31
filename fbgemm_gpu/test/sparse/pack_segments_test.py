#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import sys
import unittest
from typing import Callable, Optional

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, skipIfRocm
else:
    from fbgemm_gpu.test.test_utils import gpu_available, skipIfRocm


def get_n_rand_num_summing_to_k(n: int, k: int) -> np.ndarray:
    """Get a list of `n` integers which collectively sum to `k`, drawn
    uniformly from the set of all such lists.

    Args:
        n - The number of integers in the result list
        k - The value they should sum to
    """
    # There are a lot of ways to do this wrong, probably including
    # the ones you've just thought of. I think the following does
    # it correctly, though.
    if n == 0:
        return np.array([])
    return np.random.multinomial(k, np.ones(n) / n, size=1)[0]


# pyre-fixme[2]
# pyre-fixme[24]
def torch_compiled(model: Callable, **kwargs) -> Callable:
    if sys.version_info < (3, 12, 0):
        return torch.compile(model, **kwargs)
    else:
        return model


class PackedSegmentsTest(unittest.TestCase):
    def _pack_segments_ref(
        self,
        lengths: torch.Tensor,
        tensor: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        lengths = lengths.numpy()
        sections = np.split(tensor, np.cumsum(lengths))
        max_length = np.max(lengths, initial=0) if max_length is None else max_length
        padded_arrs = []
        for arr in sections[:-1]:  # Last section is always a blank
            arr = arr[: min(max_length, len(arr)), ...]
            padded_arr = np.pad(
                arr,
                [(0, max(max_length - arr.shape[0], 0))]
                + ([(0, 0)] * (len(arr.shape) - 1)),
                constant_values=0,
            )
            padded_arrs.append(padded_arr)

        if len(padded_arrs) == 0:
            padded_arrs = torch.empty((0, 0) + tuple(tensor.shape[1:]))
        else:
            padded_arrs = torch.Tensor(np.stack(padded_arrs))

        # pyre-fixme[7]: Expected `ndarray` but got `Tensor`.
        return padded_arrs

    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
            ]
        ),
        torch_compile=st.booleans(),
    )
    @settings(deadline=None)
    def test_pack_segments(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
        dtype: torch.dtype,
        torch_compile: bool,
    ) -> None:
        input_raw = np.random.rand(batch_size, n, k)
        input_data = torch.tensor(input_raw, dtype=dtype, requires_grad=True)
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size),
            dtype=torch.int,
        )
        max_length = lengths.max().item()

        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data, lengths=lengths, max_length=max_length
        )

        packed_ref = self._pack_segments_ref(lengths, input_raw)
        packed_ref = torch.Tensor(packed_ref).to(dtype)

        self.assertTrue(torch.equal(packed_tensor, packed_ref))

        grad_cpu = torch.tensor(
            np.random.uniform(low=0.01, high=0.5, size=packed_ref.shape).astype(
                np.float32
            )
        ).to(dtype)
        # CPU backward
        packed_tensor.backward(grad_cpu)

        if gpu_available:
            pack_segments_fun = torch.ops.fbgemm.pack_segments

            if torch_compile:
                pack_segments_fun = torch_compiled(pack_segments_fun, dynamic=True)

            packed_cuda = pack_segments_fun(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )

            self.assertTrue(torch.equal(packed_tensor, packed_cuda.cpu()))

            # GPU backward
            packed_cuda.backward(grad_cpu.cuda())

            # dynamic check
            input_raw = np.random.rand(batch_size, n + 1, k + 2)
            input_data = torch.tensor(input_raw, dtype=dtype, requires_grad=True)
            lengths = torch.tensor(
                get_n_rand_num_summing_to_k(divisions, batch_size), dtype=torch.int
            )
            max_length = lengths.max().item()
            packed_tensor = torch.ops.fbgemm.pack_segments(
                t_in=input_data, lengths=lengths, max_length=max_length
            )

            packed_ref = self._pack_segments_ref(lengths, input_raw)
            packed_ref = torch.Tensor(packed_ref).to(dtype)

            self.assertTrue(torch.equal(packed_tensor, packed_ref))

            grad_cpu = torch.tensor(
                np.random.uniform(low=0.01, high=0.5, size=packed_ref.shape).astype(
                    np.float32
                )
            ).to(dtype)
            # CPU backward
            packed_tensor.backward(grad_cpu)

            # reusing the previously compiled kernel
            packed_cuda = pack_segments_fun(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )
            self.assertTrue(torch.equal(packed_tensor, packed_cuda.cpu()))

            # GPU backward
            packed_cuda.backward(grad_cpu.cuda())

    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
        max_length=st.integers(1, 20),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
            ]
        ),
        torch_compile=st.booleans(),
    )
    @settings(deadline=None)
    def test_pack_segments_smaller_max_len(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
        max_length: int,
        dtype: torch.dtype,
        torch_compile: bool,
    ) -> None:
        input_data = torch.tensor(np.random.rand(batch_size, n, k), dtype=dtype)
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size), dtype=torch.int
        )

        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data,
            lengths=lengths,
            max_length=max_length,
        )
        self.assertEqual(packed_tensor.shape, (divisions, max_length, n, k))

        packed_ref = self._pack_segments_ref(
            lengths,
            input_data,
            max_length=max_length,
        )
        # pyre-fixme[6]: For 2nd param expected `Tensor` but got `ndarray`.
        self.assertTrue(torch.equal(packed_tensor, packed_ref))

        if gpu_available:
            pack_segments_fun = torch.ops.fbgemm.pack_segments
            if torch_compile:
                pack_segments_fun = torch_compiled(pack_segments_fun)

            packed_cuda = pack_segments_fun(
                t_in=input_data.cuda(),
                lengths=lengths.cuda(),
                max_length=max_length,
            )
            self.assertTrue(torch.equal(packed_tensor, packed_cuda.cpu()))

    @skipIfRocm()
    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
            ]
        ),
    )
    @settings(deadline=None)
    def test_pack_segments_meta_backend(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
        dtype: torch.dtype,
    ) -> None:
        input_raw = np.random.rand(batch_size, n, k)
        input_data = torch.tensor(
            input_raw, dtype=torch.float32, requires_grad=True
        ).to("meta")
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size), dtype=torch.int
        )
        max_length = lengths.max().item()

        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data, lengths=lengths, max_length=max_length
        )
        packed_ref = self._pack_segments_ref(lengths, input_raw)

        # verify forward
        assert packed_tensor.size() == torch.Tensor(packed_ref).size()


extend_test_class(PackedSegmentsTest)


if __name__ == "__main__":
    unittest.main()

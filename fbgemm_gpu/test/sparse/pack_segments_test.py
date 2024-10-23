#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import sys
import unittest
from typing import Callable, Optional

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import torch
from hypothesis import given, settings

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable


def get_n_rand_num_summing_to_k(n: int, k: int) -> npt.NDArray:
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
    """A helper function to apply torch.compile if python < 3.12.

    Args:
        model: The model to be compiled.
        kwargs: The arguments to be passed to torch.compile.

    Returns:
        The model.
    """
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
    ) -> npt.NDArray:
        """
        This function is a reference implementation of pack_segments.

        Args:
            lengths (Tensor): The lengths of tensor.
            tensor (Tensor): The tensor to be packed.
            max_length (Optional[int]): The maximum length of the packed tensor.

        Returns:
            The packed tensor.
        """
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
                torch.bfloat16,
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
        """
        This function tests pack_segments ops compared to the reference implementation.
        Both CPU and GPU (if available) are tested.

        Args:
            n - The number of rows in the input tensor
            k - The number of columns in the input tensor
            batch_size - The number of batches of the input tensor
            divisions - The number of segments to be packed
            dtype - The data type
            torch_compile - Whether to use torch.compile

        Returns:
            None
        """

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
                torch.bfloat16,
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
        """
        This function tests pack_segments ops with set max_length
        Both CPU and GPU (if available) are tested.

        Args:
            n - The number of rows in the input tensor
            k - The number of columns in the input tensor
            batch_size - The number of batches of the input tensor
            divisions - The number of segments to be packed
            max_length - The maximum length of the packed tensor
            dtype - The data type
            torch_compile - Whether to use torch.compile

        Returns:
            None
        """

        input_raw = np.random.rand(batch_size, n, k)
        input_data = torch.tensor(input_raw, dtype=dtype)
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
            input_raw,
            max_length=max_length,
        )
        packed_ref = torch.Tensor(packed_ref).to(dtype)
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

    @given(
        n=st.integers(2, 10),
        k=st.integers(2, 10),
        batch_size=st.integers(1, 30),
        divisions=st.integers(1, 10),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.half,
                torch.bfloat16,
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
        """
        This function tests pack_segments ops with meta backend.

        Args:
            n - The number of rows in the input tensor
            k - The number of columns in the input tensor
            batch_size - The number of batches of the input tensor
            divisions - The number of segments to be packed
            dtype - The data type

        Returns:
            None
        """

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

    @unittest.skipIf(*gpu_unavailable)
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
        use_cpu=st.booleans(),
    )
    @settings(deadline=None)
    def test_pack_segments_noncontig(
        self,
        n: int,
        k: int,
        batch_size: int,
        divisions: int,
        dtype: torch.dtype,
        torch_compile: bool,
        use_cpu: bool,
    ) -> None:
        """
        This function tests pack_segments ops when input gradients to backward are non-contiguous.

        Args:
            n - The number of rows in the input tensor
            k - The number of columns in the input tensor
            batch_size - The number of batches of the input tensor
            divisions - The number of segments to be packed
            dtype - The data type
            torch_compile - Whether to use torch.compile
            use_cpu - Whether to use CPU or GPU

        Returns:
            None
        """

        input_raw = np.random.rand(batch_size, n, k)
        # create input
        input_data_ref = torch.tensor(input_raw, dtype=dtype, requires_grad=True)
        input_data = torch.tensor(input_raw, dtype=dtype, requires_grad=True).cuda()
        # retain grad to compare gradients of the inputs later
        input_data.retain_grad()
        input_data_ref.retain_grad()

        # set lengths
        lengths = torch.tensor(
            get_n_rand_num_summing_to_k(divisions, batch_size),
            dtype=torch.int,
        )
        max_length = lengths.max().item()

        packed_ref = torch.ops.fbgemm.pack_segments(
            t_in=input_data_ref, lengths=lengths, max_length=max_length
        )
        packed_ref.retain_grad()

        # pack segments using fbgemm and fb
        packed_tensor = torch.ops.fbgemm.pack_segments(
            t_in=input_data, lengths=lengths.cuda(), max_length=max_length
        )
        packed_tensor.retain_grad()

        # verify forward
        self.assertTrue(torch.equal(packed_tensor.cpu(), packed_ref))

        # create non-contiguous grad
        shape = tuple(x * 2 for x in packed_ref.shape)
        grads = torch.tensor(
            np.random.uniform(low=0.01, high=0.5, size=shape).astype(np.float32)
        ).to(dtype)
        grad_noncontig_cpu = grads.as_strided(packed_ref.shape, grads.stride())
        grad_noncontig_cuda = grads.cuda().as_strided(packed_ref.shape, grads.stride())

        self.assertTrue(
            not (
                grad_noncontig_cpu.is_contiguous()
                and grad_noncontig_cuda.is_contiguous()
            ),
            msg="Expected grads to be non-contiguous but they are contiguous",
        )

        # verify backward
        packed_ref.backward(grad_noncontig_cpu)
        packed_tensor.backward(grad_noncontig_cuda)
        self.assertTrue(
            torch.equal(packed_tensor.cpu(), packed_ref),
            msg="Expected packed tensors to be equal but they are not",
        )

        # verify backward input gradients
        self.assertTrue(
            # pyre-fixme[16]: Optional type has no attribute `cpu`.
            # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Optional[Tensor]`.
            torch.equal(input_data.grad.cpu(), input_data_ref.grad.cpu()),
            msg="Expected input gradients to be equal but they are not",
        )


extend_test_class(PackedSegmentsTest)


if __name__ == "__main__":
    unittest.main()

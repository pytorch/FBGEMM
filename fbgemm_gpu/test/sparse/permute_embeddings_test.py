#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

import random
import unittest
from typing import Any, Callable, Tuple

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, open_source, permute_indices_ref_

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_available, on_oss_clang
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_available, on_oss_clang


class PermuteEmbeddingsTest(unittest.TestCase):
    @staticmethod
    def permute_embeddings_(
        permute_fn: Callable[..., Tuple[torch.Tensor, ...]],
        *args: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if permute_fn == torch.ops.fbgemm.permute_2D_sparse_data:
            permuted_lengths, permuted_embeddings, _ = permute_fn(*args, None)
            return permuted_lengths, permuted_embeddings
        else:
            return permute_fn(*args)

    @unittest.skipIf(*on_oss_clang)
    @given(
        B=st.integers(min_value=0, max_value=20),
        T=st.integers(min_value=0, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        permute_fn=st.sampled_from(
            [
                torch.ops.fbgemm.permute_2D_sparse_data,
                torch.ops.fbgemm.permute_sequence_embeddings,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_embeddings(
        self,
        B: int,
        T: int,
        L: int,
        long_index: bool,
        permute_fn: Callable[..., Tuple[torch.Tensor, ...]],
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
        #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
        embeddings = torch.rand(lengths.sum().item()).float()
        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (permuted_lengths_cpu, permuted_embeddings_cpu) = self.permute_embeddings_(
            permute_fn, permute, lengths, embeddings
        )
        (
            permuted_lengths_ref,
            permuted_embeddings_ref,
            _,
            # pyre-fixme[6]: For 4th param expected `LongTensor` but got `Tensor`.
        ) = permute_indices_ref_(lengths, embeddings, None, permute.long())
        torch.testing.assert_close(permuted_embeddings_cpu, permuted_embeddings_ref)
        torch.testing.assert_close(permuted_lengths_cpu, permuted_lengths_ref)

        if gpu_available:
            (permuted_lengths_gpu, permuted_embeddings_gpu) = self.permute_embeddings_(
                permute_fn,
                permute.cuda(),
                lengths.cuda(),
                embeddings.cuda(),
            )
            torch.testing.assert_close(
                permuted_embeddings_gpu.cpu(), permuted_embeddings_cpu
            )
            torch.testing.assert_close(permuted_lengths_gpu.cpu(), permuted_lengths_cpu)


extend_test_class(PermuteEmbeddingsTest)

if __name__ == "__main__":
    unittest.main()

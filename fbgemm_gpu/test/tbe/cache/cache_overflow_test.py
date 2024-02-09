#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import logging
import unittest

import hypothesis.strategies as st

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_embedding_utils import to_device
from fbgemm_gpu.split_table_batched_embeddings_ops_training import DEFAULT_ASSOC
from hypothesis import given, settings

from ..common import assert_torch_equal, MAX_EXAMPLES
from .cache_common import assert_cache, generate_cache_tbes, gpu_unavailable, VERBOSITY


class CacheOverflowTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        stochastic_rounding=st.booleans(),
    )
    @settings(verbosity=VERBOSITY, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_int32_overflow(self, stochastic_rounding: bool) -> None:
        """
        The unit test verifies that TBE with UVM caching can handle the cache
        access correctly when the cache access offset is larger than max int32
        """
        D_fac = 1024 // 4
        D = D_fac * 4
        cache_sets = 10**6

        current_device = torch.device(torch.cuda.current_device())
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        free_memory = total_memory - torch.cuda.memory_reserved(current_device)

        # Weight and cache precisions are fixed to FP16
        element_size = 2
        # Adjust cache_sets based on free memory
        while cache_sets * DEFAULT_ASSOC * D * element_size > free_memory:
            cache_sets = cache_sets // 10

        # Generate TBEs
        cc, cc_ref, _, _ = generate_cache_tbes(
            T=1,
            D=D_fac,
            log_E=1,
            mixed=False,
            prefetch_pipeline=True,
            cache_sets=cache_sets,
            weights_cache_precision=SparseType.FP16,
            stochastic_rounding=stochastic_rounding,
        )

        # Accessing the last cache slot
        last_cache_set = cache_sets - 1
        cache_idx = last_cache_set * DEFAULT_ASSOC + (DEFAULT_ASSOC - 1)
        if cache_idx * D < (2**31) - 1:
            logging.warning("test_cache_int32_overflow does not test int32 overflowing")
        else:
            logging.info(f"Testing overflow cache offsets: {cache_idx * D}")

        # Simply access index 0
        indices = torch.tensor([0], dtype=torch.long)
        offsets = torch.tensor([0, 1], dtype=torch.long)
        indices = to_device(indices, use_cpu=False)
        offsets = to_device(offsets, use_cpu=False)

        # Force the cache locations to be the last row in the cache
        assert cache_idx < cc.lxu_cache_weights.shape[0]
        lxu_cache_locations = torch.tensor([cache_idx], dtype=torch.int)
        lxu_cache_locations = to_device(lxu_cache_locations, use_cpu=False)

        # Does prefetch into the cache
        cc.lxu_cache_weights[cache_idx] = cc_ref.weights_dev.view(-1, D)[0]

        # Mimic cache prefetching behavior
        cc.timesteps_prefetched.append(0)
        cc.lxu_cache_locations_list.append(lxu_cache_locations)

        # Perform forward
        output = cc(indices, offsets)
        output_ref = cc_ref(indices, offsets)
        assert_torch_equal(output, output_ref)

        # Perform backward
        grad_output = to_device(torch.randn(1, D), use_cpu=False)
        output.backward(grad_output)
        output_ref.backward(grad_output)
        cc.flush()
        assert_cache(
            cc.split_embedding_weights()[0],
            cc_ref.split_embedding_weights()[0],
            stochastic_rounding,
        )


if __name__ == "__main__":
    unittest.main()

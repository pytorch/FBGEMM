#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import Tuple

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity


try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401

    # pyre-ignore[21]
    from test_utils import gpu_unavailable
except Exception:
    if torch.version.hip:
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_hip"
        )
    else:
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings"
        )

    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_cpu"
    )
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_unavailable

    open_source = False

typed_gpu_unavailable: Tuple[bool, str] = gpu_unavailable


@unittest.skipIf(*gpu_unavailable)
@unittest.skipIf(open_source, "Not supported in open source yet")
class MergePooledEmbeddingsTest(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 1, $parameter$max_value =
    #  10)` to decorator factory `hypothesis.given`.
    @given(
        num_ads=st.integers(min_value=1, max_value=10),
        embedding_dimension=st.integers(min_value=1, max_value=32),
        ads_tables=st.integers(min_value=1, max_value=32),
        num_gpus=st.integers(min_value=1, max_value=torch.cuda.device_count()),
        non_default_stream=st.booleans(),
        r=st.randoms(use_true_random=False),
        dim=st.integers(min_value=0, max_value=1),
    )
    # Can instantiate 8 contexts which takes a long time.
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_merge(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        num_ads,
        # pyre-fixme[2]: Parameter must be annotated.
        embedding_dimension,
        # pyre-fixme[2]: Parameter must be annotated.
        ads_tables,
        # pyre-fixme[2]: Parameter must be annotated.
        num_gpus,
        # pyre-fixme[2]: Parameter must be annotated.
        non_default_stream,
        # pyre-fixme[2]: Parameter must be annotated.
        r,
        dim: int,
    ) -> None:
        dst_device = r.randint(0, num_gpus - 1)
        torch.cuda.set_device(dst_device)
        ad_ds = [embedding_dimension * ads_tables for _ in range(num_gpus)]
        batch_indices = torch.zeros(num_ads).long().cuda()
        pooled_ad_embeddings = [
            torch.randn(
                num_ads, ad_d, dtype=torch.float16, device=torch.device(f"cuda:{i}")
            )
            for i, ad_d in enumerate(ad_ds)
        ]
        r.shuffle(pooled_ad_embeddings)

        streams = [torch.cuda.Stream(device=i) for i in range(num_gpus)]
        import contextlib

        uncat_size = batch_indices.size(0) if dim == 1 else ad_ds[0]

        with contextlib.ExitStack() as stack:
            if non_default_stream:
                for stream in streams:
                    stack.enter_context(torch.cuda.stream(stream))
            output = torch.ops.fbgemm.merge_pooled_embeddings(
                pooled_ad_embeddings, uncat_size, batch_indices.device, dim
            )

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def ref(pooled_ad_embeddings, batch_indices):
            return torch.cat([p.cpu() for p in pooled_ad_embeddings], dim=dim)

        output_ref = ref(pooled_ad_embeddings, batch_indices)
        output_cpu = torch.ops.fbgemm.merge_pooled_embeddings(
            [pe.cpu() for pe in pooled_ad_embeddings],
            uncat_size,
            batch_indices.cpu().device,
            dim,
        )
        self.assertEqual(output.device, torch.device(f"cuda:{dst_device}"))
        torch.testing.assert_close(output_ref, output.cpu())
        torch.testing.assert_close(output_ref, output_cpu)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 1, $parameter$max_value =
    #  10)` to decorator factory `hypothesis.given`.
    @given(
        num_inputs=st.integers(min_value=1, max_value=10),
        num_gpus=st.integers(min_value=1, max_value=torch.cuda.device_count()),
        r=st.randoms(use_true_random=False),
    )
    # Can instantiate 8 contexts which takes a long time.
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_all_to_one_device(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        num_inputs,
        # pyre-fixme[2]: Parameter must be annotated.
        num_gpus,
        # pyre-fixme[2]: Parameter must be annotated.
        r,
    ) -> None:
        dst_device = torch.device(f"cuda:{r.randint(0, num_gpus - 1)}")
        with torch.cuda.device(dst_device):
            inputs = [torch.randn(10, 20) for _ in range(num_inputs)]
            cuda_inputs = [
                input.to(f"cuda:{i % num_gpus}") for i, input in enumerate(inputs)
            ]
            cuda_outputs = torch.ops.fbgemm.all_to_one_device(cuda_inputs, dst_device)
            for i, o in zip(inputs, cuda_outputs):
                self.assertEqual(o.device, dst_device)
                torch.testing.assert_close(o.cpu(), i)

    def test_merge_pooled_embeddings_gpu_to_cpu(self) -> None:
        dst_device = torch.device("cpu")
        inputs = [torch.randn(10, 20) for _ in range(4)]
        cuda_inputs = [input.to("cuda:0") for i, input in enumerate(inputs)]
        uncat_size = inputs[0].size(1)
        output = torch.ops.fbgemm.merge_pooled_embeddings(
            cuda_inputs, uncat_size, dst_device, 0
        )
        ref_output = torch.ops.fbgemm.merge_pooled_embeddings(
            inputs, uncat_size, dst_device, 0
        )
        torch.testing.assert_close(output, ref_output)

    def test_merge_pooled_embeddings_cpu_with_different_target_device(self) -> None:
        uncat_size = 2
        pooled_embeddings = [torch.ones(uncat_size, 4), torch.ones(uncat_size, 8)]
        output_meta = torch.ops.fbgemm.merge_pooled_embeddings(
            pooled_embeddings,
            uncat_size,
            torch.device("meta"),
            1,
        )
        self.assertFalse(output_meta.is_cpu)
        self.assertTrue(output_meta.is_meta)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers($parameter$min_value = 1, $parameter$max_value =
    #  10)` to decorator factory `hypothesis.given`.
    @given(
        num_inputs=st.integers(min_value=1, max_value=10),
        num_gpus=st.integers(min_value=1, max_value=torch.cuda.device_count()),
        r=st.randoms(use_true_random=False),
    )
    # Can instantiate 8 contexts which takes a long time.
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_sum_reduce_to_one(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        num_inputs,
        # pyre-fixme[2]: Parameter must be annotated.
        num_gpus,
        # pyre-fixme[2]: Parameter must be annotated.
        r,
    ) -> None:
        dst_device = torch.device(f"cuda:{r.randint(0, num_gpus - 1)}")
        with torch.cuda.device(dst_device):
            inputs = [torch.randn(10, 20) for _ in range(num_inputs)]
            cuda_inputs = [
                input.to(f"cuda:{i % num_gpus}") for i, input in enumerate(inputs)
            ]
            cuda_output = torch.ops.fbgemm.sum_reduce_to_one(cuda_inputs, dst_device)
            self.assertEqual(cuda_output.device, dst_device)
            torch.testing.assert_close(
                cuda_output.cpu(), torch.stack(inputs).sum(dim=0)
            )

    @unittest.skipIf(*typed_gpu_unavailable)
    def test_merge_pooled_embeddings_meta(self) -> None:
        """
        Test that merge_pooled_embeddings works with meta tensor and
        dynamo export mode
        """
        uncat_size = 2
        cat_dim = 1
        pooled_embeddings = [torch.ones(uncat_size, 4), torch.ones(uncat_size, 8)]

        # pyre-fixme[53]: Captured variable `cat_dim` is not annotated.
        # pyre-fixme[53]: Captured variable `pooled_embeddings` is not annotated.
        # pyre-fixme[53]: Captured variable `uncat_size` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def fbgemm_merge_pooled_embeddings(device):
            pooled_embeddings_device = [
                pooled_embedding.to(device) for pooled_embedding in pooled_embeddings
            ]
            return torch.ops.fbgemm.merge_pooled_embeddings(
                pooled_embeddings_device, uncat_size, device, cat_dim
            )

        output_cpu = fbgemm_merge_pooled_embeddings(torch.device("cpu"))
        output_meta = fbgemm_merge_pooled_embeddings(torch.device("meta"))

        self.assertFalse(output_meta.is_cpu)
        self.assertTrue(output_meta.is_meta)

        assert output_meta.shape == output_cpu.shape


if __name__ == "__main__":
    unittest.main()

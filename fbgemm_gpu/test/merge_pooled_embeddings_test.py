#!/usr/bin/env python3

# pyre-unsafe

import unittest

import hypothesis.strategies as st
import torch
from hypothesis import Verbosity, given, settings

try:
    torch.ops.load_library("fbgemm_gpu_py.so")
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings")


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class MergePooledEmbeddingsTest(unittest.TestCase):
    @given(
        num_ads=st.integers(min_value=1, max_value=10),
        embedding_dimension=st.integers(min_value=1, max_value=32),
        ads_tables=st.integers(min_value=1, max_value=32),
        user_tables=st.integers(min_value=1, max_value=32),
        num_gpus=st.integers(min_value=1, max_value=torch.cuda.device_count()),
        non_default_stream=st.booleans(),
        r=st.randoms(use_true_random=False),
    )
    # Can instantiate 8 contexts which takes a long time.
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_merge(
        self,
        num_ads,
        embedding_dimension,
        ads_tables,
        user_tables,
        num_gpus,
        non_default_stream,
        r,
    ) -> None:
        dst_device = r.randint(0, num_gpus - 1)
        torch.cuda.set_device(dst_device)
        ad_ds = [embedding_dimension * ads_tables for _ in range(num_gpus)]
        user_ds = [user_tables * embedding_dimension]
        num_users = 1
        batch_indices = torch.zeros(num_ads).long().cuda()
        pooled_ad_embeddings = [
            torch.randn(
                num_ads, ad_d, dtype=torch.float16, device=torch.device(f"cuda:{i}")
            )
            for i, ad_d in enumerate(ad_ds)
        ]
        r.shuffle(pooled_ad_embeddings)
        pooled_user_embeddings = [
            torch.randn(
                num_users,
                user_d,
                dtype=torch.float16,
                device=torch.cuda.current_device(),
            ).half()
            for i, user_d in enumerate(user_ds)
        ]

        streams = [torch.cuda.Stream(device=i) for i in range(num_gpus)]
        import contextlib

        with contextlib.ExitStack() as stack:
            if non_default_stream:
                for stream in streams:
                    stack.enter_context(torch.cuda.stream(stream))
            output = torch.ops.fbgemm.merge_pooled_embeddings(
                pooled_ad_embeddings, pooled_user_embeddings, batch_indices
            )

        def ref(pooled_ad_embeddings, batch_indices):
            return torch.cat([p.cpu() for p in pooled_ad_embeddings], dim=1)

        output_ref = ref(pooled_ad_embeddings, batch_indices)
        self.assertEqual(output.device, torch.device(f"cuda:{dst_device}"))
        torch.testing.assert_allclose(output_ref, output.cpu())


if __name__ == "__main__":
    unittest.main()

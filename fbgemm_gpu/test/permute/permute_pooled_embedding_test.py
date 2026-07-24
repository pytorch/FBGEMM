# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import inspect
import sys
import unittest

# Import for its side effect of registering the Python abstract (fake-tensor)
# impls for the split permute ops (e.g. fbgemm::permute_pooled_embs_split).
# Without this, generate_opcheck_tests' faketensor / aot_dispatch variants of
# test_permute_pooled_embedding_split_large_grid fail with "could not find the
# abstract impl". The non-split ops register their meta impl in C++, so they do
# not need this.
import fbgemm_gpu.sparse_ops  # noqa: F401
import hypothesis.strategies as st
import torch
import torch.nn as nn
from fbgemm_gpu.permute_pooled_embedding_modules import PermutePooledEmbeddings
from hypothesis import given, settings

from .common import gpu_memory_lt_gb, Net, open_source, typed_gpu_unavailable

if open_source:
    # pyre-ignore[21]
    from test_utils import on_arm_platform, optests
else:
    from fbgemm_gpu.test.test_utils import on_arm_platform, optests

INTERN_MODULE = "fbgemm_gpu.permute_pooled_embedding_modules"
FIXED_EXTERN_API = {
    "PermutePooledEmbeddings": {
        "__init__": ["self", "embs_dims", "permute", "device"],
        "__call__": ["self", "pooled_embs"],
    },
}

FWD_COMPAT_MSG = (
    "WARNING: If this test is failing, you are probably trying "
    "to make changes to a module that has been marked external to PyPer torch packages. "
    "This can break forward compatibility of torch packages on training_platform "
    "(see https://fb.workplace.com/groups/pyper/permalink/808155810065803/). "
    "You need to split up your changes as follows:\n"
    "\t1. Edit your diff so it only contains the changes as optional, and not any usage of the"
    " new optional changes.\n"
    "\t2. Edit FIXED_EXTERN_API in this test so your diff passes the test.\n"
    "\t3. Land your diff and wait for the diff to be picked up by the production version of"
    " fbpkg training_platform.\n"
    "\t4. Once step 3. is complete, you can push the rest of your changes that use the new"
    " changes."
)


@optests.generate_opcheck_tests()
class PooledEmbeddingModulesTest(unittest.TestCase):
    def setUp(self) -> None:
        # Use a deterministic device. Previously setUp was decorated with a
        # hypothesis @given that randomized the device per example, which made
        # the generated opcheck tests intermittently exercise CPU vs GPU
        # registration (flaky). T191384137
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @settings(deadline=None)
    @given(fwd_only=st.booleans())
    def test_permutation(self, fwd_only: bool) -> None:
        net = Net(fwd_only=fwd_only).to(self.device)

        input = torch.tensor([list(range(10))], dtype=torch.float, device=self.device)
        self.assertEqual(
            net.permute_pooled_embeddings(input).view(10).tolist(),
            [6, 7, 8, 9, 0, 1, 5, 2, 3, 4],
        )

    @unittest.skipIf(*on_arm_platform)
    def test_permutation_autograd(self) -> None:
        net = Net().to(self.device)

        input = torch.randn(2, 1).to(self.device)
        input_sum = input.sum().item()

        output = net(input)
        output.sum().backward()

        # check grads for fc1 when permuted, equals to fc2 weights times input_sum
        # pyre-fixme[16]: Optional type has no attribute `view`.
        permute_res = net.permute_pooled_embeddings(net.fc1.weight.grad.view(1, 10))
        permute_ref = input_sum * net.fc2.weight
        torch.testing.assert_close(permute_res, permute_ref, rtol=1e-03, atol=1e-03)

    def test_compatibility(self) -> None:
        members = inspect.getmembers(sys.modules[INTERN_MODULE])
        for name, clazz in members:
            if getattr(clazz, "__module__", None) != INTERN_MODULE:
                continue

            self.assertIn(name, FIXED_EXTERN_API.keys(), FWD_COMPAT_MSG)

            for fn, fixed_params in FIXED_EXTERN_API[name].items():
                current_params = inspect.getfullargspec(getattr(clazz, fn)).args
                self.assertEqual(
                    fixed_params,
                    current_params,
                    msg=f"\nForward incompatible change in {name} : {fn}\n\n"
                    f"{FWD_COMPAT_MSG}",
                )

    def test_pooled_table_batched_embedding(self) -> None:
        num_emb_bags = 5
        num_embeddings = 10
        embedding_dims = [1, 2, 3, 4, 5]
        emb_weight_range = 1
        embedding_bags = [
            nn.EmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dims[i],
                mode="sum",
                sparse=True,
            )
            for i in range(num_emb_bags)
        ]
        for emb_bag in embedding_bags:
            torch.nn.init.uniform_(
                emb_bag.weight,
                -emb_weight_range,
                emb_weight_range,
            )
        indices = [[0], [1, 2], [0, 1, 2], [3, 6], [8]]
        indices = [torch.tensor(i).view(-1, len(i)) for i in indices]
        pooled_embs = [emb_bag(indices[i]) for i, emb_bag in enumerate(embedding_bags)]

        cat_pooled_embs = torch.cat(pooled_embs, dim=1)

        permute_order = [2, 1, 3, 0, 4]

        permute_pooled_embeddings = PermutePooledEmbeddings(
            embedding_dims,
            permute_order,
            device=self.device,
        )
        permuted_pooled_emb = permute_pooled_embeddings(cat_pooled_embs.to(self.device))

        ref_permuted_pooled_emb = [pooled_embs[i] for i in permute_order]
        ref_permuted_pooled_emb = torch.cat(ref_permuted_pooled_emb, dim=1)

        assert torch.allclose(
            ref_permuted_pooled_emb.to(self.device), permuted_pooled_emb
        )

    @unittest.skipIf(*on_arm_platform)
    def test_permutation_autograd_meta(self) -> None:
        """
        Test that permute_pooled_embeddings_autograd works with meta tensor and
        dynamo export mode
        """
        input = torch.randn(2, 1)
        net = Net()

        output_cpu = net(input)
        output_meta = net.to("meta")(input.to("meta"))

        assert output_meta.shape == output_cpu.shape
        assert input.shape == output_meta.shape

    @unittest.skipIf(*typed_gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(8))
    def test_permute_pooled_embedding_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in permute_pooled_embs_kernel
        (include/fbgemm_gpu/layout_transform_ops.cuh:64-117).

        Block: dim3(kMaxThreads=1024). Grid x = div_round_up(T, 32). For
        T = 2**22 + 1 and B = 32, total threads = ceil(T/32) * 32 * 1024
        > 2**32, tripping KernelLauncher::checkThreadCountNotExceeded on
        ROCm pre-fix.

        Pre-fix: AMD launch fails. Post-fix: passes (kernel grid-strides
        over t).
        """
        T = (1 << 22) + 1
        B = 32
        embs_dims = [1] * T
        permute = list(range(T))

        pooled_embs = torch.zeros(
            B, sum(embs_dims), dtype=torch.float32, device=self.device
        )
        module = PermutePooledEmbeddings(embs_dims, permute, device=self.device)
        result = module(pooled_embs)

        self.assertEqual(result.shape, pooled_embs.shape)
        self.assertTrue(torch.all(result == 0).item())

        # Tier-A correctness check at small scale with non-trivial permute
        # and non-zero values. Pure Python reference: per-feature gather.
        small_embs_dims = [3, 5, 2, 4]
        small_T = len(small_embs_dims)
        # Reverse permute: [3, 2, 1, 0].
        small_permute = list(range(small_T))[::-1]
        small_B = 4
        small_total_D = sum(small_embs_dims)
        # Distinct values across rows.
        small_pooled_cpu = torch.arange(
            small_B * small_total_D, dtype=torch.float32
        ).view(small_B, small_total_D)

        small_module = PermutePooledEmbeddings(
            small_embs_dims, small_permute, device=self.device
        )
        small_result = small_module(small_pooled_cpu.to(self.device))

        # Python reference: split per-feature, gather by permute.
        small_segments = list(small_pooled_cpu.split(small_embs_dims, dim=1))
        small_ref = torch.cat([small_segments[p] for p in small_permute], dim=1)

        torch.testing.assert_close(small_result.cpu(), small_ref)

    @unittest.skipIf(*typed_gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(8))
    def test_permute_pooled_embedding_split_large_grid(self) -> None:
        """
        Reproduces the HIP grid-overflow bug in the split frontend of
        permute_pooled_embs_kernel
        (src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_split.cu).

        Same kernel as test_permute_pooled_embedding_large_grid, exercised
        through torch.ops.fbgemm.permute_pooled_embs_auto_grad_split.
        """
        T = (1 << 22) + 1
        B = 32
        embs_dims = [1] * T
        permute = list(range(T))
        inv_permute = list(range(T))

        offsets = torch.zeros(T + 1, dtype=torch.int64, device=self.device)
        offsets[1:] = torch.cumsum(
            torch.tensor(embs_dims, dtype=torch.int64, device=self.device),
            dim=0,
        )
        permute_t = torch.tensor(permute, dtype=torch.int64, device=self.device)
        inv_permute_t = torch.tensor(inv_permute, dtype=torch.int64, device=self.device)

        pooled_embs = torch.zeros(
            B,
            int(offsets[-1].item()),
            dtype=torch.float32,
            device=self.device,
        )
        result = torch.ops.fbgemm.permute_pooled_embs_auto_grad_split(
            pooled_embs,
            offsets,
            permute_t,
            offsets,
            inv_permute_t,
        )

        self.assertEqual(result.shape, pooled_embs.shape)
        self.assertTrue(torch.all(result == 0).item())


if __name__ == "__main__":
    unittest.main()

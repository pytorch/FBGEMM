# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import fbgemm_gpu

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.utils.loader import load_torch_module

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if not open_source:
    load_torch_module(
        "//deeplearning/fbgemm/fbgemm_gpu:dram_kv_embedding_inference",
    )


@unittest.skipIf(open_source, "Not supported in open source yet")
class DramKvInferenceTest(unittest.TestCase):
    def test_serialize(self) -> None:
        num_shards = 32
        uniform_init_lower: float = -0.01
        uniform_init_upper: float = 0.01
        evict_trigger_mode: int = 1

        kv_embedding_cache = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            num_shards, uniform_init_lower, uniform_init_upper, evict_trigger_mode
        )
        serialized_result = kv_embedding_cache.serialize()

        self.assertEqual(serialized_result[0][0], num_shards)
        self.assertEqual(serialized_result[0][1], evict_trigger_mode)

        self.assertEqual(serialized_result[1][0], uniform_init_lower)
        self.assertEqual(serialized_result[1][1], uniform_init_upper)

    def test_serialize_deserialize(self) -> None:
        num_shards = 32
        uniform_init_lower: float = -0.01
        uniform_init_upper: float = 0.01
        evict_trigger_mode: int = 1

        kv_embedding_cache = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            num_shards, uniform_init_lower, uniform_init_upper, evict_trigger_mode
        )
        serialized_result = kv_embedding_cache.serialize()

        kv_embedding_cache_2 = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            0, 0.0, 0.0, 0
        )
        kv_embedding_cache_2.deserialize(serialized_result)

        self.assertEqual(str(serialized_result), str(kv_embedding_cache_2.serialize()))

    def test_set_get_embeddings(self) -> None:
        num_shards = 32
        uniform_init_lower: float = 0.0
        uniform_init_upper: float = 0.0
        evict_trigger_mode: int = 0

        kv_embedding_cache = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            num_shards, uniform_init_lower, uniform_init_upper, evict_trigger_mode
        )
        kv_embedding_cache.init(
            [(20, 4, SparseType.INT8.as_int())],
            8,
            4,
        )

        kv_embedding_cache.set_embeddings(
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            torch.tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                dtype=torch.uint8,
            ),
        )

        embs = kv_embedding_cache.get_embeddings(
            torch.tensor([1, 4, 3, 0, 5, 2], dtype=torch.int64),
        )
        assert torch.equal(
            embs[:, :4],
            torch.tensor(
                [
                    [5, 6, 7, 8],
                    [0, 0, 0, 0],
                    [13, 14, 15, 16],
                    [1, 2, 3, 4],
                    [0, 0, 0, 0],
                    [9, 10, 11, 12],
                ],
                dtype=torch.uint8,
            ),
        )

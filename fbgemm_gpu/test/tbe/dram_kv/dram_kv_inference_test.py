# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import threading
import time
import unittest
from time import sleep
from typing import List

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

        kv_embedding_cache = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            num_shards, uniform_init_lower, uniform_init_upper
        )
        serialized_result = kv_embedding_cache.serialize()

        self.assertEqual(serialized_result[0][0], num_shards)

        self.assertEqual(serialized_result[1][0], uniform_init_lower)
        self.assertEqual(serialized_result[1][1], uniform_init_upper)

    def test_serialize_deserialize(self) -> None:
        num_shards = 32
        uniform_init_lower: float = -0.01
        uniform_init_upper: float = 0.01

        kv_embedding_cache = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            num_shards, uniform_init_lower, uniform_init_upper
        )
        serialized_result = kv_embedding_cache.serialize()

        kv_embedding_cache_2 = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            0, 0.0, 0.0
        )
        kv_embedding_cache_2.deserialize(serialized_result)

        self.assertEqual(str(serialized_result), str(kv_embedding_cache_2.serialize()))

    def test_set_get_embeddings(self) -> None:
        num_shards = 32
        uniform_init_lower: float = 0.0
        uniform_init_upper: float = 0.0

        kv_embedding_cache = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            num_shards, uniform_init_lower, uniform_init_upper
        )
        kv_embedding_cache.init(
            [(20, 4, SparseType.INT8.as_int())],
            8,
            4,
            torch.tensor([0, 32], dtype=torch.int64),
        )

        kv_embedding_cache.set_embeddings(
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            torch.tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                dtype=torch.uint8,
            ),
        )

        embs = kv_embedding_cache.get_embeddings(
            torch.tensor([1, 3, 0, 2, 4, 5], dtype=torch.int64),
        )
        assert torch.equal(
            embs[:4, :4],
            torch.tensor(
                [
                    [5, 6, 7, 8],
                    [13, 14, 15, 16],
                    [1, 2, 3, 4],
                    [9, 10, 11, 12],
                ],
                dtype=torch.uint8,
            ),
        )

        def equal_one_of(t1: torch.Tensor, t2: List[torch.Tensor]) -> bool:
            any_equal = False
            for t in t2:
                any_equal = torch.equal(t1, t)
                if any_equal:
                    return any_equal
            return any_equal

        possible_embs = [
            torch.tensor([5, 6, 7, 8], dtype=torch.uint8),
            torch.tensor([13, 14, 15, 16], dtype=torch.uint8),
            torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
            torch.tensor([9, 10, 11, 12], dtype=torch.uint8),
            torch.tensor([0, 0, 0, 0], dtype=torch.uint8),
        ]
        self.assertTrue(equal_one_of(embs[4, :4], possible_embs))
        self.assertTrue(equal_one_of(embs[5, :4], possible_embs))

    def test_inplace_update(self) -> None:
        num_shards = 1
        uniform_init_lower: float = 0.0
        uniform_init_upper: float = 0.0

        kv_embedding_cache = torch.classes.fbgemm.DramKVEmbeddingInferenceWrapper(
            num_shards, uniform_init_lower, uniform_init_upper
        )
        kv_embedding_cache.init(
            [(20, 4, SparseType.INT8.as_int())],
            8,
            4,
            torch.tensor([0, 20], dtype=torch.int64),
        )
        init_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        init_weights = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=torch.uint8,
        )
        kv_embedding_cache.set_embeddings(
            init_ids,
            init_weights,
        )
        full_ids: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)

        def equal_one_of(t1: torch.Tensor, t2: List[torch.Tensor]) -> bool:
            any_equal = False
            for t in t2:
                any_equal = torch.equal(t1, t)
                if any_equal:
                    return any_equal
            return any_equal

        possible_embs = [
            torch.tensor([5, 6, 7, 8], dtype=torch.uint8),
            torch.tensor([17, 18, 19, 20], dtype=torch.uint8),
            torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
            torch.tensor([21, 22, 23, 24], dtype=torch.uint8),
            torch.tensor([0, 0, 0, 0], dtype=torch.uint8),
        ]

        import logging

        logging.basicConfig(level=logging.INFO, format="%(threadName)s: %(message)s")

        reader_start_event = threading.Event()

        reader_failed_event = threading.Event()

        def reader_thread() -> None:  # pyre-ignore
            itr = 0
            reader_start_event.wait()
            try:
                while itr < 100:
                    embs = kv_embedding_cache.get_embeddings(full_ids)
                    self.assertEqual(embs.size(0), 6)
                    self.assertEqual(embs.size(1), 8)
                    self.assertTrue(
                        torch.equal(
                            embs[0][:4],
                            torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
                        ),
                        f"id0: {embs[0][:4]} failed",
                    )
                    self.assertTrue(
                        torch.equal(
                            embs[1][:4],
                            torch.tensor([5, 6, 7, 8], dtype=torch.uint8),
                        ),
                        f"id1: {embs[1][:4]} failed",
                    )
                    self.assertTrue(
                        equal_one_of(embs[2][:4], possible_embs + [9, 10, 11, 12]),
                        f"id2: {embs[2][:4]} failed",
                    )
                    self.assertTrue(
                        equal_one_of(embs[3][:4], possible_embs + [13, 14, 15, 16]),
                        f"id3: {embs[3][:4]} failed",
                    )
                    self.assertTrue(
                        equal_one_of(embs[4][:4], possible_embs),
                        f"id3: {embs[4][:4]} failed",
                    )
                    self.assertTrue(
                        equal_one_of(embs[5][:4], possible_embs),
                        f"id3: {embs[5][:4]} failed",
                    )
                    itr += 1
            except Exception as e:
                reader_failed_event.set()
                raise e

        reader_thread = threading.Thread(target=reader_thread, name="ReaderThread")
        reader_thread.start()
        sleep(1)

        reader_start_event.set()
        sleep(0.001)  # 10 us to make sure reader thread is reading
        current_ts_sec = int(time.time())

        ipu_ids = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        ipu_weights = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [17, 18, 19, 20], [21, 22, 23, 24]],
            dtype=torch.uint8,
        )
        kv_embedding_cache.set_embeddings(
            ipu_ids,
            ipu_weights,
            current_ts_sec,
        )

        kv_embedding_cache.trigger_evict(current_ts_sec)
        kv_embedding_cache.wait_evict_completion()

        embs = kv_embedding_cache.get_embeddings(
            torch.tensor([1, 4, 0, 5, 2, 3], dtype=torch.int64),
        )
        assert torch.equal(
            embs[:4, :4],
            torch.tensor(
                [
                    [5, 6, 7, 8],
                    [17, 18, 19, 20],
                    [1, 2, 3, 4],
                    [21, 22, 23, 24],
                ],
                dtype=torch.uint8,
            ),
        )
        self.assertTrue(equal_one_of(embs[4, :4], possible_embs))
        self.assertTrue(equal_one_of(embs[5, :4], possible_embs))
        reader_thread.join()
        self.assertFalse(reader_failed_event.is_set())

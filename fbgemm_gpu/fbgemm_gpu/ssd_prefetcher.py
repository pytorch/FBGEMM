# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from collections import deque
from typing import List, Optional, Tuple

import torch

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_utils import (
    random_quant_scaled_tensor,
    rounded_row_size_in_bytes,
    unpadded_row_size_in_bytes,
)
from torch import Tensor


class SSDPrefetcher:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def prefetch(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        table_placement: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Fetch embedding table rows specified by the indicies,
        push the tuple of [Tensor, Optional[Tensor], Optional[Tensor]] into a
        queue of this prefetcher, and return a reference to the tuple. For meaning of
        elements in the tuple, see the @return section.

        @param indices Indices of the rows that we want to fetch
        @param offsets Pooling offsets
        @param table_placement Indices of the tables that we want to query from the storage

        @return A tuple of tensors:
            The first tensor will be the fetched embedding table rows (required).
            The second tensor will be the remapped indices (optional). Its size should be equal to the
              size of `indices`. If the second element is None, it means the fetched rows should be
              used sequentially by the TBE module (equivalent to `indices = torch.arange(0, indices.numel())`).
            The third tensor will be the remapped weights offsets (optional). Its size should be equal
            to the number of tables involved. Each value in this tensor indicates each individual table's
            offset (in bytes, since weights are stored as int8 regardless of the actual type) in the returned
            embedding tables tensor. If the third element is None, the returned embedding tables tensor
            will be treated as a single table (equivalent to `weights_offsets = [0]`).
        """
        pass

    @abc.abstractmethod
    def get(
        self,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return and remove a tuple of [Tensor, Optional[Tensor], Optional[Tensor]] from the queue of this prefetcher,
        which is the result of the prefetch() call, in FIFO order. If the queue is empty, raise IndexError.

        @return A tuple of tensors. Please see the @return section of prefetch() for the definition.
        """
        pass

    @abc.abstractmethod
    def is_empty(self) -> bool:
        """
        Return whether the queue of this prefetcher is empty.

        @return True if the queue is empty, False otherwise.
        """
        return True

    def split_embedding_weights(
        self, split_scale_shifts: bool = True
    ) -> List[Tuple[Tensor, Optional[Tensor]]]:
        """
        Provide a view of the embedding tables with quantized values and scales/bias
        split into two tensors.  Please implement this method if you would like to use
        the SSD prefetcher in tests or benchmarks.

        @param split_scale_shifts Whether to split the scales and bias from the embedding
               table row.

        @return A list of tuples.  Each tuple contains the quantized embedding table row
                and the scales and bias.  The scales and bias will be None if
                `split_scale_shifts` is False (meaning not splitting).
        """
        return []


class DummyPrefetcher(SSDPrefetcher):
    """
    A dummy fake SSD prefetcher intended to test if the TBE module can work with
    the SSD prefetcher interface. It simply takes a range of embedding table shapes
    and generate random weights based on the shapes during initialization, and
    returns the picked rows when the prefetch() function is called.
    """

    def _generate_weights(
        self, embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]]
    ) -> None:
        for _, rows, dim, weight_ty, _ in embedding_specs:
            weights = random_quant_scaled_tensor(
                torch.Size(
                    (
                        rows,
                        rounded_row_size_in_bytes(
                            dim,
                            weight_ty,
                            self.row_alignment,
                            self.scale_bias_size_in_bytes,
                        ),
                    ),
                ),
                device=torch.device("cpu"),
            )
            self.weights.append(weights)
            self.nrows.append(self.nrows[-1] + rows)

    def split_embedding_weights(
        self, split_scale_shifts: bool = True
    ) -> List[Tuple[Tensor, Optional[Tensor]]]:
        splits: List[Tuple[Tensor, Optional[Tensor]]] = []
        for i, (_, rows, dim, weight_ty, _) in enumerate(self.embedding_specs):
            weights_shifts = self.weights[i].detach()
            if split_scale_shifts:
                # remove the padding at the end of each row.
                weights_shifts = weights_shifts[
                    :,
                    : unpadded_row_size_in_bytes(
                        dim, weight_ty, self.scale_bias_size_in_bytes
                    ),
                ]
                if (
                    weight_ty == SparseType.INT8
                    or weight_ty == SparseType.INT4
                    or weight_ty == SparseType.INT2
                ):
                    splits.append(
                        (
                            weights_shifts[:, self.scale_bias_size_in_bytes :],
                            weights_shifts[:, : self.scale_bias_size_in_bytes],
                        )
                    )
                else:
                    assert (
                        weight_ty == SparseType.FP8
                        or weight_ty == SparseType.FP16
                        or weight_ty == SparseType.FP32
                    )
                    splits.append(
                        (
                            weights_shifts,
                            None,
                        )
                    )
            else:
                splits.append((weights_shifts, None))

        return splits

    def __init__(
        self,
        embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]],
        row_alignment: int = 1,
        scale_bias_size_in_bytes: int = DEFAULT_SCALE_BIAS_SIZE_IN_BYTES,
    ) -> None:
        self.embedding_specs = embedding_specs
        self.row_alignment = row_alignment
        self.scale_bias_size_in_bytes = scale_bias_size_in_bytes
        self.weights: List[torch.Tensor] = []
        self.nrows: List[int] = [0]
        self._generate_weights(embedding_specs)
        self.queue: deque[Tuple[Tensor, Optional[Tensor], Optional[Tensor]]] = deque()

    def prefetch(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        table_placement: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if indices.numel() == 0:
            return (torch.tensor([], dtype=torch.uint8, device="cpu"), None, None)
        assert indices.numel() == table_placement.numel()
        fetched_rows = []
        new_indices = []
        weights_offsets = [0]
        prev_table_id = None
        idx = 0
        bytes_added = 0
        for i, tbl in enumerate(table_placement):
            if prev_table_id is None:
                prev_table_id = tbl
            if prev_table_id != tbl:
                idx = 0
                prev_table_id = tbl
                weights_offsets.append(bytes_added)
            fetched_rows.append(self.weights[tbl][indices[i]])
            new_indices.append(idx)
            idx += 1
            bytes_added += self.weights[tbl][indices[i]].numel()

        fetched_tables = (
            torch.cat(fetched_rows),
            torch.tensor(new_indices, dtype=indices.dtype, device="cpu"),
            torch.tensor(weights_offsets, dtype=torch.int64, device="cpu"),
        )
        self.queue.append(fetched_tables)
        return fetched_tables

    def get(
        self,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if len(self.queue) == 0:
            raise IndexError("Prefetcher's queue is empty")
        return self.queue.popleft()

    def is_empty(self) -> bool:
        return len(self.queue) == 0


class SSDPrefetcherTestSetting:

    def __init__(self):
        self.register_prefetcher_at_tbe_init: bool = True
        self.add_ssd_placement_at_tbe_init: bool = True
        self.prefetch_before_forward: bool = False

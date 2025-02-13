# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Optional, Tuple

import torch
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
        Fetch embedding table rows specified by the indicies
        And return the requested rows and remapped indcies as two Tensors

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

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import abc

from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor


@dataclass(frozen=True)
class TBEInfo:
    """
    contains selective TBE info used for multiplexing. For more info, check https://fburl.com/code/ljnd6j65

    Args:
        table_names: table names within the tbe
        table_heights: table heights (hashsize)
        tbe_uuid: a unique identifier for the TBE
        feature_table_map: feature to table map
    """

    table_names: List[str]
    table_heights: List[int]
    tbe_uuid: str
    feature_table_map: List[int]


@dataclass(frozen=True)
class TBEInputInfo:
    """
    indices: A 1D-tensor that contains indices to be looked up
        from all embedding table.
    offsets: A 1D-tensor that conatins offsets of indices.
    batch_size_per_feature_per_rank: An optional 2D-tensor that contains batch sizes for every rank and
    every feature. this is needed to support VBE.
    """

    indices: Tensor
    offsets: Tensor
    batch_size_per_feature_per_rank: Optional[List[List[int]]] = None


class TBEInputMultiplexer(abc.ABC):
    """
    Interface for multiplex TBE input data out, actual implementation may store the data to files
    """

    @abc.abstractmethod
    def should_run(self, step: int) -> bool:
        """
        To check if should run at this step
        Args:
            step: the current step
        Returns:
            True if should run, otherwise False
        """
        pass

    @abc.abstractmethod
    def run(
        self,
        tbe_input_info: TBEInputInfo,
    ) -> None:
        """
        To run the tbe input multiplex, and this is called for every batch that needs to be dumped
        Args:
            tbe_input_info: tbe input info that contains all the necessary info for further processing
        """
        pass


@dataclass(frozen=True)
class TBEInputMultiplexerConfig:
    """
    Configuration for TBEInputMultiplexer
    """

    # first batch to start run, -1 means no run
    start_batch: int = -1
    # total batch to multiplex
    total_batch: int = 0

    def create_tbe_input_multiplexer(
        self,
        tbe_info: TBEInfo,
    ) -> Optional[TBEInputMultiplexer]:
        assert (
            self.start_batch == -1
        ), "Cannot specify monitor_start_batch without an actual implementation."
        return None

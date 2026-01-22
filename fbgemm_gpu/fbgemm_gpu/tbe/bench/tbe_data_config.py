#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import json
import logging
from typing import Any, List, Optional, Tuple

import torch

# fmt:skip
from fbgemm_gpu.tbe.utils.common import get_device
from .tbe_data_config_param_models import (
    BatchParams,
    IndicesParams,
    PoolingParams,
)  # usort:skip

try:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/src/tbe/eeg:indices_generator"
    )
except Exception:
    pass


@dataclasses.dataclass(frozen=True)
class TBEDataConfig:
    T: int
    E: int
    D: int
    mixed_dim: bool
    weighted: bool
    batch_params: BatchParams
    indices_params: IndicesParams
    pooling_params: PoolingParams
    use_cpu: bool = False
    Es: Optional[list[int]] = None
    Ds: Optional[list[int]] = None
    max_indices: Optional[int] = None
    embedding_specs: Optional[List[Tuple[int, int]]] = None
    feature_table_map: Optional[List[int]] = None
    """
    Configuration for TBE (Table Batched Embedding) benchmark data collection and generation.

    This dataclass holds parameters required to generate synthetic data for
    TBE benchmarking, including table specifications, batch parameters, indices
    distribution parameters, and pooling parameters.

    Args:
        T (int): Number of embedding tables (features). Must be positive.
        E (int): Number of rows in the embedding table (feature). If T > 1, this
            represents the averaged number of rows across all features.
        D (int): Target embedding dimension for a table (feature), i.e., number of
            columns. If T > 1, this represents the averaged dimension across
            all features.
        mixed_dim (bool): If True, generate embeddings with mixed dimensions
            across tables (features). This is automatically set to True if D is provided
            as a list with non-uniform values.
        weighted (bool): If True, the lookup rows are weighted (per-sample
            weights). The weights will be generated as FP32 tensors.
        batch_params (BatchParams): Parameters controlling batch generation.
            Contains:
            (1) `B` = target batch size (number of batch lookups per features)
            (2) `sigma_B` = optional standard deviation for variable batch size
            (3) `vbe_distribution` = distribution type ("normal" or "uniform")
            (4) `vbe_num_ranks` = number of ranks for variable batch size
            (5) `Bs` = per-feature batch sizes
        indices_params (IndicesParams): Parameters controlling index generation
            following a Zipf distribution. Contains:
            (1) `heavy_hitters` = probability density map for hot indices
            (2) `zipf_q` = q parameter in Zipf distribution (x+q)^{-s}
            (3) `zipf_s` = s parameter (alpha) in Zipf distribution
            (4) `index_dtype` = optional dtype for indices tensor
            (5) `offset_dtype` = optional dtype for offsets tensor
        pooling_params (PoolingParams): Parameters controlling pooling behavior.
            Contains:
            (1) `L` = target bag size (pooling factor, indices per lookup)
            (2) `sigma_L` = optional standard deviation for variable bag size
            (3) `length_distribution` = distribution type ("normal" or "uniform")
            (4) `Ls` = per-feature bag sizes
        use_cpu (bool = False): If True, force generated tensors to be placed
            on CPU instead of the default compute device.
        Es (Optional[List[int]] = None): Number of embeddings (rows) for each
            individual embedding feature. If provided, must have length equal
            to T. All elements must be positive.
        Ds (Optional[List[int]] = None): Target embedding dimension (columns)
            for each individual feature. If provided, must have length equal
            to T. All elements must be positive.
        max_indices (Optional[int] = None): Maximum number of indices for
            bounds checking. If Es is provided as a list and max_indices is
            None, it is automatically computed as sum(Es) - 1.
        embedding_specs (Optional[List[Tuple[int, int]]] = None): A list of
            embedding specs consisting of a list of tuples of (num_rows, embedding_dim).
            See https://fburl.com/tbe_embedding_specs for details.
        feature_table_map (Optional[List[int]] = None): An optional list that
            specifies feature-table mapping. feature_table_map[i] indicates the
            physical embedding table that feature i maps to.
    """

    def __post_init__(self) -> None:
        if isinstance(self.D, list):
            object.__setattr__(self, "mixed_dim", len(set(self.D)) > 1)
        if isinstance(self.E, list) and self.max_indices is None:
            object.__setattr__(self, "max_indices", sum(self.E) - 1)
        self.validate()

    @staticmethod
    def complex_fields() -> dict[str, Any]:
        return {
            "batch_params": BatchParams,
            "indices_params": IndicesParams,
            "pooling_params": PoolingParams,
        }

    @classmethod
    # pyre-ignore [3]
    def from_dict(cls, data: dict[str, Any]):
        for field, Type in cls.complex_fields().items():
            if not isinstance(data[field], Type):
                data[field] = Type.from_dict(data[field])
        return cls(**data)

    @classmethod
    # pyre-ignore [3]
    def from_json(cls, data: str):
        raw = json.loads(data)
        allowed = {f.name for f in dataclasses.fields(cls)}
        existing_fields = {k: v for k, v in raw.items() if k in allowed}
        missing_fields = allowed - set(existing_fields.keys())
        unknown_fields = set(raw.keys()) - allowed
        if missing_fields:
            logging.warning(
                f"TBEDataConfig.from_json: Missing expected fields not loaded: {sorted(missing_fields)}"
            )
        if unknown_fields:
            logging.info(
                f"TBEDataConfig.from_json: Ignored unknown fields from input: {sorted(unknown_fields)}"
            )
        return cls.from_dict(existing_fields)

    def dict(self) -> dict[str, Any]:
        tmp = dataclasses.asdict(self)
        for field in TBEDataConfig.complex_fields().keys():
            tmp[field] = self.__dict__[field].dict()
        return tmp

    def json(self, format: bool = False) -> str:
        return json.dumps(self.dict(), indent=(2 if format else -1), sort_keys=True)

    # pyre-ignore [3]
    def validate(self):
        # NOTE: Add validation logic here
        assert self.T > 0, "T must be positive"
        assert self.E > 0, "E must be positive"
        if self.Es is not None:
            assert all(e > 0 for e in self.Es), "All elements in Es must be positive"
        assert self.D > 0, "D must be positive"
        if self.Ds is not None:
            assert all(d > 0 for d in self.Ds), "All elements in Ds must be positive"
        if isinstance(self.Es, list) and isinstance(self.Ds, list):
            assert (
                len(self.Es) == len(self.Ds) == self.T
            ), "Lengths of Es, Lengths of Ds, and T must be equal"
            if self.max_indices is not None:
                assert self.max_indices == (
                    sum(self.Es) - 1
                ), "max_indices must be equal to sum(Es) - 1"
        self.batch_params.validate()
        if self.batch_params.Bs is not None:
            assert (
                len(self.batch_params.Bs) == self.T
            ), f"Length of Bs must be equal to T. Expected: {self.T}, but got: {len(self.batch_params.Bs)}"
        self.indices_params.validate()
        self.pooling_params.validate()
        if self.pooling_params.Ls is not None:
            assert (
                len(self.pooling_params.Ls) == self.T
            ), f"Length of Ls must be equal to T. Expected: {self.T}, but got: {len(self.pooling_params.Ls)}"
        return self

    def variable_B(self) -> bool:
        return self.batch_params.sigma_B is not None

    def variable_L(self) -> bool:
        return self.pooling_params.sigma_L is not None

    def _new_weights(self, size: int) -> Optional[torch.Tensor]:
        # Per-sample weights will always be FP32
        return None if not self.weighted else torch.randn(size, device=get_device())

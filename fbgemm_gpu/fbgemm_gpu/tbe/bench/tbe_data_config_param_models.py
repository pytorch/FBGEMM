#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import json
from typing import Any, Dict, Optional

import torch


def str_to_int_dtype(dtype: str) -> torch.dtype:
    if dtype == "torch.int32":
        return torch.int32
    elif dtype == "torch.int64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@dataclasses.dataclass(frozen=True, eq=False)
class IndicesParams:
    # Heavy hitters for the Zipf distribution, i.e. a probability density map
    # for the most hot indices.  There should not ever be more than 100
    # elements, and currently it is limited to 20 entries (kHeavyHittersMaxSize)
    heavy_hitters: torch.Tensor
    # zipf*: parameters for the Zipf distribution (x+q)^{-s}
    zipf_q: float
    # zipf_s is synonymous with alpha in the literature
    zipf_s: float
    # [Optional] dtype for indices tensor
    index_dtype: Optional[torch.dtype] = None
    # [Optional] dtype for offsets tensor
    offset_dtype: Optional[torch.dtype] = None

    @classmethod
    # pyre-ignore [3]
    def from_dict(cls, data: Dict[str, Any]):
        if not isinstance(data["heavy_hitters"], torch.Tensor):
            data["heavy_hitters"] = torch.tensor(
                data["heavy_hitters"], dtype=torch.float32
            )
            data["index_dtype"] = str_to_int_dtype(data["index_dtype"])
            data["offset_dtype"] = str_to_int_dtype(data["offset_dtype"])
        return cls(**data)

    @classmethod
    # pyre-ignore [3]
    def from_json(cls, data: str):
        return cls.from_dict(json.loads(data))

    def dict(self) -> Dict[str, Any]:
        # https://stackoverflow.com/questions/73735974/convert-dataclass-of-dataclass-to-json-string
        tmp = dataclasses.asdict(self)
        # Convert tensor to list for JSON serialization
        tmp["heavy_hitters"] = self.heavy_hitters.tolist()
        tmp["index_dtype"] = str(self.index_dtype)
        tmp["offset_dtype"] = str(self.offset_dtype)
        return tmp

    def json(self, format: bool = False) -> str:
        return json.dumps(self.dict(), indent=(2 if format else -1), sort_keys=True)

    # pyre-ignore [2]
    def __eq__(self, other) -> bool:
        return (
            (self.zipf_q, self.zipf_s, self.index_dtype, self.offset_dtype)
            == (other.zipf_q, other.zipf_s, other.index_dtype, other.offset_dtype)
        ) and bool((self.heavy_hitters - other.heavy_hitters).abs().max() < 1e-6)

    # pyre-ignore [3]
    def validate(self):
        assert self.zipf_q > 0, "zipf_q must be positive"
        assert self.zipf_s > 0, "zipf_s must be positive"
        assert self.index_dtype is None or self.index_dtype in [
            torch.int32,
            torch.int64,
        ], "index_dtype must be one of [torch.int32, torch.int64]"
        assert self.offset_dtype is None or self.offset_dtype in [
            torch.int32,
            torch.int64,
        ], "offset_dtype must be one of [torch.int32, torch.int64]"
        return self


@dataclasses.dataclass(frozen=True)
class BatchParams:
    # Target batch size, i.e. number of batch lookups per table
    B: int
    # [Optional] Standard deviation of B (for variable batch size configuration)
    sigma_B: Optional[int] = None
    # [Optional] Distribution of batch sizes (normal, uniform)
    vbe_distribution: Optional[str] = "normal"
    # Number of ranks for variable batch size generation
    vbe_num_ranks: Optional[int] = None

    @classmethod
    # pyre-ignore [3]
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)

    @classmethod
    # pyre-ignore [3]
    def from_json(cls, data: str):
        return cls.from_dict(json.loads(data))

    def dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def json(self, format: bool = False) -> str:
        return json.dumps(self.dict(), indent=(2 if format else -1), sort_keys=True)

    # pyre-ignore [3]
    def validate(self):
        assert self.B > 0, "B must be positive"
        assert not self.sigma_B or self.sigma_B > 0, "sigma_B must be positive"
        assert (
            self.vbe_num_ranks is None or self.vbe_num_ranks > 0
        ), "vbe_num_ranks must be positive"
        assert self.vbe_distribution is None or self.vbe_distribution in [
            "normal",
            "uniform",
        ], "vbe_distribution must be one of [normal, uniform]"
        return self


@dataclasses.dataclass(frozen=True)
class PoolingParams:
    # Target bag size, i.e. pooling factor, or number of indices per batch lookup
    L: int
    # [Optional] Standard deviation of L (for variable bag size configuration)
    sigma_L: Optional[int] = None
    # [Optional] Distribution of embedding sequence lengths (normal, uniform)
    length_distribution: Optional[str] = "normal"

    @classmethod
    # pyre-ignore [3]
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)

    @classmethod
    # pyre-ignore [3]
    def from_json(cls, data: str):
        return cls.from_dict(json.loads(data))

    def dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def json(self, format: bool = False) -> str:
        return json.dumps(self.dict(), indent=(2 if format else -1), sort_keys=True)

    # pyre-ignore [3]
    def validate(self):
        assert self.L > 0, "L must be positive"
        assert not self.sigma_L or self.sigma_L > 0, "sigma_L must be positive"
        assert self.length_distribution is None or self.length_distribution in [
            "normal",
            "uniform",
        ], "length_distribution must be one of [normal, uniform]"
        return self

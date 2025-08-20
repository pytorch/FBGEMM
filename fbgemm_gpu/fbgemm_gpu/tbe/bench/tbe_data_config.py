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

from fbgemm_gpu.tbe.utils.common import get_device

from .tbe_data_config_param_models import BatchParams, IndicesParams, PoolingParams

try:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu/src/tbe/eeg:indices_generator"
    )
except Exception:
    pass


@dataclasses.dataclass(frozen=True)
class TBEDataConfig:
    # Number of tables
    T: int
    # Number of rows in the embedding table
    E: int
    # Target embedding dimension for a table (number of columns)
    D: int
    # Generate mixed dimensions if true
    mixed_dim: bool
    # Whether the lookup rows are weighted or not
    weighted: bool
    # Batch parameters
    batch_params: BatchParams
    # Indices parameters
    indices_params: IndicesParams
    # Pooling parameters
    pooling_params: PoolingParams
    # Force generated tensors to be on CPU
    use_cpu: bool = False

    @staticmethod
    def complex_fields() -> Dict[str, Any]:
        return {
            "batch_params": BatchParams,
            "indices_params": IndicesParams,
            "pooling_params": PoolingParams,
        }

    @classmethod
    # pyre-ignore [3]
    def from_dict(cls, data: Dict[str, Any]):
        for field, Type in cls.complex_fields().items():
            if not isinstance(data[field], Type):
                data[field] = Type.from_dict(data[field])
        return cls(**data)

    @classmethod
    # pyre-ignore [3]
    def from_json(cls, data: str):
        return cls.from_dict(json.loads(data))

    def dict(self) -> Dict[str, Any]:
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
        assert self.D > 0, "D must be positive"
        self.batch_params.validate()
        self.indices_params.validate()
        self.pooling_params.validate()
        return self

    def variable_B(self) -> bool:
        return self.batch_params.sigma_B is not None

    def variable_L(self) -> bool:
        return self.pooling_params.sigma_L is not None

    def _new_weights(self, size: int) -> Optional[torch.Tensor]:
        # Per-sample weights will always be FP32
        return None if not self.weighted else torch.randn(size, device=get_device())

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from fbgemm_gpu.tbe.utils.common import get_device, round_up
from fbgemm_gpu.tbe.utils.requests import (
    generate_batch_sizes_from_stats,
    generate_pooling_factors_from_stats,
    get_table_batched_offsets_from_dense,
    maybe_to_dtype,
    TBERequest,
)

from .config_param_models import BatchParams, IndicesParams, PoolingParams

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
    # Whether the table is weighted or not
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
        # per sample weights will always be FP32
        return None if not self.weighted else torch.randn(size, device=get_device())

    def _generate_batch_sizes(self) -> Tuple[List[int], Optional[List[List[int]]]]:
        if self.variable_B():
            assert (
                self.batch_params.vbe_num_ranks is not None
            ), "vbe_num_ranks must be set for varaible batch size generation"
            return generate_batch_sizes_from_stats(
                self.batch_params.B,
                self.T,
                # pyre-ignore [6]
                self.batch_params.sigma_B,
                self.batch_params.vbe_num_ranks,
                # pyre-ignore [6]
                self.batch_params.vbe_distribution,
            )

        else:
            return ([self.batch_params.B] * self.T, None)

    def _generate_pooling_info(self, iters: int, Bs: List[int]) -> torch.Tensor:
        if self.variable_L():
            # Generate L from stats
            _, L_offsets = generate_pooling_factors_from_stats(
                iters,
                Bs,
                self.pooling_params.L,
                # pyre-ignore [6]
                self.pooling_params.sigma_L,
                # pyre-ignore [6]
                self.pooling_params.length_distribution,
            )

        else:
            Ls = [self.pooling_params.L] * (sum(Bs) * iters)
            L_offsets = torch.tensor([0] + Ls, dtype=torch.long).cumsum(0)

        return L_offsets

    def _generate_indices(
        self,
        iters: int,
        Bs: List[int],
        L_offsets: torch.Tensor,
    ) -> torch.Tensor:
        total_B = sum(Bs)
        L_offsets_list = L_offsets.tolist()
        indices_list = []
        for it in range(iters):
            # L_offsets is defined over the entire set of batches for a single iteration
            start_offset = L_offsets_list[it * total_B]
            end_offset = L_offsets_list[(it + 1) * total_B]

            indices_list.append(
                torch.ops.fbgemm.tbe_generate_indices_from_distribution(
                    self.indices_params.heavy_hitters,
                    self.indices_params.zipf_q,
                    self.indices_params.zipf_s,
                    # max_index = dimensions of the embedding table
                    self.E,
                    # num_indices = number of indices to generate
                    end_offset - start_offset,
                )
            )

        return torch.cat(indices_list)

    def _build_requests_jagged(
        self,
        iters: int,
        Bs: List[int],
        Bs_feature_rank: Optional[List[List[int]]],
        L_offsets: torch.Tensor,
        all_indices: torch.Tensor,
    ) -> List[TBERequest]:
        total_B = sum(Bs)
        all_indices = all_indices.flatten()
        requests = []
        for it in range(iters):
            start_offset = L_offsets[it * total_B]
            it_L_offsets = torch.concat(
                [
                    torch.zeros(1, dtype=L_offsets.dtype, device=L_offsets.device),
                    L_offsets[it * total_B + 1 : (it + 1) * total_B + 1] - start_offset,
                ]
            )
            requests.append(
                TBERequest(
                    maybe_to_dtype(
                        all_indices[start_offset : L_offsets[(it + 1) * total_B]],
                        self.indices_params.index_dtype,
                    ),
                    maybe_to_dtype(
                        it_L_offsets.to(get_device()), self.indices_params.offset_dtype
                    ),
                    self._new_weights(int(it_L_offsets[-1].item())),
                    Bs_feature_rank if self.variable_B() else None,
                )
            )
        return requests

    def _build_requests_dense(
        self, iters: int, all_indices: torch.Tensor
    ) -> List[TBERequest]:
        # NOTE: We're using existing code from requests.py to build the
        # requests, and since the existing code requires 2D view of all_indices,
        # the existing all_indices must be reshaped
        all_indices = all_indices.reshape(iters, -1)

        requests = []
        for it in range(iters):
            indices, offsets = get_table_batched_offsets_from_dense(
                all_indices[it].view(
                    self.T, self.batch_params.B, self.pooling_params.L
                ),
                use_cpu=self.use_cpu,
            )
            requests.append(
                TBERequest(
                    maybe_to_dtype(indices, self.indices_params.index_dtype),
                    maybe_to_dtype(offsets, self.indices_params.offset_dtype),
                    self._new_weights(
                        self.T * self.batch_params.B * self.pooling_params.L
                    ),
                )
            )
        return requests

    def generate_requests(
        self,
        iters: int = 1,
    ) -> List[TBERequest]:
        # Generate batch sizes
        Bs, Bs_feature_rank = self._generate_batch_sizes()

        # Generate pooling info
        L_offsets = self._generate_pooling_info(iters, Bs)

        # Generate indices
        all_indices = self._generate_indices(iters, Bs, L_offsets)

        # Build TBE requests
        if self.variable_B() or self.variable_L():
            return self._build_requests_jagged(
                iters, Bs, Bs_feature_rank, L_offsets, all_indices
            )
        else:
            return self._build_requests_dense(iters, all_indices)

    def generate_embedding_dims(self) -> Tuple[int, List[int]]:
        if self.mixed_dim:
            Ds = [
                round_up(
                    np.random.randint(low=int(0.5 * self.D), high=int(1.5 * self.D)), 4
                )
                for _ in range(self.T)
            ]
            return (int(np.average(Ds)), Ds)
        else:
            return (self.D, [self.D] * self.T)

    def generate_feature_requires_grad(self, size: int) -> torch.Tensor:
        assert size <= self.T, "size of feature_requires_grad must be less than T"
        weighted_requires_grad_tables = np.random.choice(
            self.T, replace=False, size=(size,)
        ).tolist()
        return (
            torch.tensor(
                [1 if t in weighted_requires_grad_tables else 0 for t in range(self.T)]
            )
            .to(get_device())
            .int()
        )

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from fbgemm_gpu.runtime_monitor import TBEStatsReporter, TBEStatsReporterConfig
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    MultiPassPrefetchConfig,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.utils import round_up

from hypothesis import Verbosity

from ..common import assert_torch_equal, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable, optests, running_on_github, running_on_rocm
else:
    from fbgemm_gpu.test.test_utils import (  # noqa: F401
        gpu_unavailable,  # noqa: F401
        optests,  # noqa: F401
        running_on_github,  # noqa: F401
        running_on_rocm,  # noqa: F401
    )


VERBOSITY: Verbosity = Verbosity.verbose


class TestingStatsReporter(TBEStatsReporter):
    def __init__(self, reporting_interval: int = 1) -> None:
        # Event -> args for that call
        self.reported_data: Dict[str, List[List[Union[int, str, float]]]] = {}
        self.reporting_interval = reporting_interval

    def should_report(self, iteration_step: int) -> bool:
        return (iteration_step - 1) % self.reporting_interval == 0

    def register_stats(self, stats_name: str, amplifier: int = 1) -> None:
        return

    def report_duration(
        self,
        iteration_step: int,
        event_name: str,
        duration_ms: float,
        embedding_id: str = "",
        tbe_id: str = "",
        time_unit: str = "ms",
    ) -> None:
        self.reported_data.setdefault(event_name, [])
        self.reported_data[event_name].append(
            [iteration_step, event_name, duration_ms, embedding_id, tbe_id]
        )

    def report_data_amount(
        self,
        iteration_step: int,
        event_name: str,
        data_bytes: int,
        embedding_id: str = "",
        tbe_id: str = "",
    ) -> None:
        self.reported_data.setdefault(event_name, [])
        self.reported_data[event_name].append(
            [iteration_step, event_name, data_bytes, embedding_id, tbe_id]
        )


@dataclass(frozen=True)
class TestingStatsReporterConfig(TBEStatsReporterConfig):
    def create_reporter(self) -> Optional[TBEStatsReporter]:
        return TestingStatsReporter(reporting_interval=self.interval)


def generate_cache_tbes(
    T: int,
    D: int,
    log_E: int,
    mixed: bool,
    cache_algorithm: CacheAlgorithm = CacheAlgorithm.LRU,
    prefetch_pipeline: bool = False,
    use_int_weight: bool = False,
    cache_sets: int = 0,
    weights_cache_precision: SparseType = SparseType.FP32,
    stochastic_rounding: bool = False,
    gather_uvm_cache_stats: bool = False,
    reporter_config: Optional[TestingStatsReporterConfig] = None,
    multipass_prefetch_config: Optional[MultiPassPrefetchConfig] = None,
) -> Tuple[
    SplitTableBatchedEmbeddingBagsCodegen,
    SplitTableBatchedEmbeddingBagsCodegen,
    int,
    int,
]:
    lr = 1.0 if use_int_weight else 0.02
    E = int(10**log_E)
    D = D * 4
    if not mixed:
        Ds = [D] * T
        Es = [E] * T
    else:
        Ds = [
            round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
            for _ in range(T)
        ]
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]
    managed = [EmbeddingLocation.MANAGED_CACHING] * T
    if mixed:
        average_D = sum(Ds) // T
        for t, d in enumerate(Ds):
            managed[t] = EmbeddingLocation.DEVICE if d < average_D else managed[t]
    cc_ref = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                D,
                EmbeddingLocation.DEVICE,
                ComputeDevice.CUDA,
            )
            for (E, D) in zip(Es, Ds)
        ],
        stochastic_rounding=stochastic_rounding,
        prefetch_pipeline=False,
        learning_rate=lr,
        weights_precision=weights_cache_precision,
    )
    # Init the embedding weights
    # Limiting the weights to be within a small range as larger values can
    # cause higher than 1.0e-2 absolute difference (although the relative
    # difference stays below 1.0e-2) when stochastic_rounding=True. We choose
    # to init the weights to small values instead of scaling the absolute
    # difference.
    cc_ref.init_embedding_weights_uniform(-2.0, 2.0)

    cc = SplitTableBatchedEmbeddingBagsCodegen(
        [(E, D, M, ComputeDevice.CUDA) for (E, D, M) in zip(Es, Ds, managed)],
        cache_algorithm=cache_algorithm,
        stochastic_rounding=stochastic_rounding,
        prefetch_pipeline=prefetch_pipeline,
        learning_rate=lr,
        cache_sets=cache_sets,
        weights_precision=weights_cache_precision,
        cache_precision=weights_cache_precision,
        gather_uvm_cache_stats=gather_uvm_cache_stats,
        stats_reporter_config=reporter_config,
        multipass_prefetch_config=multipass_prefetch_config,
    )

    if use_int_weight:
        min_val = -20
        max_val = +20
        for param in cc_ref.split_embedding_weights():
            p = torch.randint(
                int(min_val),
                int(max_val) + 1,
                size=param.shape,
                device=param.device,
            )
            param.data.copy_(p)

    for t in range(T):
        assert (
            cc.split_embedding_weights()[t].size()
            == cc_ref.split_embedding_weights()[t].size()
        )
        cc.split_embedding_weights()[t].data.copy_(cc_ref.split_embedding_weights()[t])

    return (cc, cc_ref, min(Es), sum(Ds))


def assert_cache(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor, stochastic_rounding: bool
) -> None:
    if stochastic_rounding:
        # Stochastic rounding randomly alters the mantissa bits during the
        # FP32->FP16 conversion in TBE backward, resulting in non-deterministic
        # results. The relative difference between the results from two
        # different runs must be <= 1.0e-2. We set absolute tolerance to 1.0e-2
        # based on the initial embedding weights.
        torch.testing.assert_close(
            tensor_a.float(), tensor_b.float(), atol=1.0e-2, rtol=1.0e-2
        )
    else:
        assert_torch_equal(tensor_a, tensor_b)

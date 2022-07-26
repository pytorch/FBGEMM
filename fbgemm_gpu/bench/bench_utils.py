# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import statistics
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import SparseType

# pyre-fixme[21]: Could not find name `default_rng` in `numpy.random` (stubbed).
from numpy.random import default_rng
from torch import Tensor

logging.basicConfig(level=logging.DEBUG)


def benchmark_torch_function(
    # pyre-fixme[2]: Parameter must be annotated.
    f,
    # pyre-fixme[2]: Parameter must be annotated.
    args,
    flush_gpu_cache_size_mb: int = 40,
    iters: int = 10,
    num_warmups: int = 2,
) -> Tuple[float, Tensor]:
    for _ in range(num_warmups):
        output = f(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # flush the cache
        if flush_gpu_cache_size_mb:
            _ = torch.rand(
                flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
            )
            torch.cuda.synchronize()
        start_event.record()
        for _ in range(iters):
            output = f(*args)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3
    else:
        start_time = time.time()
        for _ in range(iters):
            output = f(*args)
        elapsed_time = time.time() - start_time

    # pyre-fixme[61]: `output` is undefined, or not always defined.
    return float(elapsed_time) / iters, output


def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def get_device() -> torch.device:
    # pyre-fixme[7]: Expected `device` but got `Union[int, device]`.
    return (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


# Merged indices with shape (T, B, L) -> (flattened indices with shape
# (T * B * L), offsets with shape (T * B + 1))
def get_table_batched_offsets_from_dense(
    merged_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.long().contiguous().view(-1).to(get_device()),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long().to(get_device()),
    )


def get_offsets_from_dense(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    (B, L) = indices.size()
    return (
        indices.contiguous().view(-1),
        torch.tensor(
            np.cumsum(np.asarray([0] + [L for _ in range(B)])[:-1]).astype(np.int64)
        ),
    )


def b_indices(
    b: Callable[..., torch.Tensor],
    x: torch.Tensor,
    per_sample_weights: Optional[torch.Tensor] = None,
    use_cpu: bool = False,
    do_pooling: bool = True,
) -> torch.Tensor:
    (indices, offsets) = get_offsets_from_dense(x)
    if do_pooling:
        return b(
            indices.cuda(),
            offsets.cuda(),
            per_sample_weights=per_sample_weights,
        )
    else:
        return b(indices.cuda())


def generate_requests(
    iters: int,
    B: int,
    T: int,
    L: int,
    E: int,
    # inter-batch indices reuse rate
    reuse: float = 0.0,
    # alpha <= 1.0: use uniform distribution
    # alpha > 1.0: use zipf distribution
    alpha: float = 1.0,
    weights_precision: SparseType = SparseType.FP32,
    weighted: bool = False,
    requests_data_file: Optional[str] = None,
    # Comma-separated list of table numbers
    tables: Optional[str] = None,
) -> List[Tuple[torch.IntTensor, torch.IntTensor, Optional[Tensor]]]:
    if requests_data_file is not None:
        indices_tensor, offsets_tensor, lengths_tensor = torch.load(requests_data_file)

        average_L = 0
        if tables is not None:
            emb_tables = tuple(int(x) for x in tables.split(","))
            indices = torch.zeros(0, dtype=indices_tensor.dtype)
            offsets = torch.zeros(1, dtype=offsets_tensor.dtype)
            total_L = 0
            for t in emb_tables:
                t_offsets = offsets_tensor[B * t : B * (t + 1) + 1]
                total_L += t_offsets[-1] - t_offsets[0]
                indices = torch.cat(
                    (indices, indices_tensor[t_offsets[0] : t_offsets[-1]])
                )
                offsets = torch.cat(
                    (
                        offsets,
                        t_offsets[1:] - t_offsets[0] + offsets[-1],
                    )
                )
            indices_tensor = indices
            offsets_tensor = offsets
            average_L = int(total_L / B)

            assert np.prod(offsets_tensor.size()) - 1 == np.prod((T, B)), (
                f"Requested tables: {emb_tables} "
                f"does not conform to inputs (T, B) = ({T}, {B})."
            )
            logging.warning(
                f"Using (indices = {indices_tensor.size()}, offsets = {offsets_tensor.size()}) based "
                f"on tables: {emb_tables}"
            )
        else:
            average_L = int((offsets_tensor[-1] - offsets_tensor[0]) / B)
            assert (np.prod(offsets_tensor.size()) - 1) == np.prod((T, B)), (
                f"Data file (indices = {indices_tensor.size()}, "
                f"offsets = {offsets_tensor.size()}, lengths = {lengths_tensor.size()}) "
                f"does not conform to inputs (T, B) = ({T}, {B})."
            )

        assert (
            L == average_L
        ), f"Requested L does not align with provided data file ({L} vs. {average_L})"
        assert E > max(indices_tensor), (
            f"Number of embeddings is not enough to support maximum index "
            f"provided by data file {E} vs. {max(indices_tensor)}"
        )

        weights_tensor = (
            None
            if not weighted
            else torch.randn(indices_tensor.size(), device=get_device())
        )
        rs = []
        for _ in range(iters):
            rs.append(
                (
                    indices_tensor.to(get_device()),
                    offsets_tensor.to(get_device()),
                    weights_tensor,
                )
            )
        return rs

    if alpha <= 1.0:
        all_indices = torch.randint(
            low=0,
            high=E,
            size=(iters, T, B, L),
            device=get_device(),
            dtype=torch.int32,
        )
        # each bag is usually sorted
        (all_indices, _) = torch.sort(all_indices)
        all_indices = all_indices.reshape(iters, T, B * L)
    else:
        assert E >= L, "num-embeddings must be greater than equal to bag-size"
        # oversample and then remove duplicates to obtain sampling without
        # replacement
        all_indices = (np.random.zipf(a=alpha, size=(iters, T, B, 3 * L)) - 1) % E
        all_indices = torch.ops.fbgemm.bottom_unique_k_per_row(
            torch.as_tensor(all_indices), L
        )
        rng = default_rng()
        permutation = torch.as_tensor(
            rng.choice(E, size=all_indices.max().item() + 1, replace=False)
        )
        all_indices = permutation.gather(0, all_indices.flatten())
        all_indices = all_indices.to(get_device()).int().reshape(iters, T, B * L)
    for it in range(iters - 1):
        for t in range(T):
            reused_indices = torch.randperm(B * L, device=get_device())[
                : int(B * L * reuse)
            ]
            all_indices[it + 1, t, reused_indices] = all_indices[it, t, reused_indices]

    rs = []
    for it in range(iters):
        weights_tensor = (
            None if not weighted else torch.randn(T * B * L, device=get_device())
        )
        rs.append(
            get_table_batched_offsets_from_dense(all_indices[it].view(T, B, L))
            + (weights_tensor,)
        )
    return rs


def benchmark_requests(
    requests: List[Tuple[torch.IntTensor, torch.IntTensor, Optional[Tensor]]],
    func: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
    flush_gpu_cache_size_mb: int = 0,
    check_median: bool = False,
    num_warmups: int = 0,
) -> float:
    times = []

    if num_warmups > 0:
        indices, offsets, weights = requests[0]
        for _ in range(num_warmups):
            func(indices, offsets, weights)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    for (indices, offsets, weights) in requests:
        start_time = time.time()
        if torch.cuda.is_available():
            if flush_gpu_cache_size_mb:
                _ = torch.rand(
                    flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
                )
                torch.cuda.synchronize()
            start_event.record()
        func(indices, offsets, weights)
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            it_time = start_event.elapsed_time(end_event) * 1.0e-3
            times.append(it_time)
        else:
            it_time = time.time() - start_time
            times.append(it_time)
    avg_time = sum(times) / len(requests)
    median_time = statistics.median(times)
    return median_time if check_median else avg_time


def benchmark_requests_refer(
    requests: List[Tuple[torch.IntTensor, torch.IntTensor, Optional[Tensor]]],
    T: int,
    B: int,
    L: int,
    E: int,
    D: int,
    pooling_mode: str,
    weighted: bool,
    flush_gpu_cache_size_mb: int = 0,
    check_median: bool = False,
) -> float:
    do_pooling = pooling_mode in ["sum", "mean"]
    if do_pooling:
        nn_embedding_list = [
            torch.nn.EmbeddingBag(E, D, mode=pooling_mode, sparse=True).cuda()
        ] * T
    else:
        nn_embedding_list = [torch.nn.Embedding(E, D, sparse=True).cuda()] * T

    times = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    for (indices, _, weights) in requests:
        indices_list = indices.view(T, B, L).split(1)

        if weighted:
            assert weights is not None
            weights_list = weights.view(T, B, L).split(1)

        start_time = time.time()
        if torch.cuda.is_available():
            if flush_gpu_cache_size_mb:
                _ = torch.rand(
                    flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
                )
                torch.cuda.synchronize()
            start_event.record()

        nn_embedding_output = (
            [
                b_indices(nn_embedding, x, use_cpu=False, do_pooling=do_pooling)
                for (nn_embedding, x) in zip(nn_embedding_list, indices_list)
            ]
            if not weighted
            else [
                b_indices(
                    nn_embedding,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=False,
                    do_pooling=do_pooling,
                )
                for (nn_embedding, x, xw) in zip(
                    nn_embedding_list,
                    indices_list,
                    # pyre-fixme[61]: `weights_list` is undefined, or not always
                    #  defined.
                    weights_list,
                )
            ]
        )
        if do_pooling:
            final_output = torch.cat(
                [f.view(B, -1) for f in nn_embedding_output], dim=1
            )
        else:
            final_output = torch.cat(nn_embedding_output, dim=0).view(-1, D)

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            it_time = start_event.elapsed_time(end_event) * 1.0e-3
            times.append(it_time)
        else:
            it_time = time.time() - start_time
            times.append(it_time)
    avg_time = sum(times) / len(requests)
    median_time = statistics.median(times)
    return median_time if check_median else avg_time


def benchmark_pipelined_requests(
    requests: List[Tuple[torch.IntTensor, torch.IntTensor, Optional[Tensor]]],
    func1: Callable[[Tensor, Tensor, Optional[Tensor]], None],
    func2: Callable[[Tensor, Tensor, Optional[Tensor]], None],
    flush_gpu_cache_size_mb: int = 0,
    check_median: bool = False,
) -> Tuple[float, float]:
    torch.cuda.synchronize()
    start_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in requests
    ]
    end_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in requests
    ]
    for ((indices, offsets, indices_weights), start_event, end_event) in zip(
        requests, start_events, end_events
    ):
        if flush_gpu_cache_size_mb:
            _ = torch.rand(
                flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
            )
            torch.cuda.synchronize()
        start_event[0].record()
        func1(indices, offsets, indices_weights)
        end_event[0].record()
        start_event[1].record()
        func2(indices, offsets, indices_weights)
        end_event[1].record()
    torch.cuda.synchronize()
    avg_time = (
        sum(
            start_event[0].elapsed_time(end_event[0]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        )
        / len(requests),
        sum(
            start_event[1].elapsed_time(end_event[1]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        )
        / len(requests),
    )
    median_time = (
        statistics.median(
            start_event[0].elapsed_time(end_event[0]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        ),
        statistics.median(
            start_event[1].elapsed_time(end_event[1]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        ),
    )
    return median_time if check_median else avg_time

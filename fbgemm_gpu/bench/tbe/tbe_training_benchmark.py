#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging
import os
import tempfile
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional

import click
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    CacheAlgorithm,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.bench import (
    benchmark_pipelined_requests,
    benchmark_requests,
    EmbeddingOpsCommonConfigLoader,
    TBEBenchmarkingConfigLoader,
    TBEDataConfigLoader,
)
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.utils import get_device
from torch.profiler import profile

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--emb-op-type",
    default="split",
    type=click.Choice(["split", "dense", "ssd"], case_sensitive=False),
    help="The type of the embedding op to benchmark",
)
@click.option(
    "--row-wise/--no-row-wise",
    default=True,
    help="Whether to use row-wise adagrad optimzier or not",
)
@click.option(
    "--weighted-num-requires-grad",
    type=int,
    default=None,
    help="The number of weighted tables that require gradient",
)
@click.option(
    "--ssd-prefix",
    type=str,
    default="/tmp/ssd_benchmark",
    help="SSD directory prefix",
)
@click.option("--cache-load-factor", default=0.2)
@TBEBenchmarkingConfigLoader.options
@TBEDataConfigLoader.options
@EmbeddingOpsCommonConfigLoader.options(True)
@click.pass_context
def device(  # noqa C901
    context: click.Context,
    emb_op_type: click.Choice,
    row_wise: bool,
    weighted_num_requires_grad: Optional[int],
    cache_load_factor: float,
    # SSD params
    ssd_prefix: str,
    # pyre-ignore[2]
    **kwargs,
) -> None:
    # Initialize random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Load general TBE benchmarking configuration from cli arguments
    benchconfig = TBEBenchmarkingConfigLoader.load(context)

    # Load TBE data configuration from cli arguments
    tbeconfig = TBEDataConfigLoader.load(context)

    # Load common embedding op configuration from cli arguments
    embconfig = EmbeddingOpsCommonConfigLoader.load(context)

    # Generate feature_requires_grad
    feature_requires_grad = (
        tbeconfig.generate_feature_requires_grad(weighted_num_requires_grad)
        if weighted_num_requires_grad
        else None
    )

    # Generate embedding dims
    effective_D, Ds = tbeconfig.generate_embedding_dims()

    # Determine the optimizer
    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD if row_wise else OptimType.EXACT_ADAGRAD

    # Construct the common split arguments for the embedding op
    common_split_args: Dict[str, Any] = embconfig.split_args() | {
        "optimizer": optimizer,
        "learning_rate": 0.1,
        "eps": 0.1,
        "feature_table_map": list(range(tbeconfig.T)),
    }

    if emb_op_type == "dense":
        embedding_op = DenseTableBatchedEmbeddingBagsCodegen(
            [
                (
                    tbeconfig.E,
                    d,
                )
                for d in Ds
            ],
            pooling_mode=embconfig.pooling_mode,
            use_cpu=not torch.cuda.is_available(),
        )
    elif emb_op_type == "ssd":
        assert (
            torch.cuda.is_available()
        ), "SSDTableBatchedEmbeddingBags only supports GPU execution"
        cache_set = max(tbeconfig.T * tbeconfig.batch_params.B, 1)
        tempdir = tempfile.mkdtemp(prefix=ssd_prefix)
        embedding_op = SSDTableBatchedEmbeddingBags(
            embedding_specs=[(tbeconfig.E, d) for d in Ds],
            cache_sets=cache_set,
            ssd_storage_directory=tempdir,
            ssd_cache_location=EmbeddingLocation.DEVICE,
            ssd_rocksdb_shards=8,
            **common_split_args,
        )
    else:
        embedding_op = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    tbeconfig.E,
                    d,
                    embconfig.embedding_location,
                    (
                        ComputeDevice.CUDA
                        if torch.cuda.is_available()
                        else ComputeDevice.CPU
                    ),
                )
                for d in Ds
            ],
            cache_precision=(
                embconfig.weights_dtype
                if embconfig.cache_dtype is None
                else embconfig.cache_dtype
            ),
            cache_algorithm=CacheAlgorithm.LRU,
            cache_load_factor=cache_load_factor,
            **common_split_args,
        )
    embedding_op = embedding_op.to(get_device())

    if embconfig.weights_dtype == SparseType.INT8:
        # pyre-fixme[29]: `Union[(self: DenseTableBatchedEmbeddingBagsCodegen,
        #  min_val: float, max_val: float) -> None, (self:
        #  SplitTableBatchedEmbeddingBagsCodegen, min_val: float, max_val: float) ->
        #  None, Tensor, Module]` is not a function.
        embedding_op.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(d * tbeconfig.E for d in Ds)
    param_size_multiplier = embconfig.weights_dtype.bit_rate() / 8.0
    output_size_multiplier = embconfig.output_dtype.bit_rate() / 8.0
    if embconfig.pooling_mode.do_pooling():
        read_write_bytes = (
            output_size_multiplier * tbeconfig.batch_params.B * sum(Ds)
            + param_size_multiplier
            * tbeconfig.batch_params.B
            * sum(Ds)
            * tbeconfig.pooling_params.L
        )
    else:
        read_write_bytes = (
            output_size_multiplier
            * tbeconfig.batch_params.B
            * sum(Ds)
            * tbeconfig.pooling_params.L
            + param_size_multiplier
            * tbeconfig.batch_params.B
            * sum(Ds)
            * tbeconfig.pooling_params.L
        )

    logging.info(f"Managed option: {embconfig.embedding_location}")
    logging.info(
        f"Embedding parameters: {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f} GB"
    )
    logging.info(
        f"Accessed weights per batch: {tbeconfig.batch_params.B * sum(Ds) * tbeconfig.pooling_params.L * param_size_multiplier / 1.0e9: .2f} GB"
    )

    requests = tbeconfig.generate_requests(benchconfig.num_requests)

    # pyre-ignore[53]
    def _kineto_trace_handler(p: profile, phase: str) -> None:
        p.export_chrome_trace(
            benchconfig.trace_url.format(
                emb_op_type=emb_op_type, phase=phase, ospid=os.getpid()
            )
        )

    # pyre-ignore[3,53]
    def _context_factory(on_trace_ready: Callable[[profile], None]):
        return (
            profile(on_trace_ready=on_trace_ready)
            if benchconfig.export_trace
            else nullcontext()
        )

    with _context_factory(lambda p: _kineto_trace_handler(p, "fwd")):
        # forward
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: embedding_op.forward(
                indices.to(dtype=tbeconfig.indices_params.index_dtype),
                offsets.to(dtype=tbeconfig.indices_params.offset_dtype),
                per_sample_weights,
                feature_requires_grad=feature_requires_grad,
            ),
            flush_gpu_cache_size_mb=benchconfig.flush_gpu_cache_size_mb,
            num_warmups=benchconfig.warmup_iterations,
            iters=benchconfig.iterations,
        )

    logging.info(
        f"Forward, B: {tbeconfig.batch_params.B}, "
        f"E: {tbeconfig.E}, T: {tbeconfig.T}, D: {effective_D}, L: {tbeconfig.pooling_params.L}, W: {tbeconfig.weighted}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    if embconfig.output_dtype == SparseType.INT8:
        # backward bench not representative
        return

    if embconfig.pooling_mode.do_pooling():
        grad_output = torch.randn(tbeconfig.batch_params.B, sum(Ds)).to(get_device())
    else:
        grad_output = torch.randn(
            tbeconfig.batch_params.B * tbeconfig.T * tbeconfig.pooling_params.L,
            effective_D,
        ).to(get_device())

    with _context_factory(lambda p: _kineto_trace_handler(p, "fwd_bwd")):
        # backward
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: embedding_op(
                indices.to(dtype=tbeconfig.indices_params.index_dtype),
                offsets.to(dtype=tbeconfig.indices_params.offset_dtype),
                per_sample_weights,
                feature_requires_grad=feature_requires_grad,
            ),
            flush_gpu_cache_size_mb=benchconfig.flush_gpu_cache_size_mb,
            bwd_only=True,
            grad=grad_output,
            num_warmups=benchconfig.warmup_iterations,
            iters=benchconfig.iterations,
        )

    logging.info(
        f"Backward, B: {tbeconfig.batch_params.B}, E: {tbeconfig.E}, T: {tbeconfig.T}, D: {effective_D}, L: {tbeconfig.pooling_params.L}, "
        f"BW: {2 * read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@click.option("--cache-algorithm", default="lru")
@click.option("--cache-load-factor", default=0.2)
@TBEBenchmarkingConfigLoader.options
@TBEDataConfigLoader.options
@EmbeddingOpsCommonConfigLoader.options(False)
@click.pass_context
def cache(  # noqa C901
    context: click.Context,
    cache_algorithm: str,
    cache_load_factor: float,
    # pyre-ignore[2]
    **kwargs,
) -> None:
    # Initialize random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Load general TBE benchmarking configuration from cli arguments
    benchconfig = TBEBenchmarkingConfigLoader.load(context)

    # Load TBE data configuration from cli arguments
    tbeconfig = TBEDataConfigLoader.load(context)

    # Load common embedding op configuration from cli arguments
    embconfig = EmbeddingOpsCommonConfigLoader.load(context)

    E = tbeconfig.E
    T = tbeconfig.T
    D = tbeconfig.D
    L = tbeconfig.pooling_params.L
    B = tbeconfig.batch_params.B

    # Generate embedding dims
    effective_D, Ds = tbeconfig.generate_embedding_dims()

    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD

    cache_alg = CacheAlgorithm.LRU if cache_algorithm == "lru" else CacheAlgorithm.LFU

    embedding_op_nocache = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                tbeconfig.E,
                d,
                EmbeddingLocation.MANAGED,
                ComputeDevice.CUDA,
            )
            for d in Ds
        ],
        optimizer=optimizer,
        **(embconfig.split_args()),
    ).cuda()

    if embconfig.weights_dtype == SparseType.INT8:
        embedding_op_nocache.init_embedding_weights_uniform(-0.0003, 0.0003)

    embedding_op = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                tbeconfig.E,
                d,
                EmbeddingLocation.MANAGED_CACHING,
                ComputeDevice.CUDA,
            )
            for d in Ds
        ],
        optimizer=optimizer,
        cache_load_factor=cache_load_factor,
        cache_algorithm=cache_alg,
        **(embconfig.split_args()),
    ).cuda()

    if embconfig.weights_dtype == SparseType.INT8:
        embedding_op.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(w.numel() for w in embedding_op.split_embedding_weights())
    param_size_multiplier = embconfig.weights_dtype.bit_rate() / 8.0
    logging.info(
        f"Embedding tables: {E * T} rows, {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f} GB"
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
        f"{B * T * L * D * param_size_multiplier / 1.0e9: .2f} GB"
    )

    requests = tbeconfig.generate_requests(2 * benchconfig.iterations)

    warmup_requests, requests = (
        requests[: benchconfig.iterations],
        requests[benchconfig.iterations :],
    )
    grad_output = torch.randn(tbeconfig.batch_params.B, sum(Ds)).cuda()

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: embedding_op_nocache(
            indices, offsets, per_sample_weights
        ).backward(grad_output),
        flush_gpu_cache_size_mb=benchconfig.flush_gpu_cache_size_mb,
        num_warmups=benchconfig.warmup_iterations,
    )
    logging.info(
        f"ForwardBackward (UVM), B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    # warm up
    for req in warmup_requests:
        indices, offsets = req.unpack_2()
        embedding_op.forward(indices, offsets)
    # get cache miss rate (forward and backward) and exchanged cache lines (prefetch)
    cache_misses = []
    exchanged_cache_lines = []
    NOT_FOUND = -1
    for req in requests:
        indices, offsets = req.unpack_2()
        # pyre-fixme[29]: `Union[(self: TensorBase, memory_format:
        #  Optional[memory_format] = ...) -> Tensor, Tensor, Module]` is not a
        #  function.
        old_lxu_cache_state = embedding_op.lxu_cache_state.clone()
        embedding_op.prefetch(indices, offsets)
        exchanged_cache_lines.append(
            # pyre-fixme[16]: Item `bool` of `bool | Tensor` has no attribute `sum`.
            (embedding_op.lxu_cache_state != old_lxu_cache_state)
            .sum()
            .item()
        )
        cache_misses.append(
            (embedding_op.lxu_cache_locations_list[0] == NOT_FOUND).sum().item()
        )
        embedding_op.forward(indices, offsets)
    logging.info(
        f"Exchanged cache lines -- mean: {sum(exchanged_cache_lines) / len(requests): .2f}, "
        f"max: {max(exchanged_cache_lines)}, min: {min(exchanged_cache_lines)}"
    )
    logging.info(
        f"Cache miss -- mean: {sum(cache_misses) / len(requests)}, "
        f"max: {max(cache_misses)}, min: {min(cache_misses)}"
    )

    # benchmark prefetch
    embedding_op.reset_cache_states()
    for req in warmup_requests:
        indices, offsets = req.unpack_2()
        embedding_op.forward(indices, offsets)
    # TODO: Add warmup_runs
    prefetch_time, forward_backward_time = benchmark_pipelined_requests(
        requests,
        lambda indices, offsets, indices_weights: embedding_op.prefetch(
            indices, offsets
        ),
        lambda indices, offsets, indices_weights: embedding_op.forward(
            indices, offsets, indices_weights
        ).backward(grad_output),
        flush_gpu_cache_size_mb=benchconfig.flush_gpu_cache_size_mb,
    )
    e2e_time = prefetch_time + forward_backward_time

    logging.info(
        f"ForwardBackward (LXU) B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / e2e_time / 1.0e9: .2f} GB/s, "
        f"Tprefetch: {prefetch_time * 1.0e6:.0f}us, "
        f"{2 * sum(exchanged_cache_lines) * param_size_multiplier * D / prefetch_time / len(requests) / 1.0e9: .2f} GB/s, "
        f"Tfwdbwd: {forward_backward_time * 1.0e6:.0f}us, "
        f"{3 * param_size_multiplier * B * sum(Ds) * L / forward_backward_time / 1.0e9: .2f} GB/s, "
        f"Te2e: {e2e_time * 1.0e6:.0f}us, "
    )


if __name__ == "__main__":
    cli()

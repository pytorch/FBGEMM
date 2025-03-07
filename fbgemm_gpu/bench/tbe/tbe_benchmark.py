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
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    str_to_embedding_location,
    str_to_pooling_mode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.bench import (
    benchmark_requests,
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
@click.option("--weights-precision", type=SparseType, default=SparseType.FP32)
@click.option("--cache-precision", type=SparseType, default=None)
@click.option("--stoc", is_flag=True, default=False)
@click.option(
    "--managed",
    default="device",
    type=click.Choice(["device", "managed", "managed_caching"], case_sensitive=False),
)
@click.option(
    "--emb-op-type",
    default="split",
    type=click.Choice(["split", "dense", "ssd"], case_sensitive=False),
)
@click.option("--row-wise/--no-row-wise", default=True)
@click.option("--pooling", type=str, default="sum")
@click.option("--weighted-num-requires-grad", type=int, default=None)
@click.option("--bounds-check-mode", type=int, default=BoundsCheckMode.NONE.value)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP32)
@click.option(
    "--uvm-host-mapped",
    is_flag=True,
    default=False,
    help="Use host mapped UVM buffers in SSD-TBE (malloc+cudaHostRegister)",
)
@click.option(
    "--ssd-prefix", type=str, default="/tmp/ssd_benchmark", help="SSD directory prefix"
)
@click.option("--cache-load-factor", default=0.2)
@TBEBenchmarkingConfigLoader.options
@TBEDataConfigLoader.options
@click.pass_context
def device(  # noqa C901
    context: click.Context,
    emb_op_type: click.Choice,
    weights_precision: SparseType,
    cache_precision: Optional[SparseType],
    stoc: bool,
    managed: click.Choice,
    row_wise: bool,
    pooling: str,
    weighted_num_requires_grad: Optional[int],
    bounds_check_mode: int,
    output_dtype: SparseType,
    uvm_host_mapped: bool,
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

    # Determine the embedding location
    embedding_location = str_to_embedding_location(str(managed))
    if embedding_location is EmbeddingLocation.DEVICE and not torch.cuda.is_available():
        embedding_location = EmbeddingLocation.HOST

    # Determine the pooling mode
    pooling_mode = str_to_pooling_mode(pooling)

    # Construct the common split arguments for the embedding op
    common_split_args: Dict[str, Any] = {
        "weights_precision": weights_precision,
        "stochastic_rounding": stoc,
        "output_dtype": output_dtype,
        "pooling_mode": pooling_mode,
        "bounds_check_mode": BoundsCheckMode(bounds_check_mode),
        "uvm_host_mapped": uvm_host_mapped,
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
            pooling_mode=pooling_mode,
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
                    embedding_location,
                    (
                        ComputeDevice.CUDA
                        if torch.cuda.is_available()
                        else ComputeDevice.CPU
                    ),
                )
                for d in Ds
            ],
            cache_precision=(
                weights_precision if cache_precision is None else cache_precision
            ),
            cache_algorithm=CacheAlgorithm.LRU,
            cache_load_factor=cache_load_factor,
            **common_split_args,
        )
    embedding_op = embedding_op.to(get_device())

    if weights_precision == SparseType.INT8:
        # pyre-fixme[29]: `Union[(self: DenseTableBatchedEmbeddingBagsCodegen,
        #  min_val: float, max_val: float) -> None, (self:
        #  SplitTableBatchedEmbeddingBagsCodegen, min_val: float, max_val: float) ->
        #  None, Tensor, Module]` is not a function.
        embedding_op.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(d * tbeconfig.E for d in Ds)
    param_size_multiplier = weights_precision.bit_rate() / 8.0
    output_size_multiplier = output_dtype.bit_rate() / 8.0
    if pooling_mode.do_pooling():
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

    logging.info(f"Managed option: {managed}")
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

    if output_dtype == SparseType.INT8:
        # backward bench not representative
        return

    if pooling_mode.do_pooling():
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


if __name__ == "__main__":
    cli()

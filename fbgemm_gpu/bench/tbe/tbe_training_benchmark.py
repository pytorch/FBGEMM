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

try:
    from fbgemm_gpu.tbe.trace.fbgemm_kineto_trace_handler import (
        FbgemmKinetoTraceHandler,
    )
except Exception:
    pass

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
    get_available_compute_device,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.bench import (
    benchmark_requests,
    benchmark_requests_with_spec,
    EmbeddingOpsCommonConfigLoader,
    TBEBenchmarkingConfigLoader,
    TBEDataConfigLoader,
)
from fbgemm_gpu.tbe.bench.tbe_data_config_bench_helper import (
    generate_embedding_dims,
    generate_feature_requires_grad,
    generate_requests,
    generate_requests_with_Llist,
)
from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.utils import get_device
from torch.profiler import profile

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

try:
    import mtia.host_runtime.torch_mtia.dynamic_library  # pyright: ignore  # noqa: F401  # pyre-ignore[21]

    torch.mtia.init()
except Exception:
    pass


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
@EmbeddingOpsCommonConfigLoader.options
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
        generate_feature_requires_grad(tbeconfig, weighted_num_requires_grad)
        if weighted_num_requires_grad
        else None
    )

    # Generate embedding dims
    effective_D, Ds = generate_embedding_dims(tbeconfig)

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

    requests = generate_requests(tbeconfig, benchconfig.num_requests)

    # pyre-ignore[53]
    def _kineto_trace_handler(p: profile, phase: str) -> None:
        if benchconfig.trace_url is not None:
            trace_path = benchconfig.trace_url.format(
                emb_op_type=emb_op_type, phase=phase, ospid=os.getpid()
            )
            p.export_chrome_trace(trace_path)
        else:
            logger.warning("Cannot export trace: trace_url is None")

    # pyre-ignore[53]
    def _context_factory(
        on_trace_ready: Callable[[profile], None],
    ) -> tuple[Any, Optional[profile]]:
        """
        Creates a context manager for profiling based on configuration.

        Args:
            on_trace_ready: Callback function to be called when profiling is complete

        Returns:
            A tuple containing:
            - A context manager (either profile or nullcontext)
            - The profile object if profiling is enabled, otherwise None
        """
        if benchconfig.export_trace:
            prof = profile(on_trace_ready=on_trace_ready)
            return prof, prof
        else:
            return nullcontext(), None

    with _context_factory(lambda p: _kineto_trace_handler(p, "fwd"))[0] as p_obj:
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

    if benchconfig.upload_perf_data:
        if (
            benchconfig.export_trace
            and p_obj is not None
            and benchconfig.trace_url is not None
        ):
            try:
                trace_url = benchconfig.trace_url.format(
                    emb_op_type=emb_op_type, phase="fwd", ospid=os.getpid()
                )
                FbgemmKinetoTraceHandler(p_obj).sync_log(
                    run_id=str(trace_url),
                    test_phase="fwd",
                    test_name=str("tbe_training"),
                    benchmark_duration_us=float(time_per_iter * 1.0e6),
                    achieved_bw_gbps=float(read_write_bytes / time_per_iter / 1.0e9),
                )
            except Exception as e:
                logging.error(f"Failed to upload performance data to Scuba: {e}")

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

    with _context_factory(lambda p: _kineto_trace_handler(p, "fwd_bwd"))[0] as p_obj:
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

    # Upload performance data for backward pass if enabled
    if benchconfig.upload_perf_data:
        if (
            benchconfig.export_trace
            and p_obj is not None
            and benchconfig.trace_url is not None
        ):
            try:
                trace_url = benchconfig.trace_url.format(
                    emb_op_type=emb_op_type, phase="fwd_bwd", ospid=os.getpid()
                )
                FbgemmKinetoTraceHandler(p_obj).sync_log(
                    run_id=str(trace_url),
                    test_phase="fwd_bwd",
                    test_name=str("tbe_training"),
                    benchmark_duration_us=float(time_per_iter * 1.0e6),
                    achieved_bw_gbps=float(
                        2 * read_write_bytes / time_per_iter / 1.0e9
                    ),
                )
            except Exception as e:
                logging.error(f"Failed to upload performance data to Scuba: {e}")

    logging.info(
        f"Backward, B: {tbeconfig.batch_params.B}, E: {tbeconfig.E}, T: {tbeconfig.T}, D: {effective_D}, L: {tbeconfig.pooling_params.L}, "
        f"BW: {2 * read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


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
@click.option(
    "--pooling-list",
    type=str,
    default=None,
    help="override pooling list",
)
@click.option("--cache-load-factor", default=0.2)
@TBEBenchmarkingConfigLoader.options
@TBEDataConfigLoader.options
@EmbeddingOpsCommonConfigLoader.options
@click.pass_context
def device_with_speclist(  # noqa C901
    context: click.Context,
    emb_op_type: click.Choice,
    row_wise: bool,
    weighted_num_requires_grad: Optional[int],
    cache_load_factor: float,
    # SSD params
    ssd_prefix: str,
    pooling_list: Optional[str],
    # pyre-ignore[2]
    **kwargs,
) -> None:
    """
    A TBE benchmark supporting TBE param list and EEG params as input arguments. This allows for more flexible and customizable benchmarking.
    Args:
        uses optional arguments from TBEDataConfigLoader to take in TBE param list and EEG params as input arguments:
        --tbe-num-embeddings-list: the list of embedding table sizes
        --tbe-embedding-dim-list: the list of embedding dimensions
        --tbe-batch-sizes-list: the list of batch sizes
        --pooling-list: the list of pooling factors
    Example:
        buck2 run @mode/opt -c fbcode.nvcc_arch=h100 -c fbcode.platform=platform010 //deeplearning/fbgemm/fbgemm_gpu/bench:tbe_training -- device-with-speclist --bench-warmup-iterations 2 \
            --bench-iterations 10 --emb-pooling-mode sum --row-wise --tbe-num-tables 5 --tbe-num-embeddings-list 169694,66932,3717056,335,101083 --tbe-embedding-dim-list 128,128,128,128,128 \
            --tbe-batch-sizes-list 245760,245760,245760,245760,245760 \
            --pooling-list 4.454203287760417,8.075313313802083,1.5521280924479166,9.099202473958334,37.089603678385416\
            --tbe-indices-zipf 2.75 0.8900000000000006 --tbe-indices-hitters 0.0032447561639423338,0.002346034168270899,0.002270828999570933,0.0021225501825015603,0.0021215337630846732,0.0019088356649139518,0.001890480906511913,0.0018895829048911682,0.001865188838885878,0.001863886243128314,0.0018611428975176868,0.0018586561237987009,0.001858429156356095,0.0018583502111586669,0.0018583206067096312,0.001705492572638463,0.0017048511429093595,0.00170478206586161,0.0017045847028680395,0.0017042393176292915 \
            --tbe-indices-dtype 64 --tbe-offsets-dtype 64 --tbe-pooling-size 21 --tbe-pooling-vl-sigma 35 --tbe-pooling-vl-dist normal --emb-cache-dtype fp16 --emb-weights-dtype fp16 --bench-export-trace --emb-stochastic-rounding \
    """

    # Initialize random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Load general TBE benchmarking configuration from cli arguments
    benchconfig = TBEBenchmarkingConfigLoader.load(context)

    # Load TBE data configuration from cli arguments
    tbeconfig = TBEDataConfigLoader.load(context)

    # Load common embedding op configuration from cli arguments
    embconfig = EmbeddingOpsCommonConfigLoader.load(context)
    assert tbeconfig.Es is not None, "E list is not provided"
    assert tbeconfig.Ds is not None, "D list is not provided"
    # Generate feature_requires_grad
    feature_requires_grad = (
        generate_feature_requires_grad(tbeconfig, weighted_num_requires_grad)
        if weighted_num_requires_grad
        else None
    )

    # Determine the optimizer
    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD if row_wise else OptimType.EXACT_ADAGRAD

    # Construct the common split arguments for the embedding op
    common_split_args: Dict[str, Any] = embconfig.split_args() | {
        "optimizer": optimizer,
        "learning_rate": 0.1,
        "eps": 0.1,
        "feature_table_map": list(range(tbeconfig.T)),
    }
    assert tbeconfig.batch_params.Bs is not None, "B list is not provided"

    batch_size_per_feature_per_rank = None
    if tbeconfig.batch_params.sigma_B is not None:
        batch_size_per_feature_per_rank = []
        for b in tbeconfig.batch_params.Bs:
            batch_size_per_feature_per_rank.append([b])

    managed_option = (
        EmbeddingLocation.DEVICE
        if get_available_compute_device() == ComputeDevice.CUDA
        else EmbeddingLocation.HOST
    )

    if emb_op_type == "dense":
        embedding_op = DenseTableBatchedEmbeddingBagsCodegen(
            [
                (
                    e,
                    d,
                )
                for e, d in zip(tbeconfig.Es, tbeconfig.Ds)
            ],
            pooling_mode=embconfig.pooling_mode,
            use_cpu=not torch.cuda.is_available(),
        )
    elif emb_op_type == "ssd":
        assert (
            torch.cuda.is_available()
        ), "SSDTableBatchedEmbeddingBags only supports GPU execution"
        cache_set = max(sum(tbeconfig.batch_params.Bs), 1)
        tempdir = tempfile.mkdtemp(prefix=ssd_prefix)
        embedding_op = SSDTableBatchedEmbeddingBags(
            embedding_specs=[(e, d) for e, d in zip(tbeconfig.Es, tbeconfig.Ds)],
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
                    e,
                    d,
                    managed_option,
                    get_available_compute_device(),
                )
                for e, d in zip(tbeconfig.Es, tbeconfig.Ds)
            ],
            cache_precision=(
                embconfig.weights_dtype
                if embconfig.cache_dtype is None
                else embconfig.cache_dtype
            ),
            cache_algorithm=CacheAlgorithm.LRU,
            cache_load_factor=cache_load_factor,
            device=get_device(),
            **common_split_args,
        ).to(get_device())
    embedding_op = embedding_op.to(get_device())

    if embconfig.weights_dtype == SparseType.INT8:
        # pyre-fixme[29]: `Union[(self: DenseTableBatchedEmbeddingBagsCodegen,
        #  min_val: float, max_val: float) -> None, (self:
        #  SplitTableBatchedEmbeddingBagsCodegen, min_val: float, max_val: float) ->
        #  None, Tensor, Module]` is not a function.
        embedding_op.init_embedding_weights_uniform(-0.0003, 0.0003)

    avg_B = int(np.average(tbeconfig.batch_params.Bs))

    nparams = sum(d * e for e, d in zip(tbeconfig.Es, tbeconfig.Ds))
    param_size_multiplier = embconfig.weights_dtype.bit_rate() / 8.0
    output_size_multiplier = embconfig.output_dtype.bit_rate() / 8.0
    if embconfig.pooling_mode.do_pooling():
        read_write_bytes = (
            output_size_multiplier * avg_B * sum(tbeconfig.Ds)
            + param_size_multiplier
            * avg_B
            * sum(tbeconfig.Ds)
            * tbeconfig.pooling_params.L
        )
    else:
        read_write_bytes = (
            output_size_multiplier
            * avg_B
            * sum(tbeconfig.Ds)
            * tbeconfig.pooling_params.L
            + param_size_multiplier
            * avg_B
            * sum(tbeconfig.Ds)
            * tbeconfig.pooling_params.L
        )

    logging.info(f"Managed option: {embconfig.embedding_location}")
    logging.info(
        f"Embedding parameters: {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f} GB"
    )
    logging.info(
        f"Accessed weights per batch: {avg_B * sum(tbeconfig.Ds) * tbeconfig.pooling_params.L * param_size_multiplier / 1.0e9: .2f} GB"
    )

    if pooling_list is not None:
        pooling_list_extracted = [float(x) for x in pooling_list.split(",")]
        tensor_pooling_list = torch.tensor(pooling_list_extracted)
        requests = generate_requests_with_Llist(
            tbeconfig,
            tensor_pooling_list,
            benchconfig.num_requests,
            batch_size_per_feature_per_rank,
        )
    else:
        requests = generate_requests(
            tbeconfig, benchconfig.num_requests, batch_size_per_feature_per_rank
        )

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
            profile(on_trace_ready=on_trace_ready, with_stack=True, record_shapes=True)
            if benchconfig.export_trace
            else nullcontext()
        )

    #  to add batch_size_per_feature_per_rank, Yan's edit

    if torch.cuda.is_available():
        with _context_factory(lambda p: _kineto_trace_handler(p, "fwd")):
            # forward
            time_per_iter = benchmark_requests_with_spec(
                requests,
                lambda indices, offsets, per_sample_weights, batch_size_per_feature_per_rank: embedding_op.forward(
                    indices.to(dtype=tbeconfig.indices_params.index_dtype),
                    offsets.to(dtype=tbeconfig.indices_params.offset_dtype),
                    per_sample_weights,
                    feature_requires_grad=feature_requires_grad,
                    batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                ),
                flush_gpu_cache_size_mb=benchconfig.flush_gpu_cache_size_mb,
                num_warmups=benchconfig.warmup_iterations,
                iters=benchconfig.iterations,
            )
    else:
        time_per_iter = benchmark_requests_with_spec(
            requests,
            lambda indices, offsets, per_sample_weights, batch_size_per_feature_per_rank: embedding_op.forward(
                indices.to(dtype=tbeconfig.indices_params.index_dtype),
                offsets.to(dtype=tbeconfig.indices_params.offset_dtype),
                per_sample_weights,
                feature_requires_grad=feature_requires_grad,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
            ),
            flush_gpu_cache_size_mb=benchconfig.flush_gpu_cache_size_mb,
            num_warmups=benchconfig.warmup_iterations,
            iters=benchconfig.iterations,
        )

    avg_E = int(np.average(tbeconfig.E))
    avg_D = int(np.average(tbeconfig.D))
    logging.info(
        f"Forward, B: {avg_B}, "
        f"E: {avg_E}, T: {tbeconfig.T}, D: {avg_D}, L: {tbeconfig.pooling_params.L}, W: {tbeconfig.weighted}, "
        f"BW: {read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    if embconfig.output_dtype == SparseType.INT8:
        # backward bench not representative
        return

    if embconfig.pooling_mode.do_pooling():
        if batch_size_per_feature_per_rank is None:
            grad_output = torch.randn(avg_B, sum(tbeconfig.Ds)).to(get_device())
        else:
            output_size = sum(
                [b * d for (b, d) in zip(tbeconfig.batch_params.Bs, tbeconfig.Ds)]
            )
            grad_output = torch.randn(output_size).to(get_device())

    else:
        grad_output = torch.randn(
            avg_B * tbeconfig.T * tbeconfig.pooling_params.L,
            avg_D,
        ).to(get_device())
    assert (
        batch_size_per_feature_per_rank is None or grad_output.dim() == 1
    ), f"VBE expects 1D grad_output but got {grad_output.shape}"
    if torch.cuda.is_available():
        with _context_factory(lambda p: _kineto_trace_handler(p, "fwd_bwd")):
            # backward
            time_per_iter = benchmark_requests_with_spec(
                requests,
                lambda indices, offsets, per_sample_weights, batch_size_per_feature_per_rank: embedding_op(
                    indices.to(dtype=tbeconfig.indices_params.index_dtype),
                    offsets.to(dtype=tbeconfig.indices_params.offset_dtype),
                    per_sample_weights,
                    batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                    feature_requires_grad=feature_requires_grad,
                ),
                flush_gpu_cache_size_mb=benchconfig.flush_gpu_cache_size_mb,
                bwd_only=True,
                grad=grad_output,
                num_warmups=benchconfig.warmup_iterations,
                iters=benchconfig.iterations,
            )
    else:
        time_per_iter = benchmark_requests_with_spec(
            requests,
            lambda indices, offsets, per_sample_weights, batch_size_per_feature_per_rank: embedding_op(
                indices.to(dtype=tbeconfig.indices_params.index_dtype),
                offsets.to(dtype=tbeconfig.indices_params.offset_dtype),
                per_sample_weights,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                feature_requires_grad=feature_requires_grad,
            ),
            flush_gpu_cache_size_mb=benchconfig.flush_gpu_cache_size_mb,
            bwd_only=True,
            grad=grad_output,
            num_warmups=benchconfig.warmup_iterations,
            iters=benchconfig.iterations,
        )

    logging.info(
        f"Backward, B: {avg_B}, E: {avg_E}, T: {tbeconfig.T}, D: {avg_D}, L: {tbeconfig.pooling_params.L}, "
        f"BW: {2 * read_write_bytes / time_per_iter / 1.0e9: .2f} GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


if __name__ == "__main__":
    cli()

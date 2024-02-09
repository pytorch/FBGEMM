#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import signal
from typing import List, Tuple

import click
import fbgemm_gpu
import numpy as np
import tabulate
import torch

from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)

from torch import Tensor
from torch.profiler import profile, ProfilerActivity

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from bench_utils import benchmark_torch_function
else:
    from fbgemm_gpu.bench.bench_utils import benchmark_torch_function

    if torch.version.hip:
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_hip"
        )
    else:
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings"
        )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_cpu"
    )


# pyre-fixme[2]: Parameter must be annotated.
def get_gpu_device(gpu_num) -> torch.device:
    return torch.device(f"cuda:{gpu_num}")


# Merged indices with shape (T, B, L) -> (flattened indices with shape
# (T * B * L), offsets with shape (T * B + 1)).
# Reference: https://fburl.com/code/5ueyfv5j
def get_table_batched_offsets_from_dense(
    merged_indices: torch.Tensor,
    # pyre-fixme[2]: Parameter must be annotated.
    gpu_num,
) -> Tuple[torch.Tensor, torch.Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.int().contiguous().view(-1).to(device=get_gpu_device(gpu_num)),
        torch.tensor(
            ([0] + np.cumsum(flat_lengths).tolist()), device=get_gpu_device(gpu_num)
        ).int(),
    )


# Reference: https://fburl.com/code/o5600si0
def generate_requests(
    num_gpus: int,
    B: int,
    T: int,
    L: int,
    E: int,
    # inter-batch indices reuse rate
    reuse: float = 0.0,
) -> List[Tuple[torch.IntTensor, torch.IntTensor, None]]:
    rs = []
    for gpu_num in range(num_gpus):
        all_indices = torch.randint(
            low=0,
            high=E,
            size=(T, B, L),
            device=get_gpu_device(gpu_num),
            dtype=torch.int32,
        )
        # each bag is usually sorted
        (all_indices, _) = torch.sort(all_indices)
        all_indices = all_indices.reshape(T, B * L)

        rs.append(
            get_table_batched_offsets_from_dense(all_indices.view(T, B, L), gpu_num)
        )
    return rs


# pyre-fixme[3]: Return type must be annotated.
def _get_random_tensor(
    num_ads: int,
    embedding_dimension: int,
    ads_tables: int,
    data_type: str,
    gpu_idx: int,
    include_quantization: bool,
):
    if data_type == "FP16" or include_quantization:
        result_tensor = torch.randn(
            num_ads,
            embedding_dimension * ads_tables,
            dtype=torch.float16,
            device=torch.device(f"cuda:{gpu_idx}"),
        )
    elif data_type == "INT8":
        assert (
            embedding_dimension % 2
        ) == 0, "needs to align to 2 bytes (half type size) for INT8"
        result_tensor = torch.randint(
            0,
            255,
            # 2 FP16 numbers for scale and bias, total of 4 bytes overhead
            size=(num_ads, (embedding_dimension + 4) * ads_tables),
            dtype=torch.uint8,
            device=torch.device(f"cuda:{gpu_idx}"),
        )
    elif data_type == "INT4":
        assert (
            embedding_dimension % 4
        ) == 0, "needs to align to 2 bytes (half type size) for INT4"
        result_tensor = torch.randint(
            0,
            255,
            # Using torch.uint8 for int4 storage
            size=(num_ads, (embedding_dimension // 2 + 4) * ads_tables),
            dtype=torch.uint8,
            device=torch.device(f"cuda:{gpu_idx}"),
        )
    else:
        raise ValueError

    return result_tensor


# pyre-fixme[3]: Return type must be annotated.
def generate_tbe(
    # pyre-fixme[2]: Parameter must be annotated.
    batch_indices,
    num_ads: int,
    embedding_dimension: int,
    num_of_embeddings: int,
    pooling_factor: int,
    ads_tables: int,
    fused_tbe: bool,
    data_type: str,
    num_gpus: int,
):
    B = num_ads
    D = embedding_dimension
    E = num_of_embeddings
    L = pooling_factor
    T = ads_tables
    Ds = [D] * T
    managed_option = EmbeddingLocation.DEVICE

    output_dtype = SparseType.FP16
    if fused_tbe:
        assert data_type == "INT8"  # INT4 not implemented yet
        output_dtype = SparseType.INT8

    emb = [
        IntNBitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    str(idx),
                    E,
                    d,
                    SparseType.INT4,
                    managed_option,
                )
                for d in Ds
            ],
            output_dtype=output_dtype,
            device=get_gpu_device(idx),
            bounds_check_mode=BoundsCheckMode.NONE,
        )
        for idx in range(num_gpus)
    ]
    for e in emb:
        e.fill_random_weights()
    requests = generate_requests(num_gpus, B, T, L, E)
    # https://fburl.com/code/doxxjc8c
    SIZE_OF_FLOAT = 4
    num_elem_per_byte = 1 if data_type == "INT8" else 2
    assert embedding_dimension % (2 * num_elem_per_byte) == 0
    col_sizes = (
        [
            (embedding_dimension + num_elem_per_byte - 1) // num_elem_per_byte
            + 2 * SIZE_OF_FLOAT
        ]
        * ads_tables
        * num_gpus
    )
    offset = torch.tensor([0] + col_sizes, device=batch_indices.device)
    tbe_offset = torch.cumsum(offset, dim=0).to(torch.int).cuda()

    return emb, requests, tbe_offset


def print_p2p_bandwidth(
    # pyre-fixme[2]: Parameter must be annotated.
    num_gpus,
    # pyre-fixme[2]: Parameter must be annotated.
    iters,
    # pyre-fixme[2]: Parameter must be annotated.
    pooled_ad_embeddings,
    # pyre-fixme[2]: Parameter must be annotated.
    bytes_per_element,
) -> None:
    print("Pairwise GPU Copy Bandwidth (GB/s)")
    p2p_copy_bw = np.zeros((num_gpus, num_gpus))
    for i in range(num_gpus):
        for j in range(num_gpus):
            with torch.cuda.device(i):
                t, _ = benchmark_torch_function(
                    lambda: (
                        pooled_ad_embeddings[i].copy_(pooled_ad_embeddings[j])
                        if i != j
                        else pooled_ad_embeddings[i].clone()
                    ),
                    (),
                    flush_gpu_cache_size_mb=0,
                    iters=iters,
                )
                p2p_copy_bw[i, j] = (
                    pooled_ad_embeddings[i].numel() * bytes_per_element / t / 1.0e9
                )
    table = tabulate.tabulate(
        p2p_copy_bw,
        headers=[f"GPU {i}" for i in range(num_gpus)],
        tablefmt="fancy_grid",
        floatfmt=".0f",
    )
    print(table)


def benchmark(  # noqa C901
    all_to_one_only: bool,
    sum_reduce_to_one_only: bool,
    num_ads: int,
    embedding_dimension: int,
    ads_tables: int,
    iters: int = 10,
    p2p_bw: bool = False,
    dst_device: int = 0,
    data_type: str = "FP16",
    mode: str = "P2P",
    skip_dequantization: bool = False,
    num_of_embeddings: int = 10000,
    pooling_factor: int = 25,
) -> str:
    assert torch.cuda.is_available()
    torch.cuda.set_device(dst_device)
    num_gpus = torch.cuda.device_count()
    batch_indices = torch.zeros(num_ads).long().cuda()
    include_quantization = not mode == "P2P"
    # Using torch.int8 for int4 storage
    bytes_per_element = 2 if (data_type == "FP16" or include_quantization) else 1
    total_elements = num_ads * embedding_dimension * ads_tables * num_gpus

    logging.debug(
        f"B: {num_ads}, D: {embedding_dimension}, T: {ads_tables}, Data Type: {data_type}, Num GPUs: {num_gpus}, Destination GPU: {dst_device}"
    )

    fused_tbe = mode == "P2P_FUSED_TBE"
    include_tbe = fused_tbe or mode == "P2P_TBE"
    if include_tbe:
        emb, requests, tbe_offset = generate_tbe(
            batch_indices,
            num_ads,
            embedding_dimension,
            num_of_embeddings,
            pooling_factor,
            ads_tables,
            fused_tbe,
            data_type,
            num_gpus,
        )

    pooled_ad_embeddings = [
        _get_random_tensor(
            num_ads,
            embedding_dimension,
            ads_tables,
            data_type,
            gpu_idx,
            include_quantization,
        )
        for gpu_idx in range(num_gpus)
    ]

    if p2p_bw:
        print_p2p_bandwidth(num_gpus, iters, pooled_ad_embeddings, bytes_per_element)

    # pyre-fixme[53]: Captured variable `emb` is not annotated.
    # pyre-fixme[53]: Captured variable `pooled_ad_embeddings` is not annotated.
    # pyre-fixme[53]: Captured variable `requests` is not annotated.
    # pyre-fixme[53]: Captured variable `tbe_offset` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    def pool_func_with_quantization(
        # pyre-fixme[2]: Parameter must be annotated.
        batch_indices,
        # pyre-fixme[2]: Parameter must be annotated.
        include_quantization,
        # pyre-fixme[2]: Parameter must be annotated.
        include_tbe,
        # pyre-fixme[2]: Parameter must be annotated.
        fused_tbe,
        # pyre-fixme[2]: Parameter must be annotated.
        skip_dequantization,
        # pyre-fixme[2]: Parameter must be annotated.
        data_type,
    ):
        if include_tbe:
            embedding_results = []
            for idx, (indices, offsets) in enumerate(requests):
                with torch.cuda.device(idx):
                    embedding_results.append(emb[idx].forward(indices, offsets))
        else:
            embedding_results = pooled_ad_embeddings

        if data_type == "FP16" or (not fused_tbe and not include_quantization):
            if all_to_one_only:
                return torch.ops.fbgemm.all_to_one_device(
                    pooled_ad_embeddings, batch_indices.device
                )
            elif sum_reduce_to_one_only:
                return torch.ops.fbgemm.sum_reduce_to_one(
                    pooled_ad_embeddings, batch_indices.device
                )
            else:
                return torch.ops.fbgemm.merge_pooled_embeddings(
                    embedding_results, batch_indices.size(0), batch_indices.device
                )

        assert data_type == "INT8" or data_type == "INT4"
        assert not all_to_one_only  # not supported
        if fused_tbe:
            pooled_quantized_result = torch.ops.fbgemm.merge_pooled_embeddings(
                embedding_results, batch_indices.size(0), batch_indices.device
            )
        else:
            quantized = []
            for t in embedding_results:
                t_split_by_table = torch.split(t, embedding_dimension, dim=1)
                quantized_split_by_table = [
                    (
                        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(t.float())
                        if data_type == "INT8"
                        else torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                            t.float(), 4
                        )
                    )
                    for t in t_split_by_table
                ]
                result = torch.cat(quantized_split_by_table, dim=1)
                quantized.append(result)
            pooled_quantized_result = torch.ops.fbgemm.merge_pooled_embeddings(
                quantized, batch_indices.size(0), batch_indices.device
            )

        if skip_dequantization:
            return pooled_quantized_result

        PooledEmbeddingDequantizeDataTypeFP16 = 1
        if data_type == "INT8":
            return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatMixedDim(
                pooled_quantized_result,
                tbe_offset,
                PooledEmbeddingDequantizeDataTypeFP16,
            )
        else:
            # TODO: the result here is wrong. Once MixedDim version for FusedNBit quantization is done, switch to that.
            # Since their performance is similar, keep using Fused8BitRowwiseQuantizedToHalf for now.
            return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToHalf(
                pooled_quantized_result
            ).half()

    streams = [torch.cuda.Stream(device=i) for i in range(num_gpus)]
    import contextlib

    with contextlib.ExitStack() as stack:
        for stream in streams:
            stack.enter_context(torch.cuda.stream(stream))

        # warm up
        merged = pool_func_with_quantization(
            batch_indices,
            include_quantization,
            include_tbe,
            fused_tbe,
            skip_dequantization,
            data_type,
        )
        if all_to_one_only:
            merged = torch.stack(merged)
        t, _ = benchmark_torch_function(
            pool_func_with_quantization,
            (
                batch_indices,
                include_quantization,
                include_tbe,
                fused_tbe,
                skip_dequantization,
                data_type,
            ),
            flush_gpu_cache_size_mb=0,
            iters=iters,
        )
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            pool_func_with_quantization(
                batch_indices,
                include_quantization,
                include_tbe,
                fused_tbe,
                skip_dequantization,
                data_type,
            )
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    if isinstance(merged, Tensor):
        # all_to_one_only returns a list of tensors,
        # otherwise, it's a Tensor.
        merged = [merged]

    output_num_el = sum([a.numel() for a in merged])
    # Assume tensors gathered are all the same size.
    num_el_transferred = output_num_el * (num_gpus - 1) / num_gpus

    logging.debug(
        f"Mode: {mode}, Data Type: {data_type}, B: {num_ads}, D: {embedding_dimension}, T: {ads_tables}, Num GPUs: {num_gpus}, Destination GPU: {dst_device}, all_to_one_only: {all_to_one_only}, "
        f"Number of elements: {total_elements / 1.0e6:.2f}, Million, Number of elements per GPU: {total_elements / 1.0e6 / num_gpus:.2f}, Billion elements per sec: {total_elements / t / 1.0e9:.1f}, "
        f"Output Size: {output_num_el * bytes_per_element / 1.0e6:.0f}MB, Num elements transferred: {num_el_transferred / 1.0e6}, All-to-one BW: {output_num_el * bytes_per_element / t / 1.0e9:.1f}GB/s, link BW: {num_el_transferred * bytes_per_element / t / 1.0e9:.1f}GB/s, "
        f"t: {t * 1.0e3:.2f}ms"
    )
    # return result in CSV format
    return (
        f"{mode}, {data_type}, {num_ads}, {embedding_dimension}, {ads_tables}, {num_gpus}, {dst_device}, {all_to_one_only}, "
        f"{total_elements / 1.0e6:.2f}, {total_elements / 1.0e6 / num_gpus:.2f}, {total_elements / 1.0e9 / t:.1f}, "
        f"{output_num_el * bytes_per_element / 1.0e6:.0f}, {output_num_el * bytes_per_element / t / 1.0e9:.1f}, "
        f"{num_el_transferred * bytes_per_element / 1.0e9 / t:.1f}, "
        f"{t * 1.0e3:.2f}"
    )


@click.command()
@click.option("--all-to-one-only", is_flag=True, default=False)
@click.option("--sum-reduce-to-one-only", is_flag=True, default=False)
@click.option("--num_ads", default=1024, type=int)
@click.option("--embedding_dimension", default=300, type=int)
@click.option("--ads_tables", default=100, type=int)
@click.option("--iters", default=10, type=int)
@click.option("--p2p_bw", is_flag=True, default=False)
@click.option("--dst_device", default=0, type=int)
@click.option(
    "--data_type",
    type=click.Choice(["FP16", "INT8", "INT4"]),
    default="FP16",
)
# P2P: merge_pooled_embeddings() or all_to_one_device() for tensor with "--data_type"
# P2P_QUANT: for INT8/INT4 data type, start with FP16, then quantize -> P2P -> dequantize to FP16
# P2P_TBE: add TBE in front of P2P_QUANT.  When "--data_type" is FP16, the flow is TBE -> P2P; for INT8/INT4, the flow is TBE -> quantize -> P2P -> dequantize
# P2P_FUSED_TBE: similar to P2P_TBE except fuse the quantization into TBE
@click.option(
    "--mode",
    type=click.Choice(["P2P", "P2P_QUANT", "P2P_TBE", "P2P_FUSED_TBE"]),
    default="P2P",
)
# For quantized communication, do we dequantize back to FP16 in the end.
@click.option("--skip_dequantization", is_flag=True, default=False)
@click.option("--num_of_embeddings", default=100000, type=int)
@click.option("--pooling_factor", default=25, type=int)
@click.option("--sweep", is_flag=True, default=False)
def main(
    all_to_one_only: bool,
    sum_reduce_to_one_only: bool,
    num_ads: int,
    embedding_dimension: int,
    ads_tables: int,
    iters: int,
    p2p_bw: bool,
    dst_device: int,
    data_type: str,
    mode: str,
    skip_dequantization: bool,
    num_of_embeddings: int,
    pooling_factor: int,
    sweep: bool,
) -> None:
    csv_header = (
        "mode, data_type, num_ads, embedding_dimension, ads_tables, num_gpus, dst_device, all_to_one_only, "
        "number of elements (Million), number of elements per GPU (Million), throughput (billion elements per sec), "
        "output size (MB), all-to-one BW (GB/s), link BW (GB/s), t (ms)"
    )
    if sweep:
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def handler(signum, frame):
            logging.error("timeout")
            raise TimeoutError()

        results = []
        num_gpu = torch.cuda.device_count()
        for num_ads in [128, 256, 512, 1024, 2048]:
            # Scale num_ads so all GPUs have sweep through the same number of total elements
            num_ads *= 8 // num_gpu
            for embedding_dimension in [16, 64, 112, 304]:
                for ads_tables in [25, 50, 100, 400, 800]:
                    if num_ads * embedding_dimension * ads_tables > 983040000:
                        continue  # Skip tests that are too large
                    signal.signal(signal.SIGTERM, handler)
                    signal.alarm(600)
                    logging.info(
                        f"config: num_ads: {num_ads}, embedding_dimension: {embedding_dimension}, ads_tables: {ads_tables}"
                    )
                    try:
                        result = benchmark(
                            all_to_one_only,
                            sum_reduce_to_one_only,
                            num_ads,
                            embedding_dimension,
                            ads_tables,
                            iters,
                            p2p_bw,
                            dst_device,
                            data_type,
                            mode,
                            skip_dequantization,
                            num_of_embeddings,
                            pooling_factor,
                        )
                        results.append(result)
                    except (TimeoutError, RuntimeError) as err:
                        logging.error(
                            f"B: {num_ads}, D: {embedding_dimension}, T: {ads_tables}, Data Type: {data_type}, Num GPU: {num_gpu}, time out or failed: {err}"
                        )
        print(csv_header)
        print(*results, sep="\n")
        return

    result = benchmark(
        all_to_one_only,
        sum_reduce_to_one_only,
        num_ads,
        embedding_dimension,
        ads_tables,
        iters,
        p2p_bw,
        dst_device,
        data_type,
        mode,
        skip_dequantization,
        num_of_embeddings,
        pooling_factor,
    )
    print(csv_header)
    print(result)


if __name__ == "__main__":
    main()

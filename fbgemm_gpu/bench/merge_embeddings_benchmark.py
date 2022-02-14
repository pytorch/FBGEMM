#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import signal

import click
import numpy as np
import tabulate
import torch

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings")
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_cpu"
    )
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


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


def benchmark(
    all_to_one_only,
    num_ads,
    embedding_dimension,
    ads_tables,
    iters: int = 10,
    p2p_bw: bool = False,
    dst_device: int = 0,
    data_type: str = "FP16",
    include_quantization: bool = False,
) -> str:
    torch.cuda.set_device(dst_device)
    num_gpus = torch.cuda.device_count()
    batch_indices = torch.zeros(num_ads).long().cuda()
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
    # Using torch.int8 for int4 storage
    bytes_per_element = 2 if (data_type == "FP16" or include_quantization) else 1
    total_elements = num_ads * embedding_dimension * ads_tables * num_gpus

    logging.debug(
        f"B: {num_ads}, D: {embedding_dimension}, T: {ads_tables}, Data Type: {data_type}, Num GPUs: {num_gpus}, Destination GPU: {dst_device}"
    )

    def benchmark_torch_function(iters: int, f, *args) -> float:
        f(*args)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iters):
            f(*args)
        end_event.record()
        torch.cuda.synchronize()
        return (start_event.elapsed_time(end_event) * 1.0e-3) / iters

    if p2p_bw:
        print("Pairwise GPU Copy Bandwidth (GB/s)")
        p2p_copy_bw = np.zeros((num_gpus, num_gpus))
        for i in range(num_gpus):
            for j in range(num_gpus):
                with torch.cuda.device(i):
                    t = benchmark_torch_function(
                        iters,
                        lambda: pooled_ad_embeddings[i].copy_(pooled_ad_embeddings[j])
                        if i != j
                        else pooled_ad_embeddings[i].clone(),
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

    streams = [torch.cuda.Stream(device=i) for i in range(num_gpus)]
    import contextlib

    def pool_func_with_quantization(
        pooled_ad_embeddings,
        batch_indices,
        include_quantization,
        data_type,
    ):
        if include_quantization:
            assert data_type == "INT8" or data_type == "INT4"
            quantized = [
                torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(t.float())
                if data_type == "INT8"
                else torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                    t.float(), 4
                )
                for t in pooled_ad_embeddings
            ]
            pooled_quantized_result = torch.ops.fbgemm.merge_pooled_embeddings(
                quantized, batch_indices.size(0), batch_indices.device
            )
            PooledEmbeddingDequantizeDataTypeFP16 = 1

            if data_type == "INT8":
                offset = torch.cumsum(
                    torch.tensor(
                        [0] + [quantized[0].shape[1] for _ in range(len(quantized))],
                        device=batch_indices.device,
                    ),
                    dim=0,
                ).to(torch.int)
                return torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloatMixedDim(
                    pooled_quantized_result,
                    offset,
                    PooledEmbeddingDequantizeDataTypeFP16,
                )
            else:
                # TODO: the result here is wrong. Once MixedDim version for FusedNBit quantization is done, switch to that.
                # Since their performance is similar, keep using FusedNBitRowwiseQuantizedSBHalfToFloat for now.
                return torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(
                    pooled_quantized_result, 4
                ).half()

        if all_to_one_only:
            return torch.ops.fbgemm.all_to_one_device(
                pooled_ad_embeddings, batch_indices.device
            )
        else:
            return torch.ops.fbgemm.merge_pooled_embeddings(
                pooled_ad_embeddings, batch_indices.size(0), batch_indices.device
            )

    with contextlib.ExitStack() as stack:
        for stream in streams:
            stack.enter_context(torch.cuda.stream(stream))

        merged = pool_func_with_quantization(
            pooled_ad_embeddings, batch_indices, include_quantization, data_type
        )
        t = benchmark_torch_function(
            iters,
            lambda: pool_func_with_quantization(
                pooled_ad_embeddings, batch_indices, include_quantization, data_type
            ),
        )

    logging.debug(
        f"Merge, B: {num_ads}, D: {embedding_dimension}, T: {ads_tables}, Data Type: {data_type}, Num GPUs: {num_gpus}, Destination GPU: {dst_device}, "
        f"Number of elements: {total_elements / 1.0e6:.0f} Million, Billion elements per sec: {total_elements / t / 1.0e9:.1f}, "
        f"Output Size: {merged.numel() * bytes_per_element / 1.0e6:.0f}MB, BW: {merged.numel() * bytes_per_element / t / 1.0e9:.1f}GB/s, "
        f"t: {t * 1.0e3:.2f}ms"
    )
    # return result in CSV format
    return (
        f"{num_ads}, {embedding_dimension}, {ads_tables}, {data_type}, {num_gpus}, {dst_device}, "
        f"{total_elements / 1.0e6:.0f}, {total_elements / t / 1.0e9:.1f}, "
        f"{merged.numel() * bytes_per_element / 1.0e6:.0f}, {merged.numel() * bytes_per_element / t / 1.0e9:.1f}, "
        f"{t * 1.0e3:.2f}"
    )


@click.command()
@click.option("--all-to-one-only", is_flag=True, default=False)
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
# For INT8/INT4 data type, whether to start with FP16 and include quantization overhead
@click.option("--include_quantization", is_flag=True, default=False)
@click.option("--sweep", is_flag=True, default=False)
def main(
    all_to_one_only,
    num_ads,
    embedding_dimension,
    ads_tables,
    iters,
    p2p_bw,
    dst_device,
    data_type,
    include_quantization,
    sweep,
) -> None:
    assert sweep or not (
        include_quantization and data_type == "FP16"
    ), "no quantization is needed for FP16"

    csv_header = (
        "num_ads, embedding_dimension, ads_tables, data_type, num_gpus,"
        "dst_device, number of elements (Million), throughput (billion elements per sec), "
        "output size (MB), BW (GB/s), t (ms)"
    )
    if sweep:

        def handler(signum, frame):
            logging.error("timeout")
            raise TimeoutError()

        results = []
        num_gpu = torch.cuda.device_count()
        for num_ads in [128, 256, 512, 1024, 2048]:
            # Scale num_ads so all GPUs have sweep through the same number of total elements
            num_ads *= 8 // num_gpu
            for embedding_dimension in [16, 64, 104, 300]:
                for ads_tables in [25, 50, 100, 400, 800]:
                    data_type_list = (
                        ["INT8", "INT4"]
                        if include_quantization
                        else ["FP16", "INT8", "INT4"]
                    )
                    for data_type in data_type_list:
                        if num_ads * embedding_dimension * ads_tables > 1228800000:
                            continue  # Skip tests that are too large
                        signal.signal(signal.SIGTERM, handler)
                        signal.alarm(600)
                        try:
                            result = benchmark(
                                all_to_one_only,
                                num_ads,
                                embedding_dimension,
                                ads_tables,
                                iters,
                                p2p_bw,
                                dst_device,
                                data_type,
                                include_quantization,
                            )
                            results.append(result)
                        except (TimeoutError, RuntimeError) as err:
                            logging.error(f"timed out or failed: {err}")
                            logging.error(
                                f"B: {num_ads}, D: {embedding_dimension}, T: {ads_tables}, Data Type: {data_type}, Num GPU: {num_gpu}"
                            )
        print(csv_header)
        print(*results, sep="\n")
        return

    result = benchmark(
        all_to_one_only,
        num_ads,
        embedding_dimension,
        ads_tables,
        iters,
        p2p_bw,
        dst_device,
        data_type,
        include_quantization,
    )
    print(csv_header)
    print(result)


if __name__ == "__main__":
    main()

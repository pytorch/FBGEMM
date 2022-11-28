# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, List, Optional, Tuple, TypeVar

import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType  # usort:skip

# pyre-fixme[21]: Could not find name `default_rng` in `numpy.random` (stubbed).
from numpy.random import default_rng

logging.basicConfig(level=logging.DEBUG)
Deviceable = TypeVar(
    "Deviceable", torch.nn.EmbeddingBag, torch.nn.Embedding, torch.Tensor
)


def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def get_device() -> torch.device:
    # pyre-fixme[7]: Expected `device` but got `Union[int, device]`.
    return (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def to_device(t: Deviceable, use_cpu: bool) -> Deviceable:
    # pyre-fixme[7]: Expected `Deviceable` but got `Union[Tensor, Embedding,
    #  EmbeddingBag]`.
    return t.cpu() if use_cpu else t.cuda()


# Merged indices with shape (T, B, L) -> (flattened indices with shape
# (T * B * L), offsets with shape (T * B + 1))
def get_table_batched_offsets_from_dense(
    merged_indices: torch.Tensor, use_cpu: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        to_device(merged_indices.contiguous().view(-1), use_cpu),
        to_device(
            torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long(),
            use_cpu,
        ),
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
            to_device(indices, use_cpu),
            to_device(offsets, use_cpu),
            per_sample_weights=per_sample_weights,
        )
    else:
        return b(to_device(indices, use_cpu))


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
    zipf_oversample_ratio: int = 3,
    weighted: bool = False,
    requests_data_file: Optional[str] = None,
    # Comma-separated list of table numbers
    tables: Optional[str] = None,
) -> List[Tuple[torch.IntTensor, torch.IntTensor, Optional[torch.Tensor]]]:
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
        zipf_shape = (iters, T, B, zipf_oversample_ratio * L)
        if torch.cuda.is_available():
            zipf_shape_total_len = np.prod(zipf_shape)
            all_indices_list = []
            # process 8 GB at a time on GPU
            chunk_len = int(1e9)
            for chunk_begin in range(0, zipf_shape_total_len, chunk_len):
                all_indices_gpu = torch.ops.fbgemm.zipf_cuda(
                    alpha,
                    min(zipf_shape_total_len - chunk_begin, chunk_len),
                    seed=torch.randint(2**31 - 1, (1,))[0],
                )
                all_indices_list.append(all_indices_gpu.cpu())
            all_indices = torch.cat(all_indices_list).reshape(zipf_shape)
        else:
            all_indices = torch.as_tensor(np.random.zipf(a=alpha, size=zipf_shape))
        all_indices = (all_indices - 1) % E
        all_indices = torch.ops.fbgemm.bottom_unique_k_per_row(all_indices, L)
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
            None
            if not weighted
            else torch.randn(
                T * B * L, device=get_device()
            )  # per sample weights will always be FP32
        )
        rs.append(
            get_table_batched_offsets_from_dense(all_indices[it].view(T, B, L))
            + (weights_tensor,)
        )
    return rs


def quantize_embs(
    weight: torch.Tensor, weight_ty: SparseType
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if weight_ty == SparseType.FP32:
        q_weight = weight.float()
        # FIXME: How to view the PyTorch Tensor as a different type (e.g., uint8)
        # Here it uses numpy and it will introduce DtoH/HtoD overhead.
        res_weight = torch.tensor(q_weight.cpu().numpy().view(np.uint8)).contiguous()
        return (res_weight, None)

    elif weight_ty == SparseType.FP16:
        q_weight = weight.half()
        res_weight = torch.tensor(q_weight.cpu().numpy().view(np.uint8)).contiguous()
        return (res_weight, None)

    elif weight_ty == SparseType.INT8:
        q_weight = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(weight)
        res_weight = torch.tensor(q_weight[:, :-8].cpu().numpy().view(np.uint8))
        res_scale_shift = torch.tensor(
            q_weight[:, -8:]
            .contiguous()
            .cpu()
            .numpy()
            .view(np.float32)
            .astype(np.float16)
            .view(np.uint8)
        )  # [-4, -2]: scale; [-2:]: bias
        return (res_weight, res_scale_shift)

    elif weight_ty == SparseType.INT4 or weight_ty == SparseType.INT2:
        q_weight = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
            weight,
            bit_rate=weight_ty.bit_rate(),
        )
        res_weight = torch.tensor(q_weight[:, :-4].cpu().numpy().view(np.uint8))
        res_scale_shift = torch.tensor(
            q_weight[:, -4:].contiguous().cpu().numpy().view(np.uint8)
        )  # [-4, -2]: scale; [-2:]: bias
        return (res_weight, res_scale_shift)

    else:
        raise RuntimeError("Unsupported SparseType: {}".format(weight_ty))

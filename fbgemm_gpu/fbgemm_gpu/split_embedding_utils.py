# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, List, Optional, Tuple, TypeVar

import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import (
    FP8QuantizationConfig,
    SparseType,
)  # usort:skip

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
    # pyre-fixme[7]: Expected `Deviceable` but got `Union[Tensor,
    #  torch.nn.EmbeddingBag]`.
    return t.cpu() if use_cpu else t.cuda()


# Merged indices with shape (T, B, L) -> (flattened indices with shape
# (T * B * L), offsets with shape (T * B + 1))
def get_table_batched_offsets_from_dense(
    merged_indices: torch.Tensor,
    L: Optional[int] = None,
    total_B: Optional[int] = None,
    use_cpu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if L is None and total_B is None:
        (T, B, L) = merged_indices.size()
        total_B = T * B
    lengths = np.ones(total_B) * L
    return (
        to_device(merged_indices.contiguous().view(-1), use_cpu),
        to_device(
            torch.tensor(([0] + np.cumsum(lengths).tolist())).long(),
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


def generate_requests(  # noqa C901
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
    # If sigma_L is not None, treat L as mu_L and generate Ls from sigma_L
    # and mu_L
    sigma_L: Optional[int] = None,
    emulate_pruning: bool = False,
    use_cpu: bool = False,
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

    # Generate L from stats
    if sigma_L is not None:
        use_variable_L = True
        Ls = np.random.normal(loc=L, scale=sigma_L, size=T * B).astype(int)
        # Make sure that Ls are positive
        Ls[Ls < 0] = 0
        # Use the same L distribution across iters
        Ls = np.tile(Ls, iters)
        L = Ls.max()
        # Make it exclusive cumsum
        L_offsets = torch.from_numpy(np.insert(Ls.cumsum(), 0, 0)).to(torch.long)
    else:
        use_variable_L = False
        # Init to suppress the pyre error
        L_offsets = torch.empty(1)

    if alpha <= 1.0:
        all_indices = torch.randint(
            low=0,
            high=E,
            size=(iters, T, B, L),
            device="cpu" if use_variable_L else get_device(),
            dtype=torch.int32,
        )
        # each bag is usually sorted
        (all_indices, _) = torch.sort(all_indices)
        if use_variable_L:
            all_indices = torch.ops.fbgemm.bottom_k_per_row(
                all_indices.to(torch.long), L_offsets, False
            )
            all_indices = all_indices.to(get_device()).int()
        else:
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
        if use_variable_L:
            all_indices = torch.ops.fbgemm.bottom_k_per_row(
                all_indices, L_offsets, True
            )
        else:
            all_indices = torch.ops.fbgemm.bottom_k_per_row(
                all_indices, torch.tensor([0, L], dtype=torch.long), True
            )
        rng = default_rng()
        permutation = torch.as_tensor(
            rng.choice(E, size=all_indices.max().item() + 1, replace=False)
        )
        all_indices = permutation.gather(0, all_indices.flatten())
        all_indices = all_indices.to(get_device()).int()
        if not use_variable_L:
            all_indices = all_indices.reshape(iters, T, B * L)

    if reuse > 0.0:
        assert (
            not use_variable_L
        ), "Does not support generating Ls from stats for reuse > 0.0"

        for it in range(iters - 1):
            for t in range(T):
                reused_indices = torch.randperm(B * L, device=get_device())[
                    : int(B * L * reuse)
                ]
                all_indices[it + 1, t, reused_indices] = all_indices[
                    it, t, reused_indices
                ]

    # Some indices are set to -1 for emulating pruned rows.
    if emulate_pruning:
        for it in range(iters):
            for t in range(T):
                num_negative_indices = B // 2
                random_locations = torch.randint(
                    low=0,
                    high=(B * L),
                    size=(num_negative_indices,),
                    device=torch.cuda.current_device(),
                    dtype=torch.int32,
                )
                all_indices[it, t, random_locations] = -1

    rs = []
    for it in range(iters):
        if use_variable_L:
            start_offset = L_offsets[it * T * B]
            it_L_offsets = torch.concat(
                [
                    torch.zeros(1),
                    L_offsets[it * T * B + 1 : (it + 1) * T * B + 1] - start_offset,
                ]
            )
            weights_tensor = (
                None
                if not weighted
                else torch.randn(
                    int(it_L_offsets[-1].item()), device=get_device()
                )  # per sample weights will always be FP32
            )
            rs.append(
                (
                    all_indices[start_offset : L_offsets[(it + 1) * T * B]],
                    it_L_offsets.to(get_device()),
                    weights_tensor,
                )
            )
        else:
            weights_tensor = (
                None
                if not weighted
                else torch.randn(
                    T * B * L, device=get_device()
                )  # per sample weights will always be FP32
            )
            rs.append(
                get_table_batched_offsets_from_dense(
                    all_indices[it].view(T, B, L), use_cpu=use_cpu
                )
                + (weights_tensor,)
            )
    return rs


def quantize_embs(
    weight: torch.Tensor,
    weight_ty: SparseType,
    fp8_config: Optional[FP8QuantizationConfig] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    weight = weight.detach()
    if weight_ty == SparseType.FP32:
        q_weight = weight.float()
        res_weight = q_weight.view(torch.uint8)
        return (res_weight, None)

    elif weight_ty == SparseType.FP16:
        q_weight = weight.half()
        res_weight = q_weight.view(torch.uint8)
        return (res_weight, None)

    elif weight_ty == SparseType.FP8:
        assert fp8_config is not None
        # Quantize FP32 to HPF8
        res_weight = torch.ops.fbgemm.FloatToHFP8Quantized(
            weight.float(),
            fp8_config.get("exponent_bits"),
            fp8_config.get("exponent_bias"),
            fp8_config.get("max_position"),
        )
        return (res_weight, None)

    elif weight_ty == SparseType.INT8:
        # Note that FloatToFused8BitRowwiseQuantized might have additional padding
        # for alignment if embedding dimension is not a multiple of 4:
        # https://fburl.com/code/z009xsy6
        q_weight = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(weight)
        res_weight = q_weight[:, :-8].view(torch.uint8)
        res_scale_shift = torch.tensor(
            q_weight[:, -8:].view(torch.float32).to(torch.float16).view(torch.uint8)
        )  # [-4, -2]: scale; [-2:]: bias
        return (res_weight, res_scale_shift)

    elif weight_ty == SparseType.INT4 or weight_ty == SparseType.INT2:
        # Note that FP32 -> INT4/INT2 conersion op below might have additional padding
        # for alignment: https://fburl.com/code/xx9kkduf
        q_weight = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
            weight,
            bit_rate=weight_ty.bit_rate(),
        )
        res_weight = q_weight[:, :-4].view(torch.uint8)
        res_scale_shift = torch.tensor(
            q_weight[:, -4:].view(torch.uint8)
        )  # [-4, -2]: scale; [-2:]: bias
        return (res_weight, res_scale_shift)

    else:
        raise RuntimeError("Unsupported SparseType: {}".format(weight_ty))


def dequantize_embs(
    weights: torch.Tensor,
    scale_shift: torch.Tensor,
    weight_ty: SparseType,
    use_cpu: bool,
    fp8_config: Optional[FP8QuantizationConfig] = None,
) -> torch.Tensor:
    print(f"weight_ty: {weight_ty}")
    assert (
        weights.dtype == torch.uint8
    ), "The input tensor for dequantize_embs function needs to be byte tensor"
    th_weights = weights

    if scale_shift is not None:
        th_scale_shift: torch.Tensor = scale_shift.view(torch.float16).to(torch.float32)

    if weight_ty == SparseType.INT4:
        (E, D_2) = th_weights.shape
        D = D_2 * 2

        def comp(i: int) -> torch.Tensor:
            subs = th_weights.view(torch.uint8) >> (i * 4)
            sub_mask = subs & 0xF
            result = sub_mask.to(torch.float32) * th_scale_shift[:, 0].reshape(
                -1, 1
            ).to(torch.float32) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
            return result.to(torch.float32)

        comps = [comp(i) for i in range(2)]
        comps = torch.stack(comps)
        comps = comps.permute(1, 2, 0)
        comps = comps.reshape(E, D)
        return to_device(torch.tensor(comps), use_cpu)

    elif weight_ty == SparseType.INT2:
        (E, D_4) = th_weights.shape
        D = D_4 * 4

        # pyre-fixme[53]: Captured variable `scale_shift` is not annotated.
        # pyre-fixme[53]: Captured variable `weights` is not annotated.
        def comp(i: int) -> torch.Tensor:
            subs = th_weights.view(torch.uint8) >> (i * 2)
            sub_mask = subs & 0x3
            result = sub_mask.to(torch.float32) * th_scale_shift[:, 0].reshape(
                -1, 1
            ).to(torch.float32) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
            return result.to(torch.float32)

        comps = [comp(i) for i in range(4)]
        comps = torch.stack(comps)
        comps = comps.permute(1, 2, 0)
        comps = comps.reshape(E, D)
        return to_device(torch.tensor(comps), use_cpu)

    elif weight_ty == SparseType.INT8:
        (E, D) = th_weights.shape
        comps = th_weights.to(torch.float32) * th_scale_shift[:, 0].reshape(-1, 1).to(
            torch.float32
        ) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
        return to_device(torch.tensor(comps), use_cpu)

    elif weight_ty == SparseType.FP8:
        assert fp8_config is not None
        assert scale_shift is None
        # Dequantize HPF8 to FP32
        comps = torch.ops.fbgemm.HFP8QuantizedToFloat(
            weights,
            fp8_config.get("exponent_bits"),
            fp8_config.get("exponent_bias"),
        )
        return to_device(comps, use_cpu)

    elif weight_ty == SparseType.FP16:
        assert scale_shift is None
        comps = th_weights.view(torch.half)
        return to_device(torch.tensor(comps), use_cpu)

    elif weight_ty == SparseType.FP32:
        assert scale_shift is None
        comps = th_weights.view(torch.float32)
        # pyre-fixme[7]: Expected `Tensor` but got implicit return value of `None`.
        return to_device(torch.tensor(comps), use_cpu)


def fake_quantize_embs(
    weights: torch.Tensor,
    scale_shift: Optional[torch.Tensor],
    dequant_weights: torch.Tensor,
    weight_ty: SparseType,
    use_cpu: bool,
    fp8_config: Optional[FP8QuantizationConfig] = None,
) -> None:
    assert (
        weights.dtype == torch.uint8
    ), "The input tensor for dequantize_embs function needs to be byte tensor"
    th_weights = weights

    if scale_shift is not None:
        th_scale_shift: torch.Tensor = (
            scale_shift.contiguous().view(torch.float16).to(torch.float32)
        )

    if weight_ty == SparseType.INT4:
        (E, D_2) = th_weights.shape
        D = D_2 * 2

        def comp(i: int) -> torch.Tensor:
            subs = th_weights.view(torch.uint8) >> (i * 4)
            sub_mask = subs & 0xF
            result = sub_mask.to(torch.float32) * th_scale_shift[:, 0].reshape(
                -1, 1
            ).to(torch.float32) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
            return result.to(torch.float32)

        comps = [comp(i) for i in range(2)]
        comps = torch.stack(comps)
        comps = comps.permute(1, 2, 0)
        comps = comps.reshape(E, D)
        dequant_weights.copy_(to_device(comps, use_cpu))

    elif weight_ty == SparseType.INT2:
        (E, D_4) = th_weights.shape
        D = D_4 * 4

        # pyre-fixme[53]: Captured variable `scale_shift` is not annotated.
        # pyre-fixme[53]: Captured variable `weights` is not annotated.
        def comp(i: int) -> torch.Tensor:
            subs = th_weights.view(torch.uint8) >> (i * 2)
            sub_mask = subs & 0x3
            result = sub_mask.to(torch.float32) * th_scale_shift[:, 0].reshape(
                -1, 1
            ).to(torch.float32) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
            return result.to(torch.float32)

        comps = [comp(i) for i in range(4)]
        comps = torch.stack(comps)
        comps = comps.permute(1, 2, 0)
        comps = comps.reshape(E, D)
        dequant_weights.copy_(to_device(comps, use_cpu))

    elif weight_ty == SparseType.INT8:
        (E, D) = th_weights.shape
        comps = th_weights.to(torch.float32) * th_scale_shift[:, 0].reshape(-1, 1).to(
            torch.float32
        ) + th_scale_shift[:, 1].reshape(-1, 1).to(torch.float32)
        dequant_weights.copy_(to_device(comps, use_cpu))

    elif weight_ty == SparseType.FP8:
        assert fp8_config is not None
        assert scale_shift is None
        # Quantize FP32 to HPF8
        comps = torch.ops.fbgemm.FloatToHFP8Quantized(
            dequant_weights.detach().float(),
            fp8_config.get("exponent_bits"),
            fp8_config.get("exponent_bias"),
            fp8_config.get("max_position"),
        )
        weights.copy_(comps)

        # Dequantize HPF8 to FP32
        comps = torch.ops.fbgemm.HFP8QuantizedToFloat(
            comps,
            fp8_config.get("exponent_bits"),
            fp8_config.get("exponent_bias"),
        )
        dequant_weights.copy_(to_device(comps, use_cpu))

    elif weight_ty == SparseType.FP16:
        assert scale_shift is None
        comps = dequant_weights.detach().half().view(torch.uint8)
        weights.copy_(comps)
    elif weight_ty == SparseType.FP32:
        assert scale_shift is None
        comps = dequant_weights.detach().float().view(torch.uint8)
        weights.copy_(comps)

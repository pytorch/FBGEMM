# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from dataclasses import dataclass
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


@dataclass
class TBERequest:
    """
    `generate_requests`'s output wrapper
    """

    indices: torch.Tensor
    offsets: torch.Tensor
    per_sample_weights: Optional[torch.Tensor] = None
    Bs_per_feature_per_rank: Optional[List[List[int]]] = None

    def unpack_2(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.indices, self.offsets)

    def unpack_3(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return (self.indices, self.offsets, self.per_sample_weights)

    def unpack_4(
        self,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[List[List[int]]]
    ]:
        return (
            self.indices,
            self.offsets,
            self.per_sample_weights,
            self.Bs_per_feature_per_rank,
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


def generate_requests_from_data_file(
    requests_data_file: str,
    iters: int,
    B: int,
    T: int,
    L: int,
    E: int,
    weighted: bool,
    tables: Optional[str] = None,
) -> List[TBERequest]:
    """
    Generate TBE requests from the input data file (`requests_data_file`)
    """
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
            indices = torch.cat((indices, indices_tensor[t_offsets[0] : t_offsets[-1]]))
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
            TBERequest(
                indices_tensor.to(get_device()),
                offsets_tensor.to(get_device()),
                weights_tensor,
            )
        )
    return rs


def generate_int_data_from_stats(
    mu: int,
    sigma: int,
    size: int,
    distribution: str,
) -> np.ndarray:
    """
    Generate integer data based on stats
    """
    if distribution == "uniform":
        # TODO: either make these separate parameters or make a separate version of
        # generate_requests to handle the uniform dist case once whole
        # generate_requests function is refactored to split into helper functions
        # for each use case.
        # mu represents the lower bound when the uniform distribution is used
        lower_bound = mu
        # sigma represetns the upper bound when the uniform distribution is used
        upper_bound = sigma + 1
        return np.random.randint(
            lower_bound,
            upper_bound,
            (size,),
            dtype=np.int32,
        )
    else:  # normal dist
        return np.random.normal(loc=mu, scale=sigma, size=size).astype(int)


def generate_pooling_factors_from_stats(
    iters: int,
    Bs: List[int],
    L: int,
    sigma_L: int,
    # distribution of pooling factors
    length_dist: str,
) -> Tuple[int, torch.Tensor]:
    """
    Generate pooling factors for the TBE requests from the given stats
    """
    Ls_list = []
    for B in Bs:
        Ls_list.append(generate_int_data_from_stats(L, sigma_L, B, length_dist))

    # Concat all Ls
    Ls = np.concatenate(Ls_list)

    # Make sure that Ls are positive
    Ls[Ls < 0] = 0
    # Use the same L distribution across iters
    Ls = np.tile(Ls, iters)
    L = Ls.max()
    # Make it exclusive cumsum
    L_offsets = torch.from_numpy(np.insert(Ls.cumsum(), 0, 0)).to(torch.long)
    return L, L_offsets


def generate_batch_sizes_from_stats(
    B: int,
    T: int,
    sigma_B: int,
    vbe_num_ranks: int,
    # Distribution of batch sizes
    batch_size_dist: str,
) -> Tuple[List[int], List[List[int]]]:
    """
    Generate batch sizes for features from the given stats
    """
    # Generate batch size per feature per rank
    Bs_feature_rank = generate_int_data_from_stats(
        B, sigma_B, T * vbe_num_ranks, batch_size_dist
    )

    # Make sure that Bs are at least one
    Bs_feature_rank = np.absolute(Bs_feature_rank)
    Bs_feature_rank[Bs_feature_rank == 0] = 1

    # Convert numpy array to Torch tensor
    Bs_feature_rank = torch.from_numpy(Bs_feature_rank).view(T, vbe_num_ranks)
    # Compute batch sizes per feature
    Bs = Bs_feature_rank.sum(1).tolist()

    return Bs, Bs_feature_rank.tolist()


def generate_indices_uniform(
    iters: int,
    Bs: List[int],
    L: int,
    E: int,
    use_variable_L: bool,
    L_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Generate indices for the TBE requests using the uniform distribution
    """
    total_B = sum(Bs)
    indices = torch.randint(
        low=0,
        high=E,
        size=(iters, total_B, L),
        device="cpu" if use_variable_L else get_device(),
        dtype=torch.int32,
    )
    # each bag is usually sorted
    (indices, _) = torch.sort(indices)
    if use_variable_L:
        indices = torch.ops.fbgemm.bottom_k_per_row(
            indices.to(torch.long), L_offsets, False
        )
        indices = indices.to(get_device()).int()
    else:
        indices = indices.reshape(iters, total_B * L)
    return indices


def generate_indices_zipf(
    iters: int,
    Bs: List[int],
    L: int,
    E: int,
    alpha: float,
    zipf_oversample_ratio: int,
    use_variable_L: bool,
    L_offsets: torch.Tensor,
    deterministic_output: bool,
) -> torch.Tensor:
    """
    Generate indices for the TBE requests using the zipf distribution
    """
    assert E >= L, "num-embeddings must be greater than equal to bag-size"
    # oversample and then remove duplicates to obtain sampling without
    # replacement
    if L == 0:
        return torch.empty(iters, 0, dtype=torch.int).to(get_device())
    total_B = sum(Bs)
    zipf_shape = (iters, total_B, zipf_oversample_ratio * L)
    if torch.cuda.is_available():
        zipf_shape_total_len = np.prod(zipf_shape)
        indices_list = []
        # process 8 GB at a time on GPU
        chunk_len = int(1e9)
        for chunk_begin in range(0, zipf_shape_total_len, chunk_len):
            indices_gpu = torch.ops.fbgemm.zipf_cuda(
                alpha,
                min(zipf_shape_total_len - chunk_begin, chunk_len),
                seed=torch.randint(2**31 - 1, (1,))[0],
            )
            indices_list.append(indices_gpu.cpu())
        indices = torch.cat(indices_list).reshape(zipf_shape)
    else:
        indices = torch.as_tensor(np.random.zipf(a=alpha, size=zipf_shape))
    indices = (indices - 1) % E
    if use_variable_L:
        indices = torch.ops.fbgemm.bottom_k_per_row(indices, L_offsets, True)
    else:
        indices = torch.ops.fbgemm.bottom_k_per_row(
            indices, torch.tensor([0, L], dtype=torch.long), True
        )
    if deterministic_output:
        rng = default_rng(12345)
    else:
        rng = default_rng()
    permutation = torch.as_tensor(
        rng.choice(E, size=indices.max().item() + 1, replace=False)
    )
    indices = permutation.gather(0, indices.flatten())
    indices = indices.to(get_device()).int()
    if not use_variable_L:
        indices = indices.reshape(iters, total_B * L)
    return indices


def update_indices_with_random_reuse(
    iters: int,
    Bs: List[int],
    L: int,
    reuse: float,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Update the generated indices with random reuse
    """
    for it in range(iters - 1):
        B_offset = 0
        for B in Bs:
            reused_indices = torch.randperm(B * L, device=get_device())[
                : int(B * L * reuse)
            ]
            reused_indices += B_offset
            indices[it + 1, reused_indices] = indices[it, reused_indices]
            B_offset += B * L
    return indices


def update_indices_with_random_pruning(
    iters: int,
    B: int,
    T: int,
    L: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Update the generated indices with random pruning
    """
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
            indices[it, t, random_locations] = -1
    return indices


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
    # If sigma_B is not None, treat B as mu_B and generate Bs from sigma_B
    sigma_B: Optional[int] = None,
    emulate_pruning: bool = False,
    use_cpu: bool = False,
    # generate_requests uses numpy.random.default_rng without a set random seed
    # be default, causing the indices tensor to vary with each call to
    # generate_requests - set generate_repeatable_output to use a fixed random
    # seed instead for repeatable outputs
    deterministic_output: bool = False,
    # distribution of embedding sequence lengths
    length_dist: str = "normal",
    # distribution of batch sizes
    batch_size_dist: str = "normal",
    # Number of ranks for variable batch size generation
    vbe_num_ranks: Optional[int] = None,
) -> List[TBERequest]:
    # TODO: refactor and split into helper functions to separate load from file,
    # generate from distribution, and other future methods of generating data
    if requests_data_file is not None:
        assert sigma_L is None, "Variable pooling factors is not supported"
        assert sigma_B is None, "Variable batch sizes is not supported"
        return generate_requests_from_data_file(
            requests_data_file,
            iters,
            B,
            T,
            L,
            E,
            weighted,
            tables,
        )

    if sigma_B is not None:
        assert (
            vbe_num_ranks is not None
        ), "vbe_num_ranks must be set for varaible batch size generation"
        use_variable_B = True
        Bs, Bs_feature_rank = generate_batch_sizes_from_stats(
            B, T, sigma_B, vbe_num_ranks, batch_size_dist
        )
    else:
        use_variable_B = False
        Bs = [B] * T
        Bs_feature_rank = None

    if sigma_L is not None:
        # Generate L from stats
        use_variable_L = True
        L, L_offsets = generate_pooling_factors_from_stats(
            iters, Bs, L, sigma_L, length_dist
        )
    elif use_variable_B:
        use_variable_L = False
        Ls = [L] * (sum(Bs) * iters)
        L_offsets = torch.tensor([0] + Ls, dtype=torch.long).cumsum(0)
    else:
        use_variable_L = False
        # Init to suppress the pyre error
        L_offsets = torch.empty(1)

    if alpha <= 1.0:
        # Generate indices using uniform dist
        all_indices = generate_indices_uniform(
            iters, Bs, L, E, use_variable_L, L_offsets
        )
    else:
        # Generate indices using zipf dist
        all_indices = generate_indices_zipf(
            iters,
            Bs,
            L,
            E,
            alpha,
            zipf_oversample_ratio,
            use_variable_L,
            L_offsets,
            deterministic_output,
        )

    if reuse > 0.0:
        assert (
            not use_variable_L
        ), "Does not support generating Ls from stats for reuse > 0.0"
        all_indices = update_indices_with_random_reuse(iters, Bs, L, reuse, all_indices)

    # Some indices are set to -1 for emulating pruned rows.
    if emulate_pruning:
        assert (
            not use_variable_L
        ), "Does not support generating Ls from stats for emulate_pruning=True"
        assert (
            not use_variable_B
        ), "Does not support generating Bs from stats for emulate_pruning=True"

        all_indices = update_indices_with_random_pruning(
            iters, B, T, L, all_indices.view(iters, T, B * L)
        )

    # Pack requests
    rs = []
    if use_variable_L or use_variable_B:
        total_B = sum(Bs)
        all_indices = all_indices.flatten()
        for it in range(iters):
            start_offset = L_offsets[it * total_B]
            it_L_offsets = torch.concat(
                [
                    torch.zeros(1, dtype=L_offsets.dtype, device=L_offsets.device),
                    L_offsets[it * total_B + 1 : (it + 1) * total_B + 1] - start_offset,
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
                TBERequest(
                    all_indices[start_offset : L_offsets[(it + 1) * total_B]],
                    it_L_offsets.to(get_device()),
                    weights_tensor,
                    Bs_feature_rank if use_variable_B else None,
                )
            )
    else:
        for it in range(iters):
            weights_tensor = (
                None
                if not weighted
                else torch.randn(
                    T * B * L, device=get_device()
                )  # per sample weights will always be FP32
            )
            indices, offsets = get_table_batched_offsets_from_dense(
                all_indices[it].view(T, B, L), use_cpu=use_cpu
            )
            rs.append(TBERequest(indices, offsets, weights_tensor))
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

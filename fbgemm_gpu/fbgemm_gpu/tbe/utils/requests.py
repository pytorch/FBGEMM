# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

# pyre-fixme[21]: Could not find name `default_rng` in `numpy.random` (stubbed).
from numpy.random import default_rng

from .common import get_device
from .offsets import get_table_batched_offsets_from_dense

logging.basicConfig(level=logging.DEBUG)


@dataclass
class TBERequest:
    """
    `generate_requests`'s output wrapper
    """

    indices: torch.Tensor
    offsets: torch.Tensor
    per_sample_weights: torch.Tensor | None = None
    Bs_per_feature_per_rank: list[list[int]] | None = None

    def unpack_2(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.indices, self.offsets)

    def unpack_3(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return (self.indices, self.offsets, self.per_sample_weights)

    def unpack_4(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[list[int]] | None]:
        return (
            self.indices,
            self.offsets,
            self.per_sample_weights,
            self.Bs_per_feature_per_rank,
        )


def generate_requests_from_data_file(
    iters: int,
    B: int,
    T: int,
    L: int,
    E: int,
    weighted: bool,
    device: torch.device,
    requests_data_file: str | None = None,
    indices_file: str | None = None,
    offsets_file: str | None = None,
    tables: str | None = None,
    index_dtype: torch.dtype | None = None,
    offset_dtype: torch.dtype | None = None,
) -> list[TBERequest]:
    """
    Generate TBE requests from the input data file. If `requests_data_file` is provided,
    `indices_file` and `offsets_file` should not be provided. If either `indices_file`
    or `offsets_file` is provided, both must be provided.
    """
    assert not (
        requests_data_file and (indices_file or offsets_file)
    ), "If requests_data_file is provided, indices_file and offsets_file cannot be provided."

    if requests_data_file:
        indices_tensor, offsets_tensor, *rest = torch.load(requests_data_file)
    else:
        assert (
            indices_file and offsets_file
        ), "Both indices_file and offsets_file must be provided if either is provided."
        indices_tensor = torch.load(indices_file)
        offsets_tensor = torch.load(offsets_file)

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
            f"offsets = {offsets_tensor.size()}, lengths = {offsets_tensor.size() - 1}) "
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
        None if not weighted else torch.randn(indices_tensor.size(), device=device)
    )
    rs = []
    for _ in range(iters):
        rs.append(
            TBERequest(
                maybe_to_dtype(indices_tensor.to(device), index_dtype),
                maybe_to_dtype(offsets_tensor.to(device), offset_dtype),
                weights_tensor,
            )
        )
    return rs


def generate_int_data_from_stats(
    mu: int,
    sigma: int,
    size: int,
    distribution: str,
) -> npt.NDArray:
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
    Bs: list[int],
    L: int,
    sigma_L: int,
    # distribution of pooling factors
    length_dist: str,
) -> tuple[int, torch.Tensor]:
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
) -> tuple[list[int], list[list[int]]]:
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
    Bs: list[int],
    L: int,
    E: int,
    use_variable_L: bool,
    L_offsets: torch.Tensor,
    device: torch.device | None = None,
    Es: list[int] | None = None,
) -> torch.Tensor:
    """
    Generate indices for the TBE requests using the uniform distribution.
    If Es is provided, generates indices per-table so that each table t's
    indices are in [0, Es[t]).
    """
    if device is None:
        device = get_device()
    T = len(Bs)
    total_B = sum(Bs)
    dev = "cpu" if use_variable_L else device

    if Es is not None and len(set(Es)) > 1:
        assert len(Es) == T, f"len(Es)={len(Es)} must equal T={T}"
        if use_variable_L:
            # Generate per-table indices with each table's E range and
            # per-table max bag size, then flatten.
            all_table_indices = []
            for t in range(T):
                table_row_start = sum(Bs[:t])
                table_lengths = (
                    L_offsets[table_row_start + 1 : table_row_start + Bs[t] + 1]
                    - L_offsets[table_row_start : table_row_start + Bs[t]]
                )
                L_t = int(table_lengths.max().item())
                if L_t == 0:
                    all_table_indices.append(
                        torch.empty(iters, 0, dtype=torch.int).to(dev)
                    )
                    continue
                tbl_indices = torch.randint(
                    low=0,
                    high=Es[t],
                    size=(iters, Bs[t], L_t),
                    device=dev,
                    dtype=torch.int32,
                )
                tbl_indices, _ = torch.sort(tbl_indices)
                tbl_indices = tbl_indices.reshape(iters, Bs[t] * L_t)
                all_table_indices.append(tbl_indices)
            return torch.cat(all_table_indices, dim=1).flatten().to(device)
        table_indices = []
        for t in range(T):
            table_indices.append(
                torch.randint(
                    low=0,
                    high=Es[t],
                    size=(iters, Bs[t], L),
                    device=dev,
                    dtype=torch.int32,
                )
            )
        indices = torch.cat(table_indices, dim=1)
    else:
        indices = torch.randint(
            low=0,
            high=E,
            size=(iters, total_B, L),
            device=dev,
            dtype=torch.int32,
        )
    # each bag is usually sorted
    indices, _ = torch.sort(indices)
    if use_variable_L:
        # 1D layout, where row offsets are determined by L_offsets
        indices = torch.ops.fbgemm.bottom_k_per_row(
            indices.to(torch.long), L_offsets, False
        )
        indices = indices.to(device).int()
    else:
        # 2D layout
        indices = indices.reshape(iters, total_B * L)
    return indices


def _generate_zipf_indices_single_table(
    iters: int,
    B: int,
    L: int,
    E: int,
    alpha: float,
    zipf_oversample_ratio: int,
    deterministic_output: bool,
) -> torch.Tensor:
    """
    Core zipf generation for a single table (or batched tables with same E).
    Returns indices of shape (iters, B * L).
    """
    if L == 0:
        return torch.empty(iters, 0, dtype=torch.int).to(get_device())

    # For small L, the default oversample ratio may not produce enough unique
    # values.  Ensure at least 20 candidates so that even highly concentrated
    # Zipf draws are unlikely to all collide across many rows.
    effective_oversample_L = max(zipf_oversample_ratio * L, 20)
    zipf_shape = (iters, B, effective_oversample_L)

    if torch.cuda.is_available():
        zipf_shape_total_len = np.prod(zipf_shape)
        indices_list = []
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
    indices = indices.reshape(iters, B * L)

    return indices.to(get_device()).int()


def _generate_zipf_single_table_with_fallback(
    iters: int,
    B: int,
    L: int,
    E: int,
    alpha: float,
    zipf_oversample_ratio: int,
    deterministic_output: bool,
    device: torch.device,
    table_idx: int,
) -> torch.Tensor:
    """Generate Zipf indices for one table, falling back to uniform on skew error."""
    try:
        return _generate_zipf_indices_single_table(
            iters=iters,
            B=B,
            L=L,
            E=E,
            alpha=alpha,
            zipf_oversample_ratio=zipf_oversample_ratio,
            deterministic_output=deterministic_output,
        )
    except RuntimeError as e:
        if "too skewed distribution" not in str(e):
            raise
        logging.warning(
            f"Zipf index generation failed for table {table_idx} "
            f"(E={E}, L={L}, alpha={alpha}): "
            f"distribution too skewed. "
            f"Falling back to uniform distribution for this table."
        )
        indices_t = torch.randint(
            low=0,
            high=E,
            size=(iters, B, L),
            device="cpu",
            dtype=torch.int32,
        )
        indices_t, _ = torch.sort(indices_t)
        return indices_t.reshape(iters, B * L).to(device).int()


def _generate_zipf_per_table(
    iters: int,
    Bs: list[int],
    L: int,
    Es: list[int],
    T: int,
    alpha: float,
    zipf_oversample_ratio: int,
    use_variable_L: bool,
    L_offsets: torch.Tensor,
    deterministic_output: bool,
    device: torch.device,
) -> torch.Tensor:
    """Per-table Zipf generation: generate separately for each table, then concat."""
    if use_variable_L:
        all_table_indices = []
        for t in range(T):
            table_row_start = sum(Bs[:t])
            table_lengths = (
                L_offsets[table_row_start + 1 : table_row_start + Bs[t] + 1]
                - L_offsets[table_row_start : table_row_start + Bs[t]]
            )
            L_t = int(table_lengths.max().item())
            if L_t == 0:
                all_table_indices.append(
                    torch.empty(iters, 0, dtype=torch.int).to(device)
                )
                continue
            indices_t = _generate_zipf_single_table_with_fallback(
                iters,
                Bs[t],
                L_t,
                Es[t],
                alpha,
                zipf_oversample_ratio,
                deterministic_output,
                device,
                t,
            )
            all_table_indices.append(indices_t)
        return torch.cat(all_table_indices, dim=1).flatten()
    else:
        all_table_indices = []
        for t in range(T):
            indices_t = _generate_zipf_single_table_with_fallback(
                iters,
                Bs[t],
                L,
                Es[t],
                alpha,
                zipf_oversample_ratio,
                deterministic_output,
                device,
                t,
            )
            all_table_indices.append(indices_t)
        return torch.cat(all_table_indices, dim=1)


def _generate_zipf_batched(
    iters: int,
    Bs: list[int],
    L: int,
    E: int,
    alpha: float,
    zipf_oversample_ratio: int,
    use_variable_L: bool,
    L_offsets: torch.Tensor,
    deterministic_output: bool,
    device: torch.device,
) -> torch.Tensor:
    """Single-E batched Zipf generation for all tables."""
    if L == 0:
        return torch.empty(iters, 0, dtype=torch.int).to(device)
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
    try:
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
        if not use_variable_L:
            indices = indices.reshape(iters, total_B * L)

        indices = indices.to(device).int()
        return indices
    except RuntimeError as e:
        if "too skewed distribution" not in str(e):
            raise
        logging.warning(
            f"Zipf index generation failed for batched tables "
            f"(E={E}, L={L}, alpha={alpha}): "
            f"distribution too skewed. "
            f"Falling back to uniform distribution for all tables."
        )
        return generate_indices_uniform(
            iters, Bs, L, E, use_variable_L, L_offsets, device
        )


def generate_indices_zipf(
    iters: int,
    Bs: list[int],
    L: int,
    E: int,
    alpha: float,
    zipf_oversample_ratio: int,
    use_variable_L: bool,
    L_offsets: torch.Tensor,
    deterministic_output: bool,
    device: torch.device | None = None,
    Es: list[int] | None = None,
) -> torch.Tensor:
    """
    Generate indices for the TBE requests using the zipf distribution.

    If Es is provided with variable values, generates true Zipf distribution
    per table (each table t's indices follow Zipf in [0, Es[t])).
    Otherwise, uses batched generation with single E for all tables.
    """
    T = len(Bs)
    assert E >= L, "num-embeddings must be greater than equal to bag-size"

    if device is None:
        device = get_device()
    # Per-table Zipf: generate separately for each table, then concatenate
    if Es is not None and len(set(Es)) > 1:
        assert len(Es) == T, f"len(Es)={len(Es)} must equal T={T}"
        return _generate_zipf_per_table(
            iters,
            Bs,
            L,
            Es,
            T,
            alpha,
            zipf_oversample_ratio,
            use_variable_L,
            L_offsets,
            deterministic_output,
            device,
        )
    # Single E case: batched generation
    return _generate_zipf_batched(
        iters,
        Bs,
        L,
        E,
        alpha,
        zipf_oversample_ratio,
        use_variable_L,
        L_offsets,
        deterministic_output,
        device,
    )


def update_indices_with_random_reuse(
    iters: int,
    Bs: list[int],
    L: int,
    reuse: float,
    indices: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Update the generated indices with random reuse
    """
    if device is None:
        device = get_device()
    for it in range(iters - 1):
        B_offset = 0
        for B in Bs:
            reused_indices = torch.randperm(B * L, device=device)[: int(B * L * reuse)]
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


def maybe_to_dtype(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    return tensor if dtype is None else tensor.to(dtype)


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
    requests_data_file: str | None = None,
    # Path to file containing indices and offsets. If provided, this will be used
    indices_file: str | None = None,
    offsets_file: str | None = None,
    # Comma-separated list of table numbers
    tables: str | None = None,
    # If sigma_L is not None, treat L as mu_L and generate Ls from sigma_L
    # and mu_L
    sigma_L: int | None = None,
    # If Ls is not None, use these per-table bag sizes directly instead of
    # generating from sigma_L. Must have len(Ls) == T.
    Ls: list[int] | None = None,
    # If sigma_B is not None, treat B as mu_B and generate Bs from sigma_B
    sigma_B: int | None = None,
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
    vbe_num_ranks: int | None = None,
    index_dtype: torch.dtype | None = None,
    offset_dtype: torch.dtype | None = None,
    # Per-table num_embeddings. If provided, indices for table t are generated
    # in [0, Es[t]). Must have len(Es) == T.
    Es: list[int] | None = None,
) -> list[TBERequest]:
    # TODO: refactor and split into helper functions to separate load from file,
    # generate from distribution, and other future methods of generating data
    device = torch.device("cpu") if use_cpu else get_device()
    if (
        requests_data_file is not None
        or indices_file is not None
        or offsets_file is not None
    ):

        assert sigma_L is None, "Variable pooling factors is not supported"
        assert sigma_B is None, "Variable batch sizes is not supported"
        return generate_requests_from_data_file(
            iters=iters,
            B=B,
            T=T,
            L=L,
            E=E,
            weighted=weighted,
            device=device,
            requests_data_file=requests_data_file,
            indices_file=indices_file,
            offsets_file=offsets_file,
            tables=tables,
            index_dtype=index_dtype,
            offset_dtype=offset_dtype,
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

    if Ls is not None:
        # Use per-table bag sizes directly
        assert sigma_L is None, "Cannot specify both Ls and sigma_L"
        assert len(Ls) == T, f"len(Ls)={len(Ls)} must equal T={T}"
        use_variable_L = True
        # Build per-row lengths matching generate_pooling_factors_from_stats layout
        Ls_list = []
        for t in range(T):
            Ls_list.append(np.array([Ls[t]] * Bs[t], dtype=np.int32))
        Ls_np = np.concatenate(Ls_list)
        # Use the same L distribution across iters
        Ls_np = np.tile(Ls_np, iters)
        L = int(Ls_np.max())
        # Make it exclusive cumsum
        L_offsets = torch.from_numpy(np.insert(Ls_np.cumsum(), 0, 0)).to(torch.long)
    elif sigma_L is not None:
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
            iters, Bs, L, E, use_variable_L, L_offsets, device, Es=Es
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
            device=device,
            Es=Es,
        )

    if reuse > 0.0:
        assert (
            not use_variable_L
        ), "Does not support generating Ls from stats for reuse > 0.0"
        all_indices = update_indices_with_random_reuse(
            iters, Bs, L, reuse, all_indices, device
        )

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
                    int(it_L_offsets[-1].item()), device=device
                )  # per sample weights will always be FP32
            )
            rs.append(
                TBERequest(
                    maybe_to_dtype(
                        all_indices[start_offset : L_offsets[(it + 1) * total_B]],
                        index_dtype,
                    ),
                    maybe_to_dtype(it_L_offsets.to(device), offset_dtype),
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
                    T * B * L, device=device
                )  # per sample weights will always be FP32
            )
            indices, offsets = get_table_batched_offsets_from_dense(
                all_indices[it].view(T, B, L), use_cpu=use_cpu
            )
            rs.append(
                TBERequest(
                    maybe_to_dtype(indices, index_dtype),
                    maybe_to_dtype(offsets, offset_dtype),
                    weights_tensor,
                )
            )
    return rs

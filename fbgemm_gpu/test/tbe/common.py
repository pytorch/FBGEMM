#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional

import fbgemm_gpu
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    CacheAlgorithm,
    invokers,
    is_torchdynamo_compiling,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import settings, Verbosity

# pyre-fixme[16]: Module `fbgemm_gpu` has no attribute `open_source`.
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import (
        gpu_unavailable,
        running_in_oss,
        running_on_github,
        TEST_WITH_ROCM,
    )
else:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:cumem_utils")
    from fbgemm_gpu.test.test_utils import (  # noqa F401
        gpu_unavailable,
        running_in_oss,
        running_on_github,
        TEST_WITH_ROCM,
    )


torch.ops.import_module("fbgemm_gpu.sparse_ops")
settings.register_profile("derandomize", derandomize=True)
settings.load_profile("derandomize")


MAX_EXAMPLES = 40

# For long running tests reduce the number of iterations to reduce timeout errors.
MAX_EXAMPLES_LONG_RUNNING = 15

FORWARD_MAX_THREADS = 512

VERBOSITY: Verbosity = Verbosity.verbose


def gen_mixed_B_batch_sizes(B: int, T: int) -> tuple[list[list[int]], list[int]]:
    num_ranks = np.random.randint(low=1, high=4)
    low = max(int(0.25 * B), 1)
    high = int(B)
    if low == high:
        Bs_rank_feature = [[B] * num_ranks for _ in range(T)]
    else:
        Bs_rank_feature = [
            np.random.randint(low=low, high=high, size=num_ranks).tolist()
            for _ in range(T)
        ]
    Bs = [sum(Bs_feature) for Bs_feature in Bs_rank_feature]
    return Bs_rank_feature, Bs


def format_ref_tensors_in_mixed_B_layout(
    ref_tensors: list[torch.Tensor], Bs_rank_feature: list[list[int]]
) -> torch.Tensor:
    # Relayout the reference tensor
    # Jagged dimension: (rank, table, local batch)
    num_ranks = len(Bs_rank_feature[0])
    split_tensors = [[] for _ in range(num_ranks)]  # shape (rank, table)
    for t, ref_tensor in enumerate(ref_tensors):
        assert ref_tensor.shape[0] == sum(Bs_rank_feature[t])
        tensors = ref_tensor.split(Bs_rank_feature[t])
        for r, tensor in enumerate(tensors):
            split_tensors[r].append(tensor.flatten())
    concat_list = []
    for r in range(num_ranks):
        concat_list += split_tensors[r]
    return torch.cat(concat_list, dim=0)


def assert_torch_equal(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> None:
    assert torch.equal(tensor_a, tensor_b)


def get_max_thread_blocks(stream: torch.cuda.streams.Stream) -> int:
    # Based on the empirical studies, having a max grid size that is 64x larger than
    # the number of SMs gives good performance across the board
    MAX_THREAD_BLOCKS_FACTOR = 64
    device = stream.device_index
    return (
        MAX_THREAD_BLOCKS_FACTOR
        * torch.cuda.get_device_properties(device).multi_processor_count
    )


def recompute_offsets_vbe_cpu(
    original_offsets: torch.Tensor,
    T: int,
    max_B: int,
    B_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Recompute VBE offsets for CPU, since the CPU kernel does non-VBE lookup.
    We pad the offsets such that the output is in non-VBE format, i.e., [max_B, total_D]

    Args:
        original_offsets (Tensor): original offsets passed to forward
        T (int): number of tables
        max_B (int): max batch size
        B_offsets (Tensor): Batch offsets

    Returns:
        offsets: recomputed VBE offsets
    """
    # create offsets with fixed batch size max_B
    # not efficient but for now we just need a functional implementation for CPU
    max_B = max_B
    offsets = torch.empty(
        [T * max_B + 1], dtype=original_offsets.dtype, device=original_offsets.device
    )
    for t in range(T):
        assert isinstance(B_offsets, torch.Tensor)
        begin = B_offsets[t]
        end = B_offsets[t + 1]
        offsets[t * max_B : t * max_B + end - begin] = original_offsets[begin:end]
        offsets[t * max_B + end - begin : (t + 1) * max_B] = original_offsets[end]
    offsets[-1] = original_offsets[-1]
    return offsets


def recompute_output_vbe_cpu(
    output: torch.Tensor,
    T: int,
    output_size: int,
    output_offsets_feature_rank: torch.Tensor,
    B_offsets_rank_per_feature: torch.Tensor,
    D_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Recompute VBE output, since the CPU kernel does non-VBE lookup.
    This transform the output from non-VBE format ([max_B, total_D]) to VBE format, i.e., [1, output_size]

    Args:
        output (Tensor): original output returned from the kernel
        T (int): number of tables
        output_size (int): output size
        output_offsets_feature_rank (Tensor): output offsets, per feature per rank
        B_offsets_rank_per_feature (Tensor): Batch offsets per rank per feature
        D_offsets (Tensor): embedding dim offsets

    Returns:
        output_new: recomputed VBE output
    """
    output_new = torch.empty([output_size], dtype=output.dtype, device=output.device)
    assert isinstance(B_offsets_rank_per_feature, torch.Tensor)
    output_offsets_feature_rank = output_offsets_feature_rank
    assert isinstance(output_offsets_feature_rank, torch.Tensor)
    R = B_offsets_rank_per_feature.size(1) - 1
    for r in range(R):
        D_offset = 0
        for t in range(T):
            o_begin = output_offsets_feature_rank[r * T + t].item()
            o_end = output_offsets_feature_rank[r * T + t + 1].item()
            D = D_offsets[t + 1].item() - int(D_offsets[t].item())
            b_begin = B_offsets_rank_per_feature[t][r].item()
            b_end = B_offsets_rank_per_feature[t][r + 1].item()
            assert o_end - o_begin == (b_end - b_begin) * D  # pyre-ignore[6]
            output_new[o_begin:o_end] = output[
                b_begin:b_end, D_offset : D_offset + D
            ].flatten()
            D_offset += D
    return output_new


def invoke_v1_cpu(
    optimizer: OptimType,
    common_kwargs: Dict[str, Any],
    optim_kwargs: Dict[str, Any],
    vbe_metadata: invokers.lookup_args.VBEMetadata,
) -> torch.Tensor:
    """
    This function is used to invoke TBE V1 lookup function for the given optimizer on CPU.
    """
    T = common_kwargs["D_offsets"].numel() - 1
    vbe: bool = vbe_metadata.B_offsets is not None
    common_kwargs["offsets"] = (
        recompute_offsets_vbe_cpu(
            common_kwargs["offsets"],
            T,
            vbe_metadata.max_B,
            vbe_metadata.B_offsets,  # pyre-ignore[6]
        )
        if vbe
        else common_kwargs["offsets"]
    )
    if optimizer == OptimType.EXACT_ADAGRAD:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_adagrad_function_cpu(
            # optimizer_args
            learning_rate=optim_kwargs["learning_rate"],
            eps=optim_kwargs["eps"],
            momentum1_host=optim_kwargs["momentum1_host"],
            momentum1_offsets=optim_kwargs["momentum1_offsets"],
            momentum1_placements=optim_kwargs["momentum1_placements"],
            **common_kwargs,
        )
    elif optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function_cpu(
            # optimizer-specific args
            learning_rate=optim_kwargs["learning_rate"],
            eps=optim_kwargs["eps"],
            momentum1_host=optim_kwargs["momentum1_host"],
            momentum1_offsets=optim_kwargs["momentum1_offsets"],
            momentum1_placements=optim_kwargs["momentum1_placements"],
            weight_decay=optim_kwargs["weight_decay"],
            weight_decay_mode=optim_kwargs["weight_decay_mode"],
            max_norm=optim_kwargs["max_norm"],
            **common_kwargs,
        )
    elif optimizer == OptimType.EXACT_SGD:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_sgd_function_cpu(
            # optimizer-specific args
            learning_rate=optim_kwargs["learning_rate"],
            **common_kwargs,
        )
    else:
        raise AssertionError(f"{optimizer} is not supported for CPU")
    output = (
        recompute_output_vbe_cpu(
            output,
            T,
            vbe_metadata.output_size,
            vbe_metadata.output_offsets_feature_rank,  # pyre-ignore[6]
            vbe_metadata.B_offsets_rank_per_feature,  # pyre-ignore[6]
            common_kwargs["offsets"],
        )
        if vbe
        else output
    )
    return output


def invoke_v1_cuda(
    optimizer: OptimType,
    common_kwargs: Dict[str, Any],
    optim_kwargs: Dict[str, Any],
    vbe_metadata: invokers.lookup_args.VBEMetadata,
) -> torch.Tensor:
    """
    This function is used to invoke TBE V1 lookup function for the given optimizer on CUDA.
    """

    if optimizer == OptimType.EXACT_ADAGRAD:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_adagrad_function(
            momentum1_dev=optim_kwargs["momentum1_dev"],
            momentum1_uvm=optim_kwargs["momentum1_uvm"],
            momentum1_offsets=optim_kwargs["momentum1_offsets"],
            momentum1_placements=optim_kwargs["momentum1_placements"],
            eps=optim_kwargs["eps"],
            learning_rate=optim_kwargs["learning_rate"],
            **common_kwargs,
        )
    elif (
        optimizer == OptimType.EXACT_ROWWISE_ADAGRAD
        and optim_kwargs["use_rowwise_adagrad_with_counter"]
    ):
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_with_counter_function(
            momentum1_dev=optim_kwargs["momentum1_dev"],
            momentum1_uvm=optim_kwargs["momentum1_uvm"],
            momentum1_offsets=optim_kwargs["momentum1_offsets"],
            momentum1_placements=optim_kwargs["momentum1_placements"],
            prev_iter_uvm=optim_kwargs["prev_iter_uvm"],
            prev_iter_offsets=optim_kwargs["prev_iter_offsets"],
            prev_iter_placements=optim_kwargs["prev_iter_placements"],
            row_counter_dev=optim_kwargs["row_counter_dev"],
            row_counter_uvm=optim_kwargs["row_counter_uvm"],
            row_counter_offsets=optim_kwargs["row_counter_offsets"],
            row_counter_placements=optim_kwargs["row_counter_placements"],
            eps=optim_kwargs["eps"],
            learning_rate=optim_kwargs["learning_rate"],
            weight_decay=optim_kwargs["weight_decay"],
            counter_halflife=optim_kwargs["counter_halflife"],
            adjustment_iter=optim_kwargs["adjustment_iter"],
            adjustment_ub=optim_kwargs["adjustment_ub"],
            learning_rate_mode=optim_kwargs["learning_rate_mode"],
            weight_decay_mode=optim_kwargs["weight_decay_mode"],
            grad_sum_decay=optim_kwargs["grad_sum_decay"],
            max_counter=optim_kwargs["max_counter"],
            tail_id_threshold=optim_kwargs["tail_id_threshold"],
            is_tail_id_thresh_ratio=optim_kwargs["is_tail_id_thresh_ratio"],
            regularization_mode=optim_kwargs["regularization_mode"],
            weight_norm_coefficient=optim_kwargs["weight_norm_coefficient"],
            lower_bound=optim_kwargs["lower_bound"],
            **common_kwargs,
        )
    elif optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
        output = (
            torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function(
                momentum1_dev=optim_kwargs["momentum1_dev"],
                momentum1_uvm=optim_kwargs["momentum1_uvm"],
                momentum1_offsets=optim_kwargs["momentum1_offsets"],
                momentum1_placements=optim_kwargs["momentum1_placements"],
                eps=optim_kwargs["eps"],
                learning_rate=optim_kwargs["learning_rate"],
                weight_decay=optim_kwargs["weight_decay"],
                weight_decay_mode=optim_kwargs["weight_decay_mode"],
                max_norm=optim_kwargs["max_norm"],
                **common_kwargs,
            )
        )
    elif optimizer == OptimType.EXACT_SGD:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_sgd_function(
            learning_rate=optim_kwargs["learning_rate"], **common_kwargs
        )
    elif optimizer == OptimType.ADAM:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_adam_function(
            momentum1_dev=optim_kwargs["momentum1_dev"],
            momentum1_uvm=optim_kwargs["momentum1_uvm"],
            momentum1_offsets=optim_kwargs["momentum1_offsets"],
            momentum1_placements=optim_kwargs["momentum1_placements"],
            momentum2_dev=optim_kwargs["momentum2_dev"],
            momentum2_uvm=optim_kwargs["momentum2_uvm"],
            momentum2_offsets=optim_kwargs["momentum2_offsets"],
            momentum2_placements=optim_kwargs["momentum2_placements"],
            eps=optim_kwargs["eps"],
            learning_rate=optim_kwargs["learning_rate"],
            beta1=optim_kwargs["beta1"],
            beta2=optim_kwargs["beta2"],
            weight_decay=optim_kwargs["weight_decay"],
            **common_kwargs,
        )
    elif optimizer == OptimType.PARTIAL_ROWWISE_ADAM:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_partial_rowwise_adam_function(
            momentum1_dev=optim_kwargs["momentum1_dev"],
            momentum1_uvm=optim_kwargs["momentum1_uvm"],
            momentum1_offsets=optim_kwargs["momentum1_offsets"],
            momentum1_placements=optim_kwargs["momentum1_placements"],
            momentum2_dev=optim_kwargs["momentum2_dev"],
            momentum2_uvm=optim_kwargs["momentum2_uvm"],
            momentum2_offsets=optim_kwargs["momentum2_offsets"],
            momentum2_placements=optim_kwargs["momentum2_placements"],
            eps=optim_kwargs["eps"],
            learning_rate=optim_kwargs["learning_rate"],
            beta1=optim_kwargs["beta1"],
            beta2=optim_kwargs["beta2"],
            weight_decay=optim_kwargs["weight_decay"],
            **common_kwargs,
        )
    elif optimizer == OptimType.LAMB:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_lamb_function(
            momentum1_dev=optim_kwargs["momentum1_dev"],
            momentum1_uvm=optim_kwargs["momentum1_uvm"],
            momentum1_offsets=optim_kwargs["momentum1_offsets"],
            momentum1_placements=optim_kwargs["momentum1_placements"],
            momentum2_dev=optim_kwargs["momentum2_dev"],
            momentum2_uvm=optim_kwargs["momentum2_uvm"],
            momentum2_offsets=optim_kwargs["momentum2_offsets"],
            momentum2_placements=optim_kwargs["momentum2_placements"],
            eps=optim_kwargs["eps"],
            learning_rate=optim_kwargs["learning_rate"],
            beta1=optim_kwargs["beta1"],
            beta2=optim_kwargs["beta2"],
            weight_decay=optim_kwargs["weight_decay"],
            **common_kwargs,
        )
    elif optimizer == OptimType.PARTIAL_ROWWISE_LAMB:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_partial_rowwise_lamb_function(
            momentum1_dev=optim_kwargs["momentum1_dev"],
            momentum1_uvm=optim_kwargs["momentum1_uvm"],
            momentum1_offsets=optim_kwargs["momentum1_offsets"],
            momentum1_placements=optim_kwargs["momentum1_placements"],
            momentum2_dev=optim_kwargs["momentum2_dev"],
            momentum2_uvm=optim_kwargs["momentum2_uvm"],
            momentum2_offsets=optim_kwargs["momentum2_offsets"],
            momentum2_placements=optim_kwargs["momentum2_placements"],
            eps=optim_kwargs["eps"],
            learning_rate=optim_kwargs["learning_rate"],
            beta1=optim_kwargs["beta1"],
            beta2=optim_kwargs["beta2"],
            weight_decay=optim_kwargs["weight_decay"],
            **common_kwargs,
        )
    elif optimizer == OptimType.LARS_SGD:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_lars_sgd_function(
            momentum1_dev=optim_kwargs["momentum1_dev"],
            momentum1_uvm=optim_kwargs["momentum1_uvm"],
            momentum1_offsets=optim_kwargs["momentum1_offsets"],
            momentum1_placements=optim_kwargs["momentum1_placements"],
            learning_rate=optim_kwargs["learning_rate"],
            eta=optim_kwargs["eta"],
            momentum=optim_kwargs["momentum"],
            weight_decay=optim_kwargs["weight_decay"],
            **common_kwargs,
        )
    elif optimizer == OptimType.NONE:
        output = torch.ops.fbgemm.split_embedding_codegen_lookup_none_function(
            total_hash_size=optim_kwargs["total_hash_size"],
            total_unique_indices=optim_kwargs["total_unique_indices"],
            **common_kwargs,
        )
    else:
        raise AssertionError(f"{optimizer} is not supported for CUDA")
    return output


def v1_lookup(
    emb_op: SplitTableBatchedEmbeddingBagsCodegen,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    use_cpu: bool,
    per_sample_weights: Optional[torch.Tensor] = None,
    batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    feature_requires_grad: Optional[torch.Tensor] = None,
    total_unique_indices: Optional[int] = None,
) -> torch.Tensor:
    """
    This function prepares inputs and invokes TBE V1 lookup function for the given optimizer on CPU or CUDA.

    Args:
        emb_op (SplitTableBatchedEmbeddingBagsCodegen): initialized TBE operator
        indices (torch.Tensor): indices tensor
        offsets (torch.Tensor): offsets tensor
        use_cpu (bool): whether to use CPU or CUDA
        per_sample_weights (Optional[torch.Tensor]): per sample weights tensor
        batch_size_per_feature_per_rank (Optional[List[List[int]]]): batch size per feature per rank
        feature_requires_grad (Optional[torch.Tensor]): feature requires grad tensor
        total_unique_indices (Optional[int]): total unique indices, this is for NONE optimizer

    Returns:
        output tensor (torch.Tensor)
    """
    optimizer = emb_op.optimizer
    (
        indices,
        offsets,
        per_sample_weights,
        vbe_metadata,
    ) = emb_op.prepare_inputs(
        indices,
        offsets,
        per_sample_weights,
        batch_size_per_feature_per_rank,
        force_cast_input_types=True,
        prefetch_pipeline=False,
    )

    if len(emb_op.timesteps_prefetched) == 0:
        # In forward, we don't enable multi-pass prefetch as we want the process
        # to be as fast as possible and memory usage doesn't matter (will be recycled
        # by dense fwd/bwd)
        # TODO: Properly pass in the hash_zch_identities
        emb_op._prefetch(
            indices,
            offsets,
            vbe_metadata,
            multipass_prefetch_config=None,
            hash_zch_identities=None,
        )

    if len(emb_op.timesteps_prefetched) > 0:
        emb_op.timesteps_prefetched.pop(0)

    emb_op.lxu_cache_locations = (
        emb_op.lxu_cache_locations_empty
        if len(emb_op.lxu_cache_locations_list) == 0
        else emb_op.lxu_cache_locations_list.pop(0)
    )

    iter = 1
    if optimizer != OptimType.NONE:
        emb_op.iter_cpu = emb_op.iter_cpu.cpu()

        # Sync with loaded state
        if (
            not is_torchdynamo_compiling()
        ):  # wrap to make it compatible with PT2 compile
            if emb_op.iter_cpu.item() == 0:
                emb_op.iter_cpu.fill_(emb_op.iter.cpu().item())

        # Increment the iteration counter
        iter_int = int(emb_op.iter_cpu.add_(1).item())  # used for local computation
        emb_op.iter.add_(1)  # used for checkpointing
        iter = int(emb_op.iter_cpu.item())

        if emb_op._used_rowwise_adagrad_with_counter:
            if (
                emb_op._max_counter_update_freq > 0
                and iter_int % emb_op._max_counter_update_freq == 0
            ):
                row_counter_dev = emb_op.row_counter_dev.detach()
                if row_counter_dev.numel() > 0:
                    emb_op.max_counter[0] = torch.max(row_counter_dev).cpu().item() + 1
                else:
                    emb_op.max_counter[0] = 1

    common_kwargs: Dict[str, Any] = {
        "weights_placements": emb_op.weights_placements,
        "weights_offsets": emb_op.weights_offsets,
        "D_offsets": emb_op.D_offsets,
        "total_D": emb_op.total_D,
        "max_D": emb_op.max_D,
        "hash_size_cumsum": emb_op.hash_size_cumsum,
        "total_hash_size_bits": emb_op.total_hash_size_bits,
        "indices": indices,
        "offsets": offsets,
        "pooling_mode": emb_op.pooling_mode,
        "indice_weights": per_sample_weights,
        "feature_requires_grad": feature_requires_grad,
    }
    if optimizer != OptimType.NONE:
        common_kwargs.update(
            {
                # common optimizer_args
                "gradient_clipping": emb_op.optimizer_args.gradient_clipping,
                "max_gradient": emb_op.optimizer_args.max_gradient,
                "stochastic_rounding": emb_op.optimizer_args.stochastic_rounding,
            }
        )
    if use_cpu:
        common_kwargs.update(
            {
                "host_weights": emb_op.weights_host,
            }
        )
    else:
        common_kwargs.update(
            {
                "placeholder_autograd_tensor": emb_op.placeholder_autograd_tensor,
                "dev_weights": emb_op.weights_dev,
                "uvm_weights": emb_op.weights_uvm,
                "lxu_cache_weights": emb_op.lxu_cache_weights,
                "lxu_cache_locations": emb_op.lxu_cache_locations,
                "output_dtype": emb_op.output_dtype,
                # VBE metadata
                "B_offsets": vbe_metadata.B_offsets,
                "vbe_output_offsets_feature_rank": vbe_metadata.output_offsets_feature_rank,
                "vbe_B_offsets_rank_per_feature": vbe_metadata.B_offsets_rank_per_feature,
                "max_B": vbe_metadata.max_B,
                "max_B_feature_rank": vbe_metadata.max_B_feature_rank,
                "vbe_output_size": vbe_metadata.output_size,
                "is_experimental": emb_op.is_experimental,
                "use_uniq_cache_locations_bwd": emb_op.use_uniq_cache_locations_bwd,
                "use_homogeneous_placements": emb_op.use_homogeneous_placements,
                "uvm_cache_stats": (
                    emb_op.local_uvm_cache_stats
                    if (
                        emb_op.gather_uvm_cache_stats
                        # Unique conflict misses are only collected when using CacheAlgorithm.LRU
                        and emb_op.cache_algorithm == CacheAlgorithm.LRU
                    )
                    else None
                ),
                "prev_iter_dev": (
                    emb_op.prev_iter_dev if optimizer != OptimType.NONE else None
                ),
                "apply_global_weight_decay": False,
                "gwd_lower_bound": 0.0,
                "iter": iter,
            }
        )

    if optimizer != OptimType.NONE:
        optim_kwargs: Dict[str, Any] = {
            "momentum1_dev": emb_op.momentum1_dev,
            "momentum1_host": emb_op.momentum1_host,
            "momentum1_uvm": emb_op.momentum1_uvm,
            "momentum1_offsets": emb_op.momentum1_offsets,
            "momentum1_placements": emb_op.momentum1_placements,
            "momentum2_dev": emb_op.momentum2_dev,
            "momentum2_host": emb_op.momentum2_host,
            "momentum2_uvm": emb_op.momentum2_uvm,
            "momentum2_offsets": emb_op.momentum2_offsets,
            "momentum2_placements": emb_op.momentum2_placements,
            "prev_iter_host": emb_op.prev_iter_host,
            "prev_iter_uvm": emb_op.prev_iter_uvm,
            "prev_iter_offsets": emb_op.prev_iter_offsets,
            "prev_iter_placements": emb_op.prev_iter_placements,
            "row_counter_dev": emb_op.row_counter_dev,
            "row_counter_host": emb_op.row_counter_host,
            "row_counter_uvm": emb_op.row_counter_uvm,
            "row_counter_offsets": emb_op.row_counter_offsets,
            "row_counter_placements": emb_op.row_counter_placements,
            "learning_rate": emb_op.learning_rate_tensor.item(),
            "max_counter": emb_op.max_counter.item(),
            "use_rowwise_adagrad_with_counter": emb_op._used_rowwise_adagrad_with_counter,
            # optimizer_args fields
            "eps": emb_op.optimizer_args.eps,
            "weight_decay": emb_op.optimizer_args.weight_decay,
            "weight_decay_mode": emb_op.optimizer_args.weight_decay_mode,
            "max_norm": emb_op.optimizer_args.max_norm,
            "counter_halflife": emb_op.optimizer_args.counter_halflife,
            "adjustment_iter": emb_op.optimizer_args.adjustment_iter,
            "adjustment_ub": emb_op.optimizer_args.adjustment_ub,
            "learning_rate_mode": emb_op.optimizer_args.learning_rate_mode,
            "grad_sum_decay": emb_op.optimizer_args.grad_sum_decay,
            "tail_id_threshold": emb_op.optimizer_args.tail_id_threshold,
            "is_tail_id_thresh_ratio": emb_op.optimizer_args.is_tail_id_thresh_ratio,
            "regularization_mode": emb_op.optimizer_args.regularization_mode,
            "weight_norm_coefficient": emb_op.optimizer_args.weight_norm_coefficient,
            "lower_bound": emb_op.optimizer_args.lower_bound,
            "beta1": emb_op.optimizer_args.beta1,
            "beta2": emb_op.optimizer_args.beta2,
            "eta": emb_op.optimizer_args.eta,
            "momentum": emb_op.optimizer_args.momentum,
        }
    else:
        optim_kwargs = {
            "total_hash_size": emb_op.optimizer_args.total_hash_size,
            "total_unique_indices": total_unique_indices,
        }

    invoker = invoke_v1_cpu if use_cpu else invoke_v1_cuda
    return invoker(
        optimizer,
        common_kwargs,
        optim_kwargs,
        vbe_metadata,
    )

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Optional

import torch

torch.ops.load_library(
    "//deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai:trt_llm_ops_gpu"
)


def fused_one_shot_allreduce_residual_rms_norm_init(
    device: torch.device,
    rank: int,
    world_size: int,
    max_num_seqs: int,
    hidden_size: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """
    Initialize the state for fused one-shot allreduce, residual, RMS norm from
    TensorRT-LLM

    Args:
        device (torch.device): The GPU device that this module is running
            on

        rank (int): The communication rank of this GPU

        world_size (int): The total number of GPUs participate in this
            communication

        max_num_seqs (int): Max number of sequences

        hidden_size (int): The hidden size (the normalized shape in RMS
            norm)

        dist_group (Optional[ProcessGroup]): An optional ProcessGroup for
            `torch.distributed`
    """
    assert torch.distributed.is_initialized(), "torch.distributed must be initialized"
    assert world_size <= 8, f"world_size ({world_size}) must be <= 8"
    assert (
        rank >= 0 and rank < world_size
    ), f"rank ({rank}) must be >= 0 and < world_size"

    # Allocate IPC buffers
    (
        buffer,
        buffer_offsets,
        buffer_handle,
    ) = torch.ops.fbgemm._fused_one_shot_allreduce_residual_rms_norm_allocate_buffers(
        device,
        world_size,
        max_num_seqs=max_num_seqs,
        hidden_size=hidden_size,
    )

    # Exchange IPC handles
    all_buffer_handles = [torch.empty_like(buffer_handle) for _ in range(world_size)]
    torch.distributed.all_gather(all_buffer_handles, buffer_handle, group=dist_group)

    # Ensure that the handle exchange is done
    torch.cuda.synchronize()
    torch.distributed.barrier(group=dist_group)

    # Get IPC buffers from handles
    torch.ops.fbgemm._fused_one_shot_allreduce_residual_rms_norm_init(
        rank,
        world_size,
        hidden_size,
        buffer,
        all_buffer_handles,
        buffer_offsets,
    )

    # Ensure that the buffer initialization is done
    torch.cuda.synchronize()
    torch.distributed.barrier(group=dist_group)


def fused_one_shot_allreduce_residual_rms_norm(
    tensor: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
    affine: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Invoke TensorRT-LLM fused one-shot allreduce residual RMS norm

    The equivalent operation is `norm(allreduce(tensor) + residual, affine)`

    Args:
        tensor (Tensor): An input tensor for allreduce

        residual (Tensor): A residual tensor

        eps (float): A value added to the denominator for numerical
            stability of RMS norm

        affine (Tensor): A row-wise affine tensor

    Returns:
        A tensor with the same size as the input tensor
    """
    return torch.ops.fbgemm._fused_one_shot_allreduce_residual_rms_norm(
        tensor,
        residual,
        eps=eps,
        affine=affine,
    )

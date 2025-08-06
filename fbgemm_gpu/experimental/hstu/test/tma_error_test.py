# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fbgemm_gpu.experimental.hstu import cuda_hstu_attn_varlen


def main() -> None:
    # testing parameters
    batch_size = 4
    max_seq_len = 512
    max_targets = 20
    heads = 4
    attn_dim = 128
    hidden_dim = 128
    dtype = torch.float16
    alpha = 1.0

    lengths_qk = torch.randint(
        1,
        max_seq_len + 1,
        size=(batch_size,),
        device=torch.accelerator.current_accelerator(True),
        dtype=torch.int32,
    )
    seq_offsets_qk = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths_qk)
    num_targets = torch.randint(
        0,
        max_targets + 1,
        size=(batch_size,),
        device=torch.accelerator.current_accelerator(),
        dtype=torch.int32,
    )
    seq_offsets_t = torch.ops.fbgemm.asynchronous_complete_cumsum(num_targets)
    # Lengths for whole q, kv
    seq_offsets_wt = seq_offsets_qk + seq_offsets_t

    L_qk = int(seq_offsets_wt[-1].item())
    q = (
        torch.empty(
            size=(L_qk, heads, attn_dim),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(True),
        )
        .uniform_(-1, 1)
        .requires_grad_()
    )
    k = (
        torch.empty(
            size=(L_qk, heads, attn_dim),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(True),
        )
        .uniform_(-1, 1)
        .requires_grad_()
    )
    v = (
        torch.empty(
            size=(L_qk, heads, hidden_dim),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(True),
        )
        .uniform_(-1, 1)
        .requires_grad_()
    )

    flash_out = cuda_hstu_attn_varlen(  # noqa E731
        q=q,
        k=k,
        v=v,
        seq_offsets_q=seq_offsets_wt,
        seq_offsets_k=seq_offsets_wt,
        max_seqlen_q=max_seq_len + max_targets,
        max_seqlen_k=max_seq_len + max_targets,
        num_targets=num_targets,
        window_size=(-1, 0),
        alpha=alpha,
        is_train=False,
    )


if __name__ == "__main__":
    main()

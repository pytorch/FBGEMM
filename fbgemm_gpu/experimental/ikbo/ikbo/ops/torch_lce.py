# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# PyTorch reference implementations for Linear Compression Embedding (LCE).
#
# Standard LCE: output[b] = W @ E[b]
# Decomposed:   output[b] = W_cand @ E_cand[b] + W_user @ E_user[u(b)]
#
# The decomposition exploits the many-to-one mapping u(b) from candidates
# to users: W_user @ E_user is computed once per unique user and broadcast,
# reducing redundant work proportional to the candidate-to-user ratio.

import torch


def torch_lce(compression_w: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """Baseline LCE: output[b] = W @ E[b].

    Args:
        compression_w: compression weights [M, K_user + K_cand].
        embeddings: All embeddings [B, K_user + K_cand, N],
            with user embeddings pre-broadcast to match each candidate.
    """
    return compression_w @ embeddings


def torch_decomposed_lce(
    compression_w_cand: torch.Tensor,
    compression_w_user: torch.Tensor,
    embeddings_cand: torch.Tensor,
    embeddings_user: torch.Tensor,
    cand_to_user_index: torch.Tensor,
) -> torch.Tensor:
    """Decomposed LCE: output[b] = W_cand @ E_cand[b] + W_user @ E_user[u(b)].

    Avoids redundant computation by factoring the user contribution:
    W_user @ E_user is computed once per unique user (num_users < B),
    then broadcast to each user's candidates via cand_to_user_index.

    Args:
        compression_w_cand: Candidate compression weights [M, K_cand].
        compression_w_user: User compression weights [M, K_user].
        embeddings_cand: Candidate embeddings [B, K_cand, N].
        embeddings_user: User embeddings [num_users, K_user, N].
        cand_to_user_index: Maps candidate index to user index [B], int32.
    """
    cand_res = compression_w_cand @ embeddings_cand
    user_res = compression_w_user @ embeddings_user
    return cand_res + torch.index_select(user_res, dim=0, index=cand_to_user_index)

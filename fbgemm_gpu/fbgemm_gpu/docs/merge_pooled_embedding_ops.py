# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .common import add_docs

add_docs(
    torch.ops.fbgemm.merge_pooled_embeddings,
    """
merge_pooled_embeddings(pooled_embeddings, uncat_dim_size, target_device, cat_dim=1) -> Tensor

Concatenate embedding outputs from different devices (on the same host)
on to the target device.

Args:
    pooled_embeddings (List[Tensor]): A list of embedding outputs from
        different devices on the same host. Each output has 2
        dimensions.

    uncat_dim_size (int): The size of the dimension that is not
        concatenated, i.e., if `cat_dim=0`, `uncat_dim_size` is the size
        of dim 1 and vice versa.

    target_device (torch.device): The target device that aggregates all
        the embedding outputs.

    cat_dim (int = 1): The dimension that the tensors are concatenated

Returns:
    The concatenated embedding output (2D) on the target device
    """,
)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:group_gemm_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:group_gemm_ops")


class GroupLinear(torch.nn.Module):
    def __init__(
        self, linear_config: List[Tuple[int, int]], bias: bool = False
    ) -> None:
        super().__init__()
        # Use torch.nn.linear as a wrapper for weight and bias
        # TODO: Use torch.nn.parameter.Parameter instead
        self.gmm = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    in_features,
                    out_features,
                    bias=bias,
                )
                for in_features, out_features in linear_config
            ]
        )
        self.num_groups: int = len(linear_config)
        self.bias = bias

    # Backward is not supported currently
    def forward(self, input_group: List[torch.Tensor]) -> List[torch.Tensor]:
        weights = [self.gmm[i].weight for i in range(self.num_groups)]
        if self.bias:
            biases = [self.gmm[i].bias for i in range(self.num_groups)]
        else:
            biases = None
        output = torch.ops.fbgemm.group_linear_forward(input_group, weights, biases)
        return output

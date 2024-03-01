# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable

import torch


class BatchAuc(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        n_tasks: int,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        _, sorted_indices = torch.sort(predictions, descending=True, dim=-1)
        sorted_labels = torch.gather(labels, 1, sorted_indices)
        sorted_weights = torch.gather(weights, 1, sorted_indices)
        cum_fp = torch.cumsum(sorted_weights * (1.0 - sorted_labels), dim=-1)
        cum_tp = torch.cumsum(sorted_weights * sorted_labels, dim=-1)
        fac = cum_fp[:, -1] * cum_tp[:, -1]
        auc = torch.where(fac == 0, 0.5, torch.trapz(cum_tp, cum_fp, dim=-1) / fac)
        return auc


class Auc(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        n_tasks: int,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        _, sorted_indices = torch.sort(predictions, descending=True, dim=-1)
        aucs = []
        for sorted_indices_i, labels_i, weights_i in zip(
            sorted_indices, labels, weights
        ):
            sorted_labels = torch.index_select(labels_i, dim=0, index=sorted_indices_i)
            sorted_weights = torch.index_select(
                weights_i, dim=0, index=sorted_indices_i
            )
            cum_fp = torch.cumsum(sorted_weights * (1.0 - sorted_labels), dim=0)
            cum_tp = torch.cumsum(sorted_weights * sorted_labels, dim=0)
            auc = torch.where(
                cum_fp[-1] * cum_tp[-1] == 0,
                0.5,  # 0.5 is the no-signal default value for auc.
                torch.trapz(cum_tp, cum_fp) / cum_fp[-1] / cum_tp[-1],
            )
            aucs.append(auc.view(1))
        return torch.cat(aucs)


class AucJiterator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Jiterator only works with elementwise kernels
        fp_code_string = """
        template <typename T> T fp(T weights, T labels) {
            return weights * (1.0 - labels);
        }"""

        tp_code_string = """
        template <typename T> T tp(T weights, T labels) {
            return weights * labels;
        }"""

        # pyre-ignore [4]
        self.jitted_fp: Callable[..., Any] = torch.cuda.jiterator._create_jit_fn(
            fp_code_string
        )
        # pyre-ignore [4]
        self.jitted_tp: Callable[..., Any] = torch.cuda.jiterator._create_jit_fn(
            tp_code_string
        )

    def forward(
        self,
        n_tasks: int,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        _, sorted_indices = torch.sort(predictions, descending=True, dim=-1)
        aucs = []
        for sorted_indices_i, labels_i, weights_i in zip(
            sorted_indices, labels, weights
        ):
            sorted_labels = torch.index_select(labels_i, dim=0, index=sorted_indices_i)
            sorted_weights = torch.index_select(
                weights_i, dim=0, index=sorted_indices_i
            )
            cum_fp = torch.cumsum(self.jitted_fp(sorted_weights, sorted_labels), dim=0)
            cum_tp = torch.cumsum(self.jitted_tp(sorted_weights, sorted_labels), dim=0)
            auc = torch.where(
                cum_fp[-1] * cum_tp[-1] == 0,
                0.5,  # 0.5 is the no-signal default value for auc.
                torch.trapz(cum_tp, cum_fp) / cum_fp[-1] / cum_tp[-1],
            )
            aucs.append(auc.view(1))
        return torch.cat(aucs)


class BatchAucJiterator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Jiterator only works with elementwise kernels
        fp_code_string = """
        template <typename T> T fp(T weights, T labels) {
            return weights * (1.0 - labels);
        }"""

        tp_code_string = """
        template <typename T> T tp(T weights, T labels) {
            return weights * labels;
        }"""

        # pyre-ignore [4]
        self.jitted_fp: Callable[..., Any] = torch.cuda.jiterator._create_jit_fn(
            fp_code_string
        )
        # pyre-ignore [4]
        self.jitted_tp: Callable[..., Any] = torch.cuda.jiterator._create_jit_fn(
            tp_code_string
        )

    def forward(
        self,
        n_tasks: int,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        _, sorted_indices = torch.sort(predictions, descending=True, dim=-1)
        sorted_labels = torch.gather(labels, 1, sorted_indices)
        sorted_weights = torch.gather(weights, 1, sorted_indices)
        cum_fp = torch.cumsum(self.jitted_fp(sorted_weights, sorted_labels), dim=-1)
        cum_tp = torch.cumsum(self.jitted_tp(sorted_weights, sorted_labels), dim=-1)
        fac = cum_fp[:, -1] * cum_tp[:, -1]
        auc = torch.where(fac == 0, 0.5, torch.trapz(cum_tp, cum_fp, dim=-1) / fac)
        return auc


def auc(
    n_tasks: int, predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    _, sorted_indices = torch.sort(predictions, descending=True, dim=-1)
    return torch.ops.fbgemm.batch_auc(n_tasks, sorted_indices, labels, weights)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import math
from typing import cast, Optional, Tuple

import torch

from fbgemm_gpu.split_embedding_configs import QuantizationConfig, SparseType
from fbgemm_gpu.split_embedding_utils import FP8QuantizationConfig, quantize_embs
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch import Tensor  # usort:skip


# TODO: add per-feature based converter option (based on embedding_specs during inference)
# TODO: optimize embedding pruning and quantization latency.
class SplitEmbInferenceConverter:
    def __init__(
        self,
        quantize_type: SparseType,
        pruning_ratio: Optional[float],
        use_array_for_index_remapping: bool = True,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        self.quantize_type = quantize_type
        # TODO(yingz): Change the pruning ratio to per-table settings.
        self.pruning_ratio = pruning_ratio
        self.use_array_for_index_remapping = use_array_for_index_remapping
        self.quantization_config = quantization_config

    def convert_model(self, model: torch.nn.Module) -> torch.nn.Module:
        self._process_split_embs(model)
        return model

    def _prune_by_weights_l2_norm(self, new_num_rows, weights) -> Tuple[Tensor, float]:
        assert new_num_rows > 0
        from numpy.linalg import norm

        indicators = []
        for row in weights:
            indicators.append(norm(row.cpu().numpy(), ord=2))
        sorted_indicators = sorted(indicators, reverse=True)
        threshold = None
        for i in range(new_num_rows, len(sorted_indicators)):
            if sorted_indicators[i] < sorted_indicators[new_num_rows - 1]:
                threshold = sorted_indicators[i]
                break
        if threshold is None:
            threshold = sorted_indicators[-1] - 1
        return (torch.tensor(indicators), threshold)

    def _prune_embs(
        self,
        idx: int,
        num_rows: int,
        module: SplitTableBatchedEmbeddingBagsCodegen,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # TODO(yingz): Avoid DtoH / HtoD overhead.
        weights = module.split_embedding_weights()[idx].cpu()
        if self.pruning_ratio is None:
            return (weights, None)
        new_num_rows = int(math.ceil(num_rows * (1.0 - self.pruning_ratio)))  # type: ignore
        if new_num_rows == num_rows:
            return (weights, None)

        (indicators, threshold) = self._prune_by_weights_l2_norm(new_num_rows, weights)

        return torch.ops.fbgemm.embedding_bag_rowwise_prune(
            weights, indicators, threshold, torch.int32
        )

    def _get_quantization_config(self, name):
        quantization_config = self.quantization_config
        if quantization_config is None:
            raise RuntimeError("quantization_config must be set for FP8 weight")
        return quantization_config.get(name)

    def _quantize_embs(
        self, weight: Tensor, weight_ty: SparseType
    ) -> Tuple[Tensor, Optional[Tensor]]:
        fp8_quant_config = cast(FP8QuantizationConfig, self.quantization_config)
        return quantize_embs(weight, weight_ty, fp8_quant_config)

    def _process_split_embs(self, model: torch.nn.Module) -> None:
        for name, child in model.named_children():
            if isinstance(
                child,
                SplitTableBatchedEmbeddingBagsCodegen,
            ):
                embedding_specs = []
                use_cpu = child.embedding_specs[0][3] == ComputeDevice.CPU
                for E, D, _, _ in child.embedding_specs:
                    weights_ty = self.quantize_type
                    if D % weights_ty.align_size() != 0:
                        logging.warning(
                            f"Embedding dim {D} couldn't be divided by align size {weights_ty.align_size()}!"
                        )
                        assert D % 4 == 0
                        weights_ty = (
                            SparseType.FP16
                        )  # fall back to FP16 if dimension couldn't be aligned with the required size
                    embedding_specs.append(("", E, D, weights_ty))

                weight_lists = []
                new_embedding_specs = []
                index_remapping_list = []
                for t, (_, E, D, weight_ty) in enumerate(embedding_specs):
                    # Try to prune embeddings.
                    (pruned_weight, index_remapping) = self._prune_embs(t, E, child)
                    new_embedding_specs.append(
                        (
                            "",
                            pruned_weight.size()[0],
                            D,
                            weight_ty,
                            EmbeddingLocation.HOST
                            if use_cpu
                            else EmbeddingLocation.DEVICE,
                        )
                    )
                    index_remapping_list.append(index_remapping)

                    # Try to quantize embeddings.
                    weight_lists.append(self._quantize_embs(pruned_weight, weight_ty))

                is_fp8_weight = self.quantize_type == SparseType.FP8

                q_child = IntNBitTableBatchedEmbeddingBagsCodegen(
                    embedding_specs=new_embedding_specs,
                    index_remapping=index_remapping_list
                    if self.pruning_ratio is not None
                    else None,
                    pooling_mode=child.pooling_mode,
                    device="cpu" if use_cpu else torch.cuda.current_device(),
                    weight_lists=weight_lists,
                    use_array_for_index_remapping=self.use_array_for_index_remapping,
                    fp8_exponent_bits=self._get_quantization_config("exponent_bits")
                    if is_fp8_weight
                    else None,
                    fp8_exponent_bias=self._get_quantization_config("exponent_bias")
                    if is_fp8_weight
                    else None,
                )
                setattr(model, name, q_child)
            else:
                self._process_split_embs(child)

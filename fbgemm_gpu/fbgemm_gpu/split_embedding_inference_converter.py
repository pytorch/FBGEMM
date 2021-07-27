#!/usr/bin/env python3

# pyre-unsafe

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import fbgemm_gpu.split_table_batched_embeddings_ops as split_table_batched_embeddings_ops
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from torch import nn


# TODO: add per-feature based converter option (based on embedding_specs during inference)
class SplitEmbInferenceConverter:
    def __init__(self, quantize_type: SparseType):
        self.quantize_type = quantize_type

    def convert_model(self, model: torch.nn.Module) -> nn.Module:
        self._quantize_split_embs(model)
        return model

    def _quantize_split_embs(self, model: nn.Module) -> None:
        for name, child in model.named_children():
            if isinstance(
                child,
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen,
            ):
                embedding_specs = []
                use_cpu = (
                    child.embedding_specs[0][3]
                    == split_table_batched_embeddings_ops.ComputeDevice.CPU
                )
                for (E, D, _, _) in child.embedding_specs:
                    weights_ty = self.quantize_type
                    if D % weights_ty.align_size() != 0:
                        logging.warn(
                            f"Embedding dim {D} couldn't be divided by align size {weights_ty.align_size()}!"
                        )
                        assert D % 4 == 0
                        weights_ty = (
                            SparseType.FP16
                        )  # fall back to FP16 if dimension couldn't be aligned with the required size
                    embedding_specs.append((E, D, weights_ty))

                q_child = split_table_batched_embeddings_ops.IntNBitTableBatchedEmbeddingBagsCodegen(
                    embedding_specs=embedding_specs,
                    pooling_mode=child.pooling_mode,
                    use_cpu=use_cpu,
                )
                for t, (_, _, weight_ty) in enumerate(embedding_specs):
                    if weight_ty == SparseType.FP16:
                        original_weight = child.split_embedding_weights()[t]
                        q_weight = original_weight.half()
                        # FIXME: How to view the PyTorch Tensor as a different type (e.g., uint8)
                        # Here it uses numpy and it will introduce DtoH/HtoD overhead.
                        weights = torch.tensor(q_weight.cpu().numpy().view(np.uint8))
                        q_child.split_embedding_weights()[t][0].data.copy_(weights)

                    elif weight_ty == SparseType.INT8:
                        original_weight = child.split_embedding_weights()[t]
                        q_weight = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                            original_weight
                        )
                        weights = q_weight[:, :-8]
                        scale_shift = torch.tensor(
                            q_weight[:, -8:]
                            .contiguous()
                            .cpu()
                            .numpy()
                            .view(np.float32)
                            .astype(np.float16)
                            .view(np.uint8)
                        )  # [-4, -2]: scale; [-2:]: bias

                        q_child.split_embedding_weights()[t][0].data.copy_(weights)
                        q_child.split_embedding_weights()[t][1].data.copy_(scale_shift)

                    elif weight_ty == SparseType.INT4 or weight_ty == SparseType.INT2:
                        original_weight = child.split_embedding_weights()[t]
                        q_weight = (
                            torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                                original_weight,
                                bit_rate=self.quantize_type.bit_rate(),
                            )
                        )
                        weights = q_weight[:, :-4]
                        scale_shift = torch.tensor(
                            q_weight[:, -4:].contiguous().cpu().numpy().view(np.uint8)
                        )  # [-4, -2]: scale; [-2:]: bias
                        q_child.split_embedding_weights()[t][0].data.copy_(weights)
                        q_child.split_embedding_weights()[t][1].data.copy_(scale_shift)

                setattr(model, name, q_child)
            else:
                self._quantize_split_embs(child)

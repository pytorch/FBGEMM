# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import fbgemm_gpu
import fbgemm_gpu.split_table_batched_embeddings_ops
import torch

Tensor = torch.Tensor


def add_docs(method, docstr):
    method.__doc__ = docstr


add_docs(
    torch.ops.fbgemm.jagged_2d_to_dense,
    """
jagged_2d_to_dense(values, x_offsets, max_sequence_length) -> Tensor

Converts a jagged tensor, with a 2D values array into a dense tensor, padding with zeros.

Args:
    values (Tensor): 2D tensor containing the values of the jagged tensor.

    x_offsets (Tensor): 1D tensor containing the starting point of each jagged row in the values tensor.

    max_sequence_length (int): Maximum length of any row in the jagged dimension.

Returns:
    Tensor: The padded dense tensor

Example:
    >>> values = torch.tensor([[1,1],[2,2],[3,3],[4,4]])
    >>> x_offsets = torch.tensor([0, 1, 3])
    >>> torch.ops.fbgemm.jagged_2d_to_dense(values, x_offsets, 3)
    tensor([[[1, 1],
             [0, 0],
             [0, 0]],
            [[2, 2],
             [3, 3],
             [0, 0]]])

""",
)

# Example:
#
#    >>> t = torch.arange(4)


add_docs(
    torch.ops.fbgemm.jagged_1d_to_dense,
    """
jagged_1d_to_dense(values, offsets, max_sequence_length, padding_value) -> Tensor)

Converts a jagged tensor, with a 1D values array, into a dense tensor, padding with a specified padding value.

Args:
    values (Tensor): 1D tensor containing the values of the jagged tensor.

    offsets (Tensor): 1D tensor containing the starting point of each jagged row in the values tensor.

    max_sequence_length (int): Maximum length of any row in the jagged dimension.

    padding_value (int): Value to set in the empty areas of the dense output, outside of the jagged tensor coverage.

Returns:
    Tensor: the padded dense tensor

Example:
    >>> values = torch.tensor([1,2,3,4])
    >>> offsets = torch.tensor([0, 1, 3])
    >>> torch.ops.fbgemm.jagged_1d_to_dense(values, x_offsets, 3, 0)
    tensor([[1, 0, 0],
            [2, 3, 0]])

""",
)


add_docs(
    torch.ops.fbgemm.dense_to_jagged,
    """
dense_to_jagged(dense, x_offsets, total_L) -> (Tensor, Tensor[])

Converts a dense tensor into a jagged tensor, given the desired offsets of the resulting dense tensor. 

Args:
    dense (Tensor): A dense input tensor to be converted

    x_offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension. 

    total_L (int, Optional): Total number of values in the resulting jagged tensor.

Returns:
    (Tensor, Tensor[]): Values and offsets of the resulting jagged tensor. Offsets are identital to those that were input.

Example:
    >>> dense = torch.tensor([[[1, 1], [0, 0], [0, 0]], [[2, 2], [3, 3], [0, 0]]])
    >>> x_offsets = torch.tensor([0, 1, 3])
    >>> torch.ops.fbgemm.dense_to_jagged(dense, [x_offsets])
    (tensor([[1, 1],
             [2, 2],
             [3, 3]]), [tensor([0, 1, 3])])

""",
)


add_docs(
    torch.ops.fbgemm.jagged_to_padded_dense,
    """
jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0) -> Tensor

Converts a jagged tensor into a dense tensor, padding with a specified padding value.

Args:
    values (Tensor): Jagged tensor values

    offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension. 

    max_lengths (int[]): A list with max_length for each jagged dimension.

    padding_value (float): Value to set in the empty areas of the dense output, outside of the jagged tensor coverage.

Returns:
    Tensor: the padded dense tensor

Example:
    >>> values = torch.tensor([[1,1],[2,2],[3,3],[4,4]])
    >>> offsets = torch.tensor([0, 1, 3])
    >>> torch.ops.fbgemm.jagged_to_padded_dense(values, [offsets], [3], 7)
    tensor([[[1, 1],
             [7, 7],
             [7, 7]],
            [[2, 2],
             [3, 3],
             [7, 7]]])
""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_elementwise_add,
    """
jagged_dense_elementwise_add(x_values, x_offsets, y) -> Tensor

Adds a jagged tensor to a dense tensor, resulting in dense tensor. Jagged 
tensor input will be padded with zeros for the purposes of the addition.

Args:
    x_values (Tensor): Jagged tensor values

    offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension. 

    y (Tensor): A dense tensor

Returns:
    Tensor: The sum of jagged input tensor + y

""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output,
    """
jagged_dense_elementwise_add_jagged_output(x_values, x_offsets, y) -> (Tensor, Tensor[])

Adds a jagged tensor to a dense tensor and, resulting in a jagged tensor with the same structure as the input jagged tensor.

Args:
    x_values (Tensor): Jagged tensor values

    x_offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension. 

    y (Tensor): A dense tensor

Returns:
    (Tensor, Tensor[]): Values and offsets of the resulting jagged tensor. Offsets are identital to those that were input.

""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output,
    """
jagged_dense_dense_elementwise_add_jagged_output(x_values, x_offsets, y_0, y_1) -> (Tensor, Tensor[])

Adds a jagged tensor to the sum of two dense tensors, resulting in a jagged tensor with the same structure as the input jagged tensor.

Args:
    x_values (Tensor): Jagged tensor values

    x_offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension. 

    y_0 (Tensor): A dense tensor

    y_1 (Tensor): A dense tensor

Returns:
    (Tensor, Tensor[]): Values and offsets of the resulting jagged tensor. Offsets are identital to those that were input.

""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_elementwise_mul,
    """
jagged_dense_elementwise_mul(x_values, x_offsets, y) -> (Tensor, Tensor[])

Elementwise-multiplies a jagged tensor a dense tensor and, resulting in a jagged tensor with the same structure as the input jagged tensor.

Args:
    x_values (Tensor): Jagged tensor values

    x_offsets (Tensor[]): A list of jagged offset tensors, one for each jagged dimension. 

    y (Tensor): A dense tensor

Returns:
    (Tensor, Tensor[]): Values and offsets of the resulting jagged tensor. Offsets are identital to those that were input.

""",
)


add_docs(
    fbgemm_gpu.split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen,
    """
SplitTableBatchedEmbeddingBagsCodegen(embedding_specs, feature_table_map=None, cache_algorithm=CacheAlgorithm.LRU, cache_load_factor=0.2, cache_sets=0, cache_reserved_memory=0.0, cache_precision=SparseType.FP32, weights_precision=SparseType.FP32, output_dtype=SparseType.FP32, enforce_hbm=False, optimizer=OptimType.EXACT_SGD, record_cache_metrics=None, stochastic_rounding=True, gradient_clipping=False, max_gradient=1.0, learning_rate=0.01, eps=1.0e-8, momentum=0.9, weight_decay=0.0, weight_decay_mode=WeightDecayMode.NONE, eta=0.001, beta1=0.9, beta2=0.999, pooling_mode=PoolingMode.SUM, device=None, bounds_check_mode=BoundsCheckMode.WARNING) -> None

Table batched Embedding operator.  Looks up one or more embedding tables. The module is application for training. The backward operator is fused with optimizer. Thus, the embedding tables are updated during backward.

Args:
    embedding_specs (List[Tuple[int, int, EmbeddingLocation, ComputeDevice]]): A list of embedding specifications. Each spec is a tuple of (number of embedding rows, embedding dimension; must be a multiple of 4, table placement, compute device).

    feature_table_map (List[int], optional): An optional list that specifies feature-table mapping.

    cache_algorithm (CacheAlgorithm, optional): LXU cache algorithm (`CacheAlgorithm.LRU`, `CacheAlgorithm.LFU`)

    cache_load_factor (float, optional): The LXU cache capacity which is `cache_load_factor` * the total number of rows in all embedding tables

    cache_sets (int, optional): The number of cache sets

    cache_reserved_memory (float, optional): Amount of memory reserved in HBM for non-cache purpose.

    cache_precision (SparseType, optional): Data type of LXU cache (`SparseType.FP32`, `SparseType.FP16`)

    weights_precision (SparseType, optional): Data type of embedding tables (also known as weights) (`SparseType.FP32`, `SparseType.FP16`, `SparseType.INT8`)

    output_dtype (SparseType, optional): Data type of an output tensor (`SparseType.FP32`, `SparseType.FP16`, `SparseType.INT8`)

    enforce_hbm (bool, optional): If True, place all weights/momentums in HBM when using cache

    optimizer (OptimType, optional): An optimizer to use for embedding table update in the backward pass. (`OptimType.ADAM`, `OptimType.EXACT_ADAGRAD`, `OptimType.EXACT_ROWWISE_ADAGRAD`, `OptimType.EXACT_ROWWISE_WEIGHTED_ADAGRAD`, `OptimType.EXACT_SGD`, `OptimType.LAMB`, `OptimType.LARS_SGD`, `OptimType.PARTIAL_ROWWISE_ADAM`, `OptimType.PARTIAL_ROWWISE_LAMB`, `OptimType.SGD`)

    record_cache_metrics (RecordCacheMetrics, optional): Record number of hits, number of requests, etc if RecordCacheMetrics.record_cache_miss_counter is True and record the similar metrics table-wise if RecordCacheMetrics.record_tablewise_cache_miss is True (default is None).

    stochastic_rounding (bool, optional): If True, apply stochastic rounding for weight type that is not `SparseType.FP32`

    gradient_clipping (bool, optional): If True, apply gradient clipping

    max_gradient (float, optional): The value for gradient clipping

    learning_rate (float, optional): The learning rate

    eps (float, optional): The epsilon value used by Adagrad, LAMB, and Adam

    momentum (float, optional): Momentum used by LARS-SGD

    weight_decay (float, optional): Weight decay used by LARS-SGD, LAMB, ADAM, and Rowwise Adagrad

    weight_decay_mode (WeightDecayMode, optional): Weight decay mode (`WeightDecayMode.NONE`, `WeightDecayMode.L2`, `WeightDecayMode.DECOUPLE`)

    eta (float, optional): The eta value used by LARS-SGD

    beta1 (float, optional): The beta1 value used by LAMB and ADAM

    beta2 (float, optional): The beta2 value used by LAMB and ADAM

    pooling_mode (PoolingMode, optional): Pooling mode (`PoolingMode.SUM`, `PoolingMode.MEAN`, `PoolingMode.NONE`)

    device (torch.device, optional): The current device to place tensors on

    bounds_check_mode (BoundsCheckMode, optional): If not set to `BoundsCheckMode.NONE`, apply boundary check for indices (`BoundsCheckMode.NONE`, `BoundsCheckMode.FATAL`, `BoundsCheckMode.WARNING`, `BoundsCheckMode.IGNORE`)

Inputs:
    indices (torch.Tensor): A 1D-tensor that contains indices to be accessed in all embedding table

    offsets (torch.Tensor): A 1D-tensor that conatins offsets of indices.  Shape `(B * T + 1)` where `B` = batch size and `T` = number of tables.  `offsets[t * B + b + 1] - offsets[t * B + b]` is the length of bag `b` of table `t`

    per_sample_weights (torch.Tensor, optional): An optional 1D-tensor that contains positional weights. Shape `(max(bag length))`. Positional weight `i` is multiplied to all columns of row `i` in each bag after its read from the embedding table and before pooling (if pooling mode is not PoolingMode.NONE).

    feature_requires_grad (torch.Tensor, optional): An optional tensor for checking if `per_sample_weights` requires gradient

Returns:
    A 2D-tensor containing looked up data. Shape `(B, total_D)` where `B` = batch size and `total_D` = the sum of all embedding dimensions in the table

Example:
    >>> import torch
    >>> from fbgemm_gpu.split_table_batched_embeddings_ops import (
    >>>    SplitTableBatchedEmbeddingBagsCodegen,
    >>>    EmbeddingLocation,
    >>>    ComputeDevice,
    >>> )
    >>>
    >>> # Two tables
    >>> embedding_specs = [
    >>>     (3, 8, EmbeddingLocation.DEVICE, ComputeDevice.CUDA),
    >>>     (5, 4, EmbeddingLocation.MANAGED, ComputeDevice.CUDA)
    >>> ]
    >>>
    >>> tbe = SplitTableBatchedEmbeddingBagsCodegen(embedding_specs)
    >>> tbe.init_embedding_weights_uniform(-1, 1)
    >>>
    >>> print(tbe.split_embedding_weights())
    [tensor([[-0.9426,  0.7046,  0.4214, -0.0419,  0.1331, -0.7856, -0.8124, -0.2021],
            [-0.5771,  0.5911, -0.7792, -0.1068, -0.6203,  0.4813, -0.1677,  0.4790],
            [-0.5587, -0.0941,  0.5754,  0.3475, -0.8952, -0.1964,  0.0810, -0.4174]],
           device='cuda:0'), tensor([[-0.2513, -0.4039, -0.3775,  0.3273],
            [-0.5399, -0.0229, -0.1455, -0.8770],
            [-0.9520,  0.4593, -0.7169,  0.6307],
            [-0.1765,  0.8757,  0.8614,  0.2051],
            [-0.0603, -0.9980, -0.7958, -0.5826]], device='cuda:0')]


    >>> # Batch size = 3
    >>> indices = torch.tensor([0, 1, 2, 0, 1, 2, 0, 3, 1, 4, 2, 0, 0],
    >>>                        device="cuda",
    >>>                        dtype=torch.long)
    >>> offsets = torch.tensor([0, 2, 5, 7, 9, 12, 13],
    >>>                        device="cuda",
    >>>                        dtype=torch.long)
    >>>
    >>> output = tbe(indices, offsets)
    >>>
    >>> # Batch size = 3, total embedding dimension = 12
    >>> print(output.shape)
    torch.Size([3, 12])

    >>> print(output)
    tensor([[-1.5197,  1.2957, -0.3578, -0.1487, -0.4873, -0.3044, -0.9801,  0.2769,
             -0.7164,  0.8528,  0.7159, -0.6719],
            [-2.0784,  1.2016,  0.2176,  0.1988, -1.3825, -0.5008, -0.8991, -0.1405,
             -1.2637, -0.9427, -1.8902,  0.3754],
            [-1.5013,  0.6105,  0.9968,  0.3057, -0.7621, -0.9821, -0.7314, -0.6195,
             -0.2513, -0.4039, -0.3775,  0.3273]], device='cuda:0',
           grad_fn=<CppNode<SplitLookupFunction_sgd_Op>>)
""",
)


# add_docs(
#    torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul,
#    """Args:
#                {input}
#            Keyword args:
#                {out}""",
# )
#
#
# add_docs(
#    torch.ops.fbgemm.stacked_jagged_1d_to_dense,
#    """Args:
#                {input}
#            Keyword args:
#                {out}""",
# )
#
#
# add_docs(
#    torch.ops.fbgemm.stacked_jagged_2d_to_dense,
#    """Args:
#                {input}
#            Keyword args:
#                {out}""",
# )

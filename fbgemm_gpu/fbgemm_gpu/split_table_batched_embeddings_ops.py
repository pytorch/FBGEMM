#!/usr/bin/env python3

# pyre-ignore-all-errors[56]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import logging
from dataclasses import dataclass
from itertools import accumulate
from math import log2
from typing import Any, Dict, List, Optional, Tuple

import fbgemm_gpu.split_embedding_codegen_lookup_invokers as invokers
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_embedding_configs import SparseType
from torch import Tensor, nn

ASSOC = 32
# Maximum number of times prefetch() can be called without
# a corresponding forward() call
MAX_PREFETCH_DEPTH = 100
INT8_EMB_ROW_DIM_OFFSET = 8


class DoesNotHavePrefix(Exception):
    pass


class EmbeddingLocation(enum.IntEnum):
    DEVICE = 0
    MANAGED = 1
    MANAGED_CACHING = 2
    HOST = 3


class ComputeDevice(enum.IntEnum):
    CPU = 0
    CUDA = 1


class CacheAlgorithm(enum.Enum):
    LRU = 0
    LFU = 1


class PoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1
    NONE = 2


@dataclass
class SplitState:
    dev_size: int
    host_size: int
    uvm_size: int
    placements: List[EmbeddingLocation]
    offsets: List[int]


def construct_split_state(
    embedding_specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]],
    rowwise: bool,
    cacheable: bool,
    precision: SparseType = SparseType.FP32,
    int8_emb_row_dim_offset: int = INT8_EMB_ROW_DIM_OFFSET,
) -> SplitState:
    placements = []
    offsets = []
    dev_size = 0
    host_size = 0
    uvm_size = 0
    for (num_embeddings, embedding_dim, location, _) in embedding_specs:
        assert embedding_dim % 4 == 0, f"{embedding_dim}"
        if precision == SparseType.INT8:
            embedding_dim += int8_emb_row_dim_offset
        state_size = num_embeddings * embedding_dim if not rowwise else num_embeddings
        if location == EmbeddingLocation.HOST:
            placements.append(EmbeddingLocation.HOST)
            offsets.append(host_size)
            host_size += state_size
        # If table is on device, then opimtizer is on device.
        # If table is managed, then if optimizer state is rowwise, optimizer is on device, otherwise optimizer is managed.
        elif location == EmbeddingLocation.DEVICE or rowwise:
            placements.append(EmbeddingLocation.DEVICE)
            offsets.append(dev_size)
            dev_size += state_size
        else:
            if cacheable and location == EmbeddingLocation.MANAGED_CACHING:
                placements.append(EmbeddingLocation.MANAGED_CACHING)
            else:
                placements.append(EmbeddingLocation.MANAGED)
            offsets.append(uvm_size)
            uvm_size += state_size
    assert len(placements) == len(offsets)
    return SplitState(
        dev_size=dev_size,
        host_size=host_size,
        uvm_size=uvm_size,
        placements=placements,
        offsets=offsets,
    )


@dataclass
class CacheState:
    # T + 1 elements and cache_hash_size_cumsum[-1] == total_cache_hash_size
    cache_hash_size_cumsum: List[int]
    cache_index_table_map: List[int]
    total_cache_hash_size: int


def construct_cache_state(
    embedding_specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]],
    feature_table_map: List[int],
) -> CacheState:
    _cache_hash_size_cumsum = [0]
    total_cache_hash_size = 0
    for (num_embeddings, _, location, _) in embedding_specs:
        if location == EmbeddingLocation.MANAGED_CACHING:
            total_cache_hash_size += num_embeddings
        _cache_hash_size_cumsum.append(total_cache_hash_size)
    # [T], -1: non-cached table
    cache_hash_size_cumsum = []
    # [total_cache_hash_size], linear cache index -> table index
    cache_index_table_map = [-1] * total_cache_hash_size
    for t, t_ in enumerate(feature_table_map):
        for i in range(_cache_hash_size_cumsum[t_], _cache_hash_size_cumsum[t_ + 1]):
            cache_index_table_map[i] = t
        (_, _, location, _) = embedding_specs[t_]
        if location == EmbeddingLocation.MANAGED_CACHING:
            cache_hash_size_cumsum.append(_cache_hash_size_cumsum[t_])
        else:
            cache_hash_size_cumsum.append(-1)
    cache_hash_size_cumsum.append(total_cache_hash_size)
    s = CacheState(
        cache_hash_size_cumsum=cache_hash_size_cumsum,
        cache_index_table_map=cache_index_table_map,
        total_cache_hash_size=total_cache_hash_size,
    )
    return s


class SplitTableBatchedEmbeddingBagsCodegen(nn.Module):
    """
    Multiple sparse features can share one embedding table.
    'feature_table_map' specifies the feature-table mapping.
    T:  number of logical tables
    T_: number of physical tables
    T >= T_
    """

    embedding_specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]]
    optimizer_args: invokers.lookup_args.OptimizerArgs
    lxu_cache_locations_list: List[Tensor]
    lxu_cache_locations_empty: Tensor
    timesteps_prefetched: List[int]

    def __init__(  # noqa C901
        self,
        embedding_specs: List[
            Tuple[int, int, EmbeddingLocation, ComputeDevice]
        ],  # tuple of (rows, dims, placements, compute_devices)
        feature_table_map: Optional[List[int]] = None,  # [T]
        cache_algorithm: CacheAlgorithm = CacheAlgorithm.LRU,
        cache_load_factor: float = 0.2,
        cache_sets: int = 0,
        cache_reserved_memory: float = 0.0,
        cache_precision: SparseType = SparseType.FP32,
        weights_precision: SparseType = SparseType.FP32,
        enforce_hbm: bool = False,  # place all weights/momentums in HBM when using cache
        optimizer: OptimType = OptimType.EXACT_SGD,
        # General Optimizer args
        stochastic_rounding: bool = False,
        gradient_clipping: bool = False,
        max_gradient: float = 1.0,
        learning_rate: float = 0.01,
        eps: float = 1.0e-8,  # used by Adagrad, LAMB, and Adam
        momentum: float = 0.9,  # used by LARS-SGD
        weight_decay: float = 0.0,  # used by LARS-SGD, LAMB, and ADAM
        eta: float = 0.001,  # used by LARS-SGD,
        beta1: float = 0.9,  # used by LAMB and ADAM
        beta2: float = 0.999,  # used by LAMB and ADAM
        pooling_mode: PoolingMode = PoolingMode.SUM,
    ) -> None:
        super(SplitTableBatchedEmbeddingBagsCodegen, self).__init__()
        self.pooling_mode = pooling_mode
        self.weights_precision = weights_precision
        # NOTE: a placeholder to avoid multi-construction and make TorchScript work!
        self.dummy_tensor: Tensor = torch.tensor(0)

        self.embedding_specs = embedding_specs
        (rows, dims, locations, compute_devices) = zip(*embedding_specs)
        T_ = len(self.embedding_specs)
        assert T_ > 0

        assert all(
            cd == compute_devices[0] for cd in compute_devices
        ), "Heterogenous compute_devices are NOT supported!"
        self.use_cpu: bool = all(cd == ComputeDevice.CPU for cd in compute_devices)
        assert not self.use_cpu or all(
            loc == EmbeddingLocation.HOST for loc in locations
        ), "ComputeDevice.CPU is only for EmbeddingLocation.HOST!"
        self.current_device: torch.device = (
            torch.device("cpu") if self.use_cpu else torch.cuda.current_device()
        )

        # add placeholder require_grad param tensor to enable autograd with int8 weights
        self.placeholder_autograd_tensor = nn.Parameter(
            torch.zeros(0, device=self.current_device, dtype=torch.float)
        )

        self.int8_emb_row_dim_offset: int = INT8_EMB_ROW_DIM_OFFSET

        self.feature_table_map: List[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        T = len(self.feature_table_map)
        assert T_ <= T
        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        assert all(table_has_feature), "Each table must have at least one feature!"

        D_offsets = [dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(accumulate(D_offsets))
        self.total_D: int = D_offsets[-1]
        self.max_D: int = max(dims)
        cached_dims = [
            embedding_spec[1]
            for embedding_spec in embedding_specs
            if embedding_spec[2] == EmbeddingLocation.MANAGED_CACHING
        ]
        self.max_D_cache: int = max(cached_dims) if len(cached_dims) > 0 else 0

        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, device=self.current_device, dtype=torch.int32),
        )

        hash_size_cumsum = [0] + list(accumulate(rows))
        self.total_hash_size_bits = int(log2(float(hash_size_cumsum[-1])) + 1)
        # The last element is to easily access # of rows of each table by
        # hash_size_cumsum[t + 1] - hash_size_cumsum[t]
        hash_size_cumsum = [hash_size_cumsum[t] for t in self.feature_table_map] + [
            hash_size_cumsum[-1]
        ]
        self.register_buffer(
            "hash_size_cumsum",
            torch.tensor(
                hash_size_cumsum, device=self.current_device, dtype=torch.int64
            ),
        )
        weight_split = construct_split_state(
            embedding_specs,
            rowwise=False,
            cacheable=True,
            precision=weights_precision,
        )
        table_embedding_dtype = torch.float32
        if weights_precision == SparseType.FP16:
            table_embedding_dtype = torch.float16
        elif weights_precision == SparseType.INT8:
            table_embedding_dtype = torch.uint8

        self._apply_split(
            weight_split,
            prefix="weights",
            dtype=table_embedding_dtype,
            enforce_hbm=enforce_hbm,
        )

        if self.use_cpu:
            # Construct optimizer states
            assert optimizer in (
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
                OptimType.ROWWISE_ADAGRAD,
                OptimType.SGD,
            ), f"Optimizer {optimizer} is not supported in cpu mode."
        else:
            assert optimizer in (
                OptimType.ADAM,
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
                OptimType.LAMB,
                OptimType.LARS_SGD,
                OptimType.PARTIAL_ROWWISE_ADAM,
                OptimType.PARTIAL_ROWWISE_LAMB,
                OptimType.SGD,
            ), f"Optimizer {optimizer} is not supported."

        self.stochastic_rounding = stochastic_rounding
        self.optimizer = optimizer

        self.optimizer_args = invokers.lookup_args.OptimizerArgs(
            stochastic_rounding=stochastic_rounding,
            gradient_clipping=gradient_clipping,
            max_gradient=max_gradient,
            learning_rate=learning_rate,
            eps=eps,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eta=eta,
            momentum=momentum,
        )

        if optimizer in (
            OptimType.SGD,
            OptimType.EXACT_SGD,
        ):
            # NOTE: make TorchScript work!
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "momentum1_dev", torch.tensor([0], dtype=torch.int64), persistent=False
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "momentum1_host", torch.tensor([0], dtype=torch.int64), persistent=False
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "momentum1_uvm", torch.tensor([0], dtype=torch.int64), persistent=False
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "momentum1_placements",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "momentum1_offsets",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
        else:
            self._apply_split(
                construct_split_state(
                    embedding_specs,
                    rowwise=optimizer
                    in [OptimType.EXACT_ROWWISE_ADAGRAD, OptimType.ROWWISE_ADAGRAD],
                    cacheable=False,
                ),
                prefix="momentum1",
                dtype=torch.float32,
                enforce_hbm=enforce_hbm,
            )
        if optimizer in (
            OptimType.ADAM,
            OptimType.PARTIAL_ROWWISE_ADAM,
            OptimType.LAMB,
            OptimType.PARTIAL_ROWWISE_LAMB,
        ):
            self._apply_split(
                construct_split_state(
                    embedding_specs,
                    rowwise=optimizer
                    in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.PARTIAL_ROWWISE_LAMB),
                    cacheable=False,
                ),
                prefix="momentum2",
                dtype=torch.float32,
            )
            self.register_buffer("iter", torch.tensor([0], dtype=torch.int64))
        else:
            # NOTE: make TorchScript work!
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "momentum2_dev", torch.tensor([0], dtype=torch.int64), persistent=False
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "momentum2_host", torch.tensor([0], dtype=torch.int64), persistent=False
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "momentum2_uvm", torch.tensor([0], dtype=torch.int64), persistent=False
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "momentum2_placements",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "momentum2_offsets",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "iter", torch.tensor([0], dtype=torch.int64), persistent=False
            )

        cache_state = construct_cache_state(embedding_specs, self.feature_table_map)
        if cache_precision == SparseType.FP32:
            cache_embedding_dtype = torch.float32
        elif cache_precision == SparseType.FP16:
            cache_embedding_dtype = torch.float16
        else:
            raise AssertionError(f"cache_precision {cache_precision} not supported!")

        self._apply_cache_state(
            cache_state,
            cache_algorithm,
            cache_load_factor,
            cache_sets,
            cache_reserved_memory,
            dtype=cache_embedding_dtype,
        )

        logging.debug(
            f"Using fused {optimizer} with optimizer_args={self.optimizer_args}"
        )

        self.step = 0

    def get_states(self, prefix: str) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not hasattr(self, f"{prefix}_physical_placements"):
            raise DoesNotHavePrefix()
        dev_param = getattr(self, f"{prefix}_dev")
        host_param = getattr(self, f"{prefix}_host")
        uvm_param = getattr(self, f"{prefix}_uvm")
        placements = getattr(self, f"{prefix}_physical_placements")
        offsets = getattr(self, f"{prefix}_physical_offsets")
        return (
            dev_param,
            host_param,
            uvm_param,
            torch.tensor(placements, dtype=torch.int32),
            torch.tensor(offsets, dtype=torch.int64),
        )

    def get_all_states(self) -> List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        all_states = []
        for prefix in ["weights", "momentum1", "momentum2"]:
            try:
                all_states.append(self.get_states(prefix))
            except DoesNotHavePrefix:
                pass
        return all_states

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
    ) -> Tensor:
        (indices, offsets) = indices.long(), offsets.long()
        self.step += 1

        if len(self.timesteps_prefetched) == 0:
            self.prefetch(indices, offsets)

        self.timesteps_prefetched.pop(0)
        lxu_cache_locations = (
            self.lxu_cache_locations_empty
            if len(self.lxu_cache_locations_list) == 0
            else self.lxu_cache_locations_list.pop(0)
        )
        common_args = invokers.lookup_args.CommonArgs(
            placeholder_autograd_tensor=self.placeholder_autograd_tensor,
            # pyre-fixme[6]: Expected `Tensor` for 2nd param but got `Union[Tensor,
            #  nn.Module]`.
            dev_weights=self.weights_dev,
            host_weights=self.weights_host,
            uvm_weights=self.weights_uvm,
            lxu_cache_weights=self.lxu_cache_weights,
            weights_placements=self.weights_placements,
            weights_offsets=self.weights_offsets,
            D_offsets=self.D_offsets,
            total_D=self.total_D,
            max_D=self.max_D,
            hash_size_cumsum=self.hash_size_cumsum,
            total_hash_size_bits=self.total_hash_size_bits,
            indices=indices,
            offsets=offsets,
            pooling_mode=self.pooling_mode,
            indice_weights=per_sample_weights,
            feature_requires_grad=feature_requires_grad,
            lxu_cache_locations=lxu_cache_locations,
        )

        if self.optimizer == OptimType.EXACT_SGD:
            return invokers.lookup_sgd.invoke(common_args, self.optimizer_args)
        elif self.optimizer == OptimType.SGD:
            assert self.use_cpu, "Approx SGD is only supported in CPU mode"
            return invokers.lookup_approx_sgd.invoke(common_args, self.optimizer_args)

        momentum1 = invokers.lookup_args.Momentum(
            # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[Tensor,
            #  nn.Module]`.
            dev=self.momentum1_dev,
            host=self.momentum1_host,
            uvm=self.momentum1_uvm,
            offsets=self.momentum1_offsets,
            placements=self.momentum1_placements,
        )

        if self.optimizer == OptimType.LARS_SGD:
            return invokers.lookup_lars_sgd.invoke(
                common_args, self.optimizer_args, momentum1
            )
        if self.optimizer == OptimType.EXACT_ADAGRAD:
            return invokers.lookup_adagrad.invoke(
                common_args, self.optimizer_args, momentum1
            )
        if self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
            return invokers.lookup_rowwise_adagrad.invoke(
                common_args, self.optimizer_args, momentum1
            )
        if self.optimizer == OptimType.ROWWISE_ADAGRAD:
            assert self.use_cpu, "Approx rowwise AdaGrad is only supported in CPU mode"
            return invokers.lookup_approx_rowwise_adagrad.invoke(
                common_args, self.optimizer_args, momentum1
            )

        momentum2 = invokers.lookup_args.Momentum(
            # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[Tensor,
            #  nn.Module]`.
            dev=self.momentum2_dev,
            host=self.momentum2_host,
            uvm=self.momentum2_uvm,
            offsets=self.momentum2_offsets,
            placements=self.momentum2_placements,
        )
        # Ensure iter is always on CPU so the increment doesn't synchronize.
        if self.iter.is_cuda:
            # pyre-fixme[16]: `SplitTableBatchedEmbeddingBagsCodegen` has no
            #  attribute `iter`.
            self.iter = self.iter.cpu()
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.__getitem__)[[Named(self, Tensor),
        #  Named(item, typing.Any)], typing.Any], Tensor], Tensor, nn.Module]` is not a
        #  function.
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.__setitem__)[[Named(self, Tensor),
        #  Named(item, typing.Any), Named(other, typing.Any)], None], Tensor], Tensor,
        #  nn.Module]` is not a function.
        self.iter[0] += 1

        if self.optimizer == OptimType.ADAM:
            return invokers.lookup_adam.invoke(
                common_args,
                self.optimizer_args,
                momentum1,
                momentum2,
                # pyre-fixme[29]:
                #  `Union[BoundMethod[typing.Callable(Tensor.item)[[Named(self,
                #  Tensor)], typing.Union[float, int]], Tensor], Tensor, nn.Module]` is
                #  not a function.
                self.iter.item(),
            )
        if self.optimizer == OptimType.PARTIAL_ROWWISE_ADAM:
            return invokers.lookup_partial_rowwise_adam.invoke(
                common_args,
                self.optimizer_args,
                momentum1,
                momentum2,
                # pyre-fixme[29]:
                #  `Union[BoundMethod[typing.Callable(Tensor.item)[[Named(self,
                #  Tensor)], typing.Union[float, int]], Tensor], Tensor, nn.Module]` is
                #  not a function.
                self.iter.item(),
            )
        if self.optimizer == OptimType.LAMB:
            return invokers.lookup_lamb.invoke(
                common_args,
                self.optimizer_args,
                momentum1,
                momentum2,
                # pyre-fixme[29]:
                #  `Union[BoundMethod[typing.Callable(Tensor.item)[[Named(self,
                #  Tensor)], typing.Union[float, int]], Tensor], Tensor, nn.Module]` is
                #  not a function.
                self.iter.item(),
            )
        if self.optimizer == OptimType.PARTIAL_ROWWISE_LAMB:
            return invokers.lookup_partial_rowwise_lamb.invoke(
                common_args,
                self.optimizer_args,
                momentum1,
                momentum2,
                # pyre-fixme[29]:
                #  `Union[BoundMethod[typing.Callable(Tensor.item)[[Named(self,
                #  Tensor)], typing.Union[float, int]], Tensor], Tensor, nn.Module]` is
                #  not a function.
                self.iter.item(),
            )

        raise ValueError(f"Invalid OptimType: {self.optimizer}")

    def prefetch(self, indices: Tensor, offsets: Tensor) -> None:
        self.timestep += 1
        self.timesteps_prefetched.append(self.timestep)
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.numel)[[Named(self, Tensor)],
        #  int], Tensor], Tensor, nn.Module]` is not a function.
        if not self.lxu_cache_weights.numel():
            return

        (indices, offsets) = indices.long(), offsets.long()
        linear_cache_indices = torch.ops.fb.linearize_cache_indices(
            self.cache_hash_size_cumsum,
            indices,
            offsets,
        )
        if self.cache_algorithm == CacheAlgorithm.LRU:
            torch.ops.fb.lru_cache_populate(
                self.weights_uvm,
                self.cache_hash_size_cumsum,
                self.total_cache_hash_size,
                self.cache_index_table_map,
                self.weights_offsets,
                self.D_offsets,
                linear_cache_indices,
                self.lxu_cache_state,
                self.lxu_cache_weights,
                self.timestep,
                self.lxu_state,
                self.stochastic_rounding,
            )
        elif self.cache_algorithm == CacheAlgorithm.LFU:
            torch.ops.fb.lfu_cache_populate(
                self.weights_uvm,
                self.cache_hash_size_cumsum,
                self.total_cache_hash_size,
                self.cache_index_table_map,
                self.weights_offsets,
                self.D_offsets,
                linear_cache_indices,
                self.lxu_cache_state,
                self.lxu_cache_weights,
                self.lxu_state,
                self.stochastic_rounding,
            )

        assert (
            len(self.lxu_cache_locations_list) < self.max_prefetch_depth
        ), f"self.lxu_cache_locations_list has grown to size: {len(self.lxu_cache_locations_list)}, this exceeds the maximum: {self.max_prefetch_depth}. This probably indicates an error in logic where prefetch() is being called more frequently than forward()"
        self.lxu_cache_locations_list.append(
            torch.ops.fb.lxu_cache_lookup(
                linear_cache_indices,
                self.lxu_cache_state,
            )
        )

    def init_embedding_weights_uniform(self, min_val: float, max_val: float) -> None:
        splits = self.split_embedding_weights()
        if self.weights_precision == SparseType.INT8:
            # TODO: add in-place FloatToFused8BitRowwiseQuantized conversion
            for emb in splits:
                assert (
                    len(emb.shape) == 2
                ), "Int8 embedding only supported for 2D weight tensors."
                shape = [emb.shape[0], emb.shape[1] - self.int8_emb_row_dim_offset]
                tmp_emb = torch.zeros(shape, device=self.current_device)
                tmp_emb.uniform_(min_val, max_val)
                tmp_emb_i8 = torch.ops.fb.FloatToFused8BitRowwiseQuantized(tmp_emb)
                emb.data.copy_(tmp_emb_i8)
        else:
            for param in splits:
                param.uniform_(min_val, max_val)

    @torch.jit.export
    def split_embedding_weights(self) -> List[Tensor]:
        """
        Returns a list of weights, split by table
        """
        splits = []
        for t, (rows, dim, _, _) in enumerate(self.embedding_specs):
            if self.weights_precision == SparseType.INT8:
                dim += self.int8_emb_row_dim_offset
            # pyre-fixme[29]:
            #  `Union[BoundMethod[typing.Callable(Tensor.__getitem__)[[Named(self,
            #  Tensor), Named(item, typing.Any)], typing.Any], Tensor], Tensor,
            #  nn.Module]` is not a function.
            placement = self.weights_physical_placements[t]
            # pyre-fixme[29]:
            #  `Union[BoundMethod[typing.Callable(Tensor.__getitem__)[[Named(self,
            #  Tensor), Named(item, typing.Any)], typing.Any], Tensor], Tensor,
            #  nn.Module]` is not a function.
            offset = self.weights_physical_offsets[t]
            if placement == EmbeddingLocation.DEVICE.value:
                weights = self.weights_dev
            elif placement == EmbeddingLocation.HOST.value:
                weights = self.weights_host
            else:
                weights = self.weights_uvm
            splits.append(
                # pyre-fixme[29]:
                #  `Union[BoundMethod[typing.Callable(Tensor.detach)[[Named(self,
                #  Tensor)], Tensor], Tensor], Tensor, nn.Module]` is not a function.
                weights.detach()[offset : offset + rows * dim].view(rows, dim)
            )
        return splits

    @torch.jit.ignore
    def get_optimizer_buffer(self, state: str) -> torch.Tensor:
        for name, buffer in self.named_buffers():
            if name == state:
                return buffer
        return torch.tensor(0)

    @torch.jit.export
    def get_optimizer_state(self) -> List[Dict[str, torch.Tensor]]:
        r"""
        Get the optimizer state dict that matches the OSS Pytorch optims
        TODO: populate the supported list of optimizers
        """
        if (
            self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD
            or self.optimizer == OptimType.ROWWISE_ADAGRAD
        ):
            list_of_state_dict = [
                {"sum": _sum[0]} for _sum in self.split_optimizer_states()
            ]
        else:
            raise NotImplementedError(
                f"Getting optimizer state {self.optimizer} is not implmeneted"
            )

        return list_of_state_dict

    @torch.jit.ignore
    def split_optimizer_states(self) -> List[Tuple[torch.Tensor]]:
        """
        Returns a list of states, split by table
        """

        def get_optimizer_states(
            state_dev: Tensor,
            state_host: Tensor,
            state_uvm: Tensor,
            state_offsets: Tensor,
            state_placements: Tensor,
            rowwise: bool,
        ) -> List[torch.Tensor]:
            splits = []
            for t, (rows, dim, _, _) in enumerate(self.embedding_specs):
                offset = state_offsets[t]
                placement = state_placements[t]
                if placement == EmbeddingLocation.DEVICE:
                    state = state_dev
                elif placement == EmbeddingLocation.HOST:
                    state = state_host
                else:
                    state = state_uvm
                if not rowwise:
                    splits.append(
                        state.detach()[offset : offset + rows * dim].view(rows, dim)
                    )
                else:
                    splits.append(state.detach()[offset : offset + rows].view(rows))
            return splits

        states: List[List[torch.Tensor]] = []
        if self.optimizer not in (
            OptimType.SGD,
            OptimType.EXACT_SGD,
        ):
            states.append(
                get_optimizer_states(
                    # pyre-fixme[6]: Expected `Tensor` for 1st param but got
                    #  `Union[Tensor, nn.Module]`.
                    self.momentum1_dev,
                    self.momentum1_host,
                    self.momentum1_uvm,
                    self.momentum1_physical_offsets,
                    self.momentum1_physical_placements,
                    rowwise=self.optimizer
                    in [OptimType.EXACT_ROWWISE_ADAGRAD, OptimType.ROWWISE_ADAGRAD],
                )
            )
        if self.optimizer in (
            OptimType.ADAM,
            OptimType.PARTIAL_ROWWISE_ADAM,
            OptimType.LAMB,
            OptimType.PARTIAL_ROWWISE_LAMB,
        ):
            states.append(
                get_optimizer_states(
                    # pyre-fixme[6]: Expected `Tensor` for 1st param but got
                    #  `Union[Tensor, nn.Module]`.
                    self.momentum2_dev,
                    self.momentum2_host,
                    self.momentum2_uvm,
                    self.momentum2_physical_offsets,
                    self.momentum2_physical_placements,
                    rowwise=self.optimizer
                    in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.PARTIAL_ROWWISE_LAMB),
                )
            )
        return list(zip(*states))

    @torch.jit.export
    def set_learning_rate(self, lr: float) -> None:
        """
        Sets the learning rate.
        """
        self._set_learning_rate(lr)

    @torch.jit.ignore
    def _set_learning_rate(self, lr: float) -> float:
        """
        Helper function to script `set_learning_rate`.
        Note that returning None does not work.
        """
        self.optimizer_args = self.optimizer_args._replace(learning_rate=lr)
        return 0.0

    @torch.jit.export
    def flush(self) -> None:
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.numel)[[Named(self, Tensor)],
        #  int], Tensor], Tensor, nn.Module]` is not a function.
        if not self.lxu_cache_weights.numel():
            return
        torch.ops.fb.lxu_cache_flush(
            self.weights_uvm,
            self.cache_hash_size_cumsum,
            self.cache_index_table_map,
            self.weights_offsets,
            self.D_offsets,
            self.total_D,
            self.lxu_cache_state,
            self.lxu_cache_weights,
            self.stochastic_rounding,
        )

    def _apply_split(
        self,
        split: SplitState,
        prefix: str,
        dtype: torch.dtype,
        enforce_hbm: bool = False,
    ) -> None:
        setattr(self, f"{prefix}_physical_placements", split.placements)
        setattr(self, f"{prefix}_physical_offsets", split.offsets)

        offsets = [split.offsets[t] for t in self.feature_table_map]
        placements = [split.placements[t] for t in self.feature_table_map]
        self.register_buffer(
            f"{prefix}_offsets",
            torch.tensor(offsets, device=self.current_device, dtype=torch.int64),
        )
        self.register_buffer(
            f"{prefix}_placements",
            torch.tensor(placements, device=self.current_device, dtype=torch.int32),
        )
        if split.dev_size > 0:
            self.register_buffer(
                f"{prefix}_dev",
                torch.zeros(split.dev_size, device=self.current_device, dtype=dtype),
            )
        else:
            self.register_buffer(
                f"{prefix}_dev",
                torch.empty(0, device=self.current_device, dtype=dtype),
            )
        if split.host_size > 0:
            if dtype == torch.uint8:
                self.register_buffer(
                    f"{prefix}_host",
                    torch.zeros(
                        split.host_size, device=self.current_device, dtype=dtype
                    ),
                )
            else:
                setattr(
                    self,
                    f"{prefix}_host",
                    nn.Parameter(
                        torch.zeros(
                            split.host_size, device=self.current_device, dtype=dtype
                        )
                    ),
                )
        else:
            self.register_buffer(
                f"{prefix}_host",
                torch.empty(0, device=self.current_device, dtype=dtype),
            )
        if split.uvm_size > 0:
            assert not self.use_cpu
            if enforce_hbm:
                self.register_buffer(
                    f"{prefix}_uvm",
                    torch.zeros(
                        split.uvm_size, device=self.current_device, dtype=dtype
                    ),
                )
            else:
                self.register_buffer(
                    f"{prefix}_uvm",
                    torch.zeros(
                        split.uvm_size,
                        out=torch.ops.fb.new_managed_tensor(
                            torch.zeros(1, device=self.current_device, dtype=dtype),
                            [split.uvm_size],
                        ),
                    ),
                )
        else:
            self.register_buffer(
                f"{prefix}_uvm",
                torch.empty(0, device=self.current_device, dtype=dtype),
            )

    def _apply_cache_state(
        self,
        cache_state: CacheState,
        cache_algorithm: CacheAlgorithm,
        cache_load_factor: float,
        cache_sets: int,
        cache_reserved_memory: float,
        dtype: torch.dtype,
    ) -> None:
        self.cache_algorithm = cache_algorithm
        self.timestep = 1
        self.timesteps_prefetched = []

        self.max_prefetch_depth = MAX_PREFETCH_DEPTH
        self.lxu_cache_locations_list = []
        self.lxu_cache_locations_empty = torch.empty(
            0, device=self.current_device, dtype=torch.int32
        ).fill_(-1)

        # NOTE: no cache for CPU mode!
        if cache_state.total_cache_hash_size == 0 or self.use_cpu:
            self.register_buffer(
                "lxu_cache_weights",
                torch.zeros(0, 0, device=self.current_device, dtype=dtype),
            )
            # NOTE: make TorchScript work!
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "cache_hash_size_cumsum",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "total_cache_hash_size",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "cache_index_table_map",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "lxu_cache_state",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
            # pyre-fixme[28]: Unexpected keyword argument `persistent`.
            self.register_buffer(
                "lxu_state",
                torch.tensor([0], dtype=torch.int64),
                persistent=False,
            )
            return

        assert cache_load_factor > 0
        element_size = 2 if dtype == torch.float16 else 4
        if cache_sets <= 0:
            total_memory = torch.cuda.get_device_properties(
                self.current_device
            ).total_memory
            free_memory = (
                total_memory
                - torch.cuda.memory_reserved(self.current_device)
                - int(cache_reserved_memory)
            )
            assert free_memory > 0
            cache_sets = (
                int(cache_state.total_cache_hash_size * cache_load_factor) + ASSOC - 1
            ) // ASSOC
            cache_size = cache_sets * ASSOC * element_size * self.max_D_cache
            if cache_size > free_memory:
                cache_sets = (
                    int(1.0 * free_memory / self.max_D_cache / element_size) + ASSOC - 1
                ) // ASSOC
        cache_load_factor = (
            1.0 * cache_sets * ASSOC / int(cache_state.total_cache_hash_size)
        )
        assert cache_sets > 0
        if cache_algorithm == CacheAlgorithm.LFU:
            assert cache_sets < 2 ** 24 - 1
        cache_size = cache_sets * 32 * element_size * self.max_D_cache
        logging.info(
            f"Using on-device cache with admission algorithm "
            f"{cache_algorithm}, {cache_sets} sets, "
            f"load_factor: {cache_load_factor : .3f}, "
            f"{cache_size / 1024.0 / 1024.0 / 1024.0 : .2f}GB"
        )

        self.total_cache_hash_size = cache_state.total_cache_hash_size
        self.register_buffer(
            "cache_hash_size_cumsum",
            torch.tensor(
                cache_state.cache_hash_size_cumsum,
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "cache_index_table_map",
            torch.tensor(
                cache_state.cache_index_table_map,
                device=self.current_device,
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "lxu_cache_state",
            torch.zeros(
                cache_sets, ASSOC, device=self.current_device, dtype=torch.int64
            ).fill_(-1),
        )
        self.register_buffer(
            "lxu_cache_weights",
            torch.zeros(
                cache_sets * ASSOC,
                self.max_D_cache,
                device=self.current_device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "lxu_state",
            # pyre-fixme[28]: Unexpected keyword argument `size`.
            torch.zeros(
                size=(self.total_cache_hash_size + 1,)
                if cache_algorithm == CacheAlgorithm.LFU
                else (cache_sets, ASSOC),
                device=self.current_device,
                dtype=torch.int64,
            ),
        )
        if cache_algorithm not in (CacheAlgorithm.LFU, CacheAlgorithm.LRU):
            raise ValueError(
                f"cache_algorithm must be {CacheAlgorithm.LRU} "
                f"or {CacheAlgorithm.LFU}"
            )

    def reset_cache_states(self) -> None:
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.numel)[[Named(self, Tensor)],
        #  int], Tensor], Tensor, nn.Module]` is not a function.
        if not self.lxu_cache_weights.numel():
            return
        # pyre-fixme[16]: `SplitTableBatchedEmbeddingBagsCodegen` has no attribute
        #  `lxu_cache_state`.
        self.lxu_cache_state.fill_(-1)
        self.lxu_state.fill_(0)
        self.timestep = 1


# pyre-fixme[13]: Attribute `D_offsets` is never initialized.
# pyre-fixme[13]: Attribute `hash_size_cumsum` is never initialized.
# pyre-fixme[13]: Attribute `weights_offsets` is never initialized.
class DenseTableBatchedEmbeddingBagsCodegen(nn.Module):
    """
    Table-batched version of nn.EmbeddingBag(sparse=False)
    """

    weights: Tensor
    weights_offsets: Tensor
    D_offsets: Tensor
    total_D: int
    max_D: int
    hash_size_cumsum: Tensor
    total_hash_size_bits: int
    embedding_specs: List[Tuple[int, int]]

    def __init__(
        self,
        embedding_specs: List[Tuple[int, int]],  # tuple of (rows, dims)
        feature_table_map: Optional[List[int]] = None,  # [T]
        pooling_mode: PoolingMode = PoolingMode.SUM,
        use_cpu: bool = False,
    ) -> None:  # noqa C901  # tuple of (rows, dims,)
        super(DenseTableBatchedEmbeddingBagsCodegen, self).__init__()

        self.pooling_mode = pooling_mode

        self.current_device: torch.device = (
            torch.device("cpu") if use_cpu else torch.cuda.current_device()
        )

        self.embedding_specs = embedding_specs
        (rows, dims) = zip(*embedding_specs)
        T_ = len(self.embedding_specs)
        assert T_ > 0

        feature_table_map = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        T = len(feature_table_map)
        assert T_ <= T
        D_offsets = [dims[t] for t in feature_table_map]
        D_offsets = [0] + list(accumulate(D_offsets))
        self.total_D = D_offsets[-1]
        self.max_D = max(dims)
        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, device=self.current_device, dtype=torch.int32),
        )
        assert self.D_offsets.numel() == T + 1

        hash_size_cumsum = [0] + list(accumulate(rows))
        self.total_hash_size_bits = int(log2(float(hash_size_cumsum[-1])) + 1)
        # The last element is to easily access # of rows of each table by
        # hash_size_cumsum[t + 1] - hash_size_cumsum[t]
        hash_size_cumsum = [hash_size_cumsum[t] for t in feature_table_map] + [
            hash_size_cumsum[-1]
        ]
        self.register_buffer(
            "hash_size_cumsum",
            torch.tensor(
                hash_size_cumsum, device=self.current_device, dtype=torch.int64
            ),
        )
        weights_offsets = [0] + list(
            accumulate([row * dim for (row, dim) in embedding_specs])
        )
        self.weights = nn.Parameter(
            torch.randn(
                weights_offsets[-1],
                device=self.current_device,
            )
        )
        for feature in range(T):
            t = feature_table_map[feature]
            row, dim = embedding_specs[t]
            if (
                self.weights[weights_offsets[t] : weights_offsets[t + 1]].numel()
                != row * dim
            ):
                logging.info(
                    f"row {row} dim {dim} feature {feature} t {t} {self.weights[weights_offsets[t] : weights_offsets[t + 1]].numel()}"
                )
            assert (
                self.weights[weights_offsets[t] : weights_offsets[t + 1]].numel()
                == row * dim
            )
            assert self.hash_size_cumsum[feature] == sum(
                row for (row, _) in embedding_specs[:t]
            )

        self.weights_physical_offsets: List[int] = weights_offsets
        weights_offsets = [weights_offsets[t] for t in feature_table_map]
        self.register_buffer(
            "weights_offsets",
            torch.tensor(
                weights_offsets, device=self.current_device, dtype=torch.int64
            ),
        )

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
    ) -> Tensor:
        (indices, offsets) = indices.long(), offsets.long()
        return torch.ops.fb.dense_embedding_codegen_lookup_function(
            dev_weights=self.weights,
            weights_offsets=self.weights_offsets,
            D_offsets=self.D_offsets,
            total_D=self.total_D,
            max_D=self.max_D,
            hash_size_cumsum=self.hash_size_cumsum,
            total_hash_size_bits=self.total_hash_size_bits,
            indices=indices,
            offsets=offsets,
            pooling_mode=self.pooling_mode,
            indice_weights=per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        )

    @torch.jit.export
    def split_embedding_weights(self) -> List[Tensor]:
        """
        Returns a list of weights, split by table
        """
        splits = []
        for t, (rows, dim) in enumerate(self.embedding_specs):
            offset = self.weights_physical_offsets[t]
            splits.append(
                self.weights.detach()[offset : offset + rows * dim].view(rows, dim)
            )
        return splits

    def init_embedding_weights_uniform(self, min_val: float, max_val: float) -> None:
        splits = self.split_embedding_weights()
        for param in splits:
            param.uniform_(min_val, max_val)


class SequenceEmbeddingCodegen(SplitTableBatchedEmbeddingBagsCodegen):
    """
    This class wraps around SplitTableBatchedEmbeddingBagsCodegen to get
    sequence embedding op: nn.EmbeddingBag(sparse=True)
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        # assert T == 1
        assert "embedding_specs" in kwargs
        assert len(kwargs["embedding_specs"]) == 1
        super(SequenceEmbeddingCodegen, self).__init__(
            **kwargs,
        )

    # @torch.jit.ignore
    def forward(
        self,
        indices: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
    ) -> Tensor:
        offsets = torch.arange(
            0,
            indices.numel() + 1,
            device=indices.device,
            dtype=torch.int64,
        )
        return super(SequenceEmbeddingCodegen, self).forward(
            indices,
            offsets,
            per_sample_weights,
            feature_requires_grad,
        )


class DenseSequenceEmbeddingCodegen(DenseTableBatchedEmbeddingBagsCodegen):
    """
    This class wraps around DenseTableBatchedEmbeddingBagsCodegen to get
    sequence embedding op, nn.EmbeddingBag(sparse=False)
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        # assert T == 1
        assert "embedding_specs" in kwargs
        assert len(kwargs["embedding_specs"]) == 1
        super(DenseSequenceEmbeddingCodegen, self).__init__(
            **kwargs,
        )

    # @torch.jit.ignore
    def forward(
        self,
        indices: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
    ) -> Tensor:
        offsets = torch.arange(
            0,
            indices.numel() + 1,
            device=indices.device,
            dtype=torch.int64,
        )
        return super(DenseSequenceEmbeddingCodegen, self).forward(
            indices,
            offsets,
            per_sample_weights,
            feature_requires_grad,
        )

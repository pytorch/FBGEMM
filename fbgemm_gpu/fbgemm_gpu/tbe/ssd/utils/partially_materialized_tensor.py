# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import functools
from typing import Optional, Union

import torch

_HANDLED_FUNCTIONS = {}


def implements(torch_function):
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class PartiallyMaterializedTensor:
    """
    A tensor-like object that represents a partially materialized tensor in memory.

    Caller can use `narrow()` to get a view of the backing storage,
    or use `full_tensor()` to get the full tensor (this could OOM).
    """

    def __init__(self, wrapped, is_virtual: bool = False) -> None:
        """
        Ensure caller loads the module before creating this object.

        ```
        load_torch_module(
            "//deeplearning/fbgemm/fbgemm_gpu:ssd_split_table_batched_embeddings"
        )
        ```

        Args:

            wrapped: torch.classes.fbgemm.KVTensorWrapper
        """
        self._wrapped = wrapped
        self._is_virtual = is_virtual
        self._requires_grad = False

    @property
    def wrapped(self):
        """
        Get the wrapped extension class for C++ interop.
        """
        return self._wrapped

    @property
    def is_virtual(self):
        """
        Indicate whether PMT is a virtual tensor.
        This indicator is needed for checkpoint or publish.
        They need to know wheether it is PMT for kvzch or for normal emb table
        for kvzch, checkpoint and publish need to call all-gather to recalculate the correct
        metadata of the ShardedTensor
        """
        return self._is_virtual

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](cls, *args, **kwargs)

    @implements(torch.narrow)
    def narrow(self, dim: int, start: int, length: int) -> torch.Tensor:
        """
        This loads a narrowed view of the backing storage.

        Returns:
            a torch tensor
        """
        return self._wrapped.narrow(dim, start, length)

    def set_weights_and_ids(self, weights: torch.Tensor, ids: torch.Tensor) -> None:
        self._wrapped.set_weights_and_ids(weights, ids)

    def get_weights_by_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self._wrapped.get_weights_by_ids(ids)

    def full_tensor(self) -> torch.Tensor:
        """
        This loads the full tensor into memory (may OOM).

        Returns:
            a torch tensor
        """
        return self.narrow(0, 0, self.size(0))

    @implements(torch.detach)
    def detach(self) -> PartiallyMaterializedTensor:
        self._requires_grad = False
        return self

    def to(self, *args, **kwargs) -> PartiallyMaterializedTensor:
        return self

    def is_floating_point(self):
        # this class only deals with embedding vectors
        return True

    @implements(torch._has_compatible_shallow_copy_type)
    def _has_compatible_shallow_copy_type(*args, **kwargs):
        return False

    def requires_grad_(self, requires_grad=True) -> PartiallyMaterializedTensor:
        self._requires_grad = requires_grad
        return self

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def grad(self) -> Optional[torch.Tensor]:
        return None

    @property
    def is_leaf(self) -> bool:
        return True

    @property
    def shape(self) -> torch.Size:
        """
        Shape of the full tensor.
        """
        return torch.Size(self._wrapped.shape)

    def size(self, dim: Optional[int] = None) -> Union[int, torch.Size]:
        sz = self.shape
        if dim is None:
            return sz
        if dim >= len(sz) or dim < 0:
            raise IndexError(
                f"Dimension out of range (expected to be {len(sz)}, but got {dim})"
            )
        return sz[dim]

    def is_contiguous(self):
        return True

    def is_pinned(self):
        return False

    @property
    def dtype(self) -> torch.dtype:
        mapping = {"c10::Half": "half"}
        dtype_str: str = self._wrapped.dtype_str
        dtype_str = mapping.get(dtype_str, dtype_str)

        dtype = getattr(torch, dtype_str)
        assert isinstance(dtype, torch.dtype)
        return dtype

    @property
    def device(self) -> torch.device:
        device_str: str = self._wrapped.device_str
        device = torch.device(device_str)
        assert isinstance(device, torch.device)
        return device

    @property
    def layout(self) -> torch.layout:
        pass
        layout_str_mapping = {
            "SparseCsr": "sparse_csr",
            "Strided": "strided",
            "SparseCsr": "sparse_csr",
            "SparseCsc": "sparse_csc",
            "Jagged": "jagged",
        }
        layout_str: str = self._wrapped.layout_str
        layout_str = layout_str_mapping[layout_str]
        layout = getattr(torch, layout_str)
        assert isinstance(layout, torch.layout)
        return layout

    @property
    def __class__(self):
        # this is a hack to avoid assertion error in torch.nn.Module.register_parameter()
        return torch.nn.Parameter

    @property
    def grad_fn(self):
        return None

    def view(self, *args, **kwargs):
        return self

    def is_meta(*args, **kwargs):
        return False

    def copy_(self, src, non_blocking=False):
        # noop
        pass

    def numel(self):
        return torch.tensor(self.shape).prod().item()

    def nelement(self):
        return torch.tensor(self.shape).prod().item()

    def element_size(self):
        return torch.tensor([], dtype=self.dtype).element_size()

    def __deepcopy__(self, memo):
        # torch.classes.fbgemm.KVTensorWrapper doesn't support deepcopy
        new_obj = PartiallyMaterializedTensor(self._wrapped)
        memo[id(self)] = new_obj
        return new_obj

    def required_grad(self) -> bool:
        return True

    @property
    def is_quantized(self) -> bool:
        return False

    @implements(torch.equal)
    def __eq__(self, tensor1, tensor2, **kwargs):
        if not isinstance(tensor2, PartiallyMaterializedTensor):
            return False

        return torch.equal(tensor1.full_tensor(), tensor2.full_tensor())

    def __hash__(self):
        return id(self)

    @property
    def is_mps(self):
        return False

    @property
    def is_sparse(self):
        return False

    @implements(torch.isclose)
    def isclose(self, tensor1, tensor2, rtol=1e-05, atol=1e-08, equal_nan=False):
        return torch.isclose(
            tensor1.full_tensor(),
            tensor2.full_tensor(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

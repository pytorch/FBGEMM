# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

    def __init__(self, wrapped) -> None:
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

    @property
    def wrapped(self):
        """
        Get the wrapped extension class for C++ interop.
        """
        return self._wrapped

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

    def full_tensor(self) -> torch.Tensor:
        """
        This loads the full tensor into memory (may OOM).

        Returns:
            a torch tensor
        """
        return self.narrow(0, 0, self.size(0))

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

    @property
    def dtype(self) -> torch.dtype:
        dtype_str: str = self._wrapped.dtype_str()
        dtype = getattr(torch, dtype_str)
        assert isinstance(dtype, torch.dtype)
        return dtype

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
import triton
import triton.language as tl


@triton.jit
# fmt: off
def triton_add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr) -> None:
# fmt: on  # noqa E115

    # We use a 1D launch grid so axis is 0.
    pid = tl.program_id(axis=0)

    # Compute the offsets in BLOCK_SIZE chunks.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements

    # Load x and y from DRAM.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Sum and write back to DRAM.
    output = x + y
    tl.store(z_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Pre-allocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    # Create the SPMD launch grid.  It can be either Tuple[int], or
    # Callable(metaparameters) -> Tuple[int].  In this case, we use a 1D grid
    # where the size is the number of blocks:
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731

    # Launch the kernel.
    #
    # Each torch.tensor object is implicitly converted into a pointer to its
    # first element.
    #
    # `triton.jit`'ed functions can be indexed with a launch grid to obtain a
    # callable GPU kernel.
    #
    # Pass meta-parameters as keywords arguments.
    # pyre-ignore
    triton_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been
    # called, the kernel is still running asynchronously at this point.
    return output


@unittest.skip(
    "Example test intermittently fails with the following unknown error: "
    "SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats",
)
class TestTriton(unittest.TestCase):
    def test_triton_example(self) -> None:
        size = 98432
        X = torch.rand(size, device="cuda")
        Y = torch.rand(size, device="cuda")

        torch.testing.assert_close(triton_add(X, Y).cpu(), (X + Y).cpu())

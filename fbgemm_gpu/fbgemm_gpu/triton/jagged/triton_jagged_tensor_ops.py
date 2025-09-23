#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[6]

from typing import Optional, Union

import torch
import triton  # @manual
import triton.language as tl  # @manual
from torch._tensor import Tensor


@triton.jit
def jagged_jagged_elementwise_arithmetic_ops(
    # pyre-fixme[2]: Parameter must be annotated.
    x_ptr,  # x_ptr and y_ptr is pointer of jagged tensor value
    # pyre-fixme[2]: Parameter must be annotated.
    y_ptr,
    M: tl.constexpr,  # M and N would be size of the tensor with (M , N)
    N: tl.constexpr,
    stride_row: tl.constexpr,  # shared row stride for tensor
    stride_col: tl.constexpr,  # shared colume stride for tensor
    # pyre-fixme[2]: Parameter must be annotated.
    output,
    thread_block_row_size: tl.constexpr,  # row and colume size of current thread block with size (thread_block_row_size * thread_block_col_size)
    thread_block_col_size: tl.constexpr,
    ops_func: tl.constexpr,  # function use for calculation either add or multiplication
) -> None:
    pid = tl.program_id(0)
    # number of col group need for total N col
    num_group_n = (N + thread_block_col_size - 1) // thread_block_col_size
    # pid position in col perspective in range(0,num_group_n)
    pid_n = pid % num_group_n
    # pid position in row perspective since everytime row increase when we have num_group_n iteration
    pid_m = pid // num_group_n

    offset_m = pid_m * thread_block_row_size + tl.arange(0, thread_block_row_size)
    offset_n = pid_n * thread_block_col_size + tl.arange(0, thread_block_col_size)
    mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    offset = offset_m[:, None] * stride_row + offset_n[None, :] * stride_col

    x_ptr += offset
    y_ptr += offset

    x = tl.load(x_ptr, mask=mask)
    y = tl.load(y_ptr, mask=mask)

    if ops_func == "add":
        z = tensor_elementwise_add(x, y)
    else:
        z = tensor_elementwise_mul(x, y)

    output += offset
    tl.store(output, z, mask=mask)


@triton.jit
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def tensor_elementwise_add(x, y):
    return x + y


@triton.jit
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def tensor_elementwise_mul(x, y):
    return x * y


def triton_jagged_add_jagged(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    # x and y need to have same shape to do addition
    assert x.shape == y.shape

    thread_block_row_size = 32
    thread_block_col_size = 32

    # x and y would a tensor with same dimension (M,N)
    M, N = x.shape

    output = torch.empty((M, N), device="cuda", dtype=x.dtype)

    # pyre-fixme[53]: Captured variable `M` is not annotated.
    # pyre-fixme[53]: Captured variable `N` is not annotated.
    # pyre-fixme[53]: Captured variable `thread_block_col_size` is not annotated.
    # pyre-fixme[53]: Captured variable `thread_block_row_size` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def grid(META):
        return (
            triton.cdiv(M, thread_block_row_size)
            * triton.cdiv(N, thread_block_col_size),
        )

    jagged_jagged_elementwise_arithmetic_ops[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        output,
        thread_block_row_size,
        thread_block_col_size,
        ops_func="add",
    )

    return output


def triton_jagged_mul_jagged(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    # x and y need to have same shape to do addition
    assert x.shape == y.shape

    thread_block_row_size = 32
    thread_block_col_size = 32
    # x and y would a tensor with same dimension (M,N)
    M, N = x.shape

    output = torch.empty((M, N), device="cuda", dtype=x.dtype)

    # pyre-fixme[53]: Captured variable `M` is not annotated.
    # pyre-fixme[53]: Captured variable `N` is not annotated.
    # pyre-fixme[53]: Captured variable `thread_block_col_size` is not annotated.
    # pyre-fixme[53]: Captured variable `thread_block_row_size` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def grid(META):
        return (
            triton.cdiv(M, thread_block_row_size)
            * triton.cdiv(N, thread_block_col_size),
        )

    jagged_jagged_elementwise_arithmetic_ops[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        output,
        thread_block_row_size,
        thread_block_col_size,
        ops_func="mul",
    )

    return output


# with bmm([B * H , 1 , N] , [B*H , N , D])
# Each kernel function dealing with matmul of (1,N) * (N,D)
@triton.jit
def triton_batched_dense_vec_jagged_2d_matmul(
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_tensor_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_offset,
    thread_block_col_size: tl.constexpr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    D,
    H: tl.constexpr,
    # pyre-fixme[2]: Parameter must be annotated.
    output_ptr,
) -> None:

    pid = tl.program_id(0)

    # number of kernel need for with matrix (N,D) calculated by D // thread_block_col_size
    GRID_DIM_COL = (D + thread_block_col_size - 1) // thread_block_col_size

    # current output row index
    output_row_idx = pid // GRID_DIM_COL

    # current jagged tensor offset index
    jagged_offset_id = output_row_idx // H

    # current index with D reference since the real shape of jagged values is [B , N , H * D]
    D_refer_idx = output_row_idx % H

    # current part of [N * D] id
    group_id = pid % GRID_DIM_COL

    # size of tile
    offset = group_id * thread_block_col_size + tl.arange(0, thread_block_col_size)

    # begin index and end index of values
    begin = tl.load(jagged_offset + jagged_offset_id)
    end = tl.load(jagged_offset + (jagged_offset_id + 1))

    # update each pointer to the correct address
    dense_ptr += output_row_idx * dense_row_stride
    jagged_tensor_ptr += begin * jagged_value_row_stride + D_refer_idx * D
    output_ptr += D * output_row_idx

    # Number of row each kernel will go through
    num_row = tl.minimum(end - begin, dense_row_stride)

    # accumulation variable use for matmul
    acc = tl.zeros((thread_block_col_size,), dtype=tl.float32)
    mask = offset < D
    for i in range(num_row):
        val1 = tl.load(dense_ptr + i)
        val2 = tl.load(jagged_tensor_ptr + offset, mask=mask, other=0.0)
        result = val1 * val2
        acc += result
        jagged_tensor_ptr += jagged_value_row_stride

    tl.store(output_ptr + offset, acc, mask=mask)


# torch.bmm refer https://pytorch.org/docs/stable/generated/torch.bmm.html
# Operation that take dense as format [B * H , N] where N is the max_length in the logical representation we treat dense like [B * H , 1 , N]
# and 2D jagged tensor with format values format [B , N , H * D] in the logical representation we treat values like [B * H , N , D]
# in the 2D jagged tensor case offset will be tensor instead of list of tensor
# create output dense with shape [B * H , 1 , D]
# dense * jagged_tesnor = output_dense -> [B * H , 1 , N] * [B * H , N , D] = [B * H , 1 , D]
def batched_dense_vec_jagged_2d_matmul(
    dense: torch.Tensor,
    values: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    B = offset.size(0) - 1
    H = dense.size(0) // B
    D = values.size(-1) // H
    thread_block_col_size = 32

    output_dense = torch.empty((B * H, D), device="cuda", dtype=values.dtype)

    # number of thread block need for jagged tensor with [B * H , N , D]
    # pyre-fixme[53]: Captured variable `B` is not annotated.
    # pyre-fixme[53]: Captured variable `D` is not annotated.
    # pyre-fixme[53]: Captured variable `H` is not annotated.
    # pyre-fixme[53]: Captured variable `thread_block_col_size` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def grid(META):
        return (B * H * triton.cdiv(D, thread_block_col_size),)

    triton_batched_dense_vec_jagged_2d_matmul[grid](
        values,
        dense,
        offset,
        thread_block_col_size,
        dense.stride(0),
        values.stride(0),
        D,
        H,
        output_dense,
    )

    return output_dense


# each kernel will handle the conversion of one jagged tensor offset range to corresponding dense index
@triton.jit
def triton_jagged_to_dense(
    # only constexpr annotations support in triton now
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_indices_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_col_stride,  # stride of output dense with dimension (z,y,x)
    # pyre-fixme[2]: Parameter must be annotated.
    dense_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_matrix_stride,
    JAGGED_DIM: tl.constexpr,  # number of dimension of jagged tensor
    thread_block_row_size: tl.constexpr,
    thread_block_col_size: tl.constexpr,
    operation_function: tl.constexpr,  # fusion arithmetic operation function and it's input dense
    # pyre-fixme[2]: Parameter must be annotated.
    operation_dense,
) -> None:
    pid = tl.program_id(0)

    # begin index and end index of jagged tensor Values
    begin = tl.load(jagged_offsets_ptr + pid)
    end = tl.load(jagged_offsets_ptr + (pid + 1))

    # adjust the address of the jagged tensor Values to the correct address
    jagged_value_ptr += begin * jagged_value_row_stride

    # if it's 2D (or 1D) Jagged tensor we can direct use the offset in offsets ( since there is only one offset )
    # else we actually need to use the preprocess index to found the correct address of dense
    if JAGGED_DIM > 2:
        # read the index for current kernel
        dense_indice = tl.load(dense_indices_ptr + pid)

        # if the dense_indice is -1 which mean it's a truncation case
        # in that case we don't need to do anything since the dense
        # initialize with padded value
        if dense_indice == -1:
            return

        # adjust the address of output dense ptr to the correct address
        output_dense_ptr += dense_indice

        # also need to update the operation function if exist
        # notice dense_indice of two is same because we assume
        # the two dense + dense are same size
        if operation_function is not None:
            operation_dense += dense_indice
    else:
        output_dense_ptr += pid * dense_matrix_stride

        if operation_function is not None:
            operation_dense += pid * dense_matrix_stride

    offset_row = tl.arange(0, thread_block_row_size)

    # boundary need for the mask since it could be dense's size smaller than jagged tensor or revert case
    N = tl.minimum(dense_row_stride, jagged_value_row_stride)
    M = tl.minimum(dense_matrix_stride // dense_row_stride, end - begin)

    for _i in range(begin, end, thread_block_row_size):
        offset_col = tl.arange(0, thread_block_col_size)
        block_offset = (
            offset_row[:, None] * dense_row_stride
            + offset_col[None, :] * dense_col_stride
        )
        for _j in range(0, N, thread_block_col_size):
            mask = (offset_row[:, None] < M) & (offset_col[None, :] < N)
            jagged_val = tl.load(jagged_value_ptr + block_offset, mask=mask, other=0)

            # if there is some arithmetic operation we do the fusion computation
            if operation_function is not None:
                val1 = jagged_val
                val2 = tl.load(operation_dense + block_offset, mask=mask, other=0)
                # do the arithmetic operation
                if operation_function == "add":
                    jagged_val = tensor_elementwise_add(val1, val2)
                else:
                    jagged_val = tensor_elementwise_mul(val1, val2)

            # store the result
            tl.store(output_dense_ptr + block_offset, jagged_val, mask=mask)

            # update the block offset
            offset_col += thread_block_col_size
            block_offset += thread_block_col_size
        offset_row += thread_block_row_size


# This function will handle the 2d Jagged Tensor to Dense operation
# each kernel will go through all the element in each 2D tensor in
# Dense ( Notice that since it's 2d jagged tensor dense will be 3D ).
# Each kernel will check if the current value in 2d tensor is in
# range or out of range. If in the range of Jagged Tensor, it will load
# corresponding value, otherwise it will load padded value into dense.
# On the other hand, in the function triton_jagged_to_dense, we are
# only able to fill the value from jagged tensor to corresponding dense
# but we are not be able to fill the dense with padded value in kernel.
# therefore in pervious function, we fill dense with padded value first
# then load corresponding value. Instead this function can directly
# fill the value in kernel to avoid extra latency.
@triton.jit
def triton_jagged_to_dense_optimization_2d(
    # pyre-fixme[2]: Parameter must be annotated.
    input_jagged_values_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    input_jagged_offset_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    input_jagged_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_row_stride,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_matrix_stride,
    thread_block_row_size: tl.constexpr,
    thread_block_col_size: tl.constexpr,
    # pyre-fixme[2]: Parameter must be annotated.
    padded_value,
    operation_function: tl.constexpr,
    # pyre-fixme[2]: Parameter must be annotated.
    operation_dense,
) -> None:
    pid = tl.program_id(0)

    # Current corresponding offset indice
    offset_idx = pid

    # begin index and end index of jagged tensor Values
    begin = tl.load(input_jagged_offset_ptr + offset_idx)
    end = tl.load(input_jagged_offset_ptr + offset_idx + 1)

    # row size of current sub tensor
    cur_jagged_tensor_row_size = end - begin

    # update dense and jagged tensor Values to corresponding address
    output_dense_ptr += pid * output_dense_matrix_stride
    input_jagged_values_ptr += begin * input_jagged_row_stride

    # also need to update the operation function if exist
    # notice dense_indice of two is same because we assume
    # the two dense + dense are same size
    if operation_function is not None:
        operation_dense += pid * output_dense_matrix_stride

    # jagged tensor row block
    offset_row = tl.arange(0, thread_block_row_size)

    # dense row and col block
    # notice jagged tensor and dense share same col block since embedding dimension is same
    dense_col_size = output_dense_row_stride
    dense_row_size = output_dense_matrix_stride // output_dense_row_stride

    for _i in range(0, dense_row_size, thread_block_row_size):
        offset_col = tl.arange(0, thread_block_col_size)
        block_offset = (
            offset_row[:, None] * output_dense_row_stride + offset_col[None, :]
        )

        for _j in range(0, dense_col_size, thread_block_col_size):

            # create mask for dense and jagged tensor for boundary check
            dense_mask = (offset_row[:, None] < dense_row_size) & (
                offset_col[None, :] < dense_col_size
            )
            jagged_mask = (offset_row[:, None] < cur_jagged_tensor_row_size) & (
                offset_col[None, :] < input_jagged_row_stride
            )

            # get value from jagged tesnor
            jagged_val = tl.load(
                input_jagged_values_ptr + block_offset,
                mask=jagged_mask,
                other=padded_value,
            )

            # do fusion operation if need
            if operation_function is not None:
                operation_dense_val = tl.load(
                    operation_dense + block_offset, mask=dense_mask, other=0.0
                )
                jagged_val = operation_function(operation_dense_val, jagged_val)

            # load value into empty dense
            tl.store(output_dense_ptr + block_offset, jagged_val, mask=dense_mask)

            # update each block
            offset_col += thread_block_col_size
            block_offset += thread_block_col_size
        offset_row += thread_block_row_size


# this function parse the jagged tensor offsets to corresponding dense index position
# to see the detail of it see the quip note : https://fb.quip.com/gnzpA7d13vqO
# the FBGEMM implementation refer : https://www.internalfb.com/code/fbsource/[308212b2902c3182edcb5b204768321e032e8175]/fbcode/deeplearning/fbgemm/fbgemm_gpu/src/jagged_tensor_ops.cu?lines=280
# In FBGEMM it was computed by GPU but in triton currently has some compilation issue so we use CUP computation method as workaround
# However in real-world case if we only dealing with 2d jagged tensor we don't need to use this function at all
def _jagged_offsets_to_dense_indice(
    offsets: list[torch.Tensor], dense_strides: list[int], dense_sizes: list[int]
) -> torch.Tensor:

    output_offset = torch.zeros(len(offsets[-1]) - 1, device="cpu", dtype=torch.int32)

    offsets_cpu = []

    for offset in offsets:
        offsets_cpu.append(offset.cpu())

    for i in range(0, len(offsets_cpu[-1]) - 1):
        idx = i
        result = 0

        # flag to check if current offset is in the range of dense
        in_range = True
        for j in range(len(offsets_cpu) - 2, -1, -1):
            left = 0
            right = offsets_cpu[j].size(0)

            # binary search found the corresponding offset group of current index
            while left < right:
                mid = left + (right - left) // 2

                if offsets_cpu[j][mid] > idx:
                    right = mid
                else:
                    left = mid + 1

            cur_val = idx - offsets_cpu[j][left - 1]

            if dense_sizes and cur_val >= dense_sizes[j + 1]:
                in_range = False
                break

            result += cur_val * dense_strides[j + 1]
            idx = left - 1

        if in_range:
            result += idx * dense_strides[0]

            # another out of output dense range case
            if dense_sizes and idx > dense_sizes[0]:
                result = -1
            output_offset[i] = result
        else:
            output_offset[i] = -1

    return output_offset.cuda()


# transfer jagged tensor to dense for referring the quip note for wiki : https://fb.quip.com/gnzpA7d13vqO
# currently when doing the conversion if certain part of dense are not load from the jagged tensor Values
# it will be skiped. Which mean we initialize the tensor with padded value instead of fill it with padded
# value while conversion. Currently optimization approach implementation in triton faced some issue with
# LLVM compile issue but will look a work around when make a comparsion with multiple dimension of
# jagged tensot. However if currently we only dealing with 2d jagged tensor in real-world case this should
# not be affected at all
def jagged_to_dense(
    jagged_values: torch.Tensor,
    jagged_offsets: list[torch.Tensor],
    jagged_max_lengths: list[int],
    padding_value: float = 0.0,  # padding value currently use 0.0 as default value
    operation_function: Union[
        str, None
    ] = None,  # fusioned operation currently could be add or multiplication
    operation_dense: Union[
        torch.Tensor, None
    ] = None,  # dense to make the add/mul with the output dense
) -> torch.Tensor:
    outer_dense_size = len(jagged_offsets[0]) - 1
    inner_dense_size = jagged_values.size(-1)

    # dimension of jagged tensor
    JAGGED_DIM = len(jagged_offsets) + 1

    output_dense = None

    # fill the padded value into dense if is multiple dimension
    # other wise create empty dense
    # this is for avoid multiple dimension cases
    # it can create compile error if we going to fill the padding
    # value inside of kernel function
    if JAGGED_DIM > 2:
        output_dense = torch.full(
            ((outer_dense_size,) + tuple(jagged_max_lengths) + (inner_dense_size,)),
            padding_value,
            device="cuda",
            dtype=jagged_values.dtype,
        )
    else:
        output_dense = torch.empty(
            ((outer_dense_size,) + tuple(jagged_max_lengths) + (inner_dense_size,)),
            device="cuda",
            dtype=jagged_values.dtype,
        )

    thread_block_row_size = 32
    thread_block_col_size = 32

    grid = (len(jagged_offsets[-1]) - 1,)

    # dense index in address perspective
    dense_indices = None

    # if dimension of jagged tensor ( which is number of offset ) we will need calculated the related dense index referring to jagged offsets
    if JAGGED_DIM > 2:
        dense_indices = _jagged_offsets_to_dense_indice(
            jagged_offsets,
            output_dense.stride()[:-2],
            output_dense.size()[:-2],
        )

    # dense stride for each column, row, and matrix
    dense_col_stride = output_dense.stride(-1)
    dense_row_stride = output_dense.stride(-2)
    dense_matrix_stride = output_dense.stride(-3)

    if JAGGED_DIM > 2:
        triton_jagged_to_dense[grid](
            jagged_values,
            jagged_offsets[-1],
            jagged_values.stride(0),
            output_dense,
            dense_indices,
            dense_col_stride,
            dense_row_stride,
            dense_matrix_stride,
            JAGGED_DIM,
            thread_block_row_size,
            thread_block_col_size,
            operation_function=operation_function,
            operation_dense=operation_dense,
        )
    else:
        grid = (output_dense.size(0),)
        triton_jagged_to_dense_optimization_2d[grid](
            jagged_values,
            jagged_offsets[-1],
            jagged_values.stride(0),
            output_dense,
            dense_row_stride,
            dense_matrix_stride,
            thread_block_row_size,
            thread_block_col_size,
            padded_value=padding_value,
            operation_function=operation_function,
            operation_dense=operation_dense,
        )

    return output_dense


# each kernel will handle the conversion of one jagged tensor offset range from corresponding dense index
@triton.jit
def triton_dense_to_jagged(
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_value_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    jagged_offsets_ptr,
    jagged_value_row_stride: int,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dense_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_indices_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_col_stride,  # stride of output dense with dimension (z,y,x)
    dense_row_stride: int,
    # pyre-fixme[2]: Parameter must be annotated.
    dense_matrix_stride,
    JAGGED_DIM: tl.constexpr,  # number of dimension of jagged tensor
    thread_block_row_size: tl.constexpr,
    thread_block_col_size: tl.constexpr,
    operation_function: tl.constexpr,  # fusion arithmetic opeartion function and it's input dense
    # pyre-fixme[2]: Parameter must be annotated.
    operation_jagged_value_ptr,
) -> None:
    pid = tl.program_id(0)

    begin = tl.load(jagged_offsets_ptr + pid)
    end = tl.load(jagged_offsets_ptr + (pid + 1))

    # size of the current value offset range (M , N)
    N = jagged_value_row_stride
    M = end - begin

    dense_boundary_col = dense_row_stride
    # tl.minimum will change the return type cased compile issue
    # in that case use if statement instead
    if N < dense_row_stride:
        dense_boundary_col = N

    dense_boundary_row = tl.minimum(dense_matrix_stride // dense_row_stride, M)

    jagged_value_ptr += begin * jagged_value_row_stride
    if JAGGED_DIM > 2:
        dense_indice = tl.load(dense_indices_ptr + pid)
        # if dense output range we set dense_boundary to -1
        # that mean dense values will not be use with mask
        # since we still need the calculation of fusion step
        # therefore we do not do return here
        if dense_indice == -1:
            dense_boundary_col = -1
        else:
            output_dense_ptr += dense_indice
    else:
        output_dense_ptr += pid * dense_matrix_stride

    if operation_function is not None:
        operation_jagged_value_ptr += begin * jagged_value_row_stride

    offset_row = tl.arange(0, thread_block_row_size)

    for _i in range(begin, end, thread_block_row_size):
        offset_col = tl.arange(0, thread_block_col_size)
        block_offset = (
            offset_row[:, None] * dense_row_stride
            + offset_col[None, :] * dense_col_stride
        )

        for _j in range(0, N, thread_block_col_size):
            dense_mask = (offset_row[:, None] < dense_boundary_row) & (
                offset_col[None, :] < dense_boundary_col
            )
            jagged_mask = (offset_row[:, None] < M) & (offset_col[None, :] < N)
            dense_values = tl.load(
                output_dense_ptr + block_offset, mask=dense_mask, other=0
            )
            if operation_function is not None:
                operation_jagged_value = tl.load(
                    operation_jagged_value_ptr + block_offset, mask=jagged_mask, other=0
                )
                if operation_function == "add":
                    dense_values = tensor_elementwise_add(
                        dense_values, operation_jagged_value
                    )
                else:
                    dense_values = tensor_elementwise_mul(
                        dense_values, operation_jagged_value
                    )
            tl.store(jagged_value_ptr + block_offset, dense_values, mask=jagged_mask)
            offset_col += thread_block_col_size
            block_offset += thread_block_col_size
        offset_row += thread_block_row_size


def dense_to_jagged(
    dense: torch.Tensor,
    jagged_offsets: list[torch.Tensor],
    operation_function: Union[str, None] = None,
    operation_jagged_values: Union[torch.Tensor, None] = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:

    thread_block_row_size = 32
    thread_block_col_size = 32

    if operation_function is None:
        output_jagged_value = torch.empty(
            (jagged_offsets[-1][-1], dense.size(-1)),
            device="cuda",
            dtype=dense.dtype,
        )
    else:
        output_jagged_value = torch.empty(
            # pyre-fixme [16]: Optional type has no attribute `shape`.Pyre
            operation_jagged_values.shape,
            device="cuda",
            dtype=dense.dtype,
        )

    grid = (jagged_offsets[-1].size(0) - 1,)

    JAGGED_DIM = len(jagged_offsets) + 1
    dense_indices = None
    if len(jagged_offsets) > 1:
        dense_indices = _jagged_offsets_to_dense_indice(
            jagged_offsets,
            dense.stride()[:-2],
            dense.size()[:-2],
        )

    # dense stride for each column, row, and matrix
    dense_col_stride = dense.stride(-1)
    dense_row_stride = dense.stride(-2)
    dense_matrix_stride = dense.stride(-3)

    triton_dense_to_jagged[grid](
        output_jagged_value,
        jagged_offsets[-1],
        output_jagged_value.stride(0),
        dense,
        dense_indices,
        dense_col_stride,
        dense_row_stride,
        dense_matrix_stride,
        JAGGED_DIM,
        thread_block_row_size,
        thread_block_col_size,
        operation_function=operation_function,
        operation_jagged_value_ptr=operation_jagged_values,
    )

    return output_jagged_value, jagged_offsets


# jagged_tensor + dense -> dense
def jagged_dense_elementwise_add_dense_output(
    jagged_values: Tensor,
    jagged_offsets: list[Tensor],
    # pyre-fixme[2]: Parameter must be annotated.
    dense,
) -> Tensor:

    # max_length use to build output dense
    # that has same size as input dense
    max_length = dense.size()[1:-1]

    # convert jagged tensor to dense
    converted_dense = jagged_to_dense(jagged_values, jagged_offsets, max_length)

    # add opeartion add two dense with same shape
    # Once it's optimazied we can remove this statement
    # and directly return converted_dense
    return converted_dense + dense


# jagged_tensor + dense -> jagged_tensor
def jagged_dense_elementwise_add_jagged_output(
    jagged_values: Optional[Tensor], jagged_offsets: list[Tensor], dense: Tensor
) -> tuple[Tensor, list[Tensor]]:

    return dense_to_jagged(
        dense,
        jagged_offsets,
        operation_function="add",
        operation_jagged_values=jagged_values,
    )


# jagged_tensor * dense -> jagged_tensor
def jagged_dense_elementwise_mul_jagged_output(
    jagged_values: Optional[Tensor], jagged_offsets: list[Tensor], dense: Tensor
) -> tuple[Tensor, list[Tensor]]:

    return dense_to_jagged(
        dense,
        jagged_offsets,
        operation_function="mul",
        operation_jagged_values=jagged_values,
    )

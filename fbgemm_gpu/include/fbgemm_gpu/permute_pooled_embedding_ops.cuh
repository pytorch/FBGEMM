#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cuda.h>
#include "cub/block/block_reduce.cuh"
#include "cub/device/device_scan.cuh"
#include "./layout_transform_ops.cuh"

namespace at {
Tensor permute_pooled_embs(
    const Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list);
} // namespace at

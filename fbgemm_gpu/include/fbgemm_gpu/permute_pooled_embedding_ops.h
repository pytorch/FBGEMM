// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <ATen/ATen.h>

namespace at {
Tensor permute_pooled_embs(
    const Tensor& pooled_embs, // [B_local][Sum_T_global(D)]
    const Tensor& offset_dim_list,
    const Tensor& permute_list,
    const Tensor& inv_offset_dim_list,
    const Tensor& inv_permute_list);
}

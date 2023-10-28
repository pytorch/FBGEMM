/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

#include "common.h"
#include "fbgemm_gpu/cumem_utils.h"
#include "fbgemm_gpu/enum_utils.h"

namespace fbgemm_gpu {

FBGEMM_GPU_ENUM_CREATE_TAG(uvm)

} // namespace fbgemm_gpu

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <tuple>
#include <unordered_map>

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <torch/torch.h>

#if defined(USE_ROCM)

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"

#if __has_include("ck/utility/flush_icache.hpp")

#include "ck/utility/flush_icache.hpp"

#else

#include <hip/hip_runtime.h>

namespace ck {
// Copied from: https://github.com/ROCm/composable_kernel/blob/rocm-6.2.0/include/ck/utility/flush_icache.hpp
static __global__ void flush_icache() {
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}
} // namespace ck

#endif
namespace fbgemm_gpu {

void flush_icache_ck()
{
    hipDeviceProp_t deviceProps;
    hip_check_error(hipGetDeviceProperties(&deviceProps, 0));
    int32_t gpu_block3 = deviceProps.multiProcessorCount * 60;

    auto stream = at::cuda::getCurrentHIPStream().stream();

    ck::flush_icache<<<dim3(gpu_block3), dim3(64), 0, stream>>>();
    hip_check_error(hipGetLastError());
}

} // namespace fbgemm_gpu

#endif // defined(USE_ROCM)

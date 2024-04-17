# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules/Utilities.cmake)


################################################################################
# PyTorch Dependencies Setup
################################################################################

find_package(Torch REQUIRED)

#
# Toch Cuda Extensions are normally compiled with the flags below. However we
# disabled -D__CUDA_NO_HALF_CONVERSIONS__ here as it caused "error: no suitable
# constructor exists to convert from "int" to "__half" errors in
# gen_embedding_forward_quantized_split_[un]weighted_codegen_cuda.cu
#

set(TORCH_CUDA_OPTIONS
    --expt-relaxed-constexpr -D__CUDA_NO_HALF_OPERATORS__
    # -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__)

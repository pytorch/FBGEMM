/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef FBGEMM_CUB_USE_NAMESPACE

#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX

// CUB_NS_QUALIFIER must be defined alongside CUB_NS_PREFIX/POSTFIX
// (see https://github.com/NVIDIA/cub/pull/350)
#undef CUB_NS_QUALIFIER

#define CUB_NS_PREFIX namespace fbgemm_gpu {
#define CUB_NS_POSTFIX } // namespace fbgemm_gpu
#define CUB_NS_QUALIFIER ::fbgemm_gpu::cub

#endif

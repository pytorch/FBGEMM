/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#undef FBGEMM_GPU_CUB_NS_PREFIX

#ifdef FBGEMM_CUB_USE_NAMESPACE

#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX

// PR https://github.com/NVIDIA/cub/pull/350 introduced breaking change.
// When the CUB_NS_[PRE|POST]FIX macros are set, 
// CUB_NS_QUALIFIER must also be defined to the fully qualified CUB namespace
#if CUB_VERSION >= 101400
#undef CUB_NS_QUALIFIER
#endif

#define FBGEMM_GPU_CUB_NS_PREFIX fbgemm_gpu::

#else

#define FBGEMM_GPU_CUB_NS_PREFIX

#endif

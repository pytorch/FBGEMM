/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#undef FBGEMM_GPU_CUB_NS_PREFIX

#ifdef FBGEMM_CUB_USE_NAMESPACE

#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX

#define FBGEMM_GPU_CUB_NS_PREFIX fbgemm_gpu::

#else

#define FBGEMM_GPU_CUB_NS_PREFIX

#endif

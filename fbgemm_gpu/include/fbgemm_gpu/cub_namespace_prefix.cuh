/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef FBGEMM_CUB_USE_NAMESPACE

#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX

#define CUB_NS_PREFIX namespace fbgemm_gpu {
#define CUB_NS_POSTFIX } // namespace fbgemm_gpu

#endif

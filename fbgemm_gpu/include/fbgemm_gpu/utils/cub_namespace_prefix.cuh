/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef FBGEMM_CUB_USE_NAMESPACE

// Use CUB_WRAPPED_NAMESPACE (CUB >= 1.14, CUDA >= 11.6) to wrap CUB in the
// fbgemm_gpu namespace, avoiding symbol collisions with other CUB users.
#define CUB_WRAPPED_NAMESPACE fbgemm_gpu

#endif

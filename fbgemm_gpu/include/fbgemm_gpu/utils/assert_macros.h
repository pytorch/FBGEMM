/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_KERNEL_ERROR_CHECK(CODE, COND, ERROR_VAL) \
  if (!(COND)) {                                         \
    error_code = CODE;                                   \
    error_value = ERROR_VAL;                             \
    goto kernel_error_handler;                           \
  }

#define FBGEMM_KERNEL_ERROR_THROW(CODE, COND, MSG, ...)                       \
  if (error_code == CODE) {                                                   \
    printf("CUDA Kernel Assertion: " #COND " " #MSG "\n", __VA_ARGS__);       \
    CUDA_KERNEL_ASSERT(false && "Please search for 'CUDA Kernel Assertion'"); \
  }

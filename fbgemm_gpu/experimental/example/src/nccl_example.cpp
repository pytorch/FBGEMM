/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <nccl.h>

namespace fbgemm_gpu::experimental {

void example_nccl_code() {
  ncclComm_t comms[4];
  int devs[4] = {0, 1, 2, 3};
  ncclCommInitAll(comms, 4, devs);

  for (int i = 0; i < 4; i++) {
    ncclCommDestroy(comms[i]);
  }
}

} // namespace fbgemm_gpu::experimental

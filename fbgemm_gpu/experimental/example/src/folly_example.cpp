/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <folly/hash/Checksum.h>
#include <vector>

namespace fbgemm_gpu::experimental {

uint32_t example_folly_code(const std::vector<uint8_t>& buffer) {
  return folly::crc32c(buffer.data(), buffer.size());
}

} // namespace fbgemm_gpu::experimental

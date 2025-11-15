/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./CodeStorage.h"

namespace fbgemm {
namespace CodeStorage {

asmjit::JitRuntime& getRuntime() {
  // JIT Runtime for asmjit, depends on other static variables.
  // Required to prevent initialization order fiasco
  static asmjit::JitRuntime rt;

  return rt;
}

} // namespace CodeStorage
} // namespace fbgemm

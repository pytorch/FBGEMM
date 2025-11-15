/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./GenerateKernel.h" // @manual

namespace fbgemm {

namespace x86 = asmjit::x86;

/**
 * Generate instructions for initializing the C registers to 0 in 32-bit
 * Accumulation kernel.
 */
void initCRegs(x86::Emitter* a, int rowRegs, int colRegs) {
  // Take advantage of implicit zeroing out
  // i.e., zero out xmm and ymm will be zeroed out too
  for (int i = 0; i < rowRegs; ++i) {
    for (int j = 0; j < colRegs; ++j) {
      x86::Vec reg = x86::xmm(unsigned(i * colRegs + j));
      a->vpxor(reg, reg, reg);
    }
  }
}

} // namespace fbgemm

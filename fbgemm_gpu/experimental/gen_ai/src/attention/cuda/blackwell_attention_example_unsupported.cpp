/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>

int main() {
  std::cout << "SKIP: Blackwell attention examples require NVIDIA CUDA; "
               "HIP is unsupported.\n";
  return 0;
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

template <int x>
struct log2_calc_ {
  enum { value = log2_calc_<(x >> 1)>::value + 1 };
};
template <>
struct log2_calc_<0> {
  enum { value = 0 };
};

template <int x>
struct log2_calc {
  enum { value = log2_calc_<(x - 1)>::value };
};

#if 0
template <>
struct log2_calc<0> { enum { value = 0 }; };
template <>
struct log2_calc<1> { enum { value = 0 }; };
#endif

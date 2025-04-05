/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sstream>
#include <string_view>

#if __cplusplus > 201703L && __has_builtin(__builtin_source_location)
#include <source_location>
#else
#include <experimental/source_location>
#endif

namespace fbgemm_gpu::utils {

////////////////////////////////////////////////////////////////////////////////
// Wrapper around std::experimental::source_location
//
// Older versions of Clang have <source_location> but do not have
// __builtin_source_location defined, so we have to fall back to
// <experimental/source_location> instead; see:
//
//    https://youtrack.jetbrains.com/issue/CPP-27965
//    https://github.com/root-project/root/issues/14601
////////////////////////////////////////////////////////////////////////////////

#if __cplusplus > 201703L && __has_builtin(__builtin_source_location)
using source_location = std::source_location;
#else
using source_location = std::experimental::source_location;
#endif

////////////////////////////////////////////////////////////////////////////////
// Source Context
//
// This is a wrapper abstraction around two bits of context information, the
// source location and a summary string.  It is used to generate consistent
// descriptions in log messages around kernel executions.
////////////////////////////////////////////////////////////////////////////////

struct SourceContext {
  const source_location location;
  const std::string_view summary;

  constexpr inline SourceContext(
      const source_location& loc_,
      const std::string_view& sum_) noexcept
      : location(loc_), summary(sum_) {}

  inline const std::string_view description() const noexcept {
    static const std::string cached = [&] {
      std::stringstream ss;
      ss << "[" << location.file_name() << '(' << location.line() << ':'
         << location.column() << ")] [" << summary << "]";
      return ss.str();
    }();

    return cached;
  }
};

} // namespace fbgemm_gpu::utils

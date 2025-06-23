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

////////////////////////////////////////////////////////////////////////////////
// Source Location Import
//
// Handles experiemntal source location import for different versions of C++
////////////////////////////////////////////////////////////////////////////////

#if __cplusplus > 201703L && __has_builtin(__builtin_source_location)
#include <source_location>
#else
#include <experimental/source_location>
#endif

namespace fbgemm_gpu::utils {

////////////////////////////////////////////////////////////////////////////////
// Wrapper Around std::experimental::source_location
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
// This is a wrapper abstraction around some source context information,
// including the source location, template filepath, and summary string.  It is
// used to generate consistent descriptions in log messages around kernel
// executions.
////////////////////////////////////////////////////////////////////////////////

struct SourceContext {
  // The source location
  const source_location location;
  // A summary of the context (usually the kernel name)
  const std::string_view summary;
  // The originating template filepath (for template-generated source files)
  const std::string_view template_;
  // The file descriptor for DSA error reporting (needs to be generated at
  // compile-time)
  const std::string_view dsa_file_descriptor_;

  constexpr inline SourceContext(
      const source_location& loc_,
      const std::string_view& sum_,
      const std::string_view& tmpl_,
      const std::string_view& dsa_) noexcept
      : location(loc_),
        summary(sum_),
        template_(tmpl_),
        dsa_file_descriptor_(dsa_) {}

  inline const std::string description() const noexcept {
    // Generate and cache the description if it hasn't been generated yet
    std::stringstream ss;

    // Append template source file location if it exists
    if (!template_.empty()) {
      ss << "[" << template_ << "] ";
    }

    ss << "[" << location.file_name() << '(' << location.line() << ':'
       << location.column() << ")] [" << summary << "]";

    return ss.str();
  }

  inline SourceContext withSummary(
      const std::string_view& sum_) const noexcept {
    return SourceContext(location, sum_, template_, dsa_file_descriptor_);
  }
};

} // namespace fbgemm_gpu::utils

////////////////////////////////////////////////////////////////////////////////
// Macro create a compile-time concatenation of __TEMPLATE_SOURCE_FILE__ and
// __FILE__
//
// This is used for reporting the template filename into to Torch DSA.  Runtime
// strings cannot be used here because the Torch DSA error reporting mechanism
// is located higher in the stack than where the DSA launch_registry.insert() is
// called, and requires a compile-timme defined char * for the reporting to work
// correctly.
////////////////////////////////////////////////////////////////////////////////

#ifdef __TEMPLATE_SOURCE_FILE__
#define _FBGEMM_DSA_FILESRC_ "[" __TEMPLATE_SOURCE_FILE__ "] " __FILE__
#else
#define _FBGEMM_DSA_FILESRC_ __FILE__
#endif

////////////////////////////////////////////////////////////////////////////////
// Macro to define _FKL_TFILE_ to be __TEMPLATE_SOURCE_FILE__ if it is defined,
// else empty string
////////////////////////////////////////////////////////////////////////////////

#ifdef __TEMPLATE_SOURCE_FILE__
#define _FBGEMM_TFILE_ __TEMPLATE_SOURCE_FILE__
#else
#define _FBGEMM_TFILE_ ""
#endif

////////////////////////////////////////////////////////////////////////////////
// SourceContext Builder Macro
//
// This macro is used to build a SourceContext object for the kernel launcher,
// can be used elsewhere as well.
//
// NOTE: The builder is defined as a macro in specifically in this header file ,
// instead of static class method in source_context.h, so that __FILE__ and
// __TEMPLATE_SOURCE_FILE__ can be correctly expanded to point to actual call
// site.
////////////////////////////////////////////////////////////////////////////////

#define SOURCE_CONTEXT_CURRENT(LABEL)                \
  fbgemm_gpu::utils::SourceContext(                  \
      fbgemm_gpu::utils::source_location::current(), \
      #LABEL,                                        \
      _FBGEMM_TFILE_,                                \
      _FBGEMM_DSA_FILESRC_);

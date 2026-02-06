/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <exception>
#include <sstream>
#include <string>
#include <string_view>

namespace fbgemm {

template <typename... Args>
inline std::string str(const Args&... args) {
  std::ostringstream ss;
  (void)(ss << ... << args);
  return ss.str();
}

class Error : public std::exception {
 public:
  explicit Error(
      std::string_view msg,
      const char* file,
      int line,
      const char* func) {
    what_ = str("[", file, "(", line, ")] [", func, "]: ", msg);
  }

  const char* what() const noexcept override {
    return what_.c_str();
  }

 private:
  std::string what_;
};

namespace detail {

// Helper to construct check message.
// If no extra args, use the default message; otherwise, use str() to build it.
template <typename... Args>
auto fbgemmCheckMsg(const char* defaultMsg, const Args&... args) {
  if constexpr (sizeof...(args) == 0) {
    return defaultMsg;
  } else {
    return ::fbgemm::str(args...);
  }
}

} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// FBGEMM_CHECK macro - throws fbgemm::Error on failure.  Implementation is
// Based on that for the TORCH_CHECK macro.
//
// Usage:
//    FBGEMM_CHECK(should_be_true);  // Default error message
//    FBGEMM_CHECK(x == 0, "Expected x to be 0, but got ", x);
//
// On failure, this macro will raise an exception. It does NOT
// unceremoniously quit the process (unlike assert()).
////////////////////////////////////////////////////////////////////////////////

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define FBGEMM_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define FBGEMM_UNLIKELY(expr) (expr)
#endif

#define FBGEMM_CHECK(cond, ...)                                 \
  if (FBGEMM_UNLIKELY(!(cond))) {                               \
    throw ::fbgemm::Error(                                      \
        ::fbgemm::detail::fbgemmCheckMsg(                       \
            "Expected " #cond " to be true, but got false.  "   \
            "(Could this error message be improved?  If so, "   \
            "please report an enhancement request to FBGEMM.)", \
            ##__VA_ARGS__),                                     \
        __FILE__,                                               \
        __LINE__,                                               \
        __func__);                                              \
  }

} // namespace fbgemm

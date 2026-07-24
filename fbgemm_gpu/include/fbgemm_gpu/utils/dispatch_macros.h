/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
// This is NECESSARY for the PT2_COMPLIANT_TAG macro to work.
#include <torch/library.h>

////////////////////////////////////////////////////////////////////////////////
/// Type Dispatch Functions
///
/// These functions dispatch runtime at::ScalarType values to compile-time
/// types and invoke the given functor with the resolved types as explicit
/// template arguments, e.g.:
///
///   dispatch_emb_cache_types(
///       emb.scalar_type(), cache.scalar_type(), "op_name",
///       [&]<typename emb_t, typename cache_t>() { ... });
////////////////////////////////////////////////////////////////////////////////

namespace fbgemm_gpu {

#if defined(USE_ROCM)
using fp8_e4m3_t = at::Float8_e4m3fnuz;
#else
using fp8_e4m3_t = at::Float8_e4m3fn;
#endif

template <typename F>
decltype(auto) dispatch_emb_cache_types(
    const at::ScalarType emb_type,
    const at::ScalarType cache_type,
    const char* const name,
    F&& f) {
  const auto dispatch_cache = [&]<typename emb_t>() -> decltype(auto) {
    switch (cache_type) {
      case at::ScalarType::Float:
        return f.template operator()<emb_t, float>();
      case at::ScalarType::Half:
        return f.template operator()<emb_t, at::Half>();
      default:
        TORCH_CHECK(
            false,
            name,
            " not implemented for cache_t '",
            toString(cache_type),
            "'");
    }
  };
  switch (emb_type) {
    case at::ScalarType::Float:
      return dispatch_cache.template operator()<float>();
    case at::ScalarType::Half:
      return dispatch_cache.template operator()<at::Half>();
    case c10::CppTypeToScalarType<fp8_e4m3_t>::value:
      return dispatch_cache.template operator()<fp8_e4m3_t>();
    default:
      TORCH_CHECK(
          false, name, " not implemented for emb_t '", toString(emb_type), "'");
  }
}

template <typename F>
decltype(auto) dispatch_emb_cache_output_types(
    const at::ScalarType emb_type,
    const at::ScalarType cache_type,
    const at::ScalarType output_type,
    const char* const name,
    F&& f) {
  const auto dispatch_emb_cache = [&]<typename output_t>() -> decltype(auto) {
    return dispatch_emb_cache_types(
        emb_type,
        cache_type,
        name,
        [&]<typename emb_t, typename cache_t>() -> decltype(auto) {
          return f.template operator()<emb_t, cache_t, output_t>();
        });
  };
  switch (output_type) {
    case at::ScalarType::Half:
      return dispatch_emb_cache.template operator()<at::Half>();
    case at::ScalarType::Float:
      return dispatch_emb_cache.template operator()<float>();
    case at::ScalarType::BFloat16:
      return dispatch_emb_cache.template operator()<at::BFloat16>();
    default:
      TORCH_CHECK(
          false,
          name,
          " not implemented for output_t '",
          toString(output_type),
          "'");
  }
}

template <typename F>
decltype(auto) dispatch_output_types(
    const at::ScalarType output_type,
    const char* const name,
    F&& f) {
  switch (output_type) {
    case at::ScalarType::Half:
      return f.template operator()<at::Half>();
    case at::ScalarType::Float:
      return f.template operator()<float>();
    case at::ScalarType::Byte:
      return f.template operator()<uint8_t>();
#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
    case at::ScalarType::BFloat16:
      return f.template operator()<at::BFloat16>();
    case at::ScalarType::QUInt4x2:
      return f.template operator()<uint8_t>();
#endif
    default:
      TORCH_CHECK(
          false,
          name,
          " not implemented for output_t '",
          toString(output_type),
          "'");
  }
}

template <typename F>
decltype(auto) dispatch_emb_grad_cache_types(
    const at::ScalarType emb_type,
    const at::ScalarType grad_type,
    const at::ScalarType cache_type,
    const char* const name,
    F&& f) {
  const auto dispatch_emb_cache = [&]<typename grad_t>() -> decltype(auto) {
    return dispatch_emb_cache_types(
        emb_type,
        cache_type,
        name,
        [&]<typename emb_t, typename cache_t>() -> decltype(auto) {
          return f.template operator()<emb_t, grad_t, cache_t>();
        });
  };
  switch (grad_type) {
    case at::ScalarType::Float:
      return dispatch_emb_cache.template operator()<float>();
    case at::ScalarType::Half:
      return dispatch_emb_cache.template operator()<at::Half>();
    case at::ScalarType::BFloat16:
      return dispatch_emb_cache.template operator()<at::BFloat16>();
    default:
      TORCH_CHECK(
          false,
          name,
          " not implemented for grad_t '",
          toString(grad_type),
          "'");
  }
}

template <typename F>
decltype(auto) dispatch_index_types(
    const at::ScalarType index_type,
    const char* const name,
    F&& f) {
  switch (index_type) {
    case at::ScalarType::Int:
      return f.template operator()<int32_t>();
    case at::ScalarType::Long:
      return f.template operator()<int64_t>();
    default:
      TORCH_CHECK(
          false,
          name,
          " not implemented for index_t '",
          toString(index_type),
          "'");
  }
}

} // namespace fbgemm_gpu

////////////////////////////////////////////////////////////////////////////////
/// Dispatch Helper Macros
///
/// These macros cover bundled dispatch cases, similar to AT_DISPATCH_*_CASE
////////////////////////////////////////////////////////////////////////////////

#define FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define FBGEMM_DISPATCH_FLOATING_TYPES_CASE(...)       \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#if defined(USE_ROCM)

#define FBGEMM_DISPATCH_FLOAT_HALF_AND_FP8_CASE(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fnuz, __VA_ARGS__)

#else

#define FBGEMM_DISPATCH_FLOAT_HALF_AND_FP8_CASE(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__)

#endif

#define FBGEMM_DISPATCH_FLOAT_HALF_AND_DOUBLE_CASE(...) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)

#define FBGEMM_DISPATCH_FLOAT_AND_HALF_CASE(...)       \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define FBGEMM_DISPATCH_FLOAT_AND_BFLOAT16_CASE(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define FBGEMM_DISPATCH_ALL_TYPES_BUT_HALF_CASE(...)      \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__)

#define FBGEMM_DISPATCH_FLOAT_AND_DOUBLE_CASE(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////
/// Type Dispatch Macros
///
/// These macros are similar to AT_DISPATCH_*, but do not support
/// at::ScalarType::Double
////////////////////////////////////////////////////////////////////////////////

#define FBGEMM_DISPATCH_FLOAT_ONLY(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                               \
      TYPE, NAME, AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__))

#define FBGEMM_DISPATCH_FLOAT_AND_DOUBLE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                     \
      TYPE, NAME, FBGEMM_DISPATCH_FLOAT_AND_DOUBLE_CASE(__VA_ARGS__))

#define FBGEMM_DISPATCH_FLOAT_HALF_AND_DOUBLE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                          \
      TYPE, NAME, FBGEMM_DISPATCH_FLOAT_HALF_AND_DOUBLE_CASE(__VA_ARGS__))

#define FBGEMM_DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                   \
      TYPE, NAME, FBGEMM_DISPATCH_FLOAT_AND_HALF_CASE(__VA_ARGS__))

#define FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(TYPE, NAME, ...)  \
  AT_DISPATCH_SWITCH(                                         \
      TYPE,                                                   \
      NAME,                                                   \
      FBGEMM_DISPATCH_FLOAT_AND_HALF_CASE(__VA_ARGS__)        \
          AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__) \
              AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__))

#define FBGEMM_DISPATCH_FLOAT_HALF_FP8_AND_BYTE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                            \
      TYPE,                                                      \
      NAME,                                                      \
      FBGEMM_DISPATCH_FLOAT_HALF_AND_FP8_CASE(__VA_ARGS__)       \
          AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__))

#define FBGEMM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                   \
      TYPE, NAME, FBGEMM_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__))

#define FBGEMM_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                   \
      TYPE,                                                             \
      NAME,                                                             \
      FBGEMM_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__)                  \
          AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__))

#define FBGEMM_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                   \
      TYPE, NAME, FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__))

#define FBGEMM_DISPATCH_ALL_TYPES(TYPE, NAME, ...)     \
  AT_DISPATCH_SWITCH(                                  \
      TYPE,                                            \
      NAME,                                            \
      FBGEMM_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__) \
          FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__))

#define FBGEMM_DISPATCH_ALL_TYPES_AND_DOUBLE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                         \
      TYPE,                                                   \
      NAME,                                                   \
      FBGEMM_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__)        \
          FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__)    \
              AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__))

// We can cleanup the following once fbgemm uses PyTorch 2.2 in January 2024.
#ifndef PT2_COMPLIANT_TAG
#ifdef HAS_PT2_COMPLIANT_TAG
#define PT2_COMPLIANT_TAG at::Tag::pt2_compliant_tag
#else
#define PT2_COMPLIANT_TAG
#endif
#endif

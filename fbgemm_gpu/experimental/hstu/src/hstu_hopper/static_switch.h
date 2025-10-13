// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
//

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define TWO_BOOL_SWITCH(COND1, COND2, CONST_NAME1, CONST_NAME2, ...) \
  [&] {                                                              \
    if (COND1) {                                                     \
      constexpr static bool CONST_NAME1 = true;                      \
      if (COND2) {                                                   \
        constexpr static bool CONST_NAME2 = true;                    \
        return __VA_ARGS__();                                        \
      } else {                                                       \
        constexpr static bool CONST_NAME2 = false;                   \
        return __VA_ARGS__();                                        \
      }                                                              \
    } else {                                                         \
      constexpr static bool CONST_NAME1 = false;                     \
      constexpr static bool CONST_NAME2 = false;                     \
      return __VA_ARGS__();                                          \
    }                                                                \
  }()

#ifdef HSTU_DISABLE_CONTEXT
#define CONTEXT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                       \
    constexpr static bool CONST_NAME = false; \
    return __VA_ARGS__();                     \
  }()
#else
#define CONTEXT_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_TARGET
#define TARGET_SWITCH(COND, CONST_NAME, ...)  \
  [&] {                                       \
    constexpr static bool CONST_NAME = false; \
    return __VA_ARGS__();                     \
  }()
#else
#define TARGET_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_RAB
#define RAB_SWITCH(COND, CONST_NAME, ...)     \
  [&] {                                       \
    constexpr static bool CONST_NAME = false; \
    return __VA_ARGS__();                     \
  }()
#else
#define RAB_SWITCH BOOL_SWITCH
#endif

#ifdef HSTU_DISABLE_RAB
#define RAB_DRAB_SWITCH(                                       \
    RAB_COND, DRAB_COND, RAB_CONST_NAME, DRAB_CONST_NAME, ...) \
  [&] {                                                        \
    constexpr static bool RAB_CONST_NAME = false;              \
    constexpr static bool DRAB_CONST_NAME = false;             \
    return __VA_ARGS__();                                      \
  }()
#else
#ifdef HSTU_DISABLE_DRAB
#define RAB_DRAB_SWITCH(                                       \
    RAB_COND, DRAB_COND, RAB_CONST_NAME, DRAB_CONST_NAME, ...) \
  [&] {                                                        \
    constexpr static bool DRAB_CONST_NAME = false;             \
    if (RAB_COND) {                                            \
      constexpr static bool RAB_CONST_NAME = true;             \
      return __VA_ARGS__();                                    \
    } else {                                                   \
      constexpr static bool RAB_CONST_NAME = false;            \
      return __VA_ARGS__();                                    \
    }                                                          \
  }()
#else
#define RAB_DRAB_SWITCH TWO_BOOL_SWITCH
#endif
#endif

#define QUANT_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND == 1) {                                 \
      constexpr static int CONST_NAME = 1;  \
      return __VA_ARGS__();                     \
    } else if (COND == 2) {                    \
      constexpr static int CONST_NAME = 2; \
      return __VA_ARGS__();                     \
    } else if (COND == 3) {                    \
      constexpr static int CONST_NAME = 3; \
      return __VA_ARGS__();                     \
    } else if (COND == 4) {                    \
      constexpr static int CONST_NAME = 4; \
      return __VA_ARGS__();                     \
    } else if (COND == 5) {                    \
      constexpr static int CONST_NAME = 5; \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int CONST_NAME = 0; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define DTYPE_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND == 0) {     \
      using CONST_NAME = cutlass::bfloat16_t; \
      return __VA_ARGS__();                     \
    } else {                                    \
      using CONST_NAME = cutlass::half_t; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

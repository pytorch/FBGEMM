/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Compat shim for asmjit v1.17 -> v1.18+ (camelCase -> snake_case rename,
// Error typedef -> enum class). Include AFTER asmjit headers.

#pragma once

#include <asmjit/asmjit.h> // @manual

#if ASMJIT_LIBRARY_VERSION >= ASMJIT_LIBRARY_MAKE_VERSION(1, 18, 0)

// bitMask -> bit_mask<int>
namespace asmjit::Support {
template <typename... Args>
constexpr auto bitMask(Args... args) {
  return bit_mask<int>(args...);
}
} // namespace asmjit::Support

// Error: uint32_t typedef -> enum class. Wrap to keep `= 0` and `if (err)`.
namespace asmjit {
struct ErrorCompat {
  Error err;
  ErrorCompat() : err(Error::kOk) {}
  ErrorCompat(int) : err(Error::kOk) {} // NOLINT
  ErrorCompat(Error e) : err(e) {} // NOLINT
  ErrorCompat& operator=(Error e) { err = e; return *this; }
  explicit operator bool() const { return err != Error::kOk; }
};
} // namespace asmjit
#define Error ErrorCompat

// Method renames
#define setDirtyRegs set_dirty_regs
#define assignAll assign_all
#define updateFuncFrame update_func_frame
#define emitProlog emit_prolog
#define emitEpilog emit_epilog
#define emitArgsAssignment emit_args_assignment
#define newLabel new_label
#define saOffsetFromSP sa_offset_from_sp
#define stackOffset stack_offset

#endif

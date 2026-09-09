/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

#include "fbgemm/Utils.h" // inst_set_t

namespace fbgemm {

// Returns true when perf-map emission is enabled, i.e. the environment
// variable FBGEMM_JIT_PERF_MAP is set to something other than "" or "0".
// Callers should gate on this before building a symbol name so that the
// default path costs nothing.
bool jitPerfMapEnabled();

// Publishes a JIT-generated code region to /tmp/perf-<pid>.map in the format
// sampling profilers expect: one record per line, "<hex addr> <hex size>
// <name>". Without this, asmjit output lives in anonymous executable pages
// with no symbol table, and profiles attribute every sample inside a
// generated kernel to "[unknown]".
//
// `name` must not contain a newline: the reader takes everything after the
// size field to end-of-line as the symbol.
void registerJitCodeForProfiling(
    const void* addr,
    size_t size,
    const std::string& name);

// The options the EmbeddingSpMDM and EmbeddingSpMDMNBit generators share.
// Kept in one place so both keep the same flag spelling and ordering as
// options are added.
struct EmbeddingKernelOptions {
  // The generators keep one code cache per template instantiation, so every
  // template parameter that selects an instantiation has to appear here too:
  // two kernels from different caches are distinct code, and a shared name
  // would misattribute one to the other.
  bool indices64 = false;
  bool offsets64 = false;
  bool thread_local_cache = false;
  bool avx512 = false;
  // Numeric members are part of the generators' code-cache keys, so they have
  // to appear in the symbol or distinct kernels collide under one name.
  int prefetch = 0;
  int output_stride = -1;
  int input_stride = -1;
  bool weighted = false;
  bool positional = false;
  bool normalize = false;
  bool lengths = false; // i.e. !use_offsets
  bool rowwise_sparse = false;
  bool scale_bias_first = false; // i.e. !scale_bias_last
  bool fp16_out = false;
  bool bf16_out = false;
};

// Builds `fbgemm::<kernel>_<shape>_<flags...>`, e.g.
// `fbgemm::EmbeddingSpMDMNBit_2bit_D-128_idx64_off32_avx512_weighted`. `shape`
// is the part that differs between the two generators (bit rate vs input type).
//
// Underscore-separated rather than `<...>` on purpose: profilers demangle and
// normalize symbols, and Strobelight drops template arguments outright, which
// silently discards everything inside the brackets. This also matches the
// naming `CodeGenBase::getKernelName` already uses for the GEMM kernels.
std::string embeddingKernelSymbol(
    std::string_view kernel,
    std::string_view shape,
    const EmbeddingKernelOptions& options);

// Canonical spelling of an instruction set in a JIT symbol. The generators are
// templated on inst_set_t and each instantiation keeps its own code cache, so
// the ISA has to appear in the name or same-shape variants collide.
template <inst_set_t instSet>
constexpr const char* instSetName() {
  if constexpr (instSet == inst_set_t::avx512_vnni) {
    return "avx512vnni";
  } else if constexpr (instSet == inst_set_t::avx512_vnni_ymm) {
    return "avx512vnni_ymm";
  } else if constexpr (instSet == inst_set_t::avx512_ymm) {
    return "avx512_ymm";
  } else if constexpr (instSet == inst_set_t::avx512) {
    return "avx512";
  } else if constexpr (instSet == inst_set_t::avx2) {
    return "avx2";
  } else if constexpr (instSet == inst_set_t::sve) {
    return "sve";
  } else {
    return "anyarch";
  }
}

// "idx64"/"idx32" for a generator's index type, matching the spelling
// embeddingKernelSymbol() uses.
template <typename IndexType>
constexpr const char* indexWidthName() {
  return sizeof(IndexType) == 8 ? "idx64" : "idx32";
}

// Registers a freshly JIT-ed kernel with the perf map. `makeName` is invoked
// only when emission is enabled, so building the symbol costs nothing on the
// default path.
//
// The size comes from `code` AFTER asmjit's add(): relocation there can shrink
// the code, so a pre-add size is only an upper bound and would publish a range
// that overlaps the next kernel. Keeping that contract here means the call
// sites cannot drift from it.
//
// Templated on the holder so this header needs no asmjit include.
template <typename CodeHolderT, typename MakeName>
void registerJitKernel(
    const CodeHolderT& code,
    const void* fn,
    MakeName&& makeName) {
  if (!jitPerfMapEnabled()) {
    return;
  }
  registerJitCodeForProfiling(fn, code.codeSize(), makeName());
}

} // namespace fbgemm

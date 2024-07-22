/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// @lint-ignore-every LICENSELINT

/*! \file
    \brief Operation table for CUTLASS kernels
*/

// clang-format off
#pragma once

#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <unordered_map>

#include "cutlass/library/library.h"
#include "cutlass/library/util.h"

#include <cutlass_extensions/include/manifest.h>
#include <cutlass_extensions/include/utils.h>
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_extensions {

/////////////////////////////////////////////////////////////////////////////////////////////////
//                          Data Structures for Gemm Functional Maps
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tuple uniquely identifying Gemm functional behavior
struct GemmFunctionalKey {
  // Functional key items from cutlass::library
  cutlass::library::Provider provider;
  cutlass::library::GemmKind gemm_kind;
  cutlass::library::NumericTypeID element_accumulator;
  cutlass::library::NumericTypeID element_A;
  cutlass::library::LayoutTypeID layout_A;
  cutlass::library::NumericTypeID element_B;
  cutlass::library::LayoutTypeID layout_B;
  cutlass::library::NumericTypeID element_C;
  cutlass::library::LayoutTypeID layout_C;
  cutlass::library::NumericTypeID element_D;
  cutlass::library::LayoutTypeID layout_D;

  // Additional functional key imtes from cutlass_extensions
  cutlass_extensions::FusionKind fusion_kind;
  cutlass_extensions::AccumKind accum_kind;

  //
  // Methods
  //

  inline GemmFunctionalKey(
      cutlass::library::Provider provider = cutlass::library::Provider::kCUTLASS,
      cutlass::library::GemmKind gemm_kind = cutlass::library::GemmKind::kGemm,
      cutlass::library::NumericTypeID element_accumulator = cutlass::library::NumericTypeID::kF32,
      cutlass::library::NumericTypeID element_A = cutlass::library::NumericTypeID::kF16,
      cutlass::library::LayoutTypeID layout_A = cutlass::library::LayoutTypeID::kColumnMajor,
      cutlass::library::NumericTypeID element_B = cutlass::library::NumericTypeID::kF16,
      cutlass::library::LayoutTypeID layout_B = cutlass::library::LayoutTypeID::kColumnMajor,
      cutlass::library::NumericTypeID element_C = cutlass::library::NumericTypeID::kF16,
      cutlass::library::LayoutTypeID layout_C = cutlass::library::LayoutTypeID::kColumnMajor,
      cutlass::library::NumericTypeID element_D = cutlass::library::NumericTypeID::kF16,
      cutlass::library::LayoutTypeID layout_D = cutlass::library::LayoutTypeID::kColumnMajor,
      cutlass_extensions::FusionKind fusion_kind = cutlass_extensions::FusionKind::kNone,
      cutlass_extensions::AccumKind accum_kind = cutlass_extensions::AccumKind::kDefault)
      : provider(provider),
        gemm_kind(gemm_kind),
        element_accumulator(element_accumulator),
        element_A(element_A),
        layout_A(layout_A),
        element_B(element_B),
        layout_B(layout_B),
        element_C(element_C),
        layout_C(layout_C),
        element_D(element_D),
        layout_D(layout_D),
        fusion_kind(fusion_kind),
        accum_kind(accum_kind) {}

  inline bool operator==(GemmFunctionalKey const& rhs) const {
    return (provider == rhs.provider) && (gemm_kind == rhs.gemm_kind) &&
        (element_accumulator == rhs.element_accumulator) &&
        (element_A == rhs.element_A) && (layout_A == rhs.layout_A) &&
        (element_B == rhs.element_B) && (layout_B == rhs.layout_B) &&
        (element_C == rhs.element_C) && (layout_C == rhs.layout_C) &&
        (element_D == rhs.element_D) && (layout_D == rhs.layout_D) &&
        (fusion_kind == rhs.fusion_kind) && (accum_kind == rhs.accum_kind);
  }

  inline bool operator!=(GemmFunctionalKey const& rhs) const {
    return !(*this == rhs);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

inline std::ostream& operator<<(
    std::ostream& out,
    cutlass_extensions::GemmFunctionalKey const& k) {
  out << "{\n"
      << "             provider: " << cutlass::library::to_string(k.provider) << "\n"
      << "            gemm_kind: " << cutlass::library::to_string(k.gemm_kind) << "\n"
      << "  element_accumulator: " << cutlass::library::to_string(k.element_accumulator) << "\n"
      << "            element_A: " << cutlass::library::to_string(k.element_A) << "\n"
      << "             layout_A: " << cutlass::library::to_string(k.layout_A) << "\n"
      << "            element_B: " << cutlass::library::to_string(k.element_B) << "\n"
      << "             layout_B: " << cutlass::library::to_string(k.layout_B) << "\n"
      << "            element_C: " << cutlass::library::to_string(k.element_C) << "\n"
      << "             layout_C: " << cutlass::library::to_string(k.layout_C) << "\n"
      << "            element_D: " << cutlass::library::to_string(k.element_D) << "\n"
      << "             layout_D: " << cutlass::library::to_string(k.layout_D) << "\n"
      << "           fusion_kind: " << cutlass_extensions::to_string(k.fusion_kind) << "\n"
      << "            accum_kind: " << cutlass_extensions::to_string(k.accum_kind) << "\n"
      << "}";

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Hash function for GemmFunctionalKey
struct GemmFunctionalKeyHasher {
  using IntHash = std::hash<int>;

  inline static size_t rotl(size_t key, int shl) {
    return (key << shl) |
        (key >> (sizeof(key) * 8u - static_cast<size_t>(shl)));
  }

  inline size_t operator()(GemmFunctionalKey const& key) const {
    IntHash hash;

    return rotl(hash(int(key.provider)), 1) ^
        rotl(hash(int(key.gemm_kind)), 2) ^
        rotl(hash(int(key.element_accumulator)), 3) ^
        rotl(hash(int(key.element_A)), 5) ^ rotl(hash(int(key.layout_A)), 6) ^
        rotl(hash(int(key.element_B)), 8) ^ rotl(hash(int(key.layout_B)), 9) ^
        rotl(hash(int(key.element_C)), 11) ^ rotl(hash(int(key.layout_C)), 12) ^
        rotl(hash(int(key.element_D)), 13) ^ rotl(hash(int(key.layout_D)), 14) ^
        rotl(hash(int(key.fusion_kind)), 15) ^ rotl(hash(int(key.accum_kind)), 16);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//                          Data Structures for Gemm Performance Maps
/////////////////////////////////////////////////////////////////////////////////////////////////

struct GemmPerformanceKey {

  /// Describes the shape of tensor core math instruction (in elements)
  cutlass::gemm::GemmCoord instruction_shape;

  /// Describes the shape of a threadblock (in elements)
  cutlass::gemm::GemmCoord threadblock_shape;

  /// Describes the shape of a cluster (in blocks)
  cutlass::gemm::GemmCoord cluster_shape;

  /// Describles mainloop schedule (multistage, warpspecalized, etc)
  cutlass_extensions::MainloopSchedule mainloop_schedule;

  /// Describles epilogue schedule (nosmem, tma, etc)
  cutlass_extensions::EpilogueSchedule epilogue_schedule;

  /// ctor
  inline GemmPerformanceKey(
      cutlass::gemm::GemmCoord instruction_shape,
      cutlass::gemm::GemmCoord threadblock_shape,
      cutlass::gemm::GemmCoord cluster_shape,
      cutlass_extensions::MainloopSchedule mainloop_schedule,
      cutlass_extensions::EpilogueSchedule epilogue_schedule)
      : instruction_shape(instruction_shape),
        threadblock_shape(threadblock_shape), 
        cluster_shape(cluster_shape),
        mainloop_schedule(mainloop_schedule),
        epilogue_schedule(epilogue_schedule) {}

  inline bool operator==(GemmPerformanceKey const& rhs) const {
    return (instruction_shape == rhs.instruction_shape) &&
           (threadblock_shape == rhs.threadblock_shape) &&
           (cluster_shape == rhs.cluster_shape) &&
           (mainloop_schedule == rhs.mainloop_schedule) &&
           (epilogue_schedule == rhs.epilogue_schedule);
  }

  inline bool operator!=(GemmPerformanceKey const& rhs) const {
    return !(*this == rhs);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Hash function for GemmPerformanceKey
struct GemmPerformanceKeyHasher {
  using IntHash = std::hash<int>;

  inline static size_t rotl(size_t key, int shl) {
    return (key << shl) |
        (key >> (sizeof(key) * 8u - static_cast<size_t>(shl)));
  }

  inline size_t operator()(GemmPerformanceKey const& key) const {
    IntHash hash;

    return rotl(hash(int(key.instruction_shape.m())), 1) ^
           rotl(hash(int(key.instruction_shape.n())), 2) ^
           rotl(hash(int(key.instruction_shape.k())), 3) ^
           rotl(hash(int(key.threadblock_shape.m())), 4) ^
           rotl(hash(int(key.threadblock_shape.n())), 5) ^
           rotl(hash(int(key.threadblock_shape.k())), 6) ^
           rotl(hash(int(key.cluster_shape.m())), 7) ^
           rotl(hash(int(key.cluster_shape.n())), 8) ^
           rotl(hash(int(key.cluster_shape.k())), 9) ^
           rotl(hash(int(key.mainloop_schedule)), 10) ^
           rotl(hash(int(key.epilogue_schedule)), 11);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

inline std::ostream& operator<<(
    std::ostream& out,
    cutlass_extensions::GemmPerformanceKey const& k) {
  out << "{\n"
      << "         instruction shape: {" << k.instruction_shape.m() << ", "
                                         << k.instruction_shape.n() << ", " 
                                         << k.instruction_shape.k() << "}\n"
      << "         threadblock shape: {" << k.threadblock_shape.m() << ", "
                                         << k.threadblock_shape.n() << ", " 
                                         << k.threadblock_shape.k() << "}\n"
      << "             cluster shape: {" << k.cluster_shape.m() << ", "
                                         << k.cluster_shape.n() << ", " 
                                         << k.cluster_shape.k() << "}\n"
      << "         mainloop schedule:  " << cutlass_extensions::to_string(k.mainloop_schedule) << "\n"  
      << "}";

  return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//     Data structures for Operation tables
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Maps Gemm Performance Key to a vector Gemm Operation.
using GemmOperationPerformanceMap = std::unordered_map<
    GemmPerformanceKey,
    std::vector<cutlass::library::Operation const*>,
    GemmPerformanceKeyHasher>;

/// Maps a GemmFunctionalKey onto a vector of Operation
/// GemmFunctionalKey -> {GemmPerformanceKey -> [Operations*]}
using GemmOperationFunctionalMap = std::unordered_map<
    GemmFunctionalKey,
    GemmOperationPerformanceMap,
    GemmFunctionalKeyHasher>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Table of cutlass::library::Operation instances
class OperationTable {
 public:
  /// Gemm operation (D = alpha * (A x B) + beta * C)
  GemmOperationFunctionalMap gemm_operations;
  GemmOperationFunctionalMap gemm_operations_with_tensorwise;
  GemmOperationFunctionalMap gemm_operations_with_rowwise;
  GemmOperationFunctionalMap gemm_operations_with_blockwise;

 public:
  void append(Manifest const& manifest);
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass_extensions

/////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(
    std::ostream& out,
    cutlass_extensions::GemmFunctionalKey const& k);

std::ostream& operator<<(
    std::ostream& out,
    cutlass_extensions::GemmPerformanceKey const& k);

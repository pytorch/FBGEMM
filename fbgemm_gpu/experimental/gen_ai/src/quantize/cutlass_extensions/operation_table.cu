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

/*
  \file
  \brief Defines a data structure in which a set of functionally equivalent
  cutlass::library::Operation instances may be queried.
*/

#include <cutlass_extensions/include/gemm_description.h>
#include <cutlass_extensions/include/operation_table.h>
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_extensions {

/////////////////////////////////////////////////////////////////////////////////////////////////

void OperationTable::append(Manifest const& manifest) {
  for (auto const& operation : manifest) {
    cutlass::library::OperationDescription const& desc =
        operation->description();

    if (desc.kind == cutlass::library::OperationKind::kGemm) {
      cutlass_extensions::GemmDescription const& gemm_desc =
          static_cast<cutlass_extensions::GemmDescription const&>(desc);

      GemmFunctionalKey functional_key(
          gemm_desc.provider,
          gemm_desc.gemm_kind,
          gemm_desc.tile_description.math_instruction.element_accumulator,
          gemm_desc.A.element,
          gemm_desc.A.layout,
          gemm_desc.B.element,
          gemm_desc.B.layout,
          gemm_desc.C.element,
          gemm_desc.C.layout,
          gemm_desc.D.element,
          gemm_desc.D.layout,
          gemm_desc.fusion_kind,
          gemm_desc.accum_kind);
      // std::cout << "Gemm Functional Key: " << functional_key << std::endl;

      GemmPerformanceKey performance_key(
          gemm_desc.tile_description.math_instruction.instruction_shape,
          gemm_desc.tile_description.threadblock_shape,
          gemm_desc.tile_description.cluster_shape,
          gemm_desc.mainloop_schedule,
          gemm_desc.epilogue_schedule);
      // std::cout << "Gemm Performance Key " << performance_key << std::endl;

      // Populate gemm operation tables
      cutlass::library::Operation const* op_ptr = operation.get();
      if (gemm_desc.fusion_kind ==
          cutlass_extensions::FusionKind::kTensorwiseScaling) {
        gemm_operations_with_tensorwise[functional_key][performance_key]
            .push_back(op_ptr);
      } else if (
          gemm_desc.fusion_kind ==
          cutlass_extensions::FusionKind::kRowwiseScaling) {
        gemm_operations_with_rowwise[functional_key][performance_key].push_back(
            op_ptr);
      } else if (
          gemm_desc.fusion_kind ==
          cutlass_extensions::FusionKind::kBlockwiseScaling) {
        gemm_operations_with_blockwise[functional_key][performance_key]
            .push_back(op_ptr);
      }
    }
  }
}

} // namespace cutlass_extensions

/*
f8f8bf16 : it takes a scale tensor scale[1]. It is a single element tensor for
cuda graph to work.


ignore it for now: f8f8bf16_tensorwise
*/

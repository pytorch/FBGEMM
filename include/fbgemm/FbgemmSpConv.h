#pragma once

#include "fbgemm/FbgemmSpMM.h"

namespace fbgemm {

/**
 * NOTE: this is an experimental feature especially ACC_T==int32_t
 *
 * Generate a kernel that convolves A with B with specialization for sparse
 * matrix A's structure and values.
 *
 * When ACC_T == float, we assume A in RSKC layout, B and C in CNHW layout.
 * When ACC_T == int32_t, we assume A in RSKC layout, B in C/4 NHW c4 layout,
 * and C in CNHW layout (TODO: make C layout same as B)
 */
template <typename ACC_T>
FBGEMM_API std::function<void(
    const typename internal::SpMMTypeTrait<ACC_T>::b_type* BData,
    ACC_T* CData)>
generateSpConv(
    int Cin,
    int Cout,
    int IY,
    int IX,
    const typename internal::SpMMTypeTrait<ACC_T>::a_type* AData);

} // namespace fbgemm

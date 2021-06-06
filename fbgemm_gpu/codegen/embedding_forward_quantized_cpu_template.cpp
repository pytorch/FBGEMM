/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

{% set wdesc =  "weighted" if weighted else "unweighted" %}

#include <ATen/ATen.h>

#include <immintrin.h>
#include <emmintrin.h>

enum PoolingMode { SUM = 0, MEAN = 1, NONE = 2 };

using namespace at;

// From https://stackoverflow.com/questions/55084047/intel-vector-instruction-to-zero-extend-8-4-bit-values-packed-in-a-32-bit-int-to
// TODO: dispatch at architecture time?
__attribute__((always_inline)) inline __m256i cvt_nib_epi32_HSW(uint32_t x) {
    __uint64_t x_b = _pdep_u64(x, 0x0F0F0F0F0F0F0F0F);
    __m128i x_v = _mm_cvtsi64_si128(x_b);
    return _mm256_cvtepu8_epi32(x_v);
}

__attribute__((always_inline)) inline __m256i cvt_nib_epi32_SKL(uint32_t x) {
    __m256i input = _mm256_set1_epi32(x);
    __m256i shifted = _mm256_srlv_epi32(input,_mm256_set_epi32(28,24,20,16,12,8,4,0));
    return _mm256_and_si256(shifted, _mm256_set1_epi32(0xF));
}

__attribute__((always_inline)) inline __m256i cvt_hnib_epi32_SKL(uint16_t x) {
    __m256i input = _mm256_set1_epi32(x);
    __m256i shifted = _mm256_srlv_epi32(input,_mm256_set_epi32(14,12,10,8,6,4,2,0));
    return _mm256_and_si256(shifted, _mm256_set1_epi32(0x3));
}

__attribute__((always_inline)) inline __m256i cvt_byte_SKL(uint64_t x) {
    return _mm256_cvtepu8_epi32(_mm_set1_epi64x(x));
}

// Keep in sync with split_embedding_configs.py:SparseType
enum class SparseType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    INT2 = 4,
};

Tensor int_nbit_split_embedding_codegen_forward_{{ wdesc }}_cpu(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_effective_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    int64_t unused
) {
    int32_t T = D_offsets.numel() - 1;
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    TORCH_CHECK(total_D > 0);
    TORCH_CHECK(total_D % 8 == 0);
    TORCH_CHECK(max_effective_D <= {{ max_embedding_dim }});
    auto output = empty({B, total_D}, dev_weights.options().dtype(at::kHalf).pinned_memory(true));
    const auto* weights_acc = dev_weights.data_ptr<uint8_t>();
    const auto* weights_tys_acc = weights_tys.data_ptr<uint8_t>();

    auto* output_acc = output.data_ptr<Half>();
    {% if weighted %}
    const float* indice_weights_acc = indice_weights.data_ptr<float>();
    {% endif %}
    // Empty vector filled with zeros (thus accumulating to zero).
    std::vector<uint8_t> zero_row(max_effective_D * 2, 0);
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "int_nbit_split_embedding_codegen_forward_", [&] () {
        const auto* indices_acc = indices.data_ptr<index_t>();
        const auto* offsets_acc = offsets.data_ptr<index_t>();
        const auto* D_offsets_acc = D_offsets.data_ptr<int32_t>();
        const auto* weights_offsets_acc = weights_offsets.data_ptr<int64_t>();

        int32_t num_indices_m_1 = indices.numel() - 1;

        for (int32_t t = 0; t < T; ++t) {
            const int32_t D_start = D_offsets_acc[t];
            const int32_t D = D_offsets_acc[t+1] - D_offsets_acc[t];
            const uint8_t* weights = &weights_acc[weights_offsets_acc[t]];
            auto weight_ty = static_cast<SparseType>(weights_tys_acc[t]);
            const int32_t D_vecs = D / 8;

            if (weight_ty == SparseType::INT4) {
                const int32_t D_vecs = D / 8;
                // 0.5 bytes per D, plus 2 * 2 bytes for fp16 scale/shift.
                const int64_t D_bytes = D / 2 + 4;
                {% for kMaxVecsPerThread in range(33) %}
                if (D_vecs == {{ kMaxVecsPerThread }}) {
                    for (int32_t b = 0; b < B; ++b) {
                        int32_t indices_start = offsets_acc[t * B + b];
                        int32_t indices_end = offsets_acc[t * B + b + 1];
                        int32_t L = indices_end - indices_start;
                        std::array<__m256, {{ kMaxVecsPerThread }} > acc;
                        // TODO: try fbgemm when I figure out how to adjust the bias/scale setting (first vs last element) and cache the codegen state.
                        {% for i in range(kMaxVecsPerThread) %}
                            acc[{{ i }}] = _mm256_setzero_ps();
                        {% endfor %}
                        int32_t l = 0;
                        int32_t LUnroll = (L / 2) * 2;
                        for (; l < LUnroll; l += 2) {
                            int64_t idx0 = indices_acc[indices_start + l + 0];
                            int64_t idx1 = indices_acc[indices_start + l + 1];

                            const uint32_t* row0 = idx0 == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx0 * D_bytes]);
                            const uint32_t* row1 = idx1 == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx1 * D_bytes]);
                            const uint32_t* vrow0 = reinterpret_cast<const uint32_t*>(row0 + 1);
                            const uint32_t* vrow1 = reinterpret_cast<const uint32_t*>(row1 + 1);

                            uint32_t scale_shift0 = row0[0];
                            uint32_t scale_shift1 = row1[0];

                            int64_t prefetch_idx0 = indices_acc[std::min<int32_t>(indices_start + l + 2, num_indices_m_1)];
                            int64_t prefetch_idx1 = indices_acc[std::min<int32_t>(indices_start + l + 3, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx0 * D_bytes], _MM_HINT_T0);
                            _mm_prefetch(&weights[prefetch_idx1 * D_bytes], _MM_HINT_T0);

                            auto scale0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift0 & 0xFFFF)));
                            auto scale1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift1 & 0xFFFF)));
                            auto shift0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift0 >> 16) & 0xFFFF)));
                            auto shift1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift1 >> 16) & 0xFFFF)));

                            {% if weighted %}
                            auto idx_weight0 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 0]);
                            auto idx_weight1 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 1]);
                            scale0 = _mm256_mul_ps(scale0, idx_weight0);
                            scale1 = _mm256_mul_ps(scale1, idx_weight1);

                            shift0 = _mm256_mul_ps(shift0, idx_weight0);
                            shift1 = _mm256_mul_ps(shift1, idx_weight1);
                            {% endif %}
                            {% for i in range(kMaxVecsPerThread) %}
                                {% if weighted %}
                                acc[{{ i }}] = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(cvt_nib_epi32_SKL(vrow0[{{ i }}])), _mm256_add_ps(acc[{{ i }}], shift0));
                                acc[{{ i }}] = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(cvt_nib_epi32_SKL(vrow1[{{ i }}])), _mm256_add_ps(acc[{{ i }}], shift1));
                                {% else %}
                                acc[{{ i }} ] = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(cvt_nib_epi32_SKL(vrow0[{{ i }}])), _mm256_add_ps(acc[{{ i }} ], shift0));
                                acc[{{ i }} ] = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(cvt_nib_epi32_SKL(vrow1[{{ i }}])), _mm256_add_ps(acc[{{ i }} ], shift1));
                                {% endif %}
                            {% endfor %}
                        }
                        for (; l < L; ++l) {
                            int64_t idx = indices_acc[indices_start + l];
                            const uint32_t* row = idx == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx * D_bytes]);
                            const uint32_t* vrow = reinterpret_cast<const uint32_t*>(row + 1);
                            uint32_t scale_shift = row[0];

                            int64_t prefetch_idx = indices_acc[std::min<int32_t>(indices_start + l + 1, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx * D_bytes], _MM_HINT_T0);

                            auto scale = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift & 0xFFFF)));
                            auto shift = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift >> 16) & 0xFFFF)));

                            {% if weighted %}
                            auto idx_weight = _mm256_set1_ps(indice_weights_acc[indices_start + l]);
                            scale = _mm256_mul_ps(scale, idx_weight);
                            shift = _mm256_mul_ps(shift, idx_weight);
                            {% endif %}

                            {% for i in range(kMaxVecsPerThread) %}
                                {% if weighted %}
                                acc[{{ i }}] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(cvt_nib_epi32_SKL(vrow[{{ i }}])), _mm256_add_ps(acc[{{ i }}], shift));
                                {% else %}
                                acc[{{ i }} ] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(cvt_nib_epi32_SKL(vrow[{{ i }}])), _mm256_add_ps(acc[{{ i }} ], shift));
                                {% endif %}
                            {% endfor %}
                        }

                        const float scale_factor =
                        // NOTE: MEAN pooling will not work with indice_weights!
                        (pooling_mode == MEAN && L > 0)
                                        ? 1.0 / L : 1.0;

                        __m256 scale_vec, scale_acc;
                        {% for i in range(kMaxVecsPerThread) %}
                            scale_vec = _mm256_set1_ps(scale_factor);
                            scale_acc = _mm256_mul_ps(acc[{{ i }}], scale_vec);
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(&output_acc[b * total_D + D_start + 8 * {{ i }}]), _mm256_cvtps_ph(scale_acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                        {% endfor %}
                    }
                }
                {% endfor %}
            }

            if (weight_ty == SparseType::INT2) {
                // 0.25 bytes per D, plus 2 * 2 bytes for fp16 scale/shift.
                const int64_t D_bytes = D / 4 + 4;
                {% for kMaxVecsPerThread in range(1, 33) %}
                if (D_vecs == {{ kMaxVecsPerThread }}) {
                    for (int32_t b = 0; b < B; ++b) {
                        int32_t indices_start = offsets_acc[t * B + b];
                        int32_t indices_end = offsets_acc[t * B + b + 1];
                        int32_t L = indices_end - indices_start;
                        std::array<__m256, {{ kMaxVecsPerThread }} > acc;
                        // TODO: try fbgemm when I figure out how to adjust the bias/scale setting (first vs last element) and cache the codegen state.
                        {% for i in range(kMaxVecsPerThread) %}
                            acc[{{ i }}] = _mm256_setzero_ps();
                        {% endfor %}
                        int32_t l = 0;
                        int32_t LUnroll = (L / 2) * 2;
                        for (; l < LUnroll; l += 2) {
                            int64_t idx0 = indices_acc[indices_start + l + 0];
                            int64_t idx1 = indices_acc[indices_start + l + 1];

                            const uint32_t* row0 = idx0 == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx0 * D_bytes]);
                            const uint32_t* row1 = idx1 == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx1 * D_bytes]);

                            const uint16_t* vrow0 = reinterpret_cast<const uint16_t*>(row0 + 1);
                            const uint16_t* vrow1 = reinterpret_cast<const uint16_t*>(row1 + 1);

                            uint32_t scale_shift0 = row0[0];
                            uint32_t scale_shift1 = row1[0];

                            int64_t prefetch_idx0 = indices_acc[std::min<int32_t>(indices_start + l + 2, num_indices_m_1)];
                            int64_t prefetch_idx1 = indices_acc[std::min<int32_t>(indices_start + l + 3, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx0 * D_bytes], _MM_HINT_T0);
                            _mm_prefetch(&weights[prefetch_idx1 * D_bytes], _MM_HINT_T0);

                            auto scale0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift0 & 0xFFFF)));
                            auto scale1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift1 & 0xFFFF)));
                            auto shift0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift0 >> 16) & 0xFFFF)));
                            auto shift1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift1 >> 16) & 0xFFFF)));

                            {% if weighted %}
                            auto idx_weight0 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 0]);
                            auto idx_weight1 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 1]);
                            scale0 = _mm256_mul_ps(scale0, idx_weight0);
                            scale1 = _mm256_mul_ps(scale1, idx_weight1);

                            shift0 = _mm256_mul_ps(shift0, idx_weight0);
                            shift1 = _mm256_mul_ps(shift1, idx_weight1);
                            {% endif %}
                            {% for i in range(kMaxVecsPerThread) %}
                                {% if weighted %}
                                acc[{{ i }}] = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(cvt_hnib_epi32_SKL(vrow0[{{ i }}])), _mm256_add_ps(acc[{{ i }}], shift0));
                                acc[{{ i }}] = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(cvt_hnib_epi32_SKL(vrow1[{{ i }}])), _mm256_add_ps(acc[{{ i }}], shift1));
                                {% else %}
                                acc[{{ i }} ] = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(cvt_hnib_epi32_SKL(vrow0[{{ i }}])), _mm256_add_ps(acc[{{ i }} ], shift0));
                                acc[{{ i }} ] = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(cvt_hnib_epi32_SKL(vrow1[{{ i }}])), _mm256_add_ps(acc[{{ i }} ], shift1));
                                {% endif %}
                            {% endfor %}
                        }
                        for (; l < L; ++l) {
                            int64_t idx = indices_acc[indices_start + l];
                            const uint32_t* row = idx == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx * D_bytes]);
                            const uint16_t* vrow = reinterpret_cast<const uint16_t*>(row + 1);
                            uint32_t scale_shift = row[0];

                            int64_t prefetch_idx = indices_acc[std::min<int32_t>(indices_start + l + 1, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx * D_bytes], _MM_HINT_T0);

                            auto scale = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift & 0xFFFF)));
                            auto shift = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift >> 16) & 0xFFFF)));

                            {% if weighted %}
                            auto idx_weight = _mm256_set1_ps(indice_weights_acc[indices_start + l]);
                            scale = _mm256_mul_ps(scale, idx_weight);
                            shift = _mm256_mul_ps(shift, idx_weight);
                            {% endif %}

                            {% for i in range(kMaxVecsPerThread) %}
                                {% if weighted %}
                                acc[{{ i }}] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(cvt_hnib_epi32_SKL(vrow[{{ i }}])), _mm256_add_ps(acc[{{ i }}], shift));
                                {% else %}
                                acc[{{ i }} ] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(cvt_hnib_epi32_SKL(vrow[{{ i }}])), _mm256_add_ps(acc[{{ i }} ], shift));
                                {% endif %}
                            {% endfor %}
                        }

                        const float scale_factor =
                        // NOTE: MEAN pooling will not work with indice_weights!
                        (pooling_mode == MEAN && L > 0)
                                        ? 1.0 / L : 1.0;

                        __m256 scale_vec, scale_acc;
                        {% for i in range(kMaxVecsPerThread) %}
                            scale_vec = _mm256_set1_ps(scale_factor);
                            scale_acc = _mm256_mul_ps(acc[{{ i }}], scale_vec);
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(&output_acc[b * total_D + D_start + 8 * {{ i }}]), _mm256_cvtps_ph(scale_acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                        {% endfor %}
                    }
                }
                {% endfor %}
            }

            if (weight_ty == SparseType::INT8) {
                // 1 bytes per D, plus 2 * 2 bytes for fp16 scale/shift + 4 bytes for padding.
                const int64_t D_bytes = D + 8;
                {% for kMaxVecsPerThread in range(1, 33) %}
                if (D_vecs == {{ kMaxVecsPerThread }}) {
                    for (int32_t b = 0; b < B; ++b) {
                        int32_t indices_start = offsets_acc[t * B + b];
                        int32_t indices_end = offsets_acc[t * B + b + 1];
                        int32_t L = indices_end - indices_start;
                        std::array<__m256, {{ kMaxVecsPerThread }} > acc;
                        // TODO: try fbgemm when I figure out how to adjust the bias/scale setting (first vs last element) and cache the codegen state.
                        {% for i in range(kMaxVecsPerThread) %}
                            acc[{{ i }}] = _mm256_setzero_ps();
                        {% endfor %}
                        int32_t l = 0;
                        int32_t LUnroll = (L / 2) * 2;
                        for (; l < LUnroll; l += 2) {
                            int64_t idx0 = indices_acc[indices_start + l + 0];
                            int64_t idx1 = indices_acc[indices_start + l + 1];

                            const uint32_t* row0 = idx0 == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx0 * D_bytes]);
                            const uint32_t* row1 = idx1 == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx1 * D_bytes]);

                            const uint64_t* vrow0 = reinterpret_cast<const uint64_t*>(row0 + 2);
                            const uint64_t* vrow1 = reinterpret_cast<const uint64_t*>(row1 + 2);

                            uint32_t scale_shift0 = row0[0];
                            uint32_t scale_shift1 = row1[0];

                            int64_t prefetch_idx0 = indices_acc[std::min<int32_t>(indices_start + l + 2, num_indices_m_1)];
                            int64_t prefetch_idx1 = indices_acc[std::min<int32_t>(indices_start + l + 3, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx0 * D_bytes], _MM_HINT_T0);
                            _mm_prefetch(&weights[prefetch_idx1 * D_bytes], _MM_HINT_T0);

                            auto scale0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift0 & 0xFFFF)));
                            auto scale1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift1 & 0xFFFF)));
                            auto shift0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift0 >> 16) & 0xFFFF)));
                            auto shift1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift1 >> 16) & 0xFFFF)));

                            {% if weighted %}
                            auto idx_weight0 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 0]);
                            auto idx_weight1 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 1]);
                            scale0 = _mm256_mul_ps(scale0, idx_weight0);
                            scale1 = _mm256_mul_ps(scale1, idx_weight1);

                            shift0 = _mm256_mul_ps(shift0, idx_weight0);
                            shift1 = _mm256_mul_ps(shift1, idx_weight1);
                            {% endif %}
                            {% for i in range(kMaxVecsPerThread) %}
                                {% if weighted %}
                                acc[{{ i }}] = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(cvt_byte_SKL(vrow0[{{ i }}])), _mm256_add_ps(acc[{{ i }}], shift0));
                                acc[{{ i }}] = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(cvt_byte_SKL(vrow1[{{ i }}])), _mm256_add_ps(acc[{{ i }}], shift1));
                                {% else %}
                                acc[{{ i }} ] = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(cvt_byte_SKL(vrow0[{{ i }}])), _mm256_add_ps(acc[{{ i }} ], shift0));
                                acc[{{ i }} ] = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(cvt_byte_SKL(vrow1[{{ i }}])), _mm256_add_ps(acc[{{ i }} ], shift1));
                                {% endif %}
                            {% endfor %}
                        }
                        for (; l < L; ++l) {
                            int64_t idx = indices_acc[indices_start + l];
                            const uint32_t* row = idx == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx * D_bytes]);
                            const uint64_t* vrow = reinterpret_cast<const uint64_t*>(row + 2);
                            uint32_t scale_shift = row[0];

                            int64_t prefetch_idx = indices_acc[std::min<int32_t>(indices_start + l + 1, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx * D_bytes], _MM_HINT_T0);

                            auto scale = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift & 0xFFFF)));
                            auto shift = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift >> 16) & 0xFFFF)));

                            {% if weighted %}
                            auto idx_weight = _mm256_set1_ps(indice_weights_acc[indices_start + l]);
                            scale = _mm256_mul_ps(scale, idx_weight);
                            shift = _mm256_mul_ps(shift, idx_weight);
                            {% endif %}

                            {% for i in range(kMaxVecsPerThread) %}
                                {% if weighted %}
                                acc[{{ i }}] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(cvt_byte_SKL(vrow[{{ i }}])), _mm256_add_ps(acc[{{ i }}], shift));
                                {% else %}
                                acc[{{ i }} ] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(cvt_byte_SKL(vrow[{{ i }}])), _mm256_add_ps(acc[{{ i }} ], shift));
                                {% endif %}
                            {% endfor %}
                        }

                        const float scale_factor =
                        // NOTE: MEAN pooling will not work with indice_weights!
                        (pooling_mode == MEAN && L > 0)
                                        ? 1.0 / L : 1.0;

                        __m256 scale_vec, scale_acc;
                        {% for i in range(kMaxVecsPerThread) %}
                            scale_vec = _mm256_set1_ps(scale_factor);
                            scale_acc = _mm256_mul_ps(acc[{{ i }}], scale_vec);
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(&output_acc[b * total_D + D_start + 8 * {{ i }}]), _mm256_cvtps_ph(scale_acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                        {% endfor %}
                    }
                }
                {% endfor %}
            }

            if (weight_ty == SparseType::FP16) {
                // 1 bytes per D, plus 2 * 2 bytes for fp16 scale/shift + 4 bytes for padding.
                const int64_t D_bytes = D * 2;
                {% for kMaxVecsPerThread in range(1, 33) %}
                if (D_vecs == {{ kMaxVecsPerThread }}) {
                    for (int32_t b = 0; b < B; ++b) {
                        int32_t indices_start = offsets_acc[t * B + b];
                        int32_t indices_end = offsets_acc[t * B + b + 1];
                        int32_t L = indices_end - indices_start;
                        std::array<__m256, {{ kMaxVecsPerThread }} > acc;
                        // TODO: try fbgemm when I figure out how to adjust the bias/scale setting (first vs last element) and cache the codegen state.
                        {% for i in range(kMaxVecsPerThread) %}
                            acc[{{ i }}] = _mm256_setzero_ps();
                        {% endfor %}
                        int32_t l = 0;
                        int32_t LUnroll = (L / 2) * 2;
                        for (; l < LUnroll; l += 2) {
                            int64_t idx0 = indices_acc[indices_start + l + 0];
                            int64_t idx1 = indices_acc[indices_start + l + 1];

                            const __m128i* row0 = idx0 == -1 ? reinterpret_cast<const __m128i*>(zero_row.data()) : reinterpret_cast<const __m128i*>(&weights[idx0 * D_bytes]);
                            const __m128i* row1 = idx1 == -1 ? reinterpret_cast<const __m128i*>(zero_row.data()) : reinterpret_cast<const __m128i*>(&weights[idx1 * D_bytes]);

                            int64_t prefetch_idx0 = indices_acc[std::min<int32_t>(indices_start + l + 2, num_indices_m_1)];
                            int64_t prefetch_idx1 = indices_acc[std::min<int32_t>(indices_start + l + 3, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx0 * D_bytes], _MM_HINT_T0);
                            _mm_prefetch(&weights[prefetch_idx1 * D_bytes], _MM_HINT_T0);

                            {% if weighted %}
                            auto scale0 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 0]);
                            auto scale1 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 1]);

                            {% endif %}
                            {% for i in range(kMaxVecsPerThread) %}
                                {% if weighted %}
                                acc[{{ i }}] = _mm256_fmadd_ps(scale0, _mm256_cvtph_ps(row0[{{ i }}]), acc[{{ i }}]);
                                acc[{{ i }}] = _mm256_fmadd_ps(scale1, _mm256_cvtph_ps(row1[{{ i }}]), acc[{{ i }}]);
                                {% else %}
                                acc[{{ i }} ] = _mm256_add_ps(_mm256_cvtph_ps(row0[{{ i }}]), acc[{{ i }} ]);
                                acc[{{ i }} ] = _mm256_add_ps(_mm256_cvtph_ps(row1[{{ i }}]), acc[{{ i }} ]);
                                {% endif %}
                            {% endfor %}
                        }
                        for (; l < L; ++l) {
                            int64_t idx = indices_acc[indices_start + l];
                            const __m128i* row = idx == -1 ? reinterpret_cast<const __m128i*>(zero_row.data()) : reinterpret_cast<const __m128i*>(&weights[idx * D_bytes]);

                            int64_t prefetch_idx = indices_acc[std::min<int32_t>(indices_start + l + 1, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx * D_bytes], _MM_HINT_T0);

                            {% if weighted %}
                            auto scale = _mm256_set1_ps(indice_weights_acc[indices_start + l]);
                            {% endif %}

                            {% for i in range(kMaxVecsPerThread) %}
                                {% if weighted %}
                                acc[{{ i }}] = _mm256_fmadd_ps(scale, _mm256_cvtph_ps(row[{{ i }}]), acc[{{ i }}]);
                                {% else %}
                                acc[{{ i }} ] = _mm256_add_ps(_mm256_cvtph_ps(row[{{ i }}]), acc[{{ i }} ]);
                                {% endif %}
                            {% endfor %}
                        }

                        const float scale_factor =
                        // NOTE: MEAN pooling will not work with indice_weights!
                        (pooling_mode == MEAN && L > 0)
                                        ? 1.0 / L : 1.0;

                        __m256 scale_vec, scale_acc;
                        {% for i in range(kMaxVecsPerThread) %}
                            scale_vec = _mm256_set1_ps(scale_factor);
                            scale_acc = _mm256_mul_ps(acc[{{ i }}], scale_vec);
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(&output_acc[b * total_D + D_start + 8 * {{ i }}]), _mm256_cvtps_ph(scale_acc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                        {% endfor %}
                    }
                }
                {% endfor %}
            }
        }
        return;
    });
    return output;

}

#define BIG_CONSTANT(x) (x##LLU)

inline uint32_t pruned_hash_function(int32_t key, int32_t table) {
    uint64_t k = (static_cast<uint64_t>(key) << 32) | static_cast<uint64_t>(table);
    k ^= k >> 33;
    k *= BIG_CONSTANT(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return static_cast<uint32_t>(k >> 32);
}

void pruned_hashmap_insert_{{ wdesc }}_cpu(
    Tensor indices,
    Tensor dense_indices,
    Tensor offsets,
    Tensor hash_table,
    int64_t T) {

    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    const auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();

    const auto* offsets_acc = offsets.data_ptr<int32_t>();
    auto hash_table_acc = hash_table.accessor<int32_t, 2>();
    uint32_t capacity = hash_table.size(0);
    for (int32_t t = 0; t < T; ++t) {
        for (int32_t b = 0; b < B; ++b) {
            int32_t indices_start = offsets_acc[t * B + b];
            int32_t indices_end = offsets_acc[t * B + b + 1];
            int32_t L = indices_end - indices_start;
            for (int32_t l = 0; l < L; ++l) {
                int32_t idx = indices_acc[indices_start + l];
                int32_t dense_idx = dense_indices_acc[indices_start + l];

                uint32_t slot = static_cast<uint32_t>(pruned_hash_function(idx, t)) % capacity;
                while (true) {
                    int32_t sidx = hash_table_acc[slot][0];
                    int32_t stable = hash_table_acc[slot][1];

                    // empty slot
                    if (sidx == -1) {
                        hash_table_acc[slot][0] = idx;
                        hash_table_acc[slot][1] = t;
                        hash_table_acc[slot][2] = dense_idx;
                        break;
                    }
                    // already exists (shouldn't happen in practice)
                    if (sidx == idx && stable == t) {
                        hash_table_acc[slot][2] = dense_idx;
                        break;
                    }
                    // linear probe
                    slot = (slot + 1) % capacity;
                }
            }
        }
    }
    return;
}

Tensor pruned_hashmap_lookup_{{ wdesc }}_cpu(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    int64_t T) {

    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    auto dense_indices = empty_like(indices);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();

    const auto* offsets_acc = offsets.data_ptr<int32_t>();
    const auto hash_table_acc = hash_table.accessor<int32_t, 2>();
    int32_t capacity = hash_table.size(0);
    for (int32_t t = 0; t < T; ++t) {
        for (int32_t b = 0; b < B; ++b) {
            int32_t indices_start = offsets_acc[t * B + b];
            int32_t indices_end = offsets_acc[t * B + b + 1];
            int32_t L = indices_end - indices_start;
            for (int32_t l = 0; l < L; ++l) {
                int32_t idx = indices_acc[indices_start + l];

                uint32_t slot = static_cast<uint32_t>(pruned_hash_function(idx, t)) % capacity;
                while (true) {
                    int32_t sidx = hash_table_acc[slot][0];
                    int32_t stable = hash_table_acc[slot][1];

                    // empty slot
                    if (sidx == -1) {
                        dense_indices_acc[indices_start + l] = -1;
                        break;
                    }
                    // already exists
                    if (sidx == idx && stable == t) {
                        dense_indices_acc[indices_start + l] = hash_table_acc[slot][2];
                        break;
                    }
                    // linear probe
                    slot = (slot + 1) % capacity;
                }
            }
        }
    }
    return dense_indices;
}

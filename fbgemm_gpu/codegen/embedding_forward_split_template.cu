/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

{#
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
// Note: clang-format off doesn't work with this templaterized code,
// so we need to keep lint-ignore-every.
// See https://fburl.com/dw9ljh4h
#}

{% set wdesc =  "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_forward_template_helpers.cuh"
#include <unistd.h>
#include <limits.h>

#define SHFL_SYNC(val, srcLane) shfl_sync(val, srcLane, kThreadGroupSize, shfl_sync_mask)

{% if not dense %}
constexpr int32_t kCacheLocationMissing = -1;
{% endif %}

constexpr size_t kForwardMaxThreads = 512;

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{% if not weighted %}
template <
    typename emb_t,
    typename cache_t,
    {% if not dense %}
    typename output_t,
    {% endif %}
    typename index_t,
    size_t kThreadGroupSize
    >
__launch_bounds__(kForwardMaxThreads)
__global__ void {{ "dense" if dense else "split" }}_embedding_nobag_codegen_forward_unweighted_small_kernel(
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {% if not dense %}
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    int64_t D,
    FixedDivisor fd_B,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {% if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    at::PackedTensorAccessor32<output_t, 2, at::RestrictPtrTraits>
        output // [B][total_D],
    {% else %}
    at::PackedTensorAccessor32<at::acc_type<cache_t,true>, 2, at::RestrictPtrTraits>
        output // [B][total_D],
    {% endif %}
    ) {
    int32_t T = weights_offsets.size(0);
    int32_t B = (offsets.size(0) - 1) / T;
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (b_t >= B * T) {
        return;
    }
    int32_t t;
    int32_t b;
    fd_B.DivMod(b_t, &t, &b);
    int64_t weights_offset = weights_offsets[t];
    index_t indices_start = offsets[t * B + b];
    index_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    const emb_t* __restrict__ weights;
    {% if not dense %}
    const auto placement = static_cast<PlacementType>(weights_placements[t]);
    if (placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset];
    } else {
        weights = &uvm_weights[weights_offset];
    }
    {% else %}
    weights = &dev_weights[weights_offset];
    {% endif %}

    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }

    const int32_t group_start = threadIdx.x / kThreadGroupSize * kThreadGroupSize;
    const int32_t group_end = group_start + kThreadGroupSize;
    const int32_t d = threadIdx.x % kThreadGroupSize * 4;

    for (int32_t l_start = 0; l_start < L; l_start += kWarpSize) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {% if not dense %}
        int32_t cache_idx = (placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {% endif %}
        for (auto j = group_start; j < group_end && l_start + j < L; ++j) {
            int64_t idx_j = shfl_sync(idx, j);
            int64_t output_j = indices_start + l_start + j;
            {% if not dense %}
            int32_t cache_idx_j = shfl_sync(cache_idx, j);
            {% endif %}

            {% if not dense %}
            auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
                const_cast<emb_t*>(&weights[idx_j * D_emb]),
                const_cast<cache_t*>(&lxu_cache_weights[cache_idx_j][0]),
                D,
                nullptr);
            // assume cache is fp16/fp32 which doesn't require qparams
            float2 qparams_cache = make_float2(0.0f, 0.0f);

            {% endif %}
            auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
                const_cast<emb_t*>(&weights[idx_j * D_emb]),
                nullptr,
                D,
                nullptr);
            float2 qparams_emb;
            if (std::is_same<emb_t, uint8_t>::value) {
                qparams_emb = weight_row_emb.load_qparams();
            }

            if (d < D) {
                {% if not dense %}
                if (placement == PlacementType::MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                    Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                    weight.store(&output[output_j][d]);
                } else {
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    weight.store(&output[output_j][d]);
                }
                {% else %}
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    weight.store(&output[output_j][d]);
                {% endif %}
            }
        }
    }
}
{% endif %}

{% for nobag in [True, False] %}
{% if not nobag or not weighted %}
template <
    typename emb_t,
    typename cache_t,
    {% if not dense %}
    typename output_t,
    bool use_lxu_cache,
    {% endif %}
    typename index_t,
    {% if not nobag %}
    size_t kMaxVecsPerThread,
    {% endif %}
    size_t kThreadGroupSize = kWarpSize
    >
__launch_bounds__(kForwardMaxThreads)
__global__ void {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel(
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {% if not dense %}
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>
        lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {% if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {% else %}
    int64_t D,
    {% endif %}
    FixedDivisor fd_B,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {% if not nobag %}
    int64_t pooling_mode,
    {% endif %}
    {% if weighted %}
    at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>
        indice_weights,
    {% endif %}
    {% if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        lxu_cache_locations,
    at::PackedTensorAccessor32<output_t, 2, at::RestrictPtrTraits>
        output // [B][total_D],
    {% else %}
    at::PackedTensorAccessor32<at::acc_type<cache_t,true>, 2, at::RestrictPtrTraits>
        output // [B][total_D],
    {% endif %}
    ) {
    {% if not nobag %}
    int32_t B = output.size(0);
    int32_t T = D_offsets.size(0) - 1;
    {% else %}
    int32_t T = weights_offsets.size(0);
    int32_t B = (offsets.size(0) - 1) / T;
    {% endif %}
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (b_t >= B * T) {
        return;
    }
    int32_t t;
    int32_t b;
    fd_B.DivMod(b_t, &t, &b);
    int64_t weights_offset = weights_offsets[t];
    {% if not nobag %}
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;
    {% endif %}
    index_t indices_start = offsets[t * B + b];
    index_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    const emb_t* __restrict__ weights;
    {% if not dense %}
    const auto placement = static_cast<PlacementType>(weights_placements[t]);
    if (placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset];
    } else {
        weights = &uvm_weights[weights_offset];
    }
    {% else %}
    weights = &dev_weights[weights_offset];
    {% endif %}

    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }

    constexpr int VEC_WIDTH = 4;
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif

    {% if not nobag %}
    Vec4T<cache_t> accumulators[kMaxVecsPerThread];
    {% endif %}
    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {% if not dense %}
        int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {% endif %}
        {% if weighted %}
        at::acc_type<cache_t, true> idx_weight = l < L ? indice_weights[indices_start + l] : 0;
        {% endif %}
        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
            int64_t idx_j = SHFL_SYNC(idx, j);
            {% if nobag %}
            int64_t output_j = indices_start + l_start + j;
            {% endif %}
            {% if not dense %}
            int32_t cache_idx_j = use_lxu_cache ? SHFL_SYNC(cache_idx, j) : 0;
            {% endif %}

            {% if weighted %}
            at::acc_type<cache_t, true> idx_weight_j = SHFL_SYNC(idx_weight, j);
            {% endif %}

            {% if not dense %}
            // use_lxu_cache is a compile time condition
            if (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
                    const_cast<emb_t*>(&weights[idx_j * D_emb]),
                    const_cast<cache_t*>(&lxu_cache_weights[cache_idx_j][0]),
                    D,
                    nullptr);
                // assume cache is fp16/fp32 which doesn't require qparams
                float2 qparams_cache = make_float2(0.0f, 0.0f);

                {% if not nobag %}
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                    {% if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {% else %}
                    accumulators[i].add_(weight);
                    {% endif %}
                }
                {% else %}
                for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                    int32_t d = i + threadIdx.x * VEC_WIDTH;
                    if (d < D) {
                        Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                        weight.store(&output[output_j][d]);
                    }
                }
                {% endif %}
            }
            else { // else row is not in cache
            {% endif %}
                auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
                    const_cast<emb_t*>(&weights[idx_j * D_emb]),
                    nullptr,
                    D,
                    nullptr);
                float2 qparams_emb;
                if (std::is_same<emb_t, uint8_t>::value) {
                    qparams_emb = weight_row_emb.load_qparams();
                }
                {% if not nobag %}
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    {% if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {% else %}
                    accumulators[i].add_(weight);
                    {% endif %}
                }
                {% else %}
                for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                    int32_t d = i + threadIdx.x * VEC_WIDTH;
                    if (d < D) {
                        Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                        weight.store(&output[output_j][d]);
                    }
                }
                {% endif %}
            {% if not dense %}
            } // else row is not in cache
            {% endif %}
        }
    }

    {% if not nobag %}
    {% if not dense %}
    if (!std::is_same<output_t, uint8_t>::value) {
        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
            if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && L != 0) {
                accumulators[i].mul_(1.0 / L);
            }
            int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            accumulators[i].store(&output[b][D_start + d]);
        }
    } else {
        // apply per feature row-wise int8
        float thread_local_min = std::numeric_limits<float>::max();
        float thread_local_max = std::numeric_limits<float>::lowest();
        float2 qparams;

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
            ++i) {
            if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && L != 0) {
                accumulators[i].mul_(1.0 / L);
            }
            thread_local_max = max(thread_local_max, vec4_max(accumulators[i]));
            thread_local_min = min(thread_local_max, vec4_min(accumulators[i]));
        }

        qparams = warp_find_qparams(thread_local_min, thread_local_max);
        int output_D_start = D_start + t * 8;
        int output_D_end = output_D_start + D;

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
            ++i) {
            int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            nearest_rounding_vector<output_t, cache_t>(&output[b][output_D_start + d], accumulators[i], qparams);
        }
        if (threadIdx.x == 0) {
            store_qparams_to_row(&output[b][output_D_end], qparams);
        }

    }
    {% else %}
    // no pooled embedding quantization fusion for dense embeddings
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
        int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
        if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && L != 0) {
            accumulators[i].mul_(1.0 / L);
        }
        accumulators[i].store(&output[b][D_start + d]);
    }
    {% endif %}
    {% endif %}
}

Tensor {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_cuda(
    Tensor dev_weights,
    {% if not dense %}
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    {% endif %}
    Tensor weights_offsets,
    {% if not nobag %}
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    {% else %}
    int64_t D,
    {% endif %}
    Tensor indices,
    Tensor offsets,
    {% if not nobag %}
    int64_t pooling_mode,
    {% endif %}
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    {% if not dense %}
    Tensor lxu_cache_locations,
    {% endif %}
    {% if not dense %}
    int64_t output_dtype,
    {% endif %}
    int64_t unused
) {
    TENSOR_ON_CUDA_GPU(dev_weights);
    {% if not dense %}
    TENSOR_ON_CUDA_GPU(uvm_weights);
    TENSOR_ON_CUDA_GPU(lxu_cache_weights);
    TENSOR_ON_CUDA_GPU(weights_placements);
    {% endif %}
    TENSOR_ON_CUDA_GPU(weights_offsets);
    {% if not nobag %}
    TENSOR_ON_CUDA_GPU(D_offsets);
    {% endif %}
    TENSOR_ON_CUDA_GPU(indices);
    TENSOR_ON_CUDA_GPU(offsets);
    {% if weighted %}
    TENSOR_ON_CUDA_GPU(indice_weights);
    {% endif %}
    {% if not dense %}
    TENSOR_ON_CUDA_GPU(lxu_cache_locations);
    {% endif %}

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    {% if not nobag %}
    int32_t T = D_offsets.numel() - 1;
    {% else %}
    int32_t total_L = indices.numel();
    int32_t T = weights_offsets.numel();
    {% endif %}
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B >= 0);
    {% if not nobag %}
    TORCH_CHECK(total_D > 0);
    TORCH_CHECK(total_D % 4 == 0);
    TORCH_CHECK(max_D <= {{ max_embedding_dim }});
    {% else %}
    TORCH_CHECK(D > 0);
    TORCH_CHECK(D % 4 == 0);
    {% endif %}

    Tensor output;
    {% if nobag %}
    {% if dense %}
        output = at::empty({total_L, D}, dev_weights.options().dtype(at::kFloat));
    {% else %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    int64_t adjusted_D = D;
    if (o_dtype == SparseType::INT8) {
        adjusted_D += T * kINT8QparamsBytes;
    }
    output = at::empty({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    {% endif %}
    {% else %}
    {% if dense %}
    if (dev_weights.scalar_type() == at::kHalf || dev_weights.scalar_type() == at::kByte) {
        output = at::empty({B, total_D}, dev_weights.options().dtype(at::kFloat));
    } else {
        output = at::empty({B, total_D}, dev_weights.options());
    }
    {% else %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    int64_t total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
        total_adjusted_D += T * kINT8QparamsBytes;
    }
    output = at::empty({B, total_adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));

    {% endif %}
    {% endif %}

    if (B == 0) {
        return output;
    }

#ifdef __HIP_PLATFORM_HCC__  // HIP Optimal Kernel
    /*
     * current limitations
     1. sparse, bag and unweighted
     2. embedding dims in [64, 128, 192, 256]
     3. yet to support mixed embedding dims (loosely guarded below)
     4. yet to support non-uniform table locations (all be on devs)
     5. yet to support duplicate tables from some cases in exact optim (fbgemm_gpu/split_embedding_configs.py)
     */
    {% if not nobag %}
    {% if not dense %}
    {% if not weighted %}

    std::set<int> D_emb_s {64, 128, 192, 256};

    // weight param cnt
    int64_t wcnts = dev_weights.numel();
    // mixed hypothesis
    bool mixed_ls = (total_D != (max_D * T));
    // execution guards
    bool guard_ex = (wcnts > 0 && !mixed_ls);

    // all Ts on device
    std::vector<int32_t> wplas(weights_placements.data_ptr<int32_t>(), weights_placements.data_ptr<int32_t>() + weights_placements.numel());
    bool all_devs = std::accumulate(wplas.begin(), wplas.end(), 0) == 0;
    // no duplicate in weight offsets (which is the case exact optim used sometimes)
    std::vector<int64_t> woffs(weights_offsets.data_ptr<int64_t>(), weights_offsets.data_ptr<int64_t>() + weights_offsets.numel());
    std::vector<int64_t>::iterator it = std::unique(woffs.begin(), woffs.end());
    // not support duplicated weights table yet
    bool no_dupt = (it == woffs.end());

    if (guard_ex)  guard_ex = all_devs && no_dupt;

    // row dims options
    bool dims_opt = (D_emb_s.find(max_D) != D_emb_s.end());

    if (guard_ex && (dev_weights.scalar_type() == at::ScalarType::Half || dev_weights.scalar_type() == at::ScalarType::Float) && dims_opt) {
        static int init_hsaco = 0;
        static hipModule_t hip_kernel_module;
        static hipFunction_t hip_kernel_func;

        constexpr uint32_t workgroup_size = 256;
        constexpr uint32_t wave_size = 64;

        uint32_t bags_per_workgroup = workgroup_size / wave_size;
        uint32_t grids[3] = {(B + bags_per_workgroup - 1) / bags_per_workgroup, (uint32_t)T, 1};
        uint32_t blocks[3] = {workgroup_size, 1, 1};
        int64_t E = wcnts / T / max_D;

        if (init_hsaco == 0) {
            hipError_t hip_err = hipModuleLoad(&hip_kernel_module, "hip_kernel/split_tbe_fwd_hip_kernel.hsaco");  // hip kernel object
            if (hip_err != hipSuccess) {
                char cwd[PATH_MAX];
                getcwd(cwd, sizeof(cwd));
                printf("[hiperror](%d) line:%d, fail to call,(%s), cwd:%s", (int) hip_err, __LINE__, hipGetErrorString(hip_err), cwd);
                exit(1);
            }
            std::string prec = dev_weights.scalar_type() == at::ScalarType::Half  ? "fp16" : "fp32";
            std::string hip_kernel_name = std::string("split_tbe_fwd_hip_kernel_") + prec + "_e" + std::to_string(max_D);

            hip_err = hipModuleGetFunction(&hip_kernel_func, hip_kernel_module, hip_kernel_name.c_str());
            printf("kernel function: %s, elem:%ld, E:%ld, B:%d, T:%d, blocks:%dx%dx%d, grids:%dx%dx%d\n",
                hip_kernel_name.c_str(), wcnts, E, B, T,
                blocks[0], blocks[1], blocks[2], grids[0], grids[1], grids[2]);
            if (hip_err != hipSuccess) {
                printf("[hiperror](%d) line:%d, fail to call,(%s)", (int) hip_err, __LINE__, hipGetErrorString(hip_err));
                exit(1);
            }
            init_hsaco = 1;
        }

        {
            struct {
                void    *   output;
                void    *   emb_table;
                void    *   indices;
                void    *   offsets;
                uint32_t    emb_dim;
                uint32_t    batch;
                uint32_t    num_rows;
                uint32_t    num_tables;
            } args;
            size_t arg_size = sizeof(args);
            void* kconf[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                &arg_size, HIP_LAUNCH_PARAM_END};
            args.output    = output.packed_accessor32<float,2,at::RestrictPtrTraits>().data();
            if(dev_weights.scalar_type() == at::ScalarType::Half)
                args.emb_table = dev_weights.packed_accessor64<at::Half, 1, at::RestrictPtrTraits>().data();
            else
                args.emb_table = dev_weights.packed_accessor64<float, 1, at::RestrictPtrTraits>().data();
            args.indices   = indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>().data();
            args.offsets   = offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>().data();
            args.emb_dim   = (uint32_t) max_D;
            args.batch     = (uint32_t) B;
            args.num_rows  = E;
            args.num_tables = (uint32_t)T;

            hipModuleLaunchKernel(hip_kernel_func,
                grids[0], grids[1], grids[2],
                blocks[0], blocks[1], blocks[2], 0, 0, NULL, (void **) &kconf);

            return output;
        }
    }
    {% endif %}  // not weighted
    {% endif %}  // not dense
    {% endif %}  // not nobag
#endif  // HIP Optimal Kernel

    {% if not dense %}
    DISPATCH_EMB_CACHE_OUTPUT_TYPES(
    {% else %}
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    {% endif %}
        dev_weights.scalar_type(),
        {% if not dense %}
        lxu_cache_weights.scalar_type(),
        output.scalar_type(),
        {% endif %}
        "batched_embedding{{ "_nobag" if nobag else "" }}_forward_kernel_2", [&] {
        {% if not dense %}
        // Check if LXU cache is used
        bool use_lxu_cache = lxu_cache_weights.numel() > 0;
        {% endif %}
        {% if not nobag %}
        {% for use_cache in ["false", "true"] %}
        // The dense case does not have cache so we have to generate code for
        // only one case (value of use_cache does not matter)
        {% if (not dense) or (use_cache == "true") %}
        {% if not dense %}
        if (use_lxu_cache == {{ use_cache }}) {
        {% endif %}
            // kMaxElemPerThread is # of elements handled by thread if we use a full warp for a row
            // We consider kMaxElemPerThread 1 and 2, and then a multiple of 4.
            {% for kMaxElemPerThread in range(1, max_embedding_dim // (items_per_warp // 4) + 1) %}
            {% if kMaxElemPerThread in [1, 2] or kMaxElemPerThread % 4 == 0 %}
            if (max_D <= {{ items_per_warp // 4 * kMaxElemPerThread }}) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = {{ kMaxElemPerThread }} / 4 >= 1 ? {{ kMaxElemPerThread }} / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / {{ kMaxElemPerThread }}, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif
                {% if not dense %}
                split_embedding_codegen_forward_{{ wdesc }}_kernel<emb_t, cache_t, output_t, {{ use_cache }}, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                {% else %}
                dense_embedding_codegen_forward_{{ wdesc }}_kernel<scalar_t, scalar_t, int64_t, kMaxVecsPerThread, kThreadGroupSize><<<
                {% endif %}
                    div_round_up((B * T), kForwardMaxThreads / kThreadGroupSize),
                    dim3(kThreadGroupSize, kForwardMaxThreads / kThreadGroupSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    dev_weights.packed_accessor64<{{ "scalar_t" if dense else "emb_t" }}, 1, at::RestrictPtrTraits>(),
                    {% if not dense %}
                    uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                    weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    FixedDivisor(B),
                    indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    pooling_mode,
                    {% if weighted %}
                    indice_weights.packed_accessor32<at::acc_type<{{ "scalar_t" if dense else "cache_t" }}, true>, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not dense %}
                    lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    output.packed_accessor32<
                        output_t,
                        2,
                        at::RestrictPtrTraits>()
                    );
                    {% else %}
                    output.packed_accessor32<
                        at::acc_type<scalar_t, true>,
                        2,
                        at::RestrictPtrTraits>()
                    );
                    {% endif %}

                return;
            }
            {% endif %}
            {% endfor %}
        {% if not dense %}
        } // if (use_lxu_cache == {{ use_cache }})
        {% endif %}
        {% endif %} // if (not dense) or (use_cache == "true")
        {% endfor %} // for use_cache in ["false", "true"]
        {% else %}
        {% for kEmbeddingSize in [4, 8, 16, 32] %}
        if (D <= {{ kEmbeddingSize }}) {
        {% if not dense %}
        split_embedding_nobag_codegen_forward_unweighted_small_kernel<emb_t, cache_t, output_t, int64_t, {{ kEmbeddingSize // 4 }}><<<
        {% else %}
        dense_embedding_nobag_codegen_forward_unweighted_small_kernel<scalar_t, scalar_t, int64_t, {{ kEmbeddingSize // 4 }}><<<
        {% endif %}
            div_round_up((B * T), kForwardMaxThreads / kWarpSize),
            dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            dev_weights.packed_accessor64<{{ "scalar_t" if dense else "emb_t" }}, 1, at::RestrictPtrTraits>(),
            {% if not dense %}
            uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
            lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
            weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            {% endif %}
            weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            D,
            FixedDivisor(B),
            indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            {% if not dense %}
            lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            output.packed_accessor32<
                output_t,
                2,
                at::RestrictPtrTraits>()
            );
            {% else %}
            output.packed_accessor32<
                at::acc_type<scalar_t, true>,
                2,
                at::RestrictPtrTraits>()
            );
            {% endif %}

            return;
        }
        {% endfor %}
        {% for use_cache in ["false", "true"] %}
        // The dense case does not have cache so we have to generate code for
        // only one case (value of use_cache does not matter)
        {% if (not dense) or (use_cache == "true") %}
        {% if not dense %}
        if (use_lxu_cache == {{ use_cache }}) {
            split_embedding_nobag_codegen_forward_unweighted_kernel<emb_t, cache_t, output_t, {{ use_cache }}, int64_t><<<
        {% else %}
            dense_embedding_nobag_codegen_forward_unweighted_kernel<scalar_t, scalar_t, int64_t><<<
        {% endif %}
                div_round_up((B * T), kForwardMaxThreads / kWarpSize),
                dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                dev_weights.packed_accessor64<{{ "scalar_t" if dense else "emb_t" }}, 1, at::RestrictPtrTraits>(),
                {% if not dense %}
                uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                {% endif %}
                weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                D,
                FixedDivisor(B),
                indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                {% if not dense %}
                lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                output.packed_accessor32<
                    output_t,
                    2,
                    at::RestrictPtrTraits>()
                );
                {% else %}
                output.packed_accessor32<
                    at::acc_type<scalar_t, true>,
                    2,
                    at::RestrictPtrTraits>()
                );
                {% endif %}

                return;
        {% if not dense %}
        } // if (use_lxu_cache == {{ use_cache }})
        {% endif %}
        {% endif %} // if (not dense) or (use_cache == "true")
        {% endfor %} // for use_cache in ["false", "true"]
        {% endif %}
        });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
{% endif %}
{% endfor %}
    // clang-format on

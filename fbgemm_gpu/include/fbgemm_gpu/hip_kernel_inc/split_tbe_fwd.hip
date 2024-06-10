/*******************************************************************************
 * Copyright (c) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 ******************************************************************************/
#ifdef __HIP_PLATFORM_HCC__

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../codegen/embedding_forward_template_helpers_hip.cuh"

typedef int32_t  int32x4_t __attribute__((ext_vector_type(4)));
typedef float  floatx2_t __attribute__((ext_vector_type(2)));
typedef float  floatx4_t __attribute__((ext_vector_type(4)));

#define AMDGCN_BUFFER_RES_3 0x00027000
#define AMDGCN_WAVE_SIZE 64

template <typename T>
union amdgcn_buffer_resource{
    // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
    int32x4_t content;
    struct {
        T * address;
        int32_t range;
        int32_t config;
    };
};

template <typename T>
__device__ int32x4_t amdgcn_make_buffer_resource(const T* addr)
{
    amdgcn_buffer_resource<T> buffer_resource;
    buffer_resource.address = const_cast<T*>(addr);
    buffer_resource.range = 0xffffffff;
    buffer_resource.config = AMDGCN_BUFFER_RES_3;  // for gfx9

    return buffer_resource.content;
}

// buffer load fp32
__device__ half
llvm_amdgcn_raw_buffer_load_fp16(int32x4_t srsrc,
                                 int32_t voffset,
                                 int32_t soffset,
                                 int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f16");

__device__ float
llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 int32_t voffset,
                                 int32_t soffset,
                                 int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ half2
llvm_amdgcn_raw_buffer_load_fp16x2(int32x4_t srsrc,
                                   int32_t voffset,
                                   int32_t soffset,
                                   int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f16");

__device__ floatx4_t
llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc,
                                   int32_t voffset,
                                   int32_t soffset,
                                   int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                  int32x4_t rsrc,
                                  int32_t voffset,
                                  int32_t soffset,
                                  int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32x2(floatx2_t vdata,
                                    int32x4_t rsrc,
                                    int32_t voffset,
                                    int32_t soffset,
                                    int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32x4(floatx4_t vdata,
                                    int32x4_t rsrc,
                                    int32_t voffset,
                                    int32_t soffset,
                                    int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f32");

/******************************************************************************/

#define THREADS_PER_ROW 64
#define BLOCK_SIZE 256

template <typename emb_t, int32_t embedding_dim, typename index_t>
struct load_row_per_warp {
    static __device__ void run(emb_t * emb_data, index_t row_index, const emb_t * p_emb_table, int lane_id, uint32_t emb_dim) {}
};

// 64, 128, 192, 256, 384, 512, 640, 768, 896, 1024
template<typename index_t>
struct load_row_per_warp<float, 64, index_t> {
    static __device__ void run(float * emb_data, index_t row_index, const float * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        if(lane_id < emb_dim)
            emb_data[0] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, lane_id * sizeof(float), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<float, 128, index_t> {
    static __device__ void run(float * emb_data, index_t row_index, const float * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        emb_data[0] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, lane_id * sizeof(float), 0, 0);
        if((lane_id + 64) < emb_dim)
            emb_data[1] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 64) * sizeof(float), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<float, 192, index_t> {
    static __device__ void run(float * emb_data, index_t row_index, const float * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        emb_data[0] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, lane_id * sizeof(float), 0, 0);
        emb_data[1] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 64) * sizeof(float), 0, 0);
        if((lane_id + 2*64) < emb_dim)
            emb_data[2] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 2*64) * sizeof(float), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<float, 256, index_t> {
    static __device__ void run(float * emb_data, index_t row_index, const float * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        emb_data[0] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, lane_id * sizeof(float), 0, 0);
        emb_data[1] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 64) * sizeof(float), 0, 0);
        emb_data[2] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 2*64) * sizeof(float), 0, 0);
        if((lane_id + 3*64) < emb_dim)
            emb_data[3] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 3*64) * sizeof(float), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<float, 384, index_t> {
    static __device__ void run(float * emb_data, index_t row_index, const float * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        emb_data[0] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, lane_id * sizeof(float), 0, 0);
        emb_data[1] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 64) * sizeof(float), 0, 0);
        emb_data[2] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 2*64) * sizeof(float), 0, 0);
        emb_data[3] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 3*64) * sizeof(float), 0, 0);
        
        if((lane_id + 4*64) < emb_dim)
            emb_data[4] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 4*64) * sizeof(float), 0, 0);
        if((lane_id + 5*64) < emb_dim)
            emb_data[5] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 5*64) * sizeof(float), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<float, 512, index_t> {
    static __device__ void run(float * emb_data, index_t row_index, const float * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        emb_data[0] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, lane_id * sizeof(float), 0, 0);
        emb_data[1] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 64) * sizeof(float), 0, 0);
        emb_data[2] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 2*64) * sizeof(float), 0, 0);
        emb_data[3] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 3*64) * sizeof(float), 0, 0);
        emb_data[4] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 4*64) * sizeof(float), 0, 0);
        emb_data[5] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 5*64) * sizeof(float), 0, 0);

        if((lane_id + 6*64) < emb_dim)
            emb_data[6] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 6*64) * sizeof(float), 0, 0);
        if((lane_id + 7*64) < emb_dim)
            emb_data[7] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 7*64) * sizeof(float), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<float, 640, index_t> {
    static __device__ void run(float * emb_data, index_t row_index, const float * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<floatx4_t*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, lane_id * 4 * sizeof(float), 0, 0);
        *reinterpret_cast<floatx4_t*>(&emb_data[4]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, (lane_id + 64) * 4 * sizeof(float), 0, 0);

        if((lane_id + 8*64) < emb_dim)
            emb_data[8] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 8*64) * sizeof(float), 0, 0);
        if((lane_id + 9*64) < emb_dim)
            emb_data[9] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 9*64) * sizeof(float), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<float, 768, index_t> {
    static __device__ void run(float * emb_data, index_t row_index, const float * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<floatx4_t*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, lane_id * 4 * sizeof(float), 0, 0);
        *reinterpret_cast<floatx4_t*>(&emb_data[4]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, (lane_id + 64) * 4 * sizeof(float), 0, 0);
        emb_data[8] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 8*64) * sizeof(float), 0, 0);
        emb_data[9] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 9*64) * sizeof(float), 0, 0);

        if((lane_id + 10*64) < emb_dim)
            emb_data[10] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 10*64) * sizeof(float), 0, 0);
        if((lane_id + 11*64) < emb_dim)
            emb_data[11] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 11*64) * sizeof(float), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<float, 896, index_t> {
    static __device__ void run(float * emb_data, index_t row_index, const float * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<floatx4_t*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, lane_id * 4 * sizeof(float), 0, 0);
        *reinterpret_cast<floatx4_t*>(&emb_data[4]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, (lane_id + 64) * 4 * sizeof(float), 0, 0);
        *reinterpret_cast<floatx4_t*>(&emb_data[8]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, (lane_id + 128) * 4 * sizeof(float), 0, 0);

        if((lane_id + 12*64) < emb_dim)
            emb_data[12] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 12*64) * sizeof(float), 0, 0);
        if((lane_id + 13*64) < emb_dim)
            emb_data[13] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 13*64) * sizeof(float), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<float, 1024, index_t> {
    static __device__ void run(float * emb_data, index_t row_index, const float * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<floatx4_t*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, lane_id * 4 * sizeof(float), 0, 0);
        *reinterpret_cast<floatx4_t*>(&emb_data[4]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, (lane_id + 64) * 4 * sizeof(float), 0, 0);
        *reinterpret_cast<floatx4_t*>(&emb_data[8]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, (lane_id + 128) * 4 * sizeof(float), 0, 0);
        emb_data[12] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 12*64) * sizeof(float), 0, 0);
        emb_data[13] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 13*64) * sizeof(float), 0, 0);

        if((lane_id + 14*64) < emb_dim)
            emb_data[14] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 14*64) * sizeof(float), 0, 0);
        if((lane_id + 15*64) < emb_dim)
            emb_data[15] = llvm_amdgcn_raw_buffer_load_fp32(emb_res, (lane_id + 15*64) * sizeof(float), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<half, 64, index_t> {
    static __device__ void run(half * emb_data, index_t row_index, const half * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        if(lane_id < emb_dim)
            emb_data[0] = llvm_amdgcn_raw_buffer_load_fp16(emb_res, lane_id * sizeof(half), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<half, 128, index_t> {
    static __device__ void run(half * emb_data, index_t row_index, const half * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        if((lane_id * 2) < emb_dim)
            *reinterpret_cast<half2*>(emb_data) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, lane_id * sizeof(half2), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<half, 192, index_t> {
    static __device__ void run(half * emb_data, index_t row_index, const half * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<half2*>(emb_data) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, lane_id * sizeof(half2), 0, 0);
        if((lane_id + 128) < emb_dim)
            emb_data[2] = llvm_amdgcn_raw_buffer_load_fp16(emb_res, (lane_id + 128) * sizeof(half), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<half, 256, index_t> {
    static __device__ void run(half * emb_data, index_t row_index, const half * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<half2*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, lane_id * sizeof(half2), 0, 0);
        if(((lane_id + 64 ) * 2) < emb_dim)
            *reinterpret_cast<half2*>(&emb_data[2]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, (lane_id + 64 )* sizeof(half2), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<half, 384, index_t> {
    static __device__ void run(half * emb_data, index_t row_index, const half * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<half2*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, lane_id * sizeof(half2), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[2]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, (lane_id + 64 )* sizeof(half2), 0, 0);
        if(((lane_id + 64 * 2) * 2) < emb_dim)
            *reinterpret_cast<half2*>(&emb_data[4]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, (lane_id + 64 * 2)* sizeof(half2), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<half, 512, index_t> {
    static __device__ void run(half * emb_data, index_t row_index, const half * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<half2*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, lane_id * sizeof(half2), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[2]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, (lane_id + 64 )* sizeof(half2), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[4]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, (lane_id + 64 * 2 )* sizeof(half2), 0, 0);
        if(((lane_id + 64 * 3) * 2) < emb_dim)
            *reinterpret_cast<half2*>(&emb_data[6]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, (lane_id + 64 * 3 )* sizeof(half2), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<half, 640, index_t> {
    static __device__ void run(half * emb_data, index_t row_index, const half * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<half2*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, lane_id * sizeof(half2), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[2]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, (lane_id + 64 )* sizeof(half2), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[4]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, (lane_id + 64 * 2 )* sizeof(half2), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[6]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, (lane_id + 64 * 3 )* sizeof(half2), 0, 0);
        if(((lane_id + 64 * 4) * 2) < emb_dim)
            *reinterpret_cast<half2*>(&emb_data[8]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, (lane_id + 64 * 4 )* sizeof(half2), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<half, 768, index_t> {
    static __device__ void run(half * emb_data, index_t row_index, const half * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<floatx4_t*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, lane_id * sizeof(floatx4_t), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[8]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res,  64 * sizeof(floatx4_t) + (lane_id )* sizeof(half2), 0, 0);
        if((512 + (lane_id + 64) * 2) < emb_dim)
            *reinterpret_cast<half2*>(&emb_data[10]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res,  64 * sizeof(floatx4_t) + (lane_id + 64)* sizeof(half2), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<half, 896, index_t> {
    static __device__ void run(half * emb_data, index_t row_index, const half * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<floatx4_t*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, lane_id * sizeof(floatx4_t), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[8]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res,  64 * sizeof(floatx4_t) + (lane_id )* sizeof(half2), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[10]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res,  64 * sizeof(floatx4_t) + (lane_id + 64)* sizeof(half2), 0, 0);
        if((512 + (lane_id + 64 * 2) * 2) < emb_dim)
            *reinterpret_cast<half2*>(&emb_data[12]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res,  64 * sizeof(floatx4_t) + (lane_id + 64 * 2)* sizeof(half2), 0, 0);
    }
};

template<typename index_t>
struct load_row_per_warp<half, 1024, index_t> {
    static __device__ void run(half * emb_data, index_t row_index, const half * p_emb_table, int lane_id, uint32_t emb_dim) {
        int32x4_t emb_res = amdgcn_make_buffer_resource(p_emb_table + row_index * emb_dim);
        *reinterpret_cast<floatx4_t*>(&emb_data[0]) = llvm_amdgcn_raw_buffer_load_fp32x4(emb_res, lane_id * sizeof(floatx4_t), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[8]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res,  64 * sizeof(floatx4_t) + (lane_id )* sizeof(half2), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[10]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res,  64 * sizeof(floatx4_t) + (lane_id + 64)* sizeof(half2), 0, 0);
        *reinterpret_cast<half2*>(&emb_data[12]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res,  64 * sizeof(floatx4_t) + (lane_id + 64 * 2)* sizeof(half2), 0, 0);
        if((512 + (lane_id + 64 * 3) * 2) < emb_dim)
            *reinterpret_cast<half2*>(&emb_data[14]) = llvm_amdgcn_raw_buffer_load_fp16x2(emb_res,  64 * sizeof(floatx4_t) + (lane_id + 64 * 3)* sizeof(half2), 0, 0);
    }
};

template<typename emb_t, int32_t embedding_dim, typename output_t, bool weighted>
struct accumulate_row_per_warp {
    static constexpr int dword_per_row = (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    static __device__ void run(output_t * acc, emb_t * emb_data, int lane_id, float row_weight = 1.0) {
        if constexpr (!weighted) {
            #pragma unroll
            for(int i = 0; i < dword_per_row; i++){
                acc[i] += static_cast<output_t>(emb_data[i]);
            }
        } else {
            #pragma unroll
            for(int i = 0; i < dword_per_row; i++){
                acc[i] += static_cast<output_t>((float)emb_data[i] * row_weight);
            }
        }
    }
};

template<typename emb_t, int32_t embedding_dim, typename output_t>
struct store_row_per_warp {
    static constexpr int dword_per_row = (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    static __device__ void run(output_t * acc, output_t * p_output, int lane_id, uint32_t emb_dim)
    {
        #pragma unroll
        for(int i = 0; i < dword_per_row; i++){
            p_output[lane_id + i * THREADS_PER_ROW] = acc[i];
        }
    }
};

// common pattern for fp16/fp32
template<typename emb_t>
struct store_row_per_warp<emb_t, 64, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        if(lane_id < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[0], out_res, lane_id * sizeof(float), 0, 0);
    }
};

// fp32
template<>
struct store_row_per_warp<float, 128, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32(acc[0], out_res, lane_id * sizeof(float), 0, 0);
        if((lane_id + 64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[1], out_res, (lane_id + 64) * sizeof(float), 0, 0);
    }
};

template<>
struct store_row_per_warp<float, 192, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32(acc[0], out_res, lane_id * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[1], out_res, (lane_id + 64) * sizeof(float), 0, 0);
        if((lane_id + 2*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[2], out_res, (lane_id + 2*64) * sizeof(float), 0, 0);
    }
};


template<>
struct store_row_per_warp<float, 256, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32(acc[0], out_res, lane_id * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[1], out_res, (lane_id + 64) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[2], out_res, (lane_id + 2*64) * sizeof(float), 0, 0);
        if((lane_id + 3*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[3], out_res, (lane_id + 3*64) * sizeof(float), 0, 0);
    }
};

template<>
struct store_row_per_warp<float, 384, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32(acc[0], out_res, lane_id * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[1], out_res, (lane_id + 64) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[2], out_res, (lane_id + 2*64) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[3], out_res, (lane_id + 3*64) * sizeof(float), 0, 0);
        if((lane_id + 4*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[4], out_res, (lane_id + 4*64) * sizeof(float), 0, 0);
        if((lane_id + 5*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[5], out_res, (lane_id + 5*64) * sizeof(float), 0, 0);
    }
};

template<>
struct store_row_per_warp<float, 512, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32(acc[0], out_res, lane_id * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[1], out_res, (lane_id + 64) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[2], out_res, (lane_id + 2*64) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[3], out_res, (lane_id + 3*64) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[4], out_res, (lane_id + 4*64) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[5], out_res, (lane_id + 5*64) * sizeof(float), 0, 0);
        if((lane_id + 6*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[6], out_res, (lane_id + 6*64) * sizeof(float), 0, 0);
        if((lane_id + 7*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[7], out_res, (lane_id + 7*64) * sizeof(float), 0, 0);
    }
};

template<>
struct store_row_per_warp<float, 640, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[0]), out_res, lane_id * sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[4]), out_res, (lane_id + 64) * sizeof(floatx4_t), 0, 0);
        if((lane_id + 8*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[8], out_res, (lane_id + 8*64) * sizeof(float), 0, 0);
        if((lane_id + 9*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[9], out_res, (lane_id + 9*64) * sizeof(float), 0, 0);
    }
};

template<>
struct store_row_per_warp<float, 768, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[0]), out_res, lane_id * sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[4]), out_res, (lane_id + 64) * sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[8], out_res, (lane_id + 8*64) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[9], out_res, (lane_id + 9*64) * sizeof(float), 0, 0);
        if((lane_id + 10*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[10], out_res, (lane_id + 10*64) * sizeof(float), 0, 0);
        if((lane_id + 11*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[11], out_res, (lane_id + 11*64) * sizeof(float), 0, 0);
    }
};

template<>
struct store_row_per_warp<float, 896, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[0]), out_res, lane_id * sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[4]), out_res, (lane_id + 64) * sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[8]), out_res, (lane_id + 64 * 2) * sizeof(floatx4_t), 0, 0);
        if((lane_id + 12*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[12], out_res, (lane_id + 12*64) * sizeof(float), 0, 0);
        if((lane_id + 13*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[13], out_res, (lane_id + 13*64) * sizeof(float), 0, 0);
    }
};

template<>
struct store_row_per_warp<float, 1024, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[0]), out_res, lane_id * sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[4]), out_res, (lane_id + 64) * sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[8]), out_res, (lane_id + 64 * 2) * sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[12], out_res, (lane_id + 12*64) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32(acc[13], out_res, (lane_id + 13*64) * sizeof(float), 0, 0);
        if((lane_id + 14*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[14], out_res, (lane_id + 14*64) * sizeof(float), 0, 0);
        if((lane_id + 15*64) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[15], out_res, (lane_id + 15*64) * sizeof(float), 0, 0);
    }
};

// fp16
template<>
struct store_row_per_warp<half, 128, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        if((lane_id * 2) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(acc), out_res, lane_id * sizeof(floatx2_t), 0, 0);
    }
};

template<>
struct store_row_per_warp<half, 192, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(acc), out_res, lane_id * sizeof(floatx2_t), 0, 0);
        if((lane_id + 128) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32(acc[2], out_res, (lane_id + 128 )* sizeof(float), 0, 0);
    }
};


template<>
struct store_row_per_warp<half, 256, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[0]), out_res, lane_id * sizeof(floatx2_t), 0, 0);
        if(((lane_id + 64 ) * 2) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[2]), out_res, (lane_id + 64) * sizeof(floatx2_t), 0, 0);
    }
};

template<>
struct store_row_per_warp<half, 384, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[0]), out_res, lane_id * sizeof(floatx2_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[2]), out_res, (lane_id + 64) * sizeof(floatx2_t), 0, 0);
        if(((lane_id + 64 * 2 ) * 2) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[4]), out_res, (lane_id + 64 * 2) * sizeof(floatx2_t), 0, 0);
    }
};

template<>
struct store_row_per_warp<half, 512, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[0]), out_res, lane_id * sizeof(floatx2_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[2]), out_res, (lane_id + 64) * sizeof(floatx2_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[4]), out_res, (lane_id + 64 * 2) * sizeof(floatx2_t), 0, 0);
        if(((lane_id + 64 * 3 ) * 2) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[6]), out_res, (lane_id + 64 * 3) * sizeof(floatx2_t), 0, 0);
    }
};

template<>
struct store_row_per_warp<half, 640, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[0]), out_res, lane_id * sizeof(floatx2_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[2]), out_res, (lane_id + 64) * sizeof(floatx2_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[4]), out_res, (lane_id + 64 * 2) * sizeof(floatx2_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[6]), out_res, (lane_id + 64 * 3) * sizeof(floatx2_t), 0, 0);
        if(((lane_id + 64 * 4 ) * 2) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[8]), out_res, (lane_id + 64 * 4) * sizeof(floatx2_t), 0, 0);
    }
};

template<>
struct store_row_per_warp<half, 768, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[0]), out_res, lane_id * 2 *  sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[4]), out_res, lane_id * 2 *  sizeof(floatx4_t) + sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[8]), out_res, 64*2*sizeof(floatx4_t) + (lane_id) * sizeof(floatx2_t), 0, 0);
        if((512 + (lane_id + 64) * 2) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[10]), out_res, 64*2*sizeof(floatx4_t) + (lane_id + 64) * sizeof(floatx2_t), 0, 0);
    }
};

template<>
struct store_row_per_warp<half, 896, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[0]), out_res, lane_id * 2 *  sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[4]), out_res, lane_id * 2 *  sizeof(floatx4_t) + sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[8]), out_res, 64*2*sizeof(floatx4_t) + (lane_id) * sizeof(floatx2_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[10]), out_res, 64*2*sizeof(floatx4_t) + (lane_id + 64) * sizeof(floatx2_t), 0, 0);
        if((512 + (lane_id + 64 * 2) * 2) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[12]), out_res, 64*2*sizeof(floatx4_t) + (lane_id + 64 * 2) * sizeof(floatx2_t), 0, 0);
    }
};

template<>
struct store_row_per_warp<half, 1024, float> {
    static __device__ void run(float * acc, float * p_output, int lane_id, uint32_t emb_dim)
    {
        int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[0]), out_res, lane_id * 2 *  sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x4(*reinterpret_cast<floatx4_t*>(&acc[4]), out_res, lane_id * 2 *  sizeof(floatx4_t) + sizeof(floatx4_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[8]), out_res, 64*2*sizeof(floatx4_t) + (lane_id) * sizeof(floatx2_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[10]), out_res, 64*2*sizeof(floatx4_t) + (lane_id + 64) * sizeof(floatx2_t), 0, 0);
        llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[12]), out_res, 64*2*sizeof(floatx4_t) + (lane_id + 64 * 2) * sizeof(floatx2_t), 0, 0);
        if((512 + (lane_id + 64 * 3) * 2) < emb_dim)
            llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t*>(&acc[14]), out_res, 64*2*sizeof(floatx4_t) + (lane_id + 64 * 3) * sizeof(floatx2_t), 0, 0);
    }
};

template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    int32_t embedding_dim,
    int32_t bag_prefetch,
    int32_t bag_unroll,
    bool    weighted>
__device__ void split_tbe_forward_hip_kernel(
    output_t * p_output,
    const emb_t * p_emb_table,
    const index_t * p_indices,
    const index_t * p_offsets,
    const int64_t pooling_mode,
    uint32_t emb_dim,
    uint32_t batch,
    uint32_t num_rows,
    uint32_t num_tables,
    const float * p_indice_weights = nullptr)
{
    constexpr uint32_t dword_output_per_row = (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    // constexpr uint32_t input_data_per_dword = 4 / sizeof(emb_t);    // TODO: larger than 4 byte
    // constexpr uint32_t dword_input_per_row = (dword_output_per_row + input_data_per_dword - 1) / input_data_per_dword;
    // constexpr uint32_t dword_input_per_row_rounded = dword_input_per_row == 1 ? dword_input_per_row
    //                                  : ((dword_input_per_row + 1) / 2 * 2); // round to 2x
    constexpr uint32_t length_mask = ~(bag_unroll - 1);

    static_assert(bag_prefetch < bag_unroll, "");

    float accumulator[dword_output_per_row];
    index_t indices[bag_unroll];
    float indice_weights[bag_unroll];

    emb_t emb_data[dword_output_per_row * bag_prefetch];

    int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / AMDGCN_WAVE_SIZE);
    int bag_id = (blockIdx.x << 2) | wave_id;
    if(bag_id >= batch)
        return ;
    int lane_id = threadIdx.x & (AMDGCN_WAVE_SIZE - 1);

    p_offsets += blockIdx.y * batch + bag_id;
    index_t indices_start = p_offsets[0];
    index_t indices_end = p_offsets[1];

    uint64_t emb_table_stride = static_cast<uint64_t>(num_rows) * emb_dim;
    uint64_t out_bag_stride = num_tables * emb_dim;
    p_emb_table += blockIdx.y * emb_table_stride;
    p_output += blockIdx.y * emb_dim + bag_id * out_bag_stride;

    #pragma unroll
    for(int i=0; i < dword_output_per_row; i++)
    {
        accumulator[i] = .0f;
    }
    p_indices += indices_start;

    if constexpr (weighted) {
        p_indice_weights += indices_start;
    }

    int32_t length = indices_end - indices_start;
    int32_t length_mod = length & length_mask;

    int itr = 0;
    if(length_mod == 0)
        goto L_end;

    if constexpr (!weighted) {
        #pragma unroll
        for(int i=0; i < bag_unroll; i++){
            indices[i] = p_indices[i];
        }
    } else {
        #pragma unroll
        for(int i=0; i < bag_unroll; i++){
            indices[i] = p_indices[i];
	    indice_weights[i] = p_indice_weights[i];
        }
    }

    itr += bag_unroll;
    p_indices += bag_unroll;

    if constexpr (weighted) {
        p_indice_weights += bag_unroll;
    }

    // LOOP
    for( ; itr<length_mod; itr += bag_unroll){
        load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id, emb_dim);
        load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[1], p_emb_table, lane_id, emb_dim);

	if constexpr (!weighted) {
            #pragma unroll
            for(int j = 2 ; j < bag_unroll; j += 2){
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id, emb_dim);

                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id, emb_dim);
            }
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0], &emb_data[0], lane_id);
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);

            #pragma unroll
            for(int i=0; i < bag_unroll; i++){
                indices[i] = p_indices[i];
            }
            p_indices += bag_unroll;

        } else {    // row weighted
            #pragma unroll
            for(int j = 2 ; j < bag_unroll; j += 2){
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[j-2]);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id, emb_dim);

                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[j-1]);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id, emb_dim);
            }
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0], &emb_data[0], lane_id, indice_weights[bag_unroll-2]);
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[bag_unroll-1]);

            #pragma unroll
            for(int i=0; i < bag_unroll; i++){
                indices[i] = p_indices[i];
                indice_weights[i] = p_indice_weights[i];
            }
            p_indices += bag_unroll;
            p_indice_weights += bag_unroll;
        }
    }
    // LAST
    load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id, emb_dim);
    load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[1], p_emb_table, lane_id, emb_dim);

    if constexpr (!weighted) {
        #pragma unroll
        for(int j = 2 ; j < bag_unroll; j += 2){
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id, emb_dim);

            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id, emb_dim);
        }
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);

    } else {    // row weighted
        #pragma unroll
        for(int j = 2 ; j < bag_unroll; j += 2){
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[j-2]);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id, emb_dim);

            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[j-1]);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id, emb_dim);
        }
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[bag_unroll-2]);
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[bag_unroll-1]);
    }

L_end:
    if(length & (bag_unroll - 1)){
        if constexpr (!weighted) {
            // last, load one by one
            do {
                indices[0] = p_indices[0];
                p_indices++;

                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id, emb_dim);
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);

                itr++;
            } while (itr < length);
        } else {    // row weighted
            do {
                indices[0] = p_indices[0];
                indice_weights[0] = p_indice_weights[0];
                p_indices++;
                p_indice_weights++;

                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id, emb_dim);
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[0]);

                itr++;
            } while(itr < length);
        }
    }

    if (static_cast<fbgemm_gpu::PoolingMode>(pooling_mode) == fbgemm_gpu::PoolingMode::MEAN && length != 0){
#pragma unroll
        for (int i = 0; i < dword_output_per_row; i++){
            accumulator[i] *= 1.0f / length;
        }
    }

    // store out
    store_row_per_warp<emb_t, embedding_dim, output_t>::run(&accumulator[0], p_output, lane_id, emb_dim);
}


#define SPLIT_TBE_FWD_KERNEL(emb_prec, emb_type, embedding_dim, bag_prefetch, bag_unroll) \
    extern "C" __global__ void split_tbe_fwd_unweighted_hip_kernel_ ## emb_prec ## _e ## embedding_dim ( \
            float * p_output,              \
            const emb_type * p_emb_table,  \
            const int64_t * p_indices,     \
            const int64_t * p_offsets,     \
            const int64_t pooling_mode,    \
            uint32_t emb_dim,              \
            uint32_t batch,                \
            uint32_t num_rows,             \
            uint32_t num_tables)           \
    {                                      \
        split_tbe_forward_hip_kernel<emb_type, float, float, int64_t, embedding_dim, bag_prefetch, bag_unroll, false> \
                (p_output, p_emb_table, p_indices, p_offsets, pooling_mode, emb_dim, batch, num_rows, num_tables); \
    } \
    \
    extern "C" __global__ void split_tbe_fwd_weighted_hip_kernel_ ## emb_prec ## _e ## embedding_dim ( \
            float * p_output,              \
            const emb_type * p_emb_table,  \
            const int64_t * p_indices,     \
            const int64_t * p_offsets,     \
            const int64_t pooling_mode,    \
            const float * p_indice_weights,\
            uint32_t emb_dim,              \
            uint32_t batch,                \
            uint32_t num_rows,             \
            uint32_t num_tables)           \
    {                                      \
        split_tbe_forward_hip_kernel<emb_type, float, float, int64_t, embedding_dim, bag_prefetch, bag_unroll, true> \
                (p_output, p_emb_table, p_indices, p_offsets, pooling_mode, emb_dim, batch, num_rows, num_tables, p_indice_weights); \
    }


SPLIT_TBE_FWD_KERNEL(fp16,  half,  64, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 128, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 192, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 256, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 384, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 512, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 640, 2, 8)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 768, 2, 8)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 896, 2, 4)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 1024, 2, 4)

SPLIT_TBE_FWD_KERNEL(fp32, float,  64, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 128, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 192, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 256, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 384, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 512, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 640, 2, 8)
SPLIT_TBE_FWD_KERNEL(fp32, float, 768, 2, 8)
SPLIT_TBE_FWD_KERNEL(fp32, float, 896, 2, 4)
SPLIT_TBE_FWD_KERNEL(fp32, float, 1024, 2, 4)

#endif

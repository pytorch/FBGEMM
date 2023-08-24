#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdint.h>
#include <tuple>
#include <vector>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <ATen/ATen.h>
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "backward_template_helper.cuh"

using json = nlohmann::json;
using namespace fbgemm_gpu;

__global__ __launch_bounds__(kMaxThreads) void linearize_index_kernel_for_unique(
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> infos,
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> linear_indices,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    const uint32_t max_T,
    const uint32_t max_B,
    FixedDivisor fd) {
  const int32_t T = hash_size_cumsum.size(0) - 1;
  auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b;
  int32_t t;
  const auto total_B = offsets.size(0) - 1;
  bool valid = b_t < total_B;
  // info must be uint32_t (using auto will assign int32_t to info)
  uint32_t info = 0;

  fd.DivMod(b_t, &t, &b);

  const int64_t hash_offset = valid ? hash_size_cumsum[t] : -1;
  const int64_t indices_start = valid ? offsets[b_t] : -1;
  const int32_t L = valid ? offsets[b_t + 1] - indices_start : 0;
  const int32_t lane_id = threadIdx.x % kWarpSize;
    if (valid) {
        info = (reinterpret_cast<uint32_t*>(&t)[0] << info_B_num_bits) |
            reinterpret_cast<uint32_t*>(&b)[0];
    }
    for (int32_t j = 0; j < kWarpSize; ++j) {
        const int64_t indices_start_warp =
            fbgemm_gpu::shfl_sync(indices_start, j);
        const uint32_t info_warp = fbgemm_gpu::shfl_sync(info, j);
        const int32_t L_warp = fbgemm_gpu::shfl_sync(L, j);
        const int64_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
        for (int32_t i = lane_id; i < L_warp; i += kWarpSize) {
        const int64_t idx = __ldg(&indices[indices_start_warp + i]);
        reinterpret_cast<uint32_t*>(&infos[0])[indices_start_warp + i] =
            info_warp;
        linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
        }
    }
}


void load_input_int64_binary(std::string input_file, int64_t* dataset, int n_elem){
    std::ifstream myFile (input_file, std::ifstream::binary);
    if (myFile.is_open()){
        myFile.read ((char *)dataset, n_elem * sizeof(int64_t));
        myFile.close();
    }
    else{
        std::cout<<"ERROR:" << input_file << " Not exist!" << std::endl;
    }
}

at::Tensor load_float_tensor(
    std::string input_file,
    int64_t size
){
    std::cout << input_file << std::endl;

    auto load_tensor = at::zeros({size}, at::kFloat);

    std::ifstream myFile (input_file, std::ifstream::binary);
    myFile.read ((char *)(load_tensor.data_ptr<float>()), size * sizeof(float));

    return load_tensor;
}

void save_float_tensor(
    at::Tensor input_tensor,
    std::string output_file,
    int64_t size
){
    std::cout << output_file << std::endl;

    std::ofstream myFile (output_file, std::ofstream::binary);
    myFile.write ((char *)(input_tensor.data_ptr<float>()), size * sizeof(float));
}

void save_float(
    float* input_ptr,
    std::string output_file,
    int64_t size
){
    std::cout << output_file << std::endl;

    std::ofstream myFile (output_file, std::ofstream::binary);
    myFile.write ((char *)(input_ptr), size * sizeof(float));
}

class TBE_Dataloader
{
    public:
        TBE_Dataloader(
            std::string sharding_param_dir,
            std::string data_dir,
            int rank=0
        )
        {
            _rank = rank;
            _sharding_param_dir = sharding_param_dir;
            _data_dir = data_dir;

            std::string length_file = data_dir + "/index_and_offset_length_" + std::to_string(rank) + ".json";

            // Load length file and compute the total index size and offset size
            std::ifstream jsonFile(length_file);
            json j;
            jsonFile >> j;
            auto index_length = j["index"];
            auto offset_length = j["offset"];
            index_offset.push_back(0);
            offset_offset.push_back(0);
            for(int num : index_length){
                index_size += num;
                index_offset.push_back(index_size);
            }
            for(int num : offset_length){
                offset_size += num;
                offset_offset.push_back(offset_size);
            }
            nBatch = index_length.size();

            // Allocate memory and load index and offset
            h_index = (int64_t *) malloc (index_size * sizeof(int64_t));
            h_offset = (int64_t *) malloc (offset_size * sizeof(int64_t));
            std::string index_file = data_dir + "/indices_" + std::to_string(rank) + ".bin";
            std::string offset_file = data_dir + "/offsets_" + std::to_string(rank) + ".bin";
            load_input_int64_binary(index_file, h_index, index_size);
            load_input_int64_binary(offset_file, h_offset, offset_size);

            // Load Sharding Parameters
            std::string sharding_param_file = _sharding_param_dir + "/sharding_param_" + std::to_string(rank) + ".json";
            std::ifstream jsonFile_sharding(sharding_param_file);
            jsonFile_sharding >> sharding_param;

            // compute unique
            at::Tensor unique_linear_indices, inverse;
            std::vector<int> D_offsets_data = sharding_param["D_offsets"];
            auto T = D_offsets_data.size() - 1;
            int64_t total_unique_indices = 0;

            at::Device device(at::kCUDA);
            auto int64_tensor_options = at::TensorOptions().device(device).dtype(at::kLong);
            std::vector<int64_t> hash_size_cumsum_data = sharding_param["hash_size_cumsum"];
            std::vector<int64_t> hash_size_list = sharding_param["hash_size_list"];
            int64_t max_hash_size = *std::max_element(hash_size_list.begin(), hash_size_list.end());
            if(max_hash_size * (T + 1) > INT64_MAX){
                std::cout<< "TOTAL HASHSIZE TOO LARGE" << std::endl;
                exit(1);
            }
            at::Tensor hash_size_cumsum = at::tensor(hash_size_cumsum_data, int64_tensor_options);

            int32_t info_B_num_bits;
            uint32_t info_B_mask;

            // Generate GPU input tensor
            for(int i=0;i<nBatch;i++){
                at::Tensor tmp_index = at::from_blob(h_index + index_offset[i], {index_offset[i+1]-index_offset[i]}, at::kLong).to(at::kCUDA);
                at::Tensor tmp_offset = at::from_blob(h_offset + offset_offset[i], {offset_offset[i+1]-offset_offset[i]}, at::kLong).to(at::kCUDA);

                int32_t total_B = tmp_offset.size(0) - 1;
                int32_t B = total_B / T;

                at::Tensor infos = at::empty_like(tmp_index, tmp_index.options().dtype(at::kInt));
                at::Tensor linear_indices = at::empty_like(tmp_index);
                std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(total_B / T, T);

                // FBGEMM index linearlization
                // linearize_index_kernel_for_unique<<<
                //     div_round_up(total_B, kMaxThreads),
                //     kMaxThreads,
                //     0,
                //     at::cuda::getCurrentCUDAStream()>>>(
                //     hash_size_cumsum.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                //     tmp_index.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                //     tmp_offset.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                //     infos.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                //     linear_indices.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                //     info_B_num_bits,
                //     info_B_mask,
                //     (1u << (DEFAULT_INFO_NUM_BITS - info_B_num_bits)) - 1,
                //     (1u << info_B_num_bits) - 1,
                //     FixedDivisor(total_B / T)
                // );

                // max_table_length index linearlization
                for(int t=0; t<T; t++){
                    int64_t start = tmp_offset[t * B].item<int64_t>();
                    int64_t end = tmp_offset[(t+1) * B].item<int64_t>();
                    linear_indices.slice(0, start, end) = tmp_index.slice(0, start, end) + t * max_hash_size;
                }

                std::tie(unique_linear_indices, inverse) = at::_unique(linear_indices, true, true);

                total_unique_indices = unique_linear_indices.sizes()[0];
                total_unique_list.push_back(total_unique_indices);
                input_batch_list.push_back(std::make_pair(tmp_index, tmp_offset));
                linear_unique_inverse_list.push_back(std::make_pair(unique_linear_indices, inverse));
            }
        }

        ~TBE_Dataloader(){
            free(h_index);
            free(h_offset);
            // cudaFree(d_index);
            // cudaFree(d_offset);
        }

        // //next batch host
        std::tuple<int64_t*, int64_t*> next_h(){
            auto next_batch = std::make_tuple(h_index + index_offset[curBatch_h], h_offset + offset_offset[curBatch_h]);
            curBatch_h += 1;
            if(curBatch_h >= nBatch)
                curBatch_h = 0;
            return next_batch;
        }

        std::pair<at::Tensor, at::Tensor> next_input_tensor(){
            auto batch = input_batch_list[curBatch];
            curBatch += 1;
            if(curBatch >= nBatch)
                curBatch = 0;
            return batch;
        }

        std::tuple<at::Tensor, at::Tensor, int64_t> next_input_tensor_with_unique(){
            auto batch = input_batch_list[curBatch];
            int64_t unique = total_unique_list[curBatch];
            curBatch += 1;
            if(curBatch >= nBatch)
                curBatch = 0;
            return std::tuple<at::Tensor, at::Tensor, int64_t>(batch.first, batch.second, unique);
        }

        std::tuple<at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor> next_input_tensor_with_unique_inverse(){
            auto batch = input_batch_list[curBatch];
            int64_t unique = total_unique_list[curBatch];
            auto unique_inverse = linear_unique_inverse_list[curBatch];
            curBatch += 1;
            if(curBatch >= nBatch)
                curBatch = 0;
            return std::tuple<at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor>(batch.first, batch.second, unique, unique_inverse.first, unique_inverse.second);
        }

        void reset_iter(){
            curBatch = 0;
        }

        json get_sharing_param(){
            return sharding_param;
        }


    private:
        int _rank;
        std::string _data_dir;
        std::string _sharding_param_dir;
        int64_t* h_index;
        int64_t* h_offset;
        // int64_t* d_index;
        // int64_t* d_offset;
        int index_size=0;
        int offset_size=0;
        int nBatch=0;
        int curBatch=0;
        int curBatch_h=0;
        std::vector<int> index_offset;
        std::vector<int> offset_offset;
        std::vector<std::pair<at::Tensor, at::Tensor>> input_batch_list;
        std::vector<std::pair<at::Tensor, at::Tensor>> linear_unique_inverse_list;
        std::vector<int64_t> total_unique_list;
        json sharding_param;
};

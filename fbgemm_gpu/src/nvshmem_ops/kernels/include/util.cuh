// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "nccl.h"
#include <stdio.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h> // for at::cuda::current_device
#include <c10/cuda/CUDAStream.h> // for at::cuda::CUDAStream::getCurrentCUDAStream

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                    \
  do {                                      \
    cudaError_t result = (stmt);            \
    if (cudaSuccess != result) {            \
      fprintf(                              \
          stderr,                           \
          "[%s:%d] cuda failed with %s \n", \
          __FILE__,                         \
          __LINE__,                         \
          cudaGetErrorString(result));      \
      exit(-1);                             \
    }                                       \
  } while (0)


std::vector<at::Tensor> init_NCCL_AlltoAll_buffer(
  int32_t nranks,
  int32_t rank,
  std::vector<int> dim_sum_per_rank_data,
  int32_t local_batch_size
){
  auto options = at::TensorOptions().device(at::kCUDA, rank).dtype(at::kFloat);
  std::vector<at::Tensor> all_to_all_buffer = {};
  for (int i = 0; i < nranks; ++i) {
    at::Tensor buffer = at::empty({local_batch_size, dim_sum_per_rank_data[i]}, options);
    all_to_all_buffer.push_back(buffer);
  }
  return all_to_all_buffer;
}

std::vector<at::Tensor> init_NCCL_AlltoAll_buffer_bwd(
  int32_t nranks,
  int32_t rank,
  std::vector<int> dim_sum_per_rank_data,
  int32_t local_batch_size
){
  auto options = at::TensorOptions().device(at::kCUDA, rank).dtype(at::kFloat);
  std::vector<at::Tensor> all_to_all_buffer_bwd = {};
  for (int i = 0; i < nranks; ++i) {
    at::Tensor buffer_bwd = at::empty({local_batch_size, dim_sum_per_rank_data[rank]}, options);
    all_to_all_buffer_bwd.push_back(buffer_bwd);
  }
  return all_to_all_buffer_bwd;
}

at::Tensor NCCL_AlltoAll_forward(
  at::Tensor lookup_result,
  std::vector<at::Tensor> all_to_all_buffer,
  std::vector<int> dim_sum_per_rank_data,
  ncclComm_t comm,
  int32_t local_batch_size,
  int32_t nranks,
  int32_t rank
){
  int32_t comm_size = dim_sum_per_rank_data[rank] * local_batch_size;
  ncclGroupStart();
  for (int r=0; r<nranks; r++) {
    ncclSend(lookup_result.data_ptr<float>() + comm_size * r, comm_size, ncclFloat, r, comm, at::cuda::getCurrentCUDAStream());
    ncclRecv(all_to_all_buffer[r].data_ptr<float>(), dim_sum_per_rank_data[r] * local_batch_size, ncclFloat, r, comm, at::cuda::getCurrentCUDAStream());
  }
  ncclGroupEnd();
  CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));

  return at::cat(all_to_all_buffer, 1);
}


at::Tensor NCCL_AlltoAll_backward(
  at::Tensor gradients,
  std::vector<at::Tensor> all_to_all_buffer,
  std::vector<int> dim_sum_per_rank_data,
  at::IntArrayRef dim_sum_IntArrary,
  ncclComm_t comm,
  int32_t local_batch_size,
  int32_t nranks,
  int32_t rank
){
  std::vector<at::Tensor> send_tensors = at::split(gradients, dim_sum_IntArrary, 1);
  for(int i=0; i<nranks; i++){
      send_tensors[i] = send_tensors[i].contiguous();
  }

  ncclGroupStart();
  for (int r=0; r<nranks; r++) {
    ncclSend(send_tensors[r].data_ptr<float>(), dim_sum_per_rank_data[r] * local_batch_size, ncclFloat, r, comm, at::cuda::getCurrentCUDAStream());
    ncclRecv(all_to_all_buffer[r].data_ptr<float>(), dim_sum_per_rank_data[rank] * local_batch_size, ncclFloat, r, comm, at::cuda::getCurrentCUDAStream());
  }
  ncclGroupEnd();
  CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));

  return at::cat(all_to_all_buffer, 0);
}


void print_2d_tensor(at::Tensor tensor, int32_t n_row=0, int32_t n_column=0, int32_t row_offset=0, int32_t column_offset=0){
    tensor = tensor.to(at::kCPU);  // Make sure tensor is on CPU
    auto accessor = tensor.accessor<float,2>();

    int32_t R = accessor.size(0);
    int32_t C = accessor.size(1);

    if(n_row!=0 && n_row<R){
      R = n_row;
    }
    if(n_column != 0 && n_column<C){
      C = n_column;
    }

    std::cout << "[\n";
    for (int i = row_offset; i < min(R + row_offset, int(accessor.size(0))); ++i) {
        std::cout << "    [";
        for (int j = column_offset; j < min(C + column_offset, int(accessor.size(1))); ++j) {
            // Print each element with a fixed number of decimal places
            std::cout << std::fixed << std::setprecision(5) << accessor[i][j];
            // Don't print a comma after the last element of a row
            if (j != C - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "],\n";
    }
    std::cout << "]\n";
}

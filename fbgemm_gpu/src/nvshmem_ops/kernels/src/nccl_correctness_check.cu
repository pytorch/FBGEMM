#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <mpi.h>
#include <torch/torch.h>

#include <chrono>
#include <thread>

#include "nccl.h"
#include "util.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h> // for at::cuda::current_device
#include <c10/cuda/CUDAStream.h> // for at::cuda::CUDAStream::getCurrentCUDAStream



int main(int argc, char* argv[]) {
    // MPI and NVSHMEM and NCCL
    // init.===============================================================================================================================
    // MPI
    int rank, nranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int deviceId, dev_count;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(rank % dev_count));

    // NCCL
    ncclUniqueId id;
    ncclComm_t comm;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, nranks, id, rank);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int gpu_per_node = nranks;
    int local_rank = rank % gpu_per_node;
    printf(
        "rank:%d, nranks:%d, cuda_device_id:%d, device_cnt:%d, local_rank:%d\n",
        rank,
        nranks,
        deviceId,
        dev_count,
        local_rank);

    // Init nvshmem buffer =====================================================================================================
    // int32_t local_batch_size = 1024;
    // std::vector<int> dim_sum_per_rank_data = {16, 32, 64, 16, 32, 64, 16, 32};
    int32_t local_batch_size = 2;
    int32_t total_batch_size = local_batch_size * nranks;
    std::vector<int32_t> dim_sum_per_rank_data = {2, 4, 2, 4, 2, 4, 2, 4};
    std::vector<int32_t> dim_offset_per_rank_data = {};
    int32_t output_dim = 0;

    for(int i=0; i<nranks; i++){
        dim_offset_per_rank_data.push_back(output_dim);
        output_dim += dim_sum_per_rank_data[i];
    }

    at::TensorOptions options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    at::Tensor embedding = at::empty({total_batch_size, dim_sum_per_rank_data[rank]}, options);
    embedding.fill_(float(rank + 1));

    at::Tensor gradients = at::empty({local_batch_size, output_dim}, options);
    gradients.fill_(float(rank + 1));

    std::vector<at::Tensor> fwd_result_tensor_list = {};
    std::vector<at::Tensor> bwd_result_tensor_list = {};
    for(int i=0; i<nranks; i++){
        at::Tensor tmp = at::empty({local_batch_size, dim_sum_per_rank_data[i]}, options);
        tmp.fill_(float(i + 1));
        fwd_result_tensor_list.push_back(tmp);

        at::Tensor tmp_bwd = at::empty({local_batch_size, dim_sum_per_rank_data[rank]}, options);
        tmp_bwd.fill_(float(i + 1));
        bwd_result_tensor_list.push_back(tmp_bwd);
    }

    at::Tensor true_fwd_result =  at::cat(fwd_result_tensor_list, 1);
    at::Tensor true_bwd_result = at::cat(bwd_result_tensor_list, 0);
    std::vector<at::Tensor> all_to_all_buffer = init_NCCL_AlltoAll_buffer(nranks, rank, dim_sum_per_rank_data, local_batch_size);
    std::vector<at::Tensor> all_to_all_buffer_bwd = init_NCCL_AlltoAll_buffer_bwd(nranks, rank, dim_sum_per_rank_data, local_batch_size);

    // std::this_thread::sleep_for(std::chrono::seconds(rank));
    // print_2d_tensor(true_fwd_result);
    // print_2d_tensor(true_bwd_result);

    // =========================== Forward All-to-All Result Check ============================
    at::Tensor fwd_a2a_result = NCCL_AlltoAll_forward(embedding, all_to_all_buffer, dim_sum_per_rank_data, comm, local_batch_size, nranks, rank);
    bool are_close = torch::allclose(true_fwd_result, fwd_a2a_result);
    std::this_thread::sleep_for(std::chrono::seconds(rank));
    if (are_close) {
        std::cout << "rank:" << rank << "  The fwd result is correct!" << std::endl;
    } else {
        std::cout << "rank:" << rank << "  ERROR: The fwd result is incorrect!" << std::endl;
    }

    // =========================== Backward All-to-All Result Check ============================
    std::vector<int64_t> dim_sum_int64(dim_sum_per_rank_data.begin(), dim_sum_per_rank_data.end());
    at::IntArrayRef dim_sum_IntArrary(dim_sum_int64);
    at::Tensor bwd_a2a_result = NCCL_AlltoAll_backward(gradients, all_to_all_buffer_bwd, dim_sum_per_rank_data, dim_sum_IntArrary, comm, local_batch_size, nranks, rank);
    bool are_close_bwd = torch::allclose(true_bwd_result, bwd_a2a_result);
    std::this_thread::sleep_for(std::chrono::seconds(rank));
    if (are_close_bwd) {
        std::cout << "rank:" << rank << "  The bwd result is correct!" << std::endl;
    } else {
        std::cout << "rank:" << rank << "  ERROR: The bwd result is incorrect!" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // Release Memeory ========================================================================================================
    std::cout << "rank:" << rank << ", finished\n";
    MPI_Finalize();
    return 0;
}

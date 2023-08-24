#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include "tbe_data_loader.cuh"
#include "fbgemm_put_tbe.cuh"

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
    int32_t if_save_result = atoi(argv[1]); // if exp_type == 0: fbgemm kernel; exp_type == 1: nvshmem kernel, exp_type == 2: profiling
    int32_t if_all_to_all = atoi(argv[2]); // if exp_type == 0: fbgemm kernel; exp_type == 1: nvshmem kernel, exp_type == 2: profiling
    int32_t n_loop = 2048;
    if(argc > 3){
        n_loop = atoi(argv[3]);
    }

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

    // Loading sharding parameters ================================================================================================
    std::string home_directory = std::getenv("HOME");
    std::string sharding_param_dir = home_directory + "/tmp/test_1";
    std::string data_dir = home_directory + "/tmp/test_1/data";
    std::string result_dir = home_directory + "/tmp/test_1/result";
    TBE_Dataloader* dataloader = new TBE_Dataloader(sharding_param_dir, data_dir, rank);

    auto sharding_param = dataloader->get_sharing_param();
    std::cout << sharding_param <<"\n";

    at::Device device(at::kCUDA);
    auto float_tensor_options = at::TensorOptions().device(device).dtype(at::kFloat);
    auto int32_tensor_options = at::TensorOptions().device(device).dtype(at::kInt);
    auto int64_tensor_options = at::TensorOptions().device(device).dtype(at::kLong);

    torch::manual_seed(rank);

    // Load and init sharding params and embedding table parameters.
    std::string file_name = sharding_param_dir + "/weight_" + std::to_string(rank) + ".bin";
    at::Tensor dev_weights = load_float_tensor(file_name, sharding_param["dev_weights"][0]).to(device);
    // std::cout << "dev_weights:" << dev_weights.sizes() << std::endl;
    // at::Tensor dev_weight_cpu = load_float_tensor(file_name, sharding_param["dev_weights"][0]);
    // at::Tensor dev_weights = at::randn(dev_weight_cpu.sizes(), float_tensor_options);

    at::Tensor uvm_weights = at::empty({sharding_param["uvm_weights"][0]}, float_tensor_options);
    // std::cout << "uvm_weights:" << uvm_weights.sizes() << std::endl;

    at::Tensor lxu_cache_weights = at::empty({sharding_param["lxu_cache_weights"][0], sharding_param["lxu_cache_weights"][1]}, float_tensor_options);
    // std::cout << "lxu_cache_weights:" << lxu_cache_weights.sizes() << std::endl;

    std::vector<int> weights_placements_data = sharding_param["weights_placements"];
    at::Tensor weights_placements = at::tensor(weights_placements_data, int32_tensor_options);
    // std::cout << "weights_placements:" << weights_placements.sizes() << std::endl;

    std::vector<int64_t> weights_offsets_data = sharding_param["weights_offsets"];
    at::Tensor weights_offsets = at::tensor(weights_offsets_data, int64_tensor_options);
    // std::cout << "weights_offsets:" << weights_offsets.sizes() << std::endl;

    std::vector<int> D_offsets_data = sharding_param["D_offsets"];
    at::Tensor D_offsets = at::tensor(D_offsets_data, int32_tensor_options);
    // std::cout << "D_offsets:" << D_offsets.sizes() << std::endl;

    at::Tensor lxu_cache_locations = at::empty({0}, int32_tensor_options);
    // std::cout << "lxu_cache_locations:" << lxu_cache_locations.sizes() << std::endl;

    std::vector<int64_t> hash_size_cumsum_data = sharding_param["hash_size_cumsum"];
    at::Tensor hash_size_cumsum = at::tensor(hash_size_cumsum_data, int64_tensor_options);
    // std::cout << "hash_size_cumsum:" << hash_size_cumsum.sizes() << std::endl;

    // All-to-All information:
    std::vector<int> dim_sum_per_rank_data = sharding_param["dim_sum_per_rank"];
    std::vector<int> dim_offset_per_rank_data = sharding_param["dim_offset_per_rank"];
    at::Tensor dim_sum_per_rank = at::tensor(dim_sum_per_rank_data, int32_tensor_options);
    at::Tensor dim_offset_per_rank = at::tensor(dim_offset_per_rank_data, int32_tensor_options);
    int32_t total_dim_output = sharding_param["total_dim_output"];

    int64_t total_D = sharding_param["total_D"];
    int64_t max_D = sharding_param["max_D"];
    int64_t pooling_mode = sharding_param["pooling_mode"];
    int64_t output_dtype = sharding_param["output_dtype"];
    bool is_experimental = sharding_param["is_experimental"];
    int32_t n_local_Table = D_offsets.numel() - 1; // n_local_Table

    // backward params
    int64_t unused_ = 0;
    int32_t max_segment_length_per_warp = 32;
    int64_t total_hash_size_bits = sharding_param["total_hash_size_bits"];
    bool stochastic_rounding = sharding_param["stochastic_rounding"];
    double learning_rate = sharding_param["learning_rate"];
    int32_t T = D_offsets.numel() - 1; // the number of local table;
    int32_t total_B;
    int32_t info_B_num_bits;
    uint32_t info_B_mask;

    std::cout<< "total_D:" << total_D << ", max_D:" << max_D << ", pooling_mode:" <<
    pooling_mode << ", output_dtype:" << output_dtype << ", is_experimental:" << is_experimental << "\n";

    // Init nvshmem buffer =====================================================================================================
    at::Tensor tmp_idx, tmp_offset;
    std::tie(tmp_idx, tmp_offset) = dataloader->next_input_tensor();
    dataloader->reset_iter();
    total_B = tmp_offset.size(0) - 1;
    int32_t local_batch_size = total_B / n_local_Table / nranks;
    float *output_buffer, *h_output_buffer;
    h_output_buffer = (float *) malloc (local_batch_size * total_dim_output * sizeof(float)); // malloc host EMT

    std::vector<at::Tensor> all_to_all_buffer = init_NCCL_AlltoAll_buffer(nranks, rank, dim_sum_per_rank_data, local_batch_size);

    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "rank:" << rank << ", finished loading sharding params and init grad buffer\n";

    // Compute and save result ================================================================================================
    // at::Tensor indices, offsets;
    // std::tie(indices, offsets) = dataloader->next_input_tensor();

    // at::Tensor embedding = split_embedding_codegen_forward_unweighted_cuda_local(
    //     dev_weights,
    //     uvm_weights,
    //     lxu_cache_weights,
    //     weights_placements,
    //     weights_offsets,
    //     D_offsets,
    //     total_D,
    //     max_D,
    //     indices,
    //     offsets,
    //     pooling_mode,
    //     lxu_cache_locations,
    //     output_dtype,
    //     is_experimental
    // );

    // cudaDeviceSynchronize();
    // at::Tensor fwd_a2a_result = NCCL_AlltoAll_forward(embedding, all_to_all_buffer, dim_sum_per_rank_data, comm, local_batch_size, nranks, rank);

    // cudaMemcpy(h_output_buffer, fwd_a2a_result.data_ptr<float>(), local_batch_size * total_dim_output * sizeof(float), cudaMemcpyDeviceToHost);
    // std::string all_to_all_file = result_dir + "/fbgemm_all_to_all_result_" + std::to_string(rank) + "_" + std::to_string(0) + ".bin";
    // save_float(h_output_buffer, all_to_all_file, local_batch_size * total_dim_output);

    at::Tensor indices, offsets;
    if(if_save_result){
        std::tie(indices, offsets) = dataloader->next_input_tensor();

        at::Tensor embedding = split_embedding_codegen_forward_unweighted_cuda_local(
            dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            D_offsets,
            total_D,
            max_D,
            indices,
            offsets,
            pooling_mode,
            lxu_cache_locations,
            output_dtype,
            is_experimental
        );

        cudaDeviceSynchronize();
        at::Tensor fwd_a2a_result = NCCL_AlltoAll_forward(embedding, all_to_all_buffer, dim_sum_per_rank_data, comm, local_batch_size, nranks, rank);

        // save result
        cudaMemcpy(h_output_buffer, fwd_a2a_result.data_ptr<float>(), local_batch_size * total_dim_output * sizeof(float), cudaMemcpyDeviceToHost);
        std::string all_to_all_file = result_dir + "/fbgemm_all_to_all_result_" + std::to_string(rank) + "_" + std::to_string(0) + ".bin";
        save_float(h_output_buffer, all_to_all_file, local_batch_size * total_dim_output);

        std::cout << "Saved FBGEMM Result\n";
    }

    // warm up ================================================================================================
    for(int i=0; i<256; i++){
        std::tie(indices, offsets) = dataloader->next_input_tensor();

        at::Tensor embedding = split_embedding_codegen_forward_unweighted_cuda_local(
            dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            D_offsets,
            total_D,
            max_D,
            indices,
            offsets,
            pooling_mode,
            lxu_cache_locations,
            output_dtype,
            is_experimental
        );

        if(if_all_to_all){
            cudaDeviceSynchronize();
            at::Tensor fwd_a2a_result = NCCL_AlltoAll_forward(embedding, all_to_all_buffer, dim_sum_per_rank_data, comm, local_batch_size, nranks, rank);
        }
        if(rank==0 && (i+1)%16==0)
            printf("finished iter warmup:%d\n", i);
    }
    cudaDeviceSynchronize();

    // Profiling the throughput ================================================================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for(int i=0;i<n_loop;i++){
        std::tie(indices, offsets) = dataloader->next_input_tensor();

        at::Tensor embedding = split_embedding_codegen_forward_unweighted_cuda_local(
            dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            D_offsets,
            total_D,
            max_D,
            indices,
            offsets,
            pooling_mode,
            lxu_cache_locations,
            output_dtype,
            is_experimental
        );

        if(if_all_to_all){
            cudaDeviceSynchronize();
            at::Tensor fwd_a2a_result = NCCL_AlltoAll_forward(embedding, all_to_all_buffer, dim_sum_per_rank_data, comm, local_batch_size, nranks, rank);
        }
        if(rank==0 && (i+1)%1024==0)
            printf("finished iter:%d\n", i);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Rank-%d, Total (ms): %.3f, avg_kernel latency:%.3f ms/iter\n", rank, milliseconds, milliseconds/float(n_loop));

    MPI_Barrier(MPI_COMM_WORLD);

    // Release Memeory ========================================================================================================
    std::cout << "rank:" << rank << ", finished save result\n";
    free(h_output_buffer);
    MPI_Finalize();
    return 0;
}

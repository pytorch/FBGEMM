#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#include "tbe_data_loader.cuh"
#include "nvshmem_put_tbe_backward.cuh"
#include <torch/torch.h>

#include "util.cuh"
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <chrono>
#include <thread>

#include "nccl.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h> // for at::cuda::current_device
#include <c10/cuda/CUDAStream.h> // for at::cuda::CUDAStream::getCurrentCUDAStream

void find_index(const at::Tensor& tensor, float value) {
    for (int i = 0; i < tensor.size(0); ++i) {
        for (int j = 0; j < tensor.size(1); ++j) {
            if (std::abs(float(tensor[i][j].item<float>()) - value) < 0.000001) {
                std::cout <<  "[" << i << ", " << j << "]" << std::endl;
            }
        }
    }
}


int main(int argc, char* argv[]) {
    int32_t if_save_result = 0;
    int32_t put_type = 0; // put_type == 0: thread-level put;     put_type == 1: warp-level put;    put_type == 2: only_tbe + sync;     put_type >= 3: only_tbe without sync
    int32_t n_loop = 2048;
    if(argc == 3){
        if_save_result = atoi(argv[1]);
        put_type = atoi(argv[2]);
    }
    else if(argc == 4){
        if_save_result = atoi(argv[1]);
        put_type = atoi(argv[2]);
        n_loop = atoi(argv[3]);
    }

    // MPI and NVSHMEM init.===============================================================================================================================
    int rank, nranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int mype, npes, mype_node;
    mype = nvshmem_my_pe(); // global id
    npes = nvshmem_n_pes(); // global size
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE); // local id

    int deviceId, dev_count;
    cudaGetDevice(&deviceId);
    cudaGetDeviceCount(&dev_count);
    cudaSetDevice(mype % dev_count);

    // NCCL
    ncclUniqueId id;
    ncclComm_t comm;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, nranks, id, rank);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    printf("rank:%d, nranks:%d, mype:%d, npes:%d, mype_node:%d, cuda_device_id:%d, device_cnt:%d\n", rank, nranks, mype, npes, mype_node, deviceId, dev_count);

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
    // at::Tensor dev_weights = load_float_tensor(file_name, sharding_param["dev_weights"][0]).to(device);
    // std::cout << "dev_weights:" << dev_weights.sizes() << std::endl;
    // at::Tensor dev_weight_cpu = load_float_tensor(file_name, sharding_param["dev_weights"][0]);
    at::Tensor dev_weights = at::randn({sharding_param["dev_weights"][0]}, float_tensor_options);

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
    output_buffer = (float *) nvshmem_malloc (local_batch_size * total_dim_output * sizeof(float));  // NVSHMEM global memory for EMTs.
    h_output_buffer = (float *) malloc (local_batch_size * total_dim_output * sizeof(float)); // malloc host EMT

    std::vector<at::Tensor> all_to_all_buffer = init_NCCL_AlltoAll_buffer(nranks, rank, dim_sum_per_rank_data, local_batch_size);
    std::vector<at::Tensor> all_to_all_buffer_bwd = init_NCCL_AlltoAll_buffer_bwd(nranks, rank, dim_sum_per_rank_data, local_batch_size);
    std::vector<int64_t> dim_sum_int64(dim_sum_per_rank_data.begin(), dim_sum_per_rank_data.end());
    at::IntArrayRef dim_sum_IntArrary(dim_sum_int64);

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "rank:" << rank << ", finished loading sharding params\n";

    // Compute and save result ================================================================================================
    at::Tensor indices, offsets;
    std::tie(indices, offsets) = dataloader->next_input_tensor();
    at::Tensor grad_tensor = at::randn({local_batch_size, total_dim_output}, float_tensor_options);
    at::Tensor all_to_all_grad = at::zeros({local_batch_size * nranks, total_D}, float_tensor_options);
    // NCCL_AlltoAll_backward(grad_tensor, all_to_all_buffer_bwd, dim_sum_per_rank_data, dim_sum_IntArrary, comm, local_batch_size, nranks, rank);

    float* nvshmem_grad_buffer = (float *) nvshmem_malloc (local_batch_size * total_dim_output * sizeof(float));
    cudaMemcpy(nvshmem_grad_buffer, grad_tensor.data_ptr<float>(), local_batch_size * total_dim_output * sizeof(float), cudaMemcpyHostToDevice);

    float *grad_get_buffer;
    int32_t* grad_get_signal;
    CUDA_CHECK(cudaMalloc(&grad_get_buffer, local_batch_size * nranks * total_D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_get_signal, local_batch_size * nranks * n_local_Table * sizeof(int32_t)));

    total_B = offsets.size(0) - 1;
    std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(total_B / T, T);

    at::Tensor backward_result = nvshmem_split_embedding_backward_codegen_sgd_unweighted_exact_cuda_cache(
        all_to_all_grad,
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D_offsets,
        max_D,
        hash_size_cumsum,
        total_hash_size_bits,
        indices,
        offsets,
        pooling_mode,
        lxu_cache_locations,
        unused_,
        max_segment_length_per_warp,
        stochastic_rounding,
        info_B_num_bits,
        info_B_mask,
        // nvshmem_parameters:
        nvshmem_grad_buffer,
        grad_get_buffer,
        grad_get_signal,
        dim_offset_per_rank,
        local_batch_size,
        total_dim_output,
        rank,
        nranks,
        learning_rate
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaMemset(grad_get_signal, 0, local_batch_size * nranks * n_local_Table * sizeof(int32_t));

    if(if_save_result == 1){
        std::string result_file = result_dir + "/nvshmem_all_to_all_bwd_result_" + std::to_string(rank) + ".bin";
        save_float_tensor(dev_weights.to(torch::kCPU).flatten(), result_file, int(dev_weights.numel() / 10)+1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // // warm up ================================================================================================
    for(int i=0; i<256; i++){
        std::tie(indices, offsets) = dataloader->next_input_tensor();

        total_B = offsets.size(0) - 1;
        std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(total_B / T, T);

        at::Tensor backward_result = nvshmem_split_embedding_backward_codegen_sgd_unweighted_exact_cuda_cache(
            all_to_all_grad,
            dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            D_offsets,
            max_D,
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            pooling_mode,
            lxu_cache_locations,
            unused_,
            max_segment_length_per_warp,
            stochastic_rounding,
            info_B_num_bits,
            info_B_mask,
            // nvshmem_parameters:
            nvshmem_grad_buffer,
            grad_get_buffer,
            grad_get_signal,
            dim_offset_per_rank,
            local_batch_size,
            total_dim_output,
            rank,
            nranks,
            learning_rate
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmemx_barrier_all_on_stream(at::cuda::getCurrentCUDAStream());
        cudaMemset(grad_get_signal, 0, local_batch_size * nranks * n_local_Table * sizeof(int32_t));
    }
    cudaDeviceSynchronize();

    // // Profiling the throughput ================================================================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for(int i=0;i<n_loop;i++){
        std::tie(indices, offsets) = dataloader->next_input_tensor();

        total_B = offsets.size(0) - 1;
        std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(total_B / T, T);

        at::Tensor backward_result = nvshmem_split_embedding_backward_codegen_sgd_unweighted_exact_cuda_cache(
            all_to_all_grad,
            dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            D_offsets,
            max_D,
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            pooling_mode,
            lxu_cache_locations,
            unused_,
            max_segment_length_per_warp,
            stochastic_rounding,
            info_B_num_bits,
            info_B_mask,
            // nvshmem_parameters:
            nvshmem_grad_buffer,
            grad_get_buffer,
            grad_get_signal,
            dim_offset_per_rank,
            local_batch_size,
            total_dim_output,
            rank,
            nranks,
            learning_rate
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmemx_barrier_all_on_stream(at::cuda::getCurrentCUDAStream());
        cudaMemset(grad_get_signal, 0, local_batch_size * nranks * n_local_Table * sizeof(int32_t));

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
    nvshmem_free(output_buffer);
    free(h_output_buffer);
    cudaFree(grad_get_buffer);
    cudaFree(grad_get_signal);
    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}

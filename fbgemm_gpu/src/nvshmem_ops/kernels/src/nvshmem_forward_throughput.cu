#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#include "tbe_data_loader.cuh"
#include "nvshmem_put_tbe.cuh"
#include <torch/torch.h>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <chrono>
#include <thread>


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

    // std::cout<< "total_D:" << total_D << ", max_D:" << max_D << ", pooling_mode:" <<
    // pooling_mode << ", output_dtype:" << output_dtype << ", is_experimental:" << is_experimental << "\n";

    // Init nvshmem buffer =====================================================================================================
    at::Tensor tmp_idx, tmp_offset;
    std::tie(tmp_idx, tmp_offset) = dataloader->next_input_tensor();
    dataloader->reset_iter();
    const auto total_B = tmp_offset.size(0) - 1;
    int32_t local_batch_size = total_B / n_local_Table / nranks;
    float *output_buffer, *h_output_buffer;
    output_buffer = (float *) nvshmem_malloc (local_batch_size * total_dim_output * sizeof(float));  // NVSHMEM global memory for EMTs.
    h_output_buffer = (float *) malloc (local_batch_size * total_dim_output * sizeof(float)); // malloc host EMT

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "rank:" << rank << ", finished loading sharding params\n";

    // Compute and save result ================================================================================================
    // at::Tensor indices, offsets;
    // std::tie(indices, offsets) = dataloader->next_input_tensor();

    // at::Tensor embedding = nvshmem_split_embedding_codegen_forward_unweighted_cuda(
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
    //     is_experimental,
    //     // all_to_all_infos:
    //     dim_sum_per_rank,
    //     dim_offset_per_rank,
    //     total_dim_output,
    //     output_buffer,
    //     nranks,
    //     rank,
    //     put_type
    // );
    // cudaDeviceSynchronize();
    // nvshmemx_barrier_all_on_stream(at::cuda::getCurrentCUDAStream());
    // cudaMemcpy(h_output_buffer, output_buffer, local_batch_size * total_dim_output * sizeof(float), cudaMemcpyDeviceToHost);

    // std::string all_to_all_file = result_dir + "/nvshmem_all_to_all_result_" + std::to_string(rank) + "_" + std::to_string(0) + ".bin";
    // save_float(h_output_buffer, all_to_all_file, local_batch_size * total_dim_output);


    at::Tensor indices, offsets;
    if(if_save_result){
        std::tie(indices, offsets) = dataloader->next_input_tensor();
        at::Tensor embedding = nvshmem_split_embedding_codegen_forward_unweighted_cuda(
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
            is_experimental,
            // all_to_all_infos:
            dim_sum_per_rank,
            dim_offset_per_rank,
            total_dim_output,
            output_buffer,
            nranks,
            rank,
            put_type
        );
        cudaDeviceSynchronize();
        nvshmemx_barrier_all_on_stream(at::cuda::getCurrentCUDAStream());

        // save result
        cudaMemcpy(h_output_buffer, output_buffer, local_batch_size * total_dim_output * sizeof(float), cudaMemcpyDeviceToHost);
        std::string all_to_all_file = result_dir + "/nvshmem_all_to_all_result_" + std::to_string(rank) + "_" + std::to_string(0) + ".bin";
        save_float(h_output_buffer, all_to_all_file, local_batch_size * total_dim_output);

        std::cout << "Saved NVSHMEM Result\n";
    }


    // warm up ================================================================================================
    for(int i=0; i<256; i++){
        std::tie(indices, offsets) = dataloader->next_input_tensor();

        at::Tensor embedding = nvshmem_split_embedding_codegen_forward_unweighted_cuda(
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
            is_experimental,
            // all_to_all_infos:
            dim_sum_per_rank,
            dim_offset_per_rank,
            total_dim_output,
            output_buffer,
            nranks,
            rank,
            put_type
        );

        if(put_type < 3){
            cudaDeviceSynchronize();
            nvshmemx_barrier_all_on_stream(at::cuda::getCurrentCUDAStream());
        }
    }
    cudaDeviceSynchronize();

    // Profiling the throughput ================================================================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for(int i=0;i<n_loop;i++){
        std::tie(indices, offsets) = dataloader->next_input_tensor();

        at::Tensor embedding = nvshmem_split_embedding_codegen_forward_unweighted_cuda(
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
            is_experimental,
            // all_to_all_infos:
            dim_sum_per_rank,
            dim_offset_per_rank,
            total_dim_output,
            output_buffer,
            nranks,
            rank,
            put_type
        );

        if(put_type < 3){
            cudaDeviceSynchronize();
            nvshmemx_barrier_all_on_stream(at::cuda::getCurrentCUDAStream());
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
    nvshmem_free(output_buffer);
    free(h_output_buffer);
    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}

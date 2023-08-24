#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include "tbe_data_loader.cuh"
#include "util.cuh"
#include <mpi.h>
#include <chrono>
#include <thread>


int main(int argc, char* argv[]) {
    int32_t if_backward = 0;
    if(argc == 2){
        if_backward = atoi(argv[1]);
    }

    // MPI ================================================================================================
    int rank, nranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int deviceId, dev_count;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(rank % dev_count));


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

    // Load and init sharding params and embedding table parameters.
    int32_t total_dim_output = sharding_param["total_dim_output"];

    std::vector<int> D_offsets_data = sharding_param["D_offsets"];
    at::Tensor D_offsets = at::tensor(D_offsets_data, int32_tensor_options);
    int32_t n_local_Table = D_offsets.numel() - 1; // n_local_Table

    // Get local_batch_size =====================================================================================================
    at::Tensor tmp_idx, tmp_offset;
    std::tie(tmp_idx, tmp_offset) = dataloader->next_input_tensor();
    dataloader->reset_iter();
    int32_t total_B = tmp_offset.size(0) - 1;
    int32_t local_batch_size = total_B / n_local_Table / nranks;


    // load result tensor =====================================================================================================
    if(if_backward==0){
        std::string fbgemm_file = result_dir + "/fbgemm_all_to_all_result_" + std::to_string(rank) + "_" + std::to_string(0) + ".bin";
        std::string nvshmem_file = result_dir + "/nvshmem_all_to_all_result_" + std::to_string(rank) + "_" + std::to_string(0) + ".bin";

        at::Tensor fbgemm_result = load_float_tensor(fbgemm_file, local_batch_size * total_dim_output);
        at::Tensor nvshmem_result = load_float_tensor(nvshmem_file, local_batch_size * total_dim_output);

        // print first 16 elements:
        if(rank == 0){
            std::cout << "Rank:" << rank << "  FBGEMM result first 16 elements:\n";
            print_2d_tensor(fbgemm_result.view({local_batch_size, total_dim_output}), 1, 16, 38, 512);
            std::cout << "Rank:" << rank << "  NVSHMEM result first 16 elements:\n";
            print_2d_tensor(nvshmem_result.view({local_batch_size, total_dim_output}), 1, 16, 38, 512);
        }


        bool close = at::allclose(fbgemm_result, nvshmem_result, /*rtol=*/1e-03, /*atol=*/1e-03);
        if (close) {
            std::cout << "Rank:" << rank << "  FBGEMM and NVSHMEM has the same result." << std::endl;
        } else {
            std::cout << "Rank:" << rank << "  ERROR: the results of FBGEMM and NVSHMEM are different." << std::endl;
        }
    }
    else if(if_backward==1){
        std::string fbgemm_file = result_dir + "/fbgemm_all_to_all_bwd_result_" + std::to_string(rank) + ".bin";
        std::string nvshmem_file = result_dir + "/nvshmem_all_to_all_bwd_result_" + std::to_string(rank) + ".bin";

        // int32_t n_elem = int(int(sharding_param["dev_weights"][0]) * 0.1);
        int64_t n_elem = sharding_param["dev_weights"][0].get<int64_t>() / 10;
        // std::cout << sharding_param["dev_weights"][0] << " " << n_elem << std::endl;
        at::Tensor fbgemm_result = load_float_tensor(fbgemm_file, n_elem).to(device);;
        at::Tensor nvshmem_result = load_float_tensor(nvshmem_file, n_elem).to(device);;
        std::cout << "Rank:" << rank << " Finished load weight\n";

        bool close = at::allclose(fbgemm_result, nvshmem_result, /*rtol=*/1e-05, /*atol=*/1e-05);
        if (close) {
            std::cout << "Backward - Rank:" << rank << "  FBGEMM and NVSHMEM has the same result." << std::endl;
        } else {
            std::cout << "Backward - Rank:" << rank << "  ERROR: the results of FBGEMM and NVSHMEM are different." << std::endl;
        }
    }
    else if(if_backward==2){
        at::Tensor indices, offsets, unique_linear_indices, inverse;
        int64_t total_unique_indices;
        std::tie(indices, offsets, total_unique_indices, unique_linear_indices, inverse) = dataloader->next_input_tensor_with_unique_inverse();

        std::string fbgemm_file = result_dir + "/fbgemm_all_to_all_bwd_result_" + std::to_string(rank) + ".bin";
        std::string nvshmem_file = result_dir + "/nvshmem_all_to_all_bwd_result_" + std::to_string(rank) + ".bin";

        int64_t max_D = sharding_param["max_D"];
        int64_t n_elem = total_unique_indices * max_D;

        at::Tensor fbgemm_result = load_float_tensor(fbgemm_file, n_elem).to(device);;
        at::Tensor nvshmem_result = load_float_tensor(nvshmem_file, n_elem).to(device);;
        std::cout << "Rank:" << rank << " Finished load weight\n";

        bool close = at::allclose(fbgemm_result, nvshmem_result, /*rtol=*/1e-03, /*atol=*/1e-03);
        if (close) {
            std::cout << "Backward - Rank:" << rank << "  FBGEMM and NVSHMEM has the same result." << std::endl;
        } else {
            std::cout << "Backward - Rank:" << rank << "  ERROR: the results of FBGEMM and NVSHMEM are different." << std::endl;
        }
    }
    else{
        std::cout << "if_backward should be 0 to 2" << std::endl;
    }



    MPI_Finalize();
    return 0;
}

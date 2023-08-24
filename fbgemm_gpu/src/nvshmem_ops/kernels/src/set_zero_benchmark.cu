#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include "util.cuh"
#include <chrono>
#include <thread>


int main(int argc, char* argv[]) {

    // 1171 MBs
    int32_t rows = 1200000;
    int32_t dim = 256;
    int32_t n_loop = 65536;
    float milliseconds;

    if(argc == 3){
        rows = atoi(argv[1]);
        dim = atoi(argv[2]);
    }

    at::Device device(at::kCUDA);
    auto float_tensor_options = at::TensorOptions().device(device).dtype(at::kFloat);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ============================= empty =============================
    cudaEventRecord(start, 0);
    for(int i=0;i<n_loop;i++){
        auto grad_dev_weights = at::empty({rows * dim}, float_tensor_options);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("at::empty -- Total (ms): %.3f, avg_kernel latency:%.3f ms/iter\n", milliseconds, milliseconds/float(n_loop));


    // ============================= zeros =============================
    cudaEventRecord(start, 0);
    for(int i=0;i<n_loop;i++){
        auto grad_dev_weights = at::zeros({rows * dim}, float_tensor_options);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("at::zeros -- Total (ms): %.3f, avg_kernel latency:%.3f ms/iter\n", milliseconds, milliseconds/float(n_loop));

    // ============================= empty + zero =============================
    cudaEventRecord(start, 0);
    for(int i=0;i<n_loop;i++){
        auto grad_dev_weights = at::empty({rows * dim}, float_tensor_options);
        grad_dev_weights.zero_();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("empty + zero -- Total (ms): %.3f, avg_kernel latency:%.3f ms/iter\n", milliseconds, milliseconds/float(n_loop));

    // ============================= empty + cudaMemset =============================
    cudaEventRecord(start, 0);
    for(int i=0;i<n_loop;i++){
        auto grad_dev_weights = at::empty({rows * dim}, float_tensor_options);
        cudaMemset(grad_dev_weights.data_ptr<float>(), 0, rows * dim * sizeof(float));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("empty + cudaMemset -- Total (ms): %.3f, avg_kernel latency:%.3f ms/iter\n", milliseconds, milliseconds/float(n_loop));

    return 0;
}

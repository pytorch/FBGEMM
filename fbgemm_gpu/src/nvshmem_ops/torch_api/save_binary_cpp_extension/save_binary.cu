#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <iostream>
#include <tuple>
#include <fstream>

template <typename T>
void save_tensor_func(torch::Tensor input_tensor, std::string output_file, int64_t size) {
    // std::cout << output_file << std::endl;
    std::ofstream myFile(output_file, std::ofstream::binary);
    myFile.write(reinterpret_cast<char*>(input_tensor.data_ptr<T>()), size * sizeof(T));
}

void save_tensor(torch::Tensor input_tensor, std::string output_file, int64_t size) {
    torch::ScalarType tensor_type = input_tensor.scalar_type();

    if (tensor_type == torch::kFloat32) {
        save_tensor_func<float>(input_tensor, output_file, size);
    }
    else if (tensor_type == torch::kInt32) {
        save_tensor_func<int>(input_tensor, output_file, size);
    }
    else if (tensor_type == torch::kInt64) {
        save_tensor_func<int64_t>(input_tensor, output_file, size);
    }
    else {
        throw std::runtime_error("Unsupported tensor type for saving.");
    }
}

// void save_float_tensor(
//     torch::Tensor input_tensor,
//     std::string output_file,
//     int64_t size
// ){
//     std::cout << output_file << std::endl;

//     std::ofstream myFile (output_file, std::ofstream::binary);
//     myFile.write ((char *)(input_tensor.data_ptr<float>()), size * sizeof(float));
// }

// void save_int_tensor(
//     torch::Tensor input_tensor,
//     std::string output_file,
//     int64_t size
// ){
//     std::cout << output_file << std::endl;

//     std::ofstream myFile (output_file, std::ofstream::binary);
//     myFile.write ((char *)(input_tensor.data_ptr<int>()), size * sizeof(int));
// }

// void save_int64_tensor(
//     torch::Tensor input_tensor,
//     std::string output_file,
//     int64_t size
// ){
//     std::cout << output_file << std::endl;

//     std::ofstream myFile (output_file, std::ofstream::binary);
//     myFile.write ((char *)(input_tensor.data_ptr<int64_t>()), size * sizeof(int64_t));
// }


torch::Tensor load_float_tensor(
    std::string input_file,
    int64_t size
){
    // std::cout << input_file << std::endl;

    auto load_tensor = torch::zeros({1, size}, torch::kFloat);

    std::ifstream myFile (input_file, std::ifstream::binary);
    myFile.read ((char *)(load_tensor.data_ptr<float>()), size * sizeof(float));

    return load_tensor;
}

torch::Tensor load_int_tensor(
    std::string input_file,
    int64_t size
){
    // std::cout << input_file << std::endl;

    auto load_tensor = torch::zeros({1, size}, torch::kInt);

    std::ifstream myFile (input_file, std::ifstream::binary);
    myFile.read ((char *)(load_tensor.data_ptr<int>()), size * sizeof(int));

    return load_tensor;
}


torch::Tensor load_int64_tensor(
    std::string input_file,
    int64_t size
){
    // std::cout << input_file << std::endl;

    auto load_tensor = torch::zeros({1, size}, torch::kLong);

    std::ifstream myFile (input_file, std::ifstream::binary);
    myFile.read ((char *)(load_tensor.data_ptr<int64_t>()), size * sizeof(int64_t));

    return load_tensor;
}

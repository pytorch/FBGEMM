#include <iostream>
#include <string>
#include <torch/extension.h>


// void save_float_tensor(
//     torch::Tensor input_tensor,
//     std::string output_file,
//     int64_t size
// );

// void save_int_tensor(
//     torch::Tensor input_tensor,
//     std::string output_file,
//     int64_t size
// );

// void save_int64_tensor(
//     torch::Tensor input_tensor,
//     std::string output_file,
//     int64_t size
// );

void save_tensor(torch::Tensor input_tensor, std::string output_file, int64_t size);

torch::Tensor load_float_tensor(
    std::string input_file,
    int64_t size
);

torch::Tensor load_int_tensor(
    std::string input_file,
    int64_t size
);

torch::Tensor load_int64_tensor(
    std::string input_file,
    int64_t size
);

PYBIND11_MODULE(save_binary_extension, m) {
    // m.def("save_float_tensor", &save_float_tensor, "save_float_tensor");
    // m.def("save_int_tensor", &save_int_tensor, "save_int_tensor");
    // m.def("save_int64_tensor", &save_int64_tensor, "save_int64_tensor");
    m.def("save_tensor", &save_tensor, "save_tensor");

    m.def("load_float_tensor", &load_float_tensor, "load_float_tensor");
    m.def("load_int_tensor", &load_int_tensor, "load_int_tensor");
    m.def("load_int64_tensor", &load_int64_tensor, "load_int64_tensor");
}

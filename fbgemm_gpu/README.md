# FBGEMM_GPU

[![FBGEMM_GPU CI](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_ci.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_ci.yml)
[![FBGEMM_GPU-CPU Nightly Build](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_cpu_nightly.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_cpu_nightly.yml)
[![FBGEMM_GPU-CUDA Nightly Build](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_cuda_nightly.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_cuda_nightly.yml)

FBGEMM_GPU (FBGEMM GPU Kernels Library) is a collection of high-performance PyTorch
GPU operator libraries for training and inference.  The library provides efficient
table batched embedding bag, data layout transformation, and quantization supports.

FBGEMM_GPU is currently tested with CUDA 11.7.1 and 11.8 in CI, and with PyTorch
packages that are built against those CUDA versions.

Only Intel/AMD CPUs with AVX2 extensions are currently supported.


## Build Instructions

This section is intended for FBGEMM_GPU developers.  The full build instructions
for the CUDA, ROCm, and CPU-only variants of FBGEMM_GPU can be found
[here](docs/BuildInstructions.md).


## Installation

### Install through PIP

Currently only built with sm70/80 (V100/A100 GPU) wheel supports:

```
# Release GPU
conda install pytorch cuda -c pytorch -c "nvidia/label/cuda-11.7.1"
pip install fbgemm-gpu

# Release CPU-only
conda install pytorch cuda -c pytorch -c "nvidia/label/cuda-11.7.1"
pip install fbgemm-gpu-cpu

# Nightly GPU
conda install pytorch cuda -c pytorch-nightly -c "nvidia/label/cuda-11.7.1"
pip install fbgemm-gpu-nightly

# Nightly CPU-only
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
pip install fbgemm-gpu-nightly-cpu
```

### Running FBGEMM_GPU

The tests (in test folder) and benchmarks (in bench folder) are some great
examples of using FBGEMM_GPU.  To run the tests or benchmarks after building
FBGEMM_GPU (if tests or benchmarks are built), use the following command:

```
# run the tests and benchmarks of table batched embedding bag op,
# data layout transform op, quantized ops, etc.
cd test
python split_table_batched_embeddings_test.py
python quantize_ops_test.py
python sparse_ops_test.py
python split_embedding_inference_converter_test.py
cd ../bench
python split_table_batched_embeddings_benchmark.py
```

To run the tests and benchmarks on a GPU-capable device in CPU-only mode use CUDA_VISIBLE_DEVICES=-1
```
CUDA_VISIBLE_DEVICES=-1 python split_table_batched_embeddings_test.py
```

### Run the tests on ROCm

Please add `FBGEMM_TEST_WITH_ROCM=1` flag when running tests on ROCm.
```
cd test
FBGEMM_TEST_WITH_ROCM=1 python split_table_batched_embeddings_test.py
```

### Benchmark Example

```bash
cd bench
python split_table_batched_embeddings_benchmark.py uvm
```


## Documentation

### How FBGEMM_GPU works

For a high-level overview, design philosophy and brief descriptions of various
parts of FBGEMM_GPU please see our Wiki (work in progress).

We have extensively used comments in our source files. The best and up-to-date
documentation is available in the source files.

### Building the API Documentation

See [docs/README.md](docs/README.md).


## Join the FBGEMM_GPU Community

For questions or feature requests, please file a ticket over on
[GitHub Issues](https://github.com/pytorch/FBGEMM/issues) or reach out to us on
the `#fbgemm` channel in [PyTorch Slack](https://bit.ly/ptslack).

For contributions, please see the [`CONTRIBUTING`](../CONTRIBUTING.md) file for
ways to help out.


## License

FBGEMM_GPU is BSD licensed, as found in the [`LICENSE`](../LICENSE) file.

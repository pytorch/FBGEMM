# FBGEMM_GPU

[![FBGEMM_GPU CI](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_ci.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_ci.yml)
[![FBGEMM_GPU-CPU Nightly Build](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_cpu_nightly.yml/badge.svg?event=schedule)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_cpu_nightly.yml)
[![FBGEMM_GPU-CUDA Nightly Build](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_cuda_nightly.yml/badge.svg?event=schedule)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_cuda_nightly.yml)

FBGEMM_GPU (FBGEMM GPU Kernels Library) is a collection of high-performance PyTorch
GPU operator libraries for training and inference.  The library provides efficient
table batched embedding bag, data layout transformation, and quantization supports.

FBGEMM_GPU is currently tested with CUDA 11.7.1 and 11.8 in CI, and with PyTorch
packages (1.13+) that are built against those CUDA versions.

Only Intel/AMD CPUs with AVX2 extensions are currently supported.

See our [Documentation](docs/README.md) for more information.


## Installation

The full installation instructions
for the CUDA, ROCm, and CPU-only variants of FBGEMM_GPU can be found
[here](docs/InstallationInstructions.md).  In addition, instructions for running
example tests and benchmarks can be found [here](docs/TestInstructions.md).


## Build Instructions

This section is intended for FBGEMM_GPU developers only.  The full build
instructions for the CUDA, ROCm, and CPU-only variants of FBGEMM_GPU can be
found [here](docs/BuildInstructions.md).


## Join the FBGEMM_GPU Community

For questions, support, news updates, or feature requests, please feel free to:

* File a ticket in [GitHub Issues](https://github.com/pytorch/FBGEMM/issues)
* Post a discussion in [GitHub Discussions](https://github.com/pytorch/FBGEMM/discussions)
* Reach out to us on the `#fbgemm` channel in [PyTorch Slack](https://bit.ly/ptslack)

For contributions, please see the [`CONTRIBUTING`](../CONTRIBUTING.md) file for
ways to help out.


## License

FBGEMM_GPU is BSD licensed, as found in the [`LICENSE`](../LICENSE) file.

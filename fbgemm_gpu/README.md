# FBGEMM_GPU

[![FBGEMM_GPU-CPU CI](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_ci_cpu.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_ci_cpu.yml)
[![FBGEMM_GPU-CUDA CI](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_ci_cuda.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_ci_cuda.yml)
[![FBGEMM_GPU-ROCm CI](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_ci_rocm.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_gpu_ci_rocm.yml)

FBGEMM_GPU (FBGEMM GPU Kernels Library) is a collection of high-performance
PyTorch GPU operator libraries for training and inference.  The library provides
efficient table batched embedding bag, data layout transformation, and
quantization supports.

FBGEMM_GPU is currently tested with CUDA 12.1 and 11.8 in CI, and with PyTorch
packages (2.1+) that are built against those CUDA versions.

See the full [Documentation](https://pytorch.org/FBGEMM) for more information
on building, installing, and developing with FBGEMM_GPU, as well as the most
up-to-date support matrix for this library.


## Join the FBGEMM_GPU Community

For questions, support, news updates, or feature requests, please feel free to:

* File a ticket in [GitHub Issues](https://github.com/pytorch/FBGEMM/issues)
* Post a discussion in [GitHub Discussions](https://github.com/pytorch/FBGEMM/discussions)
* Reach out to us on the `#fbgemm` channel in [PyTorch Slack](https://bit.ly/ptslack)

For contributions, please see the [`CONTRIBUTING`](../CONTRIBUTING.md) file for
ways to help out.


## License

FBGEMM_GPU is BSD licensed, as found in the [`LICENSE`](../LICENSE) file.

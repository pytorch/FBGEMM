# FBGEMM_GPU

[![FBGEMMCI](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemmci.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemmci.yml)
[![Nightly Build](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_nightly_build.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_nightly_build.yml)
[![Nightly Build CPU](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_nightly_build_cpu.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_nightly_build_cpu.yml)

FBGEMM_GPU (FBGEMM GPU kernel library) is a collection of
high-performance CUDA GPU operator library for GPU training and inference.

The library provides efficient table batched embedding bag,
data layout transformation, and quantization supports.

Currently tested with CUDA 11.3, 11.5, 11.6, and 11.7 in CI. In all cases, we test with PyTorch packages which are built with CUDA 11.7.

Only Intel/AMD CPUs with AVX2 extensions are currently supported.

General build and install instructions are as follows:

Build dependencies: `scikit-build`, `cmake`, `ninja`, `jinja2`, `torch`, `cudatoolkit`,
and for testing: `hypothesis`.

```
conda install scikit-build jinja2 ninja cmake hypothesis
```

**If you're planning to build from source** and **don't** have `nvml.h` in your system, you can install it via the command
below.
```
conda install -c conda-forge cudatoolkit-dev
```

Certain operations require this library to be present. Be sure to provide the path to `libnvidia-ml.so` to
`--nvml_lib_path` if installing from source (e.g. `python setup.py install --nvml_lib_path path_to_libnvidia-ml.so`).


## PIP install

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

## Build from source

Additional dependencies: currently cuDNN is required to be installed.
Please [download][4] and follow instructions [here][5] to install cuDNN.

```
# Requires PyTorch 1.13 or later
conda install pytorch cuda -c pytorch-nightly -c "nvidia/label/cuda-11.7.1"
git clone --recursive https://github.com/pytorch/FBGEMM.git
cd FBGEMM/fbgemm_gpu
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

# Specify CUDA version to use
# (may not be needed with only a single version installed)
export CUDA_BIN_PATH=/usr/local/cuda-11.3/
export CUDACXX=/usr/local/cuda-11.3/bin/nvcc

# Specify cuDNN library and header paths.  We tested CUDA 11.6 and 11.7 with
# cuDNN version 8.5.0.96
export CUDNN_LIBRARY=${HOME}/cudnn-linux-x86_64-8.5.0.96_cuda11-archive/lib
export CUDNN_INCLUDE_DIR=${HOME}/cudnn-linux-x86_64-8.5.0.96_cuda11-archive/include

# in fbgemm_gpu folder
# build for the CUDA architecture supported by current system (or all architectures if no CUDA device present)
python setup.py install
# or build it for specific CUDA architectures (see PyTorch documentation for usage of TORCH_CUDA_ARCH_LIST)
python setup.py install -DTORCH_CUDA_ARCH_LIST="7.0;8.0"
```


## Usage Example:
```bash
cd bench
python split_table_batched_embeddings_benchmark.py uvm
```
## Build on ROCm

FBGEMM_GPU supports running on AMD (ROCm) devices. A Docker container is recommended for setting up the ROCm environment. The installation on bare metal is also available. ROCm5.3 is used as an example of the installation below.

##### Build in a Docker container
Pull Docker container and run
```
docker pull rocm/pytorch:rocm5.4_ubuntu20.04_py3.8_pytorch_staging_base
sudo docker run -it --network=host --shm-size 16G --device=/dev/kfd --device=/dev/dri \
                --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
                --ipc=host --env PYTORCH_ROCM_ARCH="gfx906;gfx908;gfx90a" -u 0 \
                rocm/pytorch:rocm5.4_ubuntu20.04_py3.8_pytorch_staging_base
```
In the container
```
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.3/
cd ~
git clone https://github.com/pytorch/FBGEMM.git
cd FBGEMM/fbgemm_gpu
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
pip install update hypothesis

# in fbgemm_gpu folder
# build for the current ROCm architecture
gpu_arch="$(/opt/rocm/bin/rocminfo | grep -o -m 1 'gfx.*')"
export PYTORCH_ROCM_ARCH=$gpu_arch
python setup.py install develop
# or build for specific ROCm architectures
export PYTORCH_ROCM_ARCH="gfx906;gfx908"
python setup.py install develop
# otherwise the build will be for the default architectures gfx906;gfx908;gfx90a
```

##### Build on bare metal
Please refer to the installation instructions of ROCm5.3 [here][6]. Take the installation on Ubuntu20.04 as an example
```
sudo apt-get update
wget https://repo.radeon.com/amdgpu-install/5.3/ubuntu/focal/amdgpu-install_5.3.50300-1_all.deb
sudo apt-get install ./amdgpu-install_5.3.50300-1_all.deb
sudo amdgpu-install --usecase=hiplibsdk,rocm --no-dkms
```
MIOpen is required and needs to be installed separately.
```
sudo apt-get install miopen-hip miopen-hip-dev
```
The remaining steps are the same as the "in the container" section.

##### Run the tests on ROCm
Please add `FBGEMM_TEST_WITH_ROCM=1` flag when running tests on ROCm.
```
cd test
FBGEMM_TEST_WITH_ROCM=1 python split_table_batched_embeddings_test.py
```

## Issues

Building is CMAKE based and keeps state across install runs.
Specifying the CUDA architectures in the command line once is enough.
However on failed builds (missing dependencies ..) this can cause problems
and using
```bash
python setup.py clean
```
to remove stale cached state can be helpful.

## Examples

The tests (in test folder) and benchmarks (in bench folder) are some great
examples of using FBGEMM_GPU.

## Build Notes
FBGEMM_GPU uses a scikit-build CMAKE-based build flow.

### Dependencies
FBGEMM_GPU requires nvcc and a Nvidia GPU with
compute capability of 3.5+.

+ ###### CUB

CUB is now included with CUDA 11.1+ - the section below will still be needed for lower CUDA versions (once they are tested).

For the [CUB][1] build time dependency, if you are using conda, you can continue with
```
conda install -c bottler nvidiacub
```
Otherwise download the CUB library from https://github.com/NVIDIA/cub/releases and unpack it to a folder of your choice. Define the environment variable CUB_DIR before building and point it to the directory that contains CMakeLists.txt for CUB. For example on Linux/Mac,

```
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_DIR=$PWD/cub-1.10.0
```

+ ###### PyTorch, Jinja2, scikit-build
[PyTorch][2], [Jinja2][3] and scikit-build are **required** to build and run the table
batched embedding bag operator. One thing to note is that the implementation
of this op relies on the version of PyTorch 1.9 or later.

```
conda install scikit-build jinja2 ninja cmake
```

## Running FBGEMM_GPU

To run the tests or benchmarks after building FBGEMM_GPU (if tests or benchmarks
are built), use the following command:
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

## How FBGEMM_GPU works
For a high-level overview, design philosophy and brief descriptions of various
parts of FBGEMM_GPU please see our Wiki (work in progress).

## Full documentation
We have extensively used comments in our source files. The best and up-to-date
documentation is available in the source files.

# Building API Documentation

See [docs/README.md](docs/README.md).

## Join the FBGEMM community
See the [`CONTRIBUTING`](../CONTRIBUTING.md) file for how to help out.

## License
FBGEMM is BSD licensed, as found in the [`LICENSE`](../LICENSE) file.

[0]:https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
[1]:https://github.com/NVIDIA/cub
[2]:https://github.com/pytorch/pytorch
[3]:https://jinja.palletsprojects.com/en/2.11.x/
[4]:https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download
[5]:https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar
[6]:https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.3/page/How_to_Install_ROCm.html#_How_to_Install

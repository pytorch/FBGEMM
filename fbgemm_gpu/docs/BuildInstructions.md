# FBGEMM_GPU Build Instructions

The most up-to-date instructions are embedded in
[`setup_env.bash`](../../.github/scripts/setup_env.bash).  The general steps for
building FBGEMM_GPU are as follows:

1. Set up an isolated environment for building (Miniconda)
1. Install the relevant build tools (C/C++ compiler)
1. Set up for either CUDA, ROCm, or CPU build
1. Install PyTorch
1. Run the build


## Set Up an Isolated Build Environment

### Install Miniconda

Setting up a [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
environment is recommended for reproducible builds:

```sh
# Set the Miniconda prefix directory
miniconda_prefix=$HOME/miniconda

# Download the Miniconda installer
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Run the installer
bash miniconda.sh -b -p "$miniconda_prefix" -u

# Load the shortcuts
. ~/.bashrc

# Run updates
conda update -n base -c defaults -y conda
```

From here on out, all installation commands will be run against or inside a
Conda environment.


### Set Up the Conda Environment

Create a Conda environment with the specified Python version:

```sh
env_name=<ENV NAME>
python_version=3.10

# Create the environment
conda create -y --name "${env_name}" python="${python_version}"

# Upgrade PIP and pyOpenSSL package
conda run -n "${env_name}" pip install --upgrade pip
conda run -n "${env_name}" python -m pip install pyOpenSSL>22.1.0
```

## Install the Build Tools

### C/C++ Compiler

Install a version of the GCC toolchain that supports **C++17**.  Note that GCC
(as opposed to Clang for example) is required for GPU (CUDA) builds because
NVIDIA's `nvcc` relies on `gcc` and `g++` in the path.  The `sysroot` package
will also need to be installed to avoid issues with missing versioned symbols
when compiling FBGEMM_CPU:

```sh
conda install -n "${env_name}" -y gxx_linux-64=10.4.0 sysroot_linux-64=2.17 -c conda-forge
```

While newer versions of GCC can be used, binaries compiled under newer versions
of GCC will not be compatible with older systems such as Ubuntu 20.04 or CentOS
Stream 8, because the compiled library will reference symbols from versions of
`GLIBCXX` that the system's `libstdc++.so.6` will not support.  To see what
versions of GLIBC and GLIBCXX the available `libstdc++.so.6` supports:

```sh
libcxx_path=/path/to/libstdc++.so.6

# Print supported for GLIBC versions
objdump -TC "${libcxx_path}" | grep GLIBC_ | sed 's/.*GLIBC_\([.0-9]*\).*/GLIBC_\1/g' | sort -Vu | cat

# Print supported for GLIBCXX versions
objdump -TC "${libcxx_path}" | grep GLIBCXX_ | sed 's/.*GLIBCXX_\([.0-9]*\).*/GLIBCXX_\1/g' | sort -Vu | cat
```

### Other Build Tools

Install the other necessary build tools such as `ninja`, `cmake`, etc:

```sh
conda install -n "${env_name}" -y \
    click \
    cmake \
    hypothesis \
    jinja2 \
    ninja \
    numpy \
    scikit-build \
    wheel
```


## Set Up for CUDA Build

The CUDA build of FBGEMM_GPU requires `nvcc` that supports compute capability
3.5+.  Setting the machine up for CUDA builds of FBGEMM_GPU can be done either
through pre-built Docker images or through Conda installation on bare metal.
Note that neither a GPU nor the NVIDIA drivers need to be present for builds,
since they are only used at runtime.

### Docker Image

For setups through Docker, simply pull the pre-installed
[Docker image for CUDA](https://hub.docker.com/r/nvidia/cuda) for the desired
Linux distribution and CUDA version.

```sh
# Run for Ubuntu 22.04, CUDA 11.8
docker run -it --entrypoint "/bin/bash" nvidia/cuda:11.8.0-devel-ubuntu22.04
```

From there, the rest of the build environment may be constructed through Conda.

### Install CUDA

Install the full CUDA package through Conda, which includes
[NVML](https://developer.nvidia.com/nvidia-management-library-nvml):

```sh
cuda_version=11.7.1

# Install the full CUDA package
conda install -n "${env_name}" -y cuda -c "nvidia/label/cuda-${cuda_version}"
```

Ensure that at the minimum, **`cuda_runtime.h`** and **`libnvidia-ml.so`** are
found:

```sh
conda_prefix=$(conda run -n "${env_name}" printenv CONDA_PREFIX)
find "${conda_prefix}" -name cuda_runtime.h
find "${conda_prefix}" -name libnvidia-ml.so
```

### Install cuDNN

[cuDNN](https://developer.nvidia.com/cudnn) is a build-time dependency for the
CUDA variant of FBGEMM_GPU.  Download and extract the cuDNN package for the
given CUDA version:

```sh
# cuDNN package URLs can be found in: https://github.com/pytorch/builder/blob/main/common/install_cuda.sh
cudnn_url=https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz

# Download and unpack cuDNN
wget -q "${cudnn_url}" -O cudnn.tar.xz
```

### [OPTIONAL] Install CUB

[CUB](https://docs.nvidia.com/cuda/cub/index.html) is a build-time dependency for
the CUDA variant FBGEMM_GPU.  This must be installed separately for
**previous versions of CUDA (prior to 11.1)** since they did not come with CUB packaged.

To install CUB through Conda:

```sh
conda install -c bottler nvidiacub
```

Alternatively, CUB may be installed manually by downloading from the
[GitHub Releases](https://github.com/NVIDIA/cub/releases ) page and unpacking
the package:

```sh
# Download and unpack CUB
wget -q https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
```


## Set Up for ROCm Build

Setting the machine up for ROCm builds of FBGEMM_GPU can be done either through
pre-built Docker images or through bare metal.

### Docker Image

For setups through Docker, simply pull the pre-installed
[Docker image for ROCm](https://hub.docker.com/r/rocm/rocm-terminal) for the
desired ROCm CUDA version.

```sh
# Run for ROCm 5.4.2
docker run -it --entrypoint "/bin/bash" rocm/rocm-terminal:5.4.2
```

From there, the rest of the build environment may be constructed through Conda.

### Install ROCm

Install the full ROCm package through the operating system package manager. The
full instructions can be found in the
[ROCm installation guide](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.3/page/How_to_Install_ROCm.html):

```sh
# [OPTIONAL] Disable apt installation prompts
export DEBIAN_FRONTEND=noninteractive

# Update the repo DB
apt update

# Download the installer
wget https://repo.radeon.com/amdgpu-install/5.4.3/ubuntu/focal/amdgpu-install_5.4.50403-1_all.deb

# Run the installer
apt install ./amdgpu-install_5.4.50403-1_all.deb

# Install ROCm
amdgpu-install -y --usecase=hiplibsdk,rocm --no-dkms
```

### Install MIOpen

[MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) is a dependency for the
ROCm variant of FBGEMM_GPU that needs to be installed:

```sh
apt install hipify-clang miopen-hip miopen-hip-dev
```


## Install PyTorch

The official [PyTorch Homepage](https://pytorch.org/get-started/locally/) contains
the most authoritative instructions on how to install PyTorch, either through
Conda or through PIP.

### Installation Through Conda

```sh
# Install the latest nightly
conda install -n "${env_name}" -y pytorch -c pytorch-nightly
# Install the latest test (RC)
conda install -n "${env_name}" -y pytorch -c pytorch-test
# Install a specific version
conda install -n "${env_name}" -y pytorch==1.13.1 -c pytorch
```

Note that installing PyTorch through Conda without specifying a version (as in
the case of nightly builds) may not always be reliable.  For example, it is known
that the GPU builds for PyTorch nightlies arrive in Conda 2 hours later than the
CPU-only builds.  As such, a Conda installation of `pytorch-nightly` in that time
window will silently fall back to installing the CPU-only version.

Also note that, because both the GPU and CPU-only versions of PyTorch are placed
into the same artifact bucket, the PyTorch variant that is selected during
installation will depend on whether or not CUDA is installed on the system.  Thus
for GPU builds, it is important to install CUDA first prior to PyTorch.

### Installation Through PIP

Note that PIP is the only choice of installation of PyTorch for ROCm builds.

```sh
# Install the latest nightly
conda run -n "${env_name}" pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu117/
# Install the latest test (RC)
conda run -n "${env_name}" pip install --pre torch --extra-index-url https://download.pytorch.org/whl/test/cu117/
# Install a specific version
conda run -n "${env_name}" pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117/
# Install the latest nightly (ROCm 5.3)
conda run -n "${env_name}" pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.3/
```

### Post-Install Checks

Verify the PyTorch installation with an `import` test:

```sh
conda run -n "${env_name}" python -c "import torch.distributed"
```

For the GPU variant of PyTorch, ensure that at the minimum, **`cuda_cmake_macros.h`**
is found:

```sh
conda_prefix=$(conda run -n "${env_name}" printenv CONDA_PREFIX)
find "${conda_prefix}" -name cuda_cmake_macros.h
```


## Build the FBGEMM_GPU Package

### Preparing the Build

Clone the repo along with its submodules, and install the `requirements.txt`:

```sh
# !! Run inside the Conda environment !!

# Select a version tag
FBGEMM_VERSION=v0.4.0

# Clone the repo along with its submodules
git clone --recursive -b ${FBGEMM_VERSION} https://github.com/pytorch/FBGEMM.git fbgemm_${FBGEMM_VERSION}

# Install additional required packages for building and testing
cd fbgemm_${FBGEMM_VERSION}/fbgemm_gpu
pip install requirements.txt
```

### The Build Process

The FBGEMM_GPU build process uses a scikit-build CMake-based build flow, and it
keeps state across install runs.  As such, builds can become stale and can cause
problems when re-runs are attempted after a build failure due to missing
dependencies, etc.  To address this, simply clear the build cache:

```sh
# !! Run in fbgemm_gpu/ directory inside the Conda environment !!

python setup.py clean
```

### CUDA Build

Building FBGEMM_GPU for CUDA requires both NVML and cuDNN to be installed and
made available to the build through environment variables:

```sh
# !! Run in fbgemm_gpu/ directory inside the Conda environment !!

# [OPTIONAL] Specify the CUDA installation paths
# This may be required if CMake is unable to find nvcc
export CUDACXX=/path/to/nvcc
export CUDA_BIN_PATH=/path/to/cuda/installation

# [OPTIONAL] Provide the CUB installation directory (applicable only to CUDA versions prior to 11.1)
export CUB_DIR=/path/to/cub

# Specify cuDNN header and library paths
export CUDNN_INCLUDE_DIR=/path/to/cudnn/include
export CUDNN_LIBRARY=/path/to/cudnn/lib

# Specify NVML path
export NVML_LIB_PATH=/path/to/libnvidia-ml.so

# Update to reflect the version of Python in the Conda environment
python_tag=py310
package_name=fbgemm_gpu

# Build for SM70/80 (V100/A100 GPU); update as needed
# If not specified, only the CUDA architecture supported by current system will be targeted
# If no CUDA device is present either, all CUDA architectures will be targeted
cuda_arch_list=7.0;8.0

# Build the wheel artifact only
python setup.py bdist_wheel \
    --package_name="${package_name}" \
    --python-tag="${python_tag}" \
    --plat-name=manylinux1_x86_64 \
    --nvml_lib_path=${NVML_LIB_PATH} \
    -DTORCH_CUDA_ARCH_LIST="${cuda_arch_list}"

# Build and install the library into the Conda environment
python setup.py install \
    --nvml_lib_path=${NVML_LIB_PATH} \
    -DTORCH_CUDA_ARCH_LIST="${cuda_arch_list}"
```

### ROCm Build

For ROCm builds, `ROCM_PATH` and `PYTORCH_ROCM_ARCH` need to be specified:

```sh
# !! Run in fbgemm_gpu/ directory inside the Conda environment !!

# Build for the ROCm architecture on current machine; update as needed (e.g. 'gfx906;gfx908;gfx90a')
export ROCM_PATH=/path/to/rocm
export PYTORCH_ROCM_ARCH=$(${ROCM_PATH}/bin/rocminfo | grep -o -m 1 'gfx.*')

python_tag=py310
package_name=fbgemm_gpu_rocm

# Build the wheel artifact only
python setup.py bdist_wheel \
    --package_name="${package_name}" \
    --python-tag="${python_tag}" \
    --plat-name=manylinux1_x86_64

# Build and install the library into the Conda environment
python setup.py install develop
```

### CPU-Only Build

For CPU-only builds, the `--cpu_only` needs to be specified:

```sh
# !! Run in fbgemm_gpu/ directory inside the Conda environment !!

python_tag=py310
package_name=fbgemm_gpu_cpu

# Build the wheel artifact only
python setup.py bdist_wheel \
    --package_name="${package_name}" \
    --python-tag="${python_tag}" \
    --plat-name=manylinux1_x86_64 \
    --cpu_only

# Build and install the library into the Conda environment
python setup.py install --cpu_only
```

### Post-Build Checks

After the build completes, it is useful to check the built library and verify
the version numbers of GLIBCXX referenced as well as the availability of certain
function symbols:

```sh
# !! Run in fbgemm_gpu/ directory inside the Conda environment !!

# Locate the built .SO file
fbgemm_gpu_lib_path=$(find . -name fbgemm_gpu_py.so)

# Note the versions of GLIBCXX referenced by the .SO
# The libstdc++.so.6 available on the install target must support these versions
objdump -TC "${fbgemm_gpu_lib_path}" | grep GLIBCXX | sed 's/.*GLIBCXX_\([.0-9]*\).*/GLIBCXX_\1/g' | sort -Vu | cat

# Test for the existence of a given function symbol in the .SO
nm -gDC "${fbgemm_gpu_lib_path}" | grep " fbgemm_gpu::merge_pooled_embeddings("
nm -gDC "${fbgemm_gpu_lib_path}" | grep " fbgemm_gpu::jagged_2d_to_dense("
```

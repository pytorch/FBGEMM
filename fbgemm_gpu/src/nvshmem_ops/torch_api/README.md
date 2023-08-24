### Pytorch API
```
Get FBGEMM + AlltoAll Throughput or Correctness Check: (--exp=fwd/bwd/correct_check/profile)
torchrec_oemae_throughput -- --nDev=8 --exp=fwd --n_loop=1024
```

### Install FBGEMM
```
# Create Conda env:
conda create -y --name my_fbgemm python=3.8
# Enter the environment
conda activate my_fbgemm
# You might need to run this due to a recent bug in conda
# https://github.com/conda/conda/issues/11885
conda init bash


# Install required packages:
# Install Pytorch-cuda Nightly
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia
# install building tools
conda install -y numpy scikit-build jinja2 ninja cmake hypothesis setuptools-git-versioning


# Install CUDNN
# Download cudnn for CUDA11.x from https://developer.nvidia.com/cudnn
tar -xvf ${CUDANN_PREFIX}.tar.xz

# Get the source code:
git clone --recursive https://github.com/pytorch/FBGEMM.git

# Build:
cd FBGEMM/fbgemm_gpu
export CUDANN_PREFIX=/pkgs/cudnn-linux-x86_64-8.9.2.26_cuda11-archive
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDNN_LIBRARY_PATH=~/${CUDANN_PREFIX}/lib
export CUDNN_INCLUDE_PATH=~/${CUDANN_PREFIX}/include

python3 setup.py -j 128 develop

python3 setup.py -j 128 install -DCUDACXX=/usr/local/cuda/bin/nvcc -DCUDNN_LIBRARY_PATH=~/${CUDANN_PREFIX}/lib -DCUDNN_INCLUDE_PATH=~/${CUDANN_PREFIX}/include
```

### Build FBGEMM kernel
##### Follow the instruction on: `https://www.internalfb.com/intern/wiki/FBGEMM/FBGEMM_GPU_OSS_Installation_Instructions/:
1. Install conda.
2. Install Pytorch using conda.
3. Install building tools.
4. `export LD_LIBRARY_PATH=$HOME/local/miniconda3/envs/my_fbgemm/lib:$LD_LIBRARY_PATH`
   `export LD_LIBRARY_PATH=$HOME/local/miniconda3/envs/my_fbgemm/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH`
5. make

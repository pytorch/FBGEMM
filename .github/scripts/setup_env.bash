#!/bin/bash

setup_miniconda () {
  miniconda_prefix="$1"
  if [ "$miniconda_prefix" == "" ]; then
    echo "Usage: setup_miniconda MINICONDA_PREFIX_PATH"
    echo "Example:"
    echo "    setup_miniconda /home/user/tmp/miniconda"
    exit 1
  fi
  if [ ! -f "${miniconda_prefix}/bin/conda" ]; then
    # Download miniconda if not exists
    mkdir -p "$miniconda_prefix"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$miniconda_prefix" -u
  fi
  # These variables will be exported outside
  export PATH="${miniconda_prefix}/bin:${PATH}"
  export CONDA="${miniconda_prefix}"
}

create_conda_environment () {
  env_name="$1"
  python_version="$2"
  pytorch_channel_name="$3"
  pytorch_cuda_version="$4"
  if [ "$python_version" == "" ]; then
    echo "Usage: create_conda_environment ENV_NAME PYTHON_VERSION PYTORCH_CHANNEL_NAME PYTORCH_CUDA_VERSION"
    echo "Example:"
    echo "    create_conda_environment build_binary 3.10 pytorch-nightly 11.7"
    exit 1
  fi
  # -y removes existing environment
  conda create -y --name "$env_name" python="$python_version"
  if [ "$pytorch_cuda_version" == "" ]; then
    # CPU version
    conda install -n "$env_name" -y pytorch cpuonly -c "$pytorch_channel_name"
  else
    # GPU version
    conda install -n "$env_name" -y pytorch pytorch-cuda="$pytorch_cuda_version" -c "$pytorch_channel_name" -c nvidia
  fi
}

install_cudnn () {
  install_path="$1"
  if [ "$install_path" == "" ]; then
    echo "Usage: install_cudnn INSTALL_PATH"
    echo "Example:"
    echo "    install_cudnn \$(pwd)/cudnn_install"
    exit 1
  fi

  rm -rf "$install_path"
  mkdir -p "$install_path"

  # Install cuDNN manually
  # See https://github.com/pytorch/builder/blob/main/common/install_cuda.sh
  mkdir -p tmp_cudnn
  cd tmp_cudnn || exit
  wget -q https://ossci-linux.s3.amazonaws.com/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz -O cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
  tar xf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
  rm -rf "${install_path:?}/include"
  rm -rf "${install_path:?}/lib"
  mv cudnn-linux-x86_64-8.5.0.96_cuda11-archive/include "$install_path"
  mv cudnn-linux-x86_64-8.5.0.96_cuda11-archive/lib "$install_path"
  cd ../
  rm -rf tmp_cudnn
}

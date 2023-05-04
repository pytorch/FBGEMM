# FBGEMM_GPU Installation Instructions

The most up-to-date instructions are embedded in
[`setup_env.bash`](../../.github/scripts/setup_env.bash).  The general steps for
building FBGEMM_GPU are as follows:

1. Set up an isolated environment for CUDA, ROCm, or CPU runtime
1. Install the relevant Python libraries
1. Install PyTorch
1. Install the FBGEMM_GPU package
1. Run post-installation checks

The shortened summary of the installation steps:

```sh
# CUDA Nightly
pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu117/
pip install fbgemm-gpu-nightly

# CUDA Release
pip install --pre torch --extra-index-url https://download.pytorch.org/whl/test/cu117/
pip install fbgemm-gpu

# CPU-only Nightly
pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu/
pip install fbgemm-gpu-nightly-cpu

# CPU-only Release
pip install --pre torch --extra-index-url https://download.pytorch.org/whl/test/cpu/
pip install fbgemm-gpu-cpu

# Test the installation
python -c "import torch; import fbgemm_gpu"
```


## Set Up CUDA Environment

The CUDA variant of FBGEMM_GPU requires an NVIDIA GPU installed to the machine,
along with working NVIDIA drivers installed; otherwise or the library will fall
back to running the CPU version of the operators.

The FBGEMM_GPU CUDA package is currently only built for the SM70 and SM80
architectures (V100 and A100 GPUs respectively).  Support for other architectures
can be achieved by building the pacakge from scratch, but is not guaranteed to
work.

### Install NVIDIA Drivers

The NVIDIA display drivers must be installed on the system prior to all other
environment setup.  The steps provided by
[NVIDIA](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
and
[PyTorch](https://github.com/pytorch/test-infra/blob/main/.github/actions/setup-nvidia/action.yml)
are the most authoritative instructions for doing this.  Driver
setup may be verified with the `nvidia-smi` command:

```sh
nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.76       Driver Version: 515.76       CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A10G         Off  | 00000000:00:1E.0 Off |                    0 |
|  0%   31C    P0    59W / 300W |      0MiB / 23028MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Set Up the Docker Container and Conda Environment

It is recommended, though not required, to install and run FBGEMM_GPU through a
Docker setup for isolation and reproducibility of the CUDA environment.

The NVIDIA-Docker runtime needs to be installed to expose the driver to the
container.  The install steps provided by
[PyTorch](https://github.com/pytorch/test-infra/blob/main/.github/actions/setup-nvidia/action.yml)
provide details on how to achieve this.

Once this is done, follow the steps in the
[Build Instructions](./BuildInstructions.md) for pulling the CUDA Docker image
and launching a container.

From there, the rest of the runtime environment may be constructed through Conda.
Follow the steps for installing MiniConda and setting up the Conda environment
from the [Build Instructions](./BuildInstructions.md).

### Install CUDA

If the OS / Docker environment does not already contain the full CUDA runtime,
follow the steps in the [Build Instructions](./BuildInstructions.md) for
installing the CUDA toolkit inside a Conda environment.


## Set Up ROCm Environment

The ROCm variant of FBGEMM_GPU requires an AMD GPU installed to the machine,
along with working AMDGPU drivers installed; otherwise or the library will fall
back to running the CPU version of the operators.

### Install AMDGPU Drivers

The AMDGPU display drivers must be installed on the system prior to all other
environment setup.  The steps provided by
[AMD](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.5/page/How_to_Install_ROCm.html)
are the most authoritative instructions for doing this.  Driver
setup may be verified with the `rocm-smi` command:

```sh
rocm-smi

======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp (DieEdge)  AvgPwr  SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
0    33.0c           37.0W   300Mhz  1200Mhz  0%   auto  290.0W    0%   0%
1    32.0c           39.0W   300Mhz  1200Mhz  0%   auto  290.0W    0%   0%
2    33.0c           37.0W   300Mhz  1200Mhz  0%   auto  290.0W    0%   0%
================================================================================
============================= End of ROCm SMI Log ==============================
```

### Set Up the Docker Container and Conda Environment

It is recommended, though not required, to install and run FBGEMM_GPU through a
Docker setup for isolation and reproducibility of the ROCm environment, which
can be difficult to set up.

Follow the steps in the [Build Instructions](./BuildInstructions.md) for pulling
the full ROCm Docker image and launching a container.

From there, the rest of the runtime environment may be constructed through Conda.
Follow the steps for installing MiniConda and setting up the Conda environment
from the [Build Instructions](./BuildInstructions.md).


## Install Python Libraries

Install the other relevant Python libraries for working with FBGEMM_GPU:

```sh
conda install -n "${env_name}" -y \
    hypothesis \
    numpy \
    scikit-build
```


## Install PyTorch

Follow the steps in the [Build Instructions](./BuildInstructions.md) for
installing PyTorch inside a Conda environment.


## Install the FBGEMM_GPU Package

FBGEMM_GPU installation is done through PIP.

### Install the CUDA Variant

```sh
# !! Run inside the Conda environment !!

# Release GPU
pip install fbgemm-gpu

# Nightly GPU
pip install fbgemm-gpu-nightly
```

### Install the ROCm Variant

As of time of writing, there no packages yet for the ROCm variant of FBGEMM_GPU.

### Install the CPU-Only Variant

```sh
# !! Run inside the Conda environment !!

# Release CPU
pip install fbgemm-gpu-cpu

# Nightly CPU
pip install fbgemm-gpu-nightly-cpu
```

## Post-Installation Checks

After installation, run an import test to ensure that the library is correctly
linked and set up.

```sh
# !! Run inside the Conda environment !!

python -c "import torch; import fbgemm_gpu; print(torch.ops.fbgemm.merge_pooled_embeddings)"
```

### Undefined Symbols

A common error that is encountered is the failure to import FBGEMM_GPU in Python,
which has the following error signature:

```sh
Traceback (most recent call last):
  File "/root/miniconda/envs/mycondaenv/lib/python3.10/site-packages/torch/_ops.py", line 565, in __getattr__
    op, overload_names = torch._C._jit_get_operation(qualified_op_name)
RuntimeError: No such operator fbgemm::jagged_2d_to_dense
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/root/miniconda/envs/mycondaenv/lib/python3.10/site-packages/fbgemm_gpu-0.4.1.post47-py3.10-linux-aarch64.egg/fbgemm_gpu/__init__.py", line 21, in <module>
    from . import _fbgemm_gpu_docs  # noqa: F401, E402
  File "/root/miniconda/envs/mycondaenv/lib/python3.10/site-packages/fbgemm_gpu-0.4.1.post47-py3.10-linux-aarch64.egg/fbgemm_gpu/_fbgemm_gpu_docs.py", line 18, in <module>
    torch.ops.fbgemm.jagged_2d_to_dense,
  File "/root/miniconda/envs/mycondaenv/lib/python3.10/site-packages/torch/_ops.py", line 569, in __getattr__
    raise AttributeError(
AttributeError: '_OpNamespace' 'fbgemm' object has no attribute 'jagged_2d_to_dense'
ERROR conda.cli.main_run:execute(47): `conda run python -c import fbgemm_gpu` failed. (See above for error)
/root/miniconda/envs/mycondaenv/lib/python3.10/site-packages/fbgemm_gpu-0.4.1.post47-py3.10-linux-aarch64.egg/fbgemm_gpu/fbgemm_gpu_py.so: undefined symbol: _ZN6fbgemm48FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfAvx2ItLi2EEEvPKT_miPh
```

Undefined symbols can appear in an FBGEMM_GPU installation for the following
reasons:

1.  The runtime libraries that FBGEMM_GPU depends on, such as `libnvidia-ml.so`
    or `libtorch.so`, are either not installed correctly or are not visible in `LD_LIBRARY_PATH`.
1.  The FBGEMM_GPU package was built incorrectly and contains declarations that
    were not linked (see [1618](https://github.com/pytorch/FBGEMM/issues/1618) for example).

In the former case, this may be resolved by re-installing the relevant packages
and/or manually updating `LD_LIBRARY_PATH`.

In the latter case, this is a serious building and packaging issue that should
be reported to the FBGEMM developers.

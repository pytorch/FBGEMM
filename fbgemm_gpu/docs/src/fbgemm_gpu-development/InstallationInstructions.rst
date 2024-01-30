Installation Instructions
=========================

**Note:** The most up-to-date installation instructions are embedded in a set
of scripts bundled in the FBGEMM repo under
`setup_env.bash <https://github.com/pytorch/FBGEMM/blob/main/.github/scripts/setup_env.bash>`_.

The general steps for installing FBGEMM_GPU are as follows:

#. Set up an isolated build environment.
#. Set up the toolchain for either a CPU-only, CUDA, or ROCm runtime.
#. Install PyTorch.
#. Install the FBGEMM_GPU package.
#. Run post-installation checks.


Set Up CPU-Only Environment
---------------------------

Follow the instructions for setting up the Conda environment at
:ref:`fbgemm-gpu.build.setup.env`, followed by
:ref:`fbgemm-gpu.install.libraries`.


Set Up CUDA Environment
-----------------------

The CUDA variant of FBGEMM_GPU requires an NVIDIA GPU installed to the machine,
along with working NVIDIA drivers installed; otherwise or the library will fall
back to running the CPU version of the operators.

The FBGEMM_GPU CUDA package is currently only built for the SM70 and SM80
architectures (V100 and A100 GPUs respectively). Support for other architectures
can be achieved by building the package from scratch, but is not guaranteed to
work (especially for older architectures).

Install NVIDIA Drivers
~~~~~~~~~~~~~~~~~~~~~~

The NVIDIA display drivers must be installed on the system prior to all other
environment setup. The steps provided by
`NVIDIA <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`__
and
`PyTorch <https://github.com/pytorch/test-infra/blob/main/.github/actions/setup-nvidia/action.yml>`__
are the most authoritative instructions for doing this. Driver setup may
be verified with the ``nvidia-smi`` command:

.. code:: sh

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

Set Up the CUDA Docker Container and Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended, though not required, to install and run FBGEMM_GPU through a
Docker setup for isolation and reproducibility of the CUDA environment.

The NVIDIA-Docker runtime needs to be installed to expose the driver to the
container. The install steps provided by
`PyTorch <https://github.com/pytorch/test-infra/blob/main/.github/actions/setup-nvidia/action.yml>`__
provide details on how to achieve this.

Once this is done, follow the instructions in
:ref:`fbgemm-gpu.build.setup.cuda.image` for pulling the CUDA Docker image
and launching a container.

From there, the rest of the runtime environment may be constructed through
Conda. Follow the instructions for setting up the Conda environment at
:ref:`fbgemm-gpu.build.setup.env`, followed by
:ref:`fbgemm-gpu.install.libraries`.

Install the CUDA Runtime
~~~~~~~~~~~~~~~~~~~~~~~~

If the OS / Docker environment does not already contain the full CUDA runtime,
follow the instructions in :ref:`fbgemm-gpu.build.setup.cuda.install` for
installing the CUDA toolkit inside a Conda environment.


Set Up ROCm Environment
-----------------------

The ROCm variant of FBGEMM_GPU requires an AMD GPU installed to the machine,
along with working AMDGPU drivers installed; otherwise or the library will fall
back to running the CPU version of the operators.

Install AMDGPU Drivers
~~~~~~~~~~~~~~~~~~~~~~

The AMDGPU display drivers must be installed on the system prior to all other
environment setup. The steps provided by
`AMD <https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.5/page/How_to_Install_ROCm.html>`__
are the most authoritative instructions for doing this. Driver setup may be
verified with the ``rocm-smi`` command:

.. code:: sh

  rocm-smi

  ======================= ROCm System Management Interface =======================
  ================================= Concise Info =================================
  GPU  Temp (DieEdge)  AvgPwr  SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
  0    33.0c           37.0W   300Mhz  1200Mhz  0%   auto  290.0W    0%   0%
  1    32.0c           39.0W   300Mhz  1200Mhz  0%   auto  290.0W    0%   0%
  2    33.0c           37.0W   300Mhz  1200Mhz  0%   auto  290.0W    0%   0%
  ================================================================================
  ============================= End of ROCm SMI Log ==============================

Set Up the ROCm Docker Container and Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended, though not required, to install and run FBGEMM_GPU through a
Docker setup for isolation and reproducibility of the ROCm environment, which
can be difficult to set up.

Follow the instructions in :ref:`fbgemm-gpu.build.setup.rocm.image` for
pulling the full ROCm Docker image and launching a container.

From there, the rest of the runtime environment may be constructed through
Conda. Follow the instructions for setting up the Conda environment at
:ref:`fbgemm-gpu.build.setup.rocm.install`, followed by
:ref:`fbgemm-gpu.install.libraries`.

.. _fbgemm-gpu.install.libraries:

Install Python Libraries
------------------------

Install the relevant Python libraries for working with FBGEMM_GPU:

.. code:: sh

  conda install -n "${env_name}" -y \
      hypothesis \
      numpy \
      scikit-build


Install PyTorch
---------------

Follow the instructions in :ref:`fbgemm-gpu.build.setup.pytorch.install`
for installing PyTorch inside a Conda environment.


Install the FBGEMM_GPU Package
------------------------------

Install through PyTorch PIP
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch PIP is the preferred channel for installing FBGEMM_GPU:

.. code:: sh

  # !! Run inside the Conda environment !!

  # CPU-only Nightly
  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu/
  pip install --pre fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cpu/

  # CPU-only Release
  pip install torch --index-url https://download.pytorch.org/whl/cpu/
  pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cpu/

  # CUDA Nightly
  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121/
  pip install --pre fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cu121/

  # CUDA Release
  pip install torch --index-url https://download.pytorch.org/whl/cu121/
  pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121/

  # ROCm Nightly
  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm5.6/
  pip install --pre fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/rocm5.6/

  # Test the installation
  python -c "import torch; import fbgemm_gpu"

Install through Public PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

  # !! Run inside the Conda environment !!

  # CPU-Only Nightly
  pip install fbgemm-gpu-nightly-cpu

  # CPU-Only Release
  pip install fbgemm-gpu-cpu

  # CUDA Nightly
  pip install fbgemm-gpu-nightly

  # CUDA Release
  pip install fbgemm-gpu

As of time of writing, packages for the ROCm variant of FBGEMM_GPU are not
released to public PyPI.


Post-Installation Checks
------------------------

After installation, run an import test to ensure that the library is correctly
linked and set up.

.. code:: sh

  # !! Run inside the Conda environment !!

  python -c "import torch; import fbgemm_gpu; print(torch.ops.fbgemm.merge_pooled_embeddings)"

Undefined Symbols
~~~~~~~~~~~~~~~~~

A common error that is encountered is the failure to import FBGEMM_GPU in
Python, which has the following error signature:

.. code:: sh

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

In general, undefined symbols can appear in an FBGEMM_GPU installation for the
following reasons:

#.  The runtime libraries that FBGEMM_GPU depends on, such as ``libnvidia-ml.so``
    or ``libtorch.so``, are either not installed correctly or are not visible
    in ``LD_LIBRARY_PATH``.

#.  The FBGEMM_GPU package was built incorrectly and contains
    declarations that were not linked (see
    `PR 1618 <https://github.com/pytorch/FBGEMM/issues/1618>`__ for example).


In the former case, this may be resolved by re-installing the relevant packages
and/or manually updating ``LD_LIBRARY_PATH``.

In the latter case, this is a serious building and/or packaging issue tha should
be reported to the FBGEMM developers.

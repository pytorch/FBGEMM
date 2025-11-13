Build Instructions
==================

**Note:** The most up-to-date build instructions are embedded in a set of
scripts bundled in the FBGEMM repo under
`setup_env.bash <https://github.com/pytorch/FBGEMM/blob/main/.github/scripts/setup_env.bash>`_.

The currently available FBGEMM GenAI build variants are:

* CUDA

The general steps for building FBGEMM GenAI are as follows:

#. Set up an isolated build environment.
#. Set up the toolchain for either a CUDA build.
#. Install PyTorch.
#. Run the build script.


.. _fbgemm-genai.build.setup.env:

Set Up an Isolated Build Environment
------------------------------------

Follow the instructions to set up the Conda environment:

#. :ref:`fbgemm-gpu.build.setup.env`
#. :ref:`fbgemm-gpu.build.setup.cuda`
#. :ref:`fbgemm-gpu.build.setup.tools.install`
#. :ref:`fbgemm-gpu.build.setup.pytorch.install`

Installing PyTorch for CUDA Builds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For CUDA builds, install PyTorch with matching CUDA version support:

.. code:: sh

  # !! Run inside the Conda environment !!

  # For CUDA 12.9 with PyTorch nightly (recommended for latest features)
  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129

  # For CUDA 12.8 with PyTorch stable
  pip install torch --index-url https://download.pytorch.org/whl/cu128

  # Verify PyTorch installation
  python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"


Other Pre-Build Setup
---------------------

As FBGEMM GenAI leverages the same build process as FBGEMM_GPU, please refer to
:ref:`fbgemm-gpu.build.prepare` for additional pre-build setup information.

.. _fbgemm-genai.build.prepare:

Preparing the Build
~~~~~~~~~~~~~~~~~~~

Clone the repo along with its submodules, and install ``requirements_genai.txt``:

.. code:: sh

  # !! Run inside the Conda environment !!

  # Select a version tag
  FBGEMM_VERSION=v1.4.0

  # Clone the repo along with its submodules
  git clone --recursive -b ${FBGEMM_VERSION} https://github.com/pytorch/FBGEMM.git fbgemm_${FBGEMM_VERSION}

  # Install additional required packages for building and testing
  cd fbgemm_${FBGEMM_VERSION}/fbgemm_gpu
  pip install -r requirements_genai.txt

Initialize Git Submodules
~~~~~~~~~~~~~~~~~~~~~~~~~

FBGEMM GenAI relies on several submodules, including CUTLASS for optimized CUDA kernels.
If you didn't use ``--recursive`` when cloning, initialize the submodules:

.. code:: sh

  # Sync and initialize all submodules including CUTLASS
  git submodule sync
  git submodule update --init --recursive

  # Verify CUTLASS is available
  ls external/cutlass/include

Install NCCL for Distributed Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For distributed communication support, install NCCL via conda:

.. code:: sh

  # !! Run inside the Conda environment !!
  conda install -c conda-forge nccl -y

Set Wheel Build Variables
~~~~~~~~~~~~~~~~~~~~~~~~~

When building out the Python wheel, the package name, Python version tag, and
Python platform name must first be properly set:

.. code:: sh

  # Set the package name depending on the build variant
  export package_name=fbgemm_genai_{cuda}

  # Set the Python version tag.  It should follow the convention `py<major><minor>`,
  # e.g. Python 3.13 --> py313
  export python_tag=py313

  # Determine the processor architecture
  export ARCH=$(uname -m)

  # Set the Python platform name for the Linux case
  export python_plat_name="manylinux_2_28_${ARCH}"
  # For the macOS (x86_64) case
  export python_plat_name="macosx_10_9_${ARCH}"
  # For the macOS (arm64) case
  export python_plat_name="macosx_11_0_${ARCH}"
  # For the Windows case
  export python_plat_name="win_${ARCH}"

.. _fbgemm-genai.build.process.cuda:

CUDA Build
----------

Building FBGEMM GenAI for CUDA requires both NVML and cuDNN to be installed and
made available to the build through environment variables.  The presence of a
CUDA device, however, is not required for building the package.

Similar to CPU-only builds, building with Clang + ``libstdc++`` can be enabled
by appending ``--cxxprefix=$CONDA_PREFIX`` to the build command, presuming the
toolchains have been properly installed.

Environment Setup for CUDA Builds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up the necessary environment variables for a CUDA build:

.. code:: sh

  # !! Run in fbgemm_gpu/ directory inside the Conda environment !!

  # Specify CUDA paths (adjust to your CUDA installation)
  export CUDA_HOME="/usr/local/cuda"
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

  # Specify NVML filepath (usually in CUDA stubs directory)
  export NVML_LIB_PATH="${CUDA_HOME}/lib64/stubs/libnvidia-ml.so"

  # Specify NCCL filepath (installed via conda)
  export NCCL_LIB_PATH="${CONDA_PREFIX}/lib/libnccl.so"

CUDA Architecture Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure the target CUDA architectures for your hardware:

.. code:: sh

  # Build for SM70/80 (V100/A100 GPU); update as needed
  # If not specified, only the CUDA architecture supported by current system will be targeted
  # If not specified and no CUDA device is present either, all CUDA architectures will be targeted
  cuda_arch_list=7.0;8.0

  # For NVIDIA Blackwell architecture (GB100, GB200):
  # cuda_arch_list=10.0a
  # export TORCH_CUDA_ARCH_LIST="10.0a"

  # Unset TORCH_CUDA_ARCH_LIST if it exists, bc it takes precedence over
  # -DTORCH_CUDA_ARCH_LIST during the invocation of setup.py
  unset TORCH_CUDA_ARCH_LIST

Optional NVCC Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional NVCC configuration options:

.. code:: sh

  # [OPTIONAL] Allow NVCC to use host compilers that are newer than what NVCC officially supports
  nvcc_prepend_flags=(
    -allow-unsupported-compiler
  )

  # [OPTIONAL] If clang is the host compiler, set NVCC to use libstdc++ since libc++ is not supported
  nvcc_prepend_flags+=(
    -Xcompiler -stdlib=libstdc++
    -ccbin "/path/to/clang++"
  )

  # [OPTIONAL] Set NVCC_PREPEND_FLAGS as needed
  export NVCC_PREPEND_FLAGS="${nvcc_prepend_flags[@]}"

  # [OPTIONAL] Enable verbose NVCC logs
  export NVCC_VERBOSE=1

Building the Package
~~~~~~~~~~~~~~~~~~~~

.. code:: sh

  # !! Run in fbgemm_gpu/ directory inside the Conda environment !!

  # [OPTIONAL] Specify the CUDA installation paths
  # This may be required if CMake is unable to find nvcc
  export CUDACXX=/path/to/nvcc
  export CUDA_BIN_PATH=/path/to/cuda/installation

  # Build the wheel artifact only
  python setup.py bdist_wheel \
      --build-target=genai \
      --build-variant=cuda \
      --python-tag="${python_tag}" \
      --plat-name="${python_plat_name}" \
      --nvml_lib_path=${NVML_LIB_PATH} \
      --nccl_lib_path=${NCCL_LIB_PATH} \
      -DTORCH_CUDA_ARCH_LIST="${cuda_arch_list}"

  # Build and install the library into the Conda environment
  python setup.py install \
      --build-target=genai \
      --build-variant=cuda \
      --nvml_lib_path=${NVML_LIB_PATH} \
      --nccl_lib_path=${NCCL_LIB_PATH} \
      -DTORCH_CUDA_ARCH_LIST="${cuda_arch_list}"

.. _fbgemm-gpu.build.process.rocm:

ROCm Build
----------

For ROCm builds, ``ROCM_PATH`` and ``PYTORCH_ROCM_ARCH`` need to be specified.
The presence of a ROCm device, however, is not required for building
the package.

Similar to CUDA builds, building with Clang + ``libstdc++`` can be enabled by
appending ``--cxxprefix=$CONDA_PREFIX`` to the build command, presuming the
toolchains have been properly installed.

.. code:: sh

  # !! Run in fbgemm_gpu/ directory inside the Conda environment !!

  export ROCM_PATH=/path/to/rocm

  # [OPTIONAL] Enable verbose HIPCC logs
  export HIPCC_VERBOSE=1

  # Build for the target architecture of the ROCm device installed on the machine (e.g. 'gfx908,gfx90a,gfx942')
  # See https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html for list
  export PYTORCH_ROCM_ARCH=$(${ROCM_PATH}/bin/rocminfo | grep -o -m 1 'gfx.*')

  # Build the wheel artifact only
  python setup.py bdist_wheel \
      --build-target=genai \
      --build-variant=rocm \
      --python-tag="${python_tag}" \
      --plat-name="${python_plat_name}" \
      -DAMDGPU_TARGETS="${PYTORCH_ROCM_ARCH}" \
      -DHIP_ROOT_DIR="${ROCM_PATH}" \
      -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" \
      -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA"

  # Build and install the library into the Conda environment
  python setup.py install \
      --build-target=genai \
      --build-variant=rocm \
      -DAMDGPU_TARGETS="${PYTORCH_ROCM_ARCH}" \
      -DHIP_ROOT_DIR="${ROCM_PATH}" \
      -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" \
      -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA"

Post-Build Checks (For Developers)
----------------------------------

As FBGEMM GenAI leverages the same build process as FBGEMM_GPU, please refer to
:ref:`fbgemm-gpu.build.process.post-build` for information on additional
post-build checks.

Troubleshooting Build Issues
-----------------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **CUTLASS not found**: Ensure git submodules are initialized:

   .. code:: sh

     git submodule sync
     git submodule update --init --recursive

2. **CUDA version mismatch**: Ensure PyTorch CUDA version matches your system CUDA:

   .. code:: sh

     # Check system CUDA version
     nvcc --version

     # Check PyTorch CUDA version
     python -c "import torch; print(torch.version.cuda)"

3. **NVML/NCCL library not found**: Verify the library paths are correct:

   .. code:: sh

     # Check NVML exists
     ls -la ${NVML_LIB_PATH}

     # Check NCCL exists
     ls -la ${NCCL_LIB_PATH}

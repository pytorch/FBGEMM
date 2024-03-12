Build Instructions
==================

**Note:** The most up-to-date build instructions are embedded in a set of
scripts bundled in the FBGEMM repo under
`setup_env.bash <https://github.com/pytorch/FBGEMM/blob/main/.github/scripts/setup_env.bash>`_.

The general steps for building FBGEMM_GPU are as follows:

#. Set up an isolated build environment.
#. Set up the toolchain for either a CPU-only, CUDA, or ROCm build.
#. Install PyTorch.
#. Run the build script.


.. _fbgemm-gpu.build.setup.env:

Set Up an Isolated Build Environment
------------------------------------

Install Miniconda
~~~~~~~~~~~~~~~~~

Setting up a `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__
environment is recommended for reproducible builds:

.. code:: sh

  export PLATFORM_NAME="$(uname -s)-$(uname -m)"

  # Set the Miniconda prefix directory
  miniconda_prefix=$HOME/miniconda

  # Download the Miniconda installer
  wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-${PLATFORM_NAME}.sh" -O miniconda.sh

  # Run the installer
  bash miniconda.sh -b -p "$miniconda_prefix" -u

  # Load the shortcuts
  . ~/.bashrc

  # Run updates
  conda update -n base -c defaults -y conda

From here on out, all installation commands will be run against or
inside a Conda environment.

Set Up the Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a Conda environment with the specified Python version:

.. code:: sh

  env_name=<ENV NAME>
  python_version=3.12

  # Create the environment
  conda create -y --name ${env_name} python="${python_version}"

  # Upgrade PIP and pyOpenSSL package
  conda run -n ${env_name} pip install --upgrade pip
  conda run -n ${env_name} python -m pip install pyOpenSSL>22.1.0


Set Up for CPU-Only Build
-------------------------

Follow the instructions for setting up the Conda environment at
:ref:`fbgemm-gpu.build.setup.env`, followed by
:ref:`fbgemm-gpu.build.setup.tools.install`.


Set Up for CUDA Build
---------------------

The CUDA build of FBGEMM_GPU requires a recent version of ``nvcc`` **that
supports compute capability 3.5+**. Setting the machine up for CUDA builds of
FBGEMM_GPU can be done either through pre-built Docker images or through Conda
installation on bare metal. Note that neither a GPU nor the NVIDIA drivers need
to be present for builds, since they are only used at runtime.

.. _fbgemm-gpu.build.setup.cuda.image:

CUDA Docker Image
~~~~~~~~~~~~~~~~~

For setups through Docker, simply pull the pre-installed `Docker image
for CUDA <https://hub.docker.com/r/nvidia/cuda>`__ for the desired Linux
distribution and CUDA version.

.. code:: sh

  # Run for Ubuntu 22.04, CUDA 11.8
  docker run -it --entrypoint "/bin/bash" nvidia/cuda:11.8.0-devel-ubuntu22.04

From here, the rest of the build environment may be constructed through Conda,
as it is still the recommended mechanism for creating an isolated and
reproducible build environment.

.. _fbgemm-gpu.build.setup.cuda.install:

Install CUDA
~~~~~~~~~~~~

Install the full CUDA package through Conda, which includes
`NVML <https://developer.nvidia.com/nvidia-management-library-nvml>`__:

.. code:: sh

  # See https://anaconda.org/nvidia/cuda for all available versions of CUDA
  cuda_version=12.1.0

  # Install the full CUDA package
  conda install -n ${env_name} -y cuda -c "nvidia/label/cuda-${cuda_version}"

Verify that ``cuda_runtime.h`` and ``libnvidia-ml.so`` are found:

.. code:: sh

  conda_prefix=$(conda run -n ${env_name} printenv CONDA_PREFIX)

  find "${conda_prefix}" -name cuda_runtime.h
  find "${conda_prefix}" -name libnvidia-ml.so

Install cuDNN
~~~~~~~~~~~~~

`cuDNN <https://developer.nvidia.com/cudnn>`__ is a build-time
dependency for the CUDA variant of FBGEMM_GPU. Download and extract the
cuDNN package for the given CUDA version:

.. code:: sh

  # cuDNN package URLs for each platform and CUDA version can be found in:
  # https://github.com/pytorch/builder/blob/main/common/install_cuda.sh
  cudnn_url=https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz

  # Download and unpack cuDNN
  wget -q "${cudnn_url}" -O cudnn.tar.xz
  tar -xvf cudnn.tar.xz


Set Up for ROCm Build
---------------------

FBGEMM_GPU supports running on AMD (ROCm) devices. Setting the machine
up for ROCm builds of FBGEMM_GPU can be done either through pre-built
Docker images or through bare metal.

.. _fbgemm-gpu.build.setup.rocm.image:

ROCm Docker Image
~~~~~~~~~~~~~~~~~

For setups through Docker, simply pull the pre-installed `Minimal Docker
image for ROCm <https://hub.docker.com/r/rocm/rocm-terminal>`__ for the
desired ROCm version:

.. code:: sh

  # Run for ROCm 5.6.1
  docker run -it --entrypoint "/bin/bash" rocm/rocm-terminal:5.6.1

While the `full ROCm Docker image <https://hub.docker.com/r/rocm/dev-ubuntu-20.04>`__
comes with all ROCm packages pre-installed, it results in a very large Docker
container, and so for this reason, the minimal image is recommended for building
and running FBGEMM_GPU.

From here, the rest of the build environment may be constructed through Conda,
as it is still the recommended mechanism for creating an isolated and
reproducible build environment.

.. _fbgemm-gpu.build.setup.rocm.install:

Install ROCm
~~~~~~~~~~~~

Install the full ROCm package through the operating system package
manager. The full instructions can be found in the `ROCm installation
guide <https://rocm.docs.amd.com/en/latest/>`__:

.. code:: sh

  # [OPTIONAL] Disable apt installation prompts
  export DEBIAN_FRONTEND=noninteractive

  # Update the repo DB
  apt update

  # Download the installer
  wget -q https://repo.radeon.com/amdgpu-install/5.6.1/ubuntu/focal/amdgpu-install_5.6.50601-1_all.deb -O amdgpu-install.deb

  # Run the installer
  apt install ./amdgpu-install.deb

  # Install ROCm
  amdgpu-install -y --usecase=hiplibsdk,rocm --no-dkms

Install MIOpen
~~~~~~~~~~~~~~

`MIOpen <https://github.com/ROCmSoftwarePlatform/MIOpen>`__ is a
dependency for the ROCm variant of FBGEMM_GPU that needs to be
installed:

.. code:: sh

  apt install hipify-clang miopen-hip miopen-hip-dev


.. _fbgemm-gpu.build.setup.tools.install:

Install the Build Tools
-----------------------

The instructions in this section apply to builds for all variants of FBGEMM_GPU.

.. _fbgemm-gpu.build.setup.tools.install.compiler.gcc:

C/C++ Compiler (GCC)
~~~~~~~~~~~~~~~~~~~~

Install a version of the GCC toolchain **that supports C++20**.  The ``sysroot``
package will also need to be installed to avoid issues with missing versioned
symbols with ``GLIBCXX`` when compiling FBGEMM_CPU:

.. code:: sh

  # Fix GCC to 10.4.0, to keep compatibility with older versions of GLIBCXX
  gcc_version=15.0.7

  conda install -n ${env_name} -c conda-forge -y gxx_linux-64=${gcc_version} sysroot_linux-64=2.17

While newer versions of GCC can be used, binaries compiled under newer versions
of GCC will not be compatible with older systems such as Ubuntu 20.04 or CentOS
Stream 8, because the compiled library will reference symbols from versions of
``GLIBCXX`` that the system’s ``libstdc++.so.6`` will not support. To see what
versions of GLIBC and GLIBCXX the available ``libstdc++.so.6`` supports:

.. code:: sh

  libcxx_path=/path/to/libstdc++.so.6

  # Print supported for GLIBC versions
  objdump -TC "${libcxx_path}" | grep GLIBC_ | sed 's/.*GLIBC_\([.0-9]*\).*/GLIBC_\1/g' | sort -Vu | cat

  # Print supported for GLIBCXX versions
  objdump -TC "${libcxx_path}" | grep GLIBCXX_ | sed 's/.*GLIBCXX_\([.0-9]*\).*/GLIBCXX_\1/g' | sort -Vu | cat

.. _fbgemm-gpu.build.setup.tools.install.compiler.clang:

C/C++ Compiler (Clang)
~~~~~~~~~~~~~~~~~~~~~~

It is possible to build FBGEMM and FBGEMM_GPU (just the CPU and CUDA variants)
using Clang as the host compiler.  To do so, install a version of the Clang
toolchain **that supports C++17**:

.. code:: sh

  # Use a recent version of LLVM+Clang
  llvm_version=15.0.7

  # NOTE: libcxx from conda-forge is outdated for linux-aarch64, so we cannot
  # explicitly specify the version number
  conda install -n ${env_name} -c conda-forge -y \
      clangxx=${llvm_version} \
      libcxx \
      llvm-openmp=${llvm_version} \
      compiler-rt=${llvm_version}

  # Append $CONDA_PREFIX/lib to $LD_LIBRARY_PATH in the Conda environment
  ld_library_path=$(conda run -n ${env_name} printenv LD_LIBRARY_PATH)
  conda_prefix=$(conda run -n ${env_name} printenv CONDA_PREFIX)
  conda env config vars set -n ${env_name} LD_LIBRARY_PATH="${ld_library_path}:${conda_prefix}/lib"

  # Set NVCC_PREPEND_FLAGS in the Conda environment for Clang to work correctly as the host compiler
  conda env config vars set -n ${env_name} NVCC_PREPEND_FLAGS=\"-std=c++17 -Xcompiler -std=c++17 -Xcompiler -stdlib=libstdc++ -ccbin ${clangxx_path} -allow-unsupported-compiler\"

**Note** that for CUDA code compilation, even though ``nvcc`` supports Clang as
the host compiler, only ``libstd++`` (GCC's implementation of the C++ standard
library) is supported for any host compiler being used by ``nvcc``.

This means that GCC is a required dependency for CUDA variant of FBGEMM_GPU,
regardless of whether it is built with Clang or not.  In this scenario, it is
recommended to first install the GCC toolchain before installing the Clang
toolchain in this scenario; see
:ref:`fbgemm-gpu.build.setup.tools.install.compiler.gcc` for instructions.

Compiler Symlinks
~~~~~~~~~~~~~~~~~

After installing the compiler toolchains, symlink the C and C++ compilers to the
binpath (override existing symlinks as needed).  In a Conda environment, the
binpath is located at ``$CONDA_PREFIX/bin``:

.. code:: sh

  conda_prefix=$(conda run -n ${env_name} printenv CONDA_PREFIX)

  ln -sf "${path_to_either_gcc_or_clang}" "$(conda_prefix)/bin/cc"
  ln -sf "${path_to_either_gcc_or_clang}" "$(conda_prefix)/bin/c++"

These symlinks will be used later in the FBGEMM_GPU build configuration stage.

Other Build Tools
~~~~~~~~~~~~~~~~~

Install the other necessary build tools such as ``ninja``, ``cmake``, etc:

.. code:: sh

  conda install -n ${env_name} -y \
      click \
      cmake \
      hypothesis \
      jinja2 \
      make \
      ncurses \
      ninja \
      numpy \
      scikit-build \
      wheel

.. _fbgemm-gpu.build.setup.pytorch.install:

Install PyTorch
---------------

The official `PyTorch
Homepage <https://pytorch.org/get-started/locally/>`__ contains the most
authoritative instructions on how to install PyTorch, either through Conda or
through PIP.

Installation Through Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

  # Install the latest nightly
  conda install -n ${env_name} -y pytorch -c pytorch-nightly

  # Install the latest test (RC)
  conda install -n ${env_name} -y pytorch -c pytorch-test

  # Install a specific version
  conda install -n ${env_name} -y pytorch==2.0.0 -c pytorch

Note that installing PyTorch through Conda without specifying a version (as in
the case of nightly builds) may not always be reliable. For example, it is known
that the GPU builds for PyTorch nightlies arrive in Conda 2 hours later than the
CPU-only builds. As such, a Conda installation of ``pytorch-nightly`` in that
time window will silently fall back to installing the CPU-only variant.

Also note that, because both the GPU and CPU-only versions of PyTorch are placed
into the same artifact bucket, the PyTorch variant that is selected during
installation will depend on whether or not CUDA is installed on the system.
Thus for GPU builds, it is important to install CUDA / ROCm first prior to
PyTorch.

Installation Through PyTorch PIP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing PyTorch through PyTorch PIP is recommended over Conda as it is much
more deterministic and thus reliable:

.. code:: sh

  # Install the latest nightly, CPU variant
  conda run -n ${env_name} pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu/

  # Install the latest test (RC), CUDA variant
  conda run -n ${env_name} pip install --pre torch --index-url https://download.pytorch.org/whl/test/cu121/

  # Install a specific version, CUDA variant
  conda run -n ${env_name} pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121/

  # Install the latest nightly, ROCm variant
  conda run -n ${env_name} pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm5.6/

For installing the ROCm variant of PyTorch, PyTorch PIP is the only available
channel as of time of writing.

Post-Install Checks
~~~~~~~~~~~~~~~~~~~

Verify the PyTorch installation (both version and variant) with an ``import`` test:

.. code:: sh

  # Ensure that the package loads properly
  conda run -n ${env_name} python -c "import torch.distributed"

  # Verify the version and variant of the installation
  conda run -n ${env_name} python -c "import torch; print(torch.__version__)"

For the CUDA variant of PyTorch, verify that at the minimum ``cuda_cmake_macros.h`` is found:

.. code:: sh

  conda_prefix=$(conda run -n ${env_name} printenv CONDA_PREFIX)
  find "${conda_prefix}" -name cuda_cmake_macros.h


Build the FBGEMM_GPU Package
----------------------------

Preparing the Build
~~~~~~~~~~~~~~~~~~~

Clone the repo along with its submodules, and install the
``requirements.txt``:

.. code:: sh

  # !! Run inside the Conda environment !!

  # Select a version tag
  FBGEMM_VERSION=v0.6.0

  # Clone the repo along with its submodules
  git clone --recursive -b ${FBGEMM_VERSION} https://github.com/pytorch/FBGEMM.git fbgemm_${FBGEMM_VERSION}

  # Install additional required packages for building and testing
  cd fbgemm_${FBGEMM_VERSION}/fbgemm_gpu
  pip install requirements.txt

The Build Process
~~~~~~~~~~~~~~~~~

The FBGEMM_GPU build process uses a scikit-build CMake-based build flow,
and it keeps state across install runs. As such, builds can become stale
and can cause problems when re-runs are attempted after a build failure
due to missing dependencies, etc. To address this, simply clear the
build cache:

.. code:: sh

  # !! Run in fbgemm_gpu/ directory inside the Conda environment !!

  python setup.py clean

Set Wheel Build Variables
~~~~~~~~~~~~~~~~~~~~~~~~~

When building out the Python wheel, the package name, Python version tag, and
Python platform name must first be properly set:

.. code:: sh

  # Set the package name depending on the build variant
  export package_name=fbgemm_gpu_{cpu, cuda, rocm}

  # Set the Python version tag.  It should follow the convention `py<major><minor>`,
  # e.g. Python 3.12 -> py312
  export python_tag=py312

  # Determine the processor architecture
  export ARCH=$(uname -m)

  # Set the Python platform name for the Linux case
  export python_plat_name="manylinux2014_${ARCH}"
  # For the macOS (x86_64) case
  export python_plat_name="macosx_10_9_${ARCH}"
  # For the macOS (arm64) case
  export python_plat_name="macosx_11_0_${ARCH}"
  # For the Windows case
  export python_plat_name="win_${ARCH}"

.. _fbgemm-gpu.build.process.cpu:

CPU-Only Build
~~~~~~~~~~~~~~

For CPU-only builds, the ``--cpu_only`` flag needs to be specified.

.. code:: sh

  # !! Run in fbgemm_gpu/ directory inside the Conda environment !!

  # Build the wheel artifact only
  python setup.py bdist_wheel \
      --package_variant=cpu \
      --package_name="${package_name}" \
      --python-tag="${python_tag}" \
      --plat-name="${python_plat_name}"

  # Build and install the library into the Conda environment (GCC)
  python setup.py install \
      --package_variant=cpu

To build using Clang + ``libstdc++`` instead of GCC, simply append the
``--cxxprefix`` flag:

.. code:: sh

  # !! Run in fbgemm_gpu/ directory inside the Conda environment !!

  # Build the wheel artifact only
  python setup.py bdist_wheel \
      --package_variant=cpu \
      --package_name="${package_name}" \
      --python-tag="${python_tag}" \
      --plat-name="${python_plat_name}" \
      --cxxprefix=$CONDA_PREFIX

  # Build and install the library into the Conda environment (Clang)
  python setup.py install \
      --package_variant=cpu
      --cxxprefix=$CONDA_PREFIX

Note that this presumes the Clang toolchain is properly installed along with the
GCC toolchain, and is made available as ``${cxxprefix}/bin/cc`` and
``${cxxprefix}/bin/c++``.

.. _fbgemm-gpu.build.process.cuda:

CUDA Build
~~~~~~~~~~

Building FBGEMM_GPU for CUDA requires both NVML and cuDNN to be installed and
made available to the build through environment variables.  The presence of a
CUDA device, however, is not required for building the package.

Similar to CPU-only builds, building with Clang + ``libstdc++`` can be enabled
by appending ``--cxxprefix=$CONDA_PREFIX`` to the build command, presuming the
toolchains have been properly installed.

.. code:: sh

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

  # Build for SM70/80 (V100/A100 GPU); update as needed
  # If not specified, only the CUDA architecture supported by current system will be targeted
  # If not specified and no CUDA device is present either, all CUDA architectures will be targeted
  cuda_arch_list=7.0;8.0

  # Unset TORCH_CUDA_ARCH_LIST if it exists, bc it takes precedence over
  # -DTORCH_CUDA_ARCH_LIST during the invocation of setup.py
  unset TORCH_CUDA_ARCH_LIST

  # Build the wheel artifact only
  python setup.py bdist_wheel \
      --package_variant=cuda \
      --package_name="${package_name}" \
      --python-tag="${python_tag}" \
      --plat-name="${python_plat_name}" \
      --nvml_lib_path=${NVML_LIB_PATH} \
      -DTORCH_CUDA_ARCH_LIST="${cuda_arch_list}"

  # Build and install the library into the Conda environment
  python setup.py install \
      --package_variant=cuda \
      --nvml_lib_path=${NVML_LIB_PATH} \
      -DTORCH_CUDA_ARCH_LIST="${cuda_arch_list}"

.. _fbgemm-gpu.build.process.rocm:

ROCm Build
~~~~~~~~~~

For ROCm builds, ``ROCM_PATH`` and ``PYTORCH_ROCM_ARCH`` need to be specified.
The presence of a ROCm device, however, is not required for building
the package.

Similar to CPU-only and CUDA builds, building with Clang + ``libstdc++`` can be
enabled by appending ``--cxxprefix=$CONDA_PREFIX`` to the build command,
presuming the toolchains have been properly installed.

.. code:: sh

  # !! Run in fbgemm_gpu/ directory inside the Conda environment !!

  export ROCM_PATH=/path/to/rocm

  # Build for the target architecture of the ROCm device installed on the machine (e.g. 'gfx906;gfx908;gfx90a')
  # See https://wiki.gentoo.org/wiki/ROCm for list
  export PYTORCH_ROCM_ARCH=$(${ROCM_PATH}/bin/rocminfo | grep -o -m 1 'gfx.*')

  # Build the wheel artifact only
  python setup.py bdist_wheel \
      --package_variant=rocm \
      --package_name="${package_name}" \
      --python-tag="${python_tag}" \
      --plat-name="${python_plat_name}" \
      -DHIP_ROOT_DIR="${ROCM_PATH}" \
      -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" \
      -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA"

  # Build and install the library into the Conda environment
  python setup.py install \
      --package_variant=rocm \
      -DHIP_ROOT_DIR="${ROCM_PATH}" \
      -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" \
      -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA"

Post-Build Checks (For Developers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the build completes, it is useful to run some checks that verify
that the build is actually correct.

Undefined Symbols Check
^^^^^^^^^^^^^^^^^^^^^^^

Because FBGEMM_GPU contains a lot of Jinja and C++ template instantiations, it
is important to make sure that there are no undefined symbols that are
accidentally generated over the course of development:

.. code:: sh

  # !! Run in fbgemm_gpu/ directory inside the Conda environment !!

  # Locate the built .SO file
  fbgemm_gpu_lib_path=$(find . -name fbgemm_gpu_py.so)

  # Check that the undefined symbols don't include fbgemm_gpu-defined functions
  nm -gDCu "${fbgemm_gpu_lib_path}" | sort

GLIBC Version Compatibility Check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also useful to verify that the version numbers of GLIBCXX
referenced as well as the availability of certain function symbols:

.. code:: sh

  # !! Run in fbgemm_gpu/ directory inside the Conda environment !!

  # Locate the built .SO file
  fbgemm_gpu_lib_path=$(find . -name fbgemm_gpu_py.so)

  # Note the versions of GLIBCXX referenced by the .SO
  # The libstdc++.so.6 available on the install target must support these versions
  objdump -TC "${fbgemm_gpu_lib_path}" | grep GLIBCXX | sed 's/.*GLIBCXX_\([.0-9]*\).*/GLIBCXX_\1/g' | sort -Vu | cat

  # Test for the existence of a given function symbol in the .SO
  nm -gDC "${fbgemm_gpu_lib_path}" | grep " fbgemm_gpu::merge_pooled_embeddings("
  nm -gDC "${fbgemm_gpu_lib_path}" | grep " fbgemm_gpu::jagged_2d_to_dense("

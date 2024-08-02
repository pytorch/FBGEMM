Build Instructions
==================

**Note:** The most up-to-date build instructions are embedded in a set of
scripts bundled in the FBGEMM repo under
`setup_env.bash <https://github.com/pytorch/FBGEMM/blob/main/.github/scripts/setup_env.bash>`_.

The general steps for building FBGEMM_GPU are as follows:

#. Set up an isolated build environment.
#. Set up the toolchain.
#. Run the build script.


FBGEMM Requirements
--------------------

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~

Building and running FBGEMM requires a CPU with support for AVX2 instruction set
or higher.

In general, FBGEMM does not have any dependency on Intel MKL. However, for
performance comparisons, some benchmarks use MKL functions. If MKL is found or
the MKL path is provided through the ``INTEL_MKL_DIR`` environment variable, the
benchmarks will be built with MKL and performance numbers will be reported for
MKL functions. Otherwise, this subset of benchmarks will not built.

Software Dependencies
~~~~~~~~~~~~~~~~~~~~~

All three dependencies are provided through the FBGEMM repo's git submodules.
However, if a custom version is desired, they can be set in the build using the
environment variables ``ASMJIT_SRC_DIR``, ``CPUINFO_SRC_DIR``, and
``GOOGLETEST_SOURCE_DIR``.

asmjit
^^^^^^

With inner kernels, FBGEMM takes a "one size doesn't fit all" approach, so the
implementation dynamically generates efficient matrix-shape specific vectorized
code using a third-party library called `asmjit <https://github.com/asmjit/asmjit>`_.

cpuinfo
^^^^^^^

FBGEMM detects CPU instruction set support at runtime using the
`cpuinfo <https://github.com/pytorch/cpuinfo>`_ library provided by the PyTorch
project, and dispatches optimized kernels for the detected instruction set.

GoogleTest
^^^^^^^^^^

`GoogleTest <https://github.com/google/googletest>`_ is required to build and
run FBGEMM's tests. However, GoogleTest is not required if you don't want to run
FBGEMM tests. Tests are built together with the library by default; to turn this
off, simply set ``FBGEMM_BUILD_TESTS=0``.


.. _fbgemm.build.setup.env:

Set Up an Isolated Build Environment
------------------------------------

Follow the instructions for setting up the Conda environment at
:ref:`fbgemm-gpu.build.setup.env`.


Install the Build Tools
-----------------------

C/C++ Compiler
~~~~~~~~~~~~~~

For Linux and macOS platforms, follow the instructions in
:ref:`fbgemm-gpu.build.setup.tools.install.compiler.gcc` to install the GCC
toolchain.  For Clang-based builds, follow the instructions in
:ref:`fbgemm-gpu.build.setup.tools.install.compiler.clang` to install the Clang
toolchain.

For builds on Windows machines, Microsoft Visual Studio 2019 or newer is
recommended.  Follow the installation instructions provided by Microsoft
`here <https://visualstudio.microsoft.com/vs/older-downloads/>`_.

Other Build Tools
~~~~~~~~~~~~~~~~~

Install the other necessary build tools such as ``ninja``, ``cmake``, etc:

.. code:: sh

  conda install -n ${env_name} -y \
      bazel \
      cmake \
      doxygen \
      make \
      ninja \
      openblas

Note that the ``bazel`` package is only necessary for Bazel builds, and the
``ninja`` package is only necessary for Windows builds.


Build the FBGEMM Library
------------------------

Preparing the Build
~~~~~~~~~~~~~~~~~~~

Clone the repo along with its submodules:

.. code:: sh

  # !! Run inside the Conda environment !!

  # Clone the repo and its submodules
  git clone --recurse-submodules https://github.com/pytorch/FBGEMM.git
  cd FBGEMM

Building on Linux and macOS (CMake + GCC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming a Conda environment with all the tools installed, the CMake build
process is straightforward:

.. code:: sh

  # !! Run inside the Conda environment !!

  # Create a build directory
  mkdir build
  cd build

  # Set CMake build arguments
  build_args=(
    -DUSE_SANITIZER=address
    -DFBGEMM_LIBRARY_TYPE=shared
    -DPYTHON_EXECUTABLE=`which python3`

    # OPTIONAL: Set to generate Doxygen documentation
    -DFBGEMM_BUILD_DOCS=ON
  )

  # Set up the build
  cmake ${build_args[@]} ..

  # Build the library
  make -j VERBOSE=1

  # Run all tests
  make test

  # Install the library
  make install

Build Issues with GCC 12+
^^^^^^^^^^^^^^^^^^^^^^^^^

As of time of writing, compilation of FBGEMM on GCC 12+ will fail due to a
`known compiler regression <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593>`__.
To work around the issue, append the following exports prior to running CMake:

.. code:: sh

  # !! Run inside the Conda environment !!

  export CFLAGS+=" -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict"
  export CXXFLAGS+=" -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict"

Please see GitHub issues
`77939 <https://github.com/pytorch/pytorch/issues/77939>`__,
`1094 <https://github.com/pytorch/FBGEMM/issues/1094>`__, and
`1666 <https://github.com/pytorch/FBGEMM/issues/1666>`__ for more details.

Building on Linux and macOS (CMake + Clang)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The steps for building FBGEMM using Clang are exactly the same as that for
building using GCC.  However, extra build arguments need to be added to the
CMake invocation to specify the Clang path, the LLVM-based C++ standard library
(``libc++``), and the LLVM-based OpenMP implementation (``libomp``):

.. code:: sh

  # !! Run inside the Conda environment !!

  # Locate Clang
  cc_path=$(which clang)
  cxx_path=$(which clang++)

  # Append to the CMake build arguments
  build_args+=(
    -DCMAKE_C_COMPILER="${cc_path}"
    -DCMAKE_CXX_COMPILER="${cxx_path}"
    -DCMAKE_C_FLAGS=\"-fopenmp=libomp -stdlib=libc++ -I $CONDA_PREFIX/include\"
    -DCMAKE_CXX_FLAGS=\"-fopenmp=libomp -stdlib=libc++ -I $CONDA_PREFIX/include\"
  )

Building on Linux (Bazel)
~~~~~~~~~~~~~~~~~~~~~~~~~

Likewise, a Bazel build is also very straightforward:

.. code:: sh

  # !! Run inside the Conda environment !!

  # Build the library
  bazel build -s :*

  # Run all tests
  bazel test -s :*

Building on Windows
~~~~~~~~~~~~~~~~~~~

.. code:: powershell

  # Specify the target architecture to bc x64
  call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64

  # Create a build directory
  mkdir %BUILD_DIR%
  cd %BUILD_DIR%

  cmake -G Ninja -DFBGEMM_BUILD_BENCHMARKS=OFF -DFBGEMM_LIBRARY_TYPE=${{ matrix.library-type }} -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="cl.exe" -DCMAKE_CXX_COMPILER="cl.exe" ..
  ninja -v all

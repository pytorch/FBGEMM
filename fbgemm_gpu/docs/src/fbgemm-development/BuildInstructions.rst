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

For Linux and macOS platforms, Install a version of the GCC toolchain
**that supports C++17**. The ``sysroot`` package will also need to be installed
to avoid issues with missing versioned symbols with ``GLIBCXX`` when compiling FBGEMM:

.. code:: sh

  conda install -n "${env_name}" -y gxx_linux-64=10.4.0 sysroot_linux-64=2.17 -c conda-forge

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

For builds on Windows machines, Microsoft Visual Studio 2019 or newer is
recommended.  Follow the installation instructions provided by Microsoft.

Other Build Tools
~~~~~~~~~~~~~~~~~

Install the other necessary build tools such as ``ninja``, ``cmake``, etc:

.. code:: sh

  conda install -n "${env_name}" -y \
      bazel \
      cmake \
      make \
      ninja \
      openblas-dev

Note that the `bazel` package is only necessary for Bazel builds, and the
`ninja` package is only necessary for Windows builds.


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

Building on Linux and macOS (CMake)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming a Conda environment with all the tools installed, the CMake build
process is straightforward:

.. code:: sh

  # !! Run inside the Conda environment !!

  # Create a build directory
  mkdir build
  cd build

  # Set up the build
  # To generate Doxygen documentation, add `-DFBGEMM_BUILD_DOCS=ON`
  cmake -DUSE_SANITIZER=address -DFBGEMM_LIBRARY_TYPE=shared -DPYTHON_EXECUTABLE=`which python3` ..

  # Build the library
  make -j VERBOSE=1

  # Run all tests
  make test

  # Install the library
  make install

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

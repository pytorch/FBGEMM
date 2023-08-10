# FBGEMM

[![FBGEMM CI](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_ci.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemm_ci.yml)

FBGEMM (Facebook GEneral Matrix Multiplication) is a low-precision,
high-performance matrix-matrix multiplications and convolution library for
server-side inference.

The library provides efficient low-precision general matrix multiplication for
small batch sizes and support for accuracy-loss minimizing techniques such as
row-wise quantization and outlier-aware quantization. FBGEMM also exploits
fusion opportunities in order to overcome the unique challenges of matrix
multiplication at lower precision with bandwidth-bound operations.

FBGEMM is used as a backend of Caffe2 and PyTorch quantized operators for x86 machines:

  * Caffe2: https://github.com/pytorch/pytorch/tree/master/caffe2/quantization/server
  * PyTorch: https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/quantized/cpu



## Build Instructions

### Build with CMake

The general instructions for building with Cmake are as follows:

```sh
# Clone the repo
git clone --recursive https://github.com/pytorch/FBGEMM.git
cd FBGEMM

# Pull down the submodules
git submodule sync
git submodule update --init --recursive

# Create a build directory
mkdir build
cd build

# Set up the build
cmake -DUSE_SANITIZER=address -DFBGEMM_LIBRARY_TYPE=shared -DPYTHON_EXECUTABLE=/usr/bin/python3 ..

# Run the build
make -j VERBOSE=1

# Run all tests
make test

# Install the package
make install
```

##### Build Issues with GCC 12

As of time of writing, compilation of FBGEMM on GCC 12 will fail due to a
[known compiler regression](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593).
To work around the issue, simply add the following exports prior to running CMake:

```sh
export CFLAGS+=" -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict"
export CXXFLAGS+=" -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict"
```

Please see GitHub issues [77939](https://github.com/pytorch/pytorch/issues/77939),
[1094](https://github.com/pytorch/FBGEMM/issues/1094), and
[1666](https://github.com/pytorch/FBGEMM/issues/1666) for more details.

### Run Examples

The tests in the `test/` directory and benchmarks in the `bench/` directory are
some great examples of using FBGEMM. For instance, the `SpMDMTest` test in
`test/PackedRequantizeAcc16Test.cc` shows how to combine row offset calculations
with packing of A (`PackAWithRowOffset`), how to pack B matrix (`PackBMatrix`)
and construct output pipeline `(sparse_matrix*dense_matrix --> requantization -->
nop)` fused with inner GEMM macro kernel.

### Dependencies

FBGEMM requires gcc 8+ and a CPU with support for AVX2 instruction set or
higher. It has been tested on Mac OS X and Linux.

#### asmjit

With inner kernels, FBGEMM takes a “one size doesn't fit all” approach, so the
implementation dynamically generates efficient matrix-shape specific vectorized
code using a third-party library called [asmjit][1]. **asmjit is required** to
build FBGEMM.

#### cpuinfo

FBGEMM detects CPU instruction set support at runtime using cpuinfo library and
dispatches optimized kernels for the detected instruction set. Therefore,
**cpuinfo is required** to detect CPU type.

#### googletest

googletest is required to build and run FBGEMM's tests. **googletest is not
required** if you don't want to run FBGEMM tests. By default, building of tests
is **on**. Turn it off by setting FBGEMM\_BUILD\_TESTS to off.

You can download [asmjit][1], [cpuinfo][2], [googletest][3] and set
ASMJIT\_SRC\_DIR, CPUINFO\_SRC\_DIR, GOOGLETEST\_SOURCE\_DIR respectively for
cmake to find these libraries. If any of these variables is not set, cmake will
build the git submodules found in the third\_party directory.

FBGEMM, in general, does not have any dependency on Intel MKL. However, for
performance comparison, some benchmarks use MKL functions. If MKL is found or
MKL path is provided with INTEL\_MKL\_DIR benchmarks are built with MKL and
performance numbers are reported for MKL functions as well. However, if MKL is
not found, the benchmarks are not built.


## Documentation

For a high-level overview, design philosophy and brief descriptions of various
parts of FBGEMM please see [our blog post][4].

### What's New?

* [New Features and Recent Improvements](https://github.com/pytorch/FBGEMM/wiki/Recent-feature-additions-and-improvements-in-FBGEMM) (January, 2020)

### API Docs

We have extensively used comments in our source files. The best and up-do-date
documentation is available in the source files.

You can also turn on the option to generate the documentation (using [Doxygen][5]
and [Sphinx][6] by setting the `-DFBGEMM_BUILD_DOCS=ON` flag when invoking CMake.

### Citation

For those looking for the appropriate article to cite regarding FBGEMM, we
recommend citing our [paper](https://arxiv.org/pdf/2101.05615.pdf):

```
@article{fbgemm,
  title={FBGEMM: Enabling High-Performance Low-Precision Deep Learning Inference},
  author={Khudia, Daya and Huang, Jianyu and Basu, Protonu and Deng, Summer and Liu, Haixin and Park, Jongsoo and Smelyanskiy, Mikhail},
  journal={arXiv preprint arXiv:2101.05615},
  year={2021}
}
```


## Join the FBGEMM community

For questions, support, news updates, or feature requests, please feel free to:

* File a ticket in [GitHub Issues](https://github.com/pytorch/FBGEMM/issues)
* Post a discussion in [GitHub Discussions](https://github.com/pytorch/FBGEMM/discussions)
* Reach out to us on the `#fbgemm` channel in [PyTorch Slack](https://bit.ly/ptslack)

For contributions, please see the [`CONTRIBUTING`](../CONTRIBUTING.md) file for
ways to help out.


## License

FBGEMM is BSD licensed, as found in the [`LICENSE`](LICENSE) file.


[1]:https://github.com/asmjit/asmjit
[2]:https://github.com/pytorch/cpuinfo
[3]:https://github.com/google/googletest
[4]:https://code.fb.com/ml-applications/fbgemm
[5]:https://www.doxygen.nl/index.html
[6]:https://www.sphinx-doc.org/en/master/

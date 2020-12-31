# FBGEMM_GPU

FBGEMM_GPU (FBGEMM GPU kernel library) is a collection of
high-performance CUDA GPU operator library for GPU training.

The library provides efficient embedding table lookup, data layout transformation,
and quantization supports.


## Examples

The tests (in test folder) and benchmarks (in bench folder) are some great
examples of using FBGEMM_GPU.

## Build Notes
FBGEMM_GPU uses the standard CMAKE-based build flow.

### Dependencies
FBGEMM_GPU requires nvcc and a Nvidia GPU with
compute capability of 3.5+.

For the CUB build time dependency, if you are using conda, you can continue with
```
conda install -c bottler nvidiacub
```
Otherwise download the CUB library from https://github.com/NVIDIA/cub/releases and unpack it to a folder of your choice. Define the environment variable CUB_DIR before building and point it to the directory that contains CMakeLists.txt for CUB. For example on Linux/Mac,

```
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_DIR=$PWD/cub-1.10.0
```

+ ###### Jinjia
third-party library [Jinja][1]. **Jinja is required** to
build FBGEMM_GPU.

+ ###### googletest
googletest is required to build and run FBGEMM_GPU's tests. **googletest is not
required** if you don't want to run FBGEMM_GPU tests. By default, building of tests
is **on**. Turn it off by setting FBGEMMGPU\_BUILD\_TESTS to off.

You can download [Jinja][1], [googletest][2] and set
GOOGLETEST\_SOURCE\_DIR respectively for
cmake to find these libraries. If any of these variables is not set, cmake will
build the git submodules found in the third\_party directory.

General build instructions are as follows:

```
git clone --recursive https://github.com/pytorch/FBGEMM.git
cd FBGEMM/fbgemm_gpu
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
cd fbgemm_gpu
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUB_DIR=${CUB_DIR}
mkdir build && cd build
cmake ..
make
```

To run the tests after building FBGEMM_GPU (if tests are built), use the following
command:
```
make test
```

## Installing  FBGEMM_GPU
```
make install
python setup.py install
```

## How FBGEMM_GPU works
For a high-level overview, design philosophy and brief descriptions of various
parts of FBGEMM_GPU please see our Wiki.

## Full documentation
We have extensively used comments in our source files. The best and up-do-date
documentation is available in the source files.

## Join the FBGEMM community
See the [`CONTRIBUTING`](../CONTRIBUTING.md) file for how to help out.

## License
FBGEMM is BSD licensed, as found in the [`LICENSE`](../LICENSE) file.


[1]:https://jinja.palletsprojects.com/en/2.11.x/
[2]:https://github.com/google/googletest

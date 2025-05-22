# FBGEMM_GPU: N-Bit Quantized Embedding Demo (NoBag Forward Unweighted)

This demo showcases the usage of the `int_nbit_split_embedding_nobag_codegen_forward_unweighted_cuda`
kernel from the FBGEMM_GPU library. It focuses on:
1.  Setting up GPU memory with randomly initialized N-bit quantized embedding tables (specifically 4-bit in this demo).
2.  Preparing input tensors (indices, offsets, etc.).
3.  Invoking the CUDA kernel wrapper to retrieve embeddings.
4.  Verifying the correctness of the retrieved embeddings against CPU-dequantized values.

## Prerequisites

1.  **CUDA Toolkit:** Ensure you have a compatible CUDA toolkit installed and `nvcc` is in your PATH.
2.  **PyTorch (LibTorch):** This demo requires LibTorch (the C++ distribution of PyTorch).
    *   Download it from the [PyTorch website](https://pytorch.org/get-started/locally/).
    *   Make sure its version is compatible with the FBGEMM_GPU version you are using.
3.  **FBGEMM_GPU:** This demo is intended to be built as part of the FBGEMM_GPU build system or against a pre-built FBGEMM_GPU library where its CMake targets are accessible. The demo links against `fbgemm_gpu_ops` (or a similar target provided by FBGEMM_GPU that includes the necessary CUDA kernels).

## Build Instructions

1.  **Navigate to the demo directory:**
    ```bash
    cd /path/to/fbgemm_gpu/demo/int_nbit_embedding_demo 
    ```
    (Replace `/path/to/fbgemm_gpu` with the actual path to your FBGEMM_GPU repository clone).

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Run CMake:**
    You need to tell CMake where to find your LibTorch installation. Set the `CMAKE_PREFIX_PATH` environment variable or pass it as a CMake variable.

    ```bash
    # Example:
    export CMAKE_PREFIX_PATH=/path/to/libtorch 
    # If FBGEMM_GPU is installed elsewhere and provides CMake config files, add its path too:
    # export CMAKE_PREFIX_PATH="/path/to/fbgemm_gpu_install;/path/to/libtorch"

    cmake .. 
    ```
    *   If FBGEMM_GPU was built using CMake and you are building this demo as a subdirectory (e.g., by adding `add_subdirectory(demo)` to a higher-level `CMakeLists.txt` in FBGEMM_GPU), then FBGEMM_GPU's targets should be automatically available.
    *   If the `fbgemm_gpu_ops` target (or equivalent) is not found, you may need to adjust the `target_link_libraries` line in `demo/int_nbit_embedding_demo/CMakeLists.txt` to match the correct FBGEMM_GPU target name that provides the CUDA kernels.

4.  **Compile the demo:**
    ```bash
    make -j$(nproc)
    ```

## Running the Demo

After successful compilation, an executable named `embedding_demo` will be created in the `build` directory.

```bash
./embedding_demo
```

## Expected Output

The program will print:
*   The configuration being used (bit-width, dimension, etc.).
*   Status messages about data preparation and tensor creation.
*   A message indicating the CUDA kernel is being invoked.
*   Verification results, comparing a few dimensions of the retrieved embeddings against values dequantized on the CPU.
*   A final "SUCCESS" or "FAILURE" message.

Example snippet of verification output:
```
--- Verification ---

Lookup 0 (Original Index: 0):
  Dim 0: Expected 0.0000, Got 0.0000, Diff 0.0000
  Dim 1: Expected 1.0000, Got 1.0000, Diff 0.0000
  Dim 2: Expected 2.0000, Got 2.0000, Diff 0.0000
  Dim 3: Expected 3.0000, Got 3.0000, Diff 0.0000
  Dim 4: Expected 4.0000, Got 4.0000, Diff 0.0000
  ... (middle dimensions match) ...
  Dim 123: Expected 123.0000, Got 123.0000, Diff 0.0000
  Dim 124: Expected 124.0000, Got 124.0000, Diff 0.0000
  Dim 125: Expected 125.0000, Got 125.0000, Diff 0.0000
  Dim 126: Expected 126.0000, Got 126.0000, Diff 0.0000
  Dim 127: Expected 127.0000, Got 127.0000, Diff 0.0000

... (similar output for other lookups) ...

SUCCESS: All retrieved embeddings match expected dequantized values.
```

Note: The exact floating-point values might have minor differences due to the nature of float-to-half conversions used for scales/biases and general floating-point arithmetic. A small tolerance is used for comparisons.

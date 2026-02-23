# cpuinfo Initialization Test

This directory contains tests to verify that the cpuinfo library is properly initialized
and consistent across PyTorch and FBGEMM.

## Background

The cpuinfo library uses **global state** to store CPU feature flags (AVX2, AVX512, etc.).
When multiple copies of cpuinfo are linked into the same process, each copy has its own
independent global state.

This bug was introduced in PyTorch commit `c5aa299b048da269e1165216a1ef3cb06edb413d`
which added `target_link_libraries(torch_python PRIVATE cpuinfo)` to `torch/CMakeLists.txt`,
causing `torch_python` to link its own cpuinfo instance separate from `libtorch`.

### The Fix

The fix (PR [#174927](https://github.com/pytorch/pytorch/pull/174927)) changes:

```cmake
# Before (bad):
target_link_libraries(torch_python PRIVATE cpuinfo)

# After (good):
target_include_directories(torch_python PRIVATE ../third_party/cpuinfo/include)
```

This ensures only the headers are included, and the actual cpuinfo library is linked
only once via `libtorch`.

## What Does the Bug Actually Cause?

### CATASTROPHIC FAILURES, Not Subtle Numerical Differences!

The bug causes **completely wrong results**, not small floating-point precision differences:

1. **FBGEMM's cpuinfo instance contains GARBAGE** (uninitialized memory), not zeros
2. **This garbage can make `cpuinfo_has_x86_avx512f()` return TRUE** even when AVX512 isn't available
3. **FBGEMM JIT-compiles AVX512 kernels** and executes them on a non-AVX512 CPU
4. **Results are COMPLETELY WRONG** (not just small FP differences)

### Example Failure (from `test_l2_norm_pruning_workflow`)

```
Mismatched elements: 256 / 512 (50.0%)
Greatest absolute difference: 0.8999999761581421 at index (0, 128)
Greatest relative difference: inf at index (0, 128)
```

Key observations:
- **50% of elements mismatch** - half the output is wrong!
- **Error at index (0, 128)** - exactly at the boundary between table 1 and table 2 (D=128)
- **Relative difference: inf** - one value is 0, the other is ~0.9
- **Absolute difference ~0.9** - matches the weight magnitudes in the test

This pattern (50% wrong, error at table boundary) indicates one entire embedding table's
output is corrupted because the wrong JIT-compiled kernel (AVX512 using ZMM registers)
is being executed on a CPU that doesn't support it.

### Why Not Just Performance Degradation?

I initially thought the bug would only cause performance degradation (scalar fallback).
However, the actual failure mode is worse:

- Uninitialized `cpuinfo_isa` contains **garbage**, not zeros
- Garbage bits can make AVX512 feature flags return **true**
- FBGEMM's JIT compiler generates **AVX512 code** (using 512-bit ZMM registers)
- This code executes on a **non-AVX512 CPU**
- Result: **Data corruption** or **SIGILL** (illegal instruction crash)

## Test Files

### `cpuinfo_initialization_test.py`

Minimal Python test using only `torch.ops.fbgemm` quantization ops:
- `test_quantize_dequantize_roundtrip`: Verifies output is valid (no NaN/Inf, bounded error)
- `test_quantize_multiple_sizes`: Tests various tensor sizes that might trigger different kernels
- `test_int4_quantize_roundtrip`: Tests INT4 quantization
- `test_quantize_performance_indicator`: Informational benchmark

**How to run:**
```bash
python cpuinfo_initialization_test.py
```

**Expected with bad PyTorch:**
- Tests FAIL with large errors (0.5+)
- Or crash with SIGILL

### `cpuinfo_initialization_test.cpp`

C++ test that directly checks cpuinfo state and feature detection consistency.

**How to compile:**
```bash
g++ -std=c++17 -o cpuinfo_test cpuinfo_initialization_test.cpp \
    -I<cpuinfo>/include -L<cpuinfo>/lib -lcpuinfo -lpthread
```

## Expected Results

### With Good PyTorch (PR #174927 applied)
- All tests PASS
- Quantization errors < 0.5

### With Bad PyTorch (bad commit)
- Tests FAIL with errors like:
  ```
  AssertionError: Quantization error 0.90 is too large!
  This may indicate cpuinfo corruption causing wrong kernel selection.
  ```
- Or crash with SIGILL (illegal instruction)

## References

- Bad commit: `c5aa299b048da269e1165216a1ef3cb06edb413d`
- Fix PR: https://github.com/pytorch/pytorch/pull/174927
- Workplace post: https://fb.workplace.com/groups/4571909969591489/permalink/25978555105166999/

# In-Kernel Broadcast Optimization (IKBO)

High-performance GPU kernels for following operations (at the moment) with In-Kernel Broadcast Optimization
- Linear Compression Embedding (LCE)
- Flash Attention

The Deep Dive Blog: (to add)

## Installation

Install `triton` TLX from: https://github.com/facebookexperimental/triton/tree/main

```bash
git clone -b tlx https://github.com/facebookexperimental/triton.git
cd triton
pip install -e . --no-build-isolation
```

```bash
# Install the source code
git clone https://github.com/pytorch/FBGEMM.git
cd fbgemm/fbgemm_gpu/experimental/ikbo
pip install -e .
```

## Quick Start

### PyTorch Reference
```python
from ikbo.ops.torch_lce import torch_decomposed_lce
output = torch_decomposed_lce(W_cand, W_user, E_cand, E_user, cand_to_user_index)
```

### Triton Kernel
```python
from ikbo.ops.triton_ikbo_lce import triton_ikbo_lce
output = triton_ikbo_lce(W_cand, W_user, E_cand, E_user, cand_to_user_index)
```

### TLX Kernel
```python
from ikbo.ops.tlx_ikbo_lce import tlx_ikbo_lce, create_user_flag
user_flag = create_user_flag(W_user, E_user)
output = tlx_ikbo_lce(W_cand, W_user, E_cand, E_user, cand_to_user_index, user_flag)
```

## Testing
```bash
FAST_TUNE=1 python -m pytest tests/ -v
```

## Benchmarks
```bash
python benchmarks/ikbo_lce_bench.py
```

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
cd FBGEMM/fbgemm_gpu/experimental/ikbo
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

## Accuracy Testing
```bash
FAST_TUNE=1 python -m pytest tests/ -v
```

## Benchmarks
Command to boost GPU power and clock to its max with persistent mode (700W, 1980MHz is for H100 SXM5 version):
```bash
sudo nvidia-smi -i {GPU_ID} -pm 1 && sudo nvidia-smi --power-limit={GPU_max_power} -i 6 && MAX_SM_CLOCK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits -i {GPU_ID}) && sudo nvidia-smi -lgc $MAX_SM_CLOCK -i {GPU_ID}
```
Run benchmark with the dedicate GPU ID and corresponding NUMA node:
```bash
CUDA_VISIBLE_DEVICES={GPU_ID} numactl -m {NUMA_node} -c {NUMA_node} python benchmarks/ikbo_lce_bench.py
CUDA_VISIBLE_DEVICES={GPU_ID} numactl -m {NUMA_node} -c {NUMA_node} python benchmarks/ikbo_fa_bench.py
```

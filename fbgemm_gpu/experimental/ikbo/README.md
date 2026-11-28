# In-Kernel Broadcast Optimization (IKBO)

High-performance GPU kernels for following operations (at the moment) with In-Kernel Broadcast Optimization
- Linear Compression Embedding (LCE)
- Flash Attention

**Blog Post:** [In-Kernel Broadcast Optimization: Co-Designing Kernels for RecSys Inference](https://pytorch.org/blog/in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/)

## Installation

Install `triton` TLX [[1]](#references) from: https://github.com/facebookexperimental/triton/tree/main

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

### TLX Kernels for Blackwell (sm_100 / sm_103)
The Hopper TLX kernels above use WGMMA and do not run on Blackwell. The Blackwell kernels use tcgen05 MMA with
TMEM accumulators and 2-CTA (cta_group::2) tiles:
```python
from ikbo.ops.tlx_ikbo_lce_blackwell import tlx_ikbo_lce_blackwell
output = tlx_ikbo_lce_blackwell(W_cand, W_user, E_cand, E_user, cand_to_user_index)  # N == 256

from ikbo.ops.tlx_ikbo_fa_blackwell import plan_flash_attn_ikbo_blackwell, tlx_flash_attn_ikbo_blackwell
plan = plan_flash_attn_ikbo_blackwell(cand_to_user_index, n_seed, num_heads, max_seq_len)  # once per candidate layout
output = tlx_flash_attn_ikbo_blackwell(query, key, value, cand_to_user_index, n_seed, max_seq_len, plan=plan)
```
The LCE kernel keeps a per-device workspace across calls (no per-call memset); with CUDA graphs, run the largest
shape eagerly once before capturing. The FA plan needs `cand_to_user_index` sorted by user.

GB300 (default 1350 W power limit, benchmark inputs below):

| | Blackwell TLX | Triton IKBO | baseline |
|---|---|---|---|
| LCE, B=1024, ratio 70 | 185-192 us | 400 us | 310 us (torch decomposed, cuBLAS) |
| FA, max_seq_len 512 / 1024 / 4096 / 16384 (TFLOP/s) | 1400 / 1621 / 1662 / 1696 | 501 / 623 / 766 / 801 | 307 / 343 / 377 / 383 (SDPA on pre-broadcast K/V) |

Both kernels run at the board power limit on GB300, so their time tracks energy per call: the LCE kernel uses ~30%
less energy per call than the cuBLAS decomposition, and the FA kernel matches cuDNN's dense attention in energy per
FLOP at long sequences.

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

## References

[1] Y. Guan, H. Yu, et al., "TLX: A Triton Language Extension for Efficient Data Movement," *arXiv preprint arXiv:2605.10905*, 2025. [[paper]](https://arxiv.org/abs/2605.10905)

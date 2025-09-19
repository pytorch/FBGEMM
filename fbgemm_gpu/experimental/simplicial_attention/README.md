# Fast 2-Simplicial Attention

High-performance GPU Kernels of 2-Simplicial Attention

2-Simplicial Attention was proposed in: [Logic and the 2-Simplicial Transformer
](https://arxiv.org/abs/1909.00668)

The idea is further optimized in: [Fast and Simplex: 2-Simplicial Attention in Triton](https://arxiv.org/pdf/2507.02754)

The Deep Dive Blog: [Fast 2-Simplicial Attention: Hardware-Efficient Kernels in TLX](https://pytorch.org/blog/fast-2-simplicial-attention-hardware-efficient-kernels-in-tlx/)

## Installation

Install `triton` TLX from the branch: https://github.com/facebookexperimental/triton/tree/tlx

> NOTE: This TLX branch provides stable functionality but not optimal performance. The high-performance branch suffers from an numeric issue caused by PTX compiler optimizations. The issue should be resolved in future releases.


```bash
git clone -b tlx https://github.com/facebookexperimental/triton.git
cd triton
pip install -e . --no-build-isolation
```

```bash
# Install the source code
git clone https://github.com/pytorch/FBGEMM.git
cd fbgemm/fbgemm_gpu/experimental/simplicial_attention
pip install -e .
```

## Quick Start

### Using Triton Implementation

```python
import torch
from simplicial.ops.triton.fwd import triton_fwd

# Setup tensors
batch_size, seq_len, num_heads, head_dim = 4, 1024, 64, 128
device = torch.cuda.current_device()

Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
K1 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
K2 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
V1 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
V2 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)

output, _ = triton_fwd(Q, K1, K2, V1, V2, w1=32, w2=256)
```

### Using TLX Implementation

```python
from simplicial.ops.tlx.fwd_ws import tlx_fwd_ws

# High-performance TLX implementation
output, _ = tlx_fwd_ws(Q, K1, K2, V1, V2, w1=32, w2=256)
```

### Using PyTorch Reference Implementation

```python
from simplicial.ops.pytorch.two_simplicial_attention import torch_fwd_ref

# Reference implementation
output = torch_fwd_ref(
    Q, K1, K2, V1, V2,
    w1=32, w2=256,
    use_fp32=True,
    disable_kv_bias=True
)
```

## Performance

Run performance benchmarks:

```bash
# Forward pass benchmarks
python benchmarks/bench_fwd.py
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v
```

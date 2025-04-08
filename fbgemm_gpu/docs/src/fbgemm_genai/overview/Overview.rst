FBGEMM GenAI Overview
=====================

High Level Overview
-------------------

FBGEMM FP8 rowwise quantization kernels have been officially adopted in the
[Llama3.1 release](https://fb.workplace.com/groups/221503021668016/permalink/1900301927121442/).
FP8 has been applied across Llama3 models with 8 B, 70 B, and 405 B.
Notably, for the 405 B model, FP8 enables the inference on a single node,
achieving a 2x throughput improvement over the baseline BF16 running on two
nodes with pipeline parallelism. Externally, it has been mentioned in
[Llama3 paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) &
[repo](https://github.com/meta-llama/llama-stack/blob/main/llama_stack/models/llama/quantize_impls.py), [HuggingFace](https://huggingface.co/docs/transformers/main/quantization/fbgemm_fp8), [vLLM](https://blog.vllm.ai/2024/07/23/llama31.html), and [TensorRT-LLM](https://developer.nvidia.com/blog/supercharging-llama-3-1-across-nvidia-platforms/).

FBGEMM GenAI FP8 supports a variety of configurations:

* GEMM Operators: {CUTLASS, CK, Triton} x {BF16, FP8} x {tensor-wise, row-wise, block-wise} x {Nvidia H100, AMD MI300x}.
* High/low Precision Conversion Kernels: (FP32 / BF16 <-> FP8) with scaling options {tensor-wise, row-wise, block-wise} across hardware platforms {Nvidia H100, AMD MI300x} and programming options of {Triton, CUDA/HIP}.

Besides FP8 support, FBGEMM GenAI operators also support:

* Customized AllReduce communications (reduce latency for small message sizes).
* GQA: optimized specifically for decoding cases, as detailed in PyTorch's blog on [INT4 decoding](https://pytorch.org/blog/int4-decoding/).
* KV cache quantizations.
* Rotary Positional Embedding (RoPE).

FP8 Core API Functions
----------------------

.. code:: python

  # Rowwise quantize (channel wise) the weight from BF16 to FP8
  wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
  # Rowwise quantize the activation (token wise) from BF16 to FP8
  xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
      x, num_tokens, activation_scale_ub
  )
  # Rowwise quantize GEMM with FP8 input and BF16 output
  y = torch.ops.fbgemm.f8f8bf16_rowwise(
      xq,
      wq,
      x_scale,
      w_scale,
      use_fast_accum=True,
  )

See :ref:`genai-quantize-ops-stable-api` for more details.

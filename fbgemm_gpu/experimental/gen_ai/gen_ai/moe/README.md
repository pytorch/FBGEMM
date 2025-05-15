# FBGEMM GenAI MoE Support

MetaShuffling MoE kernel support in FBGEMM GenAI kernel library.

# **Overview**

Mixture-of-Experts (MoE) is a popular model architecture for large language models (LLMs). Although it reduces computation in training and inference by activating less parameters per token,  it imposes additional challenges in achieving optimal computation efficiency with high memory and communication pressure, as well as the complexity to handle the dynamism and sparsity nature of the model. Here we introduce a new MoE inference solution, MetaShuffling, which enables us to efficiently deploy Llama 4 models for real scenario inference.

[Technical design blog](https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/).

# **Updates**

- 2025-05-01: Initial release of MetaShuffling MoE PyTorch examples.

- 2025-04-17: Initial release of MetaShuffling MoE GPU kernels.

# FBGEMM GenAI MoE Support

MoE Token Shuffling Kernel support in FBGEMM GenAI Kernel Library.

# **1. Overview**

Mixture-of-Experts (MoE) is a popular model architecture for large language models (LLMs). Although it reduces computation in training and inference by activating less parameters per token,  it imposes additional challenges in achieving optimal computation efficiency with high memory and communication pressure, as well as the complexity to handle the dynamism and sparsity nature of the model. Here we introduce a new MoE inference solution, token shuffling, which enables us to efficiently deploy Llama 4 models for real scenario inference.

More technical design will be coming soon.

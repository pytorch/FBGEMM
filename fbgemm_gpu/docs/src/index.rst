The FBGEMM Project Homepage
===========================

Welcome to the documentation homepage for the FBGEMM Project!

The FBGEMM Project is a repository of highly-optimized kernels used across
deep learning applications.  The codebase is organized into three related
packages: **FBGEMM**, **FBGEMM-GPU**, and **FBGEMM-GenAI**.

FBGEMM
------

**FBGEMM** (Facebook GEneral Matrix Multiplication) is a low-precision,
high-performance matrix-matrix multiplications and convolution library for
server-side inference.  This library is used as a backend of
`PyTorch <https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/quantized/cpu>`__
quantized operators on x86 machines.

See :ref:`fbgemm.main` for more information.

FBGEMM_GPU
----------

**FBGEMM_GPU** (FBGEMM GPU Kernels Library) is a collection of high-performance
PyTorch GPU operator libraries built on top of FBGEMM for training and inference,
with a focus on recommendation systems applications.  This library is built on
top of FBGEMM and provides efficient table batched embedding bag, data layout
transformation, and quantization support.

See :ref:`fbgemm-gpu.main` for more information.

FBGEMM GenAI
------------

**FBGEMM GenAI** (FBGEMM Generative AI Kernels Library) is a collection of PyTorch
GPU operator libraries that are designed for generative AI applications, such as
FP8 row-wise quantization and collective communications.

See :ref:`fbgemm-genai.main` for more information.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: General Info

   general/Contributing
   general/documentation/Overview
   general/ContactUs
   general/License

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM Development

   fbgemm/development/BuildInstructions

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM C++ API

   fbgemm/cpp-api/QuantUtils
   fbgemm/cpp-api/tbe_cpu_autovec

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM_GPU Development

   fbgemm_gpu/development/BuildInstructions
   fbgemm_gpu/development/InstallationInstructions
   fbgemm_gpu/development/TestInstructions
   fbgemm_gpu/development/FeatureGates

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM_GPU Overview

   fbgemm_gpu/overview/jagged-tensor-ops/JaggedTensorOps

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM Stable API

   fbgemm_gpu/stable-api/python_api

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM_GPU C++ API

   fbgemm_gpu/cpp-api/sparse_ops
   fbgemm_gpu/cpp-api/quantize_ops
   fbgemm_gpu/cpp-api/merge_pooled_embeddings
   fbgemm_gpu/cpp-api/split_table_batched_embeddings
   fbgemm_gpu/cpp-api/jagged_tensor_ops
   fbgemm_gpu/cpp-api/memory_utils
   fbgemm_gpu/cpp-api/input_combine
   fbgemm_gpu/cpp-api/layout_transform_ops
   fbgemm_gpu/cpp-api/embedding_ops
   fbgemm_gpu/cpp-api/ssd_embedding_ops
   fbgemm_gpu/cpp-api/experimental_ops
   fbgemm_gpu/cpp-api/feature_gates

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM_GPU Python API

   fbgemm_gpu/python-api/sparse_ops
   fbgemm_gpu/python-api/pooled_embedding_ops
   fbgemm_gpu/python-api/quantize_ops
   fbgemm_gpu/python-api/jagged_tensor_ops
   fbgemm_gpu/python-api/tbe_ops_training
   fbgemm_gpu/python-api/tbe_ops_inference
   fbgemm_gpu/python-api/pooled_embedding_modules
   fbgemm_gpu/python-api/feature_gates

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM GenAI Development

   fbgemm_genai/development/BuildInstructions
   fbgemm_genai/development/InstallationInstructions
   fbgemm_genai/development/TestInstructions

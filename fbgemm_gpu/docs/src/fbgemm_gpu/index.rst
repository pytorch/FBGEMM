.. _fbgemm-gpu.main:

FBGEMM_GPU
==========

**FBGEMM_GPU** (FBGEMM GPU Kernels Library) is a collection of high-performance
PyTorch GPU operator libraries built on top of FBGEMM for training and inference,
with a focus on recommendation systems applications.  This library is built on
top of FBGEMM and provides efficient table batched embedding bag, data layout
transformation, and quantization support.

.. _fbgemm-gpu.toc.development:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM_GPU Development

   development/BuildInstructions
   development/InstallationInstructions
   development/TestInstructions
   development/FeatureGates

.. _fbgemm-gpu.toc.overview:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM_GPU Overview

   overview/jagged-tensor-ops/JaggedTensorOps

.. _fbgemm.toc.api.stable:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM Stable API

   stable-api/python_api

.. _fbgemm-gpu.toc.api.cpp:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM_GPU C++ API

   cpp-api/sparse_ops
   cpp-api/quantize_ops
   cpp-api/merge_pooled_embeddings
   cpp-api/split_table_batched_embeddings
   cpp-api/jagged_tensor_ops
   cpp-api/memory_utils
   cpp-api/input_combine
   cpp-api/layout_transform_ops
   cpp-api/embedding_ops
   cpp-api/ssd_embedding_ops
   cpp-api/experimental_ops
   cpp-api/feature_gates

.. _fbgemm-gpu.toc.api.python:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM_GPU Python API

   python-api/sparse_ops
   python-api/pooled_embedding_ops
   python-api/quantize_ops
   python-api/jagged_tensor_ops
   python-api/tbe_ops_training
   python-api/tbe_ops_inference
   python-api/pooled_embedding_modules
   python-api/feature_gates

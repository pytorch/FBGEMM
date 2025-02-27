FBGEMM and FBGEMM_GPU Documentation Homepage
============================================

Welcome to the documentation page for the **FBGEMM** and **FBGEMM_GPU**
libraries!

**FBGEMM** (Facebook GEneral Matrix Multiplication) is a low-precision,
high-performance matrix-matrix multiplications and convolution library for
server-side inference.  This library is used as a backend of
`PyTorch <https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/quantized/cpu>`__
quantized operators on x86 machines.

**FBGEMM_GPU** (FBGEMM GPU Kernels Library) is a collection of high-performance
PyTorch GPU operator libraries for training and inference.  This library is
built on top of FBGEMM and provides efficient table batched embedding bag, data
layout transformation, and quantization support.

------------

Table of Contents

.. _home.docs.toc.general:

.. toctree::
   :maxdepth: 1
   :caption: General Info

   general/Contributing.rst
   general/documentation/Overview.rst
   general/ContactUs.rst
   general/License.rst

.. _fbgemm.toc.development:

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM Development

   fbgemm-development/BuildInstructions.rst

.. _fbgemm-gpu.toc.development:

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM_GPU Development

   fbgemm_gpu-development/BuildInstructions.rst
   fbgemm_gpu-development/InstallationInstructions.rst
   fbgemm_gpu-development/TestInstructions.rst
   fbgemm_gpu-development/FeatureGates.rst

.. _fbgemm-gpu.toc.overview:

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM_GPU Overview

   fbgemm_gpu-overview/jagged-tensor-ops/JaggedTensorOps.rst

.. _fbgemm.toc.api.stable:

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM Stable API

   fbgemm_gpu-stable-api/python_api.rst

.. _fbgemm.toc.api.cpp:

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM C++ API

   fbgemm-cpp-api/QuantUtils.rst
   fbgemm-cpp-api/tbe_cpu_autovec.rst

.. _fbgemm-gpu.toc.api.cpp:

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM_GPU C++ API

   fbgemm_gpu-cpp-api/sparse_ops.rst
   fbgemm_gpu-cpp-api/quantize_ops.rst
   fbgemm_gpu-cpp-api/merge_pooled_embeddings.rst
   fbgemm_gpu-cpp-api/split_table_batched_embeddings.rst
   fbgemm_gpu-cpp-api/jagged_tensor_ops.rst
   fbgemm_gpu-cpp-api/memory_utils.rst
   fbgemm_gpu-cpp-api/input_combine.rst
   fbgemm_gpu-cpp-api/layout_transform_ops.rst
   fbgemm_gpu-cpp-api/embedding_ops.rst
   fbgemm_gpu-cpp-api/ssd_embedding_ops.rst
   fbgemm_gpu-cpp-api/experimental_ops.rst
   fbgemm_gpu-cpp-api/feature_gates.rst

.. _fbgemm-gpu.toc.api.python.ops:

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM_GPU Python Operators API

   fbgemm_gpu-python-api/sparse_ops.rst
   fbgemm_gpu-python-api/pooled_embedding_ops.rst
   fbgemm_gpu-python-api/quantize_ops.rst
   fbgemm_gpu-python-api/jagged_tensor_ops.rst

.. _fbgemm-gpu.toc.api.python.modules:

.. toctree::
   :maxdepth: 1
   :caption: FBGEMM_GPU Python Modules API

   fbgemm_gpu-python-api/tbe_ops_training.rst
   fbgemm_gpu-python-api/tbe_ops_inference.rst
   fbgemm_gpu-python-api/pooled_embedding_modules.rst
   fbgemm_gpu-python-api/feature_gates.rst

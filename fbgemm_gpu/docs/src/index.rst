.. FBGEMM documentation master file, copied from fbgemm/docs
   on Wed Jun 8 17:19:01 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FBGEMM's documentation!
=======================================

This documentation provides a comprehensive reference of the `fbgemm_gpu`
library.

.. _fbgemm-gpu.docs.toc.general:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM_GPU General Info

   general/BuildInstructions.rst
   general/InstallationInstructions.rst
   general/TestInstructions.rst
   general/DocsInstructions.rst

.. _fbgemm-gpu.docs.toc.overview:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM_GPU Overview

   overview/jagged-tensor-ops/JaggedTensorOps.rst

.. _fbgemm-gpu.docs.toc.api.python:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM_GPU Python API

   python-api/table_batched_embedding_ops.rst
   python-api/jagged_tensor_ops.rst

.. _fbgemm-gpu.docs.toc.api.cpp:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM_GPU C++ API

   cpp-api/sparse_ops.rst
   cpp-api/quantize_ops.rst
   cpp-api/merge_pooled_embeddings.rst
   cpp-api/split_table_batched_embeddings.rst
   cpp-api/jagged_tensor_ops.rst
   cpp-api/memory_utils.rst
   cpp-api/input_combine.rst
   cpp-api/layout_transform_ops.rst
   cpp-api/embedding_ops.rst

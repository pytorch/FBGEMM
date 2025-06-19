.. _fbgemm.main:

FBGEMM
======

**FBGEMM** (Facebook GEneral Matrix Multiplication) is a low-precision,
high-performance matrix-matrix multiplications and convolution library for
server-side inference.  This library is used as a backend of
`PyTorch <https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/quantized/cpu>`__
quantized operators on x86 machines.

.. _fbgemm.toc.development:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM Development

   development/BuildInstructions

.. _fbgemm.toc.api.cpp:

.. toctree::
   :maxdepth: 2
   :caption: FBGEMM C++ API

   cpp-api/QuantUtils
   cpp-api/tbe_cpu_autovec

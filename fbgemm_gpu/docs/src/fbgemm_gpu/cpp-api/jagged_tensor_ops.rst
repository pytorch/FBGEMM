Jagged Tensor Operators
=======================

Jagged Tensor solves the issue when rows in dimension are of
different length. This often occurs in sparse feature inputs
in recommender systems, as well as natural language processing
system batched inputs.

CUDA Operators
--------------

.. doxygengroup:: jagged-tensor-ops-cuda
   :content-only:

CPU Operators
-------------

.. doxygengroup:: jagged-tensor-ops-cpu
   :content-only:

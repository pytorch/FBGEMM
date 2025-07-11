Test Instructions
=================

The tests (in the ``fbgemm_gpu/test/`` directory) and benchmarks (in the
``fbgemm_gpu/bench/`` directory) provide good examples on how to use FBGEMM_GPU
operators.

Set Uup the FBGEMM_GPU Test Environment
---------------------------------------

After an environment is available from building / installing the FBGEMM_GPU
package, additional packages need to be installed for tests to run correctly:

.. code:: sh

  # !! Run inside the Conda environment !!

  # From the /fbgemm_gpu/ directory
  python -m pip install -r requirements.txt

Running FBGEMM_GPU Tests
------------------------

To run the tests after building / installing the FBGEMM_GPU package:

.. code:: sh

  # !! Run inside the Conda environment !!

  # From the /fbgemm_gpu/test/ directory
  cd test

  python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning split_table_batched_embeddings_test.py
  python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning quantize_ops_test.py
  python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning sparse_ops_test.py
  python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning split_embedding_inference_converter_test.py

Testing with the CUDA Variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the FBGEMM_GPU CUDA package, GPUs will be automatically detected and
used for testing. To run the tests and benchmarks on a GPU-capable
machine in CPU-only mode, ``CUDA_VISIBLE_DEVICES=-1`` must be set in the
environment:

.. code:: sh

  # !! Run inside the Conda environment !!

  # Specify the specific CUDA devices to run the tests on
  # Alternatively, set to -1 for running in CPU-only mode (when on a GPU-capable machine)
  export CUDA_VISIBLE_DEVICES=-1

  # Enable for debugging failed kernel executions
  export CUDA_LAUNCH_BLOCKING=1

  # For operators involving NCCL, if the rpath is not set up correctly for
  # libnccl.so.2, LD_LIBRARY_PATH will need to be updated.
  export LD_LIBRARY_PATH="/path/to/nccl/lib:${LD_LIBRARY_PATH}"

  python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning split_table_batched_embeddings_test.py

Testing with the ROCm Variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ROCm machines, testing against a ROCm GPU needs to be enabled with
``FBGEMM_TEST_WITH_ROCM=1`` set in the environment:

.. code:: sh

  # !! Run inside the Conda environment !!

  # From the fbgemm_gpu/test/ directory
  cd test

  export FBGEMM_TEST_WITH_ROCM=1

  # Specify the specific HIP devices to run the tests on
  #
  # NOTE: This is necessary if PyTorch is unable to see the devices that
  # `rocm-smi --showproductname` can see
  export HIP_VISIBLE_DEVICES=0,1,2,3

  # Enable for debugging kernel executions
  export HIP_LAUNCH_BLOCKING=1

  python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning split_table_batched_embeddings_test.py

Running FBGEMM_GPU Benchmarks
-----------------------------

To run the benchmarks:

.. code:: sh

  # !! Run inside the Conda environment !!

  # From the fbgemm_gpu/bench/ directory
  cd bench

  python tbe_training_benchmark.py device

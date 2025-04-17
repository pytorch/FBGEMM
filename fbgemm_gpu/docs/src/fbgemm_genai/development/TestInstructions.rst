Test Instructions
=================

The tests (in the ``fbgemm_gpu/experimental/gen_ai/test/`` directory) and
benchmarks (in the ``fbgemm_gpu/experimental/gen_ai/bench/`` directory) provide
good examples on how to use FBGEMM GenAI operators.

Set Up the FBGEMM GenAI Test Environment
---------------------------------------

After an environment is available from building / installing the FBGEMM GenAI
package, additional packages need to be installed for tests to run correctly:

.. code:: sh

  # !! Run inside the Conda environment !!

  # From the fbgemm_gpu/ directory
  python -m pip install -r requirements_genai.txt

Running FBGEMM GenAI Tests
--------------------------

To run the tests after building / installing the FBGEMM GenAI package:

.. code:: sh

  # !! Run inside the Conda environment !!

  # From the fbgemm_gpu/experimental/gen_ai/test/ directory
  cd test

  python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning quantize/quantize_test.py

Running FBGEMM GenAI Benchmarks
-------------------------------

To run the benchmarks:

.. code:: sh

  # !! Run inside the Conda environment !!

  # From the fbgemm_gpu/experimental/gen_ai/bench/ directory
  cd bench

  python quantize_bench.py

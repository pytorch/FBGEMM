Installation Instructions
=========================

**Note:** The most up-to-date installation instructions are embedded in a set
of scripts bundled in the FBGEMM repo under
`setup_env.bash <https://github.com/pytorch/FBGEMM/blob/main/.github/scripts/setup_env.bash>`_.

The general steps for installing FBGEMM GenAI are as follows:

#. Set up an isolated runtime environment.
#. Set up the toolchain for either a CPU-only, CUDA, or ROCm runtime.
#. Install PyTorch.
#. Install the FBGEMM GenAI package.
#. Run post-installation checks.

Before installing FBGEMM GenAI, please check :ref:`fbgemm.releases.compatibility`
to ensure that prerequisite hardware and software you are using is compatible
with the version of FBGEMM GenAI you plan to install.


Set Up Runtime Environment
--------------------------

Follow the instructions for setting up the runtime environment:

#. :ref:`fbgemm-gpu.install.setup.cuda`
#. :ref:`fbgemm-gpu.install.libraries`
#. :ref:`fbgemm-gpu.build.setup.pytorch.install`
#. :ref:`fbgemm-gpu.install.triton`


Install the FBGEMM GenAI Package
------------------------------

Install through PyTorch PIP
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch PIP is the preferred channel for installing FBGEMM GenAI:

.. code:: sh

  # !! Run inside the Conda environment !!

  # CUDA Nightly
  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126/
  pip install --pre fbgemm-genai --index-url https://download.pytorch.org/whl/nightly/cu126/

  # CUDA Release
  pip install torch --index-url https://download.pytorch.org/whl/cu126/
  pip install fbgemm-genai --index-url https://download.pytorch.org/whl/cu126/

  # Test the installation
  python -c "import torch; import fbgemm_gpu.experimental.gen_ai"

Install through Public PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

  # !! Run inside the Conda environment !!

  # CUDA Nightly
  pip install fbgemm-gpu-nightly-genai

  # CUDA Release
  pip install fbgemm-gpu-genai


Post-Installation Checks
------------------------

After installation, run an import test to ensure that the library is correctly
linked and set up.

.. code:: sh

  # !! Run inside the Conda environment !!

  python -c "import torch; import fbgemm_gpu.experimental.gen_ai; print(torch.ops.fbgemm.quantize_fp8_per_row)"


Please refer to :ref:`fbgemm-gpu.install.post-install-checks` for information
on additional post-install checks.

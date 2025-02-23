Documentation
=============

Both FBGEMM and FBGEMM_GPU provide extensive comments in its source files, which
serve as the most authoritative and up-to-date documentation available for the
two libraries.


General Documentation Guidelines
--------------------------------

When new public API methods are added, they should be accompanied by sufficient
documentation.  Here are some guidelines for documenting FBGEMM and FBGEMM_GPU
code:

* **Code by itself is not documentation!**  Put yourself in the shoes of new
  developers who has to understand what your code does, and make their lives
  easier.

* Documentation should be added for any and all public API methods.

* Don't leave documentation as a separate task.  Instead, write docstrings
  together with the code.

* At a very minimum, add:

  * A description of the method.
  * A description of the parameters and arguments that can be passed to the method.
  * A description of the method's return value.
  * Usage examples, links to other methods, and method invocation limitations.


Specific Documentation Guides
-----------------------------

.. toctree::
   :maxdepth: 1

   Cpp.rst
   Python.rst
   Sphinx.rst


.. _general.docs.build:


Building the Documentation
--------------------------

**Note:** The most up-to-date documentation build instructions are embedded in
a set of scripts bundled in the FBGEMM repo under
`setup_env.bash <https://github.com/pytorch/FBGEMM/blob/main/.github/scripts/setup_env.bash>`_.

The general steps for building the FBGEMM and FBGEMM_GPU documentation are as
follows:

#. Set up an isolated build environment.
#. Build FBGEMM_GPU (CPU variant).
#. Set up the documentation toolchain.
#. Run documentation build scripts.

Set Up Build Environment
~~~~~~~~~~~~~~~~~~~~~~~~

Follow the instructions for setting up the Conda environment at
:ref:`fbgemm-gpu.build.setup.env`.

Build FBGEMM_GPU
~~~~~~~~~~~~~~~~

A build pass of **FBGEMM_GPU** is required for the documentation to be built
correctly.  Follow the instructions in
:ref:`fbgemm-gpu.build.setup.tools.install`, followed by
:ref:`fbgemm-gpu.build.process.cpu`, to build FBGEMM_GPU (CPU variant).

Set Up the Documentation Toolchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

  # !! Run inside the Conda environment !!

  # From the /fbgemm_gpu/ directory
  cd docs

  # Install Sphinx and other Python packages
  pip install -r requirements.txt

  # Install Doxygen and and other tools
  conda install -c conda-forge -y doxygen graphviz make

Build the Documentation
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

  # Generate the C++ documentation, the Python documentation, and assemble
  # together
  make clean doxygen html

After the build completes, view the generated documentation:

.. code:: sh

  sphinx-serve -b build


Linting the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~

The same command used for building can be used for linting, by prepending the
``SPHINX_LINT`` flag:

.. code:: sh

  SPHINX_LINT=1 make clean doxygen html

For technical reasons, running a Sphinx build with linting turned on will cause
the documentation to be assembled incorrectly, which is why linting is invoked
separately from the build.

Occasionally, unresolved references might show up while linting, which have the
following error signature:

.. code:: sh

  /opt/build/repo/fbgemm_gpu/docs/docstring of torch._ops.fbgemm.PyCapsule.jagged_2d_to_dense:1:py:class reference target not found: Tensor

If these errors turn out to be false negatives, they can be silenced by being
added into the ``nitpick.ignore`` file (in the same directory as Sphinx
``conf.py``):

.. code:: yaml

  # Add in `{domain} {reference}` format, with space in between.
  py:class Tensor


Deployment Preview
~~~~~~~~~~~~~~~~~~

A preview of the FBGEMM and FBGEMM_GPU documentation will be automatically built
and deployed by `Netlify <https://www.netlify.com/>`__ when pull requests are
made.  When the build completes, the deployment preview can be found at:

.. code:: sh

  https://deploy-preview-{PR NUMBER}--pytorch-fbgemm-docs.netlify.app/

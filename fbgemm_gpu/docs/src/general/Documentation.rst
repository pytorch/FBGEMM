Documentation
=============

Both FBGEMM and FBGEMM_GPU provide extensive comments in its source files, which
serve as the most authoritative and up-to-date documentation available for the
two libraries.


.. _fbgemm-gpu.docs.build:

Building the API Documentation
------------------------------

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

  # Install Sphinx and other docs tools
  pip install -r requirements.txt

  # Install Doxygen and Make
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

Deployment Preview
~~~~~~~~~~~~~~~~~~

As a PyTorch project, a preview of the FBGEMM and FBGEMM_GPU documentation will
be automatically built and deployed by `Netlify <https://www.netlify.com/>`__
when pull requests are made.  When the build completes, the deployment preview
can be found at:

.. code:: sh

  https://deploy-preview-<PR NUMBER>>--pytorch-fbgemm-docs.netlify.app/


General Documentation Guidelines
--------------------------------

When new public API methods are added, they should be accompanied by sufficient
documentation.  Here are some guidelines for documenting FBGEMM and FBGEMM_GPU
code:

* Code by itself is not documentation! Put yourself in the shoes of new
  developers who has to understand what your code does, and make their lives
  easier.

* Documentation should be added for any and all public API methods.

* Don't leave docstring-writing as a separate task.

* Write docstrings together with the code.

* At a very minimum, add:

  *  A description of the method.
  *  A description for each argument that can be passed into the method.
  *  A description of the method's return value.

*  Add usage examples, links to other methods, and method invocation limitations.


Adding Documentation to Python Code
-----------------------------------

Documentation for Python is provided through docstrings and generated using
`Sphinx <https://www.sphinx-doc.org/en/master/>`__.  Please reference the
`Google-style Python docstrings
<https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>`__
guide for docstring formatting examples.

Follow these instructions to document, generate, and publish a new Python
docstring:

#.  Add the docstring directly under the name of the target method.  At a very
    minimum, please add descriptions of:

    * The method's functional behavior
    * The arguments, as denoted by the ``Args`` section
    * The return value, as denoted by the ``Returns`` section
    * The exceptions that can be thrown (if applicable), as denoted by the
      ``Raises`` section

    Other sections such as ``Todo``, ``Note``, and ``Example`` should be added
    as needed.

    Here is an example Python docstring:

    .. literalinclude::  ../../../fbgemm_gpu/docs/examples.py
      :language: python
      :start-after: fbgemm-gpu.python.docs.examples.docstring.start
      :end-before: fbgemm-gpu.python.docs.examples.docstring.end

#.  On the Sphinx documentation side, add an ``autofunction`` directive to the
    corresponding ``.rst`` file.  If an ``.rst`` file for the corresponding
    Python source file does not exist, create a new one by the same name as the
    Python source file.  Using the above example:

    .. code:: rst

      .. autofunction:: fbgemm_gpu.docs.examples.example_method

#.  Make sure the ``.rst`` file is included in to the ``toctree`` in
    ``index.rst`` (e.g. :ref:`fbgemm-gpu.toc.api.python`).

#.  Verify the changes by building the docs locally with
    :ref:`fbgemm-gpu.docs.build` or submitting a PR for a Netlify preview.

------------

The Python docstring example above generates the following HTML output:

.. autofunction:: fbgemm_gpu.docs.examples.example_method

------------


Adding Documentation to C++ Code
--------------------------------

Documentation for C++ is provided through
`Javadoc-style comments <https://www.oracle.com/technical-resources/articles/java/javadoc-tool.html>`__
and generated using Sphinx, `Doxygen <https://www.doxygen.nl/>`__, and
`Breathe <https://www.breathe-doc.org/>`__.

Documentation is kept in header files with the ``.h`` extension as well as in
``.cpp``, ``cu``, and ``cuh`` files.  In these files, everything between
``#ifndef DOXYGEN_THIS_WILL_BE_SKIPPED`` and ``#endif`` will be hidden from the
HTML output.  When you add descriptionss to a function, make sure that the
``#ifndef`` and ``#endif`` are configured correctly.

Follow these instructions to document, generate, and publish a new C++
docstring:

#.  API methods are grouped together by group tags for better organization in
    Sphinx.  If a desired method group for the target method is not defined yet,
    define it near the top of the relevant header file with the ``@defgroup``
    command:

    .. literalinclude::  ../../../src/docs/example_code.cpp
      :language: cpp
      :start-after: fbgemm-gpu.cpp.docs.examples.defgroup.start
      :end-before: fbgemm-gpu.cpp.docs.examples.defgroup.end

#.  Add the docstring directly above the target method's declaration.  At a very
    minimum, please add descriptions of:

    * The method's functional behavior
    * The type parameters, as denoted by the ``@tparam`` tag
    * The arguments, as denoted by the ``@param`` tag
    * The return value, as denoted by the ``@return`` tag
    * The exceptions that can be thrown (if applicable), as denoted by the
      ``@throw`` tag

    Other commands such as ``@note``, ``@warning``, and ``@see`` should be added
    as needed.

    Here is an example C++ docstring:

    .. literalinclude::  ../../../src/docs/example_code.cpp
      :language: cpp
      :start-after: fbgemm-gpu.cpp.docs.examples.docstring.start
      :end-before: fbgemm-gpu.cpp.docs.examples.docstring.end

#.  On the Sphinx documentation side, add a ``doxygengroup`` directive to the
    corresponding ``.rst`` file.  If an ``.rst`` file for the corresponding
    header file does not exist, create a new one by the same name as the header
    file.  Using the above example:

    .. code:: rst

      .. doxygengroup:: example-method-group
        :content-only:

#.  Make sure the ``.rst`` file is included in to the ``toctree`` in
    ``index.rst`` (e.g. :ref:`fbgemm-gpu.toc.api.cpp`).

#.  The C++ source header file needs to be in one of the directories listed in
    the ``INPUT`` parameter in ``Doxygen.ini``.  In general, this has already
    been taken care of, but if it's in a directory not listed, be sure to
    append the directory path to the parameter.

#.  Verify the changes by building the docs locally with
    :ref:`fbgemm-gpu.docs.build` or submitting a PR for a Netlify preview.

------------

The Doxygen example above generates the following HTML output:

.. doxygengroup:: example-method-group
  :content-only:

------------


Sphinx Documentation Pointers
-----------------------------

References Other Sections of the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reference other sections in the documentation, an anchor must first be
created above the target section:

.. code:: rst

  .. _fbgemm-gpu.example.reference:

  Example Section Header
  ----------------------

  NOTES:

  #.  The reference anchor must start with an underscore, i.e. ``_``.

  #.  !! There must be an empty line between the anchor and its target !!

The anchor can then be referenced elsewhere in the docs:

.. code:: rst

  Referencing the section :ref:`fbgemm-gpu.example.reference` from
  another page in the docs.

  Referencing the section with
  :ref:`custom text <fbgemm-gpu.example.reference>` from another page
  in the docs.

  Note that the prefix underscore is not needed when referencing the anchor.


Referencing the Source Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``literalinclude`` directive can be used to display the source code inside a
Sphinx document.  To display the full file content:

.. code:: rst

    .. literalinclude::  relative/path/from/this/rst/file/to/the/source.txt


To display only a section of the file, a pair of unique tags must first be added
to the target source file, as comments with the tag string enclosed in brackets.

For Python source files:

.. code:: python

  # [example.tag.start]

  # ... code section that will be referenced ...

  # [example.tag.end]

For C++ source files:

.. code:: cpp

  /// @skipline [example.tag.start]

  /// ... code section that will be referenced ...

  /// @skipline [example.tag.end]

The tags then need to be supplied to the ``literalinclude`` directive:

.. code:: rst

    .. literalinclude::  relative/path/from/this/rst/file/to/the/source.cpp
      :language: cpp
      :start-after: example.tag.start
      :end-before: example.tag.end

See the Sphinx documentation
`here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-literalinclude>`__
for more information.


Adding LaTeX
~~~~~~~~~~~~

Math expressions with LaTeX can be added inline to Sphinx docs using the
``math`` directive:

.. code:: rst

  Example text: :math:`k_{n+1} = n^2 + k_n^2 - k_{n-1}`

The above example will be rendered as: :math:`k_{n+1} = n^2 + k_n^2 - k_{n-1}`.

Math expressinos can also be inserted as a code block:

.. code:: rst

  .. math::

    \int_a^bu \frac{d^2v}{dx^2} \,dx
      = \left.u \frac{dv}{dx} \right|_a^b
      - \int_a^b \frac{du}{dx} \frac{dv}{dx} \,dx

.. math::

  \int_a^bu \frac{d^2v}{dx^2} \,dx
    = \left.u \frac{dv}{dx} \right|_a^b
    - \int_a^b \frac{du}{dx} \frac{dv}{dx} \,dx

See the Sphinx documentation
`here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#math>`__
and `here <https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax>`__
for more information.


Adding Graphs
~~~~~~~~~~~~~

Graphs can be generated in Sphinx using ``graphviz`` directive.  Graph
descriptions can be added inside a block:

.. code:: rst

  .. graphviz::

    digraph example {
      "From" -> "To";
    }

.. graphviz::

  digraph example {
    "From" -> "To";
  }

Alternatively, they can be imported from an external ``.dot`` file:

.. code:: rst

  .. graphviz:: ExampleGraph.dot

.. graphviz:: ExampleGraph.dot

See the
`Sphinx <https://www.sphinx-doc.org/en/master/usage/extensions/graphviz.html>`__
and `Graphviz <https://graphviz.org/documentation/>`__ documentation more
information.

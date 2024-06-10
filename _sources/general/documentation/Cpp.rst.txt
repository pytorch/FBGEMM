.. _general.docs.add.cpp:

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

    .. literalinclude::  ../../../../src/docs/example_code.cpp
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

    .. literalinclude::  ../../../../src/docs/example_code.cpp
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
    :ref:`general.docs.build` or submitting a PR for a Netlify preview.

------------

The Doxygen example above generates the following HTML output:

.. doxygengroup:: example-method-group
  :content-only:

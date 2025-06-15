.. _general.docs.add.python:

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

    .. literalinclude::  ../../../../fbgemm_gpu/docs/examples.py
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
    :ref:`general.docs.build` or submitting a PR for a Netlify preview.

------------

The Python docstring example above generates the following HTML output:

.. autofunction:: fbgemm_gpu.docs.examples.example_method

------------


.. _general.docs.add.autogen:

Adding Documentation to Auto-Generated Python Code
--------------------------------------------------

Many FBGEMM_GPU Python API methods are auto-generated through PyTorch during the
build process, and require docstrings to be attached after the fact.  Follow
these instructions to document auto-generated Python methods:

#.  If needed, create a Python file under ``fbgemm_gpu/fbgemm_gpu/docs`` in the
    repo.

#.  In the Python file, use the provided helper methods in
    ``fbgemm_gpu.docs.common`` to add attach a docstring to the target
    auto-generated method by method name.  Here is an example from the codebase:

    .. literalinclude::  ../../../../fbgemm_gpu/docs/jagged_tensor_ops.py
      :language: python
      :start-after: fbgemm-gpu.autogen.docs.examples.docstring.start
      :end-before: fbgemm-gpu.autogen.docs.examples.docstring.end

#.  If not already present, append the Python file to the imports list in
    ``fbgemm_gpu/fbgemm_gpu/docs/__init__.py``.  This will force the ad-hoc
    documentation to be loaded on ``fbgemm_gpu`` module load.  For example:

    .. code:: rst

      from . import the_new_doc_module


#.  Follow the remaining steps in :ref:`general.docs.add.python` to render the
    docstring in the documentation.

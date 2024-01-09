.. _general.docs.add.cpp:

Sphinx Documentation Pointers
-----------------------------

References Other Sections of the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reference other sections in the documentation, an anchor must first be
created above the target section:

.. code:: rst

  .. _docs.example.reference:

  Example Section Header
  ----------------------

  NOTES:

  #.  The reference anchor must start with an underscore, i.e. ``_``.

  #.  !! There must be an empty line between the anchor and its target !!

The anchor can then be referenced elsewhere in the docs:

.. code:: rst

  Referencing the section :ref:`docs.example.reference` from
  another page in the docs.

  Referencing the section with
  :ref:`custom text <docs.example.reference>` from another page
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

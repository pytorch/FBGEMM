Jagged Tensor Operators
=======================

High Level Overview
-------------------

The purpose of jagged tensor operators is to handle the case where some
dimension of the input data is "jagged," i.e. each consecutive row in a given
dimension may be a different length.  This is similar to the ``NestedTensor``
`implementation <https://github.com/pytorch/pytorch/issues/25032>`__
in PyTorch and the ``RaggedTensor``
`implementation <https://www.tensorflow.org/guide/ragged_tensor>`__ in
Tensorflow.

Two notable examples of this type of input are:

* Sparse feature inputs in recommendation systems

* Batches of tokenized sentences which may be input to natural language
  processing systems.


Jagged Tensor Format
-------------------

Jagged tensors are effectively represented in FBGEMm_GPU as a three-tensor
object.  The three tensors are: **Values**, **MaxLengths**, and **Offsets**.

Values
~~~~~~

``Values`` is defined as a 2D tensor that contains all the element values
in the jagged tensor, i.e. ``Values.numel()`` is the number of elements in the
jagged tensor.  The size of each row in ``Values`` is derived from the greatest
common divisor for the smallest (most-inner) dimension sub-tensor
(excluding tensors of size 0) in the jagged tensor.

Offsets
~~~~~~~

``Offsets`` is a list of tensors, where each tensor ``Offsets[i]`` represents
the partitioning indices of the values of the next tensor in the list,
``Offsets[i + 1]``.

For example, ``Offset[i] = [ 0, 3, 4 ]`` implies that the current
dimension ``i`` is divided into two groups, denoted by index bounds
``[0 , 3)`` and ``[3, 4)``.  For each ``Offsets[i]``, where
``0 <= i < len(Offests) - 1``, ``Offsets[i][0] = 0``, and
``Offsets[i][-1] = Offsets[i+1].length``.

``Offsets[-1]`` refers to the outer dimension index of ``Values`` (row index),
i.e. ``offsets[-1]`` would be the partition index of ``Values`` itself.  As
such, ``Offsets[-1]``, the tensor begins with ``0`` and ends with
``Values.size(0)`` (i.e. the number of rows for ``Values``).

Max Lengths
~~~~~~~~~~~

``MaxLengths`` is a list of integers, where each value ``MaxLengths[i]``
represents the maximum value between corresponding offset values in
``Offsets[i]``:

.. code:: cpp

  MaxLengths[i] = max( Offsets[i][j] - Offsets[i][j-1]  |  0 < j  < len(Offsets[i]) )

The information in ``MaxLengths`` is used for performing the conversion from
jagged tensor to normal (dense) densor where it will be used to determine the
shape of the tensor's dense form.

.. _fbgemm-gpu.overview.ops.jagged.example:

Jagged Tensor Example
~~~~~~~~~~~~~~~~~~~~~

The figure below shows an example jagged tensor that contains three 2D
sub-tensors, with each sub-tensor having a different dimension:

.. image:: JaggedTensorExample.png

In this example, the sizes of the rows in the inner-most dimension of the jagged
tensor are ``8``, ``4``, and ``0``, and so number of elements per row in
``Values`` is set to ``4`` (greatest common divisor).  This means ``Values``
must be of size ``9 x 4`` in order to accomodate all values in the jagged
tensor.

Because the example jagged tensor contains 2D sub-tensors, the ``Offsets`` list
will need to have a length of 2 to create the partitioning indices.
``Offsets[0]`` represents the partition for dimension ``0`` and ``Offsets[1]``
represents the partition for dimension ``1``.

The ``MaxLengths`` values in the example jagged tensor are ``[4 , 2]``.
``MaxLengths[0]`` is derived from ``Offsets[0]`` range ``[4, 0)`` and
``MaxLengths[1]`` is derived from ``Offsets[1]`` range ``[0, 2)`` (or
``[7, 9]``, ``[3,5]``).

Below is a table of the partition indices applied to the ``Values`` tensor to
construct the logical representation of the example jagged tensor:

.. _fbgemm-gpu.overview.ops.jagged.example.table:

.. list-table::
    :header-rows: 1

    * - ``Offsets[0]``
      - ``Offsets[0]`` Range
      - ``Offsets[0]`` Group
      - Corresponding ``Offsets[1]``
      - ``Offsets[1]`` Range
      - ``Values`` Group
      - Corresponding ``Values``
    * - ``[ 0, 4, 6, 8 ]``
      - ``[0, 4)``
      - Group 1
      - ``[ 0, 2, 3, 3, 5 ]``
      - ``[ 0, 2 )``
      - Group 1
      - ``[ [ 1, 2, 3, 4 ], [ 5, 6, 7, 8 ] ]``
    * -
      -
      -
      -
      - ``[ 2, 3 )``
      - Group 2
      - ``[ [ 1, 2, 3, 4 ] ]``
    * -
      -
      -
      -
      - ``[ 3, 3 )``
      - Group 3
      - ``[ ]``
    * -
      -
      -
      -
      - ``[ 3, 5 )``
      - Group 4
      - ``[ [ 1, 2, 3, 4 ], [ 5, 6, 7, 8 ] ]``
    * -
      - ``[4, 6)``
      - Group 2
      - ``[ 5, 6, 7 ]``
      - ``[ 5, 6 )``
      - Group 5
      - ``[ [ 1, 2, 3, 4 ] ]``
    * -
      -
      -
      -
      - ``[ 6, 7 )``
      - Group 6
      - ``[ [ 1, 2, 7, 9 ] ]``
    * -
      - ``[6, 8)``
      - Group 3
      - ``[ 7, 9 ]``
      - ``[ 7, 9 )``
      - Group 7
      - ``[ [ 1, 2, 3, 4 ], [ 8, 8, 9, 6 ] ]``


Jagged Tensor Operations
------------------------

At the current stage, FBGEMM_GPU only supports element-wise addition,
multiplication, and conversion operations for jagged tensors.

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

Jagged Tensor addition and multiplication works similar to the
`Hadamard Product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>`__
and involves only the ``Values`` of the jagged tensor.  For example:

.. math::

    \begin{bmatrix}
    \begin{bmatrix}
        1. & 2. \\
        3. & 4. \\
    \end{bmatrix} \\
    \begin{bmatrix}
        5. & 6. \\
    \end{bmatrix} \\
    \begin{bmatrix}
        7. & 8. \\
        9. & 10. \\
        11. & 12. \\
    \end{bmatrix} \\
    \end{bmatrix}
    \times
    \begin{bmatrix}
    \begin{bmatrix}
        1. & 2. \\
        3. & 4. \\
    \end{bmatrix} \\
    \begin{bmatrix}
        5. & 6. \\
    \end{bmatrix} \\
    \begin{bmatrix}
        7. & 8. \\
        9. & 5. \\
        2. & 3. \\
    \end{bmatrix} \\
    \end{bmatrix}
    \rightarrow
    \begin{bmatrix}
    \begin{bmatrix}
        1. & 4. \\
        9. & 16. \\
    \end{bmatrix} \\
    \begin{bmatrix}
        25. & 36. \\
    \end{bmatrix} \\
    \begin{bmatrix}
        49. & 64. \\
        81. & 50. \\
        22. & 36. \\
    \end{bmatrix} \\
    \end{bmatrix}

As such, arithmetic operations on jagged tensors require the two operand to have
same shape.  In other words, if we have jagged tensors, :math:`A`, :math:`X`,
:math:`B`, and :math:`C`, where :math:`C = AX + B`, then the following
properties hold:

.. code:: cpp

  // MaxLengths are the same
  C.maxlengths == A.maxlengths == X.maxlengths == B.maxlengths

  // Offsets are the same
  C.offsets == A.offsets == X.offsets == B.offsets

  // Values are elementwise equal to the operations applied
  C.values[i][j] == A.values[i][j] * X.values[i][j] + B.values[i][j]

Conversion Operations
~~~~~~~~~~~~~~~~~~~~~

Jagged to Dense
^^^^^^^^^^^^^^^

.. image:: JaggedTensorConversion1.png

Conversions of a jagged tensor :math:`J` to the equivalent dense tensor :math:`D`
starts with an empty dense tensor.  The shape of :math:`D` is based on the
``MaxLengths``, the inner dimension of ``Values``, and the length of
``Offsets[0]``. The number of dimensions in :math:`D` is:

.. code:: cpp

  rank(D) = len(MaxLengths) + 2

For each dimension in :math:`D`, the dimension size is:

.. code:: cpp

  dim(i) = MaxLengths[i-1]  // (0 < i < D.rank-1)

Using the example jagged tensor from
:ref:`fbgemm-gpu.overview.ops.jagged.example`, ``len(MaxLengths) = 2``, so
the equivalent dense tensor's rank (number of dimension) will be ``4``.  The
example jagged tensor two offset tensors, ``Offsets[0]`` and ``Offsets[1]``.
During the conversion process, elements from ``Values`` will be loaded onto the
dense tensor based on the ranges denoted in the partition indices of
``Offsets[0]`` and ``Offsets[1]`` (see the
:ref:`table <fbgemm-gpu.overview.ops.jagged.example.table>` for the mapping
of the groups to corresponding rows in the dense table):

.. image:: JaggedTensorConversion2.png

Some parts of :math:`D` will not have values from :math:`J` loaded into it since
not every partition range denoted in ``Offsets[i]`` has a size equal to
``MaxLengths[i]``. In that case, those parts will be padded with a pad value.
In the above example, the pad value is ``0``.

Dense to Jagged
^^^^^^^^^^^^^^^

For conversons from dense to jagged tensors, values in the dense tensor are
loaded into the jagged tensor's ``Values``.  However, it's possible that the
given dense tensor is not same shape referring to the ``Offsets``.  It could
lead to the case where jagged tensor can not read in corresponding dense location
if dense's related dimension is smaller than expected.  When this happens we
give the padded value to corresponding ``Values`` (see below):

.. image:: JaggedTensorConversion3.png

Combined Arithmetic + Conversion Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some situations, we would like to perform the following operation:

.. code:: cpp

  dense_tensor + jagged_tensor → dense_tensor (or jagged_tensor)

We can break such an operation into two steps:

#.  **Conversion Operation** - convert from jagged → dense or dense → jagged
    depending on the desired format for the target tensor.  After conversion,
    the operand tensors, be it dense or jagged, should have the exact same
    shapes.

#.  **Arithmetic operation** - perform the arithmetic operations as usual for dense
    or jagged tensors.

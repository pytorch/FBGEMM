# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ctypes import c_size_t


# [fbgemm-gpu.python.docs.examples.docstring.start]
def example_method(alignment: c_size_t, param: float) -> int:
    """
    This class is an example of how you can write docstrings.
    You can add multiple lines of those descriptions. Make sure to include
    useful information about your method.

    **Code Example:**

    .. code-block:: cpp

        // Here is a C++ code block
        std::vector<int32_t> foo(const std::vector<int32_t> &lst) {
            std::vector<int32_t> ret;
            for (const auto x : lst) {
                ret.emplace_back(x * 2);
            }
            return ret;
        }

    And here is a verbatim-text diagram example:

    .. code-block:: text

        .------+---------------------------------.-----------------------------
        |            Block A (first)             |       Block B (second)
        +------+------+--------------------------+------+------+---------------
        | Next | Prev |   usable space           | Next | Prev | usable space..
        +------+------+--------------------------+------+--+---+---------------
        ^  |                                     ^         |
        |  '-------------------------------------'         |
        |                                                  |
        '----------- Block B's prev points to Block A -----'

    Todo:
        * This is a TODO item.
        * And a second TODO item.

    Args:
        alignment (c_size_t): Description of the `alignment` value.
        param (float): Description of `param1`.

    Returns:
        Description of the method's return value.

    Raises:
        AttributeError: If there is an error with the attributes.
        ValueError: If `param` is equal to 3.14.

    Example:
        This is how you can use this function

        >>> print("Code blocks are supported")

    Note:
        For more info on reStructuredText docstrings, see
        `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__
        and
        `here <https://peps.python.org/pep-0287/>`__.
    """
    return 42


# [fbgemm-gpu.python.docs.examples.docstring.end]

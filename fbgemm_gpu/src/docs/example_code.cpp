/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstddef>

/// @skipline [fbgemm-gpu.cpp.docs.examples.defgroup.start]
/// @defgroup example-method-group Example Method Group
/// This is a description of the example method group.
/// @skipline [fbgemm-gpu.cpp.docs.examples.defgroup.end]

/// @skipline [fbgemm-gpu.cpp.docs.examples.docstring.start]
/// @ingroup example-method-group
///
/// @brief A very short description of `example_method`.
///
/// Here is a much longer description of `example_method` with code examples:
///
/// **Example:**
/// ```python
/// # Here is a Python code block
/// def foo(lst: list[int]) -> list[int]:
///   return [ x ** 2 for x in lst ]
/// ```
///
/// And here is a verbatim-text diagram example:
///
/// @code{.unparsed}
///   .------+---------------------------------.-----------------------------
///   |            Block A (first)             |       Block B (second)
///
///   +------+------+--------------------------+------+------+---------------
///   | Next | Prev |   usable space           | Next | Prev | usable space..
///   +------+------+--------------------------+------+--+---+---------------
///   ^  |                                     ^         |
///   |  '-------------------------------------'         |
///   |                                                  |
///   '----------- Block B's prev points to Block A -----'
/// @endcode
///
/// @tparam T Description of `T`
/// @tparam Alignment Description of the `Alignment` value
/// @param param1 Description of `param1`
/// @param param2 Description of `param2`
///
/// @return Description of the method's return value.
///
/// @throw std::bad_alloc if allocation failed
/// @throw std::logic_error if a logic error occurs
///
/// @note This is an example note.
///
/// @warning This is an example warning.
///
/// @see For more info on Doxygen docstrings, see
/// <a href="https://www.doxygen.nl/manual/commands.html#cmdlink">here</a>.
template <typename T, std::size_t Alignment>
int32_t example_method(T param1, float param2);
/// @skipline [fbgemm-gpu.cpp.docs.examples.docstring.end]

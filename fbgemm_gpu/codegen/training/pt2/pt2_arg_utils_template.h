/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#pragma once

namespace fbgemm_gpu {

/* This file is code-generated from generate_backward_split.py to enumerate index of
 * auxiliary arguments in the list. The enum is generated from the dict `aux_args`,
 * which is a single source that controls and maintains the argument position.
*/

{% for name in aux_names %}
enum ArgIndex_{{ name }} {
  {%- for var in aux_args[name] %}
  IDX_{{ var | upper }} = {{ loop.index - 1 }},
  {%- endfor %}
  {{ name | upper }}_SIZE = {{ name | length }}
};

{% endfor %}

} // namespace fbgemm_gpu

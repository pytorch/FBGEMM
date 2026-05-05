#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Group benchmark report rows by their C++ kernel base name.

Kineto / Chrome trace kernel names look like
``foo_kernel_warp_per_row_1<Half, float, 32, true>``. The "base" is
everything before the ``<...>`` template argument list; the "template
args" are the contents of the angle brackets.

Report design: one table per base name. When different configs of the
same op launch different kernels (e.g., gate-on vs gate-off dispatch),
each base kernel gets its own table, and configs that did not dispatch
to a given kernel appear as ``count=0`` rows.
"""

from __future__ import annotations

import re
from typing import Any

_TEMPLATE_RE = re.compile(r"^(?P<base>[^<]+?)\s*(?:<(?P<args>.*)>)?\s*$")


def base_name_of(kernel_name: str) -> str:
    """Return the kernel base name (everything before ``<``)."""
    match = _TEMPLATE_RE.match(kernel_name)
    if match is None:
        return kernel_name
    return match.group("base").strip()


def template_args_of(kernel_name: str) -> str:
    """Return the kernel template arguments (inside the outermost
    ``<...>``), or the empty string when there are none."""
    match = _TEMPLATE_RE.match(kernel_name)
    if match is None:
        return ""
    args = match.group("args")
    return args.strip() if args else ""


def group_by_base(
    rows: list[dict[str, Any]],
    base_field: str = "kernel_base",
    fallback_field: str = "kernel_name",
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group rows by ``kernel_base``.

    Returns a list of ``(base_name, rows)`` pairs in alphabetical order
    by base name (the stability-contract-mandated table order).
    Each group preserves the relative order of the input rows.

    If a row lacks ``kernel_base``, the value is derived from
    ``kernel_name`` via :func:`base_name_of`.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        base = row.get(base_field)
        if not base:
            base = base_name_of(row.get(fallback_field, "") or "")
        if base not in buckets:
            buckets[base] = []
            order.append(base)
        buckets[base].append(row)

    return [(name, buckets[name]) for name in sorted(order)]

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Natural / numeric-aware sorting for benchmark report rows.

The same benchmark run on two different machines must produce the same
row order so that a merger can align rows by key. Numeric columns are
compared numerically (so ``1024 < 4096``, not ``"1024" > "4096"``
lexicographically) and mixed-content columns are split into
numeric / non-numeric runs and compared piecewise.
"""

from __future__ import annotations

import re
from typing import Any

_NUMERIC_CHUNK = re.compile(r"(\d+(?:\.\d+)?)")


def natural_sort_key(value: Any) -> tuple:
    """Return a hashable sort key with natural numeric ordering.

    ``None`` sorts first. Numbers sort numerically. Strings sort by
    their ``(numeric-or-casefolded-text)`` runs. Anything else falls
    through to ``repr()`` on the lowest type bucket.
    """
    if value is None:
        return (0,)

    if isinstance(value, bool):
        # bool before int because ``isinstance(True, int) is True``.
        return (1, int(value))

    if isinstance(value, (int, float)):
        return (2, float(value))

    if isinstance(value, str):
        # Fast path: pure number string.
        stripped = value.strip()
        try:
            return (2, float(stripped))
        except ValueError:
            pass
        chunks: list[tuple[int, Any]] = []
        for part in _NUMERIC_CHUNK.split(value):
            if part == "":
                continue
            if _NUMERIC_CHUNK.fullmatch(part):
                chunks.append((0, float(part)))
            else:
                chunks.append((1, part.casefold()))
        return (3, tuple(chunks))

    return (4, repr(value))


def sort_rows(
    rows: list[dict[str, Any]],
    config_columns: list[str],
    extra_key_fields: tuple[str, ...] = ("kernel_base", "kernel_name"),
) -> list[dict[str, Any]]:
    """Return ``rows`` sorted by ``(config[c] for c in config_columns)``
    then by each field in ``extra_key_fields``.

    ``rows`` must carry config values in a nested ``config`` dict (e.g.
    ``{"config": {"batch_size": 1024}, "kernel_base": "...", ...}``). The
    ``config_columns`` list is the canonical ordering from
    ``regression_stats.json``; entries may be passed as either ``"batch_size"``
    or ``"config.batch_size"`` — the prefix is stripped transparently.
    """
    stripped_cols = [
        c[len("config.") :] if c.startswith("config.") else c for c in config_columns
    ]

    def key_for(row: dict[str, Any]) -> tuple:
        cfg = row.get("config", {}) or {}
        parts = [natural_sort_key(cfg.get(c)) for c in stripped_cols]
        for field in extra_key_fields:
            parts.append(natural_sort_key(row.get(field)))
        return tuple(parts)

    return sorted(rows, key=key_for)

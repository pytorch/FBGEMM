#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Generic trace parsing utilities."""

from __future__ import annotations


def detect_phase(filename: str) -> str:
    """Determine benchmark phase from a trace filename.

    Returns ``"fwd"``, ``"fwd_bwd"``, ``"bwd"``, or ``"unknown"``.

    The check order matters: ``_benchphase_fwd_bwd_`` and ``_fwd_bwd_`` must be
    tested before ``_fwd_`` to avoid misclassifying combined traces as
    forward-only.
    """
    if "_benchphase_fwd_bwd_" in filename or "_fwd_bwd_" in filename:
        return "fwd_bwd"
    if "_benchphase_fwd_" in filename or "_fwd_" in filename:
        return "fwd"
    if "_bwd_" in filename or "bwd" in filename.lower():
        return "bwd"
    return "unknown"

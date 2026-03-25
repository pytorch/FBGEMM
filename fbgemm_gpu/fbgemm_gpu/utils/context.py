#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from contextlib import contextmanager
from typing import Generator

_UNSET = object()


@contextmanager
def updated_env(
    overrides: dict[str, str],
) -> Generator[None, None, None]:
    """
    Context manager that temporarily sets environment variables and restores
    them on exit.  Variables that did not exist before are removed; variables
    that had a previous value are restored.

    Example::

        with updated_env({"FBGEMM_NO_JK": "1", "MY_FLAG": "true"}):
            # env vars are active here
            ...
        # original env is restored here
    """
    saved: dict[str, object] = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key, _UNSET)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, prev in saved.items():
            if prev is _UNSET:
                os.environ.pop(key, None)
            else:
                # pyre-ignore[6]: `prev` is narrowed to `str` by the _UNSET check
                os.environ[key] = prev

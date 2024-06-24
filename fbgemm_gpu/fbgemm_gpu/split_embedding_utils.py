# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings

from fbgemm_gpu.tbe.utils import (  # noqa: F401
    b_indices,  # noqa: F401
    fake_quantize_embs,  # noqa: F401
    generate_requests,  # noqa: F401
    get_device,  # noqa: F401
    get_table_batched_offsets_from_dense,  # noqa: F401
    quantize_embs,  # noqa: F401
    round_up,  # noqa: F401
    TBERequest,  # noqa: F401
    to_device,  # noqa: F401
)

warnings.warn(  # noqa: B028
    f"""\033[93m
    The Python module {__name__} is now DEPRECATED and will be removed in the
    future.  Users should import fbgemm_gpu.tbe.utils into their scripts instead.
    \033[0m""",
    DeprecationWarning,
)

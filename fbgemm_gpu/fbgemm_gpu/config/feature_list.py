# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from enum import auto, Enum

import torch

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:config_cpp")
except Exception:
    import fbgemm_gpu  # noqa F401


# Note: ENUM name must match EXACTLY with the JK knob name in the UI
class FeatureGateName(Enum):
    """
    FBGEMM_GPU feature gates enum (Python).

    **Code Example:**

    .. code-block:: python

        from deeplearning.fbgemm.fbgemm_gpu.config import FeatureGateName

        def foo():
            if FeatureGateName.TBE_V2.is_enabled():

                # Do something if feature is enabled
                ...
            else:
                # Do something different if feature is disabled
                ...

    Note:
        While not required, it is best to mirror the enum values in C++,
        in `fbgemm_gpu/config/feature_gates.h`.

        For fbcode: The ENUM name must match EXACTLY with the JK knob name in the UI
        For OSS: The environment variable will be evaluated as f"FBGEMM_{ENUM}"

    """

    TBE_V2 = auto()

    def is_enabled(self) -> bool:
        return torch.ops.fbgemm.check_feature_gate_key(self.name)

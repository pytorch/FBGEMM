# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from enum import auto, Enum

from .feature_gates import EnvVarFeatureGates

try:
    from deeplearning.fbgemm.fbgemm_gpu.fb.config import JKFeatureGates  # noqa: F401

    open_source: bool = False
except Exception:
    open_source: bool = True


# Note: ENUM name must match EXACTLY with the JK knob name in the UI
class FeatureGateName(Enum):
    """
    Implementation of FBGEMM feature gate names.

    **Code Example:**

    .. code-block:: python

        from deeplearning.fbgemm.fbgemm_gpu.config import FeatureGateName

        def foo():
            if FeatureGateName.ENABLE_TBE.is_enabled():
                # Do something if enabled
                ...
            else:
                # Do something different if disabled
                ...

    Note:
        For fbcode: The ENUM name must match EXACTLY with the JK knob name in the UI
        For OSS: The environment variable will be evaluated as f"FBGEMM_{ENUM}"

    """

    TBE_V2 = auto()

    def is_enabled(self) -> bool:
        if open_source:
            return EnvVarFeatureGates.is_feature_enabled(self)
        else:
            return JKFeatureGates.is_feature_enabled(self)

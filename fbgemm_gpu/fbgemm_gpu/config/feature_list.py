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

        from fbgemm_gpu.config import FeatureGateName

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

    # Enable TBE V2 APIs
    TBE_V2 = auto()

    # Enable Ensemble Rowwise Adagrad (D60189486 stack)
    TBE_ENSEMBLE_ROWWISE_ADAGRAD = auto()

    # Enable ROCm packed bags optimization in TBE inference
    TBE_ROCM_INFERENCE_PACKED_BAGS = auto()

    # Enable HIP-based backward kernels in TBE training
    TBE_ROCM_HIP_BACKWARD_KERNEL = auto()

    # Enable bounds_check_indices_v2
    BOUNDS_CHECK_INDICES_V2 = auto()

    # Disable FP8 quantization vectorization
    DISABLE_FP8_QUANT_VECTORIZATION = auto()

    def is_enabled(self) -> bool:
        return FeatureGate.is_enabled(self)


class FeatureGate:
    """
    FBGEMM_GPU feature gate.

    This class exists because methods defined on enums cannot be invoked when
    the enum is packaged into a model (the mechanism is unclear).

    **Code Example:**

    .. code-block:: python

        from deeplearning.fbgemm.fbgemm_gpu.config import FeatureGate, FeatureGateName

        FeatureGate.is_enabled(FeatureGateName.TBE_V2)
    """

    @classmethod
    def is_enabled(cls, feature: FeatureGateName) -> bool:
        return torch.ops.fbgemm.check_feature_gate_key(feature.name)

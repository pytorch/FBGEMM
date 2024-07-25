# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from typing import Protocol


class IFeatureGateName(Protocol):
    name: str


class EnvVarFeatureGates:
    """
    Open source implementation of feature gates. Relies on reading environment
    variables to determine if a feature is enabled.
    """

    @staticmethod
    def is_feature_enabled(feature_name: IFeatureGateName) -> bool:
        return os.environ.get(f"FBGEMM_{feature_name.name}", "0") == "1"

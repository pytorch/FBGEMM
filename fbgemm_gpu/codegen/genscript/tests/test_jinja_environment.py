#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import unittest
from collections.abc import Callable
from typing import cast
from unittest.mock import patch

from deeplearning.fbgemm.fbgemm_gpu.codegen.genscript import jinja_environment


WaveConfigs = list[tuple[int, int, str]]
_has_dual_wave_support = hasattr(
    jinja_environment, "get_max_vecs_template_configs_union"
)
_get_max_vecs_template_configs_union = cast(
    Callable[..., WaveConfigs],
    getattr(jinja_environment, "get_max_vecs_template_configs_union", None),
)
_get_max_vecs_template_configs_union_forward = cast(
    Callable[..., WaveConfigs],
    getattr(jinja_environment, "get_max_vecs_template_configs_union_forward", None),
)


@unittest.skipUnless(_has_dual_wave_support, "requires D115263090")
class WaveConfigUnionTest(unittest.TestCase):
    def test_wave64_only_union_and_dispatch(self) -> None:
        with patch.dict(
            jinja_environment.env.globals,
            {
                "has_wave32": False,
                "has_wave64": True,
                "items_per_wave64": 256,
            },
        ):
            configs = _get_max_vecs_template_configs_union(
                fixed_max_vecs_per_thread=2,
                use_subwarp_shuffle=True,
                use_vec_blocking=True,
            )
            dispatch = jinja_environment.env.from_string(
                "{{ dispatch_optimal_kernel(items_per_wave64, 2, true) }}"
            ).render()

        self.assertEqual(
            [
                (2, 1, "true"),
                (1, 8, "false"),
                (1, 4, "false"),
                (1, 2, "false"),
                (1, 1, "false"),
                (2, 1, "false"),
            ],
            configs,
        )
        self.assertIn("(MAX_D + 256 - 1) / 256", dispatch)
        self.assertIn("if (MAX_D > 512)", dispatch)
        self.assertNotIn("(MAX_D + 128 - 1) / 128", dispatch)

    def test_mixed_wave_union_preserves_order_and_deduplicates(self) -> None:
        with patch.dict(
            jinja_environment.env.globals,
            {
                "has_wave32": True,
                "has_wave64": True,
                "items_per_warp32": 128,
                "items_per_wave64": 256,
            },
        ):
            configs = _get_max_vecs_template_configs_union(
                fixed_max_vecs_per_thread=2,
                use_subwarp_shuffle=True,
                use_vec_blocking=True,
            )

        self.assertEqual(
            [
                (2, 1, "true"),
                (1, 8, "false"),
                (1, 4, "false"),
                (1, 2, "false"),
                (1, 1, "false"),
                (2, 1, "false"),
            ],
            configs,
        )

    def test_missing_wave_flags_fall_back_to_items_per_warp(self) -> None:
        with patch.dict(
            jinja_environment.env.globals,
            {
                "has_wave32": False,
                "has_wave64": False,
                "items_per_warp": 128,
            },
        ):
            configs = _get_max_vecs_template_configs_union(
                fixed_max_vecs_per_thread=2,
                use_subwarp_shuffle=True,
                use_vec_blocking=True,
            )

        self.assertEqual(
            [
                (2, 1, "true"),
                (1, 4, "false"),
                (1, 2, "false"),
                (1, 1, "false"),
                (2, 1, "false"),
            ],
            configs,
        )

    def test_forward_union_uses_each_waves_vector_count(self) -> None:
        with patch.dict(
            jinja_environment.env.globals,
            {
                "has_wave32": True,
                "has_wave64": True,
                "items_per_warp32": 128,
                "items_per_wave64": 256,
            },
        ):
            configs = _get_max_vecs_template_configs_union_forward(
                max_forward_embedding_dim=256,
                use_subwarp_shuffle=False,
                use_vec_blocking=True,
            )

        self.assertEqual(
            [
                (1, 1, "true"),
                (1, 1, "false"),
                (2, 1, "true"),
                (2, 1, "false"),
            ],
            configs,
        )


@unittest.skipUnless(_has_dual_wave_support, "requires D115263090")
class DispatchCodeGenerationTest(unittest.TestCase):
    def test_non_vec_blocking_dispatch_uses_runtime_warp_size(self) -> None:
        code = jinja_environment.dispatch_non_vec_blocking_kernel(
            items_per_warp=256,
            fixed_max_vecs_per_thread=2,
            use_subwarp_shuffle=True,
        )

        self.assertEqual(
            ["32", "64", "128", "256", "512"],
            re.findall(r"if \(MAX_D <= (\d+)\)", code),
        )
        self.assertEqual(
            ["8", "4", "2", "1", "1"],
            re.findall(r"kSubwarpDivisor =\s+\\\n\s+(\d+);", code),
        )
        self.assertEqual(5, code.count("kWarpSizeHost() / kSubwarpDivisor"))
        self.assertNotIn("constexpr int kThreadGroupSize", code)

    def test_vec_blocking_dispatch_uses_full_runtime_warp(self) -> None:
        code = jinja_environment.dispatch_vec_blocking_kernel(
            items_per_warp=256,
            fixed_max_vecs_per_thread=2,
        )

        self.assertIn("if (MAX_D > 512)", code)
        self.assertIn("(MAX_D + 256 - 1) / 256", code)
        self.assertIn("constexpr int kSubwarpDivisor = 1", code)
        self.assertIn("kThreadGroupSize = kWarpSizeHost()", code)

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import unittest
from unittest.mock import patch

from deeplearning.fbgemm.fbgemm_gpu.codegen.genscript import jinja_environment


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
            configs = jinja_environment.get_max_vecs_template_configs_union(
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

    def test_wave32_only_union_and_dispatch(self) -> None:
        with patch.dict(
            jinja_environment.env.globals,
            {
                "has_wave32": True,
                "has_wave64": False,
                "items_per_wave32": 128,
            },
        ):
            configs = jinja_environment.get_max_vecs_template_configs_union(
                fixed_max_vecs_per_thread=2,
                use_subwarp_shuffle=True,
                use_vec_blocking=True,
            )
            dispatch = jinja_environment.dispatch_optimal_kernel(
                items_per_warp=128,
                fixed_max_vecs_per_thread=2,
                use_subwarp_shuffle=True,
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
        divisors = re.findall(r"kSubwarpDivisor =\s+\\\n\s+(\d+);", dispatch)
        self.assertTrue(divisors)
        self.assertNotIn("8", divisors)
        self.assertIn("if (MAX_D <= 256)", dispatch)
        self.assertIn("if (MAX_D > 256)", dispatch)

    def test_mixed_wave_union_preserves_order_and_deduplicates(self) -> None:
        with patch.dict(
            jinja_environment.env.globals,
            {
                "has_wave32": True,
                "has_wave64": True,
                "items_per_wave32": 128,
                "items_per_wave64": 256,
            },
        ):
            configs = jinja_environment.get_max_vecs_template_configs_union(
                fixed_max_vecs_per_thread=2,
                use_subwarp_shuffle=True,
                use_vec_blocking=True,
            )
            dispatch = jinja_environment.dispatch_non_vec_blocking_kernel(
                items_per_warp=256,
                fixed_max_vecs_per_thread=2,
                use_subwarp_shuffle=True,
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
        self.assertEqual(
            ["4", "2", "1", "1"],
            re.findall(r"kSubwarpDivisor =\s+\\\n\s+(\d+);", dispatch),
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
            configs = jinja_environment.get_max_vecs_template_configs_union(
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
                "items_per_wave32": 128,
                "items_per_wave64": 256,
            },
        ):
            configs = jinja_environment.get_max_vecs_template_configs_union_forward(
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


class DispatchCodeGenerationTest(unittest.TestCase):
    def test_non_vec_blocking_dispatch_uses_runtime_warp_size(self) -> None:
        with patch.dict(
            jinja_environment.env.globals,
            {"has_wave32": False, "has_wave64": True},
        ):
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

    def test_non_vec_and_vec_thresholds_have_no_gap(self) -> None:
        for items_per_warp in (128, 256):
            with self.subTest(items_per_warp=items_per_warp):
                code = jinja_environment.dispatch_optimal_kernel(
                    items_per_warp=items_per_warp,
                    fixed_max_vecs_per_thread=2,
                    use_subwarp_shuffle=True,
                )
                non_vec_thresholds = [
                    int(value) for value in re.findall(r"if \(MAX_D <= (\d+)\)", code)
                ]
                vec_thresholds = [
                    int(value) for value in re.findall(r"if \(MAX_D > (\d+)\)", code)
                ]

                self.assertEqual(1, len(vec_thresholds))
                self.assertEqual(max(non_vec_thresholds), vec_thresholds[0])

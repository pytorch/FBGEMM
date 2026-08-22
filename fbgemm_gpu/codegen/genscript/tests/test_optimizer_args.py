#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.optimizer_args import (
    decl_surfaces,
    DeclSurface,
    make_cpu_kernel_arg,
    make_function_arg,
    make_kernel_arg,
    OptimizerArgsSet,
    OptimizerArgsSetItem as OptimItem,
)
from deeplearning.fbgemm.fbgemm_gpu.codegen.genscript.torch_type_utils import ArgType


def _rowwise_adagrad_spec() -> list[OptimItem]:
    """The argument spec of an optimizer with tensor, placeholder and scalar args."""
    return [
        OptimItem(ArgType.TENSOR, "momentum1"),
        OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
        OptimItem(ArgType.FLOAT, "eps"),
        OptimItem(ArgType.FLOAT, "weight_decay", 0.0),
    ]


class DeclSurfaceTest(unittest.TestCase):
    def test_every_surface_named_by_the_plan_exists(self) -> None:
        self.assertEqual(
            sorted(surface.value for surface in DeclSurface),
            [
                "autograd",
                "cpu_kernel",
                "gpu_kernel",
                "host_function",
                "meta",
                "pt2_cpu",
                "pt2_cuda",
            ],
        )

    def test_strings_normalize_to_surfaces(self) -> None:
        self.assertEqual(
            decl_surfaces(["meta", DeclSurface.PT2_CUDA]),
            frozenset({DeclSurface.META, DeclSurface.PT2_CUDA}),
        )

    def test_an_unknown_surface_is_rejected(self) -> None:
        # A typo must fail generation rather than silently annotate nothing.
        with self.assertRaises(ValueError) as caught:
            decl_surfaces(["gpu_kernels"])
        self.assertIn("gpu_kernels", str(caught.exception))

    def test_an_unknown_surface_on_an_item_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OptimItem(ArgType.FLOAT, "eps", unused_on=["cuda_kernel"])

    def test_the_default_policy_is_empty(self) -> None:
        item = OptimItem(ArgType.FLOAT, "eps")
        self.assertEqual(item.unused_on, frozenset())
        for surface in DeclSurface:
            self.assertFalse(item.is_unused_on(surface))


class DeclarationRendererTest(unittest.TestCase):
    def test_default_policy_renders_the_bare_declaration(self) -> None:
        self.assertEqual(make_function_arg(ArgType.FLOAT, "eps", None), "double eps")
        self.assertEqual(make_kernel_arg(ArgType.FLOAT, "eps", None), "float eps")
        self.assertEqual(
            make_cpu_kernel_arg(ArgType.FLOAT, "eps", 0.0), "float eps = 0.0"
        )
        self.assertEqual(
            make_kernel_arg(ArgType.INT_TENSOR, "momentum1_placements", None),
            "at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> "
            "momentum1_placements",
        )

    def test_a_selected_surface_is_prefixed(self) -> None:
        self.assertEqual(
            make_function_arg(
                ArgType.FLOAT,
                "eps",
                None,
                unused_on=frozenset({DeclSurface.META}),
                surface=DeclSurface.META,
            ),
            "[[maybe_unused]] double eps",
        )

    def test_an_unselected_surface_is_untouched(self) -> None:
        self.assertEqual(
            make_function_arg(
                ArgType.FLOAT,
                "eps",
                None,
                unused_on=frozenset({DeclSurface.META}),
                surface=DeclSurface.HOST_FUNCTION,
            ),
            "double eps",
        )


class OptimizerArgsPolicyTest(unittest.TestCase):
    def test_default_projections_are_unchanged(self) -> None:
        args = OptimizerArgsSet.create(_rowwise_adagrad_spec()).cuda
        for rendered in (
            args.split_function_args
            + args.split_function_args_no_defaults
            + args.split_kernel_args
            + args.split_cpu_kernel_args
            + args.split_function_args_autograd
        ):
            self.assertNotIn("[[maybe_unused]]", rendered)
        for surface in DeclSurface:
            self.assertEqual(
                args.split_function_args_by_surface[surface.value],
                args.split_function_args,
            )
            self.assertEqual(
                args.split_function_args_no_defaults_by_surface[surface.value],
                args.split_function_args_no_defaults,
            )

    def test_marking_one_surface_changes_only_that_declaration(self) -> None:
        spec = _rowwise_adagrad_spec()
        spec[2] = OptimItem(ArgType.FLOAT, "eps", unused_on=[DeclSurface.META])
        args = OptimizerArgsSet.create(spec).cuda
        baseline = OptimizerArgsSet.create(_rowwise_adagrad_spec()).cuda

        meta = args.split_function_args_no_defaults_by_surface["meta"]
        self.assertIn("[[maybe_unused]] double eps", meta)
        self.assertEqual(
            [decl.replace("[[maybe_unused]] ", "") for decl in meta],
            baseline.split_function_args_no_defaults,
        )

        # Schemas, call-site names, and kernel constructors are contract text
        # and must never pick up the annotation.
        self.assertEqual(args.split_function_schemas, baseline.split_function_schemas)
        self.assertEqual(
            args.split_function_arg_names, baseline.split_function_arg_names
        )
        self.assertEqual(
            args.split_kernel_arg_constructors, baseline.split_kernel_arg_constructors
        )
        self.assertEqual(args.split_kernel_arg_names, baseline.split_kernel_arg_names)
        self.assertEqual(
            args.unified_pt2.split_function_schemas,
            baseline.unified_pt2.split_function_schemas,
        )
        # Every other declaration surface is untouched.
        self.assertEqual(
            args.split_function_args_no_defaults_by_surface["host_function"],
            baseline.split_function_args_no_defaults,
        )

    def test_cpu_and_gpu_projections_can_differ(self) -> None:
        spec = _rowwise_adagrad_spec()
        spec[0] = OptimItem(
            ArgType.TENSOR, "momentum1", unused_on=[DeclSurface.CPU_KERNEL]
        )
        argsset = OptimizerArgsSet.create(spec)

        annotated_cpu = [
            decl
            for decl in argsset.cpu.split_cpu_kernel_args
            if "[[maybe_unused]]" in decl
        ]
        self.assertEqual(len(annotated_cpu), 3)  # host, placements, offsets
        self.assertTrue(
            all("momentum1" in decl for decl in annotated_cpu), annotated_cpu
        )
        self.assertEqual(
            [
                decl
                for decl in argsset.cuda.split_kernel_args
                if "[[maybe_unused]]" in decl
            ],
            [],
        )

    def test_expansion_preserves_the_policy_on_every_derived_projection(self) -> None:
        spec = [
            OptimItem(
                ArgType.PLACEHOLDER_TENSOR,
                "row_counter",
                ph_tys=[ArgType.FLOAT_TENSOR],
                is_optional=True,
                unused_on=[DeclSurface.PT2_CUDA],
            ),
            OptimItem(ArgType.TENSOR, "learning_rate_tensor"),
        ]
        argsset = OptimizerArgsSet.create(spec)

        expected = {
            "cpu": [
                "row_counter_host",
                "row_counter_placements",
                "row_counter_offsets",
            ],
            "cuda": [
                "row_counter_dev",
                "row_counter_uvm",
                "row_counter_placements",
                "row_counter_offsets",
            ],
            "any": [
                "row_counter_host",
                "row_counter_dev",
                "row_counter_uvm",
                "row_counter_placements",
                "row_counter_offsets",
            ],
        }
        for projection, names in expected.items():
            rendered = getattr(argsset, projection).split_function_args_by_surface[
                "pt2_cuda"
            ]
            annotated = [decl for decl in rendered if "[[maybe_unused]]" in decl]
            self.assertEqual(len(annotated), len(names), (projection, rendered))
            for name in names:
                self.assertTrue(
                    any(decl.endswith(f" {name}") for decl in annotated),
                    (projection, name, annotated),
                )
            # learning_rate_tensor carries no policy and is never annotated.
            self.assertTrue(
                all("learning_rate_tensor" not in decl for decl in annotated)
            )

    def test_a_generated_common_argument_is_declared_not_inferred_from_its_name(
        self,
    ) -> None:
        # `generate_index_select.py` builds a synthetic `OptimItem(FLOAT,
        # "unused")` purely to satisfy the shared TBE templates. The metadata,
        # not the spelling of the name, is what marks it unread.
        by_name_only = OptimizerArgsSet.create([OptimItem(ArgType.FLOAT, "unused")])
        self.assertEqual(
            by_name_only.cuda.split_function_args_by_surface["gpu_kernel"],
            ["double unused = 0"],
        )

        declared = OptimizerArgsSet.create(
            [
                OptimItem(
                    ArgType.FLOAT,
                    "unused",
                    unused_on=list(DeclSurface),
                )
            ]
        )
        for surface in DeclSurface:
            self.assertEqual(
                declared.cuda.split_function_args_by_surface[surface.value],
                ["[[maybe_unused]] double unused = 0"],
            )
        self.assertEqual(
            declared.cuda.split_function_schemas,
            by_name_only.cuda.split_function_schemas,
        )

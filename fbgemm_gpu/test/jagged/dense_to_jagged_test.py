#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest

import hypothesis.strategies as st
import torch
import torch._dynamo
from hypothesis import given, settings, Verbosity

from .common import additional_decorators, generate_jagged_tensor, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import cpu_and_maybe_gpu, gpu_memory_lt_gb, gpu_unavailable, optests
else:
    from fbgemm_gpu.test.test_utils import (
        cpu_and_maybe_gpu,
        gpu_memory_lt_gb,
        gpu_unavailable,
        optests,
    )


@optests.generate_opcheck_tests(additional_decorators=additional_decorators)
class DenseToJaggedTest(unittest.TestCase):
    def _test_dense_to_jagged(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        precompute_total_L: bool,
    ) -> None:
        # Generate multi-dim jagged tensor
        values_2d, offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)

        # jagged -> dense
        dense = torch.ops.fbgemm.jagged_to_padded_dense(values_2d, offsets, max_lengths)

        # dense -> jagged (op which is being tested)
        if precompute_total_L:
            total_L = values_2d.size(0)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets, total_L
            )
        else:
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets
            )

        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        torch.testing.assert_close(dense, dense2)

        # verify backward
        dense.retain_grad()
        ref_output_values = jagged_values.clone().detach().requires_grad_(True)
        ref_values = dense.clone().detach().requires_grad_(True)
        jagged_values.backward(ref_output_values)
        torch.testing.assert_close(dense.grad, ref_values)

        torch.library.opcheck(
            torch.ops.fbgemm.dense_to_jagged,
            (dense.detach().requires_grad_(True), offsets),
        )

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            precompute_total_L,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.integers(0, 6000),
        inner_dense_size=st.sampled_from([8, 16, 23, 24, 48, 50, 64, 72, 96, 192]),
        dtype=st.just(torch.half),
        device=cpu_and_maybe_gpu(),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_opt(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            precompute_total_L,
        )

    # (8000+1) * 8 (size of the element of LongTensor/int64_t offsets)
    # = ~62.5KB > 48KB default shared memory on V100/A100.
    @unittest.skipIf(*gpu_unavailable)
    @given(
        num_jagged_dim=st.just(1),
        outer_dense_size=st.just(8000),
        inner_dense_size=st.just(16),
        dtype=st.just(torch.half),
        device=cpu_and_maybe_gpu(),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_dense_to_jagged_opt_large_batch(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        precompute_total_L: bool,
    ) -> None:
        self._test_dense_to_jagged(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            precompute_total_L,
        )

    @optests.dontGenerateOpCheckTests("regression test, not an op-shape check")
    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipIf(*gpu_memory_lt_gb(16))
    def test_dense_to_jagged_backward_int32_overflow(self) -> None:
        """Regression: int32 overflow in dense_to_jagged backward at
        B * max_L * D > INT_MAX, even when B * max_L itself fits in int32.

        dense_to_jagged.backward dispatches to fbgemm::jagged_to_padded_dense
        forward, which fills a dense (B, max_L, D) gradient using
        jagged_dense_elementwise_dense_output_kernel_ in common.cuh. Pre-fix,
        that kernel uses PackedTensorAccessor32 for the dense `y` and
        `output` parameters: stride[0] is stored as int32 and the access
        `output[oidx][jidx][iidx]` lowers to `oidx * stride[0]` in int32.
        For shapes where (B - 1) * (max_L * D) > INT_MAX the multiply wraps
        negative and writes go to addresses outside the destination buffer.
        Outer-loop bound and grid sizing are unaffected (B * max_L need not
        overflow), so there is no FBGEMM_LAUNCH_KERNEL grid abort - the
        kernel runs to completion silently writing wrong memory, leaving
        ~25% of high-oidx rows of the dense gradient at the at::empty bit
        pattern. Downstream NaN. (T264042859, observed in production at
        Instagram Reels MTML pos_encoding training.)

        Workload: B=1024, max_L=40960, D=64, bf16. B * max_L = 42M < INT_MAX
        (so no outer-loop / grid hazard), B * max_L * D ~ 2.68B > INT_MAX
        (so PTA32 stride math wraps for oidx >= 819, i.e. the last ~25% of
        batches). Peak GPU memory ~ B * max_L * D * 2 bytes ~ 5 GB.

        The single jagged value of 1.0 is placed in the LAST batch (at
        oidx = B - 1 = 1023, well into the wrap zone). On unfixed code the
        write to out[1023, 0, :] never happens; it stays at uninitialized
        memory. After the fix routes this shape to the int64 (PTA64) kernel
        path the write lands correctly.
        """
        device = torch.accelerator.current_accelerator()
        B = 1024
        max_L = 40960
        D = 64
        dtype = torch.bfloat16
        assert B * max_L < (1 << 31), (
            "test must keep B * max_L below INT_MAX (we are testing the PTA32 "
            "stride-overflow path, not the outer-loop overflow path)"
        )
        assert (
            B * max_L * D > (1 << 31) - 1
        ), "test must exceed INT_MAX on B * max_L * D"
        # PTA32 stride wraps for oidx >= ceil(INT_MAX / (max_L * D)).
        wrap_oidx = (((1 << 31) - 1) + (max_L * D) - 1) // (max_L * D)
        assert (
            B - 1 >= wrap_oidx
        ), "last batch must fall inside the PTA32 stride-overflow zone"

        # Tiny jagged: only the LAST batch has length=1, all others empty.
        # That keeps total_L = 1 (and thus the (total_L, D) values tensor
        # microscopic) so the only large allocation is the (B, max_L, D)
        # padded output.
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[B] = 1
        grad_jagged = torch.ones((1, D), dtype=dtype, device=device)

        # Exact code path that dense_to_jagged.backward executes.
        out = torch.ops.fbgemm.jagged_to_padded_dense(
            grad_jagged, [offsets], [max_L], padding_value=0.0
        )

        self.assertEqual(tuple(out.shape), (B, max_L, D))

        # The strongest signal: the in-range position MUST equal the jagged
        # value. On unfixed code oidx=B-1 falls in the PTA32 wrap zone and
        # the write goes to a wrong address; this position stays as
        # whatever at::empty left in memory - anything but 1.0.
        self.assertEqual(out[B - 1, 0, 0].item(), 1.0)
        self.assertEqual(out[B - 1, 0, D - 1].item(), 1.0)

        # Spot-check padding inside the wrap zone (oidx=B-1) and outside
        # it (oidx=0). 0.0 here means the kernel correctly wrote
        # padding_value; an unwritten cell would surface as garbage / NaN
        # after the autograd reduce.
        for pos in [1, max_L // 2, max_L - 1]:
            self.assertEqual(out[B - 1, pos, 0].item(), 0.0)
            self.assertEqual(out[0, pos, 0].item(), 0.0)
        # Spot-check a batch right at the wrap boundary.
        self.assertEqual(out[wrap_oidx, 0, 0].item(), 0.0)
        self.assertEqual(out[wrap_oidx, max_L - 1, 0].item(), 0.0)

    @given(
        num_jagged_dim=st.integers(1, 5),
        outer_dense_size=st.integers(0, 5),
        inner_dense_size=st.integers(0, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=st.sampled_from([torch.device("meta")]),
        precompute_total_L=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_meta_backend(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
        precompute_total_L: bool,
    ) -> None:
        device = torch.device("cpu")
        values_2d, offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)

        # jagged -> dense
        dense = torch.ops.fbgemm.jagged_to_padded_dense(values_2d, offsets, max_lengths)

        # dense -> jagged (op which is being tested)
        if precompute_total_L:
            total_L = values_2d.size(0)
            dense.to(device)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets, total_L
            )
        else:
            dense.to(device)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets
            )

        jagged_values.to(device)
        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        assert dense.size() == dense2.size()

    @optests.dontGenerateOpCheckTests("tests that call torch.compile are slow")
    @given(
        num_jagged_dim=st.integers(1, 5),
        # TODO: size = 0/1 will be incorrectly specialized
        outer_dense_size=st.integers(2, 5),
        inner_dense_size=st.integers(2, 5),
        dtype=st.sampled_from([torch.float, torch.half, torch.bfloat16]),
        device=cpu_and_maybe_gpu(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dense_to_jagged_dynamic_shape(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Start a fresh compile for each parameter of the test case
        torch._dynamo.reset()

        values_2d, offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            dtype,
            device,
            mark_dynamic=True,
        )
        values_2d = values_2d.clone().detach().requires_grad_(True)

        def jagged_to_dense(
            values: torch.Tensor,
            offsets: list[torch.LongTensor],
            max_lengths: list[int],
        ) -> torch.Tensor:
            return torch.ops.fbgemm.jagged_to_padded_dense(values, offsets, max_lengths)

        # jagged -> dense
        dense = jagged_to_dense(values_2d, offsets, max_lengths.tolist())

        # dense -> jagged, it is required to pre-compute totalL
        total_L = values_2d.size(0)
        dense = dense.clone().detach().to(device)

        torch._dynamo.mark_dynamic(dense, 0)
        torch._dynamo.mark_dynamic(dense, -1)

        def dense_to_jagged_withL(
            dense: torch.Tensor, offsets: list[torch.LongTensor], total_L: list[int]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.fbgemm.dense_to_jagged(dense, offsets, total_L)

        def dense_to_jagged_noL(
            dense: torch.Tensor, offsets: list[torch.LongTensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.fbgemm.dense_to_jagged(dense, offsets)

        jagged_values, jagged_offsets = dense_to_jagged_noL(dense, offsets)
        jagged_values, jagged_offsets = dense_to_jagged_withL(dense, offsets, total_L)

        jagged_values.to(device)
        # jagged -> dense
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )

        # verify forward
        assert dense.size() == dense2.size()

    @optests.dontGenerateOpCheckTests("regression test, not an op-shape check")
    @unittest.skipIf(*gpu_unavailable)
    def test_dense_to_jagged_heap_symint_total_L(self) -> None:
        """Regression: dense_to_jagged_forward crashes on AMD/HIP when the
        total_L argument is a heap-allocated SymInt rather than an inline
        concrete int.

        Production failure (Stories LSR on MI350X, MAST f1096341099): the
        outer forward is wrapped by torch.compile; downstream a Python int
        num_events was symbolicalized into a heap SymInt and forwarded to
        torch.ops.fbgemm.dense_to_jagged. dense_to_jagged_forward.cu then
        calls
            at::empty_symint({total_L_computed, D}, ...)
        with the heap SymInt. On the hipified empty.memory_format dispatcher
        the shape array goes through asIntArrayRefSlow which raises
            "SymIntArrayRef expected to contain only concrete integers"
        or, when the SymNode pointer slips through unchecked, lands in
        empty_generic with the pointer reinterpreted as int64, producing
            "Trying to create tensor with negative dimension <huge negative>".

        This test bypasses dynamo and constructs the heap SymInt directly
        via ShapeEnv.create_unbacked_symint(), then calls the real op with
        a real CUDA/HIP tensor. That reproduces the exact kernel-level
        condition without depending on which torch.compile backend / version
        chooses to preserve vs. realize the SymInt.

        Fix: dense_to_jagged_forward must realize the total_L SymInt to a
        concrete int64 (via .guard_int(__FILE__, __LINE__)) before
        constructing the empty output, instead of forwarding the raw SymInt
        to empty_symint.
        """
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        device = torch.accelerator.current_accelerator()

        B = 4
        D = 8
        max_L = 16
        total_L_concrete = 17

        lengths = torch.tensor([3, 5, 2, 7], dtype=torch.long, device=device)
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[1:] = lengths.cumsum(0)
        self.assertEqual(int(offsets[-1].item()), total_L_concrete)

        dense = torch.randn(B, max_L, D, dtype=torch.float32, device=device)

        # Construct an unbacked, heap-allocated SymInt with no hint. The
        # IValue(c10::SymInt) constructor only preserves heap-ness when the
        # SymNode's maybe_as_int() returns None -- i.e. when the SymInt is
        # truly unbacked (no hint, no constant simplification). A SymInt
        # with a hint is collapsed to Tag::Int (inline) by IValue and would
        # not exercise the kernel-level crash path.
        shape_env = ShapeEnv()
        total_L_sym = shape_env.create_unbacked_symint()

        # Pre-fix kernel forwards this heap SymInt to at::empty_symint,
        # which crashes in the empty.memory_format HIP wrapper at
        #   "SymIntArrayRef expected to contain only concrete integers".
        # Post-fix the kernel realizes total_L via guard_int(__FILE__, __LINE__)
        # before constructing the empty output. guard_int on a truly
        # unbacked SymNode (no hint, no runtime guard) raises a clean
        # GuardOnDataDependentSymNode -- which is the correct user-visible
        # behavior: "the kernel cannot allocate without a concrete size."
        # That clean error is what we assert here. The buggy pre-fix path
        # raised the low-level SymIntArrayRef error from inside the empty
        # wrapper, which fails this regex match.
        with self.assertRaisesRegex(
            RuntimeError,
            r"Could not extract specialized integer from data-dependent expression",
        ):
            torch.ops.fbgemm.dense_to_jagged(dense, [offsets], total_L_sym)


if __name__ == "__main__":
    unittest.main()

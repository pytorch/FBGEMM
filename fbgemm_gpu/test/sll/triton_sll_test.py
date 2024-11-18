# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import fbgemm_gpu
import fbgemm_gpu.sll.cpu_sll  # noqa F401
import fbgemm_gpu.sll.triton_sll  # noqa F401

import torch
from hypothesis import given, settings, strategies as st
from torch.testing._internal.optests import opcheck

# pyre-ignore[16]: Module `fbgemm_gpu` has no attribute `open_source`
open_source: bool = getattr(fbgemm_gpu, "open_source", False)

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    from fbgemm_gpu.test.test_utils import gpu_unavailable

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")


class TritonBmmTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 512),
        D=st.integers(1, 256),
        N=st.integers(1, 1000),
        T=st.integers(1, 256),
        use_fbgemm_kernel=st.booleans(),
        allow_tf32=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_triton_jagged_dense_bmm(
        self,
        B: int,
        D: int,
        N: int,
        T: int,
        allow_tf32: bool,
        use_fbgemm_kernel: bool,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)
        lengths = torch.randint(0, N + 1, (B,), device=device)
        offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths.cumsum(dim=0),
            ],
            dim=0,
        )
        x = torch.rand(int(lengths.sum().item()), D, device=device)
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
            x,
            [offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        y = torch.rand((B, D, T), device=device)
        padded_ref = torch.bmm(padded_x, y)
        ref = torch.ops.fbgemm.dense_to_jagged(padded_ref, [offsets])[0]
        ret = torch.ops.fbgemm.sll_jagged_dense_bmm(
            x, y, offsets, N, allow_tf32=allow_tf32, use_fbgemm_kernel=use_fbgemm_kernel
        )
        if allow_tf32:
            assert torch.allclose(ref, ret, atol=1e-3, rtol=1e-3)
        else:
            assert torch.allclose(ref, ret, 1e-5)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 512),
        D=st.integers(1, 256),
        N=st.integers(1, 1000),
        T=st.integers(1, 256),
        use_fbgemm_kernel=st.booleans(),
        allow_tf32=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_triton_jagged_dense_bmm_with_grad(
        self,
        B: int,
        D: int,
        N: int,
        T: int,
        allow_tf32: bool,
        use_fbgemm_kernel: bool,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)
        lengths = torch.randint(0, N + 1, (B,), device=device)
        offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths.cumsum(dim=0),
            ],
            dim=0,
        )

        torch.manual_seed(0)
        x1 = torch.rand(
            (int(lengths.sum().item()), D), requires_grad=True, device=device
        )
        padded_x1 = torch.ops.fbgemm.jagged_to_padded_dense(
            x1,
            [offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        y1 = torch.rand((B, D, T), requires_grad=True, device=device)
        padded_ref = torch.bmm(padded_x1, y1)
        ref = torch.ops.fbgemm.dense_to_jagged(padded_ref, [offsets])[0]

        torch.manual_seed(0)
        x2 = torch.rand(
            (int(lengths.sum().item()), D), requires_grad=True, device=device
        )
        y2 = torch.rand((B, D, T), requires_grad=True, device=device)
        ret = torch.ops.fbgemm.sll_jagged_dense_bmm(
            x2,
            y2,
            offsets,
            N,
            allow_tf32=allow_tf32,
            use_fbgemm_kernel=use_fbgemm_kernel,
        )

        if allow_tf32:
            assert torch.allclose(ref, ret, atol=1e-3, rtol=1e-3)
        else:
            assert torch.allclose(ref, ret, 1e-5)

        grad_output = torch.rand((int(lengths.sum().item()), T), device=device) * 0.01
        ref.backward(grad_output)
        ret.backward(grad_output)

        if allow_tf32:
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(x1.grad, x2.grad, atol=1e-3, rtol=1e-3)
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(y1.grad, y2.grad, atol=1e-3, rtol=1e-3)
        else:
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(x1.grad, x2.grad, 1e-5)
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(y1.grad, y2.grad, 1e-5)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 512),
        D=st.integers(1, 256),
        N=st.integers(1, 1000),
        T=st.integers(1, 256),
        use_fbgemm_kernel=st.booleans(),
        allow_tf32=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_triton_jagged_jagged_bmm(
        self,
        B: int,
        D: int,
        N: int,
        T: int,
        use_fbgemm_kernel: bool,
        allow_tf32: bool,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)
        lengths = torch.randint(0, N + 1, (B,), device=device)
        offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths.cumsum(dim=0),
            ],
            dim=0,
        )
        x = torch.rand(int(lengths.sum().item()), D, device=device)
        y = torch.rand(int(lengths.sum().item()), T, device=device)
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
            x,
            [offsets],
            max_lengths=[N],
            padding_value=0.0,
        )  # [B, N, D]
        padded_y = torch.ops.fbgemm.jagged_to_padded_dense(
            y,
            [offsets],
            max_lengths=[N],
            padding_value=0.0,
        )  # [B, N, T]
        ret = torch.ops.fbgemm.sll_jagged_jagged_bmm(
            x, y, offsets, N, allow_tf32=allow_tf32, use_fbgemm_kernel=use_fbgemm_kernel
        )
        ref = torch.bmm(padded_x.permute(0, 2, 1), padded_y)

        if allow_tf32:
            assert torch.allclose(ref, ret, atol=1e-3, rtol=1e-3)
        else:
            assert torch.allclose(ref, ret, 1e-5)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 512),
        D=st.integers(1, 256),
        N=st.integers(1, 1000),
        T=st.integers(1, 256),
        use_fbgemm_kernel=st.booleans(),
        allow_tf32=st.booleans(),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_triton_jagged_jagged_bmm_with_grad(
        self,
        B: int,
        D: int,
        N: int,
        T: int,
        use_fbgemm_kernel: bool,
        allow_tf32: bool,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)
        lengths = torch.randint(0, N + 1, (B,), device=device)
        offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths.cumsum(dim=0),
            ],
            dim=0,
        )

        torch.manual_seed(0)
        x1 = torch.rand(
            (int(lengths.sum().item()), D), requires_grad=True, device=device
        )  # [Sum_B, D]
        padded_x1 = torch.ops.fbgemm.jagged_to_padded_dense(
            x1,
            [offsets],
            max_lengths=[N],
            padding_value=0.0,
        )  # [B, N, D]
        y1 = torch.rand(
            (int(lengths.sum().item()), T), requires_grad=True, device=device
        )  # [Sum_B, T]
        padded_y1 = torch.ops.fbgemm.jagged_to_padded_dense(
            y1,
            [offsets],
            max_lengths=[N],
            padding_value=0.0,
        )  # [B, N, T]

        ref = torch.bmm(padded_x1.permute(0, 2, 1), padded_y1)

        # triton version
        torch.manual_seed(0)
        x2 = torch.rand(
            (int(lengths.sum().item()), D), requires_grad=True, device=device
        )  # [Sum_B, D]
        y2 = torch.rand(
            (int(lengths.sum().item()), T), requires_grad=True, device=device
        )  # [Sum_B, T]
        ret = torch.ops.fbgemm.sll_jagged_jagged_bmm(
            x2,
            y2,
            offsets,
            N,
            allow_tf32=allow_tf32,
            use_fbgemm_kernel=use_fbgemm_kernel,
        )

        if allow_tf32:
            assert torch.allclose(ref, ret, atol=1e-3, rtol=1e-3)
        else:
            assert torch.allclose(ref, ret, 1e-5)

        # grad check
        grad_output = torch.rand((B, D, T), device=device) * 0.01
        ref.backward(grad_output)
        ret.backward(grad_output)

        if allow_tf32:
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(y1.grad, y2.grad, atol=1e-3, rtol=1e-3)
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(x1.grad, x2.grad, atol=1e-3, rtol=1e-3)
        else:
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(y1.grad, y2.grad, 1e-5)
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.allclose`, for 1st positional argument, expected `Tensor` but got `Optional[Tensor]`.Pyre
            assert torch.allclose(x1.grad, x2.grad, 1e-5)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 512),
        max_L=st.integers(1, 200),
        device_type=st.sampled_from(["cpu", "cuda"]),
        enable_pt2=st.sampled_from([True, False]),
    )
    @settings(deadline=None)
    def test_dense_jagged_cat_jagged_out(
        self,
        B: int,
        max_L: int,
        device_type: str,
        enable_pt2: bool,
    ) -> None:
        device = torch.device(device_type)
        lengths = torch.randint(0, max_L + 1, (B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        c_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths + 1)
        a = torch.randint(0, 100000000, (B,), device=device)
        b = torch.randint(0, 100000000, (int(lengths.sum().item()),), device=device)

        ref = torch.cat(
            [
                (
                    torch.cat((a[i : i + 1], b[offsets[i] : offsets[i + 1]]), dim=-1)
                    if lengths[i] > 0
                    else a[i : i + 1]
                )
                for i in range(B)
            ],
            dim=-1,
        )

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def model(a, b, offsets, max_L):
            return torch.ops.fbgemm.sll_dense_jagged_cat_jagged_out(
                a, b, offsets, max_L
            )

        if enable_pt2:
            model = torch.compile(model)

        ret, c_offsets_computed = model(a, b, offsets, max_L)

        assert torch.allclose(ref, ret)
        assert torch.equal(c_offsets, c_offsets_computed)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(10, 100),
        L=st.integers(1, 200),
        device_type=st.sampled_from(["cpu", "cuda"]),
        enable_pt2=st.booleans(),
    )
    @settings(deadline=None)
    def test_triton_jagged_self_substraction_jagged_out(
        self,
        B: int,
        L: int,
        device_type: str,
        enable_pt2: bool,
    ) -> None:
        device = torch.device(device_type)
        torch.manual_seed(0)
        lengths_a = torch.randint(1, L + 2, (B,), device=device)
        offsets_a = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths_a.cumsum(dim=0),
            ],
            dim=0,
        )

        lengths_b = (lengths_a - 1) * (lengths_a - 1)

        offsets_b = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                lengths_b.cumsum(dim=0),
            ],
            dim=0,
        )

        jagged_A = torch.randint(
            0, 100000000, (int(lengths_a.sum().item()),), device=device
        )

        def model(
            jagged_A: torch.Tensor,
            offsets_a: torch.Tensor,
            offsets_b: torch.Tensor,
            L: int,
        ) -> torch.Tensor:
            return torch.ops.fbgemm.sll_jagged_self_substraction_jagged_out(
                jagged_A,
                offsets_a,
                offsets_b,
                L,
            )

        if enable_pt2:
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.config.capture_scalar_outputs = True
            opcheck(
                torch.ops.fbgemm.sll_jagged_self_substraction_jagged_out,
                (jagged_A, offsets_a, offsets_b, L),
            )
            model = torch.compile(model)

        result = model(jagged_A, offsets_a, offsets_b, L)

        for i in range(B):
            if lengths_a[i] == 1:
                continue

            a = jagged_A[offsets_a[i] : offsets_a[i + 1]]
            ref = a[:-1].unsqueeze(1) - a[1:].unsqueeze(0)

            assert torch.equal(result[offsets_b[i] : offsets_b[i + 1]], ref.flatten())

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @given(
        B=st.integers(1, 10),
        max_L=st.integers(1, 100),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(deadline=None)
    def test_jagged2_to_padded_dense(
        self,
        B: int,
        max_L: int,
        device_type: str,
    ) -> None:
        device = torch.device(device_type)

        lengths = torch.randint(1, max_L + 1, (B,), device=device)
        lengths_square = lengths * lengths
        offsets = torch.cat(
            [
                torch.tensor([0], device=device, dtype=torch.int),
                lengths_square.cumsum(dim=0),
            ],
            dim=0,
        )

        x = torch.rand(
            int(lengths_square.sum().item()),
            requires_grad=True,
            device=device,
        )

        def ref_jagged2_to_padded_dense(
            x: torch.Tensor, offsets: torch.Tensor, max_L: int, padding_value: float
        ) -> torch.Tensor:
            B = offsets.size(0) - 1
            dense_output = torch.full(
                (B, max_L, max_L),
                padding_value,
                dtype=x.dtype,
                device=x.device,
            )
            for b in range(B):
                begin = offsets[b]
                end = offsets[b + 1]
                Ni = int(torch.sqrt(end - begin))
                if Ni == 0:
                    continue
                dense_output[b, 0:Ni, 0:Ni] = x[begin:end].view(Ni, Ni)

            return dense_output

        x_clone = (
            x.detach().clone().requires_grad_()
            if x.requires_grad
            else x.detach().clone()
        )
        padding_value = 0.0
        ref_out = ref_jagged2_to_padded_dense(x, offsets, max_L, padding_value)
        test_out = torch.ops.fbgemm.sll_jagged2_to_padded_dense(
            x_clone, offsets, max_L, padding_value
        )
        assert torch.allclose(ref_out, test_out)

        # Backward pass
        dout = torch.rand((B, max_L, max_L), dtype=x.dtype, device=x.device) * 0.1
        test_out.backward(dout)
        ref_out.backward(dout)

        assert x.grad is not None
        assert x_clone.grad is not None
        assert torch.allclose(x.grad, x_clone.grad)

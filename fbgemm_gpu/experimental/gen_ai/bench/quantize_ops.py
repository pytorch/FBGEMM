# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Keep a registry of all quantize operators.
import abc
from typing import List, Tuple

import fbgemm_gpu.experimental.gen_ai  # noqa: F401

import torch
import triton  # @manual=//triton:triton
from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
    matmul_fp8_block,
    matmul_fp8_row,
    quantize_fp8_block,
    quantize_fp8_row,
    scale_fp8_row,
)
from tinygemm.utils import group_quantize_tensor

if torch.cuda.is_available() and torch.version.cuda:
    torch.ops.load_library("//tinygemm:tinygemm")

# Marlin currently only is supported only internally at Meta.
try:
    from marlin.quantize import marlin_quantize

    torch.ops.load_library("//ai_codesign/gen_ai/marlin:marlin_ops")
    MARLIN_ENABLED = True
except ImportError:
    MARLIN_ENABLED = False


quantize_op_registry = []


class QuantizeOpBase(metaclass=abc.ABCMeta):
    """Helper abstract class to define expected methods of quantize ops."""

    @abc.abstractmethod
    def quantize(self, *args):
        """Function which quantizes inputs."""
        pass

    @abc.abstractmethod
    def compute(self, *args):
        """Function which performs main compute operation."""
        pass

    @abc.abstractmethod
    def quantize_and_compute(self, *args):
        """Function which quantizes inputs and performs main compute operation."""
        pass

    def bench_with_rotating_buffer(self, fn, args):
        import copy
        import pickle

        # torch.cuda.get_device_properties does not have L2/L3 cache size,
        # so hard code an overapproximation of L2/L3 cache size to ensure L2/L3 cache flush
        total_buffer_size = 512 * 1024 * 1024

        # Use pickle to serialize model input to estimate total sizes of input
        input_sizes = len(pickle.dumps(args))

        # Make at least one copy of the inputs
        copy_cnt = total_buffer_size // input_sizes
        if copy_cnt == 0:
            copy_cnt = 1

        args_list = [args]
        for _ in range(copy_cnt):
            args_list.append(copy.deepcopy(args))

        def rotating_buffer_fn(fn, args_list, copy_cnt):
            for i in range(copy_cnt):
                fn(*(args_list[i]))

        with torch.cuda.stream(torch.cuda.Stream()):
            # A rotating_buffer_fn contains multiple runs of the fn to benchmark,
            # so divide time accordingly
            return triton.testing.do_bench_cudagraph(
                lambda: rotating_buffer_fn(self.compute, args_list, copy_cnt + 1),
                rep=200,
            ) / (copy_cnt + 1)

    def benchmark(
        self,
        *args,
        bench_quantize: bool = False,
        use_rotating_buffer_bench: bool = False
    ) -> float:
        """Benchmark runtime of this operator."""
        if bench_quantize:
            with torch.cuda.stream(torch.cuda.Stream()):
                t = triton.testing.do_bench_cudagraph(
                    lambda: self.quantize_and_compute(*args)
                )
        else:
            if use_rotating_buffer_bench:
                t = self.bench_with_rotating_buffer(self.compute, args)
            else:
                with torch.cuda.stream(torch.cuda.Stream()):
                    t = triton.testing.do_bench_cudagraph(lambda: self.compute(*args))
        return t

    @abc.abstractproperty
    def name(self) -> str:
        """Name of the operator."""
        pass

    @abc.abstractproperty
    def hip(self) -> bool:
        """Whether this operator supports AMD or not."""
        pass

    @abc.abstractproperty
    def cuda(self) -> bool:
        """Whether this operator supports Nvidia or not."""
        pass

    @property
    def supported(self) -> bool:
        """Whether this op will run on the current device."""
        if torch.version.hip is not None:
            return self.hip
        elif torch.version.cuda is not None:
            return self.cuda
        else:
            return False


def register_quantize_op(op):
    """Decorator function for assembling all quantize ops."""
    quantize_op_registry.append(op())
    return op


def get_quantize_ops() -> List[QuantizeOpBase]:
    """Get all registered quantize ops."""
    return quantize_op_registry


@register_quantize_op
class BF16Baseline(QuantizeOpBase):
    """
    Baseline BF16 matmul.
    """

    def quantize(self, x, w):
        return x.bfloat16(), w.t().bfloat16()

    def compute(self, x, w):
        return torch.matmul(x, w)

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "bf16_baseline"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class ScaledMMBaseline(QuantizeOpBase):
    """
    Reference FP8 matmul implemented in native torch with cublas or hipblas.
    """

    def __init__(self):
        if torch.version.cuda is not None:
            self.fp8_dtype = torch.float8_e4m3fn
        else:
            self.fp8_dtype = torch.float8_e4m3fnuz
        self.E4M3_MAX_POS: float = torch.finfo(self.fp8_dtype).max
        self.E5M2_MAX_POS: float = torch.finfo(torch.float8_e5m2).max
        self.FP16_MAX_POS: float = torch.finfo(torch.float16).max
        self.EPS: float = 1e-12

    def _amax_to_scale(
        self, amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
    ) -> torch.Tensor:
        # To make scale dtype to be fp32 for accuracy
        amax = amax.float()
        if float8_dtype == self.fp8_dtype:
            # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
            res = self.E4M3_MAX_POS / torch.clamp(amax, min=self.EPS)
        else:  # e5m2
            # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
            res = self.E5M2_MAX_POS / torch.clamp(amax, min=self.EPS)

        # pyre-fixme[7]: Expected `Tensor` but got `Union[float, Tensor]`.
        return res

    def _to_fp8_saturated(
        self, x: torch.Tensor, float8_dtype: torch.dtype
    ) -> torch.Tensor:
        if float8_dtype == torch.float8_e4m3fn:
            x = x.clamp(min=-1 * self.E4M3_MAX_POS, max=self.E4M3_MAX_POS)
        else:
            x = x.clamp(min=-1 * self.E5M2_MAX_POS, max=self.E5M2_MAX_POS)
        return x.to(float8_dtype)

    def _quantize_tensor(self, x):
        x_amax = torch.max(torch.abs(x))
        scale = self._amax_to_scale(x_amax, self.fp8_dtype, x.dtype)
        scaled_x = self._to_fp8_saturated(x * scale, self.fp8_dtype)
        x_inverse_scale = scale.reciprocal()
        return scaled_x, x_inverse_scale

    def quantize(self, x, w):
        xq, x_scale = self._quantize_tensor(x)
        wq, w_scale = self._quantize_tensor(w.t())
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        output = torch._scaled_mm(
            xq,
            wq,
            bias=None,
            out_dtype=torch.bfloat16,
            scale_a=x_scale,
            scale_b=w_scale,
            scale_result=None,
            use_fast_accum=True,
        )
        return output

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "scaled_mm"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class ScaledMMRowwise(QuantizeOpBase):
    def quantize(self, x, w):
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        dummy_scale = torch.tensor([1.0], device=x.device, dtype=torch.float32)
        return xq, wq.t(), x_scale, w_scale, dummy_scale

    def compute(self, xq, wq, x_scale, w_scale, dummy_scale):
        output = torch._scaled_mm(
            xq,
            wq,
            bias=None,
            out_dtype=torch.bfloat16,
            scale_a=dummy_scale,
            scale_b=dummy_scale,
            scale_result=None,
            use_fast_accum=True,
        )
        # Apply separate rowwise scaling.
        output = scale_fp8_row(output, x_scale, w_scale)
        return output

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "scaled_mm_rowwise"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class FP8TensorwiseGemm(QuantizeOpBase):
    """
    FP8 matmul with tensorwise scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.fbgemm.f8f8bf16(xq, wq, x_scale * w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cutlass_tensorwise"

    @property
    def hip(self) -> bool:
        # Need to add support for better quantize kernel.
        # Also may have an issue with cuda graphs.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class FP8CublasRowwiseGemm(QuantizeOpBase):
    """
    FP8 matmul with tensorwise scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        out = torch.ops.fbgemm.f8f8bf16_cublas(xq, wq)
        scaled_out = scale_fp8_row(out, x_scale, w_scale)
        return scaled_out

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cublas_rowwise"

    @property
    def hip(self) -> bool:
        # This implementation is specific to cublas.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class FP8RowwiseGemm(QuantizeOpBase):
    """
    FP8 matmul with rowwise scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.fbgemm.f8f8bf16_rowwise(xq, wq, x_scale, w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_rowwise"
        else:
            return "ck_rowwise"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class TritonFP8RowwiseGemm(QuantizeOpBase):
    """
    FP8 matmul with rowwise scaling implemented with Triton.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        bias = torch.randn(w.shape[0], device=x.device, dtype=torch.float32)
        return xq, wq, x_scale, w_scale, bias

    def compute(self, xq, wq, x_scale, w_scale, bias):
        return matmul_fp8_row(xq, wq, x_scale, w_scale, bias=bias)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "triton_rowwise"

    @property
    def hip(self) -> bool:
        # triton FP8 matmuls do not currently compile on AMD.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class FP8TritonBlockwiseGemm(QuantizeOpBase):
    """
    FP8 matmul with block scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_block(x, 128, 128)
        wq, w_scale = quantize_fp8_block(w, 128, 128)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return matmul_fp8_block(xq, wq, x_scale, w_scale, 128, 128, 128)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "triton_blockwise"

    @property
    def hip(self) -> bool:
        # Currently has some issues.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class FP8CutlassBlockwiseGemm(QuantizeOpBase):
    """
    FP8 matmul with block scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_block(x, 128, 128)
        wq, w_scale = quantize_fp8_block(w, 128, 128)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.fbgemm.f8f8bf16_blockwise(
            xq, wq, x_scale, w_scale, 128, 128, 128
        )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_blockwise"
        else:
            return "ck_blockwise"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


# CUTLASS kernel v2
@register_quantize_op
class CutlassFP8TensorwiseGemm_v2(QuantizeOpBase):
    """
    FP8 matmul with tensorwise scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.fbgemm.f8f8bf16_v2(xq, wq, x_scale * w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cutlass_tensorwise_v2"

    @property
    def hip(self) -> bool:
        # Need to add support for better quantize kernel.
        # Also may have an issue with cuda graphs.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class F8I4RowwiseGemm(QuantizeOpBase):
    """
    Mixed Precision FP8 Activations with Int4 Weights.
    """

    def _int4_row_quantize(
        self,
        x: torch.Tensor,
        group_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_bit = 4  # Number of target bits.
        to_quant = x.reshape(-1, group_size).to(torch.float)

        max_val = to_quant.amax(dim=1, keepdim=True)
        min_val = to_quant.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-6) / max_int

        zeros = min_val + scales * (2 ** (n_bit - 1))

        out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)

        # Recenter output and move to int8.
        out = (out - 2 ** (n_bit - 1)).to(dtype=torch.int8).reshape(x.shape)

        # Cutlass expects column major layout for scale and zero point,
        # so we transpose here and make them contiguous.
        scales = scales.view(x.shape[0], -1).t().contiguous()
        zeros = zeros.view(x.shape[0], -1).t().contiguous()

        return out, scales, zeros

    def _pack_int4(self, x: torch.Tensor) -> torch.Tensor:
        # Given int8 x, pack adjacent int4 values into a single int8.
        low_x = x[:, ::2]
        high_x = x[:, 1::2]

        # High bits need to left shift, this also masks off extra bits.
        high_x = torch.bitwise_left_shift(high_x, 4)
        # Low bits need to have sign bits removed.
        low_x = torch.bitwise_and(low_x, 0xF)

        # Recombine into a single value with bitwise or.
        return torch.bitwise_or(low_x, high_x).contiguous()

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale, w_zp = self._int4_row_quantize(w)
        # Pack int4 values together.
        wq = self._pack_int4(wq)
        return xq, wq, x_scale, w_scale, w_zp

    def compute(self, xq, wq, x_scale, w_scale, w_zp):
        return torch.ops.fbgemm.f8i4bf16_rowwise(xq, wq, x_scale, w_scale, w_zp)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale, w_zp = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale, w_zp)

    @property
    def name(self) -> str:
        return "cutlass_f8i4_rowwise"

    @property
    def hip(self) -> bool:
        # Not yet supported on AMD.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class BF16I4RowwiseGemm(F8I4RowwiseGemm):
    """
    Mixed Precision BF16 Activations with Int4 Weights.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        wq, w_scale, w_zp = self._int4_row_quantize(w)
        # Pack int4 values together.
        wq = self._pack_int4(wq)
        return x.to(torch.bfloat16), wq, w_scale, w_zp

    def compute(self, x, wq, w_scale, w_zp):
        return torch.ops.fbgemm.bf16i4bf16_rowwise(x, wq, w_scale, w_zp)

    def quantize_and_compute(self, x, w):
        x, wq, w_scale, w_zp = self.quantize(x, w)
        return self.compute(x, wq, w_scale, w_zp)

    @property
    def name(self) -> str:
        return "cutlass_bf16i4_rowwise"

    @property
    def hip(self) -> bool:
        # Not yet supported on AMD.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class TinyGemmBF16I4(QuantizeOpBase):
    """
    Mixed Precision BF16 Activations with Int4 Weights using tinygemm.
    """

    def quantize(self, x, w):
        # Quantize and pack weights to int4 using tinygemm utils.
        w_int32, w_scales_and_zeros = group_quantize_tensor(
            w, n_bit=4, q_group_size=128
        )
        wq = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, 4)
        return x, wq, w_scales_and_zeros

    def compute(self, x, wq, scale):
        return torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_int4TC(
            wq, x, 128, scale, False
        )

    def quantize_and_compute(self, x, w):
        x, wq, scale = self.quantize(x, w)
        return self.compute(x, wq, scale)

    @property
    def name(self) -> str:
        return "tinygemm_bf16i4"

    @property
    def hip(self) -> bool:
        # Tinygemm only supported for cuda.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class MarlinBF16I4(QuantizeOpBase):
    """
    Mixed Precision BF16 Activations with Int4 Weights using Marlin.
    """

    def quantize(self, x, w):
        # Marlin quantize expects weights in [K, N] layout.
        _, wq, scale = marlin_quantize(w.t().contiguous(), 128)
        return x, wq, scale

    def compute(self, x, wq, scale):
        return torch.ops.marlin.marlin_gemm(x, wq, scale)

    def quantize_and_compute(self, x, w):
        x, wq, scale = self.quantize(x, w)
        return self.compute(x, wq, scale)

    @property
    def name(self) -> str:
        return "marlin_bf16i4"

    @property
    def hip(self) -> bool:
        # Marlin only supported for cuda.
        return False

    @property
    def cuda(self) -> bool:
        # This op is not always supported.
        return MARLIN_ENABLED

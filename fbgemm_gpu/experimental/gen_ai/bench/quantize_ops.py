# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Keep a registry of all quantize operators.
import abc
from typing import List

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

    def benchmark(self, *args, bench_quantize: bool = False) -> float:
        """Benchmark runtime of this operator."""
        if bench_quantize:
            return triton.testing.do_bench(lambda: self.quantize_and_compute(*args))
        else:
            return triton.testing.do_bench(lambda: self.compute(*args))

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
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return matmul_fp8_row(xq, wq, x_scale, w_scale)

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

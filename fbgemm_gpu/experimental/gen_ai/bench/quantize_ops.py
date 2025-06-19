# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Keep a registry of all quantize operators.
import abc
from typing import List, Tuple

import fbgemm_gpu.experimental.gen_ai  # noqa: F401
import numpy as np

import torch
import triton  # @manual=//triton:triton

from fbgemm_gpu.experimental.gemm.triton_gemm.fp4_quantize import (
    triton_quantize_mx4_unpack,
    triton_scale_nvfp4_quant,
    triton_scale_nvfp4_quant_rms,
    triton_scale_nvfp4_quant_silu,
)

from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import (
    get_fp8_constants,
    matmul_fp8_block,
    matmul_fp8_row,
    quantize_fp8_block,
    quantize_fp8_row,
    scale_fp8_row,
    triton_quantize_fp8_row,
)
from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import (
    grouped_gemm,
    grouped_gemm_fp8_rowwise,
)
from fbgemm_gpu.experimental.gen_ai.quantize import (
    ck_preshuffle,
    quantize_int4_preshuffle,
)

from gen_ai.llm_inference.fb.llm.kernel.rms_norm import rms_norm
from gen_ai.llm_inference.fb.llm.kernel.silu_mul import silu_mul

try:
    from tinygemm.utils import group_quantize_tensor

    if torch.cuda.is_available() and torch.version.cuda:
        torch.ops.load_library("//tinygemm:tinygemm")
    TINYGEMM_ENABLED = True
except ImportError:
    TINYGEMM_ENABLED = False

# Marlin currently only is supported only internally at Meta.
try:
    from marlin.quantize import marlin_quantize

    torch.ops.load_library("//ai_codesign/gen_ai/marlin:marlin_ops")
    MARLIN_ENABLED = True
except ImportError:
    MARLIN_ENABLED = False

try:
    from deep_gemm import (
        gemm_fp8_fp8_bf16_nt,
        get_col_major_tma_aligned_tensor,
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
        m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    )

    DEEPGEMM_ENABLED = True
except ImportError:
    DEEPGEMM_ENABLED = False


# Machete is also only supported internally at Meta for now.
try:
    from machete.machete import machete_gemm
    from machete.quantize import machete_quantize_and_pack

    MACHETE_ENABLED = True
except ImportError:
    MACHETE_ENABLED = False


quantize_op_registry = []


class QuantizeOpBase(metaclass=abc.ABCMeta):
    """Helper abstract class to define expected methods of quantize ops."""

    @abc.abstractmethod
    def quantize(self, *args):
        """Function which quantizes inputs."""
        pass

    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        """Function which performs main compute operation."""
        pass

    @abc.abstractmethod
    def quantize_and_compute(self, *args, **kwargs):
        """Function which quantizes inputs and performs main compute operation."""
        pass

    def preprocess(self, *args):
        """Preprocess inputs before benchmarking. These outputs will be passed to quantize."""
        return args

    def bench_with_rotating_buffer(self, fn, args, use_cuda_graph: bool = True):
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

        if use_cuda_graph:
            with torch.cuda.stream(torch.cuda.Stream()):
                # A rotating_buffer_fn contains multiple runs of the fn to benchmark,
                # so divide time accordingly
                return triton.testing.do_bench_cudagraph(
                    lambda: rotating_buffer_fn(self.compute, args_list, copy_cnt + 1),
                    rep=200,
                ) / (copy_cnt + 1)
        else:
            return triton.testing.do_bench(
                lambda: rotating_buffer_fn(self.compute, args_list, copy_cnt + 1),
                rep=200,
            ) / (copy_cnt + 1)

    def benchmark(
        self,
        *args,
        bench_quantize: bool = False,
        use_rotating_buffer_bench: bool = False,
        use_cuda_graph: bool = True,
        **kwargs,
    ) -> float:
        """Benchmark runtime of this operator."""
        if bench_quantize:
            if use_cuda_graph:
                with torch.cuda.stream(torch.cuda.Stream()):
                    t = triton.testing.do_bench_cudagraph(
                        lambda: self.quantize_and_compute(*args, **kwargs)
                    )
            else:
                t = triton.testing.do_bench(
                    lambda: self.quantize_and_compute(*args, **kwargs)
                )
        else:
            if use_rotating_buffer_bench:
                t = self.bench_with_rotating_buffer(self.compute, args, use_cuda_graph)
            else:
                if use_cuda_graph:
                    with torch.cuda.stream(torch.cuda.Stream()):
                        t = triton.testing.do_bench_cudagraph(
                            lambda: self.compute(*args, **kwargs)
                        )
                else:
                    t = triton.testing.do_bench(lambda: self.compute(*args, **kwargs))
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
        if isinstance(x, list):
            x = [i.bfloat16() for i in x]
            w = [torch.transpose(i, -2, -1).bfloat16() for i in w]
        else:
            x = x.bfloat16()
            w = torch.transpose(w, -2, -1).bfloat16()
        return x, w

    def compute(self, x, w):
        # Handle both grouped and standard gemm.
        if isinstance(x, list):
            output = []
            for i in range(len(x)):
                output.append(torch.matmul(x[i], w[i]))
            return output
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
        self.fp8_dtype, _, _, _ = get_fp8_constants()
        self.E4M3_MAX_POS: float = torch.finfo(self.fp8_dtype).max
        self.E5M2_MAX_POS: float = torch.finfo(torch.float8_e5m2).max
        self.FP16_MAX_POS: float = torch.finfo(torch.float16).max
        self.EPS: float = 1e-12
        self.fast_accum = True

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
            use_fast_accum=self.fast_accum,
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
    def __init__(self):
        self.fast_accum = True

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
            use_fast_accum=self.fast_accum,
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
class BF16OSSFastGemv(QuantizeOpBase):
    """
    BF16 OSS fast gemv kernel.
    """

    def quantize(self, x, w):
        # dummy quantize
        return x, w

    def compute(self, x, w):
        out = torch.ops.fbgemm.bf16_fast_gemv(x, w)
        return out

    def quantize_and_compute(self, x, w):
        x, w = self.quantize(x, w)
        return self.compute(x, w)

    @property
    def name(self) -> str:
        return "bf16_oss_fast_gemv"

    @property
    def hip(self) -> bool:
        # This implementation is specific to cublas.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class BF16Fp8OSSFastGemv(QuantizeOpBase):
    """
    BF16FP8 OSS fast gemv kernel.
    """

    def quantize(self, x, w):
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
        return x, wq, w_scale

    def compute(self, x, wq, w_scale):
        out = torch.ops.fbgemm.bf16fp8bf16_fast_gemv(x, wq, w_scale)
        return out

    def quantize_and_compute(self, x, w):
        x, wq, w_scale = self.quantize(x, w)
        return self.compute(x, wq, w_scale)

    @property
    def name(self) -> str:
        return "bf16fp8_oss_fast_gemv"

    @property
    def hip(self) -> bool:
        # This implementation is specific to cublas.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class Fp8Fp8OSSFastGemv(QuantizeOpBase):
    """
    FP8FP8 OSS fast gemv kernel.
    """

    def quantize(self, x, w):
        # rowwise quantize
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        out = torch.ops.fbgemm.fp8fp8bf16_fast_gemv(
            xq, wq, x_scale, w_scale, is_batched=False
        )
        return out

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "fp8fp8_oss_fast_gemv"

    @property
    def hip(self) -> bool:
        # This implementation is specific to cublas.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class Fp8OSSFastGemvBatched(QuantizeOpBase):
    """
    Batched fp8 fast gemv kernel
    """

    def quantize(self, x, w):
        # rowwise quantize
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x)
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        out = torch.ops.fbgemm.fp8fp8bf16_fast_gemv(
            xq, wq, x_scale, w_scale, is_batched=True
        )
        return out

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "fp8fp8_oss_fast_gemv_batched"

    @property
    def hip(self) -> bool:
        # This implementation is specific to cublas.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class FP8CublasRowwiseGemm(QuantizeOpBase):
    """
    FP8 cublas matmul with rowwise scaling.
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
class FP8CublasTensorwiseGemm(QuantizeOpBase):
    """
    FP8 cublas matmul with tensorwise scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.fbgemm.f8f8bf16_cublas(xq, wq, x_scale * w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale * w_scale)

    @property
    def name(self) -> str:
        return "cublas_tensorwise"

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

    def __init__(self):
        self.fast_accum = True
        self.gemm_op = torch.ops.fbgemm.f8f8bf16_rowwise

    def preprocess(self, x, w):
        # Prequantize weights.
        if isinstance(w, (list, tuple)):
            wq, w_scale = zip(*[quantize_fp8_row(i) for i in w])
        else:
            wq, w_scale = quantize_fp8_row(w)
            if wq.dim() == 3:
                w_scale = w_scale.view(wq.size(0), -1)
        return x, wq, w_scale

    def quantize(self, x, wq, w_scale):
        # Quantize both input tensors.
        # Handle both grouped and standard gemm.
        if isinstance(x, (list, tuple)):
            xq, x_scale = zip(*[quantize_fp8_row(i) for i in x])
        else:
            xq, x_scale = quantize_fp8_row(x)
            # Set proper batch dimension shapes.
            if xq.dim() == 3:
                x_scale = x_scale.view(xq.size(0), -1)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        # Handle group gemm if inputs are grouped.
        if isinstance(xq, (list, tuple)):
            output = []
            for i in range(len(xq)):
                output.append(
                    self.gemm_op(
                        xq[i],
                        wq[i],
                        x_scale[i],
                        w_scale[i],
                        use_fast_accum=self.fast_accum,
                    )
                )
            return output
        # Unroll batched gemm if needed.
        elif xq.dim() == 3 and wq.dim() == 3:
            B, M, _ = xq.shape
            _, N, _ = wq.shape
            y = torch.empty((B, M, N), device=xq.device, dtype=torch.bfloat16)
            for i in range(B):
                y[i] = self.gemm_op(
                    xq[i], wq[i], x_scale[i], w_scale[i], use_fast_accum=self.fast_accum
                )
            return y
        # Otherwise return normal gemm result.
        return self.gemm_op(xq, wq, x_scale, w_scale, use_fast_accum=self.fast_accum)

    def quantize_and_compute(self, x, wq, w_scale):
        xq, wq, x_scale, w_scale = self.quantize(x, wq, w_scale)
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
class FP8RowwisePreshuffleGemm(FP8RowwiseGemm):
    """
    FP8 matmul with rowwise scaling and preshuffling of input B.
    """

    def __init__(self):
        self.fast_accum = True
        if self.supported:
            self.gemm_op = torch.ops.fbgemm.f8f8bf16_rowwise_preshuffle

    def preprocess(self, x, w):
        x, wq, w_scale = super().preprocess(x, w)
        return x, ck_preshuffle(wq, 16), w_scale

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_rowwise_preshuffle"
        else:
            return "ck_rowwise_preshuffle"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        # Not yet supported on nvidia.
        return False


@register_quantize_op
class FP8RowwiseGroupedGemm(QuantizeOpBase):
    """
    FP8 grouped matmul with rowwise scaling.
    """

    def preprocess(self, x, w):
        # Apply sparsity to inputs if appropriate.
        # First check if N and K are fixed.
        m_values = [i.shape[0] for i in x]
        n_values = [i.shape[0] for i in w]
        k_values = [i.shape[1] for i in w]
        # If so, do specialized version of initialization.
        if len(np.unique(n_values)) == 1 and len(np.unique(k_values)) == 1:
            m_values = [i.shape[0] for i in x]
            # Inputs for fixed nk mode must be contiguous, however in the benchmark
            # script they typically are not. Do a little special processing to make them
            # work. In practice this wont be needed.
            # Start by padding along m dimension with zeros.
            max_m = max(m_values)
            x = [
                torch.nn.functional.pad(i, (0, 0, 0, max_m - i.shape[0]), value=0)
                for i in x
            ]
            # Stack inputs into groups.
            x = torch.stack(x).contiguous()
            w = torch.stack(w).contiguous()

            # Preapply weight quantization.
            wq, w_scale = quantize_fp8_row(w)
            # Return processed tensors.
            return (
                x,
                wq,
                w_scale,
                torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device),
            )
        # Otherwise run without sparsity.
        wq, w_scale = zip(*[quantize_fp8_row(i) for i in w])
        return x, wq, w_scale, None

    def quantize(self, x, wq, w_scale, m_values=None):
        # Handle case where inputs are explicitly grouped and non-sparse.
        if isinstance(x, (tuple, list)):
            xq, x_scale = zip(*[triton_quantize_fp8_row(i) for i in x])
            return xq, wq, x_scale, w_scale, m_values
        # Otherwise inputs are unified tensors and sparse.
        else:
            B = x.shape[0]
            xq, x_scale = triton_quantize_fp8_row(x, zero_start_index_M=m_values)
            x_scale = x_scale.view(B, -1)
            return xq, wq, x_scale, w_scale, m_values

    def compute(self, xq, wq, x_scale, w_scale, m_values):
        if m_values is None:
            return torch.ops.fbgemm.f8f8bf16_rowwise_grouped(
                xq,
                wq,
                x_scale,
                w_scale,
            )
        else:
            # Break tensor into groups, simulates what is done e2e.
            return torch.ops.fbgemm.f8f8bf16_rowwise_grouped_dynamic(
                xq,
                wq,
                x_scale,
                w_scale,
                zero_start_index_M=m_values,
            )

    def quantize_and_compute(self, x, wq, w_scale, m_values=None):
        xq, wq, x_scale, w_scale, m_values = self.quantize(x, wq, w_scale, m_values)
        return self.compute(xq, wq, x_scale, w_scale, m_values)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_rowwise_grouped"
        else:
            return "ck_rowwise_grouped"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class BF16TritonStackedGroupedGemm(QuantizeOpBase):
    """
    BF16 grouped matmul with stacked inputs implemented with triton.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int32, device=x[0].device)
        w = torch.concat(w, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, w, m_sizes

    def quantize(self, x, w, m_sizes):
        return x, w, m_sizes

    def compute(self, x, w, m_sizes):
        return grouped_gemm(x, w, m_sizes, _use_warp_specialization=True)

    def quantize_and_compute(self, x, w, m_sizes):
        x, w, m_sizes = self.quantize(x, w, m_sizes)
        return self.compute(x, w, m_sizes)

    @property
    def name(self) -> str:
        return "triton_bf16_grouped_stacked"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class BF16TritonStackedGroupedGemmFuseScatterAdd(BF16TritonStackedGroupedGemm):
    """
    BF16 grouped matmul with stacked inputs implemented with triton. Fused with ScatterAdd.
    """

    def preprocess(self, x, w):
        x, w, m_sizes = super().preprocess(x, w)
        M = x.shape[0]
        N = w.shape[0] // m_sizes.shape[0]
        output = torch.zeros(M, N, dtype=torch.bfloat16, device=x.device)
        indices = torch.randperm(M, dtype=torch.int32, device=x.device)
        return x, w, m_sizes, output, indices

    def quantize(self, x, w, m_sizes, *args):
        return *super().quantize(x, w, m_sizes), *args

    def compute(self, x, w, m_sizes, output, indices):
        return grouped_gemm(
            x,
            w,
            m_sizes,
            _use_warp_specialization=True,
            _output_tensor=output,
            _scatter_add_indices=indices,
        )

    def quantize_and_compute(self, x, w, m_sizes, *args):
        x, w, m_sizes, *ret = self.quantize(x, w, m_sizes, *args)
        return self.compute(x, w, m_sizes, *ret)

    @property
    def name(self) -> str:
        return "triton_bf16_grouped_stacked_fuse_scatter_add"


@register_quantize_op
class FP8TritonStackedGroupedGemm(QuantizeOpBase):
    """
    FP8 grouped matmul with rowwise scaling and stacked inputs implemented with triton.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int32, device=x[0].device)
        # Quantize weights.
        wq, w_scale = zip(*[quantize_fp8_row(i) for i in w])
        # Group weights as single tensor.
        wq = torch.concat(wq, dim=0).contiguous()
        w_scale = torch.concat(w_scale, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, w_scale, m_sizes

    def quantize(self, x, wq, w_scale, m_sizes):
        B = x.shape[0]
        xq, x_scale = triton_quantize_fp8_row(x)
        x_scale = x_scale.view(B, -1)
        return xq, wq, x_scale, w_scale, m_sizes

    def compute(self, xq, wq, x_scale, w_scale, m_sizes):
        return grouped_gemm_fp8_rowwise(
            xq, wq, m_sizes, x_scale, w_scale, _use_warp_specialization=True
        )

    def quantize_and_compute(self, x, wq, w_scale, m_sizes):
        xq, wq, x_scale, w_scale, m_sizes = self.quantize(x, wq, w_scale, m_sizes)
        return self.compute(xq, wq, x_scale, w_scale, m_sizes)

    @property
    def name(self) -> str:
        return "triton_grouped_stacked"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class FP8TritonStackedGroupedGemmFuseScatterAdd(FP8TritonStackedGroupedGemm):
    """
    FP8 grouped matmul with stacked inputs implemented with triton. Fused with ScatterAdd.
    """

    def preprocess(self, x, w):
        x, wq, w_scale, m_sizes = super().preprocess(x, w)
        M = x.shape[0]
        N = wq.shape[0] // m_sizes.shape[0]
        output = torch.zeros(M, N, dtype=torch.bfloat16, device=x.device)
        indices = torch.randperm(M, dtype=torch.int32, device=x.device)
        return x, wq, w_scale, m_sizes, output, indices

    def quantize(self, x, wq, w_scale, m_sizes, *args):
        return *super().quantize(x, wq, w_scale, m_sizes), *args

    def compute(self, xq, wq, x_scale, w_scale, m_sizes, output, indices):
        return grouped_gemm_fp8_rowwise(
            xq,
            wq,
            m_sizes,
            x_scale,
            w_scale,
            _use_warp_specialization=True,
            _output_tensor=output,
            _scatter_add_indices=indices,
        )

    def quantize_and_compute(self, x, wq, w_scale, m_sizes, *args):
        xq, wq, x_scale, w_scale, m_sizes, *ret = self.quantize(
            x, wq, w_scale, m_sizes, *args
        )
        return self.compute(xq, wq, x_scale, w_scale, m_sizes, *ret)

    @property
    def name(self) -> str:
        return "triton_grouped_stacked_fuse_scatter_add"


@register_quantize_op
class DeepGemmStacked(QuantizeOpBase):
    """
    FP8 grouped matmul with blockwise scaling implemented with DeepGemm.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        indices = torch.arange(len(m_values))
        m_indices = indices.repeat_interleave(torch.tensor(m_values)).to(
            device=x[0].device, dtype=torch.int
        )
        # Quantize weights.
        wq, w_scale = zip(*[quantize_fp8_block(i, block_k=128, block_m=128) for i in w])
        # Group weights as single tensor.
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, w_scale, m_indices

    def quantize(self, x, wq, w_scale, m_indices):
        xq, x_scale = quantize_fp8_block(x, block_m=1, block_k=128)
        # Pretranspose scales to deepgemm format.
        x_scale = get_col_major_tma_aligned_tensor(x_scale)
        return xq, wq, x_scale, w_scale, m_indices

    def compute(self, xq, wq, x_scale, w_scale, m_indices):
        # Preallocate output.
        out = torch.empty(
            [xq.shape[0], wq.shape[1]], device=xq.device, dtype=torch.bfloat16
        )
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (xq, x_scale), (wq, w_scale), out, m_indices
        )
        return out

    def quantize_and_compute(self, x, wq, w_scale, m_indices):
        xq, wq, x_scale, w_scale, m_indices = self.quantize(x, wq, w_scale, m_indices)
        return self.compute(xq, wq, x_scale, w_scale, m_indices)

    @property
    def name(self) -> str:
        return "deepgemm_stacked"

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return DEEPGEMM_ENABLED


@register_quantize_op
class DeepGemmMaskedStacked(DeepGemmStacked):
    def preprocess(self, x, w):
        # Quantize weights.
        wq, w_scale = zip(*[quantize_fp8_block(i, block_k=128, block_m=128) for i in w])
        # Group weights as single tensor.
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()

        # Also view input as flattened.
        m_values = [i.shape[0] for i in x]
        expected_m = max(m_values)
        padded_m_max = ((max(m_values) + 127) // 128) * 128
        masked_m = torch.tensor(m_values).to(dtype=torch.int32, device=x[0].device)

        num_groups = len(m_values)
        k = x[0].shape[1]
        x_padded = torch.zeros(
            [num_groups, padded_m_max, k], device=x[0].device, dtype=x[0].dtype
        )
        for g in range(num_groups):
            x_padded[g, : m_values[g], :] = x[g]

        # Return processed tensors.
        return x_padded, wq, w_scale, masked_m, expected_m, m_values

    def quantize(self, x, wq, w_scale, masked_m, expected_m, m_values):
        g, m_max, k = x.shape
        xq, x_scale = quantize_fp8_block(x.view(-1, k), block_m=1, block_k=128)
        # Pretranspose scales to deepgemm format.
        x_scale = get_col_major_tma_aligned_tensor(x_scale)
        return (
            xq.view(g, m_max, -1),
            wq,
            x_scale.view(g, m_max, -1),
            w_scale,
            masked_m,
            expected_m,
            m_values,
        )

    def compute(self, xq, wq, x_scale, w_scale, masked_m, expected_m, m_values):
        # Preallocate output.
        out = torch.empty(
            [xq.shape[0], xq.shape[1], wq.shape[1]],
            device=xq.device,
            dtype=torch.bfloat16,
        )
        m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            (xq, x_scale), (wq, w_scale), out, masked_m, expected_m
        )
        num_groups = xq.shape[0]
        out_list = [out[g, : m_values[g], :] for g in range(num_groups)]
        return out_list

    def quantize_and_compute(self, x, wq, w_scale, masked_m, expected_m, m_values):
        xq, wq, x_scale, w_scale, masked_m, expected_m = self.quantize(
            x, wq, w_scale, masked_m, expected_m, m_values
        )
        return self.compute(xq, wq, x_scale, w_scale, masked_m, expected_m, m_values)

    @property
    def name(self) -> str:
        return "deepgemm_masked_stacked"


@register_quantize_op
class DeepGemmBlockwise(QuantizeOpBase):
    """
    FP8 matmul with blockwise scaling implemented with DeepGemm.
    """

    def preprocess(self, x, w):
        # Quantize weights.
        wq, w_scale = quantize_fp8_block(w, block_m=128, block_k=128)
        # allocate output.
        out = torch.empty(
            x.shape[0], wq.shape[0], device=x.device, dtype=torch.bfloat16
        )
        # Return processed tensors.
        return x, wq, w_scale, out

    def quantize(self, x, wq, w_scale, out):
        xq, x_scale = quantize_fp8_block(x, block_m=1, block_k=128)
        # Pretranspose scales to deepgemm format.
        x_scale = get_col_major_tma_aligned_tensor(x_scale)
        return xq, wq, x_scale, w_scale, out

    def compute(self, xq, wq, x_scale, w_scale, out):
        gemm_fp8_fp8_bf16_nt((xq, x_scale), (wq, w_scale), out)
        return out

    def quantize_and_compute(self, x, wq, w_scale, out):
        xq, wq, x_scale, w_scale, out = self.quantize(x, wq, w_scale, out)
        return self.compute(xq, wq, x_scale, w_scale, out)

    @property
    def name(self) -> str:
        return "deepgemm_blockwise"

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class DeepGemmRowwise(QuantizeOpBase):
    """
    FP8 matmul with rowwise scaling implemented with DeepGemm.
    """

    def preprocess(self, x, w):
        # Quantize weights.
        wq, w_scale = quantize_fp8_row(w)
        # allocate output.
        out = torch.empty(
            x.shape[0], wq.shape[0], device=x.device, dtype=torch.bfloat16
        )
        # Return processed tensors.
        return x, wq, w_scale, out

    def quantize(self, x, wq, w_scale, out):
        xq, x_scale = quantize_fp8_row(x)
        # Pretranspose scales to deepgemm format.
        x_scale = get_col_major_tma_aligned_tensor(x_scale, rowwise_scaling=True)
        return xq, wq, x_scale, w_scale, out

    def compute(self, xq, wq, x_scale, w_scale, out):
        gemm_fp8_fp8_bf16_nt((xq, x_scale), (wq, w_scale), out)
        return out

    def quantize_and_compute(self, x, wq, w_scale, out):
        xq, wq, x_scale, w_scale, out = self.quantize(x, wq, w_scale, out)
        return self.compute(xq, wq, x_scale, w_scale, out)

    @property
    def name(self) -> str:
        return "deepgemm_rowwise"

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return DEEPGEMM_ENABLED


@register_quantize_op
class FP8StackedGroupedGemm(QuantizeOpBase):
    """
    FP8 grouped matmul with rowwise scaling and stacked inputs.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        # Quantize weights.
        wq, w_scale = zip(*[quantize_fp8_row(i) for i in w])
        # Group weights as single tensor.
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, w_scale, m_sizes

    def quantize(self, x, wq, w_scale, m_sizes):
        B = x.shape[0]
        xq, x_scale = triton_quantize_fp8_row(x)
        x_scale = x_scale.view(B, -1)
        return xq, wq, x_scale, w_scale, m_sizes

    def compute(self, xq, wq, x_scale, w_scale, m_sizes):
        return torch.ops.fbgemm.f8f8bf16_rowwise_grouped_stacked(
            xq, wq, x_scale, w_scale, m_sizes
        )

    def quantize_and_compute(self, x, wq, w_scale, m_sizes):
        xq, wq, x_scale, w_scale, m_sizes = self.quantize(x, wq, w_scale, m_sizes)
        return self.compute(xq, wq, x_scale, w_scale, m_sizes)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_grouped_stacked"
        else:
            return "ck_grouped_stacked"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class BF16GroupedGemm(QuantizeOpBase):
    """
    BF16 grouped matmul implemented with CK or Cutlass.
    """

    def preprocess(self, x, w):
        # Apply sparsity to inputs if appropriate.
        # First check if N and K are fixed.
        m_values = [i.shape[0] for i in x]
        n_values = [i.shape[0] for i in w]
        k_values = [i.shape[1] for i in w]
        # If so, do specialized version of initialization.
        if len(np.unique(n_values)) == 1 and len(np.unique(k_values)) == 1:
            m_values = [i.shape[0] for i in x]
            # Inputs for fixed nk mode must be contiguous, however in the benchmark
            # script they typically are not. Do a little special processing to make them
            # work. In practice this wont be needed.
            # Start by padding along m dimension with zeros.
            max_m = max(m_values)
            x = [
                torch.nn.functional.pad(i, (0, 0, 0, max_m - i.shape[0]), value=0)
                for i in x
            ]
            # Stack inputs into groups.
            x = torch.stack(x).contiguous()
            w = torch.stack(w).contiguous()
            return (
                x,
                w,
                torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device),
            )
        return x, w, None

    def quantize(self, x, w, m_values=None):
        # No action required.
        return x, w, m_values

    def compute(self, x, w, m_values):
        if m_values is None:
            return torch.ops.fbgemm.bf16bf16bf16_grouped(x, w)
        else:
            return torch.ops.fbgemm.bf16bf16bf16_grouped_dynamic(x, w, m_values)

    def quantize_and_compute(self, x, w, m_values):
        return self.compute(x, w, m_values)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_bf16_grouped"
        else:
            return "ck_bf16_grouped"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class FP8RowwiseBatchedGemm(QuantizeOpBase):
    """
    FP8 batched matmul with rowwise scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.fbgemm.f8f8bf16_rowwise_batched(xq, wq, x_scale, w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_rowwise_batched"
        else:
            return "ck_rowwise_batched"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class FP8LiteGemm(QuantizeOpBase):
    """
    FP8 lite matmul for memory bound.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(x)
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_tensor(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.fbgemm.f8f8bf16_lite(xq, wq, x_scale * w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cuda_lite"

    @property
    def hip(self) -> bool:
        # Need to add support for better quantize kernel.
        # Also may have an issue with cuda graphs.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class TritonFP8RowwiseGemm(QuantizeOpBase):
    """
    FP8 matmul with rowwise scaling implemented with Triton.
    """

    def __init__(self):
        self.fast_accum = True

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        bias = torch.randn(w.shape[0], device=x.device, dtype=torch.float32)
        return xq, wq, x_scale, w_scale, bias

    def compute(self, xq, wq, x_scale, w_scale, bias):
        return matmul_fp8_row(
            xq,
            wq,
            x_scale,
            w_scale,
            bias=bias,
            fp8_fast_accum=self.fast_accum,
            use_warp_specialization=True,
        )

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


####################################################################################################
# CUTLASS kernel v2
####################################################################################################


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
        if hasattr(torch.ops.cutlass_extensions, "f8f8bf16"):
            return torch.ops.cutlass_extensions.f8f8bf16(xq, wq, x_scale * w_scale)
        else:
            raise RuntimeError(
                "Skipping cutlass_extensions_v2 runs as it is not supported"
            )

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


# CUTLASS kernel v2
@register_quantize_op
class CutlassFP8RowwiseGemm_v2(QuantizeOpBase):
    """
    FP8 matmul with rowwise scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        if hasattr(torch.ops.cutlass_extensions, "f8f8bf16_rowwise"):
            return torch.ops.cutlass_extensions.f8f8bf16_rowwise(
                xq, wq, x_scale, w_scale
            )
        else:
            raise RuntimeError(
                "Skipping cutlass_extensions_v2 runs as it is not supported"
            )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cutlass_rowwise_v2"

    @property
    def hip(self) -> bool:
        # Need to add support for better quantize kernel.
        # Also may have an issue with cuda graphs.
        return False

    @property
    def cuda(self) -> bool:
        return True


####################################################################################################


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
class F8I4ShuffledGemm(QuantizeOpBase):
    def preprocess(self, x, w):
        # Prequantize and pack weights.
        wq, (group_scale, row_scale) = quantize_int4_preshuffle(w)
        return x, wq, row_scale, group_scale

    def quantize(self, x, wq, row_scale, group_scale):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        return xq, wq, x_scale, row_scale, group_scale

    def compute(self, xq, wq, x_scale, row_scale, group_scale):
        # Handle batched cases by looping over each batch.
        if xq.dim() == 3:
            B, M, _ = xq.shape
            _, N, _ = wq.shape
            y = torch.empty((B, M, N), device=xq.device, dtype=torch.bfloat16)
            for i in range(B):
                y[i] = torch.ops.fbgemm.f8i4bf16_shuffled(
                    xq[i], wq[i], x_scale[i], row_scale[i], group_scale[i]
                )
            return y
        # Otherwise run gemm normally.
        return torch.ops.fbgemm.f8i4bf16_shuffled(
            xq, wq, x_scale, row_scale, group_scale
        )

    def quantize_and_compute(self, x, wq, row_scale, group_scale):
        xq, wq, x_scale, row_scale, group_scale = self.quantize(
            x, wq, row_scale, group_scale
        )
        return self.compute(xq, wq, x_scale, row_scale, group_scale)

    @property
    def name(self) -> str:
        return "cutlass_f8i4_preshuffle"

    @property
    def hip(self) -> bool:
        # Not yet supported on AMD.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class BF16I4ShuffledGemm(QuantizeOpBase):
    def preprocess(self, x, w):
        # Prequantize and pack weights.
        wq, (group_scale, group_zero) = quantize_int4_preshuffle(w, dtype="bf16")
        return x, wq, group_scale, group_zero

    def quantize(self, x, wq, group_scale, group_zero):
        # No extra action required.
        return x, wq, group_scale, group_zero

    def compute(self, x, wq, group_scale, group_zero):
        # Handle batched cases by looping over each batch.
        if x.dim() == 3:
            B, M, _ = x.shape
            _, N, _ = wq.shape
            y = torch.empty((B, M, N), device=x.device, dtype=torch.bfloat16)
            for i in range(B):
                y[i] = torch.ops.fbgemm.bf16i4bf16_shuffled(
                    x[i], wq[i], group_scale[i], group_zero[i]
                )
            return y
        # Otherwise run gemm normally.
        return torch.ops.fbgemm.bf16i4bf16_shuffled(x, wq, group_scale, group_zero)

    def quantize_and_compute(self, x, wq, group_scale, group_zero):
        x, wq, group_scale, group_zero = self.quantize(x, wq, group_scale, group_zero)
        return self.compute(x, wq, group_scale, group_zero)

    @property
    def name(self) -> str:
        return "cutlass_bf16i4_preshuffle"

    @property
    def hip(self) -> bool:
        # Not yet supported on AMD.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class F8I4ShuffledGroupedGemm(QuantizeOpBase):
    """
    FP8 x Int4 mixed dtype grouped gemm with preshuffling.
    """

    def preprocess(self, x, w):
        assert isinstance(x, list) and isinstance(
            w, list
        ), "Only supported for grouped inputs."
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int32, device=x[0].device)
        # Quantize weights.
        wq, scales = zip(*[quantize_int4_preshuffle(i) for i in w])
        group_scale, row_scale = zip(*scales)
        # Group weights as single tensor.
        wq = torch.stack(wq, dim=0).contiguous()
        row_scale = torch.stack(row_scale, dim=0).contiguous()
        group_scale = torch.stack(group_scale, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, row_scale, group_scale, m_sizes

    def quantize(self, x, wq, row_scale, group_scale, m_sizes):
        B = x.shape[0]
        xq, x_scale = triton_quantize_fp8_row(x)
        x_scale = x_scale.view(B, -1)
        return xq, wq, x_scale, row_scale, group_scale, m_sizes

    def compute(self, xq, wq, x_scale, row_scale, group_scale, m_sizes):
        out = torch.ops.fbgemm.f8i4bf16_shuffled_grouped(
            xq, wq, x_scale, row_scale, group_scale, m_sizes
        )
        return out

    def quantize_and_compute(self, x, wq, row_scale, group_scale, m_sizes):
        xq, wq, x_scale, row_scale, group_scale, m_sizes = self.quantize(
            x, wq, row_scale, group_scale, m_sizes
        )
        return self.compute(xq, wq, x_scale, row_scale, group_scale, m_sizes)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_f8i4_grouped_preshuffle"
        else:
            return "ck_f8i4_grouped_preshuffle"

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class BF16I4ShuffledGroupedGemm(QuantizeOpBase):
    """
    BF16 x Int4 mixed dtype grouped gemm with preshuffling.
    """

    def preprocess(self, x, w):
        assert isinstance(x, list) and isinstance(
            w, list
        ), "Only supported for grouped inputs."
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int32, device=x[0].device)
        # Quantize weights.
        wq, scales = zip(
            *[quantize_int4_preshuffle(i, dtype="bf16", use_zp=False) for i in w]
        )
        # Group weights as single tensor.
        group_scale, group_zero = zip(*scales)
        wq = torch.stack(wq, dim=0).contiguous()
        group_scale = torch.stack(group_scale, dim=0).contiguous()
        group_zero = torch.stack(group_zero, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, group_scale, group_zero, m_sizes

    def quantize(self, x, wq, group_scale, group_zero, m_sizes):
        return x, wq, group_scale, group_zero, m_sizes

    def compute(self, x, wq, group_scale, group_zero, m_sizes):
        # TODO Zero points arent currently supported in grouped gemm.
        # We leave them as inputs for future compatibility but they are ignored.
        return torch.ops.fbgemm.bf16i4bf16_shuffled_grouped(
            x, wq, group_scale, group_zero, m_sizes
        )

    def quantize_and_compute(self, x, wq, group_scale, group_zero, m_sizes):
        x, wq, group_scale, group_zero, m_sizes = self.quantize(
            x, wq, group_scale, group_zero, m_sizes
        )
        return self.compute(x, wq, group_scale, group_zero, m_sizes)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_bf16i4_grouped_preshuffle"
        else:
            return "ck_bf16i4_grouped_preshuffle"

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class BF16GroupedStacked(QuantizeOpBase):
    """
    BF16 grouped matmul with stacked inputs backed by cutlass or ck.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        # Group weights as single tensor.
        w = torch.stack(w, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, w, m_sizes

    def quantize(self, x, w, m_sizes):
        return x, w, m_sizes

    def compute(self, x, w, m_sizes):
        return torch.ops.fbgemm.bf16bf16bf16_grouped_stacked(x, w, m_sizes)

    def quantize_and_compute(self, x, w, m_sizes):
        x, w, m_sizes = self.quantize(x, w, m_sizes)
        return self.compute(x, w, m_sizes)

    @property
    def name(self) -> str:
        return "bf16_grouped_stacked"

    @property
    def hip(self) -> bool:
        return True

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
        return (
            x.to(torch.bfloat16),
            wq,
            w_scale,
            w_zp,
        )

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
        # Only enabled if import works.
        return TINYGEMM_ENABLED


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


@register_quantize_op
class MacheteBF16I4(QuantizeOpBase):
    """
    Mixed Precision BF16 Activations with Int4 Weights using Machete.
    """

    def quantize(self, x, w):
        # Marlin quantize expects weights in [K, N] layout.
        _, wq, scale, _ = machete_quantize_and_pack(
            w.t().contiguous(), bits=4, groupsize=128
        )
        return x, wq, scale

    def compute(self, x, wq, scale):
        return machete_gemm(x, wq, bits=4, groupsize=128, scales=scale)

    def quantize_and_compute(self, x, w):
        x, wq, scale = self.quantize(x, w)
        return self.compute(x, wq, scale)

    @property
    def name(self) -> str:
        return "machete_bf16i4"

    @property
    def hip(self) -> bool:
        # Machete only supported for cuda.
        return False

    @property
    def cuda(self) -> bool:
        # This op is not always supported.
        return MACHETE_ENABLED


@register_quantize_op
class NVFP4Gemm(QuantizeOpBase):
    """
    NVFP4 matmul with block-wise scaling.
    """

    def quantize(self, x, w):
        x_global_scale = ((448.0 * 6.0) / torch.amax(x.flatten(), dim=-1)).to(
            torch.float32
        )
        w_global_scale = ((448.0 * 6.0) / torch.amax(w.flatten(), dim=-1)).to(
            torch.float32
        )
        global_scale = 1 / (x_global_scale * w_global_scale)

        xq, x_scale = triton_scale_nvfp4_quant(x, x_global_scale)
        wq, w_scale = triton_scale_nvfp4_quant(w, w_global_scale)

        return xq, wq, x_scale, w_scale, global_scale

    def compute(self, xq, wq, x_scale, w_scale, global_scale):
        return torch.ops.fbgemm.f4f4bf16(
            xq, wq, x_scale, w_scale, global_scale=global_scale, use_mx=False
        )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale, global_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale, global_scale=global_scale)

    @property
    def name(self) -> str:
        return "cutlass_nv_f4f4bf16"

    @property
    def hip(self) -> bool:
        # F4F4BF16 only supported for cuda.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class NVFP4Quantize(QuantizeOpBase):
    """
    NVFP4 quantization with block-wise scaling.
    """

    def quantize_rms(self, x, w):
        M, N = w.shape
        group_size = 16
        w = torch.randn(group_size, dtype=torch.bfloat16, device=w.device)
        x_global_scale = torch.tensor([448.0 * 6.0]).to(device=x.device) / torch.amax(
            x.flatten(), dim=-1
        )
        xq_ref, x_scale_ref = triton_scale_nvfp4_quant_rms(
            x,
            w.repeat(M * N // group_size),
            x_global_scale,
            group_size=group_size,
            EPS=1e-5,
        )

        intermediate = rms_norm(x.reshape(-1, 16), w, eps=1e-5)
        intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
        xq, x_scale = triton_scale_nvfp4_quant(
            intermediate,
            x_global_scale,
            group_size=group_size,
        )

    def quantize_silu(self, x, w):
        M, N = x.shape
        group_size = 16
        x_global_scale = torch.tensor([448.0 * 6.0]).to(device=x.device) / torch.amax(
            x.flatten(), dim=-1
        )
        xq_ref, x_scale_ref = triton_scale_nvfp4_quant_silu(
            x,
            w,
            x_global_scale,
            group_size=group_size,
        )

        intermediate = silu_mul(x.reshape(-1, 16), w.reshape(-1, 16))
        intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
        xq, x_scale = triton_scale_nvfp4_quant(
            intermediate,
            x_global_scale,
            group_size=group_size,
        )

    def quantize(self, x, w):
        x_global_scale = ((448.0 * 6.0) / torch.amax(x.flatten(), dim=-1)).to(
            torch.float32
        )
        w_global_scale = ((448.0 * 6.0) / torch.amax(w.flatten(), dim=-1)).to(
            torch.float32
        )
        global_scale = 1 / (x_global_scale * w_global_scale)

        xq, x_scale = triton_scale_nvfp4_quant(x, x_global_scale)
        wq, w_scale = triton_scale_nvfp4_quant(w, w_global_scale)
        return xq, wq, x_scale, w_scale, global_scale

    def compute(self, xq, wq, x_scale, w_scale, global_scale):
        return torch.ops.fbgemm.f4f4bf16(
            xq, wq, x_scale, w_scale, global_scale=global_scale, use_mx=False
        )

    def quantize_and_compute(self, x, w):
        return self.quantize(x, w)

    @property
    def name(self) -> str:
        return "nvfp4_quantize"

    @property
    def hip(self) -> bool:
        # F4F4BF16 only supported for cuda.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class MXFP4Gemm(QuantizeOpBase):
    """
    MXFP4 matmul with block-wise scaling.
    """

    def quantize(self, x, w):
        xq, x_scale = triton_quantize_mx4_unpack(x)
        wq, w_scale = triton_quantize_mx4_unpack(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.fbgemm.f4f4bf16(xq, wq, x_scale, w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cutlass_f4f4bf16"

    @property
    def hip(self) -> bool:
        # F4F4BF16 only supported for cuda.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class MXFP4GroupedGemm(QuantizeOpBase):
    """
    MXFP4 grouped matmul with blockwise scaling.
    """

    def preprocess(self, x, w):
        wq, w_scale = zip(*[triton_quantize_mx4_unpack(i) for i in w])
        return x, wq, w_scale

    def quantize(self, x, wq, w_scale):
        xq, x_scale = zip(*[triton_quantize_mx4_unpack(i) for i in x])
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.fbgemm.f4f4bf16_grouped(
            xq,
            wq,
            x_scale,
            w_scale,
        )

    def quantize_and_compute(self, x, wq, w_scale):
        xq, wq, x_scale, w_scale = self.quantize(x, wq, w_scale)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cutlass_f4f4bf16_grouped"

    @property
    def hip(self) -> bool:
        # F4F4BF16_grouped only supported for cuda.
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class NVFP4GroupedGemm(QuantizeOpBase):
    """
    NVFP4 grouped matmul with blockwise scaling.
    """

    def quantize(self, x, w):
        def get_global_scale(x, w):
            x_global_scale = ((448.0 * 6.0) / torch.amax(x.flatten(), dim=-1)).to(
                torch.float32
            )
            w_global_scale = ((448.0 * 6.0) / torch.amax(w.flatten(), dim=-1)).to(
                torch.float32
            )
            global_scale = 1 / (x_global_scale * w_global_scale)
            return x_global_scale, w_global_scale, global_scale

        # Compute global scale for each group
        G = len(x)
        x_global_scale = []
        w_global_scale = []
        global_scale = []
        for i in range(G):
            x_global_scale_, w_global_scale_, global_scale_ = get_global_scale(
                x[i], w[i]
            )
            x_global_scale.append(x_global_scale_)
            w_global_scale.append(w_global_scale_)
            global_scale.append(global_scale_)

        # Quantize weights and activations
        wq, w_scale = zip(
            *[triton_scale_nvfp4_quant(w[i], w_global_scale[i]) for i in range(G)]
        )
        xq, x_scale = zip(
            *[triton_scale_nvfp4_quant(x[i], x_global_scale[i]) for i in range(G)]
        )
        return xq, wq, x_scale, w_scale, global_scale

    def compute(self, xq, wq, x_scale, w_scale, global_scale):
        return torch.ops.fbgemm.f4f4bf16_grouped(
            xq, wq, x_scale, w_scale, global_scale, use_mx=False
        )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale, global_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale, global_scale)

    @property
    def name(self) -> str:
        return "cutlass_nv_f4f4bf16_grouped"

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class MXFP4StackedGroupedGemm(QuantizeOpBase):
    """
    MXFP4 grouped matmul with blockwise scaling and stacked inputs.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        wq, w_scale = zip(*[triton_quantize_mx4_unpack(i) for i in w])
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()
        return x, wq, w_scale, m_sizes

    def quantize(self, x, wq, w_scale, m_sizes):
        xq, x_scale = zip(*[triton_quantize_mx4_unpack(i) for i in x])
        xq = torch.stack(xq, dim=0).contiguous()
        x_scale = torch.stack(x_scale, dim=0).contiguous()
        xq = xq.view(-1, xq.shape[-1])
        return xq, wq, x_scale, w_scale, m_sizes

    def compute(self, xq, wq, x_scale, w_scale, m_sizes):
        return torch.ops.fbgemm.f4f4bf16_grouped_stacked(
            xq, wq, x_scale, w_scale, m_sizes
        )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale, m_sizes = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale, m_sizes)

    @property
    def name(self) -> str:
        return "cutlass_f4f4bf16_grouped_stacked"

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_quantize_op
class NVFP4StackedGroupedGemm(QuantizeOpBase):
    """
    NVFP4 grouped matmul with blockwise scaling and stacked inputs.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        return x, w, m_sizes

    def quantize(self, x, w, m_sizes):
        def get_global_scale(x, w):

            w_global_scale = ((448.0 * 6.0) / torch.amax(w.flatten(), dim=-1)).to(
                torch.float32
            )

            if x.shape[0] != 0:
                x_global_scale = ((448.0 * 6.0) / torch.amax(x.flatten(), dim=-1)).to(
                    torch.float32
                )
            else:
                x_global_scale = w_global_scale

            if x.shape[0] != 0:
                global_scale = 1 / (x_global_scale * w_global_scale)
            else:
                global_scale = 1 / (w_global_scale)

            return x_global_scale, w_global_scale, global_scale

        # Compute global scale for each group
        G = len(x)
        x_global_scale = []
        w_global_scale = []
        global_scale = []
        for i in range(G):
            x_global_scale_, w_global_scale_, global_scale_ = get_global_scale(
                x[i], w[i]
            )
            x_global_scale.append(x_global_scale_)
            w_global_scale.append(w_global_scale_)
            global_scale.append(global_scale_)

        wq, w_scale = zip(
            *[triton_scale_nvfp4_quant(w[i], w_global_scale[i]) for i in range(G)]
        )
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()

        xq = []
        x_scale = []
        for i in range(G):
            if x[i].shape[0] != 0:
                o_a, o_b = triton_scale_nvfp4_quant(x[i], x_global_scale[i])
                xq.append(o_a)
                o_b = o_b.reshape(-1, x[i].shape[1] // 16)
                x_scale.append(o_b)
        xq = torch.vstack(xq).contiguous()
        x_scale = torch.vstack(x_scale).contiguous()

        x = torch.concat(x, dim=0).contiguous()
        global_scale = torch.stack(global_scale, dim=0).contiguous()
        return xq, wq, x_scale, w_scale, m_sizes, global_scale

    def compute(self, xq, wq, x_scale, w_scale, m_sizes, global_scale):
        return torch.ops.fbgemm.f4f4bf16_grouped_stacked(
            xq, wq, x_scale, w_scale, m_sizes, global_scale, use_mx=False
        )

    def quantize_and_compute(self, x, w, m_sizes):
        xq, wq, x_scale, w_scale, m_sizes, global_scale = self.quantize(x, w, m_sizes)
        return self.compute(xq, wq, x_scale, w_scale, m_sizes, global_scale)

    @property
    def name(self) -> str:
        return "cutlass_nv_f4f4bf16_grouped_stacked"

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return True

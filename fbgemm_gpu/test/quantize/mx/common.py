# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Microsoft Corporation.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import struct
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# -------------------------------------------------------------------------
# Helper funcs
# -------------------------------------------------------------------------

FP32_EXPONENT_BIAS: int = 127
FP32_MIN_NORMAL: float = 2 ** (-FP32_EXPONENT_BIAS + 1)


# Enum for rounding modes
class RoundingMode(IntEnum):
    nearest = 0
    floor = 1
    even = 2

    @staticmethod
    def string_enums() -> List[str]:
        return [s.name for s in list(RoundingMode)]


# Enum for scalar data formats
class ElemFormat(Enum):
    int8 = 1
    int4 = 2
    int2 = 3
    fp8_e5m2 = 4
    fp8_e4m3 = 5
    fp6_e3m2 = 6
    fp6_e2m3 = 7
    fp4 = 8
    fp4_e2m1 = 8
    float16 = 9
    fp16 = 9
    bfloat16 = 10
    bf16 = 10

    @staticmethod
    def from_str(s: str) -> int:
        assert s is not None, "String elem_format == None"
        s = s.lower()
        if hasattr(ElemFormat, s):
            return getattr(ElemFormat, s)
        else:
            raise Exception("Undefined elem format", s)


def _get_min_norm(ebits: int) -> int:
    """Valid for all float formats"""
    emin = 2 - (2 ** (ebits - 1))
    return 0 if ebits == 0 else 2**emin


def _get_max_norm(ebits: int, mbits: int) -> float:
    """Valid only for floats that define NaN"""
    assert ebits >= 5, "invalid for floats that don't define NaN"
    emax = 0 if ebits == 0 else 2 ** (ebits - 1) - 1
    return 2**emax * float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)


_FORMAT_CACHE: Dict[ElemFormat, Tuple[int, int, int, float, float]] = {}


def _get_format_params(  # noqa
    fmt: Union[ElemFormat, str, None],
) -> Tuple[int, int, int, float, float]:
    """Allowed formats:
    - intX:         2 <= X <= 32, assume sign-magnitude, 1.xxx representation
    - floatX/fpX:   16 <= X <= 28, assume top exp is used for NaN/Inf
    - bfloatX/bfX:  9 <= X <= 32
    - fp4,                  no NaN/Inf
    - fp6_e3m2/e2m3,        no NaN/Inf
    - fp8_e4m3/e5m2,        e5m2 normal NaN/Inf, e4m3 special behavior

    Returns:
      ebits: exponent bits
      mbits: mantissa bits: includes sign and implicit bits
      emax: max normal exponent
      max_norm: max normal number
      min_norm: min normal number
    """

    if type(fmt) is str:
        fmt = ElemFormat.from_str(fmt)

    if fmt in _FORMAT_CACHE:
        return _FORMAT_CACHE[fmt]

    if fmt == ElemFormat.int8:
        ebits, mbits = 0, 8
        emax = 0
    elif fmt == ElemFormat.int4:
        ebits, mbits = 0, 4
        emax = 0
    elif fmt == ElemFormat.int2:
        ebits, mbits = 0, 2
        emax = 0
    elif fmt == ElemFormat.fp8_e5m2:
        ebits, mbits = 5, 4
        emax = 2 ** (ebits - 1) - 1
    elif fmt == ElemFormat.fp8_e4m3:
        ebits, mbits = 4, 5
        emax = 2 ** (ebits - 1)
    elif fmt == ElemFormat.fp6_e3m2:
        ebits, mbits = 3, 4
        emax = 2 ** (ebits - 1)
    elif fmt == ElemFormat.fp6_e2m3:
        ebits, mbits = 2, 5
        emax = 2 ** (ebits - 1)
    elif fmt == ElemFormat.fp4:
        ebits, mbits = 2, 3
        emax = 2 ** (ebits - 1)
    elif fmt == ElemFormat.float16:
        ebits, mbits = 5, 12
        emax = 2 ** (ebits - 1) - 1
    elif fmt == ElemFormat.bfloat16:
        ebits, mbits = 8, 9
        emax = 2 ** (ebits - 1) - 1
    else:
        raise Exception("Unknown element format %s" % fmt)

    if fmt != ElemFormat.fp8_e4m3:
        max_norm = 2**emax * float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)
    else:
        max_norm = 2**emax * 1.75  # FP8 has custom max_norm

    min_norm = _get_min_norm(ebits)

    # pyre-fixme[6]: For 1st argument expected `ElemFormat` but got `Union[None,
    #  ElemFormat, int]`.
    _FORMAT_CACHE[fmt] = (ebits, mbits, emax, max_norm, min_norm)

    return ebits, mbits, emax, max_norm, min_norm


def _reshape_to_blocks(
    A: torch.Tensor, axes: List[int], block_size: int
) -> Tuple[torch.Tensor, List[int], torch.Size, torch.Size]:
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for _ in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape: List[int], reshape_block_size: int) -> List[int]:
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape


def _undo_reshape_to_blocks(
    A: torch.Tensor, padded_shape: torch.Size, orig_shape: torch.Size, axes: List[int]
) -> torch.Tensor:
    # Undo tile reshaping
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A


def get_s_e_m(value_in_float: float) -> Tuple[int, int, int]:
    def float_to_bits(value_in_float: float) -> int:
        s = struct.pack("@f", value_in_float)
        return struct.unpack("@I", s)[0]

    bits = float_to_bits(value_in_float)  # bits in form of uint32
    sign = (bits & 0x80000000) >> 31  # sign bit
    exp = ((bits & 0x7F800000) >> 23) - 127  # exponent
    mant = bits & 0x007FFFFF
    return sign, exp, mant


def all_encodings(
    _e: int,
    _m: int,
    device: torch.device,
    encodes_infs: bool = True,
) -> torch.Tensor:
    _CACHE = {}
    if (_e, _m, encodes_infs) in _CACHE:
        x = _CACHE[(_e, _m, encodes_infs)]
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    # Holds all positive and negative
    x = np.zeros((2 ** (_e + _m + 1)), dtype=np.float32)
    for _i in range(2 ** (_e + _m)):
        if _e > 0:
            _exp = _i >> _m
            # Skip exp == all ones
            if encodes_infs and _exp == 2**_e - 1:
                continue
            # Normal or subnormal encoding
            if _exp == 0:
                _exp = 1 - (2 ** (_e - 1) - 1)
                _explicit = 0.0
            else:
                _exp -= 2 ** (_e - 1) - 1
                _explicit = 1.0
            # Obtain mantissa value
            _mant = _i & ((2**_m) - 1)
            _mmant = _mant / (2**_m)

            # FP8 e4m3 hack
            if _e == 4 and _m == 3 and _exp == 8 and _mmant == 0.875:
                _value = 0
            else:
                _value = 2 ** (_exp) * (_explicit + _mmant)
        else:
            _value = _i / (2 ** (_m - 1))

        x[_i] = _value
        x[_i + 2 ** (_e + _m)] = -_value

    _CACHE[(_e, _m, encodes_infs)] = x

    return torch.as_tensor(x, dtype=torch.float32, device=device)


# -------------------------------------------------------------------------
# Helper funcs for test
# -------------------------------------------------------------------------


def check_diff_quantize(
    x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    tol: int = 0,
    ntol: int = 0,
    handle_infs: bool = False,
) -> bool:
    """In floating-point x==y with inf on both sides returns NaN.
    If handle_infs is True, then we allow inf==inf to pass.
    """
    __tracebackhide__ = True

    # Check shapes
    if y1.size() != y2.size():
        print("Size mismatch: ", list(y1.size()), "!=", list(y2.size()))
        raise IndexError

    # Convert to numpy
    # pyre-fixme[9]: x has type `Tensor`; used as `Union[ndarray, Tensor]`.
    x = np.array(x) if type(x) is list else x
    # pyre-fixme[9]: x has type `Tensor`; used as `Union[ndarray[Any, Any], Tensor]`.
    x = x.cpu().numpy() if type(x) is torch.Tensor else x
    y1 = y1.detach().cpu().numpy()
    y2 = y2.detach().cpu().numpy()

    torch_infs = np.isinf(y1) | np.isnan(y1)
    cuda_infs = np.isinf(y2) | np.isnan(y2)
    y1_ = np.where(torch_infs, 0.0, y1)
    y2_ = np.where(cuda_infs, 0.0, y2)
    diff = abs(y1_ - y2_)

    # Don't compare infs if requested
    if not handle_infs:
        torch_infs = None
        cuda_infs = None

    # Check for differences
    max_diff = np.max(diff)
    ndiff = np.sum(diff > tol)  # num of violations
    if (max_diff > tol and ndiff > ntol) or np.any(torch_infs != cuda_infs):
        where_diff = (diff != 0) | (torch_infs != cuda_infs)
        print("%d/%d mismatches" % (np.count_nonzero(where_diff), where_diff.size))
        print("First and last mismatch:")
        # pyre-fixme[6]: For 1st argument expected `float` but got `Tensor`.
        print("Orig:", x[where_diff][0], get_s_e_m(x[where_diff][0]))
        print("y1:  ", y1[where_diff][0], get_s_e_m(y1[where_diff][0]))
        print("y2:  ", y2[where_diff][0], get_s_e_m(y2[where_diff][0]))
        if np.count_nonzero(where_diff) > 1:
            print("--------------------")
            # pyre-fixme[6]: For 1st argument expected `float` but got `Tensor`.
            print("Orig:", x[where_diff][-1], get_s_e_m(x[where_diff][-1]))
            print("y1:  ", y1[where_diff][-1], get_s_e_m(y1[where_diff][-1]))
            print("y2:  ", y2[where_diff][-1], get_s_e_m(y2[where_diff][-1]))
        # raise ValueError
        return False
    return True


# -------------------------------------------------------------------------
# Helper funcs for quantization
# -------------------------------------------------------------------------


# Never explicitly compute 2**(-exp) since subnorm numbers have
# exponents smaller than -126
def _safe_lshift(x: torch.Tensor, bits: int, exp: Optional[int]) -> torch.Tensor:
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2**exp) * (2**bits)


def _safe_rshift(x: torch.Tensor, bits: int, exp: Optional[int]) -> torch.Tensor:
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2**exp)


def _round_mantissa(
    A: torch.Tensor, bits: int, round: RoundingMode, clamp: bool = False
) -> torch.Tensor:
    """
    Rounds mantissa to nearest bits depending on the rounding method 'round'
    Args:
      A     {PyTorch tensor} -- Input tensor
      round {str}            --  Rounding method
                                 "floor" rounds to the floor
                                 "nearest" rounds to ceil or floor, whichever is nearest
    Returns:
      A {PyTorch tensor} -- Tensor with mantissas rounded
    """

    if round == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round == "even":
        absA = torch.abs(A)
        # find 0.5, 2.5, 4.5 ...
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise Exception("Unrecognized round method %s" % (round))

    # Clip values that cannot be expressed by the specified number of bits
    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A


def _shared_exponents(
    A: torch.Tensor,
    method: str = "max",
    rounding_mode: str = "even",
    axes: Optional[List[int]] = None,
    ebits: int = 0,
) -> torch.Tensor:
    """
    Get shared exponents for the passed matrix A.
    Args:
      A      {PyTorch tensor} -- Input tensor
      method {str}            -- Exponent selection method.
                                 "max" uses the max absolute value
                                 "none" uses an exponent for each value (i.e., no sharing)
      axes   {list(int)}      -- List of integers which specifies the axes across which
                                 shared exponents are calculated.
    Returns:
      shared_exp {PyTorch tensor} -- Tensor of shared exponents
    """

    if method == "max":
        if axes is None:
            shared_exp = torch.max(torch.abs(A))
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    elif method == "none":
        shared_exp = torch.abs(A)
    else:
        raise Exception("Unrecognized shared exponent selection method %s" % (method))

    if rounding_mode == "even":
        MBITS_FP32 = 23
        SBITS = 1
        M_ROUND = (1 << (MBITS_FP32 - SBITS - 1)) - 1
        shared_exp = shared_exp.view(dtype=torch.int32) + M_ROUND
        return torch.floor(torch.log2(shared_exp.view(dtype=torch.float32)))
        """
        roundup_idx = (shared_exp_old != shared_exp).nonzero()
        if roundup_idx.numel() > 0:
            idx = roundup_idx[0]
            raise Exception(
                f"{roundup_idx.numel() / len(shared_exp.shape) / shared_exp.numel() * 100}% exp round up: {idx=} {shared_exp.shape=} {(shared_exp - shared_exp_old)[idx]=}"
            )
        """
    elif rounding_mode == "ceil":
        shared_exp = torch.ceil(
            torch.log2(
                shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
            )
        )
    elif rounding_mode == "floor":
        shared_exp = torch.floor(
            torch.log2(
                shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
            )
        )
    else:
        raise Exception("Unrecognized rounding mode %s" % (rounding_mode))

    # Restrict to [-emax, emax] range
    if ebits > 0:
        emax = 2 ** (ebits - 1) - 1
        # shared_exp = torch.clamp(shared_exp, -emax, emax)
        # Overflow to Inf
        shared_exp[shared_exp > emax] = float("NaN")
        # Underflows are set to -127 which causes them to be
        # flushed to 0 later
        shared_exp[shared_exp < -emax] = -emax

    return shared_exp


# -------------------------------------------------------------------------
# Main funcs
# -------------------------------------------------------------------------
def _quantize_elemwise_core(
    A: torch.Tensor,
    bits: int,
    exp_bits: int,
    max_norm: float,
    round: str = "nearest",
    saturate_normals: bool = False,
    allow_denorm: bool = True,
    custom_cuda: bool = False,
) -> torch.Tensor:
    """
    Core function used for element-wise quantization
    Arguments:
      A         {PyTorch tensor} -- A tensor to be quantized
      bits      {int}            -- Number of mantissa bits. Includes
                                    sign bit and implicit one for floats
      exp_bits  {int}            -- Number of exponent bits, 0 for ints
      max_norm  {float}          -- Largest representable normal number
      round     {str}            -- Rounding mode: (floor, nearest, even)
      saturate_normals {bool}    -- If True, normal numbers (i.e., not NaN/Inf)
                                    that exceed max norm are clamped.
                                    Must be True for correct MX conversion.
      allow_denorm     {bool}    -- If False, flush denorm numbers in the
                                    elem_format to zero.
      custom_cuda      {str}     -- If True, use custom CUDA kernels
    Returns:
      quantized tensor {PyTorch tensor} -- A tensor that has been quantized
    """
    A_is_sparse = A.is_sparse
    if A_is_sparse:
        if A.layout != torch.sparse_coo:
            raise NotImplementedError(
                "Only COO layout sparse tensors are currently supported."
            )

        sparse_A = A.coalesce()
        A = sparse_A.values().clone()

    # Flush values < min_norm to zero if denorms are not allowed
    if not allow_denorm and exp_bits > 0:
        min_norm = _get_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(torch.abs(A) + (A == 0).type(A.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2 ** (exp_bits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of bits are in the integer portion of the number
    # pyre-fixme[6]: For 3rd argument expected `Optional[int]` but got
    #  `Optional[Tensor]`.
    out = _safe_lshift(out, bits - 2, private_exp)

    # pyre-fixme[6]: For 3rd argument expected `RoundingMode` but got `str`.
    out = _round_mantissa(out, bits, round, clamp=False)

    # Undo scaling
    # pyre-fixme[6]: For 3rd argument expected `Optional[int]` but got
    #  `Optional[Tensor]`.
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where(
            (torch.abs(out) > max_norm), torch.sign(out) * float("Inf"), out
        )

    # handle Inf/NaN
    if not custom_cuda:
        out[A == float("Inf")] = float("Inf")
        out[A == -float("Inf")] = -float("Inf")
        out[A == float("NaN")] = float("NaN")

    if A_is_sparse:
        out = torch.sparse_coo_tensor(
            sparse_A.indices(),
            out,
            sparse_A.size(),
            dtype=sparse_A.dtype,
            device=sparse_A.device,
            requires_grad=sparse_A.requires_grad,
        )

    return out


def _quantize_elemwise(
    A: torch.Tensor,
    elem_format: Union[ElemFormat, None],
    round: str = "nearest",
    custom_cuda: bool = False,
    saturate_normals: bool = False,
    allow_denorm: bool = True,
) -> torch.Tensor:
    """Quantize values to a defined format. See _quantize_elemwise_core()"""
    if elem_format is None:
        return A

    ebits, mbits, _, max_norm, _ = _get_format_params(elem_format)

    output = _quantize_elemwise_core(
        A,
        mbits,
        ebits,
        max_norm,
        round=round,
        allow_denorm=allow_denorm,
        saturate_normals=saturate_normals,
        custom_cuda=custom_cuda,
    )

    return output

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from . import common  # noqa E402

# pyre-fixme[21]: Could not find name `open_source` in
#  `deeplearning.fbgemm.fbgemm_gpu.test.quantize.common`.
from .common import open_source

# pyre-fixme[16]: Module `common` has no attribute `open_source`.
if open_source:
    # pyre-ignore[21]
    from test_utils import optests
else:
    from fbgemm_gpu.test.test_utils import optests

    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")

torch.ops.import_module("fbgemm_gpu.sparse_ops")

E4M3_MIN_SUBNORMAL = 2.0**-9
E4M3_MAX = 448.0
QMAX = 7.0


def grouped_reference(
    rows: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent torch port of the grouped fp8_e4m3fn arm of D126999147:
    scale = e4m3fn(max|x| / 7) per block (at least 2^-9), code = rint(x / scale)
    clipped to [-7, 7]. Returns (codes, scales, decoded fp16 values)."""
    x = rows.float().reshape(rows.shape[0], -1, group_size)
    ideal = x.abs().amax(dim=2, keepdim=True) / QMAX
    scale = ideal.clamp(E4M3_MIN_SUBNORMAL, E4M3_MAX).to(torch.float8_e4m3fn)
    # Integer codes, so a code of 0 decodes to +0 (never -0) as stored codes do.
    codes = torch.round(x / scale.float()).clamp(-QMAX, QMAX).to(torch.int8)
    values = (codes.float() * scale.float()).reshape(rows.shape).half()
    return codes.reshape(rows.shape), scale.squeeze(2), values


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def unpack(
    packed: torch.Tensor, dim: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Codes (even column in the low nibble, two's complement) and fp8 scales."""
    code_bytes = packed[:, : dim // 2]
    nibbles = torch.stack([code_bytes & 0x0F, code_bytes >> 4], dim=2)
    nibbles = nibbles.reshape(packed.shape[0], dim).to(torch.int8)
    codes = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    scales = packed[:, dim // 2 : dim // 2 + dim // group_size].contiguous()
    return codes, scales.view(torch.float8_e4m3fn)


def sample_rows(dim: int, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator().manual_seed(dim)
    rows = torch.randn(64, dim, generator=gen) * 0.05
    # Small rows give e4m3 subnormal scales; a zero row hits the scale floor.
    rows[1] *= 1e-3
    rows[2] = 0.0
    rows[3, :4] = torch.tensor([7.0, -3.5, 3.5, 0.5])
    return rows.to(dtype)


@optests.generate_opcheck_tests(fast=True)
class GroupedInt4E4M3Test(unittest.TestCase):
    def test_packed_rows_match_reference(self) -> None:
        for dim, group_size in ((128, 8), (128, 16), (256, 32), (16, 8)):
            for dtype in (torch.float32, torch.float16):
                with self.subTest(dim=dim, group_size=group_size, dtype=dtype):
                    rows = sample_rows(dim, dtype)
                    packed = torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(
                        rows, group_size
                    )
                    self.assertEqual(
                        packed.shape, (rows.shape[0], dim // 2 + dim // group_size)
                    )
                    ref_codes, ref_scales, ref_values = grouped_reference(
                        rows, group_size
                    )
                    codes, scales = unpack(packed, dim, group_size)
                    self.assertTrue(torch.equal(codes, ref_codes))
                    self.assertTrue(
                        torch.equal(
                            scales.view(torch.uint8), ref_scales.view(torch.uint8)
                        )
                    )
                    decoded = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(
                        packed, group_size, dim
                    )
                    self.assertTrue(same_bits(decoded, ref_values))

    def test_decode_ignores_row_padding(self) -> None:
        dim, group_size = 128, 8
        packed = torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(
            sample_rows(dim, torch.float16), group_size
        )
        padding = torch.full((packed.shape[0], 16), 0xFF, dtype=torch.uint8)
        padded = torch.cat([packed, padding], dim=1)
        self.assertTrue(
            same_bits(
                torch.ops.fbgemm.GroupedInt4E4M3ToHalf(padded, group_size, dim),
                torch.ops.fbgemm.GroupedInt4E4M3ToHalf(packed, group_size, dim),
            )
        )

    def test_decode_as_bytes_returns_the_fp16_bytes(self) -> None:
        dim, group_size = 128, 8
        packed = torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(
            sample_rows(dim, torch.float16), group_size
        )
        as_bytes = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(
            packed, group_size, dim, as_bytes=True
        )
        self.assertEqual(as_bytes.dtype, torch.uint8)
        self.assertEqual(as_bytes.shape, (packed.shape[0], 2 * dim))
        self.assertTrue(
            torch.equal(
                as_bytes,
                torch.ops.fbgemm.GroupedInt4E4M3ToHalf(packed, group_size, dim).view(
                    torch.uint8
                ),
            )
        )

    def test_round_trip_reaches_fixed_point(self) -> None:
        dim, group_size = 128, 8
        values = sample_rows(dim, torch.float16)
        for _ in range(4):
            decoded = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(
                torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(values, group_size),
                group_size,
                dim,
            )
            if same_bits(decoded, values):
                break
            values = decoded
        self.assertTrue(same_bits(decoded, values))

    def test_fixed_point_rows_requantize_to_the_same_bytes(self) -> None:
        """A row training stored as a fixed point publishes to codes that serving
        decodes back to that row exactly, from fp16 or fp32 storage and from a
        column slice of a wider row, as publish passes it."""
        dim, group_size = 128, 8
        rows = sample_rows(dim, torch.float16).float()
        rows[4] = 0.0  # every group at the scale floor
        rows[5, :8] = 2.0**-12  # absmax below 7 * 2^-9
        rows[6, :8] = torch.tensor([5000.0, -4000.0, 3136.0, 1.0, 0, 0, 0, 0])  # clamp
        rows[7, :8] = 6.0e-8  # fp16 subnormals
        rows[8, :8] = torch.tensor([7.0, -7.0, 7.0, -7.0, 1.0, -1.0, 0.5, 0.0])
        values = rows.half()
        for _ in range(4):
            decoded = torch.ops.fbgemm.GroupedInt4E4M3ToHalf(
                torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(values, group_size),
                group_size,
                dim,
            )
            if same_bits(decoded, values):
                break
            values = decoded
        self.assertTrue(same_bits(decoded, values))

        packed = torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(values, group_size)
        wide = torch.zeros(values.shape[0], dim + 16, dtype=torch.float16)
        wide[:, 8 : 8 + dim] = values
        for name, source in (
            ("fp32", values.float()),
            ("column slice", wide[:, 8 : 8 + dim]),
        ):
            with self.subTest(source=name):
                self.assertTrue(
                    torch.equal(
                        torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(
                            source, group_size
                        ),
                        packed,
                    )
                )
        self.assertTrue(
            same_bits(
                torch.ops.fbgemm.GroupedInt4E4M3ToHalf(packed, group_size, dim), values
            )
        )

    def test_rejects_bad_dims(self) -> None:
        for device in ("cpu", "meta"):
            with self.subTest(device=device):
                rows = torch.zeros(2, 12, dtype=torch.float16, device=device)
                for group_size in (8, 3, 0):
                    with self.assertRaises(RuntimeError):
                        torch.ops.fbgemm.FloatOrHalfToGroupedInt4E4M3(rows, group_size)
                packed = torch.zeros(2, 70, dtype=torch.uint8, device=device)
                for group_size in (8, 0):
                    with self.assertRaises(RuntimeError):
                        torch.ops.fbgemm.GroupedInt4E4M3ToHalf(packed, group_size, 128)


if __name__ == "__main__":
    unittest.main()

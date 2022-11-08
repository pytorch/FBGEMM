/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
namespace fbgemm_gpu {

at::Tensor _hfp8_to_float_cpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias);
at::Tensor _float_to_hfp8_cpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias,
    const double max_pos);

using fint32 = union fint32 {
  uint32_t I;
  float F;
};

// TODO: add a flag later to control whether underflow
// flushes to 0 or clips to smallest denorm number.
inline C10_HOST_DEVICE uint8_t
float_to_hfp8(float val_fp, int ebits, int exponent_bias, float max_pos) {
  int mbits = 7 - ebits;
  fint32 val_out, bouncer, smallest_normal;

  val_out.F = val_fp;
  uint32_t sign_bit = val_out.I & 0x80000000;
  val_out.I = val_out.I & 0x7FFFFFFF;
  val_out.F = fminf(val_out.F, max_pos);

  smallest_normal.I = (127 - exponent_bias + 1)
      << 23; // smallest hfp8 normal number in FP32
  // I don't know if the input "min_pos" is the smallest denormalized number
  // or the smallest normalized number. The test below needs to be done with
  // the smallest normal number, which is the numerical value 2^(1-bias)

  // The conversion for denormalized values are slightly different. HFP8 is so
  // low precision that gradual underflow is probably crucial
  if (val_out.F >= smallest_normal.F) {
    // Use round to nearest even. We make use of the standard rounding mechanism
    // in FP32 rather than rounding the mantissa and handling tie-to-even and
    // incrementing exponent We want to round of 23-mbits of the FP32 value
    // val_in This can be done by adding a power of 2 exactly 23-mbits larger
    // than the exponent of val_in This forces val_in to be moved to the right
    // and rounding exact at the location corresponding to having mbits of
    // explicit mantissa left
    bouncer.I = (val_out.I & 0xFF800000) + ((23 - mbits) << 23);
    val_out.F = (bouncer.F + val_out.F) - bouncer.F;
    // adding the bouncer rounds off bits, and subtracting bouncer
    // leaves the desired value, albeit in FP32 encoding
    // All we need is to change the exponent encoding to using "bias"
    val_out.I = uint32_t(val_out.I - ((127 - exponent_bias) << 23))
        << (8 - ebits);
    val_out.I =
        ((val_out.I | sign_bit) >>
         24); // the 8 lsbs is the desired HFP8 encoding

  } else {
    // When the value is in the denormal range, IEEE numbers essentially becomes
    // a fixed point number. The lsb is the smallest non-zero number
    // 2^(1-bias-mbits) Hence, we define the bouncer so that its lsb is this
    // smallest non-zero number Adding the input to this bouncer forces rounding
    // to occur appropriately Also, in this situation, after adding the bouncer,
    // the 8 least significant bits of the sum is already the HFP8 encoding of
    // the desired result. Just need to restore the sign bit
    bouncer.I = (127 + (23 + (1 - exponent_bias - mbits))) << 23;
    val_out.F = bouncer.F + val_out.F;
    val_out.I = val_out.I | (sign_bit >> 24);
    ;
  }

  uint8_t bfp8_val = val_out.I; // get the 8 lsbs
  return bfp8_val;
}

inline C10_HOST_DEVICE float
hfp8_to_float(uint8_t hfp8_val, int ebits, int exponent_bias) {
  fint32 val_out, sign, multiplier;

  sign.I = (hfp8_val & 0x80) << 24;
  val_out.I = (hfp8_val & 0x7F) << (24 - (8 - ebits));
  // so that the mantissa bits start at the mantissa bit positions of FP32
  // encoding

  // Let the hfp8 mantissa bits correspond to the value frac, 0 <= frac < 1
  // So if the hfp8 value is a normal number, it's value is 2^e x (1+frac)
  // where e is its (true, unbiased) exponent
  // If the hfp8 value is denormal, the value is 2^(1-bias) x frac

  // However, the bit pattern in the 8-bit exponent field of val_out.F
  // is bias+e when hfp8 is normal, and 0 when hfp8 is subnormal.
  // So, as an FP32 value, when hfp8 is normal, val_out.F represents the value
  // of 2^(bias+e-127) * (1+frac)
  // And when hfp8 is subnormal, val_out.F is also subnormal, and represents the
  // value of 2^(-126) * frac In either case, val_out.F corresponds to
  // 2^(bias-127) * (value of hfp8 input) Thus, if we multiply val_out.F by
  // 2^(127-bias), we obtain the hfp8 value as an FP32 number

  multiplier.I = (127 + (127 - exponent_bias))
      << 23; // multiplier.F is 2^(127-bias)
  val_out.F *= multiplier.F;
  val_out.I |= sign.I;
  return val_out.F;
}

inline C10_HOST_DEVICE float float_to_flexp(
    const float val_fp,
    const int mbits,
    const int bias,
    const float max_pos) {
  fint32 X, bouncer, scale, inv_scale;
  uint32_t sign_bit;
  int32_t E, expo, emin, delta_E, nbits2round;

  X.F = val_fp;
  sign_bit = X.I & 0x80000000;
  X.I = X.I & 0x7FFFFFFF;

  emin = 1 - bias;

  // Because the input value can be of extreme magnitude
  // We scale them into less extreme to avoid potential exception during
  // manipulation
  E = ((X.I & 0x7F800000) >> 23) - 127;
  if (E >= 0) {
    scale.I = 0X2F800000;
    inv_scale.I = 0X4F800000; // scale is 2^-32, inv_scale is 2^32
    delta_E = -32;
  } else {
    scale.I = 0x4F800000;
    inv_scale.I = 0x2F800000;
    delta_E = 32;
  }
  X.F *= scale.F; // at this point X is never close to over/underflow
  expo = ((X.I & 0x7F800000) >> 23) - 127 - delta_E;

  // If expo >= emin
  // We round to mbits explicit mantissa bits
  // That is, we want to round off 23-mbits of the trailing bits in X
  nbits2round = 23 - mbits;
  // However, if expo < emin, we need to round more bits off
  nbits2round += std::max(emin - expo, 0);

  bouncer.I = (nbits2round << 23) + (X.I & 0x7F800000);
  X.F = X.F + bouncer.F; // Because bouncer is exactly 2^nbits2round bigger
                         // this addition forces the rounding off of nbits2round
  X.F = X.F - bouncer.F; // X.F is the original X with nbits2round rounded off

  // restore the true magnitude by undoing the previous scale
  X.F *= inv_scale.F;
  // clip on the large end of the domain
  X.F = fminf(X.F, max_pos);
  // restores the original sign
  X.I |= sign_bit;

  const float val_flexp = X.F;
  return val_flexp;
}

inline C10_HOST_DEVICE float float_to_msfp(
    const float val_fp,
    const int shared_expo,
    const int mbits,
    const int bias,
    const float max_pos) {
  fbgemm_gpu::fint32 X, bouncer, scale, inv_scale;
  int32_t expo, emin, delta_E, nbits2round;

  X.F = val_fp;
  const uint32_t sign_bit = X.I & 0x80000000;
  X.I = X.I & 0x7FFFFFFF; // 31 bits

  emin = 1 - bias;

  // Because the input value can be of extreme magnitude
  // We scale them into less extreme to avoid potential exception during
  // manipulation
  const int32_t E = ((X.I & 0x7F800000) >> 23) - 127;
  if (E >= 0) {
    scale.I = 0X2F800000;
    inv_scale.I = 0X4F800000; // scale is 2^-32, inv_scale is 2^32
    delta_E = -32;
  } else {
    scale.I = 0x4F800000;
    inv_scale.I = 0x2F800000;
    delta_E = 32;
  }
  X.F *= scale.F; // at this point X is never close to over/underflow
  expo = ((X.I & 0x7F800000) >> 23) - 127 - delta_E;

  // If expo >= emin
  // We round to mbits explicit mantissa bits
  // That is, we want to round off 23-mbits of the trailing bits in X
  nbits2round = 23 - mbits + 1;
  // However, if expo < emin, we need to round more bits off
  nbits2round += std::max(emin - expo, 0); // max(emin - expo, 0);
  // also need to right shift mantissa with the shared expoennt
  nbits2round += std::max(shared_expo - expo, 0);

  bouncer.I = (nbits2round << 23) + (X.I & 0x7F800000);
  X.F = X.F + bouncer.F; // Because bouncer is exactly 2^nbits2round bigger
                         // this addition forces the rounding off of nbits2round
  X.F = X.F - bouncer.F; // X.F is the original X with nbits2round rounded off

  // restore the true magnitude by undoing the previous scale
  X.F *= inv_scale.F;
  // clip on the large end of the domain
  X.F = std::min(X.F, max_pos);
  // restores the original sign
  X.I |= sign_bit;

  const float val_msfp = X.F;
  return val_msfp;
}

inline C10_HOST_DEVICE float dequantize_msfp_float_from_uint8(
    const uint8_t mantissa,
    const uint8_t exponent,
    int ebits,
    int mbits,
    int bias) {
  fbgemm_gpu::fint32 val_out, sign, mbit_offset;
  int32_t signed_exponent = 0;
  // convert exponent from uint8_t to int32_t
  if (exponent > 127) {
    signed_exponent = ~(exponent - 1) & 0xFF;
    signed_exponent = -signed_exponent;
  } else {
    signed_exponent = exponent;
  }
  val_out.I = (mantissa & 0x7F) << (23 - mbits);
  val_out.I |= (signed_exponent + 127) << 23;
  mbit_offset.I = (signed_exponent + 127) << 23;
  val_out.F = val_out.F - mbit_offset.F;
  val_out.F = val_out.F * 2;
  sign.I = (mantissa & 0x80) << 24;
  val_out.I |= sign.I;
  /*
  printf(
      "dequantize_msfp_float_from_uint8 exponent %d signed exponent %d mantissa
  %d value %f\n", exponent, signed_exponent, mantissa, val_out.F);
  */
  return val_out.F;
}
inline C10_HOST_DEVICE uint8_t quantize_msfp_float_to_uint8(
    const float val_fp,
    const uint8_t shared_expo,
    const int mbits,
    const int bias,
    const float max_pos) {
  fbgemm_gpu::fint32 X, scale, inv_scale;
  int32_t expo, emin, delta_E, nbits2round, signed_shared_expo;
  uint8_t out = 0;
  X.F = val_fp;
  const uint32_t sign_bit = X.I & 0x80000000;
  X.I = X.I & 0x7FFFFFFF; // 31 bits
  if (X.I == 0) {
    return out;
  }
  emin = 1 - bias;

  // convert shared_expo from uint8_t to int32_t
  if (shared_expo > 127) {
    signed_shared_expo = ~(shared_expo - 1) & 0xFF;
    signed_shared_expo = -signed_shared_expo;
  } else {
    signed_shared_expo = shared_expo;
  }

  // Because the input value can be of extreme magnitude
  // We scale them into less extreme to avoid potential exception during
  // manipulation
  const int32_t E = ((X.I & 0x7F800000) >> 23) - 127;
  if (E >= 0) {
    scale.I = 0X2F800000;
    inv_scale.I = 0X4F800000; // scale is 2^-32, inv_scale is 2^32
    delta_E = -32;
  } else {
    scale.I = 0x4F800000;
    inv_scale.I = 0x2F800000;
    delta_E = 32;
  }
  X.F *= scale.F; // at this point X is never close to over/underflow
  expo = ((X.I & 0x7F800000) >> 23) - 127 - delta_E;

  // If expo >= emin
  // We round to mbits explicit mantissa bits
  // That is, we want to round off 23-mbits of the trailing bits in X
  // needs 1 bit for the leading bit in mantissa
  nbits2round = 23 - mbits + 1;
  // However, if expo < emin, we need to round more bits off
  nbits2round += std::max(emin - expo, 0); // max(emin - expo, 0);
  // also need to right shift mantissa with the shared expoennt
  nbits2round += std::max(signed_shared_expo - expo, 0);

  // restore the true magnitude by undoing the previous scale
  X.F *= inv_scale.F;
  // clip on the large end of the domain
  X.F = std::min(X.F, max_pos);
  // restores the original sign

  out = ((X.I & 0x7F0000) | 0x800000) >> nbits2round;
  if (sign_bit) {
    out = out | 0x80;
  }
  uint8_t out_without_leading_bit = (X.I & 0x7F0000) >> (nbits2round - 1);
  X.I = (X.I & 0x7F800000) | (out_without_leading_bit << (23 - mbits));
  X.I |= sign_bit;

  /*
  printf(
      "quantize value %f E %d expo %d with shared_expo %d signed shared expo %d
  to %f %d err %f\n", val_fp, E, expo, shared_expo, signed_shared_expo, X.F,
      out,
      val_fp - X.F);
  */
  return out;
}
} // namespace fbgemm_gpu

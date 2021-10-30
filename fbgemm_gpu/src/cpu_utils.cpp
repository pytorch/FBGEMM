/*
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "include/fbgemm_gpu/cpu_utils.h"
#include <ATen/Parallel.h>
namespace fbgemm_gpu {

alignas(64) static thread_local uint32_t rnd_state_tl[64];
static thread_local bool rnd_state_tl_initialized = false;

void internal_rng_float_jump(
    uint32_t* state0,
    uint32_t* state1,
    uint32_t* state2,
    uint32_t* state3) {
  static const uint32_t jump_table[] = {
      0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b};
  uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int i, b;

  for (i = 0; i < 4; ++i) {
    for (b = 0; b < 32; ++b) {
      if (jump_table[i] & (1U << b)) {
        s0 ^= *state0;
        s1 ^= *state1;
        s2 ^= *state2;
        s3 ^= *state3;
      }
      { /* draw one more integer */
        const uint32_t t = *state1 << 9;
        *state2 ^= *state0;
        *state3 ^= *state1;
        *state1 ^= *state2;
        *state0 ^= *state3;
        *state2 ^= t;
        *state3 = ((*state3 << 11) | (*state3 >> (32 - 11)));
      }
    }
  }
  *state0 = s0;
  *state1 = s1;
  *state2 = s2;
  *state3 = s3;
}

void init_threadlocal_rnd_state(unsigned int seed) {
  if (rnd_state_tl_initialized) return;
  auto &state = rnd_state_tl;
  static const uint32_t temp_state[] = {
      31,  30,  29,  28,  27,  26,  25,  24,  23,  22,  21,  20,  19,
      18,  17,  16,  131, 130, 129, 128, 127, 126, 125, 124, 123, 122,
      121, 120, 119, 118, 117, 116, 231, 230, 229, 228, 227, 226, 225,
      224, 223, 222, 221, 220, 219, 218, 217, 216, 331, 330, 329, 328,
      327, 326, 325, 324, 323, 322, 321, 320, 319, 318, 317, 316};

  int i;
  unsigned int tseed = at::get_thread_num() + seed;
  /* finish initializing the state */
  for (i = 0; i < 16; ++i) {
    state[i] = tseed + temp_state[i];
    state[i + 16] = tseed + temp_state[i + 16];
    state[i + 32] = tseed + temp_state[i + 32];
    state[i + 48] = tseed + temp_state[i + 48];
  }
  for (i = 0; i < 16; ++i) {
    internal_rng_float_jump(/* progress each sequence by 2^64 */
                            state + i,
                            state + 16 + i,
                            state + 32 + i,
                            state + 48 + i);
  }

  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
  rnd_state_tl_initialized = true;
}

__m512i MM512_RNG_XOSHIRO128P_EXTSTATE_EPI32(unsigned int* stateptr) {
  __m512i state_0 = _mm512_loadu_si512(stateptr);
  __m512i state_1 = _mm512_loadu_si512(stateptr + 16);
  __m512i state_2 = _mm512_loadu_si512(stateptr + 32);
  __m512i state_3 = _mm512_loadu_si512(stateptr + 48);
  const __m512i result = _mm512_add_epi32(state_0, state_3);
  const __m512i s = _mm512_slli_epi32(state_1, 9);
  __m512i t;
  state_2 = _mm512_xor_epi32(state_2, state_0);
  state_3 = _mm512_xor_epi32(state_3, state_1);
  state_1 = _mm512_xor_epi32(state_1, state_2);
  state_0 = _mm512_xor_epi32(state_0, state_3);
  state_2 = _mm512_xor_epi32(state_2, s);
  _mm512_storeu_si512(stateptr, state_0);
  _mm512_storeu_si512(stateptr + 16, state_1);
  _mm512_storeu_si512(stateptr + 32, state_2);
  t = _mm512_slli_epi32(state_3, 11);
  state_3 = _mm512_or_epi32(t, _mm512_srli_epi32(state_3, 32 - 11));
  _mm512_storeu_si512(stateptr + 48, state_3);
  return result;
}

__m256i _mm512_cvtps_ph_stoc(__m512 src) {
  static thread_local __m512i random[4];
  static thread_local int i = 0;

  if (i == 0) {
    auto rnd = MM512_RNG_XOSHIRO128P_EXTSTATE_EPI32(rnd_state_tl);
    __m512i mm512_rng_mask = _mm512_set1_epi32(0x00001fe0);
    random[0] = _mm512_and_si512(mm512_rng_mask, _mm512_slli_epi32(rnd, 5));
    random[1] = _mm512_and_si512(mm512_rng_mask, _mm512_srli_epi32(rnd, 3));
    random[2] = _mm512_and_si512(mm512_rng_mask, _mm512_srli_epi32(rnd, 11));
    random[3] = _mm512_and_si512(mm512_rng_mask, _mm512_srli_epi32(rnd, 19));
  }
  src = _mm512_castsi512_ps(
      _mm512_add_epi32(random[i], _mm512_castps_si512(src)));
  i = (i + 1) & 0x03;

  return _mm512_cvtps_ph(src, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}
}

// @lint-ignore-every LICENSELINT
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include <cpuinfo.h>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "fbgemm/Utils.h"

namespace fbgemm {

namespace {
inst_set_t g_forced_isa = inst_set_t::anyarch;
std::optional<bool> g_Avx512_Ymm_enabled{std::nullopt};

inst_set_t fbgemmEnvGetIsa() {
  static const char* isa_env = "FBGEMM_ENABLE_INSTRUCTIONS";
  static const std::unordered_map<std::string, inst_set_t> isaMap = {
      {"AVX2", inst_set_t::avx2},
      {"AVX512", inst_set_t::avx512},
      {"AVX512_E1", inst_set_t::avx512_vnni},
      {"AVX512_256", inst_set_t::avx512_ymm},
      {"AVX512_E1_256", inst_set_t::avx512_vnni_ymm},
  };
  const char* env = std::getenv(isa_env);
  if (env == nullptr) {
    return inst_set_t::anyarch;
  }

#ifdef __aarch64__
#ifdef VLOG
  VLOG(0) << "[" << env << "] not supported on aarch64";
#endif
  return inst_set_t::anyarch;
#endif

  std::string val(env);
  std::ranges::transform(val, val.begin(), ::toupper);
  auto it = isaMap.find(val);
  return it == isaMap.end() ? inst_set_t::anyarch : it->second;
}

bool fbgemmEnvAvx512_256Enabled() {
  static const char* isa_env = "FBGEMM_ENABLE_AVX512_256";
  const char* env = std::getenv(isa_env);
  if (env == nullptr) {
    return false;
  }

#ifdef __aarch64__
#ifdef VLOG
  VLOG(0) << "[" << env << "] not supported on aarch64";
#endif
  return false;
#endif

  std::string val(env);
  std::ranges::transform(val, val.begin(), ::tolower);
  return val == "true" || val == "1";
}

std::unordered_map<inst_set_t, std::unordered_set<inst_set_t>> isaSupportMap = {
    {inst_set_t::anyarch, {inst_set_t::anyarch}},
    {inst_set_t::avx2, {inst_set_t::avx2, inst_set_t::anyarch}},
    {inst_set_t::avx512,
     {inst_set_t::avx512, inst_set_t::avx512_ymm, inst_set_t::avx2}},
    {inst_set_t::avx512_ymm,
     {inst_set_t::avx512, inst_set_t::avx512_ymm, inst_set_t::avx2}},
    {inst_set_t::avx512_vnni,
     {inst_set_t::avx512_vnni,
      inst_set_t::avx512_vnni_ymm,
      inst_set_t::avx512,
      inst_set_t::avx512_ymm,
      inst_set_t::avx2}},
    {inst_set_t::avx512_vnni_ymm,
     {inst_set_t::avx512_vnni,
      inst_set_t::avx512_vnni_ymm,
      inst_set_t::avx512,
      inst_set_t::avx512_ymm,
      inst_set_t::avx2}},
};

} // namespace

/**
 * @brief Force specific architecure to for GEMM kernel execution
 *        overides FBGEMM_ENABLE_AVX512_256 env. variable
 * @param isa the ISA to enforce, supported optionsi
 *     AVX2          inst_set_t::avx2
 *     AVX512        inst_set_t::avx512
 *     AVX512_E1     inst_set_t::avx512_vnni
 *     AVX512_256    inst_set_t::avx512_ymm
 *     AVX512_E1_256 inst_set_t::avx512_vnni_ymm
 */
void fbgemmForceIsa(inst_set_t isa) {
  g_forced_isa = isa;
#ifdef __aarch64__
#ifdef VLOG
  VLOG(0) << "[anyarch] forced on aarch64";
#endif
  g_forced_isa = inst_set_t::anyarch;
#endif
}

/**
 * @brief Enables AVX512-256 if appriate. Inteded for Skylake based Xeon-D
 *        processors, wherein AXV512-256 is preferred due to higher
 *        Turbo frequencis
 * @param flag True enables / False disables
 */
void fbgemmEnableAvx512Ymm(bool flag) {
  g_Avx512_Ymm_enabled = flag;
}

/**
 * @brief Determine the best available x86 machine ISA to be used for
 *        GEMM kernels.
 *        FBGEMM_ENABLE_AVX512_256 env. or fbgemmForceIsa() are set
 *        forces to specific architecture if supported by the processor.
 *        Enforcing on Skylake to AVX2 will execute AVX2 version of the kernel
 *        However, enforcing AVX512-256 on Broadwell will fail, and AVX2 version
 *        of the kernels will be executed.
 */
inst_set_t fbgemmInstructionSet() {
  static const inst_set_t env_forced_isa = fbgemmEnvGetIsa();
  static const bool isAvx512_Ymm_enabled = fbgemmEnvAvx512_256Enabled();

  if (fbgemmHasArmSveSupport()) {
    return inst_set_t::sve;
  }

  inst_set_t forced_isa =
      g_forced_isa != inst_set_t::anyarch ? g_forced_isa : env_forced_isa;
  static const inst_set_t detected_isa = ([]() {
    inst_set_t isa = inst_set_t::anyarch;
    // Check environment
    if (cpuinfo_initialize()) {
      const bool isXeonD = fbgemmIsIntelXeonD() &&
          (g_Avx512_Ymm_enabled.value_or(isAvx512_Ymm_enabled));
      if (fbgemmHasAvx512VnniSupport()) {
        if (isXeonD) {
          isa = inst_set_t::avx512_vnni_ymm;
        } else {
          isa = inst_set_t::avx512_vnni;
        }
      } else if (fbgemmHasAvx512Support()) {
        if (isXeonD) {
          isa = inst_set_t::avx512_ymm;
        } else {
          isa = inst_set_t::avx512;
        }
      } else if (fbgemmHasAvx2Support()) {
        isa = inst_set_t::avx2;
      }
    }
    return isa;
  })();

  if (forced_isa == inst_set_t::anyarch) {
    return detected_isa;
  }
  const auto supported_isa = isaSupportMap.find(detected_isa);
  assert(
      supported_isa != isaSupportMap.end() &&
      "Detected ISA can't be located in Supported ISA map");
  if (supported_isa == isaSupportMap.end()) {
    return detected_isa;
  }
  return supported_isa->second.count(forced_isa) ? forced_isa : detected_isa;
}

bool isZmm(inst_set_t isa) {
  return isa == inst_set_t::avx512 || isa == inst_set_t::avx512_vnni;
}

bool isYmm(inst_set_t isa) {
  return isa == inst_set_t::avx512_ymm || isa == inst_set_t::avx512_vnni_ymm ||
      isa == inst_set_t::avx2;
}
} // namespace fbgemm

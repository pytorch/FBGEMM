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
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include "./PlatformSpecificKernels.h"
#include "fbgemm/Utils.h"

namespace fbgemm {

namespace {
enum class Avx512YmmOverride : uint8_t { unset, disabled, enabled };

constinit std::atomic<inst_set_t> g_forced_isa{inst_set_t::anyarch};
constinit std::atomic<Avx512YmmOverride> g_Avx512_Ymm_override{
    Avx512YmmOverride::unset};

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

bool isForcedIsaSupported(inst_set_t detected, inst_set_t forced) {
  using enum inst_set_t;
  switch (detected) {
    case avx512_vnni:
    case avx512_vnni_ymm:
      if (forced == avx512_vnni || forced == avx512_vnni_ymm) {
        return true;
      }
      [[fallthrough]];
    case avx512:
    case avx512_ymm:
      if (forced == avx512 || forced == avx512_ymm) {
        return true;
      }
      [[fallthrough]];
    case avx2:
      if (forced == avx2) {
        return true;
      }
      [[fallthrough]];
    case anyarch:
      return forced == anyarch;
    default:
      return false;
  }
}

inst_set_t getForcedIsa() {
  static const inst_set_t env_forced_isa = fbgemmEnvGetIsa();
  const inst_set_t forced_isa = g_forced_isa.load(std::memory_order_relaxed);
  return forced_isa != inst_set_t::anyarch ? forced_isa : env_forced_isa;
}

bool getUseYmmOverZmm() {
  static const bool env_avx512_ymm_enabled = fbgemmEnvAvx512_256Enabled();
  const Avx512YmmOverride ymm_override =
      g_Avx512_Ymm_override.load(std::memory_order_relaxed);
  const bool avx512_ymm_enabled = ymm_override == Avx512YmmOverride::unset
      ? env_avx512_ymm_enabled
      : ymm_override == Avx512YmmOverride::enabled;
  return fbgemmIsIntelXeonD() && avx512_ymm_enabled;
}

inst_set_t detectIsa() {
  using enum inst_set_t;
  if (!cpuinfo_initialize()) {
    return anyarch;
  }
  const bool useYmm = getUseYmmOverZmm();
  if (fbgemmHasAvx512VnniSupport()) {
    return useYmm ? avx512_vnni_ymm : avx512_vnni;
  }
  if (fbgemmHasAvx512Support()) {
    return useYmm ? avx512_ymm : avx512;
  }
  if (fbgemmHasAvx2Support()) {
    return avx2;
  }
  return anyarch;
}

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
#ifdef __aarch64__
#ifdef VLOG
  VLOG(0) << "[anyarch] forced on aarch64";
#endif
  isa = inst_set_t::anyarch;
#endif
  g_forced_isa.store(isa, std::memory_order_relaxed);
  initializePlatformDispatch(/*force=*/true);
}

/**
 * @brief Enables AVX512-256 if appriate. Inteded for Skylake based Xeon-D
 *        processors, wherein AXV512-256 is preferred due to higher
 *        Turbo frequencis
 * @param flag True enables / False disables
 */
void fbgemmEnableAvx512Ymm(bool flag) {
  g_Avx512_Ymm_override.store(
      flag ? Avx512YmmOverride::enabled : Avx512YmmOverride::disabled,
      std::memory_order_relaxed);
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
  if (fbgemmHasArmSveSupport()) {
    return inst_set_t::sve;
  }

  const inst_set_t forced_isa = getForcedIsa();
  static const inst_set_t detected_isa = detectIsa();

  if (forced_isa == inst_set_t::anyarch) {
    return detected_isa;
  }
  return isForcedIsaSupported(detected_isa, forced_isa) ? forced_isa
                                                        : detected_isa;
}

bool isZmm(inst_set_t isa) {
  return isa == inst_set_t::avx512 || isa == inst_set_t::avx512_vnni;
}

bool isYmm(inst_set_t isa) {
  return isa == inst_set_t::avx512_ymm || isa == inst_set_t::avx512_vnni_ymm ||
      isa == inst_set_t::avx2;
}
} // namespace fbgemm

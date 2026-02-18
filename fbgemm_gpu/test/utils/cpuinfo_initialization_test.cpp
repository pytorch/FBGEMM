/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * C++ test to verify cpuinfo initialization consistency between PyTorch and
 * FBGEMM.
 *
 * This test directly checks cpuinfo state to detect issues caused by linking
 * multiple instances of the cpuinfo library.
 *
 * Background:
 * -----------
 * The cpuinfo library uses global state to store CPU feature flags.
 * When multiple copies are linked (e.g., one from PyTorch and another from
 * FBGEMM), each copy has independent global state. If PyTorch initializes its
 * cpuinfo instance but FBGEMM's instance remains uninitialized, FBGEMM's
 * feature detection returns incorrect values.
 *
 * This bug was introduced in PyTorch commit
 * c5aa299b048da269e1165216a1ef3cb06edb413d which added
 * `target_link_libraries(torch_python PRIVATE cpuinfo)`.
 *
 * The fix (PR #174927) changes it to `target_include_directories` to only
 * include headers.
 *
 * References:
 * -----------
 * - Bad commit: c5aa299b048da269e1165216a1ef3cb06edb413d
 * - Fix PR: https://github.com/pytorch/pytorch/pull/174927
 *
 * How to compile (standalone test, not through BUCK):
 * ---------------------------------------------------
 *   g++ -std=c++17 -o cpuinfo_test cpuinfo_initialization_test.cpp \
 *       -I<path_to_cpuinfo>/include \
 *       -L<path_to_cpuinfo>/lib -lcpuinfo -lpthread
 *
 *   # Or with FBGEMM:
 *   g++ -std=c++17 -o cpuinfo_test cpuinfo_initialization_test.cpp \
 *       -I<fbgemm>/include -I<cpuinfo>/include \
 *       -L<fbgemm>/lib -lfbgemm -lcpuinfo -lpthread
 */

#include <cstdlib>
#include <iostream>
#include <string>

// cpuinfo headers
#include <cpuinfo.h>

// Optional: FBGEMM headers if available
#if __has_include("fbgemm/Utils.h")
#include "fbgemm/Utils.h"
#define HAS_FBGEMM 1
#else
#define HAS_FBGEMM 0
#endif

namespace {

// ANSI color codes for output
constexpr const char* RED = "\033[31m";
constexpr const char* GREEN = "\033[32m";
constexpr const char* YELLOW = "\033[33m";
constexpr const char* RESET = "\033[0m";

void print_pass(const std::string& test_name) {
  std::cout << GREEN << "[PASS] " << RESET << test_name << std::endl;
}

void print_fail(const std::string& test_name, const std::string& reason) {
  std::cout << RED << "[FAIL] " << RESET << test_name << ": " << reason
            << std::endl;
}

void print_warn(const std::string& test_name, const std::string& reason) {
  std::cout << YELLOW << "[WARN] " << RESET << test_name << ": " << reason
            << std::endl;
}

void print_info(const std::string& msg) {
  std::cout << "[INFO] " << msg << std::endl;
}

// Check if we're on x86/x86_64
bool is_x86_platform() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
  return true;
#else
  return false;
#endif
}

} // namespace

/**
 * Test 1: Verify cpuinfo_initialize() returns true
 *
 * If cpuinfo has multiple instances, the "wrong" instance may not be
 * properly initialized, causing cpuinfo_initialize() to fail or
 * cpuinfo_is_initialized to be false.
 */
bool test_cpuinfo_initializes() {
  const std::string test_name = "cpuinfo_initialize returns true";

  bool result = cpuinfo_initialize();

  if (result) {
    print_pass(test_name);
    return true;
  } else {
    print_fail(
        test_name,
        "cpuinfo_initialize() returned false. "
        "This may indicate multiple cpuinfo instances with corrupted state.");
    return false;
  }
}

/**
 * Test 2: Verify AVX2 detection is sensible on modern x86_64
 *
 * On any modern x86_64 CPU (Haswell 2013+), AVX2 should be available.
 * If cpuinfo reports no AVX2 on such a system, the cpuinfo_isa struct
 * is likely reading from an uninitialized instance.
 */
bool test_avx2_detection() {
  const std::string test_name = "AVX2 detection sensibility";

  if (!is_x86_platform()) {
    print_warn(test_name, "Skipped - not on x86 platform");
    return true;
  }

  if (!cpuinfo_initialize()) {
    print_fail(test_name, "cpuinfo_initialize() failed");
    return false;
  }

  bool has_avx2 = cpuinfo_has_x86_avx2();
  bool has_avx = cpuinfo_has_x86_avx();
  bool has_fma3 = cpuinfo_has_x86_fma3();

  print_info("CPU feature detection results:");
  print_info("  AVX:  " + std::string(has_avx ? "yes" : "no"));
  print_info("  AVX2: " + std::string(has_avx2 ? "yes" : "no"));
  print_info("  FMA3: " + std::string(has_fma3 ? "yes" : "no"));

  // On x86_64, we expect at least SSE2 (required by the architecture)
  // Most modern CPUs (2013+) have AVX2
  // We can't strictly require AVX2, but we CAN check for consistency

  // Inconsistency check: if AVX2 is present, AVX must also be present
  if (has_avx2 && !has_avx) {
    print_fail(
        test_name,
        "Inconsistent: AVX2 detected but not AVX. "
        "This indicates corrupted cpuinfo state (possibly multiple instances).");
    return false;
  }

  // Check package info is valid (another sign of proper initialization)
  const cpuinfo_package* pkg = cpuinfo_get_packages();
  if (pkg == nullptr) {
    print_fail(
        test_name,
        "cpuinfo_get_packages() returned nullptr. "
        "cpuinfo may not be properly initialized.");
    return false;
  }

  print_info("  CPU: " + std::string(pkg->name));

  print_pass(test_name);
  return true;
}

/**
 * Test 3: Verify AVX512 detection consistency
 *
 * If AVX512F is reported, the other AVX512 features should also be consistent.
 */
bool test_avx512_detection_consistency() {
  const std::string test_name = "AVX512 detection consistency";

  if (!is_x86_platform()) {
    print_warn(test_name, "Skipped - not on x86 platform");
    return true;
  }

  if (!cpuinfo_initialize()) {
    print_fail(test_name, "cpuinfo_initialize() failed");
    return false;
  }

  bool has_avx512f = cpuinfo_has_x86_avx512f();
  bool has_avx512bw = cpuinfo_has_x86_avx512bw();
  bool has_avx512dq = cpuinfo_has_x86_avx512dq();
  bool has_avx512vl = cpuinfo_has_x86_avx512vl();
  bool has_avx512vnni = cpuinfo_has_x86_avx512vnni();

  print_info("AVX512 feature detection results:");
  print_info("  AVX512F:    " + std::string(has_avx512f ? "yes" : "no"));
  print_info("  AVX512BW:   " + std::string(has_avx512bw ? "yes" : "no"));
  print_info("  AVX512DQ:   " + std::string(has_avx512dq ? "yes" : "no"));
  print_info("  AVX512VL:   " + std::string(has_avx512vl ? "yes" : "no"));
  print_info("  AVX512VNNI: " + std::string(has_avx512vnni ? "yes" : "no"));

  // Consistency check: AVX512BW/DQ/VL typically require AVX512F
  if ((has_avx512bw || has_avx512dq || has_avx512vl) && !has_avx512f) {
    print_fail(
        test_name,
        "Inconsistent: AVX512 sub-features detected but not AVX512F. "
        "This indicates corrupted cpuinfo state.");
    return false;
  }

  print_pass(test_name);
  return true;
}

/**
 * Test 4: Verify multiple calls to cpuinfo_initialize return same results
 *
 * With proper single-instance cpuinfo, multiple init calls should be
 * idempotent.
 */
bool test_initialization_idempotency() {
  const std::string test_name = "cpuinfo initialization idempotency";

  // First call
  bool result1 = cpuinfo_initialize();
  bool avx2_1 = cpuinfo_has_x86_avx2();

  // Second call
  bool result2 = cpuinfo_initialize();
  bool avx2_2 = cpuinfo_has_x86_avx2();

  // Third call
  bool result3 = cpuinfo_initialize();
  bool avx2_3 = cpuinfo_has_x86_avx2();

  if (result1 != result2 || result2 != result3) {
    print_fail(
        test_name,
        "cpuinfo_initialize() returned different values on repeated calls");
    return false;
  }

  if (avx2_1 != avx2_2 || avx2_2 != avx2_3) {
    print_fail(
        test_name,
        "cpuinfo_has_x86_avx2() returned different values on repeated calls. "
        "This strongly suggests multiple cpuinfo instances with inconsistent "
        "state.");
    return false;
  }

  print_pass(test_name);
  return true;
}

#if HAS_FBGEMM
/**
 * Test 5: Verify FBGEMM's CPU feature detection matches cpuinfo
 *
 * FBGEMM wraps cpuinfo. If they disagree, multiple instances exist.
 */
bool test_fbgemm_matches_cpuinfo() {
  const std::string test_name = "FBGEMM matches cpuinfo detection";

  if (!is_x86_platform()) {
    print_warn(test_name, "Skipped - not on x86 platform");
    return true;
  }

  if (!cpuinfo_initialize()) {
    print_fail(test_name, "cpuinfo_initialize() failed");
    return false;
  }

  bool cpuinfo_avx2 = cpuinfo_has_x86_avx2();
  bool fbgemm_avx2 = fbgemm::fbgemmHasAvx2Support();

  bool cpuinfo_avx512 = cpuinfo_has_x86_avx512f() &&
      cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq() &&
      cpuinfo_has_x86_avx512vl();
  bool fbgemm_avx512 = fbgemm::fbgemmHasAvx512Support();

  print_info("cpuinfo vs FBGEMM detection:");
  print_info(
      "  AVX2:   cpuinfo=" + std::string(cpuinfo_avx2 ? "yes" : "no") +
      ", FBGEMM=" + std::string(fbgemm_avx2 ? "yes" : "no"));
  print_info(
      "  AVX512: cpuinfo=" + std::string(cpuinfo_avx512 ? "yes" : "no") +
      ", FBGEMM=" + std::string(fbgemm_avx512 ? "yes" : "no"));

  if (cpuinfo_avx2 != fbgemm_avx2) {
    print_fail(
        test_name,
        "AVX2 detection mismatch between cpuinfo and FBGEMM! "
        "This confirms multiple cpuinfo instances are linked.");
    return false;
  }

  if (cpuinfo_avx512 != fbgemm_avx512) {
    print_fail(
        test_name,
        "AVX512 detection mismatch between cpuinfo and FBGEMM! "
        "This confirms multiple cpuinfo instances are linked.");
    return false;
  }

  // Also check FBGEMM's instruction set detection
  fbgemm::inst_set_t isa = fbgemm::fbgemmInstructionSet();
  std::string isa_name;
  switch (isa) {
    case fbgemm::inst_set_t::anyarch:
      isa_name = "anyarch (scalar fallback)";
      break;
    case fbgemm::inst_set_t::avx2:
      isa_name = "avx2";
      break;
    case fbgemm::inst_set_t::avx512:
      isa_name = "avx512";
      break;
    case fbgemm::inst_set_t::avx512_ymm:
      isa_name = "avx512_ymm";
      break;
    case fbgemm::inst_set_t::avx512_vnni:
      isa_name = "avx512_vnni";
      break;
    case fbgemm::inst_set_t::avx512_vnni_ymm:
      isa_name = "avx512_vnni_ymm";
      break;
    default:
      isa_name = "unknown";
  }
  print_info("  FBGEMM instruction set: " + isa_name);

  // If AVX2 is available but FBGEMM uses scalar, something is wrong
  if (cpuinfo_avx2 && isa == fbgemm::inst_set_t::anyarch) {
    print_fail(
        test_name,
        "cpuinfo detects AVX2 but FBGEMM fell back to scalar! "
        "FBGEMM is reading from wrong cpuinfo instance.");
    return false;
  }

  print_pass(test_name);
  return true;
}
#endif // HAS_FBGEMM

int main(int argc, char** argv) {
  std::cout << "=== cpuinfo Initialization Test ===" << std::endl;
  std::cout << "Testing for multiple cpuinfo instance issues..." << std::endl;
  std::cout << std::endl;

  int failed = 0;
  int passed = 0;

  // Run tests
  if (test_cpuinfo_initializes()) {
    passed++;
  } else {
    failed++;
  }

  if (test_avx2_detection()) {
    passed++;
  } else {
    failed++;
  }

  if (test_avx512_detection_consistency()) {
    passed++;
  } else {
    failed++;
  }

  if (test_initialization_idempotency()) {
    passed++;
  } else {
    failed++;
  }

#if HAS_FBGEMM
  if (test_fbgemm_matches_cpuinfo()) {
    passed++;
  } else {
    failed++;
  }
#else
  print_warn("FBGEMM test", "Skipped - FBGEMM headers not available");
#endif

  // Summary
  std::cout << std::endl;
  std::cout << "=== Summary ===" << std::endl;
  std::cout << "Passed: " << passed << std::endl;
  std::cout << "Failed: " << failed << std::endl;

  if (failed > 0) {
    std::cout << std::endl;
    std::cout << RED << "FAILURE: cpuinfo initialization issues detected!"
              << RESET << std::endl;
    std::cout << "This likely indicates multiple cpuinfo instances are linked."
              << std::endl;
    std::cout << "Fix: Ensure PyTorch is built with PR #174927 applied."
              << std::endl;
    return 1;
  }

  std::cout << GREEN << "SUCCESS: All cpuinfo tests passed!" << RESET
            << std::endl;
  return 0;
}

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)

# Shared compiler warning flags for both CPU and GPU builds
#
# Produces up to four lists from ONE source of truth:
#
#   MSVC_FLAGS_VAR   MSVC warning flags
#   CC_FLAGS_VAR     host C/C++ flags (gcc or clang, per CMAKE_CXX_COMPILER_ID)
#   NVCC_FLAGS_VAR   the CC list, each entry wrapped as -Xcompiler=<flag>
#   HIPCC_FLAGS_VAR  the clang-shaped list, for hipcc
#
# Adding a warning to `_cc_common` (or, when it is populated, `_cc_clang_only`)
# must automatically reach CXX, nvcc and hipcc with no further edits. That is the
# entire point of deriving them here rather than at the call sites.
#
# NOTE: this function PRODUCES the nvcc/hipcc lists; it does not apply them.
# Wiring them into COMPILE_OPTIONS / HIPCC_OPTIONS is deliberately separate, so a
# bad activation can be reverted without losing this refactor.
function(fbgemm_get_warning_flags)
  cmake_parse_arguments(ARG ""
    "MSVC_FLAGS_VAR;CC_FLAGS_VAR;NVCC_FLAGS_VAR;HIPCC_FLAGS_VAR"
    "EXTRA_MSVC_FLAGS;EXTRA_CC_FLAGS" ${ARGN})

  # MSVC flags
  set(_msvc
    ${ARG_EXTRA_MSVC_FLAGS}
    /wd4244
    /wd4267
    /wd4305
    /wd4309)

  # Portable warning flags. Additions that apply to every compiler go here.
  #
  # The `-Wall`-implied group below is redundant with `-Wall` today. It is listed
  # explicitly on purpose: the flags survive any future narrowing of `-Wall`.
  # Do not "clean up" as redundant.
  #
  # Every flag here must be accepted by BOTH gcc and clang -- the OSS CI matrix
  # builds with each. A clang-only flag belongs in `_cc_clang_only`; putting one
  # here makes gcc emit "unrecognized command line option", which `-Werror`
  # turns into a build failure.
  set(_cc_common
    -Wall
    -Wextra
    -Werror
    # -Wall-implied
    -Waddress
    -Wenum-compare
    -Wmisleading-indentation
    -Wparentheses
    # -Wall-implied. Note the PLURAL -Wunused-local-typedefs: clang's singular
    # spelling is rejected by g++ outright. The plural is accepted by both and
    # produces the same diagnostic, so it goes here rather than in
    # `_cc_clang_only`, buying coverage on the gcc leg.
    -Wpessimizing-move
    -Wunused-label
    -Wunused-local-typedefs
    # `-Wall`-implied and therefore inert on the host surface (clang enables it
    # by default; gcc via -Wall). Its value is on DEVICE code, which reaches it
    # via `_hipcc` -- but only once that list is applied. Until then this is
    # host-only and should produce nothing.
    -Wunused-value
    # Init and control flow. `-Winfinite-recursion` and `-Wself-assign` are
    # g++-rejected and live in `_cc_clang_only`.
    -Wimplicit-fallthrough
    -Wuninitialized
    # g++ accepts `-Wvexing-parse`, so it goes in the portable bucket for
    # gcc-leg coverage.
    -Wvexing-parse
    # This one is NOT free. `-Wall` implies it on clang but NOT on gcc
    # (measured: g++ -Wall gives 0 diagnostics on a missing-braces probe, 1 only
    # when the flag is explicit), so it genuinely enables a new warning on the
    # OSS gcc host leg, under `-Werror`.
    #
    # It is portable, so `_cc_common` is the correct bucket. HIP is unaffected,
    # because hipcc is clang and `_hipcc` already carries `-Wall`.
    -Wmissing-braces
    # Portable: accepted and diagnosing on BOTH g++ 11.5 and clang 22
    # (probed). Subplan 04 grouped it with the clang-only A2.2 flags, which
    # would have stranded it behind the clang guard and silently lost gcc-leg
    # coverage. It belongs here.
    -Wmismatched-tags
    # ---- A2.4 (portable half) ------------------------------------------
    # Accepted and diagnosing on BOTH g++ 11.5 and clang 22 (probed), so it
    # goes here rather than behind the clang guard. Subplan 04 called this out
    # correctly.
    -Waddress-of-packed-member
    # ---- Phase B2: shadowing ----------------------------------------------
    # Portable: probed accepted AND diagnosing on both g++ 11.5 and clang 22
    # (the bare flag; -Wshadow-all, -Wshadow-uncaptured-local and
    # -Wshadow-field-in-constructor are clang-only and are NOT added here).
    #
    # This is enabled DEMOTED -- see -Wno-error=shadow in the shared
    # suppressions. The B0 census measured 299 first-party diagnostics across
    # 22 files, so enabling it at full strength would break the build outright.
    # Demoting makes those 299 visible in CI and keeps the census reproducible
    # instead of being a one-off local probe, while the fixups land
    # incrementally. The escape comes out when the count reaches zero.
    #
    # -Wshadow is the right first Phase B flag on two measurements: it is the
    # most concentrated (79% of diagnostics in 5 files) and it is the only one
    # of the three with ZERO third-party exposure, so unlike
    # -Wshorten-64-to-32 and -Wzero-as-null-pointer-constant it is not blocked
    # on giving PyTorch/ATen/c10 headers -isystem treatment.
    -Wshadow
    # ---- Phase B3: zero as null pointer -----------------------------------
    # Portable: probed accepted AND diagnosing on g++ 11.5 and clang 22.
    # Enabled DEMOTED -- see -Wno-error=zero-as-null-pointer-constant.
    #
    # B0 measured 449 first-party diagnostics across 130 files. That is the
    # LARGEST job of the three B flags despite having ~260 fewer diagnostics
    # than -Wshorten-64-to-32, because it is diffuse: only 19% sit in the top
    # five files, versus 79% for -Wshadow. Rank by file count and
    # concentration, not by raw count.
    -Wzero-as-null-pointer-constant
    # ---- Parity gap G1: unused-family flags fbcode enables globally ---------
    # Probed accepted on g++ 11.5 AND clang 22, so portable.
    # -Wunused-variable and -Wunused-but-set-variable are already [enabled] by
    # -Wall/-Wextra (verified with `g++ -Wall -Wextra -Q --help=warning`), so
    # naming them explicitly is parity, not a behaviour change. It also makes
    # the pre-existing -Wno-error=unused-but-set-variable escapes meaningful:
    # an escape without its positive flag is inert.
    # -Wunused-const-variable IS a real addition (gcc maps it to level =2).
    -Wunused-variable
    -Wunused-const-variable
    -Wunused-but-set-variable)

  # Clang-only warning flags. These are appended to `_cc` ONLY when the host
  # compiler is clang (see the guarded append below), because the OSS CI matrix
  # builds with gcc on half its legs and g++ treats an unknown `-W` as a hard
  # error -- not a warning, so `-Werror` is not even required to break the build.
  #
  # They ARE included unconditionally in `_hipcc`, because hipcc is always clang.
  #
  # Prefer a portable spelling in `_cc_common` over a clang-only one here when
  # both exist: `-Wunused-local-typedefs` (plural) is accepted and diagnoses on
  # both compilers, whereas clang's `-Wunused-local-typedef` (singular) is a hard
  # error on gcc. Verified by compiling a probe with each.
  set(_cc_clang_only
    # clang has no gcc equivalent for this one.
    -Wmove
    # g++ rejects both of these outright.
    #
    # `-Winfinite-recursion` is `-Wall`-implied in clang (measured: 0 diagnostics
    # at baseline, 1 with `-Wall`), so the OSS clang leg already has it and is
    # green. No suppression is needed.
    -Winfinite-recursion
    -Wself-assign
    # g++ rejects these two; only `-Wvexing-parse` is portable.
    -Wnull-conversion
    -Wstring-concatenation
    # ---- A2.1: deprecated C++ constructs -------------------------------
    # All four are rejected outright by g++ 11.5 ("unrecognized command line
    # option"), so they must stay behind the clang guard. Probed, not assumed.
    #
    # `-Wdeprecated-dynamic-exception-spec` is NOT on by default: measured
    # `-Wall -Wextra -Werror` against a `throw()` declaration and got a clean
    # compile, so this flag genuinely enables a new diagnostic rather than
    # restating one. See the paired `-Wno-error=` escape in
    # `_cc_suppressions_clang_base` for why it is demoted.
    -Wdeprecated-dynamic-exception-spec
    # `[=]` implicitly capturing `this` is deprecated in C++20 and FBGEMM is a
    # C++20 codebase with heavy lambda use around kernel dispatch. If this
    # fires, the fix is `[=, this]` or an explicit capture list -- not a
    # suppression.
    -Wdeprecated-this-capture
    -Wdeprecated-copy-with-user-provided-copy
    # ---- A2.2: type and template hygiene -------------------------------
    # g++ 11.5 rejects all three (probed). `-Wmismatched-tags`, the fourth
    # member of this chunk in subplan 04, is portable and lives in
    # `_cc_common` instead.
    #
    # `-Wundefined-var-template` may fire where an explicit instantiation lives
    # in a different TU; the fix is an explicit instantiation declaration, not
    # a suppression.
    -Wundefined-var-template
    # Warns where clang accepts something gcc will not. Genuinely useful here
    # because OSS builds with both, and this flag is the only one in the set
    # that actively protects the gcc leg from clang-only constructs.
    -Wgcc-compat
    # ---- A2.3: conversions and literals --------------------------------
    # All four are g++-rejected (probed).
    #
    # `-Wstring-conversion` is already enabled for `deeplearning/` in fbcode
    # and globally at 100%, so this is OSS-only catch-up.
    -Wstring-conversion
    -Wimplicitly-unsigned-literal
    -Wuninitialized-const-reference
    # Inert unless the code carries clang thread-safety annotations
    # (`guarded_by` and friends). FBGEMM does not use them, so this is list
    # parity rather than new coverage. Kept because it costs nothing and the
    # annotations may arrive later.
    -Wthread-safety
    # ---- A2.4 (clang half) ---------------------------------------------
    # Intentionally enable clang's full lifetime-diagnostic umbrella. g++ 11.5
    # has no bare `-Wdangling` (it has `-Wdangling-pointer` and
    # `-Wdangling-reference` instead), so this spelling is clang-only.
    -Wdangling
    # ---- A2.5: shift-sign-overflow -------------------------------------
    # The first genuinely risky flag in Phase A. Fires on signed left-shift
    # overflow, and FBGEMM is full of hand-written bit manipulation in
    # AVX2/AVX512/NEON/SVE intrinsic paths and in quantization code. The fix is
    # casting to unsigned before shifting -- and a wrong "fix" in that code is
    # a silent numerical bug, not a compile error.
    #
    # Deliberately alone in its own diff so review can focus. If the OSS legs
    # report a non-trivial count, split the fixups by ISA
    # (src/*Avx2.cc, src/*Avx512.cc, src/*Neon.cc, src/*Sve.cc) and treat it as
    # Phase B work rather than a parity chunk.
    -Wshift-sign-overflow
    # ---- A2.6a: unused exception parameter ------------------------------
    # g++-rejected (probed). Already on for `deeplearning/` in fbcode and at
    # 100% globally, so OSS-only catch-up.
    #
    # NOTE: the other half of subplan 04's A2.6, `-Wheader-hygiene`, is NOT
    # here. It reaches HIP device code through `_hipcc`, and fbcode carries
    # `-Wno-header-hygiene` in `_hip_warning_suppressions` precisely because it
    # fires on ROCm/CK headers. Enabling it needs a HIP-only suppression list
    # that does not exist yet -- see the A2.6b notes in subplan 04.
    -Wunused-exception-parameter
    # ---- A2.6b: header hygiene ------------------------------------------
    # `using namespace` at header scope. g++-rejected (probed), so clang-only.
    #
    # This reaches HIP DEVICE code via `_hipcc`, and fbcode carries a blanket
    # `-Wno-header-hygiene` in `_hip_warning_suppressions` because it fires on
    # ROCm/CK headers. We do NOT mirror that blanket suppression here: measured
    # that `-isystem` fully neutralises this diagnostic (an `-I` include of a
    # header-scope `using namespace` errors; the same header via `-isystem` is
    # silent), and subplan 01 already moved CUTLASS/CK/asmjit/ATen to SYSTEM
    # includes. The blanket suppression would therefore give up device coverage
    # we may no longer need to give up.
    #
    # Instead the HIP path gets `-Wno-error=header-hygiene` (see
    # `_hipcc_suppressions`), which keeps the diagnostic VISIBLE so 02c.3's
    # census can count it, without breaking the build if some CK header is
    # still reached via `-I`.
    -Wheader-hygiene
    # ---- Phase B4: 64-to-32 narrowing -------------------------------------
    # clang-only: g++ 11.5 rejects the flag outright (probed).
    # Enabled DEMOTED -- see -Wno-error=shorten-64-to-32 in the clang
    # suppressions, which must ALSO be clang-guarded (see there).
    #
    # B0 measured 711 first-party diagnostics across 76 files, 55% in the top
    # five. Subplan 09 calls this the hardest chunk in the project: the fixups
    # are casts in index arithmetic, where a wrong cast is a silent numerical
    # or OOB bug rather than a compile error. Enabling it demoted makes the
    # backlog visible without forcing rushed casts.
    -Wshorten-64-to-32)

  # Clang-only flags that also need a RECENT clang. An older clang does not
  # know these flags, and an unknown `-W` option stops the build when `-Werror`
  # is present. The CXX path adds them only when the host clang is new enough.
  #
  # Use `VERSION_GREATER_EQUAL`. `VERSION_GREATER 16.0.0` is true for clang
  # 16.0.6, so that form does not exclude clang 16.
  set(_cc_clang_only_gt17
    # clang 17 added this flag. clang 16 and older stop with an error. No older
    # clang has a flag with the same meaning, so a gate is the only repair.
    -Wdeprecated-redundant-constexpr-static-def
    # clang 17 added this flag too. clang 16 and older stop with an error.
    -Wpacked-non-pod)

  # Suppressions. These are appended LAST so they win over everything above.
  #
  # To ENABLE a warning suppressed here, DELETE it from these lists. Adding the
  # positive flag to `_cc_common` will not work; these come later.
  set(_cc_suppressions_common
    -Wno-deprecated-declarations
    -Wno-deprecated-enum-enum-conversion
    -Wno-strict-aliasing
    -Wno-sign-compare
    -Wno-vla
    -Wno-error=unused-parameter
    -Wno-error=unknown-pragmas
    -Wno-error=attributes
    # Phase B2 is landing demoted while its 299 first-party diagnostics are
    # fixed; see -Wshadow in _cc_common. Portable escape -- probed demoting to
    # a warning on both g++ and clang, unlike the A2.1 escape which g++
    # hard-errors on and which therefore had to be clang-guarded.
    # TODO(T169200065): delete once the -Wshadow count reaches zero. This is
    # the whole point of enabling it demoted rather than silently.
    -Wno-error=shadow
    # Phase B3, landing demoted while its 449 diagnostics across 130 files
    # are fixed; see -Wzero-as-null-pointer-constant in _cc_common. Portable
    # escape -- probed demoting to a warning on both g++ and clang.
    #
    # This escape also covers the 48 diagnostics coming from PyTorch / ATen /
    # c10 / folly / thrift headers, which are NOT FBGEMM's to fix. Those need
    # -isystem treatment (unowned work that B0 newly identified) before this
    # flag can go to hard error, independently of the first-party backlog.
    # TODO(T169200065): delete once the count reaches zero AND the external
    # headers are -isystem.
    -Wno-error=zero-as-null-pointer-constant)

  # Clang suppressions, split by the clang version that made them necessary.
  # The CXX path applies these conditionally on the HOST clang version (below);
  # the hipcc path takes all of them, because hipcc is always a recent clang.
  set(_cc_suppressions_clang_base
    -Wno-unused-command-line-argument
    -Wno-c99-extensions
    -Wno-gnu-zero-variadic-macro-arguments
    # CUDA 13.3's `crt/host_runtime.h` declares `atexit(...) throw()`, a
    # deprecated dynamic exception spec. It is an NVIDIA system header we
    # cannot edit, and since 02c.1 forwards the host warning set to nvcc via
    # `-Xcompiler=`, nvcc's host pass sees it too. fbcode carries the identical
    # escape in `buck2/platform/cxx/fbcode/warnings.bzl`
    # (`CLANG_WARNINGS_TO_DISABLE`), so fbcode is not clean for this flag
    # either -- it is enabled and demoted, exactly as here.
    #
    # This MUST live in the clang-guarded list. g++ hard-errors on
    # `-Wno-error=<warning it does not know>`:
    #   cc1plus: error: '-Wno-error=deprecated-dynamic-exception-spec':
    #                   no option '-Wdeprecated-dynamic-exception-spec'
    # Note that this is stricter than a bare unknown `-Wno-<name>`, which g++
    # accepts silently. The leniency does not extend to the `=` form.
    # TODO(T169200065): drop once the toolchain stops surfacing this header.
    -Wno-error=deprecated-dynamic-exception-spec
    # Phase B4, landing demoted; see -Wshorten-64-to-32 in _cc_clang_only.
    #
    # This escape MUST stay clang-guarded. g++ hard-errors on
    # -Wno-error=<warning it does not know>, even without -Werror:
    #   cc1plus: error: '-Wno-error=shorten-64-to-32': no option
    #                   '-Wshorten-64-to-32'
    # Same landmine as the A2.1 escape. Putting it in _cc_suppressions_common
    # would break every gcc leg of the OSS matrix.
    # TODO(T169200065): delete once the count reaches zero.
    -Wno-error=shorten-64-to-32)

  set(_cc_suppressions_clang_gt13
    -Wno-error=unused-but-set-parameter
    -Wno-error=unused-but-set-variable)

  set(_cc_suppressions_clang_gt17
    -Wno-vla-cxx-extension
    -Wno-error=global-constructors)

  # Full clang-shaped suppression set, assembled unconditionally so it is
  # available even when the HOST compiler is GCC. Used only for the hipcc list.
  set(_cc_suppressions_clang
    ${_cc_suppressions_clang_base}
    ${_cc_suppressions_clang_gt13}
    ${_cc_suppressions_clang_gt17})

  set(_cc_suppressions_gcc
    -Wno-error=unused-but-set-parameter
    -Wno-error=unused-but-set-variable
    -Wno-error=array-bounds
    -Wno-error=maybe-uninitialized)

  # Host-compiler-conditional suppression set for the CXX path. The version gates
  # are preserved exactly as before this refactor.
  set(_cc_suppressions ${_cc_suppressions_common})

  if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    list(APPEND _cc_suppressions ${_cc_suppressions_clang_base})

    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 13.0.0)
      list(APPEND _cc_suppressions ${_cc_suppressions_clang_gt13})
    endif()

    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 17.0.0)
      list(APPEND _cc_suppressions ${_cc_suppressions_clang_gt17})
    endif()

  # GNU-specific
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    list(APPEND _cc_suppressions ${_cc_suppressions_gcc})
  endif()

  # NOTE: ARG_EXTRA_CC_FLAGS stays FIRST, preserving the pre-refactor behaviour
  # that per-target extras are overridable by the shared suppressions.
  #
  # `_cc_clang_only` is guarded: it must not reach a gcc command line. Now that
  # the list is populated the guard is load-bearing -- `-Wmove` is clang-only and
  # g++ rejects it outright.
  set(_cc
    ${ARG_EXTRA_CC_FLAGS}
    ${_cc_common})

  if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    list(APPEND _cc ${_cc_clang_only})
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 17.0.0)
      list(APPEND _cc ${_cc_clang_only_gt17})
    endif()
  endif()

  list(APPEND _cc ${_cc_suppressions})

  # nvcc does not understand host warning flags directly; they must be forwarded
  # to the host compiler. Use the `-Xcompiler=<flag>` single-token form: the
  # two-token `-Xcompiler <flag>` form can be split or de-duplicated when CMake
  # expands a `;`-separated list into COMPILE_OPTIONS.
  set(_nvcc "")
  foreach(_flag IN LISTS _cc)
    list(APPEND _nvcc "-Xcompiler=${_flag}")
  endforeach()

  # hipcc is ALWAYS clang, regardless of CMAKE_CXX_COMPILER_ID -- the OSS ROCm CI
  # matrix builds with both gcc and clang as the HOST compiler
  # (.github/workflows/fbgemm_gpu_ci_rocm.yml). So this list is built from the
  # clang-shaped inputs unconditionally. Deriving it from `_cc` would silently
  # drop every clang-only flag on the gcc leg, with no failure to signal it.
  #
  # Unlike nvcc, these reach DEVICE code -- the only surface where they do.
  #
  # ⚠ `_cc_suppressions_clang` here is the FULL clang set, including the entries
  # the CXX path gates behind CMAKE_CXX_COMPILER_VERSION. Those gates describe the
  # HOST compiler and say nothing about hipcc's clang, which is a separate and
  # generally newer toolchain. Before these flags are ever applied, confirm the
  # hipcc in the supported ROCm versions accepts all of them -- an unknown
  # `-Wno-*` becomes an error under `-Werror`. Producing the list is harmless;
  # applying it is not.
  # `_cc_suppressions_common` is included deliberately. Omitting it would give
  # hipcc `-Werror` while withholding -Wno-deprecated-declarations,
  # -Wno-strict-aliasing, -Wno-sign-compare, -Wno-vla and the -Wno-error=*
  # entries that the host path relies on, and applying such a list would fail
  # immediately. The common suppressions are portable `-Wno-*` that hipcc, being
  # clang, accepts. Including them makes `_hipcc` the clang analogue of `_cc`:
  # everything except the caller's EXTRA_CC_FLAGS, with the FULL clang
  # suppression set in place of the host-conditional one.

  # Suppressions that apply ONLY to the hipcc/device list.
  #
  # `_hipcc` otherwise shares the host suppression lists, so before this there
  # was no way to relax a flag on device code without also relaxing it on the
  # host. `-Wheader-hygiene` is the first flag that needs exactly that.
  #
  # Appended LAST in `_hipcc` so it wins, matching the ordering contract the
  # host path uses.
  set(_hipcc_suppressions
    # ROCm/CK headers use `using namespace` at header scope. fbcode silences
    # this outright for HIP (`_hip_warning_suppressions` in build_utils.bzl);
    # we demote instead of silencing so the diagnostic still appears in the
    # 02c.3 census and we can tell when it reaches zero.
    #
    # Measured: `-isystem` alone neutralises this diagnostic, and subplan 01
    # already made the third-party include dirs SYSTEM, so this escape may
    # already be unnecessary. It is kept until a full ROCm census confirms a
    # zero count -- 02c.3 has only covered 13% of the build so far.
    # TODO(T169200065): drop once the ROCm census shows zero header-hygiene.
    -Wno-error=header-hygiene)

  set(_hipcc
    ${_cc_common}
    ${_cc_clang_only}
    # hipcc is a recent clang in every supported ROCm version, so it also gets
    # the gated flags. The same rule already applies to the gated suppressions
    # in `_cc_suppressions_clang` below.
    ${_cc_clang_only_gt17}
    ${_cc_suppressions_common}
    ${_cc_suppressions_clang}
    ${_hipcc_suppressions})

  set(${ARG_MSVC_FLAGS_VAR} ${_msvc} PARENT_SCOPE)
  set(${ARG_CC_FLAGS_VAR}   ${_cc}   PARENT_SCOPE)

  if(ARG_NVCC_FLAGS_VAR)
    set(${ARG_NVCC_FLAGS_VAR} ${_nvcc} PARENT_SCOPE)
  endif()

  if(ARG_HIPCC_FLAGS_VAR)
    set(${ARG_HIPCC_FLAGS_VAR} ${_hipcc} PARENT_SCOPE)
  endif()
endfunction()

function(cpp_library)
    # NOTE: This function is meant for building targets in FBGEMM, not
    # FBGEMM_GPU or FBGEMM GenAI, which have much more complicated setups.
    #
    # This function does the following:
    #
    #   1. Builds the .SO file for the target
    #   1. Handles MSVC-specific compilation flags
    #   1. Handles dependencies linking
    #   1. Adds common target properties as needed
    #   1. Adds the target to the install package

    set(flags)
    set(singleValueArgs
        PREFIX              # Desired name for the library target (and by extension, the prefix for naming intermediate targets)
        TYPE                # Target type, e.g., MODULE, OBJECT.  See https://cmake.org/cmake/help/latest/command/add_library.html
        DESTINATION         # The install destination directory to place the build target into
        ENABLE_IPO          # Whether to enable interprocedural optimization (IPO) for the target
        SANITIZER_OPTIONS   # Sanitizer options to pass to the target
    )
    set(multiValueArgs
        SRCS            # Sources for CPU-only build
        CC_FLAGS        # General compilation flags applicable to all build variants
        MSVC_FLAGS      # Compilation flags specific to MSVC
        DEFINITIONS     # Preprocessor definitions
        INCLUDE_DIRS    # First-party include directories for compilation
        SYSTEM_INCLUDE_DIRS # Third-party include directories, passed as SYSTEM to suppress their warnings
        DEPS            # Target dependencies, i.e. built STATIC targets
    )

    cmake_parse_arguments(
        args
        "${flags}" "${singleValueArgs}" "${multiValueArgs}"
        ${ARGN})

    ############################################################################
    # Prepare Sources
    ############################################################################

    # Set the build target sources
    set(lib_sources ${args_SRCS})

    # If the sources list is empty, add a placeholder source file so that the
    # library can be built without failure
    if(NOT lib_sources)
        # Create a salt value
        STRING(RANDOM LENGTH 6 salt)

        # Generate a placeholder source file
        file(WRITE ${CMAKE_BINARY_DIR}/gen_placeholder_${salt}.cc "")

        # Append to lib_sources
        list(APPEND lib_sources
            ${CMAKE_BINARY_DIR}/gen_placeholder_${salt}.cc)
    endif()

    ############################################################################
    # Build the Library
    ############################################################################

    # Set the build target name
    set(lib_name ${args_PREFIX})

    # Create the library
    add_library(${lib_name} ${args_TYPE}
        ${lib_sources})

    ############################################################################
    # Compilation Flags and Definitions
    ############################################################################

    if(MSVC)
        # MSVC needs to define these variables to avoid generating _dllimport
        # functions.
        if(args_TYPE STREQUAL STATIC)
            target_compile_definitions(${lib_name}
                PUBLIC ASMJIT_STATIC
                PUBLIC FBGEMM_STATIC)
        endif()

        fbgemm_get_warning_flags(
            MSVC_FLAGS_VAR  _msvc_flags
            CC_FLAGS_VAR    _cc_flags
            NVCC_FLAGS_VAR  _nvcc_warning_flags
            HIPCC_FLAGS_VAR _hipcc_warning_flags
            EXTRA_MSVC_FLAGS ${args_MSVC_FLAGS}
            EXTRA_CC_FLAGS   ${args_CC_FLAGS})
        set(lib_cc_flags ${_msvc_flags})

    else()
        fbgemm_get_warning_flags(
            MSVC_FLAGS_VAR  _msvc_flags
            CC_FLAGS_VAR    _cc_flags
            NVCC_FLAGS_VAR  _nvcc_warning_flags
            HIPCC_FLAGS_VAR _hipcc_warning_flags
            EXTRA_MSVC_FLAGS ${args_MSVC_FLAGS}
            EXTRA_CC_FLAGS   ${args_CC_FLAGS})
        set(lib_cc_flags ${_cc_flags})
    endif()

    target_compile_options(${lib_name} PRIVATE
        ${lib_cc_flags})

    if(args_DEFINITIONS)
        target_compile_definitions(${lib_name}
            PUBLIC ${args_DEFINITIONS})
    endif()

    ############################################################################
    # Library Includes and Linking
    ############################################################################

    # Add the include directories
    target_include_directories(${lib_name} PUBLIC
        ${args_INCLUDE_DIRS})

    # Third-party include directories are marked SYSTEM so their diagnostics do
    # not surface in this target.
    target_include_directories(${lib_name} SYSTEM PUBLIC
        ${args_SYSTEM_INCLUDE_DIRS})

    # Link against the external libraries as needed
    target_link_libraries(${lib_name} PUBLIC ${args_DEPS})

    # Link against OpenMP if available
    if(OpenMP_FOUND)
        target_link_libraries(${lib_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # Add sanitizer options if needed
    if(args_SANITIZER_OPTIONS)
        target_link_options(${lib_name} PUBLIC
            "-fsanitize=${args_SANITIZER_OPTIONS}"
            -fno-omit-frame-pointer)
        target_compile_options(${lib_name} PUBLIC
            "-fsanitize=${args_SANITIZER_OPTIONS}"
            -fno-omit-frame-pointer)
    endif()

    # Set PIC
    set_target_properties(${lib_name} PROPERTIES
        # Enforce -fPIC for STATIC library option, since they are to be
        # integrated into other libraries down the line
        # https://stackoverflow.com/questions/3961446/why-does-gcc-not-implicitly-supply-the-fpic-flag-when-compiling-static-librarie
        POSITION_INDEPENDENT_CODE ON)

    # Set IPO
    if(args_ENABLE_IPO)
        set_target_properties(${lib_name} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION ON)
    endif()

    ############################################################################
    # Add to Install Package
    ############################################################################

    if(args_DESTINATION)
        set(lib_install_destination ${args_DESTINATION})
    else()
        set(lib_install_destination ${CMAKE_INSTALL_LIBDIR})
    endif()

    install(
        TARGETS ${lib_name}
        EXPORT fbgemmLibraryConfig
        ARCHIVE DESTINATION ${lib_install_destination}
        LIBRARY DESTINATION ${lib_install_destination}
        # For Windows
        RUNTIME DESTINATION ${lib_install_destination})

    ############################################################################
    # Set the Output Variable(s)
    ############################################################################

    set(${args_PREFIX} ${lib_name} PARENT_SCOPE)

    ############################################################################
    # Debug Summary
    ############################################################################

    BLOCK_PRINT(
        "CPP Library Target: ${args_PREFIX} (${args_TYPE})"
        " "
        "SRCS:"
        "${args_SRCS}"
        " "
        "CC_FLAGS:"
        "${args_CC_FLAGS}"
        " "
        "RESOLVED WARNING FLAGS (CC):"
        "${_cc_flags}"
        " "
        "RESOLVED WARNING FLAGS (NVCC, produced not applied):"
        "${_nvcc_warning_flags}"
        " "
        "RESOLVED WARNING FLAGS (HIPCC, produced not applied):"
        "${_hipcc_warning_flags}"
        " "
        "MSVC_FLAGS:"
        "${args_MSVC_FLAGS}"
        " "
        "DEFINITIONS:"
        "${args_DEFINITIONS}"
        " "
        "ENABLE_IPO: "
        "${args_ENABLE_IPO}"
        " "
        "INCLUDE_DIRS:"
        "${args_INCLUDE_DIRS}"
        " "
        "SYSTEM_INCLUDE_DIRS:"
        "${args_SYSTEM_INCLUDE_DIRS}"
        " "
        "Library Dependencies:"
        "${args_DEPS}"
        " "
        "Output Library:"
        "${lib_name}"
        " "
        "Install Destination:"
        "${lib_install_destination}"
    )
endfunction()

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
    -Wuninitialized)

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
    -Wself-assign)

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
    -Wno-error=attributes)

  # Clang suppressions, split by the clang version that made them necessary.
  # The CXX path applies these conditionally on the HOST clang version (below);
  # the hipcc path takes all of them, because hipcc is always a recent clang.
  set(_cc_suppressions_clang_base
    -Wno-unused-command-line-argument
    -Wno-c99-extensions
    -Wno-gnu-zero-variadic-macro-arguments)

  set(_cc_suppressions_clang_gt13
    -Wno-error=unused-but-set-parameter
    -Wno-error=unused-but-set-variable)

  set(_cc_suppressions_clang_gt17
    -Wno-vla-cxx-extension
    -Wno-error=global-constructors
    -Wno-error=shadow)

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
  set(_hipcc
    ${_cc_common}
    ${_cc_clang_only}
    ${_cc_suppressions_common}
    ${_cc_suppressions_clang})

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

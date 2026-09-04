# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)

# Shared compiler warning flags for both CPU and GPU builds
#
# This function makes up to four lists from one set of flags:
#
#   MSVC_FLAGS_VAR   MSVC warning flags
#   CC_FLAGS_VAR     host C/C++ flags (gcc or clang, per CMAKE_CXX_COMPILER_ID)
#   NVCC_FLAGS_VAR   the CC list, each flag with an -Xcompiler= prefix
#   HIPCC_FLAGS_VAR  the list for hipcc, which is always clang
#
# Add a warning to `_cc_common`, or to `_cc_clang_only`. The warning then
# reaches CXX, nvcc and hipcc. No other file needs a change. This is the reason
# to make the lists here, and not at each call site.
#
# This function makes the lists. Other files apply them. The two steps stay
# separate, so a change to one of them is easy to reverse.
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

  # Portable flags. g++ and clang both accept these flags. A test with each
  # compiler shows this. Do not assume the result from the name of a flag.
  #
  # Use this list, and not `_cc_clang_only`, when both compilers accept a
  # flag. A flag in this list also applies to the gcc builds.
  #
  # `-Wall` includes some of the flags below. This list gives their names,
  # because a later version of `-Wall` can remove them. Do not delete them.
  set(_cc_common
    -Wall
    -Wextra
    -Werror
    -Waddress
    -Wenum-compare
    -Wmisleading-indentation
    -Wparentheses
    -Wpessimizing-move
    -Wunused-label
    # Use the plural name. g++ stops with an error if it gets the singular
    # name `-Wunused-local-typedef`. Both compilers accept the plural name,
    # and both give the same warning.
    -Wunused-local-typedefs
    -Wunused-value
    -Wimplicit-fallthrough
    -Wuninitialized
    -Wvexing-parse
    # `-Wall` includes this flag on clang, but not on gcc. The flag
    # therefore adds a new warning to the gcc builds.
    -Wmissing-braces
    -Wmismatched-tags
    -Waddress-of-packed-member
    # The two flags below are on, but they do not stop the build. See the
    # `-Wno-error=` lines in `_cc_suppressions_common`. Those lines give the
    # condition to remove them.
    -Wshadow
    -Wzero-as-null-pointer-constant
    -Wunused-variable
    -Wunused-const-variable
    -Wunused-but-set-variable)

  # Clang-only flags. g++ does not know these flags. g++ stops with an error
  # when it gets an unknown `-W` option. This error occurs even when `-Werror`
  # is absent. The code therefore adds these flags to `_cc` only when the host
  # compiler is clang.
  #
  # `_hipcc` always contains these flags, because hipcc is always clang.
  #
  # Put the `-Wno-error=` form of these flags in this list too. g++ stops with
  # an error if it gets `-Wno-error=` for a warning that it does not know. g++
  # accepts an unknown `-Wno-` form without an error, but it does not accept
  # the `-Wno-error=` form.
  set(_cc_clang_only
    -Wmove
    -Winfinite-recursion
    -Wself-assign
    -Wnull-conversion
    -Wstring-concatenation
    # This flag is on, but it does not stop the build. See
    # `_cc_suppressions_clang_base`.
    -Wdeprecated-dynamic-exception-spec
    # C++20 deprecates the capture of `this` with `[=]`. To correct a site,
    # use `[=, this]` or a full capture list. Do not add a suppression.
    -Wdeprecated-this-capture
    -Wdeprecated-copy-with-user-provided-copy
    -Wundefined-var-template
    # This flag finds code that only clang accepts. It protects the gcc
    # builds.
    -Wgcc-compat
    -Wstring-conversion
    -Wimplicitly-unsigned-literal
    -Wuninitialized-const-reference
    # This flag does nothing until the code uses clang thread-safety
    # annotations.
    -Wthread-safety
    # This is clang's full lifetime-diagnostic umbrella. g++ has only
    # `-Wdangling-pointer` and `-Wdangling-reference`.
    -Wdangling
    # This flag finds a signed left shift. The kernels contain much bit
    # manipulation, so the warning can occur frequently. To correct a site,
    # change the value to unsigned before the shift. A wrong change makes a
    # silent numerical error, and the compiler does not report it.
    -Wshift-sign-overflow
    -Wunused-exception-parameter
    # This flag finds a `using namespace` line in a header. `-isystem` stops
    # the warning completely, so it does not occur for third-party headers.
    # On the HIP path the warning does not stop the build. See
    # `_hipcc_suppressions`.
    -Wheader-hygiene
    # This flag is on, but it does not stop the build. See
    # `_cc_suppressions_clang_base`.
    -Wshorten-64-to-32
    # The four flags below do nothing for C++, CUDA or HIP code. They apply
    # to Objective-C methods and to C function pointers. This list keeps them,
    # so that the set of flags stays complete.
    -Wdeprecated-implementations
    -Wsemicolon-before-method-body
    -Wimport-preprocessor-directive-pedantic
    -Wincompatible-function-pointer-types
    -Wambiguous-reversed-operator
    -Wbitwise-instead-of-logical
    -Wunreachable-code-fallthrough
    # This is the singular name. This list keeps it, so that the set of
    # flags stays complete. The plural name in `_cc_common` already applies to
    # both compilers.
    -Wunused-local-typedef)

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

  # Suppressions. The code adds these lines last, so they have priority over
  # all the lists above.
  #
  # To enable a warning in these lists, delete its line. Do not add the flag
  # to `_cc_common`. These lists come later, so they have priority.
  #
  # `-Wno-<name>` hides a warning completely. `-Wno-error=<name>` shows the
  # warning but does not let it stop the build. Use the second form, because
  # it keeps the work visible. A `-Wno-error=` line does nothing if no list
  # above enables the warning.
  set(_cc_suppressions_common
    -Wno-deprecated-declarations
    -Wno-deprecated-enum-enum-conversion
    -Wno-strict-aliasing
    -Wno-sign-compare
    -Wno-vla
    -Wno-error=unused-parameter
    -Wno-error=unknown-pragmas
    -Wno-error=attributes
    # Both compilers accept this line. Remove it when the warning count is
    # zero.
    -Wno-error=shadow
    # Both compilers accept this line. The warning also occurs in
    # third-party headers, and those headers need `-isystem`. Remove this line
    # when the count is zero and the headers are system includes.
    -Wno-error=zero-as-null-pointer-constant)

  # Clang suppressions. The clang version controls which lines apply. The
  # CXX path uses the version of the host clang. The hipcc path uses all of
  # the lines, because hipcc is always a recent clang.
  set(_cc_suppressions_clang_base
    -Wno-unused-command-line-argument
    -Wno-c99-extensions
    -Wno-gnu-zero-variadic-macro-arguments
    # A CUDA header declares `atexit(...) throw()`. This is a deprecated
    # dynamic exception specification. The header comes from the vendor, so we
    # cannot change it. Remove this line when the vendor corrects the header.
    -Wno-error=deprecated-dynamic-exception-spec
    # Remove this line when the warning count is zero.
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

  # nvcc does not accept host warning flags. It must send them to the host
  # compiler. Use the one-token form `-Xcompiler=<flag>`. CMake can divide
  # or remove the two-token form `-Xcompiler <flag>` when it expands a list
  # into COMPILE_OPTIONS.
  set(_nvcc "")
  foreach(_flag IN LISTS _cc)
    list(APPEND _nvcc "-Xcompiler=${_flag}")
  endforeach()

  # hipcc is always clang. The host compiler can be gcc or clang, because
  # the ROCm CI builds with both. This list therefore always uses the clang
  # inputs. A list made from `_cc` loses every clang-only flag on the gcc
  # builds, and no error reports the loss.
  #
  # These flags reach device code. No other path does this.
  #
  # WARNING: `_cc_suppressions_clang` is the full clang set. It includes the
  # lines that the CXX path selects by the version of the host clang. Those
  # version tests apply to the host compiler only. The clang in hipcc is a
  # different and usually newer compiler. Make sure that the hipcc in each
  # supported ROCm version accepts all of these lines. An unknown `-Wno-`
  # line becomes an error when `-Werror` is present.
  # This list contains `_cc_suppressions_common` on purpose. Without it,
  # hipcc gets `-Werror`, but it does not get -Wno-deprecated-declarations,
  # -Wno-strict-aliasing, -Wno-sign-compare, -Wno-vla and the -Wno-error=
  # lines. The host path needs those lines, and a build without them fails
  # immediately.
  #
  # The common suppressions are portable `-Wno-` lines. hipcc is clang, so it
  # accepts them. With them, `_hipcc` becomes the clang equivalent of `_cc`.
  # It contains everything except the EXTRA_CC_FLAGS from the caller, and it
  # uses the full clang suppression set.

  # Suppressions for device code only. `_hipcc` uses the same suppression
  # lists as the host. Without this list, a change to a flag on device code
  # also changes it on the host. The code adds this list last in `_hipcc`, so
  # it has priority. The host path uses the same order.
  set(_hipcc_suppressions
    # ROCm headers contain `using namespace` lines. This line shows the
    # warning but does not let it stop the build, so the count stays visible.
    # The third-party include directories are already SYSTEM, and that alone
    # stops the warning. This line can therefore be unnecessary. Remove it
    # when a full ROCm build reports zero.
    -Wno-error=header-hygiene
    # ROCm can miss Composable Kernel's requested occupancy for some gfx908
    # kernels. This is a backend tuning diagnostic, not an invalid program.
    -Wno-error=pass-failed)

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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${CMAKE_CURRENT_LIST_DIR}/CppLibrary.cmake)

function(prepare_target_sources)
    # This function does the following:
    #
    #   1. Take all the specified project sources for a target
    #   1. Filter files out based on CPU-only, CUDA, and HIP build modes
    #   1. Bucketize them into sets of CXX, CU, and HIP files
    #   1. Apply common source file properties for each bucket
    #   1. Merge the buckets back into a single list of sources
    #   1. Export the file list as ${args_PREFIX}_sources

    set(flags)
    set(singleValueArgs PREFIX)
    set(multiValueArgs
        CPU_SRCS
        GPU_SRCS
        CUDA_SPECIFIC_SRCS
        HIP_SPECIFIC_SRCS
        NVCC_FLAGS
        INCLUDE_DIRS
    )

    cmake_parse_arguments(
        args
        "${flags}" "${singleValueArgs}" "${multiValueArgs}"
        ${ARGN})

    ############################################################################
    # Collect and Annotate, and Append CXX sources
    ############################################################################

    # Add the CPU CXX sources
    set(${args_PREFIX}_sources_cpp ${args_CPU_SRCS})
    list(FILTER ${args_PREFIX}_sources_cpp INCLUDE REGEX "^.+\.cpp$")

    # For GPU mode, add the CXX sources from GPU_SRCS
    if(NOT FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CPU)
        set(_gpu_sources_cpp ${args_GPU_SRCS})
        list(FILTER _gpu_sources_cpp INCLUDE REGEX "^.+\.cpp$")
        list(APPEND ${args_PREFIX}_sources_cpp ${_gpu_sources_cpp})
    endif()

    # Set source properties
    set_source_files_properties(${${args_PREFIX}_sources_cpp}
        PROPERTIES INCLUDE_DIRECTORIES
        "${args_INCLUDE_DIRS}")

    if(CXX_AVX2_FOUND)
        set_source_files_properties(${${args_PREFIX}_sources_cpp}
            PROPERTIES COMPILE_OPTIONS
            "${CXX_AVX2_FLAGS}")
    else()
        set_source_files_properties(${${args_PREFIX}_sources_cpp}
            PROPERTIES COMPILE_OPTIONS
            "-fopenmp")
    endif()

    # Append to the full sources list
    list(APPEND ${args_PREFIX}_sources_combined ${${args_PREFIX}_sources_cpp})

    ############################################################################
    # Collect, Annotate, and Append CU sources
    ############################################################################

    if(NOT FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CPU)
        # Filter GPU_SRCS for CU sources - these may be HIPified later if building in ROCm mode
        set(${args_PREFIX}_sources_cu ${args_GPU_SRCS})
        list(FILTER ${args_PREFIX}_sources_cu INCLUDE REGEX "^.+\.cu$")

        # Append CUDA-specific sources, but ONLY when building in CUDA mode
        if(NOT FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM)
            list(APPEND ${args_PREFIX}_sources_cu ${args_CUDA_SPECIFIC_SRCS})
        endif()

        set_source_files_properties(${${args_PREFIX}_sources_cu}
            PROPERTIES INCLUDE_DIRECTORIES
            "${args_INCLUDE_DIRS}")

        # Set source properties
        set_source_files_properties(${${args_PREFIX}_sources_cu}
            PROPERTIES COMPILE_OPTIONS
            "${args_NVCC_FLAGS}")

        # Starting with CUDA 13.0, nvcc changed the default visibility of
        # __global__ functions to `hidden`, which causes symbol lookup errors
        # during linking.  This can be worked around by setting -cudart=shared
        # and --device-entity-has-hidden-visibility=false.
        #
        # https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/
        if( (FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA) AND
            (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0") )
            set(_nvcc_flags ${args_NVCC_FLAGS}
                -static-global-template-stub=false
                --device-entity-has-hidden-visibility=false)
        else()
            set(_nvcc_flags ${args_NVCC_FLAGS})
        endif()

        # Set compilation flags
        set_source_files_properties(${${args_PREFIX}_sources_cu}
            PROPERTIES COMPILE_OPTIONS
            "${_nvcc_flags}")

        # Append to the full sources list
        list(APPEND ${args_PREFIX}_sources_combined ${${args_PREFIX}_sources_cu})
    endif()

    ############################################################################
    # Collect, Annotate, and Append HIP sources
    ############################################################################

    if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM)
        # Filter GPU_SRCS for HIP sources
        set(${args_PREFIX}_sources_hip ${args_GPU_SRCS})
        list(FILTER ${args_PREFIX}_sources_hip INCLUDE REGEX "^.+\.hip$")

        # Append HIP-specific sources, but ONLY when building in HIP mode
        list(APPEND ${args_PREFIX}_sources_hip ${args_HIP_SPECIFIC_SRCS})

        # Set source properties
        set_source_files_properties(${${args_PREFIX}_sources_hip}
            PROPERTIES INCLUDE_DIRECTORIES
            "${args_INCLUDE_DIRS}")

        # Append to the full sources list
        list(APPEND ${args_PREFIX}_sources_combined ${${args_PREFIX}_sources_hip})
    endif()

    ############################################################################
    # Set the Output Variable(s)
    ############################################################################

    set(${args_PREFIX}_sources ${${args_PREFIX}_sources_combined} PARENT_SCOPE)
endfunction()

function(gpu_cpp_library)
    # This function does the following:
    #
    #   1. Take all the target sources and select relevant sources based on build type (CPU-only, CUDA, HIP)
    #   1. Apply source file properties as needed
    #   1. Fetch the HIPified versions of the files as needed (presumes that `hipify()` has already been run)
    #   1. Build the .SO file, either as STATIC or MODULE
    #
    # Building as STATIC allows the target to be linked to other library targets:
    #   https://www.reddit.com/r/cpp_questions/comments/120p0ey/how_to_create_a_composite_shared_library_out_of
    #   https://github.com/ROCm/hipDNN/blob/master/Examples/hipdnn-training/cmake/FindHIP.cmake

    set(flags)
    set(singleValueArgs
        PREFIX          # Desired name for the library target (and by extension, the prefix for naming intermediate targets)
        TYPE            # Target type, e.g., MODULE, OBJECT.  See https://cmake.org/cmake/help/latest/command/add_library.html
        DESTINATION     # The install destination directory to place the build target into
        KEEP_PREFIX     # Whether to keep the prefix for the library target, e.g. libfoo.so vs foo.so
    )
    set(multiValueArgs
        CPU_SRCS            # Sources for CPU-only build
        GPU_SRCS            # Sources common to both CUDA and HIP builds.  .CU files specified here will be HIPified when building a HIP target
        CUDA_SPECIFIC_SRCS  # Sources available only for CUDA build
        HIP_SPECIFIC_SRCS   # Sources available only for HIP build
        OTHER_SRCS          # Sources from third-party libraries
        CC_FLAGS            # General compilation flags applicable to all build variants
        NVCC_FLAGS          # Compilation flags specific to NVCC
        HIPCC_FLAGS         # Compilation flags specific to HIPCC
        EXCLUDED_WARNING_FLAGS # Shared warning flags not applied to this target
        INCLUDE_DIRS        # First-party include directories for compilation
        SYSTEM_INCLUDE_DIRS # Third-party include directories, passed as SYSTEM to suppress their warnings
        DEPS                # Target dependencies, i.e. built STATIC targets
        TORCH_LIBS          # PyTorch libraries to link against. Note that we provide the TORCH_LIBS automatically - this is for PyTorch build.
    )

    cmake_parse_arguments(
        args
        "${flags}" "${singleValueArgs}" "${multiValueArgs}"
        ${ARGN})

    ############################################################################
    # Prepare CXX and CU sources
    ############################################################################

    # Take all the sources, and filter them into CPU and GPU buckets depending
    # on the source type and build mode
    # NOTE: Only first-party include dirs are set as source file properties here.
    # Third-party dirs are attached at the target level as SYSTEM includes; see
    # gpu_cpp_library() below.  They must NOT also appear as plain `-I` entries:
    # clang de-duplicates header search paths keeping the first occurrence, so a
    # directory listed under both `-I` and `-isystem` stays non-system and its
    # warnings are NOT suppressed.
    prepare_target_sources(
        PREFIX ${args_PREFIX}
        CPU_SRCS ${args_CPU_SRCS}
        GPU_SRCS ${args_GPU_SRCS}
        CUDA_SPECIFIC_SRCS ${args_CUDA_SPECIFIC_SRCS}
        HIP_SPECIFIC_SRCS ${args_HIP_SPECIFIC_SRCS}
        NVCC_FLAGS ${args_NVCC_FLAGS}
        INCLUDE_DIRS ${args_INCLUDE_DIRS})
    set(lib_sources ${${args_PREFIX}_sources})

    # If the overall sources list is empty (e.g. the target is GPU-only and we
    # are currently building in CPU-only mode), add a placeholder source file
    # so that the library can be built without failure
    if(NOT lib_sources AND NOT args_OTHER_SRCS)
        # Create a salt value
        STRING(RANDOM LENGTH 6 salt)

        # Generate a placeholder source file
        file(COPY_FILE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/placeholder.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/gen_placeholder_${salt}.cpp)

        # Append to lib_sources
        list(APPEND lib_sources
            ${CMAKE_CURRENT_BINARY_DIR}/gen_placeholder_${salt}.cpp)
    endif()

    ############################################################################
    # Compilation Flags
    ############################################################################

    # Computed BEFORE the library is created, because `hip_add_library()` takes
    # HIPCC_OPTIONS as a creation-time argument (it is a legacy FindHIP concept,
    # not a target property that can be set afterwards), so the HIPCC list has to
    # exist by then. Both lists are now consumed, and both minus `-Werror`:
    # `_nvcc_warning_flags` by the CUDA genex near the bottom of this function,
    # and `_hipcc_warning_flags` by `hip_add_library()` below.
    #
    # Only the flag computation moved here. The MSVC `target_compile_definitions`
    # that used to sit in the same block stays after library creation, because it
    # needs ${lib_name} to exist.

    fbgemm_get_warning_flags(
        MSVC_FLAGS_VAR  _msvc_flags
        CC_FLAGS_VAR    _cc_flags
        NVCC_FLAGS_VAR  _nvcc_warning_flags
        HIPCC_FLAGS_VAR _hipcc_warning_flags
        EXTRA_MSVC_FLAGS ${args_MSVC_FLAGS}
        EXTRA_CC_FLAGS   ${args_CC_FLAGS})

    foreach(_flag IN LISTS args_EXCLUDED_WARNING_FLAGS)
        list(REMOVE_ITEM _cc_flags "${_flag}")
        list(REMOVE_ITEM _nvcc_warning_flags "-Xcompiler=${_flag}")
        list(REMOVE_ITEM _hipcc_warning_flags "${_flag}")
    endforeach()

    if(MSVC)
        set(lib_cc_flags ${_msvc_flags})
    else()
        set(lib_cc_flags ${_cc_flags})
    endif()

    ############################################################################
    # Build the Library
    ############################################################################

    # Set the build target name
    set(lib_name ${args_PREFIX})

    if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM)
        if(lib_sources)
            # Fetch the equivalent HIPified sources if available.  The mapping
            # is provided by a table that is generated during transpilation
            # process, so this presumes that `hipify()` has already been run.
            #
            # This code is placed under an if-guard so that it won't fail for
            # targets that have nothing to do with HIP, e.g. asmjit
            get_hipified_list("${lib_sources}" lib_sources_hipified)

            # Set properties for the HIPified sources
            set_source_files_properties(${lib_sources_hipified} PROPERTIES
                HIP_SOURCE_PROPERTY_FORMAT 1)
        endif()

        # Set the include directories for HIP.  First-party only: the legacy
        # FindHIP `hip_include_directories()` has no SYSTEM variant and emits
        # plain `-I`, so third-party dirs are passed separately as `-isystem`
        # entries in HIPCC_OPTIONS below.
        hip_include_directories("${args_INCLUDE_DIRS}")

        # Build `-isystem` flags for the third-party include dirs.  hipcc is
        # clang, and clang de-duplicates header search paths keeping the first
        # occurrence, so these directories must appear ONLY here and never in
        # `hip_include_directories()` -- otherwise the `-I` entry wins and the
        # directory is not treated as a system header path.
        set(lib_hipcc_system_includes "")
        foreach(include_dir IN LISTS args_SYSTEM_INCLUDE_DIRS)
            list(APPEND lib_hipcc_system_includes "-isystem${include_dir}")
        endforeach()

        # RECONNAISSANCE: warnings ON, `-Werror` OFF.
        #
        # OSS HIP compilation has never had `-Wall`/`-Wextra` at all, and there is
        # no local ROCm build to measure with, so this lands warn-only to size the
        # fixup work from CI before it can break anyone. **This is temporary** --
        # a follow-up deletes the two lines below and passes
        # `_hipcc_warning_flags` directly, at which point ROCm CI becomes a real
        # device-code gate.
        #
        # Strip every `-Werror*` entry, not just the bare token. `REMOVE_ITEM`
        # with the literal would leave a targeted `-Werror=<name>` behind if the
        # warning list ever grows one, and the surface would silently stop being
        # warn-only. The `^-Werror` anchor deliberately does not match the
        # `-Wno-error=<name>` entries -- those are inert once `-Werror` is gone,
        # and removing them would be a behaviour change.
        set(_hipcc_recon ${_hipcc_warning_flags})
        list(FILTER _hipcc_recon EXCLUDE REGEX "^-Werror")

        # Create the HIP library
        #
        # Warning flags come FIRST, before HIP_HCC_FLAGS. The tail of
        # HIP_HCC_FLAGS is the HIP-specific `-Wno-*` block appended by
        # RocmSetup.cmake, and those suppressions must keep winning -- otherwise
        # `-Wall` re-enables things HIP deliberately turned off, such as
        # `-Wformat`. Do not reorder these two.
        hip_add_library(${lib_name} ${args_TYPE}
            ${lib_sources_hipified}
            ${args_OTHER_SRCS}
            ${FBGEMM_HIP_HCC_LIBRARIES}
            HIPCC_OPTIONS ${_hipcc_recon} ${HIP_HCC_FLAGS} ${lib_hipcc_system_includes} ${args_HIPCC_FLAGS})

        # Append ROCM includes
        target_include_directories(${lib_name} PUBLIC
            ${args_INCLUDE_DIRS})

        # ROCm toolchain and third-party headers are system headers
        target_include_directories(${lib_name} SYSTEM PUBLIC
            ${FBGEMM_HIP_INCLUDE}
            ${args_SYSTEM_INCLUDE_DIRS})

    else()
        # Create the CPU-only / CUDA library
        add_library(${lib_name} ${args_TYPE}
            ${lib_sources}
            ${args_OTHER_SRCS})
    endif()

    ############################################################################
    # Compilation Definitions
    ############################################################################

    # The flag computation that used to be here moved above the "Build the
    # Library" section -- see the note there. This part cannot move, because it
    # operates on the target.
    if(MSVC AND args_TYPE STREQUAL STATIC)
        # MSVC needs to define these variables to avoid generating _dllimport
        # functions.
        target_compile_definitions(${lib_name}
            PUBLIC ASMJIT_STATIC
            PUBLIC FBGEMM_STATIC)
    endif()


    ############################################################################
    # Library Includes and Linking
    ############################################################################

    # Add external include directories.  These are third-party (PyTorch, NCCL,
    # CUTLASS, CK, asmjit, ...) and are marked SYSTEM so their diagnostics do not
    # surface in FBGEMM targets that consume them.
    target_include_directories(${lib_name} SYSTEM PRIVATE
        ${args_SYSTEM_INCLUDE_DIRS})

    # Set additional target properties
    if(NOT args_KEEP_PREFIX)
        # Remove `lib` prefix from the output artifact name, e.g.
        # `libfoo.so` -> `foo.so`
        set_target_properties(${lib_name} PROPERTIES PREFIX "")
    endif()

    set_target_properties(${lib_name} PROPERTIES
        # Enforce -fPIC for STATIC library option, since they are to be
        # integrated into other libraries down the line
        # https://stackoverflow.com/questions/3961446/why-does-gcc-not-implicitly-supply-the-fpic-flag-when-compiling-static-librarie
        POSITION_INDEPENDENT_CODE ON)

    if (args_DEPS OR CMAKE_INSTALL_RPATH)
        # Only set this if the library has dependencies that we also build,
        # otherwise we will hit the following error:
        #   `No valid ELF RPATH or RUNPATH entry exists in the file`
        # However, if CMAKE_INSTALL_RPATH is set, respect that logic. Such as when we build with PyTorch.
        set_target_properties(${lib_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH ON
            # Set the RPATH for the library to include $ORIGIN, so it can look
            # into the same directory for dependency .SO files to load, e.g.
            # fbgemm_gpu.so -> fbgemm.so, asmjit.so
            #
            # More info on RPATHS:
            #   https://amir.rachum.com/shared-libraries/#debugging-cheat-sheet
            #   https://stackoverflow.com/questions/43330165/how-to-link-a-shared-library-with-cmake-with-relative-path
            #   https://stackoverflow.com/questions/57915564/cmake-how-to-set-rpath-to-origin-with-cmake
            #   https://stackoverflow.com/questions/58360502/how-to-set-rpath-origin-in-cmake
            INSTALL_RPATH "\$ORIGIN")
    endif()

    # Collect external libraries for linking
    set(library_dependencies
        ${TORCH_LIBRARIES}
        ${args_TORCH_LIBS}
        ${NCCL_LIBRARIES}
        ${CUDA_DRIVER_LIBRARIES}
        ${args_DEPS})

    # Add NVML if available
    if(NVML_LIB_PATH)
        list(APPEND library_dependencies ${NVML_LIB_PATH})
    endif()

    # Add AMD SMI if available (ROCm builds)
    if(FBGEMM_AMDSMI_LIB)
        list(APPEND library_dependencies ${FBGEMM_AMDSMI_LIB})
    endif()

    # Link against the external libraries as needed
    target_link_libraries(${lib_name} PRIVATE ${library_dependencies})

    ############################################################################
    # Other Compilation Flags
    ############################################################################

    # Set the additional compilation flags
    #
    # ⚠ `args_CC_FLAGS` is applied to EVERY language, unwrapped. No current caller
    # passes a `-W` flag through it (audited), and one that did would
    # reach nvcc raw and fail the build. Keep it that way.
    target_compile_options(${lib_name} PRIVATE
        ${args_CC_FLAGS}
        $<$<COMPILE_LANGUAGE:CXX>:${lib_cc_flags}>)

    # Forward the host warning set to nvcc's host compiler.
    #
    # `_nvcc_warning_flags` is the CXX warning set with every entry wrapped as
    # `-Xcompiler=<flag>` (see fbgemm_get_warning_flags). nvcc rejects a bare
    # `-W...`, so the wrapping is mandatory, not cosmetic -- and it must be the
    # single-token `-Xcompiler=<flag>` form, because the two-token
    # `-Xcompiler <flag>` form can be split when CMake expands a `;`-list.
    #
    # The list form inside the genex is deliberate and matches the CXX line above:
    # a `;`-list inside `$<...>` was measured NOT to leak to other languages, so
    # a per-flag `foreach` would be a no-op.
    #
    # Skipped under MSVC: `_nvcc_warning_flags` is derived from the gcc/clang list
    # regardless of host compiler, so on Windows nvcc would forward `-Wextra`,
    # `-Wno-strict-aliasing` and friends to cl.exe, which does not accept them.
    # The host CXX path already handles this by selecting `_msvc_flags` into
    # `lib_cc_flags`; there is no MSVC-shaped equivalent for the nvcc list today.
    # RECONNAISSANCE: warnings ON, `-Werror` OFF -- the same treatment the HIPCC
    # list gets above, for a different reason.
    #
    # nvcc hands its own generated stubs (`/tmp/tmpxft_*.cudafe1.stub.c`) to the
    # host compiler, so `-Xcompiler=-Werror` lands on code this repo does not
    # write and cannot annotate. Those stubs spell out template arguments as bare
    # decimal literals, and an `int64_t` NTTP of `INT64_MIN` becomes
    # `9223372036854775808` -- which does not fit a signed 64-bit type, so GCC
    # emits `integer constant is so large that it is unsigned`. That diagnostic
    # carries NO `-W<name>`, so unlike every other noisy warning here it cannot
    # be demoted with `-Wno-error=<name>`; dropping `-Werror` is the only lever.
    #
    # Same anchor rationale as the HIPCC list: `^-Xcompiler=-Werror` strips bare
    # `-Werror` and any future `-Werror=<name>`, and deliberately does not match
    # `-Xcompiler=-Wno-error=<name>`, which is inert once `-Werror` is gone.
    set(_nvcc_recon ${_nvcc_warning_flags})
    list(FILTER _nvcc_recon EXCLUDE REGEX "^-Xcompiler=-Werror")

    if(NOT MSVC)
        target_compile_options(${lib_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${_nvcc_recon}>)
    endif()

    ############################################################################
    # Post-Build Steps
    ############################################################################

    if (args_DEPS OR CMAKE_INSTALL_RPATH)
        # Only set this if the library has dependencies that we also build,
        # otherwise we will hit the following error:
        #   `No valid ELF RPATH or RUNPATH entry exists in the file`
        set(set_rpath_to_origin 1)
    endif()

    # Add a post-build step to remove errant RPATHs from the .SO
    add_custom_target(${lib_name}_postbuild ALL
        DEPENDS
        WORKING_DIRECTORY ${OUTPUT_DIR}
        COMMAND bash ${FBGEMM}/.github/scripts/fbgemm_gpu_postbuild.bash $<TARGET_FILE:${lib_name}> ${set_rpath_to_origin})

    # Set the post-build steps to run AFTER the build completes
    add_dependencies(${lib_name}_postbuild ${lib_name})

    ############################################################################
    # Set the Output Variable(s)
    ############################################################################

    set(${args_PREFIX} ${lib_name} PARENT_SCOPE)

    ############################################################################
    # Add to Install Package
    ############################################################################

    if(args_DESTINATION)
        install(
            TARGETS ${args_PREFIX}
            # Allows args_PREFIX to be exported as a target, used for PyTorch build integration
            EXPORT fbgemmGenAILibraryConfig
            DESTINATION ${args_DESTINATION})
    endif()

    ############################################################################
    # Debug Summary
    ############################################################################

    BLOCK_PRINT(
        "GPU CPP Library Target: ${args_PREFIX} (${args_TYPE})"
        " "
        "CPU_SRCS:"
        "${args_CPU_SRCS}"
        " "
        "GPU_SRCS:"
        "${args_GPU_SRCS}"
        " "
        "CUDA_SPECIFIC_SRCS:"
        "${args_CUDA_SPECIFIC_SRCS}"
        " "
        "HIP_SPECIFIC_SRCS:"
        "${args_HIP_SPECIFIC_SRCS}"
        " "
        "OTHER_SRCS:"
        "${args_OTHER_SRCS}"
        " "
        "CC_FLAGS:"
        "${args_CC_FLAGS}"
        " "
        "RESOLVED WARNING FLAGS (CC):"
        "${_cc_flags}"
        " "
        "RESOLVED WARNING FLAGS (NVCC, full set):"
        "${_nvcc_warning_flags}"
        " "
        "RESOLVED WARNING FLAGS (NVCC, AS APPLIED -- recon strips -Werror*):"
        "${_nvcc_recon}"
        " "
        "RESOLVED WARNING FLAGS (HIPCC, full set):"
        "${_hipcc_warning_flags}"
        " "
        "RESOLVED WARNING FLAGS (HIPCC, AS APPLIED -- recon strips -Werror*):"
        "${_hipcc_recon}"
        " "
        "NVCC_FLAGS:"
        "${args_NVCC_FLAGS}"
        " "
        "HIPCC_FLAGS:"
        "${args_HIPCC_FLAGS}"
        " "
        "INCLUDE_DIRS:"
        "${args_INCLUDE_DIRS}"
        " "
        "SYSTEM_INCLUDE_DIRS:"
        "${args_SYSTEM_INCLUDE_DIRS}"
        " "
        "Selected Source Files:"
        "${lib_sources}"
        " "
        "HIPified Source Files:"
        "${lib_sources_hipified}"
        " "
        "Library Dependencies:"
        "${library_dependencies}"
        " "
        "Output Library:"
        "${lib_name}"
        " "
        "Destination Directory:"
        "${args_DESTINATION}"
    )
endfunction()

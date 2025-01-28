# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules/Utilities.cmake)

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
        GPU_FLAGS
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
    LIST_FILTER(
        INPUT ${args_CPU_SRCS}
        OUTPUT cpu_sources_cpp
        REGEX "^.+\.cpp$"
    )
    set(${args_PREFIX}_sources_cpp ${cpu_sources_cpp})

    # For GPU mode, add the CXX sources from GPU_SRCS
    if(NOT FBGEMM_CPU_ONLY)
        LIST_FILTER(
            INPUT ${args_GPU_SRCS}
            OUTPUT gpu_sources_cpp
            REGEX "^.+\.cpp$"
        )
        list(APPEND ${args_PREFIX}_sources_cpp ${gpu_sources_cpp})
    endif()

    # Set source properties
    set_source_files_properties(${${args_PREFIX}_sources_cpp}
        PROPERTIES INCLUDE_DIRECTORIES
        "${args_INCLUDE_DIRS}")

    if(CXX_AVX2_FOUND)
        set_source_files_properties(${${args_PREFIX}_sources_cpp}
            PROPERTIES COMPILE_OPTIONS
            "${AVX2_FLAGS}")
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

    if(NOT FBGEMM_CPU_ONLY)
        # Filter GPU_SRCS for CU sources - these may be HIPified later if building in ROCm mode
        LIST_FILTER(
            INPUT ${args_GPU_SRCS}
            OUTPUT ${args_PREFIX}_sources_cu
            REGEX "^.+\.cu$"
        )

        # Append CUDA-specific sources, but ONLY when building in CUDA mode
        if(NOT USE_ROCM)
            list(APPEND ${args_PREFIX}_sources_cu ${args_CUDA_SPECIFIC_SRCS})
        endif()

        # Set source properties
        set_source_files_properties(${${args_PREFIX}_sources_cu}
            PROPERTIES COMPILE_OPTIONS
            "${args_GPU_FLAGS}")

        set_source_files_properties(${${args_PREFIX}_sources_cu}
            PROPERTIES INCLUDE_DIRECTORIES
            "${args_INCLUDE_DIRS}")

        # Append to the full sources list
        list(APPEND ${args_PREFIX}_sources_combined ${${args_PREFIX}_sources_cu})
    endif()

    ############################################################################
    # Collect, Annotate, and Append HIP sources
    ############################################################################

    if(NOT FBGEMM_CPU_ONLY AND USE_ROCM)
        # Filter GPU_SRCS for HIP sources
        LIST_FILTER(
            INPUT ${args_GPU_SRCS}
            OUTPUT ${args_PREFIX}_sources_hip
            REGEX "^.+\.hip$"
        )

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
    )
    set(multiValueArgs
        CPU_SRCS            # Sources for CPU-only build
        GPU_SRCS            # Sources common to both CUDA and HIP builds.  .CU files specified here will be HIPified when building a HIP target
        CUDA_SPECIFIC_SRCS  # Sources available only for CUDA build
        HIP_SPECIFIC_SRCS   # Sources available only for HIP build
        OTHER_SRCS          # Sources from third-party libraries
        GPU_FLAGS           # Compile flags for GPU builds
        INCLUDE_DIRS        # Include directories for compilation
        DEPS                # Target dependencies, i.e. built STATIC targets
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
    prepare_target_sources(
        PREFIX ${args_PREFIX}
        CPU_SRCS ${args_CPU_SRCS}
        GPU_SRCS ${args_GPU_SRCS}
        CUDA_SPECIFIC_SRCS ${args_CUDA_SPECIFIC_SRCS}
        HIP_SPECIFIC_SRCS ${args_HIP_SPECIFIC_SRCS}
        GPU_FLAGS ${args_GPU_FLAGS}
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
    # Build the Library
    ############################################################################

    # Set the build target name
    set(lib_name ${args_PREFIX})

    if(USE_ROCM)
        if(lib_sources)
            # Fetch the equivalent HIPified sources if available.  The mapping
            # is provided by a table that is generated during transpilation
            # process, so this presumes that `hipify()` has already been run.
            #
            # This code is placed under an if-guard so that it won't fail for
            # targets that have nothing to do with HIP, e.g. asmjit
            get_hipified_list("${lib_sources}" lib_sources_hipified)

            # Set properties for the HIPified sources
            set_source_files_properties(${lib_sources_hipified}
                                        PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
        endif()

        # Set the include directories for HIP
        hip_include_directories("${args_INCLUDE_DIRS}")

        # Create the HIP library
        hip_add_library(${lib_name} ${args_TYPE}
            ${lib_sources_hipified}
            ${args_OTHER_SRCS}
            ${FBGEMM_HIP_HCC_LIBRARIES}
            HIPCC_OPTIONS
            ${HIP_HCC_FLAGS})

        # Append ROCM includes
        target_include_directories(${lib_name} PUBLIC
            ${FBGEMM_HIP_INCLUDE}
            ${ROCRAND_INCLUDE}
            ${ROCM_SMI_INCLUDE}
            ${args_INCLUDE_DIRS})

    else()
        # Create the CPU-only / CUDA library
        add_library(${lib_name} ${args_TYPE}
            ${lib_sources}
            ${args_OTHER_SRCS})
    endif()

    ############################################################################
    # Library Includes and Linking
    ############################################################################

    # Add external include directories
    target_include_directories(${lib_name} PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${NCCL_INCLUDE_DIRS})

    # Set additional target properties
    set_target_properties(${lib_name} PROPERTIES
        # Remove `lib` prefix from the output artifact name, e.g.
        # `libfoo.so` -> `foo.so`
        PREFIX ""
        # Enforce -fPIC for STATIC library option, since they are to be
        # integrated into other libraries down the line
        # https://stackoverflow.com/questions/3961446/why-does-gcc-not-implicitly-supply-the-fpic-flag-when-compiling-static-librarie
        POSITION_INDEPENDENT_CODE ON)

    if (args_DEPS)
        # Only set this if the library has dependencies that we also build,
        # otherwise we will hit the following error:
        #   `No valid ELF RPATH or RUNPATH entry exists in the file`
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
        ${NCCL_LIBRARIES}
        ${CUDA_DRIVER_LIBRARIES}
        ${args_DEPS})

    # Add NVML if available
    if(NVML_LIB_PATH)
        list(APPEND library_dependencies ${NVML_LIB_PATH})
    endif()

    # Link against the external libraries as needed
    target_link_libraries(${lib_name} PRIVATE ${library_dependencies})

    # Silence compiler warnings (in asmjit)
    target_compile_options(${lib_name} PRIVATE
        -Wno-deprecated-anon-enum-enum-conversion
        -Wno-deprecated-declarations)

    ############################################################################
    # Post-Build Steps
    ############################################################################

    if (args_DEPS)
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
        install(TARGETS ${args_PREFIX}
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
        "GPU_FLAGS:"
        "${args_GPU_FLAGS}"
        " "
        "INCLUDE_DIRS:"
        "${args_INCLUDE_DIRS}"
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

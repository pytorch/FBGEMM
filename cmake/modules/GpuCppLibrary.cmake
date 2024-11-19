# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules/Utilities.cmake)

function(prepare_target_sources)
    # This function does the following:
    #   1. Take all the specified project sources for a target
    #   1. Filter the files out based on CPU-only, CUDA, and HIP build modes
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
    set(${args_PREFIX}_sources_cpp ${args_CPU_SRCS})

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

function(prepare_hipified_target_sources)
    # This function does the following:
    #   1. Take all the specified target sources
    #   1. Look up their equivalent HIPified files if applicable (presumes that hipify() already been run)
    #   1. Apply source file properties
    #   1. Update the HIP include directories

    set(flags)
    set(singleValueArgs PREFIX)
    set(multiValueArgs SRCS INCLUDE_DIRS)

    cmake_parse_arguments(
        args
        "${flags}" "${singleValueArgs}" "${multiValueArgs}"
        ${ARGN})

    get_hipified_list("${args_SRCS}" args_SRCS)

    set_source_files_properties(${args_SRCS}
                                PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

    # Add include directories
    hip_include_directories("${args_INCLUDE_DIRS}")

    ############################################################################
    # Set the Output Variable(s)
    ############################################################################

    set(${args_PREFIX}_sources_hipified ${args_SRCS} PARENT_SCOPE)
endfunction()

function(gpu_cpp_library)
    # This function does the following:
    #   1. Take all the target sources and select relevant sources based on build type (CPU-only, CUDA, HIP)
    #   1. Apply source file properties as needed
    #   1. HIPify files as needed
    #   1. Build the .SO file

    set(flags)
    set(singleValueArgs
        PREFIX          # Desired name prefix for the library target
    )
    set(multiValueArgs
        CPU_SRCS            # Sources for CPU-only build
        GPU_SRCS            # Sources common to both CUDA and HIP builds.  .CU files specified here will be HIPified when building a HIP target
        CUDA_SPECIFIC_SRCS  # Sources available only for CUDA build
        HIP_SPECIFIC_SRCS   # Sources available only for HIP build
        GPU_FLAGS           # Compile flags for GPU builds
        INCLUDE_DIRS        # Include directories for compilation
    )

    cmake_parse_arguments(
        args
        "${flags}" "${singleValueArgs}" "${multiValueArgs}"
        ${ARGN})

    ############################################################################
    # Prepare CXX and CU sources
    ############################################################################

    prepare_target_sources(
        PREFIX ${args_PREFIX}
        CPU_SRCS ${args_CPU_SRCS}
        GPU_SRCS ${args_GPU_SRCS}
        CUDA_SPECIFIC_SRCS ${args_CUDA_SPECIFIC_SRCS}
        HIP_SPECIFIC_SRCS ${args_HIP_SPECIFIC_SRCS}
        GPU_FLAGS ${args_GPU_FLAGS}
        INCLUDE_DIRS ${args_INCLUDE_DIRS})
    set(lib_sources ${${args_PREFIX}_sources})


    ############################################################################
    # Build the Library
    ############################################################################

    set(lib_name ${args_PREFIX}_py)
    if(USE_ROCM)
        # Fetch the HIPified sources
        prepare_hipified_target_sources(
            PREFIX ${args_PREFIX}
            SRCS ${lib_sources}
            INCLUDE_DIRS ${args_INCLUDE_DIRS})
        set(lib_sources_hipified ${${args_PREFIX}_sources_hipified})

        # Create the HIP library
        hip_add_library(${lib_name} SHARED
            ${lib_sources_hipified}
            ${args_OTHER_SRCS}
            ${FBGEMM_HIP_HCC_LIBRARIES}
            HIPCC_OPTIONS
            ${HIP_HCC_FLAGS})

        # Append ROCM includes
        target_include_directories(${lib_name} PUBLIC
            ${FBGEMM_HIP_INCLUDE}
            ${ROCRAND_INCLUDE}
            ${ROCM_SMI_INCLUDE})

    else()
        # Create the C++/CUDA library
        add_library(${lib_name} MODULE
            ${lib_sources}
            ${args_OTHER_SRCS})
    endif()

    ############################################################################
    # Library Includes and Linking
    ############################################################################

    # Add PyTorch include/
    target_include_directories(${lib_name} PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${NCCL_INCLUDE_DIRS})

    # Remove `lib` from the output artifact name, i.e. `libfoo.so` -> `foo.so`
    set_target_properties(${lib_name}
        PROPERTIES PREFIX "")

    # Link to PyTorch
    target_link_libraries(${lib_name}
        ${TORCH_LIBRARIES}
        ${NCCL_LIBRARIES}
        ${CUDA_DRIVER_LIBRARIES})

    # Link to NVML if available
    if(NVML_LIB_PATH)
        target_link_libraries(${lib_name} ${NVML_LIB_PATH})
    endif()

    # Silence warnings (in asmjit)
    target_compile_options(${lib_name} PRIVATE
        -Wno-deprecated-anon-enum-enum-conversion
        -Wno-deprecated-declarations)

    ############################################################################
    # Post-Build Steps
    ############################################################################

    # Add a post-build step to remove errant RPATHs from the .SO
    add_custom_target(${lib_name}_postbuild ALL
        DEPENDS
        WORKING_DIRECTORY ${OUTPUT_DIR}
        COMMAND bash ${FBGEMM}/.github/scripts/fbgemm_gpu_postbuild.bash)

    # Run the post-build steps AFTER the build itself
    add_dependencies(${lib_name}_postbuild ${lib_name})

    ############################################################################
    # Set the Output Variable(s)
    ############################################################################

    # PREFIX = `foo` --> Target Library = `foo_py`
    set(${args_PREFIX}_py ${lib_name} PARENT_SCOPE)

    BLOCK_PRINT(
        "GPU CPP Library Target: ${args_PREFIX}"
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
        "HIP_SPECIFIC_SRCS"
        "${args_HIP_SPECIFIC_SRCS}"
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
        "Output Library:"
        "${lib_name}"
    )
endfunction()

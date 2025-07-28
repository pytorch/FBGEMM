# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
        INCLUDE_DIRS    # Include directories for compilation
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

        set(lib_cc_flags
            ${args_MSVC_FLAGS}
            /wd4244
            /wd4267
            /wd4305
            /wd4309)

    else()
        set(lib_cc_flags
            ${args_CC_FLAGS}
            -Wno-deprecated-declarations
            -Wall
            -Wextra
            -Werror
            -Wunknown-pragmas
            -Wimplicit-fallthrough
            -Wno-strict-aliasing
            -Wunused-variable
            -Wno-sign-compare
            -Wno-vla
            -Wno-error=unused-parameter
            -Wno-error=attributes)

        if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
            list(APPEND lib_cc_flags
                -Wno-c99-extensions
                -Wno-gnu-zero-variadic-macro-arguments
                -Wno-deprecated-enum-enum-conversion)

            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 13.0.0)
                list(APPEND lib_cc_flags
                    -Wno-error=unused-but-set-parameter
                    -Wno-error=unused-but-set-variable)
            endif()

            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 17.0.0)
                list(APPEND lib_cc_flags
                    -Wno-vla-cxx-extension
                    -Wno-error=global-constructors
                    -Wno-error=shadow)
            endif()

        elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
            list(APPEND lib_cc_flags
                -Wmaybe-uninitialized
                -Wno-error=unused-but-set-parameter
                -Wno-error=unused-but-set-variable)

            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
                list(APPEND lib_cc_flags
                    -Wno-deprecated-enum-enum-conversion)
            endif()

            if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3.0)
                # Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
                list(APPEND lib_cc_flags
                    -Wno-attributes)
            endif()
        endif()
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

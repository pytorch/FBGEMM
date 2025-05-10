# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Utility Functions
################################################################################

function(BLOCK_PRINT)
  message("")
  message("")
  message("================================================================================")
  foreach(ARG IN LISTS ARGN)
     message("${ARG}")
  endforeach()
  message("================================================================================")
  message("")
endfunction()

function(LIST_FILTER)
  set(flags)
  set(singleValueArgs OUTPUT REGEX)
  set(multiValueArgs INPUT)

  cmake_parse_arguments(
    args
    "${flags}" "${singleValueArgs}" "${multiValueArgs}"
    ${ARGN})

  set(${args_OUTPUT})

  foreach(value ${args_INPUT})
    if("${value}" MATCHES "${args_REGEX}")
      list(APPEND ${args_OUTPUT} ${value})
    endif()
  endforeach()

  set(${args_OUTPUT} ${${args_OUTPUT}} PARENT_SCOPE)
endfunction()

function(prepend_filepaths)
  set(flags)
  set(singleValueArgs PREFIX OUTPUT)
  set(multiValueArgs INPUT)

  cmake_parse_arguments(
    args
    "${flags}" "${singleValueArgs}" "${multiValueArgs}"
    ${ARGN})

  set(${args_OUTPUT})

  foreach(filepath ${args_INPUT})
    list(APPEND ${args_OUTPUT} "${args_PREFIX}/${filepath}")
  endforeach()

  set(${args_OUTPUT} ${${args_OUTPUT}} PARENT_SCOPE)
endfunction()

macro(handle_genfiles variable)
  prepend_filepaths(
    PREFIX ${CMAKE_BINARY_DIR}
    INPUT ${${variable}}
    OUTPUT ${variable})
endmacro()

macro(handle_genfiles_rocm variable)
  if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM)
    handle_genfiles(${variable})
  endif()
endmacro()

function(add_to_package)
  set(flags)
  set(singleValueArgs
    DESTINATION       # The destination directory, RELATIVE to the root of the installation package directory
  )
  set(multiValueArgs
    FILES             # The list of files to place into the DESTINATION directory
    TARGETS           # THe list of CMake targets whose build artifacts to place into the DESTINATION directory
  )

  cmake_parse_arguments(
    args
    "${flags}" "${singleValueArgs}" "${multiValueArgs}"
    ${ARGN})

  install(TARGETS ${args_TARGETS} DESTINATION ${args_DESTINATION})
  install(FILES ${args_FILES} DESTINATION ${args_DESTINATION})

  BLOCK_PRINT(
    "Adding to Package: ${args_DESTINATION}"
    " "
    "TARGETS:"
    "${args_TARGETS}"
    " "
    "FILES:"
    "${args_FILES}"
  )
endfunction()

function(glob_files variable)
  # This function is similar to file(GLOB) in that it returns a list of files
  # that match the given file patterns but filters out those that match the
  # exclude regexes

  set(options)
  set(oneValueArgs EXCLUDE_REGEX)
  set(multiValueArgs PATTERNS)  # List of glob patterns
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Set default exclude regex to match nothing
  if(NOT ARG_EXCLUDE_REGEX)
    set(ARG_EXCLUDE_REGEX "^$")
  endif()

  set(all_matched_files)

  # Loop over each pattern and glob files
  foreach(pattern IN LISTS ARG_PATTERNS)
    file(GLOB matched_files "${pattern}")
    list(APPEND all_matched_files ${matched_files})
  endforeach()

  # Remove duplicates and apply exclusion filter
  if(all_matched_files)
    list(REMOVE_DUPLICATES all_matched_files)
    list(FILTER all_matched_files EXCLUDE REGEX "${ARG_EXCLUDE_REGEX}")
  endif()

  # Set output variable in parent scope
  set(${variable} ${all_matched_files} PARENT_SCOPE)
endfunction()

function(glob_files_nohip variable)
  # This function is a wrapper around glob_files that excludes files with the
  # *_hip.cpp suffix

  set(args ${ARGN})  # All arguments except function name

  glob_files(
    tmp_list
    PATTERNS ${ARGN}
    EXCLUDE_REGEX  ".*_hip\\.cpp$")

  set(${variable} ${tmp_list} PARENT_SCOPE)
endfunction()

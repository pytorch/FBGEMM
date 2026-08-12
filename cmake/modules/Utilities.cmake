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

macro(handle_genfiles variable)
  list(TRANSFORM ${variable} PREPEND "${CMAKE_BINARY_DIR}/")
endmacro()

# Re-export a third-party target's public include directories as SYSTEM
# includes, so that warnings originating inside its headers are suppressed in
# every target that consumes it.
#
# This is needed for dependencies pulled in via `add_subdirectory()` (asmjit,
# cpuinfo), whose INTERFACE_INCLUDE_DIRECTORIES propagate as plain `-I` and
# therefore leak their diagnostics into FBGEMM targets.  We cannot fix those
# projects, so we mark their headers system instead of blanket-disabling the
# warning for our own code as well.
function(fbgemm_mark_target_includes_system target_name)
  if(NOT TARGET ${target_name})
    return()
  endif()

  get_target_property(_includes ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
  # The property is `<name>-NOTFOUND` when unset; passing that through produces
  # a confusing configure-time error.
  if(_includes)
    set_target_properties(${target_name} PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_includes}")
  endif()
endfunction()

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

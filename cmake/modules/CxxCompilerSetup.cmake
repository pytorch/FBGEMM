# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CMake C++ Setup
################################################################################

# SET THE C AND C++ VERSIONS HERE
set(CXX_VERSION 20)

# Set the default C++ standard to CXX_VERSION if CMAKE_CXX_STANDARD is not
# supplied by CMake command invocation.
# Individual targets can have this value overridden; see
# https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
# https://cmake.org/cmake/help/latest/prop_tgt/HIP_STANDARD.html
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD ${CXX_VERSION})
  set(CMAKE_HIP_STANDARD ${CXX_VERSION})
  set(CXX_STANDARD ${CXX_VERSION})
  set(HIP_STANDARD ${CXX_VERSION})
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(HIP_STANDARD_REQUIRED ON)


BLOCK_PRINT(
  "Default C++ compiler flags"
  "(values may be overridden by CMAKE_CXX_STANDARD and CXX_STANDARD):"
  ""
  "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}"
  ""
  "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}"
  ""
  "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}"
)

# Strip all symbols from the .SO file after building
if(NOT MSVC AND NOT APPLE)
  add_link_options($<$<CONFIG:RELEASE>:-s>)
endif()

# Enable compile commands to compile_commands.json for debugging
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

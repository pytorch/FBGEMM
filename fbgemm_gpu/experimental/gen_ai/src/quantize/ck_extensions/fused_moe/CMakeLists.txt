# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(TILE_EXAPMLE_FUSED_MOE "tile_example_fused_moe")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message("adding ${TILE_EXAPMLE_FUSED_MOE}")
file(GLOB INSTANCE_SRCS instances/*.cpp)
add_executable(${TILE_EXAPMLE_FUSED_MOE} EXCLUDE_FROM_ALL main.cpp)
target_include_directories(${TILE_EXAPMLE_FUSED_MOE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_sources(${TILE_EXAPMLE_FUSED_MOE} PRIVATE ${INSTANCE_SRCS})

set(TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
list(APPEND TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS -DCK_TILE_BUFFER_LOAD_AGPR=1) # TODO: enable load to a
list(APPEND TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=4) # rta
# list(APPEND TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS  -mllvm -greedy-reverse-local-assignment=1)
# list(APPEND TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)

target_compile_options(${TILE_EXAPMLE_FUSED_MOE} PRIVATE ${TILE_EXAPMLE_FUSED_MOE_COMPILE_OPTIONS})

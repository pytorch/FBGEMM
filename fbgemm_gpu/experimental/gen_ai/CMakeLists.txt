# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################

set(experimental_gen_ai_cpp_source_files
    src/attention/attention.cpp
    src/attention/gqa_attn_splitk.cu)

set(experimental_gen_ai_python_source_files
    gen_ai/__init__.py)


################################################################################
# Build Shared Library
################################################################################

add_library(fbgemm_gpu_experimental_gen_ai_py MODULE
    ${experimental_gen_ai_cpp_source_files})

target_include_directories(fbgemm_gpu_experimental_gen_ai_py PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(fbgemm_gpu_experimental_gen_ai_py ${TORCH_LIBRARIES})

# Remove `lib` from the output artifact name
set_target_properties(fbgemm_gpu_experimental_gen_ai_py PROPERTIES PREFIX "")


################################################################################
# Install Shared Library and Python Files
################################################################################

install(TARGETS fbgemm_gpu_experimental_gen_ai_py
        DESTINATION fbgemm_gpu/experimental/gen_ai)

install(FILES ${experimental_gen_ai_python_source_files}
        DESTINATION fbgemm_gpu/experimental/gen_ai)

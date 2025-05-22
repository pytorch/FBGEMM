cmake_minimum_required(VERSION 3.18)
project(FBGEMM_GPU_Demo CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch (LibTorch)
# Users should point CMAKE_PREFIX_PATH to their LibTorch installation
# e.g., export CMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # For utils.h
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Add the executable
add_executable(embedding_demo main.cpp)

# Target link libraries
target_link_libraries(embedding_demo PRIVATE
    ${TORCH_LIBRARIES}
)

# CUDA-specific linking if necessary (often handled by TORCH_LIBRARIES)
# If PyTorch is built with CUDA, TORCH_LIBRARIES should include CUDA runtime.
# Explicitly adding CUDA::cudart might be needed in some setups.
target_link_libraries(embedding_demo PRIVATE CUDA::cudart)


# Link against the FBGEMM_GPU library that provides the kernel implementation.
# This assumes that FBGEMM_GPU has been built and installed or is available
# in the build tree such that CMake can find its targets.
#
# Option 1: If FBGEMM_GPU exports a CMake target (e.g., fbgemm_gpu::fbgemm_gpu)
# find_package(FBGEMM_GPU REQUIRED) # This would require FBGEMM_GPU to install a FindFBGEMM_GPU.cmake
# target_link_libraries(embedding_demo PRIVATE fbgemm_gpu::fbgemm_gpu_codegen_ops) # Or similar target

# Option 2: If FBGEMM_GPU is built as part of the same CMake superproject
# and exposes a target like 'fbgemm_gpu_codegen_ops' or 'fbgemm_gpu'.
# This is more likely if building this demo within the FBGEMM_GPU source tree.
# We assume a target named 'fbgemm_gpu' or 'fbgemm_gpu_ops' for now.
# The actual name depends on how FBGEMM_GPU defines its CMake targets.
# For the purpose of this demo, we'll try a common convention.
# This will likely need to be adjusted based on the FBGEMM_GPU build system.
#
# Looking at the root CMakeLists.txt and fbgemm_gpu/CMakeLists.txt,
# FBGEMM_GPU itself is a library. We need to link to the specific
# component that builds the CUDA operations.
# The `fbgemm_gpu_codegen_ops` or similar seems plausible.
# Let's assume a general target `fbgemm_gpu_ops` or `fbgemm_gpu_codegen_tbe`.
# The file `FbgemmGpu.cmake` might define `fbgemm_gpu_ops`.
# From `fbgemm_gpu/CMakeLists.txt` it seems `fbgemm_gpu_ops` is a likely candidate
# as an interface library.
#
# **IMPORTANT**: This linking step is the most likely to need adjustment.
# For now, let's placeholder with a common pattern. The user will need to ensure
# FBGEMM_GPU is built and its targets are available.
# We will try to link to "fbgemm_gpu_ops" which seems to be defined in fbgemm_gpu/CMakeLists.txt
# as an interface library that includes TBE ops.
#
# If FBGEMM_GPU is installed system-wide or in a known prefix,
# find_package(FBGEMM_GPU) might work if it provides config files.
# If building alongside FBGEMM_GPU (e.g. this demo is a subdirectory),
# then its targets should be visible.

# Attempt to link against a plausible FBGEMM_GPU target.
# This assumes this demo's CMakeLists.txt is processed after FBGEMM_GPU's targets are defined.
# If FBGEMM_GPU is a parent project or added via add_subdirectory:
target_link_libraries(embedding_demo PRIVATE fbgemm_gpu_ops) # Common pattern seen in fbgemm_gpu tests

# If Torch was found, it sets TORCH_CXX_FLAGS
set_property(TARGET embedding_demo PROPERTY CXX_FLAGS "${TORCH_CXX_FLAGS}")

# Enable CUDA for the target
set_target_properties(embedding_demo PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;90") # Common GPU architectures
set_property(TARGET embedding_demo PROPERTY CUDA_STANDARD 17)

# Print some helpful messages
message(STATUS "PyTorch (LibTorch) found:")
message(STATUS "  TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "  TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "CUDA found:")
message(STATUS "  CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "  CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "  CUDA_VERSION: ${CUDA_VERSION}")

message(STATUS "Make sure FBGEMM_GPU is built and its CMake targets are available.")
message(STATUS "You might need to build this demo as part of the FBGEMM_GPU build tree,")
message(STATUS "or ensure CMAKE_PREFIX_PATH points to both LibTorch and FBGEMM_GPU install locations if applicable.")

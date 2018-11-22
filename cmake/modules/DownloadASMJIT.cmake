cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(asmjit-download NONE)

include(ExternalProject)

ExternalProject_Add(asmjit
  GIT_REPOSITORY https://github.com/asmjit/asmjit
  GIT_TAG 673dcefaa048c5f5a2bf8b85daf8f7b9978d018a
  SOURCE_DIR "${FBGEMM_THIRDPARTY_DIR}/asmjit"
  BINARY_DIR "${FBGEMM_BINARY_DIR}/asmjit"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

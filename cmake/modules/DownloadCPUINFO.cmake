cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

project(cpuinfo-download NONE)

include(ExternalProject)

ExternalProject_Add(cpuinfo
  GIT_REPOSITORY https://github.com/pytorch/cpuinfo
  GIT_TAG master
  SOURCE_DIR "${FBGEMM_THIRDPARTY_DIR}/cpuinfo"
  BINARY_DIR "${FBGEMM_BINARY_DIR}/cpuinfo"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(googletest-download NONE)

include(ExternalProject)

ExternalProject_Add(googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG 0fc5466dbb9e623029b1ada539717d10bd45e99e
  SOURCE_DIR "${FBGEMM_THIRDPARTY_DIR}/googletest"
  BINARY_DIR "${FBGEMM_BINARY_DIR}/googletest"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

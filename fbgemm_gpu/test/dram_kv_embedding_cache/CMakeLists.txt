find_package(folly REQUIRED)
find_package(gflags REQUIRED)

include_directories(
        ${FBGEMM_SOURCE_DIR}
)

add_executable(fixed_block_pool_test ${CMAKE_CURRENT_SOURCE_DIR}/fixed_block_pool_test.cpp)
target_compile_features(fixed_block_pool_test PUBLIC cxx_std_17)
target_compile_options(fixed_block_pool_test PUBLIC "-O3")
target_link_libraries(fixed_block_pool_test gtest gtest_main Folly::folly)

add_executable(sharded_map_test ${CMAKE_CURRENT_SOURCE_DIR}/sharded_map_test.cpp)
target_compile_features(sharded_map_test PUBLIC cxx_std_17)
target_compile_options(sharded_map_test PUBLIC "-O3")
target_link_libraries(sharded_map_test gtest gtest_main Folly::folly)

add_executable(feature_evict_test ${CMAKE_CURRENT_SOURCE_DIR}/feature_evict_test.cpp)
target_compile_features(feature_evict_test PUBLIC cxx_std_17)
target_compile_options(feature_evict_test PUBLIC "-O3")
target_link_libraries(feature_evict_test gtest gtest_main Folly::folly)
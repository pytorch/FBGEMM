/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "fbgemm_gpu/split_embeddings_cache_cuda.cuh"

using namespace ::testing;

// Helper function that generates input tensor for emulate_cache_miss testing.
at::Tensor generate_lxu_cache_locations(
    const int64_t num_requests,
    const int64_t num_sets,
    const int64_t associativity = 32) {
  const auto lxu_cache_locations = at::randint(
      0,
      num_sets * associativity,
      {num_requests},
      at::device(at::kCPU).dtype(at::kInt));
  return lxu_cache_locations;
}

// Wrapper function that takes lxu_cache_locations on CPU, copies it to GPU,
// runs emulate_cache_miss(), and then returns the result, placed on CPU.
std::pair<at::Tensor, at::Tensor> run_emulate_cache_miss(
    at::Tensor lxu_cache_locations,
    const int64_t enforced_misses_per_256,
    const bool gather_uvm_stats = false) {
  at::Tensor lxu_cache_locations_copy = at::_to_copy(lxu_cache_locations);
  const auto options =
      lxu_cache_locations.options().device(at::kCUDA).dtype(at::kInt);
  const auto uvm_cache_stats =
      gather_uvm_stats ? at::zeros({6}, options) : at::empty({0}, options);

  const auto lxu_cache_location_with_cache_misses = emulate_cache_miss(
      lxu_cache_locations_copy.to(at::kCUDA),
      enforced_misses_per_256,
      gather_uvm_stats,
      uvm_cache_stats);
  return {lxu_cache_location_with_cache_misses.cpu(), uvm_cache_stats.cpu()};
}

TEST(uvm_cache_miss_emulate_test, no_cache_miss) {
  constexpr int64_t num_requests = 10000;
  constexpr int64_t num_sets = 32768;
  constexpr int64_t associativity = 32;

  auto lxu_cache_locations_cpu =
      generate_lxu_cache_locations(num_requests, num_sets, associativity);
  auto lxu_cache_location_with_cache_misses_and_uvm_cache_stats =
      run_emulate_cache_miss(lxu_cache_locations_cpu, 0);
  auto lxu_cache_location_with_cache_misses =
      lxu_cache_location_with_cache_misses_and_uvm_cache_stats.first;
  EXPECT_TRUE(
      at::equal(lxu_cache_locations_cpu, lxu_cache_location_with_cache_misses));
}

TEST(uvm_cache_miss_emulate_test, enforced_cache_miss) {
  constexpr int64_t num_requests = 10000;
  constexpr int64_t num_sets = 32768;
  constexpr int64_t associativity = 32;
  constexpr std::array<int64_t, 6> enforced_misses_per_256_for_testing = {
      1, 5, 7, 33, 100, 256};

  for (const bool miss_in_lxu_cache_locations : {false, true}) {
    for (const bool gather_cache_stats : {false, true}) {
      for (const auto enforced_misses_per_256 :
           enforced_misses_per_256_for_testing) {
        auto lxu_cache_locations_cpu =
            generate_lxu_cache_locations(num_requests, num_sets, associativity);
        if (miss_in_lxu_cache_locations) {
          // one miss in the original lxu_cache_locations; shouldn't be counted
          // as enforced misses from emulate_cache_miss().
          auto z = lxu_cache_locations_cpu.data_ptr<int32_t>();
          z[0] = -1;
        }
        auto lxu_cache_location_with_cache_misses_and_uvm_cache_stats =
            run_emulate_cache_miss(
                lxu_cache_locations_cpu,
                enforced_misses_per_256,
                gather_cache_stats);
        auto lxu_cache_location_with_cache_misses =
            lxu_cache_location_with_cache_misses_and_uvm_cache_stats.first;
        EXPECT_FALSE(at::equal(
            lxu_cache_locations_cpu, lxu_cache_location_with_cache_misses));

        auto x = lxu_cache_locations_cpu.data_ptr<int32_t>();
        auto y = lxu_cache_location_with_cache_misses.data_ptr<int32_t>();
        int64_t enforced_misses = 0;
        for (int32_t i = 0; i < lxu_cache_locations_cpu.numel(); ++i) {
          if (x[i] != y[i]) {
            EXPECT_EQ(y[i], -1);
            enforced_misses++;
          }
        }
        int64_t num_requests_over_256 =
            static_cast<int64_t>(num_requests / 256);
        int64_t expected_misses = num_requests_over_256 *
                enforced_misses_per_256 +
            std::min((num_requests - num_requests_over_256 * 256),
                     enforced_misses_per_256);
        if (miss_in_lxu_cache_locations) {
          expected_misses--;
        }
        EXPECT_EQ(expected_misses, enforced_misses);
        if (gather_cache_stats) {
          auto uvm_cache_stats =
              lxu_cache_location_with_cache_misses_and_uvm_cache_stats.second;
          auto cache_stats_ptr = uvm_cache_stats.data_ptr<int32_t>();
          // enforced misses are recorded as conflict misses.
          EXPECT_EQ(expected_misses, cache_stats_ptr[5]);
        }
      }
    }
  }
}

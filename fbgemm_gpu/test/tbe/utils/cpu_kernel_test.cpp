/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include "fbgemm_gpu/embedding_backward_split_cpu.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "torch/types.h" // @manual=//caffe2:torch-cpp-cpu

#if FBGEMM_GPU_MEMCHECK
#define FBGEMM_MEM_CHECK_ONLY
#else
#define FBGEMM_MEM_CHECK_ONLY maybe_unused
#endif

template <typename T>
void test_csr2csc() {
  constexpr auto DType = c10::CppTypeToScalarType<T>::value;

  internal::HyperCompressedSparseColumn csc;
  int B = 2;
  at::Tensor offsets =
      torch::tensor({0, 4, 8}, torch::TensorOptions().dtype(DType));
  at::Tensor indices = torch::tensor(
      {1, 2, 4, 5, 4, 3, 2, 9}, torch::TensorOptions().dtype(DType));
  int64_t pooling_mode = (int64_t)fbgemm_gpu::PoolingMode::SUM;
  int table_to_feature_offset[2] = {0, 1};
  int num_embeddings = 10;

  const auto no_weights = at::empty({0}, at::TensorOptions().dtype(at::kFloat));
  [[FBGEMM_MEM_CHECK_ONLY]] const auto func_name1 = "::internal::csr2csc_1";
  ::internal::csr2csc(
      csc,
      B,
      MAKE_TA_WITH_NAME(func_name1, offsets, T, 1),
      MAKE_TA_WITH_NAME(func_name1, indices, T, 1),
      MAKE_TA_WITH_NAME(func_name1, no_weights, float, 1),
      pooling_mode,
      table_to_feature_offset,
      num_embeddings);

  // sorted list of unique elements in indices
  std::array<int, 6> expect_cs_indices = {{1, 2, 3, 4, 5, 9}};
  for (int i = 0; i < expect_cs_indices.size(); ++i) {
    EXPECT_EQ(expect_cs_indices[i], csc.column_segment_indices[i]);
  }

  // column_segment_ptr[i+1]-column_segment_ptr[i] gives the count of
  // column_segment_indices[i] in indices
  std::array<int, 7> expect_cs_ptr = {{0, 1, 3, 4, 6, 7, 8}};
  for (int i = 0; i < expect_cs_ptr.size(); ++i) {
    EXPECT_EQ(expect_cs_ptr[i], csc.column_segment_ptr[i]);
  }

  // gives the bag of the ith lowest value in indices, where the bag is
  // determined according to offsets
  std::array<int, 8> expect_row_indices = {{0, 0, 1, 1, 0, 1, 0, 1}};
  for (int i = 0; i < expect_row_indices.size(); ++i) {
    EXPECT_EQ(expect_row_indices[i], csc.row_indices[i]);
  }

  internal::HyperCompressedSparseColumn csc_weighted;
  at::Tensor indice_weights = torch::tensor(
      {1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f}, torch::kFloat32);

  [[maybe_unused]] const auto func_name2 = "::internal::csr2csc_2";
  using weight_t = at::acc_type<float, true>;

  ::internal::csr2csc(
      csc_weighted,
      B,
      MAKE_TA_WITH_NAME(func_name2, offsets, T, 1),
      MAKE_TA_WITH_NAME(func_name2, indices, T, 1),
      MAKE_TA_WITH_NAME(func_name2, indice_weights, weight_t, 1),
      pooling_mode,
      table_to_feature_offset,
      num_embeddings);

  for (int i = 0; i < expect_cs_indices.size(); ++i) {
    EXPECT_EQ(expect_cs_indices[i], csc_weighted.column_segment_indices[i]);
  }

  for (int i = 0; i < expect_cs_ptr.size(); ++i) {
    EXPECT_EQ(expect_cs_ptr[i], csc_weighted.column_segment_ptr[i]);
  }

  for (int i = 0; i < expect_row_indices.size(); ++i) {
    EXPECT_EQ(expect_row_indices[i], csc_weighted.row_indices[i]);
  }

  // sorting should be exact, no arithmetic needed. check for strict equality
  // of floats, not relative error
  std::array<float, 8> expect_weights = {
      {1.0f, 1.1f, 1.6f, 1.5f, 1.2f, 1.4f, 1.3f, 1.7f}};
  for (int i = 0; i < expect_weights.size(); ++i) {
    EXPECT_EQ(expect_weights[i], csc_weighted.weights[i]);
  }
}

TEST(CpuKernelTest, csr2csc_test_int32) {
  test_csr2csc<int32_t>();
}

TEST(CpuKernelTest, csr2csc_test_int64) {
  test_csr2csc<int64_t>();
}

namespace {

constexpr std::string_view kFinalOffsetLog =
    "The last element in offsets is incorrect";
constexpr std::string_view kOffsetsLog =
    "Setting indices_start and indices_end within the range";
constexpr std::string_view kIndicesLog = "Setting idx to zero";

struct BoundsCheckResult {
  at::Tensor indices;
  at::Tensor offsets;
  int64_t num_warnings;
  std::string log;

  bool logged(std::string_view message) const {
    return log.find(message) != std::string::npos;
  }
};

// Runs the op through the dispatcher in WARNING mode on one 4-row table with
// two bags.
BoundsCheckResult bounds_check_warning_mode(
    const std::vector<int64_t>& indices_values,
    const std::vector<int64_t>& offsets_values) {
  static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "fbgemm::bounds_check_indices", "");
  const auto indices = torch::tensor(indices_values, torch::kInt64);
  const auto offsets = torch::tensor(offsets_values, torch::kInt64);
  const auto warning = torch::zeros({1}, torch::kInt64);
  std::vector<c10::IValue> stack{
      /*rows_per_table=*/torch::tensor({4}, torch::kInt64),
      indices,
      offsets,
      static_cast<int64_t>(fbgemm_gpu::BoundsCheckMode::WARNING),
      warning,
      /*weights=*/c10::IValue(),
      /*B_offsets=*/c10::IValue(),
      /*max_B=*/int64_t{-1},
      /*b_t_map=*/c10::IValue(),
      /*info_B_num_bits=*/int64_t{-1},
      /*info_B_mask=*/int64_t{-1},
      /*bounds_check_version=*/int64_t{1},
      /*prefetch_pipeline=*/false};
  testing::internal::CaptureStderr();
  op.callBoxed(&stack);
  const auto log = testing::internal::GetCapturedStderr();
  return {indices, offsets, warning.item<int64_t>(), log};
}

BoundsCheckResult bounds_check_two_out_of_range_indices() {
  return bounds_check_warning_mode({1, 7, 9, 3}, {0, 2, 4});
}

struct BoundsCheckCase {
  const char* name;
  std::vector<int64_t> indices;
  std::vector<int64_t> offsets;
  std::vector<int64_t> expected_indices;
  std::vector<int64_t> expected_offsets;
  int64_t expected_num_warnings;
};

// Checks that the call corrects its inputs, counts every warning, and logs
// exactly expected_log of the three warning messages.
void expect_bounds_check(
    const BoundsCheckCase& bounds_case,
    std::optional<std::string_view> expected_log) {
  SCOPED_TRACE(bounds_case.name);
  const auto result =
      bounds_check_warning_mode(bounds_case.indices, bounds_case.offsets);
  EXPECT_TRUE(
      torch::equal(
          result.indices,
          torch::tensor(bounds_case.expected_indices, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(
          result.offsets,
          torch::tensor(bounds_case.expected_offsets, torch::kInt64)));
  EXPECT_EQ(result.num_warnings, bounds_case.expected_num_warnings);
  for (const auto message : {kFinalOffsetLog, kOffsetsLog, kIndicesLog}) {
    EXPECT_EQ(result.logged(message), expected_log == message)
        << message << "\nCaptured log:\n"
        << result.log;
  }
}

} // namespace

TEST(CpuKernelTest, bounds_check_warning_mode_rate_limits_log) {
  const auto expected_indices = torch::tensor({1, 0, 0, 3}, torch::kInt64);

  // The limiter is process-wide, so another test or a --gtest_repeat iteration
  // may have used up the current interval; wait for the next allowed log.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  auto logged_result = bounds_check_two_out_of_range_indices();
  while (!logged_result.logged(kIndicesLog) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    logged_result = bounds_check_two_out_of_range_indices();
  }
  ASSERT_TRUE(logged_result.logged(kIndicesLog));
  EXPECT_TRUE(torch::equal(logged_result.indices, expected_indices));
  EXPECT_EQ(logged_result.num_warnings, 2);

  const auto next_result = bounds_check_two_out_of_range_indices();
  EXPECT_FALSE(next_result.logged(kIndicesLog));
  EXPECT_TRUE(torch::equal(next_result.indices, expected_indices));
  EXPECT_EQ(next_result.num_warnings, 2);
}

TEST(CpuKernelTest, bounds_check_warning_mode_limiters_are_independent) {
  const BoundsCheckCase bad_final_offset{
      "bad final offset",
      /*indices=*/{1, 2, 3, 0},
      /*offsets=*/{0, 2, 5},
      /*expected_indices=*/{1, 2, 3, 0},
      /*expected_offsets=*/{0, 2, 4},
      /*expected_num_warnings=*/1};
  const BoundsCheckCase bad_offset{
      "bad non-final offset",
      /*indices=*/{1, 2, 3, 0},
      /*offsets=*/{0, 5, 4},
      /*expected_indices=*/{1, 2, 3, 0},
      /*expected_offsets=*/{0, 4, 4},
      /*expected_num_warnings=*/1};
  const BoundsCheckCase bad_index{
      "bad index",
      /*indices=*/{1, 2, 3, 9},
      /*offsets=*/{0, 2, 4},
      /*expected_indices=*/{1, 2, 3, 0},
      /*expected_offsets=*/{0, 2, 4},
      /*expected_num_warnings=*/1};
  const BoundsCheckCase bad_offset_and_index{
      "bad non-final offset and bad index",
      /*indices=*/{1, 2, 3, 9},
      /*offsets=*/{0, 5, 4},
      /*expected_indices=*/{1, 2, 3, 0},
      /*expected_offsets=*/{0, 4, 4},
      /*expected_num_warnings=*/2};

  // The limiters are process-wide, so wait out any interval started by another
  // test or a --gtest_repeat iteration.
  std::this_thread::sleep_for(std::chrono::milliseconds(10'500));

  expect_bounds_check(bad_final_offset, kFinalOffsetLog);
  expect_bounds_check(bad_offset, kOffsetsLog);
  // Only the first warning of a call is a log candidate, so this call's bad
  // index must neither log nor start the indices interval.
  expect_bounds_check(bad_offset_and_index, std::nullopt);
  expect_bounds_check(bad_index, kIndicesLog);

  expect_bounds_check(bad_final_offset, std::nullopt);
  expect_bounds_check(bad_offset, std::nullopt);
  expect_bounds_check(bad_index, std::nullopt);
}

#undef FBGEMM_MEM_CHECK_ONLY

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/fb/src/dram_kv_embedding_cache/fake_enrichment.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_cache.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/serving_qdq.h"

#include <ATen/ATen.h>
#include <c10/util/Float8_e4m3fn.h>
#include <fbgemm/QuantUtils.h>
#include <folly/coro/BlockingWait.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace kv_mem {
namespace {

constexpr int64_t kDim = 128;
constexpr int kBitRate = 4;
constexpr int64_t kGroupSize = 8;
constexpr int64_t kPackedBytes = kDim / 2 + 4;
constexpr int64_t kNumShards = 4;
constexpr serving_qdq::Format kRowwise{kBitRate, 0};
constexpr serving_qdq::Format kGrouped{kBitRate, kGroupSize};

// The publish quantizer's scalar reference, independent of the optimized path
// the code under test calls.
std::vector<std::uint8_t> quantize_ref(const c10::Half* row) {
  std::vector<std::uint8_t> packed(kPackedBytes);
  fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef<fbgemm::float16>(
      kBitRate,
      reinterpret_cast<const fbgemm::float16*>(row),
      1,
      kDim,
      packed.data());
  return packed;
}

std::vector<c10::Half> dequantize_ref(const std::vector<std::uint8_t>& packed) {
  std::vector<c10::Half> row(kDim);
  fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfRef<fbgemm::float16>(
      kBitRate,
      packed.data(),
      1,
      kPackedBytes,
      reinterpret_cast<fbgemm::float16*>(row.data()));
  return row;
}

std::vector<std::uint8_t> codes_of(const c10::Half* row) {
  const auto packed = quantize_ref(row);
  return {packed.begin(), packed.begin() + kDim / 2};
}

bool same_bits(const c10::Half* a, const c10::Half* b, int64_t n) {
  return std::memcmp(a, b, n * sizeof(c10::Half)) == 0;
}

// Publishing a row decodes back to it bit-for-bit.
bool is_serving_fixed_point(const c10::Half* row) {
  return same_bits(dequantize_ref(quantize_ref(row)).data(), row, kDim);
}

// The group-wise INT4 publish, written from its spec: per group of kGroupSize
// values, scale = e4m3fn(max|x| / 7) clamped to [2^-9, 448] and
// code = rint(x / scale) in [-7, 7]. Returns the decoded row.
std::vector<c10::Half> grouped_publish_ref(const c10::Half* row) {
  std::vector<c10::Half> decoded(kDim);
  for (int64_t start = 0; start < kDim; start += kGroupSize) {
    float abs_max = 0;
    for (int64_t j = start; j < start + kGroupSize; ++j) {
      abs_max = std::max(abs_max, std::abs(static_cast<float>(row[j])));
    }
    const float scale = static_cast<float>(
        c10::Float8_e4m3fn(std::clamp(abs_max / 7.0f, 1.0f / 512.0f, 448.0f)));
    for (int64_t j = start; j < start + kGroupSize; ++j) {
      // Integer codes: a stored code of 0 decodes to +0, never -0.
      const int code = static_cast<int>(std::clamp(
          std::nearbyint(static_cast<float>(row[j]) / scale), -7.0f, 7.0f));
      decoded[j] = c10::Half(static_cast<float>(code) * scale);
    }
  }
  return decoded;
}

bool is_grouped_serving_fixed_point(const c10::Half* row) {
  return same_bits(grouped_publish_ref(row).data(), row, kDim);
}

// Repeats the publish round trip from `row` until it stops changing. Unlike
// rowwise, a group whose first scale rounded up far enough to leave its largest
// code at 6 re-derives a smaller scale on the second pass, so the stored codes
// can differ from the first publish of the original row.
std::vector<c10::Half> grouped_fixed_point_ref(const c10::Half* row) {
  std::vector<c10::Half> current(row, row + kDim);
  for (int pass = 0; pass < 4; ++pass) {
    auto next = grouped_publish_ref(current.data());
    if (same_bits(next.data(), current.data(), kDim)) {
      break;
    }
    current = std::move(next);
  }
  return current;
}

const c10::Half* row_ptr(const at::Tensor& rows, int64_t r) {
  return rows.const_data_ptr<at::Half>() + r * rows.size(1);
}

// Unit-norm rows around a shared mean direction, like DEX embeddings: the mean
// widens each row's min/max range, which is what INT4 rounds against.
at::Tensor dex_like_rows(int64_t num_rows, uint64_t seed) {
  auto gen = at::make_generator<at::CPUGeneratorImpl>(seed);
  const auto mean = at::randn({1, kDim}, gen, at::kFloat) * 0.6;
  const auto rows = mean + at::randn({num_rows, kDim}, gen, at::kFloat) * 0.3;
  return (rows / rows.norm(2, {1}, /*keepdim=*/true)).to(at::kHalf);
}

c10::intrusive_ptr<EnrichmentConfig> fake_enrichment_config(
    int64_t serving_qdq_bit_rate,
    int64_t serving_qdq_group_size) {
  return c10::make_intrusive<EnrichmentConfig>(
      static_cast<int64_t>(EnrichmentType::IN_MEMORY_TEST_ONLY),
      /*provider_name=*/"",
      /*client_id=*/"",
      kDim,
      static_cast<int64_t>(EnrichmentResponseFormat::THRIFT_FLOAT),
      /*opentab_tier_name=*/"",
      /*opentab_payload_ids=*/"",
      /*opentab_payload_types=*/"",
      /*opentab_column_group_ids=*/"",
      /*opentab_vec_payload_indexes=*/"",
      /*opentab_timeout_ms=*/5000,
      /*opentab_batch_size=*/100,
      /*fs_tier=*/"",
      /*fs_caller_id=*/"",
      /*fs_timeout_ms=*/5000,
      /*fs_batch_size=*/500,
      /*fs_feature_group_id=*/0,
      /*fs_feature_group_name=*/"",
      /*fs_feature_name=*/"",
      /*laser_batch_size=*/0,
      /*laser_key_suffix=*/"",
      /*laser_key_suffixes=*/std::vector<std::string>{},
      serving_qdq_bit_rate,
      serving_qdq_group_size);
}

std::shared_ptr<DramKVEmbeddingCache<at::Half>> make_cache(
    int64_t serving_qdq_bit_rate,
    int64_t serving_qdq_group_size = 0) {
  return std::make_shared<DramKVEmbeddingCache<at::Half>>(
      kDim,
      /*uniform_init_lower=*/-0.1,
      /*uniform_init_upper=*/0.1,
      /*feature_evict_config=*/std::nullopt,
      kNumShards,
      /*num_threads=*/4,
      /*row_storage_bitwidth=*/16,
      /*backend_return_whole_row=*/false,
      /*enable_async_update=*/false,
      /*table_dims=*/std::nullopt,
      /*hash_size_cumsum=*/at::tensor({0, 100000}, at::kLong),
      /*is_training=*/true,
      /*disable_random_init=*/true,
      /*enable_raw_embedding_streaming=*/false,
      /*res_store_shards=*/0,
      /*res_server_port=*/0,
      /*table_names=*/std::vector<std::string>{},
      /*table_offsets=*/std::vector<int64_t>{},
      /*table_sizes=*/std::vector<int64_t>{},
      fake_enrichment_config(serving_qdq_bit_rate, serving_qdq_group_size));
}

at::Tensor read_rows(
    DramKVEmbeddingCache<at::Half>& cache,
    const at::Tensor& ids) {
  auto rows = at::zeros({ids.size(0), kDim}, at::kHalf);
  folly::coro::blockingWait(
      cache.get_kv_db_async(ids, rows, at::tensor({ids.size(0)}, at::kLong)));
  return rows;
}

at::Tensor wait_for_enriched_rows(
    DramKVEmbeddingCache<at::Half>& cache,
    const at::Tensor& ids) {
  for (int attempt = 0; attempt < 3000; ++attempt) {
    auto rows = read_rows(cache, ids);
    if ((rows.abs().amax({1}) > 0).all().item<bool>()) {
      return rows;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ADD_FAILURE() << "enrichment never filled the requested rows";
  return read_rows(cache, ids);
}

// What the fake enrichment backend returns for `ids`, cast to the cache's FP16.
at::Tensor fake_enrichment_rows(const std::vector<int64_t>& ids) {
  auto rows = at::empty({static_cast<int64_t>(ids.size()), kDim}, at::kFloat);
  for (size_t r = 0; r < ids.size(); ++r) {
    for (int64_t j = 0; j < kDim; ++j) {
      rows[r][j] = fake_enrichment::fakeEmbeddingValue(ids[r], j);
    }
  }
  return rows.to(at::kHalf);
}

TEST(ServingQdqTest, RoundTripRowIsServingFixedPointWithPublishCodes) {
  const auto original = dex_like_rows(4096, /*seed=*/0);
  auto rows = original.clone();

  const auto stats = serving_qdq::round_trip_rows(
      rows.mutable_data_ptr<at::Half>(), rows.size(0), kDim, kDim, kRowwise);

  EXPECT_EQ(stats.rows, rows.size(0));
  EXPECT_EQ(stats.not_converged, 0);
  EXPECT_GT(stats.changed, 0);
  for (int64_t r = 0; r < rows.size(0); ++r) {
    EXPECT_TRUE(is_serving_fixed_point(row_ptr(rows, r))) << "row " << r;
    EXPECT_EQ(codes_of(row_ptr(rows, r)), codes_of(row_ptr(original, r)))
        << "row " << r;
  }
}

TEST(ServingQdqTest, RoundTripRowLeavesColumnsPastDimUntouched) {
  constexpr int64_t kStride = kDim + 4;
  auto rows = at::full({8, kStride}, 7.0, at::kHalf);
  rows.narrow(1, 0, kDim).copy_(dex_like_rows(8, /*seed=*/1));
  const auto expected_tail = rows.narrow(1, kDim, 4).clone();

  serving_qdq::round_trip_rows(
      rows.mutable_data_ptr<at::Half>(), 8, kStride, kDim, kRowwise);

  EXPECT_TRUE(at::equal(rows.narrow(1, kDim, 4), expected_tail));
}

TEST(ServingQdqTest, ZeroRowStaysZero) {
  auto rows = at::zeros({1, kDim}, at::kHalf);

  const auto stats = serving_qdq::round_trip_rows(
      rows.mutable_data_ptr<at::Half>(), 1, kDim, kDim, kRowwise);

  EXPECT_EQ(stats.changed, 0);
  EXPECT_TRUE(at::equal(rows, at::zeros({1, kDim}, at::kHalf)));
}

TEST(ServingQdqTest, NonFiniteRowIsLeftAsIs) {
  auto rows = dex_like_rows(1, /*seed=*/2);
  rows[0][3] = std::numeric_limits<float>::infinity();
  const auto original = rows.clone();

  const auto stats = serving_qdq::round_trip_rows(
      rows.mutable_data_ptr<at::Half>(), 1, kDim, kDim, kRowwise);

  EXPECT_EQ(stats.non_finite, 1);
  EXPECT_TRUE(same_bits(row_ptr(rows, 0), row_ptr(original, 0), kDim));
}

TEST(ServingQdqTest, RowWhoseDecodeOverflowsIsLeftAsIs) {
  // Range [0, 65504]: the fp16 scale rounds up to 4368, so code 15 decodes to
  // 65520, which is fp16 inf.
  auto rows =
      at::linspace(0, 65504, kDim, at::kFloat).to(at::kHalf).view({1, kDim});
  const auto original = rows.clone();

  const auto stats = serving_qdq::round_trip_rows(
      rows.mutable_data_ptr<at::Half>(), 1, kDim, kDim, kRowwise);

  EXPECT_EQ(stats.non_finite, 1);
  EXPECT_TRUE(same_bits(row_ptr(rows, 0), row_ptr(original, 0), kDim));
}

TEST(ServingQdqTest, RestoredRowsAreRewrittenOnce) {
  auto cache = make_cache(kBitRate);
  const auto original = dex_like_rows(256, /*seed=*/3);
  const auto ids = at::arange(256, at::kLong);
  // The plain write path is what a checkpoint restore uses; it bypasses the
  // enrichment hooks.
  folly::coro::blockingWait(
      cache->set_kv_db_async(ids, original, at::tensor({256}, at::kLong)));

  const int64_t changed = cache->serving_qdq_existing_rows();
  const auto stored = read_rows(*cache, ids);

  EXPECT_GT(changed, 0);
  for (int64_t r = 0; r < stored.size(0); ++r) {
    EXPECT_TRUE(is_serving_fixed_point(row_ptr(stored, r))) << "row " << r;
    EXPECT_EQ(codes_of(row_ptr(stored, r)), codes_of(row_ptr(original, r)))
        << "row " << r;
  }
  EXPECT_EQ(cache->serving_qdq_existing_rows(), 0);
}

TEST(ServingQdqTest, RestoredRowsAreUntouchedWhenOff) {
  auto cache = make_cache(/*serving_qdq_bit_rate=*/0);
  const auto original = dex_like_rows(64, /*seed=*/4);
  const auto ids = at::arange(64, at::kLong);
  folly::coro::blockingWait(
      cache->set_kv_db_async(ids, original, at::tensor({64}, at::kLong)));

  EXPECT_EQ(cache->serving_qdq_existing_rows(), 0);
  const auto stored = read_rows(*cache, ids);
  EXPECT_TRUE(
      same_bits(row_ptr(stored, 0), row_ptr(original, 0), stored.numel()));
}

TEST(ServingQdqTest, EnrichmentFillStoresServingValues) {
  auto cache = make_cache(kBitRate);
  const std::vector<int64_t> ids = {1, 2, 3, 7};
  const auto id_tensor = at::tensor(ids, at::kLong);
  const auto fetched = fake_enrichment_rows(ids);

  cache->set_embedding_cache_enrich_query_id_async(
      id_tensor, id_tensor, at::tensor({4}, at::kLong));
  const auto stored = wait_for_enriched_rows(*cache, id_tensor);

  for (int64_t r = 0; r < stored.size(0); ++r) {
    EXPECT_TRUE(is_serving_fixed_point(row_ptr(stored, r))) << "row " << r;
    EXPECT_EQ(codes_of(row_ptr(stored, r)), codes_of(row_ptr(fetched, r)))
        << "row " << r;
  }
  EXPECT_FALSE(at::equal(stored, fetched));
}

TEST(ServingQdqTest, EnrichmentFillStoresFetchedRowsWhenOff) {
  auto cache = make_cache(/*serving_qdq_bit_rate=*/0);
  const std::vector<int64_t> ids = {1, 2, 3, 7};
  const auto id_tensor = at::tensor(ids, at::kLong);

  cache->set_embedding_cache_enrich_query_id_async(
      id_tensor, id_tensor, at::tensor({4}, at::kLong));
  const auto stored = wait_for_enriched_rows(*cache, id_tensor);

  EXPECT_TRUE(at::equal(stored, fake_enrichment_rows(ids)));
}

TEST(ServingQdqTest, GroupedRoundTripIsTheServingFixedPoint) {
  const auto original = dex_like_rows(4096, /*seed=*/5);
  auto rows = original.clone();

  const auto stats = serving_qdq::round_trip_rows(
      rows.mutable_data_ptr<at::Half>(), rows.size(0), kDim, kDim, kGrouped);

  EXPECT_EQ(stats.rows, rows.size(0));
  EXPECT_EQ(stats.not_converged, 0);
  EXPECT_GT(stats.multi_pass, 0);
  for (int64_t r = 0; r < rows.size(0); ++r) {
    EXPECT_TRUE(is_grouped_serving_fixed_point(row_ptr(rows, r)))
        << "row " << r;
    EXPECT_TRUE(same_bits(
        row_ptr(rows, r),
        grouped_fixed_point_ref(row_ptr(original, r)).data(),
        kDim))
        << "row " << r;
  }
}

TEST(ServingQdqTest, GroupSizeNeedsBitRate4) {
  auto rows = dex_like_rows(1, /*seed=*/6);
  EXPECT_THROW(
      serving_qdq::round_trip_rows(
          rows.mutable_data_ptr<at::Half>(),
          1,
          kDim,
          kDim,
          serving_qdq::Format{2, kGroupSize}),
      c10::Error);
}

TEST(ServingQdqTest, RestoredRowsAreRewrittenToGroupedServingValues) {
  auto cache = make_cache(kBitRate, kGroupSize);
  const auto original = dex_like_rows(256, /*seed=*/7);
  const auto ids = at::arange(256, at::kLong);
  folly::coro::blockingWait(
      cache->set_kv_db_async(ids, original, at::tensor({256}, at::kLong)));

  EXPECT_GT(cache->serving_qdq_existing_rows(), 0);
  const auto stored = read_rows(*cache, ids);

  for (int64_t r = 0; r < stored.size(0); ++r) {
    EXPECT_TRUE(same_bits(
        row_ptr(stored, r),
        grouped_fixed_point_ref(row_ptr(original, r)).data(),
        kDim))
        << "row " << r;
  }
  EXPECT_EQ(cache->serving_qdq_existing_rows(), 0);
}

TEST(ServingQdqTest, EnrichmentFillStoresGroupedServingValues) {
  auto cache = make_cache(kBitRate, kGroupSize);
  const std::vector<int64_t> ids = {1, 2, 3, 7};
  const auto id_tensor = at::tensor(ids, at::kLong);
  const auto fetched = fake_enrichment_rows(ids);

  cache->set_embedding_cache_enrich_query_id_async(
      id_tensor, id_tensor, at::tensor({4}, at::kLong));
  const auto stored = wait_for_enriched_rows(*cache, id_tensor);

  for (int64_t r = 0; r < stored.size(0); ++r) {
    EXPECT_TRUE(same_bits(
        row_ptr(stored, r),
        grouped_fixed_point_ref(row_ptr(fetched, r)).data(),
        kDim))
        << "row " << r;
  }
  EXPECT_FALSE(at::equal(stored, fetched));
}

} // namespace
} // namespace kv_mem

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
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
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
    int64_t serving_qdq_group_size,
    bool packed_storage = false,
    bool int8_storage = false) {
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
      serving_qdq_group_size,
      packed_storage,
      int8_storage);
}

struct CacheOptions {
  bool packed_storage{false};
  int64_t num_shards{kNumShards};
  int64_t max_D{kDim};
  bool backend_return_whole_row{false};
  std::optional<c10::intrusive_ptr<FeatureEvictConfig>> feature_evict_config;
  bool enable_ssd_backend{false};
  bool int8_storage{false};
};

std::shared_ptr<DramKVEmbeddingCache<at::Half>> make_cache(
    int64_t serving_qdq_bit_rate,
    int64_t serving_qdq_group_size = 0,
    const CacheOptions& options = {}) {
  return std::make_shared<DramKVEmbeddingCache<at::Half>>(
      options.max_D,
      /*uniform_init_lower=*/-0.1,
      /*uniform_init_upper=*/0.1,
      options.feature_evict_config,
      options.num_shards,
      /*num_threads=*/4,
      /*row_storage_bitwidth=*/16,
      options.backend_return_whole_row,
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
      fake_enrichment_config(
          serving_qdq_bit_rate,
          serving_qdq_group_size,
          options.packed_storage,
          options.int8_storage),
      options.enable_ssd_backend);
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

constexpr int64_t kTail = 4;
constexpr int64_t kPackedMaxD = kDim + kTail;
constexpr serving_qdq::PackedLayout kPackedLayout{kDim, kGroupSize, 0};

at::Tensor grouped_fixed_rows(int64_t num_rows, uint64_t seed) {
  auto rows = dex_like_rows(num_rows, seed);
  serving_qdq::round_trip_rows(
      rows.mutable_data_ptr<at::Half>(), num_rows, kDim, kDim, kGrouped);
  return rows;
}

// Grouped fixed points in the first kDim columns, arbitrary fp16 after.
at::Tensor fixed_rows_with_tail(int64_t num_rows, uint64_t seed) {
  auto gen = at::make_generator<at::CPUGeneratorImpl>(seed + 1000);
  return at::cat(
      {grouped_fixed_rows(num_rows, seed),
       at::randn({num_rows, kTail}, gen, at::kFloat).to(at::kHalf)},
      1);
}

std::shared_ptr<DramKVEmbeddingCache<at::Half>> make_packed_cache(
    CacheOptions options = {}) {
  options.packed_storage = true;
  options.max_D = kPackedMaxD;
  return make_cache(kBitRate, kGroupSize, options);
}

void write_rows(
    DramKVEmbeddingCache<at::Half>& cache,
    const at::Tensor& ids,
    const at::Tensor& rows) {
  folly::coro::blockingWait(
      cache.set_kv_db_async(ids, rows, at::tensor({ids.size(0)}, at::kLong)));
}

at::Tensor read_rows(
    DramKVEmbeddingCache<at::Half>& cache,
    const at::Tensor& ids,
    int64_t width,
    int64_t offset = 0) {
  auto rows = at::zeros({ids.size(0), width}, at::kHalf);
  cache.get_kv_from_storage_by_snapshot(ids, rows, nullptr, offset, width);
  return rows;
}

at::Tensor read_whole_rows(
    DramKVEmbeddingCache<at::Half>& cache,
    const at::Tensor& ids) {
  const int64_t meta_dim = FixedBlockPool::get_metaheader_dim<at::Half>();
  auto rows = at::zeros({ids.size(0), meta_dim + kPackedMaxD}, at::kHalf);
  cache.get_kv_with_metaheader_from_storage(ids, rows);
  return rows;
}

bool same_bits(const at::Tensor& a, const at::Tensor& b) {
  return a.sizes() == b.sizes() &&
      std::memcmp(
          a.const_data_ptr(),
          b.const_data_ptr(),
          a.numel() * a.element_size()) == 0;
}

std::vector<c10::Half> dequantize_grouped_ref(
    const std::vector<std::uint8_t>& packed,
    int64_t dim,
    int64_t group_size) {
  std::vector<c10::Half> row(dim);
  fbgemm_gpu::grouped_int4_e4m3::dequantize_row(
      packed.data(), dim, group_size, row.data());
  return row;
}

TEST(PackedStorageTest, GroupedFixedPointsPackExactly) {
  auto rows = dex_like_rows(4096, /*seed=*/10);
  const auto stats = serving_qdq::round_trip_rows(
      rows.mutable_data_ptr<at::Half>(), rows.size(0), kDim, kDim, kGrouped);
  ASSERT_GT(stats.multi_pass, 0);

  std::vector<std::uint8_t> payload(kPackedLayout.payload_bytes());
  std::vector<c10::Half> decoded(kDim);
  for (int64_t r = 0; r < rows.size(0); ++r) {
    ASSERT_TRUE(
        serving_qdq::pack_row(row_ptr(rows, r), kPackedLayout, payload.data()))
        << "row " << r;
    serving_qdq::unpack_row(payload.data(), kPackedLayout, decoded.data());
    EXPECT_TRUE(same_bits(decoded.data(), row_ptr(rows, r), kDim))
        << "row " << r;
  }
}

TEST(PackedStorageTest, UnpackMatchesReferenceDecodeForEveryScale) {
  // 17 groups of 8 exercise the vector loop and its odd tail; group size 16
  // takes the generic path.
  for (const auto& [dim, group_size] :
       std::vector<std::pair<int64_t, int64_t>>{{136, 8}, {128, 16}}) {
    const serving_qdq::PackedLayout layout{dim, group_size, 0};
    std::mt19937 rng(static_cast<uint32_t>(dim + group_size));
    std::vector<std::uint8_t> payload(layout.packed_bytes());
    std::vector<c10::Half> decoded(dim);
    const int64_t num_groups = dim / group_size;
    for (int64_t first_scale = 0; first_scale < 256;
         first_scale += num_groups) {
      for (int64_t b = 0; b < layout.code_bytes(); ++b) {
        payload[b] = static_cast<std::uint8_t>(rng());
      }
      for (int64_t g = 0; g < num_groups; ++g) {
        const auto scale = static_cast<std::uint8_t>((first_scale + g) % 256);
        // e4m3fn NaN scales never come out of the encoder.
        payload[layout.code_bytes() + g] = (scale & 0x7F) == 0x7F ? 0 : scale;
      }
      serving_qdq::unpack_row(payload.data(), layout, decoded.data());
      EXPECT_TRUE(same_bits(
          decoded.data(),
          dequantize_grouped_ref(payload, dim, group_size).data(),
          dim))
          << "dim " << dim << " group_size " << group_size << " scales from "
          << first_scale;
    }
  }
}

TEST(PackedStorageTest, OnlyServingFixedPointsPackExactly) {
  const auto rows = dex_like_rows(1024, /*seed=*/11);
  std::vector<std::uint8_t> payload(kPackedLayout.payload_bytes());
  int64_t exact = 0;
  for (int64_t r = 0; r < rows.size(0); ++r) {
    const bool packed =
        serving_qdq::pack_row(row_ptr(rows, r), kPackedLayout, payload.data());
    EXPECT_EQ(packed, is_grouped_serving_fixed_point(row_ptr(rows, r)))
        << "row " << r;
    exact += packed;
  }
  EXPECT_LT(exact, rows.size(0));

  for (const float bad :
       {std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    auto row = grouped_fixed_rows(1, /*seed=*/12);
    row[0][5] = bad;
    EXPECT_FALSE(
        serving_qdq::pack_row(row_ptr(row, 0), kPackedLayout, payload.data()));
  }
}

TEST(PackedStorageTest, ZeroRowPacksToZeroBytes) {
  const auto row = at::zeros({1, kDim}, at::kHalf);
  std::vector<std::uint8_t> payload(kPackedLayout.payload_bytes(), 0xAB);

  ASSERT_TRUE(
      serving_qdq::pack_row(row_ptr(row, 0), kPackedLayout, payload.data()));

  EXPECT_EQ(payload, std::vector<std::uint8_t>(payload.size(), 0));
}

TEST(PackedStorageTest, RowsRoundTripThroughEveryReadWidth) {
  auto cache = make_packed_cache();
  const auto rows = fixed_rows_with_tail(512, /*seed=*/13);
  const auto ids = at::arange(512, at::kLong);

  write_rows(*cache, ids, rows);

  EXPECT_TRUE(same_bits(read_rows(*cache, ids, kPackedMaxD), rows));
  EXPECT_TRUE(same_bits(
      read_rows(*cache, ids, kTail, /*offset=*/kDim),
      rows.narrow(1, kDim, kTail).contiguous()));
  EXPECT_TRUE(same_bits(
      read_rows(*cache, ids, 8, /*offset=*/4),
      rows.narrow(1, 4, 8).contiguous()));
  EXPECT_EQ(cache->get_packed_write_counts(), std::make_pair(0L, 0L));
}

TEST(PackedStorageTest, UnquantizedRowsAreStoredAsTheirServingFixedPoints) {
  auto cache = make_packed_cache();
  const auto original = dex_like_rows(256, /*seed=*/14);
  const auto ids = at::arange(256, at::kLong);

  write_rows(*cache, ids, original);

  const auto stored = read_rows(*cache, ids, kDim);
  int64_t not_fixed = 0;
  for (int64_t r = 0; r < stored.size(0); ++r) {
    not_fixed += !is_grouped_serving_fixed_point(row_ptr(original, r));
    EXPECT_TRUE(same_bits(
        row_ptr(stored, r),
        grouped_fixed_point_ref(row_ptr(original, r)).data(),
        kDim))
        << "row " << r;
  }
  EXPECT_EQ(cache->get_packed_write_counts(), std::make_pair(not_fixed, 0L));
  EXPECT_EQ(cache->serving_qdq_existing_rows(), 0);
}

TEST(PackedStorageTest, RowsWithoutFixedPointAreKeptExactly) {
  auto cache = make_packed_cache();
  auto rows = fixed_rows_with_tail(3, /*seed=*/15);
  rows[0][7] = std::numeric_limits<float>::infinity();
  rows[1][9] = std::numeric_limits<float>::quiet_NaN();
  const auto ids = at::arange(3, at::kLong);

  write_rows(*cache, ids, rows);

  EXPECT_TRUE(same_bits(read_rows(*cache, ids, kPackedMaxD), rows));
  EXPECT_EQ(cache->get_packed_write_counts(), std::make_pair(0L, 2L));
  EXPECT_EQ(cache->serving_qdq_existing_rows(), 0);

  const auto replacement = fixed_rows_with_tail(3, /*seed=*/16);
  write_rows(*cache, ids, replacement);
  EXPECT_TRUE(same_bits(read_rows(*cache, ids, kPackedMaxD), replacement));
}

TEST(PackedStorageTest, CheckpointRowsInterchangeWithFp16Storage) {
  const auto ids = at::arange(300, at::kLong);
  const CacheOptions whole_row{.backend_return_whole_row = true};
  auto packed = make_packed_cache(whole_row);
  write_rows(*packed, ids, fixed_rows_with_tail(300, /*seed=*/17));
  auto saved = read_whole_rows(*packed, ids);
  // Distinct saved timestamps, which a restore must carry over.
  for (int64_t r = 0; r < saved.size(0); ++r) {
    FixedBlockPool::set_timestamp(saved[r].data_ptr(), 1000000 + r);
  }

  auto fp16 = make_cache(
      kBitRate,
      kGroupSize,
      {.max_D = kPackedMaxD, .backend_return_whole_row = true});
  fp16->set_kv_with_metaheader_to_storage(saved.clone());
  EXPECT_TRUE(same_bits(read_whole_rows(*fp16, ids), saved));

  auto restored = make_packed_cache(whole_row);
  restored->set_kv_with_metaheader_to_storage(read_whole_rows(*fp16, ids));
  EXPECT_TRUE(same_bits(read_whole_rows(*restored, ids), saved));
  EXPECT_EQ(restored->get_packed_write_counts(), std::make_pair(0L, 0L));
}

TEST(PackedStorageTest, TtlEvictionDropsOnlyExpiredRows) {
  auto evict_config = c10::make_intrusive<FeatureEvictConfig>(
      /*trigger_mode=*/static_cast<int64_t>(EvictTriggerMode::MANUAL),
      /*trigger_strategy=*/
      static_cast<int64_t>(EvictTriggerStrategy::BY_TIMESTAMP),
      /*trigger_step_interval=*/std::nullopt,
      /*mem_util_threshold_in_GB=*/std::nullopt,
      /*ttls_in_mins=*/std::vector<int64_t>{60},
      /*counter_thresholds=*/std::nullopt,
      /*counter_decay_rates=*/std::nullopt,
      /*feature_score_counter_decay_rates=*/std::nullopt,
      /*training_id_eviction_trigger_count=*/std::nullopt,
      /*training_id_keep_count=*/std::nullopt,
      /*enable_eviction_for_feature_score_eviction_policy=*/std::nullopt,
      /*l2_weight_thresholds=*/std::nullopt,
      /*embedding_dims=*/std::nullopt);
  auto cache = make_packed_cache(
      {.backend_return_whole_row = true, .feature_evict_config = evict_config});
  const auto ids = at::arange(200, at::kLong);
  write_rows(*cache, ids, fixed_rows_with_tail(200, /*seed=*/18));
  auto rows = read_whole_rows(*cache, ids);
  const uint32_t now = FixedBlockPool::current_timestamp();
  for (int64_t r = 0; r < rows.size(0); ++r) {
    FixedBlockPool::set_timestamp(
        rows[r].data_ptr(), r % 2 == 0 ? now - 2 * 3600 : now);
  }
  cache->set_kv_with_metaheader_to_storage(rows.clone());

  cache->trigger_feature_evict();
  cache->resume_ongoing_eviction(/*force_resume=*/true);
  for (int attempt = 0; attempt < 1000 && cache->is_evicting(); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  cache->wait_until_eviction_done();

  const auto kept = at::arange(1, 200, 2, at::kLong);
  EXPECT_EQ(cache->get_num_rows(), kept.size(0));
  EXPECT_TRUE(
      same_bits(read_whole_rows(*cache, kept), rows.index_select(0, kept)));
}

TEST(PackedStorageTest, EnrichmentFillStoresGroupedServingValues) {
  auto cache = make_packed_cache();
  const std::vector<int64_t> ids = {1, 2, 3, 7};
  const auto id_tensor = at::tensor(ids, at::kLong);
  const auto fetched = fake_enrichment_rows(ids);

  cache->set_embedding_cache_enrich_query_id_async(
      id_tensor, id_tensor, at::tensor({4}, at::kLong));
  wait_for_enriched_rows(*cache, id_tensor);
  const auto stored = read_rows(*cache, id_tensor, kPackedMaxD);

  for (int64_t r = 0; r < stored.size(0); ++r) {
    EXPECT_TRUE(same_bits(
        row_ptr(stored, r),
        grouped_fixed_point_ref(row_ptr(fetched, r)).data(),
        kDim))
        << "row " << r;
  }
  EXPECT_EQ(cache->get_packed_write_counts(), std::make_pair(0L, 0L));
}

TEST(PackedStorageTest, PackedRowsUseFewerBytes) {
  auto fp16 = make_cache(kBitRate, kGroupSize, {.max_D = kPackedMaxD});
  auto packed = make_packed_cache();
  constexpr int64_t kRows = 200000;
  const auto ids = at::arange(kRows, at::kLong);
  const auto rows = fixed_rows_with_tail(kRows, /*seed=*/19);

  write_rows(*fp16, ids, rows);
  write_rows(*packed, ids, rows);

  // 16-byte MetaHeader + payload: 132 fp16 values vs 64 code bytes, 16 scale
  // bytes and 4 fp16 tail values.
  EXPECT_EQ(fp16->get_block_size(), 16 + kPackedMaxD * 2);
  EXPECT_EQ(packed->get_block_size(), 16 + kDim / 2 + kDim / kGroupSize + 8);
  const auto per_row = [](size_t bytes) {
    return static_cast<double>(bytes) / kRows;
  };
  std::cout << "bytes/row in pool chunks: fp16="
            << per_row(fp16->get_map_actual_used_chunk_in_bytes()) << " packed="
            << per_row(packed->get_map_actual_used_chunk_in_bytes())
            << "; blocks + map entries: fp16="
            << per_row(fp16->get_map_used_memsize_in_bytes())
            << " packed=" << per_row(packed->get_map_used_memsize_in_bytes())
            << '\n';
  EXPECT_LT(
      per_row(packed->get_map_actual_used_chunk_in_bytes()) * 2.5,
      per_row(fp16->get_map_actual_used_chunk_in_bytes()));
}

TEST(PackedStorageTest, ConcurrentReadersSeeWholeWrittenRows) {
  auto cache = make_packed_cache();
  constexpr int64_t kRows = 2048;
  const auto ids = at::arange(kRows, at::kLong);
  const std::array<at::Tensor, 2> versions = {
      fixed_rows_with_tail(kRows, /*seed=*/20),
      fixed_rows_with_tail(kRows, /*seed=*/21)};
  write_rows(*cache, ids, versions[0]);

  std::atomic<bool> stop{false};
  std::atomic<int64_t> torn{0};
  std::vector<std::thread> threads;
  threads.reserve(2);
  for (int w = 0; w < 2; ++w) {
    threads.emplace_back([&, w] {
      for (int iter = 0; !stop.load(); ++iter) {
        write_rows(*cache, ids, versions[(iter + w) % 2]);
      }
    });
  }
  for (int r = 0; r < 4; ++r) {
    threads.emplace_back([&] {
      while (!stop.load()) {
        const auto read = read_rows(*cache, ids, kPackedMaxD);
        for (int64_t i = 0; i < kRows; ++i) {
          const auto* row = row_ptr(read, i);
          if (!same_bits(row, row_ptr(versions[0], i), kPackedMaxD) &&
              !same_bits(row, row_ptr(versions[1], i), kPackedMaxD)) {
            torn++;
          }
        }
      }
    });
  }
  std::this_thread::sleep_for(std::chrono::seconds(2));
  stop = true;
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(torn.load(), 0);
}

TEST(PackedStorageTest, UnsupportedConfigsAreRefused) {
  EXPECT_THROW(
      make_cache(
          kBitRate, /*serving_qdq_group_size=*/0, {.packed_storage = true}),
      c10::Error);
  EXPECT_THROW(make_packed_cache({.enable_ssd_backend = true}), c10::Error);
  auto l2_config = c10::make_intrusive<FeatureEvictConfig>(
      /*trigger_mode=*/static_cast<int64_t>(EvictTriggerMode::MANUAL),
      /*trigger_strategy=*/
      static_cast<int64_t>(EvictTriggerStrategy::BY_L2WEIGHT),
      /*trigger_step_interval=*/std::nullopt,
      /*mem_util_threshold_in_GB=*/std::nullopt,
      /*ttls_in_mins=*/std::nullopt,
      /*counter_thresholds=*/std::nullopt,
      /*counter_decay_rates=*/std::nullopt,
      /*feature_score_counter_decay_rates=*/std::nullopt,
      /*training_id_eviction_trigger_count=*/std::nullopt,
      /*training_id_keep_count=*/std::nullopt,
      /*enable_eviction_for_feature_score_eviction_policy=*/std::nullopt,
      /*l2_weight_thresholds=*/std::vector<double>{0.1},
      /*embedding_dims=*/std::vector<int64_t>{kPackedMaxD});
  EXPECT_THROW(
      make_packed_cache({.feature_evict_config = l2_config}), c10::Error);
}

// Throughput of the packed read path; set KVZCH_PACKED_BENCH=1 and build in
// opt mode.
TEST(PackedStorageTest, Benchmark) {
  if (std::getenv("KVZCH_PACKED_BENCH") == nullptr) {
    GTEST_SKIP() << "set KVZCH_PACKED_BENCH=1";
  }
  using Clock = std::chrono::steady_clock;
  const auto seconds_since = [](Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  constexpr int64_t kRows = 1 << 20;
  const auto rows = fixed_rows_with_tail(kRows, /*seed=*/22);

  // Single-core decode of rows laid out back to back, as in a pool chunk.
  const int64_t packed_bytes = kPackedLayout.packed_bytes();
  std::vector<std::uint8_t> packed(kRows * packed_bytes);
  const auto pack_start = Clock::now();
  for (int64_t r = 0; r < kRows; ++r) {
    ASSERT_TRUE(
        serving_qdq::pack_row(
            row_ptr(rows, r), kPackedLayout, packed.data() + r * packed_bytes));
  }
  std::cout << "1 core, " << kRows << " rows: pack (write path) "
            << kRows / seconds_since(pack_start) / 1e6 << " M rows/s" << '\n';
  std::vector<c10::Half> out(kRows * kDim);
  for (int rep = 0; rep < 3; ++rep) {
    auto start = Clock::now();
    for (int64_t r = 0; r < kRows; ++r) {
      serving_qdq::unpack_row(
          packed.data() + r * packed_bytes,
          kPackedLayout,
          out.data() + r * kDim);
    }
    const double decode_s = seconds_since(start);
    start = Clock::now();
    for (int64_t r = 0; r < kRows; ++r) {
      std::memcpy(
          out.data() + r * kDim, row_ptr(rows, r), kDim * sizeof(c10::Half));
    }
    const double copy_s = seconds_since(start);
    std::cout << "1 core, " << kRows << " rows: decode "
              << kRows / decode_s / 1e6 << " M rows/s, fp16 memcpy "
              << kRows / copy_s / 1e6 << " M rows/s" << '\n';
  }
  // Same, visiting rows in random order, so every row comes from DRAM.
  const auto order = at::randperm(kRows, at::kLong);
  const auto* order_ptr = order.const_data_ptr<int64_t>();
  for (int rep = 0; rep < 3; ++rep) {
    auto start = Clock::now();
    for (int64_t i = 0; i < kRows; ++i) {
      const int64_t r = order_ptr[i];
      serving_qdq::unpack_row(
          packed.data() + r * packed_bytes,
          kPackedLayout,
          out.data() + i * kDim);
    }
    const double decode_s = seconds_since(start);
    start = Clock::now();
    for (int64_t i = 0; i < kRows; ++i) {
      std::memcpy(
          out.data() + i * kDim,
          row_ptr(rows, order_ptr[i]),
          kDim * sizeof(c10::Half));
    }
    const double copy_s = seconds_since(start);
    start = Clock::now();
    for (int64_t i = 0; i < kRows; ++i) {
      if (i + 8 < kRows) {
        const auto* ahead = packed.data() + order_ptr[i + 8] * packed_bytes;
        __builtin_prefetch(ahead);
        __builtin_prefetch(ahead + 64);
      }
      serving_qdq::unpack_row(
          packed.data() + order_ptr[i] * packed_bytes,
          kPackedLayout,
          out.data() + i * kDim);
    }
    std::cout << "1 core, random order, prefetch 8 ahead: decode "
              << kRows / seconds_since(start) / 1e6 << " M rows/s" << '\n';
    std::cout << "1 core, random order: decode " << kRows / decode_s / 1e6
              << " M rows/s, fp16 memcpy " << kRows / copy_s / 1e6
              << " M rows/s" << '\n';
  }

  // End-to-end get() of unique ids in random order, with the read knobs the
  // mb9 CU cache runs with (KVZCH_CU_CACHE_QPS_ENVS, 32 shards). The opt-in
  // gate keeps these out of every other test's process.
  for (const auto& [name, value] :
       std::vector<std::pair<const char*, const char*>>{
           {"KVZCH_DRAM_EXECUTOR_THREAD_CAP", "8"},
           {"KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE", "8"},
           {"KVZCH_DRAM_SEGMENTED_SHARD_READS", "1"},
           {"KVZCH_PARALLEL_DRAM_SHARD_INPUT", "1"},
           {"KVZCH_SKIP_DISABLED_INITIALIZERS", "1"},
           {"KVZCH_SKIP_DRAM_PERF_TIMING", "1"}}) {
    setenv(name, value, /*overwrite=*/0);
  }
  for (const int64_t num_rows : {kRows, int64_t{5'000'000}}) {
    const auto table =
        num_rows == kRows ? rows : fixed_rows_with_tail(num_rows, /*seed=*/23);
    auto fp16 = make_cache(
        kBitRate, kGroupSize, {.num_shards = 32, .max_D = kPackedMaxD});
    auto packed_cache = make_packed_cache({.num_shards = 32});
    const auto ids = at::randperm(num_rows, at::kLong);
    write_rows(*fp16, ids, table);
    write_rows(*packed_cache, ids, table);
    // Faulted in up front, so the timing is the read path alone.
    auto weights = at::zeros({num_rows, kPackedMaxD}, at::kHalf);
    for (int rep = 0; rep < 3; ++rep) {
      for (auto* cache : {fp16.get(), packed_cache.get()}) {
        const auto lookup = at::randperm(num_rows, at::kLong);
        const auto start = Clock::now();
        folly::coro::blockingWait(cache->get_kv_db_async(
            lookup, weights, at::tensor({num_rows}, at::kLong)));
        std::cout << (cache == fp16.get() ? "fp16  " : "packed") << " get() of "
                  << num_rows << " rows (" << cache->get_executor_num_threads()
                  << " threads, 32 shards): " << seconds_since(start) * 1e3
                  << " ms" << '\n';
      }
    }
  }
}

// The exclusive HBM tier with the GPU cache emulated by CPU tensors: tags,
// INT8 rows (codes then fp32 scale and bias) and last-access seconds.
struct FakeHbm {
  at::Tensor cache_state;
  at::Tensor cache_weights;
  at::Tensor last_access;

  explicit FakeHbm(int64_t slots, int64_t dim = kDim)
      : cache_state(at::full({slots, 1}, -1, at::kLong)),
        cache_weights(at::zeros({slots, dim + 8}, at::kByte)),
        last_access(at::zeros({slots}, at::kInt)) {}

  int64_t slot(int64_t key) const {
    auto h = static_cast<uint64_t>(key);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return static_cast<int64_t>(h % static_cast<uint32_t>(cache_state.size(0)));
  }

  int64_t tag(int64_t key) const {
    return cache_state[slot(key)][0].item<int64_t>();
  }

  int64_t occupied() const {
    return (cache_state >= 0).sum().item<int64_t>();
  }

  // kvzch_exclusive_cache_insert_int8: claims empty slots only, returns the
  // claimed keys.
  std::vector<int64_t>
  insert(const at::Tensor& keys, const at::Tensor& rows, int32_t now) {
    std::vector<int64_t> claimed;
    const int64_t dim = rows.size(1);
    for (int64_t n = 0; n < keys.size(0); ++n) {
      const auto key = keys[n].item<int64_t>();
      const auto row = rows[n].to(at::kFloat);
      if (key < 0 || !(row != 0).any().item<bool>()) {
        continue;
      }
      const auto s = slot(key);
      auto* tag = cache_state.data_ptr<int64_t>() + s;
      if (*tag != -1) {
        continue;
      }
      const auto* x = row.const_data_ptr<float>();
      // fminf/fmaxf, as the kernel reduces: NaN columns are skipped.
      float row_min = std::numeric_limits<float>::max();
      float row_max = std::numeric_limits<float>::lowest();
      for (int64_t d = 0; d < dim; ++d) {
        row_min = std::fmin(row_min, x[d]);
        row_max = std::fmax(row_max, x[d]);
      }
      const float range = row_max - row_min;
      const float scale = range == 0.0f ? 1.0f : range / 255.0f;
      auto* dst = cache_weights.data_ptr<uint8_t>() + s * (dim + 8);
      for (int64_t d = 0; d < dim; ++d) {
        dst[d] = static_cast<uint8_t>(std::fmin(
            std::fmax(std::rint((x[d] - row_min) / scale), 0.0f), 255.0f));
      }
      std::memcpy(dst + dim, &scale, sizeof(float));
      std::memcpy(dst + dim + 4, &row_min, sizeof(float));
      last_access[s] = now;
      *tag = key;
      claimed.push_back(key);
    }
    return claimed;
  }

  // What the GPU gather returns for a resident key.
  std::vector<c10::Half> decode(int64_t key, int64_t dim = kDim) const {
    const auto s = slot(key);
    const auto* src = cache_weights.const_data_ptr<uint8_t>() + s * (dim + 8);
    float scale = 0;
    float bias = 0;
    std::memcpy(&scale, src + dim, sizeof(float));
    std::memcpy(&bias, src + dim + 4, sizeof(float));
    std::vector<c10::Half> row(dim);
    for (int64_t d = 0; d < dim; ++d) {
      row[d] = c10::Half(static_cast<float>(src[d]) * scale + bias);
    }
    return row;
  }
};

constexpr int64_t kHbmSlots = 1 << 14;

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    setenv(name, value, /*overwrite=*/1);
  }
  ~ScopedEnv() {
    unsetenv(name_);
  }

 private:
  const char* name_;
};

std::shared_ptr<DramKVEmbeddingCache<at::Half>> make_exclusive_cache(
    FakeHbm& hbm,
    int64_t bit_rate = 0,
    int64_t group_size = 0,
    CacheOptions options = {}) {
  auto cache = make_cache(bit_rate, group_size, options);
  cache->exclusive_hbm_register(
      hbm.cache_state, hbm.cache_weights, hbm.last_access);
  return cache;
}

// Writes `rows` for `ids`, moves every row the HBM tier accepts into it and
// returns the moved keys.
std::vector<int64_t> write_and_promote(
    DramKVEmbeddingCache<at::Half>& cache,
    FakeHbm& hbm,
    const at::Tensor& ids,
    const at::Tensor& rows) {
  write_rows(cache, ids, rows);
  auto claimed = hbm.insert(
      ids, rows, static_cast<int32_t>(FixedBlockPool::current_timestamp()));
  cache.exclusive_hbm_promote_async(claimed);
  cache.exclusive_hbm_wait();
  return claimed;
}

TEST(ExclusiveHbmTest, PromotionMovesTheOnlyCopyIntoHbm) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(2000, at::kLong);
  const auto rows = dex_like_rows(2000, /*seed=*/30);

  const auto moved = write_and_promote(*cache, hbm, ids, rows);

  ASSERT_GT(static_cast<int64_t>(moved.size()), 1500);
  EXPECT_EQ(
      static_cast<int64_t>(cache->get_num_rows()),
      2000 - static_cast<int64_t>(moved.size()));
  EXPECT_EQ(
      cache->exclusive_hbm_stats()[0], static_cast<int64_t>(moved.size()));
  EXPECT_EQ(hbm.occupied(), static_cast<int64_t>(moved.size()));
  // Checkpoint reads return the HBM rows as the GPU gather decodes them, and
  // backend rows unchanged; the key listing covers both tiers.
  const auto stored = read_rows(*cache, ids, kDim, /*offset=*/0);
  const std::unordered_set<int64_t> in_hbm(moved.begin(), moved.end());
  for (int64_t id = 0; id < 2000; ++id) {
    if (in_hbm.contains(id)) {
      EXPECT_TRUE(same_bits(row_ptr(stored, id), hbm.decode(id).data(), kDim))
          << "id " << id;
    } else {
      EXPECT_TRUE(same_bits(row_ptr(stored, id), row_ptr(rows, id), kDim))
          << "id " << id;
    }
  }
  EXPECT_EQ(cache->get_keys_in_range_impl(0, 100000).size(0), 2000);
  // A training read only reaches the backend for GPU misses; one that finds
  // a tombstone is counted.
  read_rows(*cache, at::tensor({moved[0]}, at::kLong));
  EXPECT_EQ(cache->exclusive_hbm_stats()[7], 1);
}

TEST(ExclusiveHbmTest, FailedPromotionEmptiesItsSlotsAndRaisesLater) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(100, at::kLong);
  const auto rows = dex_like_rows(100, /*seed=*/31);
  write_rows(*cache, ids, rows);
  const auto claimed = hbm.insert(
      ids, rows, static_cast<int32_t>(FixedBlockPool::current_timestamp()));
  ASSERT_FALSE(claimed.empty());

  cache->fail_next_hbm_promotion_for_test();
  cache->exclusive_hbm_promote_async(claimed);
  EXPECT_THROW(cache->exclusive_hbm_wait(), std::runtime_error);

  // No slot keeps a row the backend lost track of: DRAM still has every row.
  EXPECT_EQ(hbm.occupied(), 0);
  EXPECT_EQ(static_cast<int64_t>(cache->get_num_rows()), 100);
  EXPECT_TRUE(same_bits(
      read_rows(*cache, ids, kDim, /*offset=*/0),
      rows.narrow(1, 0, kDim).contiguous()));
  // The error is reported once; promotion works again.
  EXPECT_EQ(cache->exclusive_hbm_stats()[0], 0);
  const auto moved = write_and_promote(*cache, hbm, ids, rows);
  EXPECT_EQ(
      cache->exclusive_hbm_stats()[0], static_cast<int64_t>(moved.size()));
}

TEST(ExclusiveHbmTest, FoundMaskReadsReportHbmRowsWithoutCountingALoss) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(500, at::kLong);
  const auto moved =
      write_and_promote(*cache, hbm, ids, dex_like_rows(500, /*seed=*/44));
  const std::unordered_set<int64_t> in_hbm(moved.begin(), moved.end());
  // The write-side cache fill reads every written id with a found mask; ids
  // already in HBM must come back as present-in-HBM, not as missing.
  auto rows = at::zeros({500, kDim}, at::kHalf);
  auto found = at::full({500}, 7, at::kByte);

  folly::coro::blockingWait(cache->get_kv_db_async_with_found_mask(
      ids, rows, at::tensor({500}, at::kLong), found));

  const auto* mask = found.const_data_ptr<uint8_t>();
  for (int64_t id = 0; id < 500; ++id) {
    EXPECT_EQ(mask[id], in_hbm.contains(id) ? 2 : 1) << "id " << id;
  }
  const auto stats = cache->exclusive_hbm_stats();
  EXPECT_EQ(stats[7], 0);
  EXPECT_EQ(stats[8], static_cast<int64_t>(moved.size()));
}

TEST(ExclusiveHbmTest, SlotCollisionLeavesTheRowInDram) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  std::vector<int64_t> same_slot;
  for (int64_t key = 0; same_slot.size() < 2; ++key) {
    if (hbm.slot(key) == hbm.slot(0)) {
      same_slot.push_back(key);
    }
  }
  const auto ids = at::tensor(same_slot, at::kLong);
  const auto rows = dex_like_rows(2, /*seed=*/31);

  const auto moved = write_and_promote(*cache, hbm, ids, rows);

  ASSERT_EQ(moved, std::vector<int64_t>{same_slot[0]});
  EXPECT_EQ(static_cast<int64_t>(cache->get_num_rows()), 1);
  EXPECT_TRUE(same_bits(
      row_ptr(read_rows(*cache, ids.narrow(0, 1, 1), kDim, 0), 0),
      row_ptr(rows, 1),
      kDim));
}

TEST(ExclusiveHbmTest, PromotionOfAKeyNoLongerInTheSlotIsSkipped) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(4, at::kLong);
  write_rows(*cache, ids, dex_like_rows(4, /*seed=*/32));

  // Promotions are checked against the slot, so a stale one is a no-op.
  cache->exclusive_hbm_promote_async({0, 1, 2, 3});
  cache->exclusive_hbm_wait();

  EXPECT_EQ(static_cast<int64_t>(cache->get_num_rows()), 4);
  EXPECT_EQ(cache->exclusive_hbm_stats()[2], 4);
}

TEST(ExclusiveHbmTest, PromotionAfterAnInPlaceWriteKeepsTheNewRow) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(200, at::kLong);
  const auto rows = dex_like_rows(200, /*seed=*/45);
  write_rows(*cache, ids, rows);
  // The GPU claims slots for the rows it read...
  const auto claimed = hbm.insert(
      ids, rows, static_cast<int32_t>(FixedBlockPool::current_timestamp()));
  ASSERT_GT(claimed.size(), 100);
  // ...then a write (e.g. a Laser response) updates some of them in place
  // before the promotion runs.
  const auto rewritten = at::tensor(
      std::vector<int64_t>(claimed.begin(), claimed.begin() + 10), at::kLong);
  const auto fresh = dex_like_rows(10, /*seed=*/46);
  write_rows(*cache, rewritten, fresh);

  cache->exclusive_hbm_promote_async(claimed);
  cache->exclusive_hbm_wait();

  const auto stats = cache->exclusive_hbm_stats();
  EXPECT_EQ(stats[0], static_cast<int64_t>(claimed.size()) - 10);
  EXPECT_EQ(stats[2], 10);
  EXPECT_TRUE(same_bits(read_rows(*cache, rewritten, kDim, 0), fresh));
  for (int64_t r = 0; r < 10; ++r) {
    EXPECT_EQ(hbm.tag(claimed[r]), -1);
  }
  EXPECT_EQ(hbm.occupied(), stats[0]);
}

TEST(ExclusiveHbmTest, PromotionOfARowTheBackendNoLongerHasEmptiesItsSlot) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  // The GPU read these rows, then the evictor dropped them from the backend.
  const auto ids = at::arange(50, at::kLong);
  const auto claimed = hbm.insert(
      ids,
      dex_like_rows(50, /*seed=*/47),
      static_cast<int32_t>(FixedBlockPool::current_timestamp()));
  ASSERT_FALSE(claimed.empty());

  cache->exclusive_hbm_promote_async(claimed);
  cache->exclusive_hbm_wait();

  EXPECT_EQ(cache->exclusive_hbm_stats()[0], 0);
  EXPECT_EQ(hbm.occupied(), 0);
  EXPECT_EQ(cache->get_num_rows(), 0);
  EXPECT_EQ(cache->get_keys_in_range_impl(0, 100000).size(0), 0);
}

TEST(ExclusiveHbmTest, FillOnlyWritesSkipHbmRows) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(100, at::kLong);
  const auto moved =
      write_and_promote(*cache, hbm, ids, dex_like_rows(100, /*seed=*/48));
  ASSERT_FALSE(moved.empty());
  const auto key = at::tensor({moved[0]}, at::kLong);
  const auto before = hbm.decode(moved[0]);

  folly::coro::blockingWait(cache->set_kv_db_async_on_enrichment_executor(
      key,
      at::full({1, kDim}, -7.0, at::kHalf),
      at::tensor({1}, at::kLong),
      0,
      /*skip_nonzero_rows=*/true));
  cache->exclusive_hbm_wait();

  EXPECT_EQ(hbm.tag(moved[0]), moved[0]);
  EXPECT_EQ(
      cache->exclusive_hbm_stats()[0], static_cast<int64_t>(moved.size()));
  EXPECT_TRUE(same_bits(
      row_ptr(read_rows(*cache, key, kDim, 0), 0), before.data(), kDim));
}

TEST(ExclusiveHbmTest, IdCountEvictionCountsHbmRows) {
  // ID_COUNT trigger at 600 rows, keep 500, all rows past their TTL.
  auto evict_config = c10::make_intrusive<FeatureEvictConfig>(
      /*trigger_mode=*/static_cast<int64_t>(EvictTriggerMode::ID_COUNT),
      /*trigger_strategy=*/
      static_cast<int64_t>(EvictTriggerStrategy::BY_TIMESTAMP),
      /*trigger_step_interval=*/std::nullopt,
      /*mem_util_threshold_in_GB=*/std::nullopt,
      /*ttls_in_mins=*/std::vector<int64_t>{60},
      /*counter_thresholds=*/std::nullopt,
      /*counter_decay_rates=*/std::nullopt,
      /*feature_score_counter_decay_rates=*/std::nullopt,
      /*training_id_eviction_trigger_count=*/std::vector<int64_t>{600},
      /*training_id_keep_count=*/std::vector<int64_t>{500},
      /*enable_eviction_for_feature_score_eviction_policy=*/
      std::vector<int8_t>{1},
      /*l2_weight_thresholds=*/std::nullopt,
      /*embedding_dims=*/std::nullopt);
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(
      hbm,
      0,
      0,
      {.backend_return_whole_row = true, .feature_evict_config = evict_config});
  const auto ids = at::arange(550, at::kLong);
  write_rows(*cache, ids, dex_like_rows(550, /*seed=*/51));
  auto rows = read_whole_rows(*cache, ids);
  const uint32_t old = FixedBlockPool::current_timestamp() - 2 * 3600;
  for (int64_t r = 0; r < rows.size(0); ++r) {
    FixedBlockPool::set_timestamp(rows[r].data_ptr(), old);
  }
  cache->set_kv_with_metaheader_to_storage(rows.clone());
  // Move ~200 rows into HBM, then add rows so that DRAM alone stays under
  // the trigger while DRAM + HBM passes it.
  const auto hot = at::arange(200, at::kLong);
  cache->exclusive_hbm_promote_async(hbm.insert(
      hot,
      read_rows(*cache, hot),
      static_cast<int32_t>(FixedBlockPool::current_timestamp())));
  cache->exclusive_hbm_wait();
  const auto in_hbm = cache->exclusive_hbm_stats()[0];
  ASSERT_GT(in_hbm, 150);
  const auto more = at::arange(550, 650, at::kLong);
  write_rows(*cache, more, dex_like_rows(100, /*seed=*/52));
  auto more_rows = read_whole_rows(*cache, more);
  for (int64_t r = 0; r < more_rows.size(0); ++r) {
    FixedBlockPool::set_timestamp(more_rows[r].data_ptr(), old);
  }
  cache->set_kv_with_metaheader_to_storage(more_rows.clone());
  // DRAM alone (650 - in_hbm) is under the trigger; DRAM + HBM is not.
  ASSERT_LT(static_cast<int64_t>(cache->get_num_rows()), 600);

  cache->maybe_evict();
  cache->resume_ongoing_eviction(/*force_resume=*/true);
  for (int attempt = 0; attempt < 1000 && cache->is_evicting(); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  cache->wait_until_eviction_done();

  // Eviction (DRAM rows only) brings DRAM + HBM down to the keep count.
  EXPECT_EQ(cache->exclusive_hbm_stats()[0], in_hbm);
  EXPECT_EQ(static_cast<int64_t>(cache->get_num_rows()) + in_hbm, 500);
}

TEST(ExclusiveHbmTest, MetadataOfAKeyInNeitherTierIsZero) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(100, at::kLong);
  const auto moved =
      write_and_promote(*cache, hbm, ids, dex_like_rows(100, /*seed=*/49));
  ASSERT_FALSE(moved.empty());
  // A key listed for a checkpoint can leave both tiers before its metadata
  // read (a TTL drop); that read must not abort the trainer.
  const auto metadata = cache->get_kv_zch_eviction_metadata_impl(
      at::tensor({moved[0], int64_t{99999}}, at::kLong),
      at::tensor({2}, at::kLong));
  EXPECT_NE(metadata[0].item<int64_t>(), 0);
  EXPECT_EQ(metadata[1].item<int64_t>(), 0);
}

TEST(ExclusiveHbmTest, DemoteAllMovesEveryRowBackWithItsTimestamp) {
  FakeHbm hbm(kHbmSlots);
  auto cache =
      make_exclusive_cache(hbm, 0, 0, {.backend_return_whole_row = true});
  const auto ids = at::arange(1000, at::kLong);
  const auto rows = dex_like_rows(1000, /*seed=*/33);
  const auto moved = write_and_promote(*cache, hbm, ids, rows);
  // A GPU hit after promotion is newer than the row's backend timestamp.
  const int32_t hit_time = FixedBlockPool::current_timestamp() + 1000;
  hbm.last_access[hbm.slot(moved[0])] = hit_time;
  std::vector<std::vector<c10::Half>> decoded;
  decoded.reserve(moved.size());
  for (const auto key : moved) {
    decoded.push_back(hbm.decode(key));
  }

  EXPECT_EQ(
      cache->exclusive_hbm_demote_all(), static_cast<int64_t>(moved.size()));

  EXPECT_EQ(static_cast<int64_t>(cache->get_num_rows()), 1000);
  EXPECT_EQ(cache->exclusive_hbm_stats()[0], 0);
  EXPECT_EQ(hbm.occupied(), 0);
  const auto stored = read_rows(*cache, ids);
  for (size_t i = 0; i < moved.size(); ++i) {
    EXPECT_TRUE(same_bits(row_ptr(stored, moved[i]), decoded[i].data(), kDim));
  }
  const auto metadata = cache->get_kv_zch_eviction_metadata_impl(
      at::tensor({moved[0]}, at::kLong), at::tensor({1}, at::kLong));
  EXPECT_EQ(
      static_cast<uint32_t>(metadata[0].item<int64_t>() & 0xFFFFFFFF),
      static_cast<uint32_t>(hit_time));
}

TEST(ExclusiveHbmTest, WriteToAnHbmRowMovesItBackAndEmptiesTheSlot) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(200, at::kLong);
  const auto moved =
      write_and_promote(*cache, hbm, ids, dex_like_rows(200, /*seed=*/34));
  const auto key = at::tensor({moved[0]}, at::kLong);
  const auto fresh = dex_like_rows(1, /*seed=*/35);

  write_rows(*cache, key, fresh);
  cache->exclusive_hbm_wait();

  EXPECT_EQ(hbm.tag(moved[0]), -1);
  EXPECT_EQ(cache->exclusive_hbm_stats()[5], 1);
  EXPECT_EQ(
      static_cast<int64_t>(cache->get_num_rows()),
      200 - static_cast<int64_t>(moved.size()) + 1);
  EXPECT_TRUE(same_bits(
      row_ptr(read_rows(*cache, key, kDim, 0), 0), row_ptr(fresh, 0), kDim));
}

TEST(ExclusiveHbmTest, EnrichmentDoesNotRefetchHbmRows) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(1, 65, at::kLong);
  const auto moved =
      write_and_promote(*cache, hbm, ids, dex_like_rows(64, /*seed=*/36));
  const auto rows_before = static_cast<int64_t>(cache->get_num_rows());
  const auto moved_ids = at::tensor(moved, at::kLong);

  cache->set_embedding_cache_enrich_query_id_async(
      moved_ids, moved_ids, at::tensor({moved_ids.size(0)}, at::kLong));
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  cache->exclusive_hbm_wait();

  EXPECT_EQ(static_cast<int64_t>(cache->get_num_rows()), rows_before);
  EXPECT_EQ(
      cache->exclusive_hbm_stats()[0], static_cast<int64_t>(moved.size()));
  for (const auto key : moved) {
    EXPECT_EQ(hbm.tag(key), key);
  }
}

TEST(ExclusiveHbmTest, TtlSweepMovesBackOnlyRowsIdleLongerThanTheTtl) {
  ScopedEnv ttl("KVZCH_EXCLUSIVE_HBM_TTL_MINS", "1");
  ScopedEnv sweep("KVZCH_EXCLUSIVE_HBM_SWEEP_S", "0");
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(500, at::kLong);
  const auto moved =
      write_and_promote(*cache, hbm, ids, dex_like_rows(500, /*seed=*/37));
  const int64_t later = FixedBlockPool::current_timestamp() + 600;
  std::unordered_set<int64_t> hot;
  for (size_t i = 0; i < moved.size(); i += 2) {
    hot.insert(moved[i]);
    hbm.last_access[hbm.slot(moved[i])] = static_cast<int32_t>(later - 5);
  }

  const auto left = cache->exclusive_hbm_sweep(later);

  EXPECT_EQ(
      left,
      static_cast<int64_t>(moved.size()) - static_cast<int64_t>(hot.size()));
  EXPECT_EQ(cache->exclusive_hbm_stats()[0], static_cast<int64_t>(hot.size()));
  EXPECT_EQ(
      static_cast<int64_t>(cache->get_num_rows()),
      500 - static_cast<int64_t>(hot.size()));
  for (const auto key : moved) {
    EXPECT_EQ(hbm.tag(key), hot.contains(key) ? key : -1) << "key " << key;
  }
}

TEST(ExclusiveHbmTest, TtlSweepCanDropFromBothTiers) {
  ScopedEnv ttl("KVZCH_EXCLUSIVE_HBM_TTL_MINS", "1");
  ScopedEnv demote("KVZCH_EXCLUSIVE_HBM_TTL_DEMOTE", "0");
  ScopedEnv sweep("KVZCH_EXCLUSIVE_HBM_SWEEP_S", "0");
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  const auto ids = at::arange(300, at::kLong);
  const auto moved =
      write_and_promote(*cache, hbm, ids, dex_like_rows(300, /*seed=*/38));

  EXPECT_EQ(
      cache->exclusive_hbm_sweep(FixedBlockPool::current_timestamp() + 600),
      static_cast<int64_t>(moved.size()));

  EXPECT_EQ(
      static_cast<int64_t>(cache->get_num_rows()),
      300 - static_cast<int64_t>(moved.size()));
  EXPECT_EQ(
      cache->exclusive_hbm_stats()[4], static_cast<int64_t>(moved.size()));
  EXPECT_EQ(hbm.occupied(), 0);
}

TEST(ExclusiveHbmTest, CheckpointRoundTripRestoresHbmRowsIntoDram) {
  FakeHbm hbm(kHbmSlots);
  const CacheOptions whole_row{.backend_return_whole_row = true};
  auto cache = make_exclusive_cache(hbm, 0, 0, whole_row);
  const auto ids = at::arange(800, at::kLong);
  const auto moved =
      write_and_promote(*cache, hbm, ids, dex_like_rows(800, /*seed=*/39));
  const int64_t meta_dim = FixedBlockPool::get_metaheader_dim<at::Half>();
  const auto keys = cache->get_keys_in_range_impl(0, 100000).view(-1);
  ASSERT_EQ(keys.size(0), 800);

  auto saved = at::zeros({800, meta_dim + kDim}, at::kHalf);
  cache->get_kv_from_storage_by_snapshot(keys, saved, nullptr, 0, std::nullopt);
  auto restored = make_cache(0, 0, whole_row);
  restored->set_kv_with_metaheader_to_storage(saved.clone());

  EXPECT_EQ(static_cast<int64_t>(restored->get_num_rows()), 800);
  auto reread = at::zeros_like(saved);
  restored->get_kv_from_storage_by_snapshot(
      keys, reread, nullptr, 0, std::nullopt);
  EXPECT_TRUE(same_bits(reread, saved));
  const std::unordered_set<int64_t> in_hbm(moved.begin(), moved.end());
  const auto* key_ptr = keys.const_data_ptr<int64_t>();
  for (int64_t r = 0; r < 800; ++r) {
    EXPECT_EQ(FixedBlockPool::get_key(saved[r].data_ptr()), key_ptr[r]);
    if (in_hbm.contains(key_ptr[r])) {
      EXPECT_TRUE(same_bits(
          row_ptr(saved, r) + meta_dim, hbm.decode(key_ptr[r]).data(), kDim));
    }
  }
}

TEST(ExclusiveHbmTest, PromotionDropsTheFp16FallbackRowOfAPackedRow) {
  FakeHbm hbm(kHbmSlots);
  auto cache =
      make_exclusive_cache(hbm, kBitRate, kGroupSize, {.packed_storage = true});
  // A row with no serving fixed point is kept as fp16 outside its block.
  const auto key = at::tensor({7}, at::kLong);
  auto nan_row = grouped_fixed_rows(1, /*seed=*/41);
  nan_row[0][3] = std::numeric_limits<float>::quiet_NaN();
  write_rows(*cache, key, nan_row);
  EXPECT_EQ(cache->packed_fallback_stats(), (std::vector<int64_t>{1, 0}));

  // Promotion frees the block, and its fallback row with it.
  const auto claimed = hbm.insert(
      key, nan_row, static_cast<int32_t>(FixedBlockPool::current_timestamp()));
  ASSERT_EQ(claimed.size(), 1);
  cache->exclusive_hbm_promote_async(claimed);
  cache->exclusive_hbm_wait();
  EXPECT_EQ(cache->exclusive_hbm_stats()[0], 1);
  EXPECT_EQ(cache->packed_fallback_stats(), (std::vector<int64_t>{0, 0}));

  // The freed block is reused by another key, which reads its own row.
  const auto other = at::tensor({8}, at::kLong);
  const auto other_row = grouped_fixed_rows(1, /*seed=*/42);
  write_rows(*cache, other, other_row);
  EXPECT_TRUE(same_bits(
      row_ptr(read_rows(*cache, other, kDim, 0), 0),
      row_ptr(other_row, 0),
      kDim));
  EXPECT_EQ(cache->packed_fallback_stats(), (std::vector<int64_t>{0, 0}));
}

TEST(ExclusiveHbmTest, DemotionToPackedStorageStoresGroupedFixedPoints) {
  FakeHbm hbm(kHbmSlots);
  auto cache =
      make_exclusive_cache(hbm, kBitRate, kGroupSize, {.packed_storage = true});
  constexpr int64_t kRows = 4000;
  const auto ids = at::arange(kRows, at::kLong);
  const auto original = grouped_fixed_rows(kRows, /*seed=*/40);
  const auto moved = write_and_promote(*cache, hbm, ids, original);

  cache->exclusive_hbm_demote_all();

  const auto stored = read_rows(*cache, ids, kDim, 0);
  int64_t exact = 0;
  for (const auto key : moved) {
    EXPECT_TRUE(is_grouped_serving_fixed_point(row_ptr(stored, key)));
    exact += same_bits(row_ptr(stored, key), row_ptr(original, key), kDim);
  }
  EXPECT_EQ(cache->get_packed_write_counts().second, 0);
  std::cout << "g8 rows through INT8 HBM and back: " << exact << "/"
            << static_cast<int64_t>(moved.size()) << " bit-exact; "
            << cache->exclusive_hbm_stats()[6]
            << " INT8 decodes were not g8 fixed points and were re-snapped"
            << '\n';
  EXPECT_GT(exact * 10, static_cast<int64_t>(moved.size()) * 9);
}

TEST(ExclusiveHbmTest, GroupedRowDriftOverRepeatedHbmTripsIsBounded) {
  FakeHbm hbm(kHbmSlots);
  auto cache =
      make_exclusive_cache(hbm, kBitRate, kGroupSize, {.packed_storage = true});
  constexpr int64_t kRows = 4000;
  constexpr int kCycles = 8;
  const auto ids = at::arange(kRows, at::kLong);
  const auto original = grouped_fixed_rows(kRows, /*seed=*/50);
  write_rows(*cache, ids, original);
  auto previous = original.clone();
  std::vector<int64_t> changed_per_cycle;
  float max_drift = 0;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    // Every row goes to HBM as the GPU reads it, and comes back.
    cache->exclusive_hbm_promote_async(hbm.insert(
        ids,
        read_rows(*cache, ids),
        static_cast<int32_t>(FixedBlockPool::current_timestamp())));
    cache->exclusive_hbm_wait();
    ASSERT_GT(cache->exclusive_hbm_stats()[0] * 10, kRows * 8);
    cache->exclusive_hbm_demote_all();
    const auto stored = read_rows(*cache, ids, kDim, 0);
    int64_t changed = 0;
    for (int64_t r = 0; r < kRows; ++r) {
      ASSERT_TRUE(is_grouped_serving_fixed_point(row_ptr(stored, r)));
      changed += !same_bits(row_ptr(stored, r), row_ptr(previous, r), kDim);
    }
    changed_per_cycle.push_back(changed);
    max_drift = std::max(
        max_drift,
        (stored.to(at::kFloat) - original.to(at::kFloat))
            .abs()
            .max()
            .item<float>());
    previous = stored;
  }
  std::cout << "g8 rows changed per HBM round trip:";
  for (const auto changed : changed_per_cycle) {
    std::cout << ' ' << changed;
  }
  std::cout << "; max |drift| from the original after " << kCycles
            << " trips: " << max_drift << '\n';
  // At most the first trip changes a row; after it every row is a fixed point
  // of the round trip, so repeated trips do not drift.
  EXPECT_LT(changed_per_cycle[0] * 10, kRows);
  for (int cycle = 1; cycle < kCycles; ++cycle) {
    EXPECT_EQ(changed_per_cycle[cycle], 0) << "trip " << cycle;
  }
}

TEST(ExclusiveHbmTest, Fp16RowsLoseOnlyInt8PrecisionOnce) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm);
  constexpr int64_t kRows = 4000;
  const auto ids = at::arange(kRows, at::kLong);
  const auto original = dex_like_rows(kRows, /*seed=*/41);
  const auto moved = write_and_promote(*cache, hbm, ids, original);
  cache->exclusive_hbm_demote_all();
  const auto once = read_rows(*cache, ids, kDim, 0);

  // A second trip changes nothing: an INT8 grid row re-quantizes to itself.
  const auto moved_again = write_and_promote(*cache, hbm, ids, once);
  cache->exclusive_hbm_demote_all();
  const auto twice = read_rows(*cache, ids, kDim, 0);

  float max_abs = 0;
  float max_rel = 0;
  for (const auto key : moved) {
    const auto err = (once[key].to(at::kFloat) - original[key].to(at::kFloat))
                         .abs()
                         .max()
                         .item<float>();
    const auto range = (original[key].max() - original[key].min())
                           .to(at::kFloat)
                           .item<float>();
    max_abs = std::max(max_abs, err);
    max_rel = std::max(max_rel, err / range);
  }
  int64_t stable = 0;
  for (const auto key : moved_again) {
    stable += same_bits(row_ptr(twice, key), row_ptr(once, key), kDim);
  }
  std::cout << "fp16 rows through INT8 HBM: max abs err " << max_abs
            << ", max err / row range " << max_rel << "; second trip " << stable
            << "/" << static_cast<int64_t>(moved_again.size()) << " unchanged"
            << '\n';
  EXPECT_LE(max_rel, 0.5f / 255.0f + 1e-3f);
  EXPECT_EQ(stable, static_cast<int64_t>(moved_again.size()));
}

// The row a key has after one trip through the INT8 HBM tier.
std::vector<c10::Half> int8_trip(const c10::Half* row) {
  FakeHbm one(1);
  auto rows = at::empty({1, kDim}, at::kHalf);
  std::memcpy(rows.data_ptr(), row, kDim * sizeof(c10::Half));
  const auto claimed = one.insert(at::zeros({1}, at::kLong), rows, 0);
  return claimed.empty() ? std::vector<c10::Half>(row, row + kDim)
                         : one.decode(0);
}

TEST(ExclusiveHbmTest, ConcurrentWritesPromotionsAndDemotionsLoseNoRow) {
  ScopedEnv ttl("KVZCH_EXCLUSIVE_HBM_TTL_MINS", "1");
  ScopedEnv sweep("KVZCH_EXCLUSIVE_HBM_SWEEP_S", "0");
  FakeHbm hbm(1 << 12);
  auto cache = make_exclusive_cache(hbm);
  constexpr int64_t kKeys = 3000;
  const auto all = at::arange(kKeys, at::kLong);
  // Key-derived rows: the last row written to each key, which the key must
  // hold at the end (as written, or after an INT8 trip through HBM).
  std::vector<at::Tensor> last(kKeys);
  {
    const auto initial = dex_like_rows(kKeys, /*seed=*/42);
    write_rows(*cache, all, initial);
    for (int64_t k = 0; k < kKeys; ++k) {
      last[k] = initial[k].clone();
    }
  }

  std::atomic<bool> stop{false};
  std::thread writer([&] {
    for (int iter = 0; !stop.load(); ++iter) {
      // Distinct ids, so the batch's last write to a key is unambiguous.
      const auto ids = at::randperm(kKeys, at::kLong).narrow(0, 0, 64);
      const auto rows = dex_like_rows(64, /*seed=*/1000 + iter);
      write_rows(*cache, ids, rows);
      for (int64_t r = 0; r < 64; ++r) {
        last[ids[r].item<int64_t>()] = rows[r].clone();
      }
    }
  });
  // Every HBM row is past the TTL at now + 120 s, so each pass demotes all.
  std::thread demoter([&] {
    while (!stop.load()) {
      cache->exclusive_hbm_sweep(FixedBlockPool::current_timestamp() + 120);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });
  // Checkpoint reads run alongside: no key is ever in neither tier, so no
  // row may read back as zero.
  std::atomic<int64_t> zero_reads{0};
  std::thread checkpointer([&] {
    while (!stop.load()) {
      const auto stored = read_rows(*cache, all, kDim, 0);
      zero_reads += (stored.abs().amax({1}) == 0).sum().item<int64_t>();
    }
  });
  // The "GPU": promotes rows it reads from the backend, like read-populate.
  // Only this thread claims slots; the executor alone empties them.
  for (int iter = 0; iter < 200; ++iter) {
    const auto ids = at::randint(kKeys, {64}, at::kLong);
    const auto rows = read_rows(*cache, ids);
    cache->exclusive_hbm_promote_async(hbm.insert(
        ids, rows, static_cast<int32_t>(FixedBlockPool::current_timestamp())));
    if (iter % 50 == 49) {
      cache->exclusive_hbm_wait();
      cache->exclusive_hbm_demote_all();
    }
  }
  stop = true;
  writer.join();
  demoter.join();
  checkpointer.join();
  cache->exclusive_hbm_wait();

  // Every key is in exactly one tier, and every tombstone holds its slot.
  const auto stats = cache->exclusive_hbm_stats();
  EXPECT_EQ(static_cast<int64_t>(cache->get_num_rows()) + stats[0], kKeys);
  EXPECT_EQ(hbm.occupied(), stats[0]);
  EXPECT_GT(stats[3], 0);
  EXPECT_EQ(zero_reads.load(), 0);
  const auto stored = read_rows(*cache, all, kDim, 0);
  int64_t lost = 0;
  for (int64_t k = 0; k < kKeys; ++k) {
    const auto* expected = last[k].const_data_ptr<at::Half>();
    lost += !same_bits(row_ptr(stored, k), expected, kDim) &&
        !same_bits(row_ptr(stored, k), int8_trip(expected).data(), kDim);
  }
  EXPECT_EQ(lost, 0);
}

TEST(ExclusiveHbmTest, ColumnsPastTheHbmRowReadBackAsZero) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(hbm, 0, 0, {.max_D = kPackedMaxD});
  const auto ids = at::arange(300, at::kLong);
  const auto rows = at::cat(
      {dex_like_rows(300, /*seed=*/43), at::ones({300, kTail}, at::kHalf)}, 1);
  write_rows(*cache, ids, rows);
  cache->exclusive_hbm_promote_async(hbm.insert(
      ids,
      rows.narrow(1, 0, kDim).contiguous(),
      static_cast<int32_t>(FixedBlockPool::current_timestamp())));
  cache->exclusive_hbm_wait();
  ASSERT_GT(cache->exclusive_hbm_stats()[0], 0);

  cache->exclusive_hbm_demote_all();

  const auto stored = read_rows(*cache, ids, kPackedMaxD, 0);
  for (int64_t id = 0; id < 300; ++id) {
    const bool was_in_hbm =
        stored[id].narrow(0, kDim, kTail).abs().sum().item<float>() == 0;
    EXPECT_TRUE(same_bits(
        row_ptr(stored, id),
        was_in_hbm ? hbm.decode(id).data() : row_ptr(rows, id),
        kDim));
  }
  EXPECT_EQ(static_cast<int64_t>(cache->get_num_rows()), 300);
}

TEST(ExclusiveHbmTest, UnsupportedConfigsAreRefused) {
  FakeHbm narrow(kHbmSlots, kDim - 4);
  EXPECT_THROW(make_exclusive_cache(narrow), c10::Error);
  FakeHbm hbm(kHbmSlots);
  auto ssd = make_cache(0, 0, {.enable_ssd_backend = true});
  EXPECT_THROW(
      ssd->exclusive_hbm_register(
          hbm.cache_state, hbm.cache_weights, hbm.last_access),
      c10::Error);
}

constexpr int64_t kInt8RowBytes = kDim + 8;
constexpr int64_t kInt8CheckpointRowBytes =
    16 + kInt8RowBytes + kTail * static_cast<int64_t>(sizeof(c10::Half));

// The INT8 publish (torch.ops.quantized.embedding_bag_byte_prepack) through
// fbgemm's scalar reference: dim codes, then fp32 scale and bias.
std::vector<std::uint8_t> int8_publish_ref(const c10::Half* row) {
  std::vector<std::uint8_t> packed(kInt8RowBytes);
  fbgemm::FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef<fbgemm::float16>(
      reinterpret_cast<const fbgemm::float16*>(row), 1, kDim, packed.data());
  return packed;
}

std::vector<std::uint8_t> int8_publish_of_float_ref(const c10::Half* row) {
  std::vector<float> values(kDim);
  for (int64_t d = 0; d < kDim; ++d) {
    values[d] = static_cast<float>(row[d]);
  }
  std::vector<std::uint8_t> packed(kInt8RowBytes);
  fbgemm::FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef<float>(
      values.data(), 1, kDim, packed.data());
  return packed;
}

std::vector<c10::Half> int8_decode_ref(const std::uint8_t* packed) {
  float scale = 0;
  float bias = 0;
  std::memcpy(&scale, packed + kDim, sizeof(float));
  std::memcpy(&bias, packed + kDim + 4, sizeof(float));
  std::vector<c10::Half> row(kDim);
  for (int64_t d = 0; d < kDim; ++d) {
    row[d] = c10::Half(static_cast<float>(packed[d]) * scale + bias);
  }
  return row;
}

// DEX-like rows in the first kDim columns, arbitrary fp16 after.
at::Tensor rows_with_tail(int64_t num_rows, uint64_t seed) {
  auto gen = at::make_generator<at::CPUGeneratorImpl>(seed + 2000);
  return at::cat(
      {dex_like_rows(num_rows, seed),
       at::randn({num_rows, kTail}, gen, at::kFloat).to(at::kHalf)},
      1);
}

std::shared_ptr<DramKVEmbeddingCache<at::Half>> make_int8_cache(
    CacheOptions options = {}) {
  options.int8_storage = true;
  options.max_D = kPackedMaxD;
  return make_cache(0, 0, options);
}

at::Tensor read_int8_rows(
    DramKVEmbeddingCache<at::Half>& cache,
    const at::Tensor& ids) {
  auto rows = at::zeros({ids.size(0), kInt8CheckpointRowBytes}, at::kByte);
  cache.get_kv_from_storage_by_snapshot(ids, rows, nullptr, 0, std::nullopt);
  return rows;
}

const std::uint8_t* int8_row_ptr(const at::Tensor& rows, int64_t r) {
  return rows.const_data_ptr<std::uint8_t>() + r * rows.size(1);
}

TEST(Int8StorageTest, EncodedRowsAreFixedPointsOfTheInt8Publish) {
  const auto rows = dex_like_rows(4096, /*seed=*/50);
  std::vector<std::uint8_t> encoded(kInt8RowBytes);
  int64_t multi_pass = 0;
  float max_rel = 0;
  for (int64_t r = 0; r < rows.size(0); ++r) {
    const auto result =
        serving_qdq::encode_int8_row(row_ptr(rows, r), kDim, encoded.data());
    ASSERT_NE(result, serving_qdq::Int8Result::kNotConverged) << "row " << r;
    multi_pass += result == serving_qdq::Int8Result::kMultiPass;
    auto decoded = int8_decode_ref(encoded.data());
    // Publishing the stored row (fp16 or widened to fp32) re-derives its codes,
    // scale and bias bit for bit.
    EXPECT_EQ(int8_publish_ref(decoded.data()), encoded) << "row " << r;
    EXPECT_EQ(int8_publish_of_float_ref(decoded.data()), encoded)
        << "row " << r;
    const auto original = rows[r].to(at::kFloat);
    const auto error =
        (at::from_blob(decoded.data(), {kDim}, at::kHalf).to(at::kFloat) -
         original)
            .abs()
            .max()
            .item<float>();
    const auto range = (original.max() - original.min()).item<float>();
    max_rel = std::max(max_rel, error / range);
  }
  std::cout << "FP16 -> INT8 -> FP16: max err / row range " << max_rel
            << "; rows needing a second re-quantization " << multi_pass << "/"
            << rows.size(0) << '\n';
  // Half an INT8 step plus fp16 rounding of the decode.
  EXPECT_LE(max_rel, 0.5f / 255.0f + 1e-3f);
}

TEST(Int8StorageTest, ZeroRowEncodesToZeroBytes) {
  const auto row = at::zeros({1, kDim}, at::kHalf);
  std::vector<std::uint8_t> encoded(kInt8RowBytes, 0xAB);

  serving_qdq::encode_int8_row(row_ptr(row, 0), kDim, encoded.data());

  EXPECT_EQ(encoded, std::vector<std::uint8_t>(kInt8RowBytes, 0));
}

TEST(Int8StorageTest, ReadsReturnTheInt8DecodeAtEveryWidth) {
  auto cache = make_int8_cache();
  const auto rows = rows_with_tail(512, /*seed=*/51);
  const auto ids = at::arange(512, at::kLong);

  write_rows(*cache, ids, rows);

  const auto stored = read_rows(*cache, ids, kPackedMaxD);
  std::vector<std::uint8_t> encoded(kInt8RowBytes);
  for (int64_t r = 0; r < 512; ++r) {
    serving_qdq::encode_int8_row(row_ptr(rows, r), kDim, encoded.data());
    EXPECT_TRUE(same_bits(
        row_ptr(stored, r), int8_decode_ref(encoded.data()).data(), kDim))
        << "row " << r;
  }
  EXPECT_TRUE(same_bits(
      stored.narrow(1, kDim, kTail).contiguous(),
      rows.narrow(1, kDim, kTail).contiguous()));
  EXPECT_TRUE(same_bits(
      read_rows(*cache, ids, 8, /*offset=*/4),
      stored.narrow(1, 4, 8).contiguous()));
  EXPECT_TRUE(same_bits(
      read_rows(*cache, ids, kTail, /*offset=*/kDim),
      rows.narrow(1, kDim, kTail).contiguous()));
  // Rewriting what was read stores the same row: no drift.
  write_rows(*cache, ids, stored);
  EXPECT_TRUE(same_bits(read_rows(*cache, ids, kPackedMaxD), stored));
  EXPECT_EQ(cache->get_int8_write_counts()[2], 0);
}

TEST(Int8StorageTest, NarrowWritesKeepTheRestOfTheRow) {
  auto cache = make_int8_cache();
  const auto ids = at::arange(16, at::kLong);
  write_rows(*cache, ids, rows_with_tail(16, /*seed=*/52));
  const auto before = read_rows(*cache, ids, kPackedMaxD);
  const auto head = dex_like_rows(16, /*seed=*/53).narrow(1, 0, 8).contiguous();

  write_rows(*cache, ids, head);

  const auto after = read_rows(*cache, ids, kPackedMaxD);
  EXPECT_TRUE(same_bits(
      after.narrow(1, kDim, kTail).contiguous(),
      before.narrow(1, kDim, kTail).contiguous()));
  EXPECT_LT(
      (after.narrow(1, 0, 8).to(at::kFloat) - head.to(at::kFloat))
          .abs()
          .max()
          .item<float>(),
      0.01f);
}

TEST(Int8StorageTest, NonFiniteValuesAreClampedAndCounted) {
  auto cache = make_int8_cache();
  auto rows = rows_with_tail(2, /*seed=*/54);
  rows[0][3] = std::numeric_limits<float>::quiet_NaN();
  rows[1][5] = std::numeric_limits<float>::infinity();
  const auto ids = at::arange(2, at::kLong);

  write_rows(*cache, ids, rows);

  const auto stored = read_rows(*cache, ids, kPackedMaxD);
  EXPECT_TRUE(at::isfinite(stored).all().item<bool>());
  EXPECT_EQ(cache->get_int8_write_counts()[1], 2);
}

TEST(Int8StorageTest, Int8CheckpointRoundTripIsBitExact) {
  const CacheOptions whole_row{.backend_return_whole_row = true};
  auto cache = make_int8_cache(whole_row);
  const auto ids = at::arange(300, at::kLong);
  write_rows(*cache, ids, rows_with_tail(300, /*seed=*/55));
  // Distinct saved timestamps, which the restore must carry over.
  auto with_ts = read_whole_rows(*cache, ids);
  for (int64_t r = 0; r < with_ts.size(0); ++r) {
    FixedBlockPool::set_timestamp(with_ts[r].data_ptr(), 1000000 + r);
  }
  cache->set_kv_with_metaheader_to_storage(with_ts.clone());
  const auto saved = read_int8_rows(*cache, ids);
  ASSERT_EQ(cache->get_int8_row_bytes(), kInt8CheckpointRowBytes);

  auto restored = make_int8_cache(whole_row);
  restored->set_range_to_storage(saved.clone(), 0, saved.size(0));

  EXPECT_TRUE(same_bits(read_int8_rows(*restored, ids), saved));
  EXPECT_TRUE(same_bits(read_whole_rows(*restored, ids), with_ts));
  for (int64_t r = 0; r < saved.size(0); ++r) {
    EXPECT_EQ(FixedBlockPool::get_key(saved[r].data_ptr()), r);
    EXPECT_EQ(
        FixedBlockPool::get_timestamp(saved[r].data_ptr()),
        static_cast<uint32_t>(1000000 + r));
  }
}

TEST(Int8StorageTest, RawRestoreRestampsWithKvzchRestampOnRestore) {
  const CacheOptions whole_row{.backend_return_whole_row = true};
  auto cache = make_int8_cache(whole_row);
  const auto ids = at::arange(100, at::kLong);
  write_rows(*cache, ids, rows_with_tail(100, /*seed=*/57));
  auto saved = read_int8_rows(*cache, ids);
  for (int64_t r = 0; r < saved.size(0); ++r) {
    FixedBlockPool::set_timestamp(saved[r].data_ptr(), 1000);
  }

  ScopedEnv restamp("KVZCH_RESTAMP_ON_RESTORE", "1");
  auto restored = make_int8_cache(whole_row);
  const auto before = FixedBlockPool::current_timestamp();
  restored->set_range_to_storage(saved.clone(), 0, saved.size(0));

  const auto reread = read_int8_rows(*restored, ids);
  for (int64_t r = 0; r < reread.size(0); ++r) {
    EXPECT_GE(FixedBlockPool::get_timestamp(reread[r].data_ptr()), before);
    // Only the timestamp changes: key, codes, qparams and tail are restored.
    EXPECT_EQ(FixedBlockPool::get_key(reread[r].data_ptr()), r);
    EXPECT_EQ(
        std::memcmp(
            int8_row_ptr(reread, r) + sizeof(FixedBlockPool::MetaHeader),
            int8_row_ptr(saved, r) + sizeof(FixedBlockPool::MetaHeader),
            kInt8CheckpointRowBytes - sizeof(FixedBlockPool::MetaHeader)),
        0);
  }
}

TEST(Int8StorageTest, Fp16CheckpointLoadsIntoAnInt8Table) {
  const auto ids = at::arange(300, at::kLong);
  const auto rows = rows_with_tail(300, /*seed=*/56);
  auto fp16 = make_cache(
      0, 0, {.max_D = kPackedMaxD, .backend_return_whole_row = true});
  write_rows(*fp16, ids, rows);
  auto saved = read_whole_rows(*fp16, ids);
  for (int64_t r = 0; r < saved.size(0); ++r) {
    FixedBlockPool::set_timestamp(saved[r].data_ptr(), 2000000 + r);
  }

  auto int8 = make_int8_cache({.backend_return_whole_row = true});
  int8->set_kv_with_metaheader_to_storage(saved.clone());

  const auto restored = read_whole_rows(*int8, ids);
  const auto restored_codes = read_int8_rows(*int8, ids);
  const int64_t meta_dim = FixedBlockPool::get_metaheader_dim<at::Half>();
  std::vector<std::uint8_t> encoded(kInt8RowBytes);
  for (int64_t r = 0; r < 300; ++r) {
    EXPECT_TRUE(same_bits(row_ptr(restored, r), row_ptr(saved, r), meta_dim));
    serving_qdq::encode_int8_row(row_ptr(rows, r), kDim, encoded.data());
    EXPECT_TRUE(same_bits(
        row_ptr(restored, r) + meta_dim,
        int8_decode_ref(encoded.data()).data(),
        kDim))
        << "row " << r;
    EXPECT_TRUE(same_bits(
        row_ptr(restored, r) + meta_dim + kDim,
        row_ptr(saved, r) + meta_dim + kDim,
        kTail));
    EXPECT_EQ(
        std::vector<std::uint8_t>(
            int8_row_ptr(restored_codes, r) + 16,
            int8_row_ptr(restored_codes, r) + 16 + kInt8RowBytes),
        encoded);
  }
}

TEST(Int8StorageTest, PublishOfStoredRowsReproducesTheStoredCodes) {
  auto cache = make_int8_cache();
  const auto ids = at::arange(2000, at::kLong);
  write_rows(*cache, ids, rows_with_tail(2000, /*seed=*/57));
  const auto codes = read_int8_rows(*cache, ids);

  // Publish reads the live table as fp16 and quantizes it.
  const auto published = read_rows(*cache, ids, kDim, 0);

  for (int64_t r = 0; r < 2000; ++r) {
    const std::vector<std::uint8_t> stored(
        int8_row_ptr(codes, r) + 16,
        int8_row_ptr(codes, r) + 16 + kInt8RowBytes);
    EXPECT_EQ(int8_publish_ref(row_ptr(published, r)), stored) << "row " << r;
  }
}

TEST(Int8StorageTest, HbmPromotionAndDemotionMoveBytesWithoutLoss) {
  FakeHbm hbm(kHbmSlots);
  auto cache = make_exclusive_cache(
      hbm, 0, 0, {.max_D = kPackedMaxD, .int8_storage = true});
  constexpr int64_t kRows = 3000;
  const auto ids = at::arange(kRows, at::kLong);
  write_rows(*cache, ids, rows_with_tail(kRows, /*seed=*/58));
  const auto before = read_int8_rows(*cache, ids);
  const auto values_before = read_rows(*cache, ids, kDim, 0);

  // The GPU quantizes the fp16 rows it reads from the backend.
  auto claimed = hbm.insert(
      ids,
      values_before,
      static_cast<int32_t>(FixedBlockPool::current_timestamp()));
  cache->exclusive_hbm_promote_async(claimed);
  cache->exclusive_hbm_wait();
  ASSERT_GT(static_cast<int64_t>(claimed.size()), 2000);

  int64_t same_hbm_bytes = 0;
  for (const auto key : claimed) {
    same_hbm_bytes += std::memcmp(
                          hbm.cache_weights.const_data_ptr<std::uint8_t>() +
                              hbm.slot(key) * kInt8RowBytes,
                          int8_row_ptr(before, key) + 16,
                          kInt8RowBytes) == 0;
  }
  // A checkpoint of the HBM rows carries the HBM bytes.
  const auto during = read_int8_rows(*cache, ids);
  cache->exclusive_hbm_demote_all();
  const auto after = read_int8_rows(*cache, ids);
  const auto values_after = read_rows(*cache, ids, kDim, 0);

  std::cout << "INT8 DRAM rows promoted to HBM: " << same_hbm_bytes << "/"
            << static_cast<int64_t>(claimed.size())
            << " HBM rows byte-identical to the DRAM row" << '\n';
  EXPECT_EQ(same_hbm_bytes, static_cast<int64_t>(claimed.size()));
  EXPECT_TRUE(same_bits(values_after, values_before));
  for (int64_t r = 0; r < kRows; ++r) {
    // Codes, scale and bias come back unchanged; HBM rows carry no tail.
    EXPECT_EQ(
        std::memcmp(
            int8_row_ptr(after, r) + 16,
            int8_row_ptr(before, r) + 16,
            kInt8RowBytes),
        0)
        << "row " << r;
    EXPECT_EQ(
        std::memcmp(
            int8_row_ptr(during, r) + 16,
            int8_row_ptr(before, r) + 16,
            kInt8RowBytes),
        0)
        << "row " << r;
  }
  EXPECT_EQ(cache->exclusive_hbm_stats()[6], 0);
}

TEST(Int8StorageTest, Int8RowsUseFewerBytes) {
  auto fp16 = make_cache(0, 0, {.max_D = kPackedMaxD});
  auto int8 = make_int8_cache();
  constexpr int64_t kRows = 200000;
  const auto ids = at::arange(kRows, at::kLong);
  const auto rows = rows_with_tail(kRows, /*seed=*/59);

  write_rows(*fp16, ids, rows);
  write_rows(*int8, ids, rows);

  // 16-byte MetaHeader + payload: 132 fp16 values vs 128 codes, fp32 scale
  // and bias and 4 fp16 tail values.
  EXPECT_EQ(fp16->get_block_size(), 16 + kPackedMaxD * 2);
  EXPECT_EQ(int8->get_block_size(), kInt8CheckpointRowBytes);
  const auto per_row = [](size_t bytes) {
    return static_cast<double>(bytes) / kRows;
  };
  const int64_t meta_dim = FixedBlockPool::get_metaheader_dim<at::Half>();
  const auto fp16_ckpt_row_bytes = (meta_dim + kPackedMaxD) * 2;
  std::cout << "bytes/row in pool chunks: fp16="
            << per_row(fp16->get_map_actual_used_chunk_in_bytes())
            << " int8=" << per_row(int8->get_map_actual_used_chunk_in_bytes())
            << "; blocks + map entries: fp16="
            << per_row(fp16->get_map_used_memsize_in_bytes())
            << " int8=" << per_row(int8->get_map_used_memsize_in_bytes())
            << "; checkpoint bytes/row: fp16=" << fp16_ckpt_row_bytes
            << " int8=" << int8->get_int8_row_bytes() << '\n';
  EXPECT_LT(
      per_row(int8->get_map_actual_used_chunk_in_bytes()) * 1.6,
      per_row(fp16->get_map_actual_used_chunk_in_bytes()));
  EXPECT_LT(int8->get_int8_row_bytes() * 1.7, fp16_ckpt_row_bytes);
}

TEST(Int8StorageTest, ConcurrentReadersSeeWholeWrittenRows) {
  auto cache = make_int8_cache();
  constexpr int64_t kRows = 2048;
  const auto ids = at::arange(kRows, at::kLong);
  // Rows already on the INT8 grid read back exactly.
  write_rows(*cache, ids, rows_with_tail(kRows, /*seed=*/60));
  const auto first = read_rows(*cache, ids, kPackedMaxD);
  write_rows(*cache, ids, rows_with_tail(kRows, /*seed=*/61));
  const auto second = read_rows(*cache, ids, kPackedMaxD);
  const std::array<at::Tensor, 2> versions = {first, second};

  std::atomic<bool> stop{false};
  std::atomic<int64_t> torn{0};
  std::vector<std::thread> threads;
  threads.reserve(2);
  for (int w = 0; w < 2; ++w) {
    threads.emplace_back([&, w] {
      for (int iter = 0; !stop.load(); ++iter) {
        write_rows(*cache, ids, versions[(iter + w) % 2]);
      }
    });
  }
  for (int r = 0; r < 4; ++r) {
    threads.emplace_back([&] {
      while (!stop.load()) {
        const auto read = read_rows(*cache, ids, kPackedMaxD);
        for (int64_t i = 0; i < kRows; ++i) {
          const auto* row = row_ptr(read, i);
          if (!same_bits(row, row_ptr(versions[0], i), kPackedMaxD) &&
              !same_bits(row, row_ptr(versions[1], i), kPackedMaxD)) {
            torn++;
          }
        }
      }
    });
  }
  std::this_thread::sleep_for(std::chrono::seconds(2));
  stop = true;
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(torn.load(), 0);
}

TEST(Int8StorageTest, UnsupportedConfigsAreRefused) {
  EXPECT_THROW(
      make_cache(
          kBitRate,
          kGroupSize,
          {.packed_storage = true, .max_D = kPackedMaxD, .int8_storage = true}),
      c10::Error);
  EXPECT_THROW(
      make_cache(kBitRate, 0, {.max_D = kPackedMaxD, .int8_storage = true}),
      c10::Error);
  EXPECT_THROW(make_int8_cache({.enable_ssd_backend = true}), c10::Error);
  // Without int8_storage, uint8 rows are not INT8 rows.
  auto fp16 = make_cache(0, 0, {.max_D = kPackedMaxD});
  EXPECT_EQ(fp16->get_int8_row_bytes(), 0);
  auto bytes = at::zeros({1, kInt8CheckpointRowBytes}, at::kByte);
  EXPECT_THROW(fp16->set_int8_rows_with_metaheader(bytes), c10::Error);
}

} // namespace
} // namespace kv_mem

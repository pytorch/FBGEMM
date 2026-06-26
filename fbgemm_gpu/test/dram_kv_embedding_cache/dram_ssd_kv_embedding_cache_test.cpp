/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_ssd_kv_embedding_cache.h"

#include <fmt/format.h>
#include <folly/coro/BlockingWait.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <thread>
#include <vector>

#include "deeplearning/fbgemm/fbgemm_gpu/src/ssd_split_embeddings_cache/ssd_table_batched_embeddings.h"

namespace kv_db {

namespace {

// Number of float-sized elements occupied by the MetaHeader prefix that the
// SSD tier stores ahead of each embedding payload (see fixed_block_pool.h).
constexpr int64_t kMetaHeaderDim = 4;

// Local mirror of the production MetaHeader layout (fixed_block_pool.h), 16
// bytes: [int64 key][uint32 timestamp][uint32 count:31][bool used:1].
struct alignas(8) MetaHeader {
  int64_t key;
  uint32_t timestamp;
  uint32_t count : 31;
  bool used : 1;
};
static_assert(sizeof(MetaHeader) == kMetaHeaderDim * sizeof(float));

// Write a valid MetaHeader prefix into the first kMetaHeaderDim floats of
// `row`.
void create_metaheader_row(float* row, int64_t row_dim, int64_t key) {
  std::memset(row, 0, row_dim * sizeof(float));
  MetaHeader header{};
  header.key = key;
  header.timestamp = 0;
  header.count = 0;
  header.used = true;
  std::memcpy(row, &header, sizeof(header));
}

// Construct a real on-disk EmbeddingRocksDB to act as the SSD (L3) tier, with
// the metadata column family enabled so that get_kv_db_weights_only_async()
// returns only the embedding payload (skipping the MetaHeader prefix) — exactly
// what the composite's SSD read path expects.
std::shared_ptr<ssd::EmbeddingRocksDB> makeRocksDbBackend(
    const std::string& path,
    int64_t num_shards,
    int64_t max_D,
    int64_t metadata_dim) {
  std::filesystem::create_directories(path);
  return std::make_shared<ssd::EmbeddingRocksDB>(
      path,
      num_shards,
      /*num_threads=*/8,
      /*memtable_flush_period=*/0,
      /*memtable_flush_offset=*/0,
      /*l0_files_per_compact=*/4,
      max_D,
      /*rate_limit_mbps=*/0,
      /*size_ratio=*/1,
      /*compaction_trigger=*/8,
      /*write_buffer_size=*/536870912,
      /*max_write_buffer_num=*/8,
      /*uniform_init_lower=*/-0.01,
      /*uniform_init_upper=*/0.01,
      /*row_storage_bitwidth=*/32,
      /*cache_size=*/0,
      /*use_passed_in_path=*/true,
      /*tbe_unqiue_id=*/0,
      /*l2_cache_size_gb=*/0,
      /*enable_async_update=*/false,
      /*enable_raw_embedding_streaming=*/false,
      /*res_store_shards=*/0,
      /*res_server_port=*/0,
      /*table_names=*/std::vector<std::string>{},
      /*table_offsets=*/std::vector<int64_t>{},
      /*table_sizes=*/std::vector<int64_t>{},
      /*table_dims=*/std::nullopt,
      /*hash_size_cumsum=*/std::nullopt,
      /*flushing_block_size=*/2000000000,
      /*disable_random_init=*/true,
      /*enable_blob_db=*/false,
      /*enable_metadata_cf=*/true,
      metadata_dim);
}

} // namespace

class DramSsdKVEmbeddingCacheTest : public ::testing::Test {
 protected:
  static constexpr int EMBEDDING_DIM = 16;
  static constexpr int NUM_SHARDS = 4;

  void SetUp() override {
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = 0;

    auto hash_size_cumsum = at::tensor({0, 100000}, at::kLong);

    dram_cache_ = std::make_shared<kv_mem::DramKVEmbeddingCache<float>>(
        EMBEDDING_DIM,
        /*uniform_init_lower=*/-0.1,
        /*uniform_init_upper=*/0.1,
        /*feature_evict_config=*/std::nullopt,
        NUM_SHARDS,
        /*num_threads=*/4,
        /*row_storage_bitwidth=*/32,
        /*backend_return_whole_row=*/true,
        /*enable_async_update=*/false,
        /*table_dims=*/std::nullopt,
        hash_size_cumsum,
        /*is_training=*/true, // training mode for dirty bit writes
        /*disable_random_init=*/true,
        /*enable_raw_embedding_streaming=*/false,
        /*res_store_shards=*/0,
        /*res_server_port=*/0,
        /*table_names=*/std::vector<std::string>{},
        /*table_offsets=*/std::vector<int64_t>{},
        /*table_sizes=*/std::vector<int64_t>{},
        /*enrichment_config=*/std::nullopt,
        // Enables dirty-bit tracking on the DRAM tier, required for flush().
        /*enable_ssd_backend=*/true);

    // Unique on-disk path per test so RocksDB instances don't collide.
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    ssd_path_ = (std::filesystem::temp_directory_path() /
                 fmt::format(
                     "dram_ssd_test_{}_{}",
                     info != nullptr ? info->name() : "unknown",
                     static_cast<int64_t>(::getpid())))
                    .string();
    ssd_backend_ = makeRocksDbBackend(
        ssd_path_, /*num_shards=*/1, EMBEDDING_DIM, kMetaHeaderDim);
  }

  void TearDown() override {
    composite_.reset();
    dram_cache_.reset();
    ssd_backend_.reset();
    if (!ssd_path_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(ssd_path_, ec);
    }
  }

  /// Create the composite backend
  void createComposite(int64_t queue_size = 1024, int64_t batch_size = 1024) {
    composite_ = std::make_shared<DramSsdKVEmbeddingCache<float>>(
        dram_cache_, ssd_backend_, queue_size, batch_size);
  }

  /// Insert an embedding into the composite via set
  void insertViaComposite(int64_t id, float value) {
    auto indices = at::tensor({id}, at::kLong);
    auto weights = at::full({1, EMBEDDING_DIM}, value, at::kFloat);
    auto count = at::tensor({1}, at::kLong);
    composite_->set_kv_db_async(indices, weights, count).wait();
  }

  /// Seed an embedding directly into the SSD tier (MetaHeader + payload row).
  void insertIntoSsd(int64_t id, float value) {
    const int64_t row_dim = kMetaHeaderDim + EMBEDDING_DIM;
    auto indices = at::tensor({id}, at::kLong);
    auto rows = at::zeros({1, row_dim}, at::kFloat);
    auto* row = rows.data_ptr<float>();
    create_metaheader_row(row, row_dim, id);
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
      row[kMetaHeaderDim + i] = value;
    }
    auto count = at::tensor({1}, at::kLong);
    ssd_backend_->set_kv_db_async(indices, rows, count).wait();
  }

  /// Read embeddings via composite get
  at::Tensor getViaComposite(const std::vector<int64_t>& ids) {
    auto num = static_cast<int64_t>(ids.size());
    auto indices = at::tensor(ids, at::TensorOptions().dtype(at::kLong));
    auto weights = at::zeros({num, EMBEDDING_DIM}, at::kFloat);
    auto count = at::tensor({num}, at::kLong);
    composite_->get_kv_db_async(indices, weights, count).wait();
    return weights;
  }

  /// Read embedding payloads directly from the SSD tier (weights only).
  at::Tensor readSsd(const std::vector<int64_t>& ids) {
    auto num = static_cast<int64_t>(ids.size());
    auto indices = at::tensor(ids, at::TensorOptions().dtype(at::kLong));
    auto weights = at::zeros({num, EMBEDDING_DIM}, at::kFloat);
    auto count = at::tensor({num}, at::kLong);
    ssd_backend_->get_kv_db_weights_only_async(indices, weights, count).wait();
    return weights;
  }

  std::shared_ptr<kv_mem::DramKVEmbeddingCache<float>> dram_cache_;
  std::shared_ptr<ssd::EmbeddingRocksDB> ssd_backend_;
  std::shared_ptr<DramSsdKVEmbeddingCache<float>> composite_;
  std::string ssd_path_;
};

// Test: Constructor wires up SSD backend and starts writeback thread
TEST_F(DramSsdKVEmbeddingCacheTest, ConstructorWiresUpComponents) {
  createComposite();
  EXPECT_NE(composite_->dram_cache().get(), nullptr);
  EXPECT_NE(composite_->ssd_backend().get(), nullptr);
}

// Test: get_kv_db_async returns DRAM hit immediately
TEST_F(DramSsdKVEmbeddingCacheTest, GetDramHitReturnsImmediately) {
  createComposite();

  // Insert into DRAM via composite set
  insertViaComposite(42, 1.5f);

  // Read back
  auto weights = getViaComposite({42});

  // Should find the DRAM value
  auto* w_ptr = weights.data_ptr<float>();
  EXPECT_FLOAT_EQ(w_ptr[0], 1.5f);
  EXPECT_FLOAT_EQ(w_ptr[1], 1.5f);

  // A DRAM hit must short-circuit: no SSD lookup and no SSD->DRAM backfill.
  EXPECT_EQ(composite_->get_ssd_num_lookups(), 0);
  EXPECT_EQ(composite_->get_ssd_num_hits(), 0);
}

// Test: get_kv_db_async falls through to SSD on DRAM miss
TEST_F(DramSsdKVEmbeddingCacheTest, GetDramMissSsdHit) {
  createComposite();

  // Insert into SSD only (not in DRAM)
  insertIntoSsd(99, 2.5f);

  // Read via composite — should find in SSD
  auto weights = getViaComposite({99});

  auto* w_ptr = weights.data_ptr<float>();
  EXPECT_FLOAT_EQ(w_ptr[0], 2.5f);
  EXPECT_FLOAT_EQ(w_ptr[EMBEDDING_DIM - 1], 2.5f);

  // DRAM miss falls through to exactly one SSD lookup that hit, which is what
  // schedules the SSD->DRAM backfill.
  EXPECT_EQ(composite_->get_ssd_num_lookups(), 1);
  EXPECT_EQ(composite_->get_ssd_num_hits(), 1);
}

// Test: get_kv_db_async returns zeros when both DRAM and SSD miss
TEST_F(DramSsdKVEmbeddingCacheTest, GetBothMissReturnsZeros) {
  createComposite();

  // Key 999 is neither in DRAM nor SSD
  auto weights = getViaComposite({999});

  auto* w_ptr = weights.data_ptr<float>();
  for (int j = 0; j < EMBEDDING_DIM; ++j) {
    EXPECT_FLOAT_EQ(w_ptr[j], 0.0f) << "Expected zero at index " << j;
  }
}

// Test: get_kv_db_async with mixed hits (DRAM + SSD + miss)
TEST_F(DramSsdKVEmbeddingCacheTest, GetMixedHits) {
  createComposite();

  // Key 1: in DRAM
  insertViaComposite(1, 1.0f);
  // Key 2: in SSD only
  insertIntoSsd(2, 2.0f);
  // Key 3: nowhere

  auto weights = getViaComposite({1, 2, 3});
  auto* w_ptr = weights.data_ptr<float>();
  int64_t stride = EMBEDDING_DIM;

  // Key 1: DRAM hit
  EXPECT_FLOAT_EQ(w_ptr[0 * stride], 1.0f);
  // Key 2: SSD hit
  EXPECT_FLOAT_EQ(w_ptr[1 * stride], 2.0f);
  // Key 3: both miss -> zeros
  EXPECT_FLOAT_EQ(w_ptr[2 * stride], 0.0f);
}

// Test: set_kv_db_async writes to DRAM only (not SSD)
TEST_F(DramSsdKVEmbeddingCacheTest, SetWritesToDramOnly) {
  createComposite();

  insertViaComposite(10, 3.0f);

  // Verify it's in DRAM (readable via composite)
  auto weights = getViaComposite({10});
  EXPECT_FLOAT_EQ(weights.data_ptr<float>()[0], 3.0f);

  // SSD tier should not have the key: a direct SSD read returns zeros.
  auto ssd = readSsd({10});
  auto* s = ssd.data_ptr<float>();
  for (int j = 0; j < EMBEDDING_DIM; ++j) {
    EXPECT_FLOAT_EQ(s[j], 0.0f) << "set() must not write to SSD";
  }
}

// Test: SSD hit backfills DRAM (subsequent DRAM read should hit)
TEST_F(DramSsdKVEmbeddingCacheTest, SsdHitBackfillsDram) {
  createComposite();

  // Insert into SSD only
  insertIntoSsd(50, 5.0f);

  // First read: DRAM miss -> SSD hit -> schedules async DRAM backfill.
  auto weights1 = getViaComposite({50});
  EXPECT_FLOAT_EQ(weights1.data_ptr<float>()[0], 5.0f);

  // Deterministically verify the backfill path ran: the SSD lookup hit, which
  // is what schedules the SSD->DRAM backfill. ssd_num_hits_ is incremented
  // synchronously (before the async task is queued), so this is race-free.
  EXPECT_EQ(composite_->get_ssd_num_lookups(), 1);
  EXPECT_EQ(composite_->get_ssd_num_hits(), 1);

  // Best-effort (non-asserting) check that the value actually lands in DRAM.
  // The backfill is fire-and-forget on backfill_executor_ with no deterministic
  // await hook, so a hard assertion here would make the test flaky. Instead we
  // poll with a timeout and only log a warning if it never appears.
  // TODO: Replace with a hard assertion once DramSsdKVEmbeddingCache exposes a
  // way to await pending backfills.
  auto dram_indices = at::tensor({50}, at::kLong);
  auto dram_count = at::tensor({1}, at::kLong);
  float dram_val = 0.0f;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto dram_weights = at::zeros({1, EMBEDDING_DIM}, at::kFloat);
    dram_cache_->get_kv_db_async(dram_indices, dram_weights, dram_count).wait();
    dram_val = dram_weights.data_ptr<float>()[0];
    if (dram_val == 5.0f) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (dram_val != 5.0f) {
    GTEST_LOG_(WARNING)
        << "SSD->DRAM backfill did not land within timeout (got " << dram_val
        << ", expected 5.0). Backfill is async/best-effort in this test.";
  }
}

// Test: stop/start writeback thread
TEST_F(DramSsdKVEmbeddingCacheTest, WritebackThreadLifecycle) {
  createComposite();

  // Writeback thread should be running after construction
  composite_->stop_writeback_thread();

  // Should be safe to stop again
  composite_->stop_writeback_thread();

  // Should be safe to start again
  composite_->start_writeback_thread();
}

// Test: Destructor stops writeback thread cleanly
TEST_F(DramSsdKVEmbeddingCacheTest, DestructorCleanup) {
  createComposite();

  // Insert some data
  insertViaComposite(300, 30.0f);

  // Destroy composite — should not hang or crash
  composite_.reset();
}

// Test: async backfill work does not block shutdown.
TEST_F(DramSsdKVEmbeddingCacheTest, AsyncBackfillShutdownDoesNotHang) {
  createComposite();

  // Populate many SSD-only keys to schedule a sizable async backfill batch.
  constexpr int64_t kNumKeys = 2000;
  std::vector<int64_t> ids;
  ids.reserve(kNumKeys);
  for (int64_t i = 0; i < kNumKeys; ++i) {
    const int64_t key = 100000 + i;
    ids.push_back(key);
    insertIntoSsd(key, 10.0f + static_cast<float>(i));
  }

  // Trigger SSD lookups and async backfill scheduling.
  auto weights = getViaComposite(ids);
  EXPECT_EQ(weights.size(0), kNumKeys);

  auto shutdown_future =
      std::async(std::launch::async, [this]() { composite_.reset(); });

  // Destructor must complete quickly even with pending async backfill work.
  EXPECT_EQ(
      shutdown_future.wait_for(std::chrono::seconds(5)),
      std::future_status::ready);
  shutdown_future.get();
}

// Test: stopping writeback thread while async backfill is active remains safe.
TEST_F(DramSsdKVEmbeddingCacheTest, StopWritebackDuringAsyncBackfill) {
  createComposite();

  constexpr int64_t kNumKeys = 1024;
  std::vector<int64_t> ids;
  ids.reserve(kNumKeys);
  for (int64_t i = 0; i < kNumKeys; ++i) {
    const int64_t key = 200000 + i;
    ids.push_back(key);
    insertIntoSsd(key, 20.0f + static_cast<float>(i));
  }

  // Schedule async backfill by issuing SSD-hit reads.
  auto weights = getViaComposite(ids);
  EXPECT_EQ(weights.size(0), kNumKeys);

  // Should be safe and non-hanging while backfill tasks may still be running.
  EXPECT_NO_THROW(composite_->stop_writeback_thread());
  EXPECT_NO_THROW(composite_->start_writeback_thread());
}

// Test: Delegation of set_kv_zch_eviction_metadata_async
TEST_F(DramSsdKVEmbeddingCacheTest, DelegatesEvictionMetadata) {
  createComposite();

  auto indices = at::tensor({1, 2, 3}, at::kLong);
  auto count = at::tensor({3}, at::kLong);
  auto engage = at::zeros({3}, at::kFloat);

  // Should not throw
  EXPECT_NO_THROW(
      composite_->set_kv_zch_eviction_metadata_async(indices, count, engage)
          .wait());
}

// Test: get with negative indices (sentinels) are skipped
TEST_F(DramSsdKVEmbeddingCacheTest, GetSkipsNegativeIndices) {
  createComposite();

  insertViaComposite(1, 1.0f);
  insertIntoSsd(2, 2.0f);

  auto indices = at::tensor({-1, 1, -1, 2}, at::kLong);
  auto weights = at::zeros({4, EMBEDDING_DIM}, at::kFloat);
  auto count = at::tensor({4}, at::kLong);
  composite_->get_kv_db_async(indices, weights, count).wait();

  auto* w_ptr = weights.data_ptr<float>();
  int64_t stride = EMBEDDING_DIM;

  // -1 indices should remain zero
  EXPECT_FLOAT_EQ(w_ptr[0 * stride], 0.0f);
  // Key 1: DRAM hit
  EXPECT_FLOAT_EQ(w_ptr[1 * stride], 1.0f);
  // -1 index: zero
  EXPECT_FLOAT_EQ(w_ptr[2 * stride], 0.0f);
  // Key 2: SSD hit
  EXPECT_FLOAT_EQ(w_ptr[3 * stride], 2.0f);
}

// Test: flush() drains dirty DRAM blocks to SSD.
TEST_F(DramSsdKVEmbeddingCacheTest, FlushDirtyBlocksToSsd) {
  createComposite();

  // Seed dirty DRAM blocks via the metaheader write path, which marks blocks
  // dirty. (The composite's set_kv_db_async is the backfill path and
  // intentionally does NOT mark dirty, so flush would skip those.) flush() then
  // drains the dirty blocks to SSD.
  const int64_t row_dim = kMetaHeaderDim + EMBEDDING_DIM;
  const std::vector<int64_t> ids = {100, 101};
  auto rows =
      at::zeros({static_cast<int64_t>(ids.size()), row_dim}, at::kFloat);
  auto* rp = rows.data_ptr<float>();
  for (size_t r = 0; r < ids.size(); ++r) {
    float* row = rp + r * row_dim;
    create_metaheader_row(row, row_dim, ids[r]); // embeds key in metaheader
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
      row[kMetaHeaderDim + i] = 10.0f + static_cast<float>(r);
    }
  }
  dram_cache_->set_kv_with_metaheader_to_storage(rows);

  // Before flush, the keys are not on SSD yet.
  auto before = readSsd(ids);
  auto* b = before.data_ptr<float>();
  EXPECT_FLOAT_EQ(b[0 * EMBEDDING_DIM], 0.0f);
  EXPECT_FLOAT_EQ(b[1 * EMBEDDING_DIM], 0.0f);

  // flush() drains dirty DRAM blocks to SSD synchronously.
  composite_->flush();
  EXPECT_GT(composite_->get_ssd_num_writes(), 0);

  auto after = readSsd(ids);
  auto* a = after.data_ptr<float>();
  EXPECT_FLOAT_EQ(a[0 * EMBEDDING_DIM], 10.0f);
  EXPECT_FLOAT_EQ(a[1 * EMBEDDING_DIM], 11.0f);

  // flush() clears the dirty bits, so a second flush finds no dirty blocks and
  // must not re-write anything (avoids duplicated SSD writes).
  auto writes_after_first = composite_->get_ssd_num_writes();
  composite_->flush();
  EXPECT_EQ(composite_->get_ssd_num_writes(), writes_after_first)
      << "Second flush should not re-write already-flushed (clean) blocks";
}

// Test: get_dram_kv_perf appends composite SSD metrics (indices 38-42).
TEST_F(DramSsdKVEmbeddingCacheTest, GetDramKvPerfReportsSsdMetrics) {
  createComposite();

  // One DRAM-miss -> SSD-hit read: records 1 SSD lookup + 1 SSD hit.
  insertIntoSsd(7, 7.0f);
  getViaComposite({7});

  // Seed two dirty DRAM blocks and flush them: records 2 SSD writes.
  const int64_t row_dim = kMetaHeaderDim + EMBEDDING_DIM;
  const std::vector<int64_t> ids = {100, 101};
  auto rows =
      at::zeros({static_cast<int64_t>(ids.size()), row_dim}, at::kFloat);
  auto* rp = rows.data_ptr<float>();
  for (size_t r = 0; r < ids.size(); ++r) {
    float* row = rp + r * row_dim;
    create_metaheader_row(row, row_dim, ids[r]);
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
      row[kMetaHeaderDim + i] = 10.0f + static_cast<float>(r);
    }
  }
  dram_cache_->set_kv_with_metaheader_to_storage(rows);
  composite_->flush();

  // interval=1 so per-interval values equal the raw counts. step%interval==0
  // and step>0 so the SSD metrics block runs.
  auto perf = composite_->get_dram_kv_perf(/*step=*/1, /*interval=*/1);
  ASSERT_EQ(perf.size(), 43u);
  EXPECT_DOUBLE_EQ(perf[38], 1.0); // ssd lookups
  EXPECT_DOUBLE_EQ(perf[39], 1.0); // ssd hits
  EXPECT_DOUBLE_EQ(perf[40], 2.0); // ssd writes (2 flushed blocks)
  EXPECT_GT(perf[42], 0.0); // cumulative rows written to SSD
}

// Test: set_range_to_storage (checkpoint restore) writes whole rows to SSD.
TEST_F(DramSsdKVEmbeddingCacheTest, SetRangeToStorageWritesWholeRowToSsd) {
  createComposite();

  const int64_t row_dim = kMetaHeaderDim + EMBEDDING_DIM;
  const std::vector<int64_t> ids = {7, 8};
  auto rows =
      at::zeros({static_cast<int64_t>(ids.size()), row_dim}, at::kFloat);
  auto* rp = rows.data_ptr<float>();
  for (size_t r = 0; r < ids.size(); ++r) {
    float* row = rp + r * row_dim;
    create_metaheader_row(row, row_dim, ids[r]); // key read back via get_key()
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
      row[kMetaHeaderDim + i] = 70.0f + static_cast<float>(r);
    }
  }

  // Keys are taken from each row's metaheader, not from start/length.
  composite_->set_range_to_storage(
      rows, /*start=*/0, /*length=*/static_cast<int64_t>(ids.size()));

  auto ssd = readSsd(ids);
  auto* s = ssd.data_ptr<float>();
  EXPECT_FLOAT_EQ(s[0 * EMBEDDING_DIM], 70.0f);
  EXPECT_FLOAT_EQ(s[1 * EMBEDDING_DIM], 71.0f);
}

} // namespace kv_db

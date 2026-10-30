/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

#include "fbgemm_gpu/utils/cpu_output_pool.h"

using fbgemm_gpu::CpuOutputPool;
using fbgemm_gpu::pooled_empty_cpu;

namespace {

constexpr size_t kMiB = size_t{1} << 20;

TEST(CpuOutputPoolTest, BucketsRoundUpToQuarterPowerOfTwo) {
  EXPECT_EQ(CpuOutputPool::bucket_bytes(4 * kMiB), 4 * kMiB);
  EXPECT_EQ(CpuOutputPool::bucket_bytes(4 * kMiB + 1), 5 * kMiB);
  EXPECT_EQ(CpuOutputPool::bucket_bytes(7 * kMiB + 1), 8 * kMiB);
  EXPECT_EQ(CpuOutputPool::bucket_bytes(3), 3);
}

TEST(CpuOutputPoolTest, ReleasedBlockIsReusedForSameBucket) {
  CpuOutputPool pool(/*min_bytes=*/kMiB, /*max_cached_bytes=*/64 * kMiB);
  void* first = nullptr;
  {
    const auto t = pooled_empty_cpu({1000, 4500}, at::kByte, pool);
    first = t.data_ptr();
  }
  EXPECT_EQ(pool.cached_bytes(), CpuOutputPool::bucket_bytes(1000 * 4500));

  // A different size in the same bucket gets the same resident block back.
  const auto t = pooled_empty_cpu({1100, 4500}, at::kByte, pool);
  EXPECT_EQ(t.data_ptr(), first);
  EXPECT_EQ(pool.cached_bytes(), 0);
}

TEST(CpuOutputPoolTest, ReleaseBeyondCapFreesInsteadOfCaching) {
  CpuOutputPool pool(/*min_bytes=*/kMiB, /*max_cached_bytes=*/6 * kMiB);
  {
    const auto a = pooled_empty_cpu({4 * kMiB}, at::kByte, pool);
    const auto b = pooled_empty_cpu({4 * kMiB}, at::kByte, pool);
  }
  EXPECT_EQ(pool.cached_bytes(), 4 * kMiB);
}

TEST(CpuOutputPoolTest, SmallOutputsBypassThePool) {
  CpuOutputPool pool(/*min_bytes=*/kMiB, /*max_cached_bytes=*/64 * kMiB);
  {
    const auto t = pooled_empty_cpu({16, 16}, at::kFloat, pool);
  }
  EXPECT_EQ(pool.cached_bytes(), 0);
}

TEST(CpuOutputPoolTest, QuantizedDtypeMatchesAtEmpty) {
  CpuOutputPool pool(/*min_bytes=*/kMiB, /*max_cached_bytes=*/64 * kMiB);
  {
    const auto pooled = pooled_empty_cpu({40000, 72}, at::kQUInt4x2, pool);
    const auto reference = at::empty({40000, 72}, at::kQUInt4x2);
    EXPECT_TRUE(pooled.is_quantized());
    EXPECT_EQ(pooled.scalar_type(), at::kQUInt4x2);
    EXPECT_EQ(pooled.key_set(), reference.key_set());
    EXPECT_EQ(pooled.sizes(), reference.sizes());
    EXPECT_EQ(pooled.strides(), reference.strides());
    EXPECT_EQ(pooled.storage().nbytes(), reference.storage().nbytes());
  }
  // 4-bit elements pack two per byte.
  EXPECT_EQ(pool.cached_bytes(), CpuOutputPool::bucket_bytes(40000 * 72 / 2));
}

TEST(CpuOutputPoolTest, ViewKeepsBlockAliveUntilDropped) {
  CpuOutputPool pool(/*min_bytes=*/kMiB, /*max_cached_bytes=*/64 * kMiB);
  at::Tensor view;
  {
    const auto t = pooled_empty_cpu({1000, 4500}, at::kByte, pool);
    t.fill_(7);
    view = t.split(500, 0)[1];
  }
  EXPECT_EQ(pool.cached_bytes(), 0);
  EXPECT_TRUE(view.eq(7).all().item<bool>());
  view.reset();
  EXPECT_EQ(pool.cached_bytes(), CpuOutputPool::bucket_bytes(1000 * 4500));
}

// Registry of storage addresses currently owned by a live tensor. If the pool
// ever handed one block to two owners, insert() would see a duplicate.
class LiveBlocks {
 public:
  bool claim(const void* p) {
    std::scoped_lock lock(mu_);
    ever_.insert(p);
    return live_.insert(p).second;
  }

  void drop(const void* p) {
    std::scoped_lock lock(mu_);
    live_.erase(p);
  }

  size_t distinct_blocks() const {
    std::scoped_lock lock(mu_);
    return ever_.size();
  }

 private:
  mutable std::mutex mu_;
  std::unordered_set<const void*> live_;
  std::unordered_set<const void*> ever_;
};

struct Owned {
  at::Tensor tensor;
  uint8_t tag = 0;
};

// True iff every byte still holds the owner's tag, i.e. no other owner wrote
// to this block while it was live.
bool intact(const Owned& o) {
  const auto* p = static_cast<const uint8_t*>(o.tensor.data_ptr());
  const auto n = o.tensor.nbytes();
  for (size_t i = 0; i < n; ++i) {
    if (p[i] != o.tag) {
      return false;
    }
  }
  return true;
}

TEST(CpuOutputPoolTest, ConcurrentAllocateAndCrossThreadReleaseIsSafe) {
  constexpr int kThreads = 8;
  constexpr int kItersPerThread = 400;
  constexpr size_t kKiB = size_t{1} << 10;
  constexpr size_t kCap = 32 * kMiB;
  CpuOutputPool pool(/*min_bytes=*/64 * kKiB, /*max_cached_bytes=*/kCap);
  LiveBlocks live;

  // Shared handoff slots: a thread swaps its fresh tensor in and takes the
  // previous occupant, which another thread allocated, and releases it.
  std::vector<Owned> slots(kThreads * 2);
  std::vector<std::mutex> slot_mu(slots.size());
  std::atomic<int> violations{0};

  const auto release = [&](Owned& o) {
    if (!o.tensor.defined()) {
      return;
    }
    if (!intact(o)) {
      violations.fetch_add(1);
    }
    live.drop(o.tensor.data_ptr());
    o.tensor.reset();
  };

  const auto worker = [&](int tid) {
    std::mt19937 rng(tid);
    std::uniform_int_distribution<int> rows(64, 512);
    std::uniform_int_distribution<size_t> slot(0, slots.size() - 1);
    std::vector<Owned> kept;
    for (int i = 0; i < kItersPerThread; ++i) {
      Owned o;
      o.tensor = pooled_empty_cpu({rows(rng), 1024}, at::kByte, pool);
      o.tag = static_cast<uint8_t>((tid * kItersPerThread + i) % 251 + 1);
      if (!live.claim(o.tensor.data_ptr())) {
        violations.fetch_add(1);
      }
      std::memset(o.tensor.data_ptr(), o.tag, o.tensor.nbytes());

      if (i % 3 == 0) {
        kept.push_back(std::move(o));
        if (kept.size() > 4) {
          release(kept.front());
          kept.erase(kept.begin());
        }
      } else {
        const size_t s = slot(rng);
        Owned prev;
        {
          std::scoped_lock lock(slot_mu[s]);
          std::swap(prev, slots[s]);
          slots[s] = std::move(o);
        }
        release(prev);
      }
    }
    for (auto& k : kept) {
      release(k);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& t : threads) {
    t.join();
  }
  for (auto& s : slots) {
    release(s);
  }

  EXPECT_EQ(violations.load(), 0);
  EXPECT_LE(pool.cached_bytes(), kCap);
  EXPECT_GT(pool.cached_bytes(), 0);
  // 3200 allocations were served by far fewer blocks: freed blocks were
  // reused across threads rather than reallocated.
  EXPECT_LT(live.distinct_blocks(), size_t{kThreads} * kItersPerThread / 4);
}

} // namespace

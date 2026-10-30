/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Microbenchmark: pooled CPU TBE output allocation vs plain at::empty.
//
// Every allocation mirrors what the NOBAG CPU TBE does with its output:
// allocate a {rows, 1024} uint8 tensor, write every byte (the gather
// overwrites the whole output), then drop it. Two scenarios:
//
//  steady: each thread loops allocate/write/drop. jemalloc keeps the block it
//          just freed hot in its own cache, so this is at::empty's best case.
//  burst:  each round every thread holds `depth` outputs live at once (a load
//          burst with many requests in flight), writes them, drops them, and
//          then jemalloc returns freed memory to the OS (arena purge), as the
//          predictor's decay settings do between bursts. at::empty then
//          re-faults every page on the next burst; the pool keeps its blocks.
//
// Reports wall time per allocation (alloc + full write + free) and minor page
// faults per allocation.
//
// Usage: cpu_output_pool_bench [--rounds N] [--depth D] [--threads 1,8,32]
//                              [--mb 4,32,128]

#include <ATen/ATen.h>
#include <sys/resource.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "fbgemm_gpu/utils/cpu_output_pool.h"

// Provided by jemalloc when it is the process allocator; null otherwise.
extern "C" int mallctl(const char*, void*, size_t*, void*, size_t)
    __attribute__((weak));

namespace {

constexpr int64_t kRowBytes = 1024;

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(std::stoi(item));
  }
  return out;
}

long minor_faults() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_minflt;
}

bool using_jemalloc() {
  return mallctl != nullptr;
}

void purge_freed_memory() {
  if (using_jemalloc()) {
    mallctl("arena.4096.purge", nullptr, nullptr, nullptr, 0); // all arenas
  }
}

struct Result {
  double us_per_alloc = 0;
  double faults_per_alloc = 0;
};

using AllocFn = at::Tensor (*)(int64_t rows, void* ctx);

// `rounds` rounds; in each, every thread allocates and writes `depth` outputs,
// keeps them all live until every thread is done, then drops them. With
// `purge`, freed memory is returned to the OS between rounds (untimed).
Result run(
    int threads,
    int rounds,
    int depth,
    size_t bytes,
    bool purge,
    AllocFn alloc,
    void* ctx) {
  const int64_t rows = static_cast<int64_t>(bytes) / kRowBytes;
  std::barrier sync(threads + 1);
  std::atomic<long long> busy_ns{0};
  const auto body = [&] {
    std::vector<at::Tensor> live(depth);
    for (int r = 0; r < rounds; ++r) {
      sync.arrive_and_wait(); // round start
      const auto t0 = std::chrono::steady_clock::now();
      for (int d = 0; d < depth; ++d) {
        live[d] = alloc(rows, ctx);
        std::memset(live[d].data_ptr(), (r + d) & 0xff, live[d].nbytes());
      }
      sync.arrive_and_wait(); // all live at once
      for (auto& t : live) {
        t.reset();
      }
      busy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
      sync.arrive_and_wait(); // round end
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(threads);
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back(body);
  }
  long faults = 0;
  for (int r = 0; r < rounds; ++r) {
    if (purge) {
      purge_freed_memory();
    }
    const long f0 = minor_faults();
    sync.arrive_and_wait();
    sync.arrive_and_wait();
    sync.arrive_and_wait();
    faults += minor_faults() - f0;
  }
  for (auto& t : pool) {
    t.join();
  }
  const double allocs = static_cast<double>(threads) * rounds * depth;
  return {busy_ns.load() / 1e3 / allocs, faults / allocs};
}

at::Tensor plain_alloc(int64_t rows, void* /*ctx*/) {
  return at::empty({rows, kRowBytes}, at::kByte);
}

at::Tensor pooled_alloc(int64_t rows, void* ctx) {
  return fbgemm_gpu::pooled_empty_cpu(
      {rows, kRowBytes},
      at::kByte,
      *static_cast<fbgemm_gpu::CpuOutputPool*>(ctx));
}

} // namespace

int main(int argc, char** argv) {
  int rounds = 20;
  int depth = 4;
  std::vector<int> thread_counts = {1, 8, 32};
  std::vector<int> sizes_mb = {4, 32, 128};
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    if (flag == "--rounds") {
      rounds = std::stoi(argv[i + 1]);
    } else if (flag == "--depth") {
      depth = std::stoi(argv[i + 1]);
    } else if (flag == "--threads") {
      thread_counts = parse_list(argv[i + 1]);
    } else if (flag == "--mb") {
      sizes_mb = parse_list(argv[i + 1]);
    }
  }
  std::printf(
      "jemalloc=%d rounds=%d depth=%d\n",
      using_jemalloc() ? 1 : 0,
      rounds,
      depth);
  std::printf(
      "%-7s %7s %7s %12s %12s %12s %12s %8s\n",
      "mode",
      "size_MB",
      "threads",
      "empty_us",
      "pool_us",
      "empty_flt",
      "pool_flt",
      "speedup");
  for (const bool burst : {false, true}) {
    for (const int mb : sizes_mb) {
      const size_t bytes = static_cast<size_t>(mb) << 20;
      for (const int threads : thread_counts) {
        const int d = burst ? depth : 1;
        const auto plain =
            run(threads, rounds, d, bytes, burst, &plain_alloc, nullptr);
        // Pool sized to hold the whole burst, as the default cap does for the
        // predictor's working set; warmed once like a long-running process.
        fbgemm_gpu::CpuOutputPool pool(
            /*min_bytes=*/size_t{1} << 20,
            /*max_cached_bytes=*/
            fbgemm_gpu::CpuOutputPool::bucket_bytes(bytes) * threads * d);
        run(threads, 1, d, bytes, false, &pooled_alloc, &pool);
        const auto pooled =
            run(threads, rounds, d, bytes, burst, &pooled_alloc, &pool);
        std::printf(
            "%-7s %7d %7d %12.1f %12.1f %12.1f %12.1f %7.2fx\n",
            burst ? "burst" : "steady",
            mb,
            threads,
            plain.us_per_alloc,
            pooled.us_per_alloc,
            plain.faults_per_alloc,
            pooled.faults_per_alloc,
            plain.us_per_alloc / pooled.us_per_alloc);
      }
    }
  }
  std::printf(
      "us = wall time per allocation (alloc + full write + free) per thread; "
      "flt = minor page faults per allocation (4 KiB pages: 4 MB = 1024)\n");
  return 0;
}

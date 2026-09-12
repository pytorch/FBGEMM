/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "src/JitPerfMap.h" // @manual

#ifndef _MSC_VER
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>
#endif

using namespace fbgemm;

// JitSymbolBuilder decides the whole on-disk symbol format, and symbols are
// only built when FBGEMM_JIT_PERF_MAP is set, so a formatting regression would
// otherwise surface only in a profile.
TEST(JitPerfMapTest, SymbolBuilderFormat) {
  EXPECT_EQ(
      JitSymbolBuilder("gemm")
          .field("MC", 16)
          .field("isa", "avx2")
          .flag("accum", true)
          .flag("trans", false)
          .str(),
      "fbgemm::gemm_MC-16_isa-avx2_accum-1_trans-0");
  // A kernel with no fields is still namespace-qualified.
  EXPECT_EQ(JitSymbolBuilder("bare").str(), "fbgemm::bare");
}

// Every inst_set_t has to name itself. instSetName() ends in an "anyarch"
// fallback, so an enumerator added without a branch there would silently share
// a name with anyarch, and kernels built for it would read in a profile as the
// wrong instruction set. Distinctness is what catches that.
TEST(JitPerfMapTest, EveryInstSetHasADistinctName) {
  // The list below is hand-written because instSetName() is templated, so this
  // switch is what keeps it honest: it has no default, so adding an
  // inst_set_t enumerator makes it non-exhaustive and -Wswitch fails the
  // build until both it and the list are updated.
  const auto exhaustive = [](inst_set_t set) {
    switch (set) {
      case inst_set_t::anyarch:
      case inst_set_t::avx2:
      case inst_set_t::avx512:
      case inst_set_t::avx512_ymm:
      case inst_set_t::avx512_vnni:
      case inst_set_t::avx512_vnni_ymm:
      case inst_set_t::sve:
        return true;
    }
    return false;
  };
  EXPECT_TRUE(exhaustive(inst_set_t::anyarch));

  const std::vector<std::string> names = {
      instSetName<inst_set_t::anyarch>(),
      instSetName<inst_set_t::avx2>(),
      instSetName<inst_set_t::avx512>(),
      instSetName<inst_set_t::avx512_ymm>(),
      instSetName<inst_set_t::avx512_vnni>(),
      instSetName<inst_set_t::avx512_vnni_ymm>(),
      instSetName<inst_set_t::sve>(),
  };
  for (const auto& name : names) {
    EXPECT_FALSE(name.empty());
  }
  const std::set<std::string> unique(names.begin(), names.end());
  EXPECT_EQ(unique.size(), names.size())
      << "two inst_set_t values share a perf-map name";
}

#ifdef _MSC_VER

TEST(JitPerfMapTest, PosixOnly) {
  GTEST_SKIP() << "perf maps are a Linux profiling convention";
}

#else

namespace {

// A plausible code address and size; the map only records them, so any
// non-null address works.
const void* const kAddr = reinterpret_cast<const void*>(0x400000);
constexpr size_t kSize = 0x40;
constexpr const char* kSymbol = "fbgemm::test_kernel";

std::string mapPathFor(pid_t pid) {
  return "/tmp/perf-" + std::to_string(pid) + ".map";
}

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

// Runs `body` in a forked child and returns the child's pid once it has
// exited. Each scenario needs its own process: the descriptor is opened once
// per pid, so a single process can only exercise the open path once.
pid_t runInChild(void (*body)()) {
  const pid_t pid = fork();
  if (pid == 0) {
    body();
    _exit(0);
  }
  EXPECT_GT(pid, 0);
  int status = 0;
  EXPECT_EQ(waitpid(pid, &status, 0), pid);
  return pid;
}

void enablePerfMap() {
  // Must happen before the first registration in this process; the flag is
  // read once and cached.
  setenv("FBGEMM_JIT_PERF_MAP", "1", 1);
}

} // namespace

// A map left behind by a dead process that happened to hold this pid must not
// be appended to: its address ranges describe code that no longer exists, so a
// profiler would attribute our samples to its symbols.
TEST(JitPerfMapTest, StaleMapFromRecycledPidIsDiscarded) {
  enablePerfMap();
  const pid_t child = runInChild([] {
    const std::string path = mapPathFor(getpid());
    {
      std::ofstream out(path);
      out << "deadbeef 10 stale::kernel_from_previous_process\n";
    }
    // Backdate it an hour so it clearly predates this process.
    struct timeval times[2];
    gettimeofday(&times[0], nullptr);
    times[0].tv_sec -= 3600;
    times[1] = times[0];
    utimes(path.c_str(), times);

    registerJitCodeForProfiling(kAddr, kSize, kSymbol);
  });

  const std::string path = mapPathFor(child);
  const std::string contents = readFile(path);
  EXPECT_EQ(
      contents.find("stale::kernel_from_previous_process"), std::string::npos)
      << "stale content survived:\n"
      << contents;
  EXPECT_NE(contents.find(kSymbol), std::string::npos)
      << "own record missing:\n"
      << contents;
  unlink(path.c_str());
}

// Discarding a stale map and writing records race each other: if a thread is
// handed an O_APPEND descriptor before another finishes truncating, its record
// is thrown away and the kernel reads as [unknown] in the profile.
TEST(JitPerfMapTest, StaleMapTruncationDoesNotDropConcurrentRecords) {
  enablePerfMap();
  constexpr int kThreads = 32;

  const pid_t child = runInChild([] {
    const std::string path = mapPathFor(getpid());
    {
      std::ofstream out(path);
      out << "deadbeef 10 stale::kernel_from_previous_process\n";
    }
    struct timeval times[2];
    gettimeofday(&times[0], nullptr);
    times[0].tv_sec -= 3600;
    times[1] = times[0];
    utimes(path.c_str(), times);

    // Released together so several threads reach their first registration --
    // and therefore the open that truncates -- at the same moment.
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&ready, &go] {
        ready.fetch_add(1, std::memory_order_relaxed);
        while (!go.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        registerJitCodeForProfiling(kAddr, kSize, "fbgemm::race_kernel");
      });
    }
    while (ready.load(std::memory_order_relaxed) < kThreads) {
      std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    for (auto& thread : threads) {
      thread.join();
    }
  });

  const std::string path = mapPathFor(child);
  const std::string contents = readFile(path);
  int records = 0;
  for (size_t i = contents.find("fbgemm::race_kernel"); i != std::string::npos;
       i = contents.find("fbgemm::race_kernel", i + 1)) {
    ++records;
  }
  EXPECT_EQ(records, kThreads)
      << "records were discarded by the stale-map truncation";
  EXPECT_EQ(
      contents.find("stale::kernel_from_previous_process"), std::string::npos)
      << "stale content survived";
  unlink(path.c_str());
}

// Content written during this process's lifetime belongs to another JIT writer
// sharing the map, and must survive.
TEST(JitPerfMapTest, ContentFromThisProcessIsPreserved) {
  enablePerfMap();
  const pid_t child = runInChild([] {
    const std::string path = mapPathFor(getpid());
    std::ofstream out(path);
    out << "cafe1000 20 other_jit::kernel\n";
    out.close();

    registerJitCodeForProfiling(kAddr, kSize, kSymbol);
  });

  const std::string path = mapPathFor(child);
  const std::string contents = readFile(path);
  EXPECT_NE(contents.find("other_jit::kernel"), std::string::npos)
      << "co-writer's record was discarded:\n"
      << contents;
  EXPECT_NE(contents.find(kSymbol), std::string::npos)
      << "own record missing:\n"
      << contents;
  unlink(path.c_str());
}

// The path is predictable, so a symlink planted there must not redirect the
// write into someone else's file.
TEST(JitPerfMapTest, SymlinkAtMapPathIsRefused) {
  enablePerfMap();
  const std::string victim = "/tmp/fbgemm_perf_map_victim.txt";
  unlink(victim.c_str());
  {
    std::ofstream create(victim);
  }

  const pid_t child = runInChild([] {
    const std::string path = mapPathFor(getpid());
    unlink(path.c_str());
    symlink("/tmp/fbgemm_perf_map_victim.txt", path.c_str());
    registerJitCodeForProfiling(kAddr, kSize, kSymbol);
  });

  EXPECT_EQ(readFile(victim), "") << "write followed the symlink";
  unlink(victim.c_str());
  unlink(mapPathFor(child).c_str());
}

// Nothing serialises writers in userspace, so the guarantee that a record
// arrives whole comes from handing each one to the kernel as a single
// O_APPEND write. Concurrent writers must not tear or interleave records.
TEST(JitPerfMapTest, ConcurrentWritesProduceWellFormedRecords) {
  enablePerfMap();
  constexpr int kThreads = 16;
  constexpr int kPerThread = 200;

  const pid_t child = runInChild([] {
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([t] {
        for (int i = 0; i < kPerThread; ++i) {
          registerJitCodeForProfiling(
              kAddr, kSize, "fbgemm::thread" + std::to_string(t));
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  });

  const std::string path = mapPathFor(child);
  std::ifstream in(path);
  std::string line;
  int lines = 0;
  int malformed = 0;
  while (std::getline(in, line)) {
    ++lines;
    unsigned long long addr = 0;
    unsigned long long size = 0;
    char symbol[128] = {};
    // Exactly three fields, and the symbol must be one of the ones written.
    if (sscanf(line.c_str(), "%llx %llx %127s", &addr, &size, symbol) != 3 ||
        !std::string(symbol).starts_with("fbgemm::thread")) {
      ++malformed;
    }
  }
  EXPECT_EQ(lines, kThreads * kPerThread);
  EXPECT_EQ(malformed, 0) << malformed << " of " << lines
                          << " records were torn or interleaved";
  unlink(path.c_str());
}

// fork() clones only the calling thread. A lock held by any other thread at
// that instant would be inherited permanently locked, and the child would hang
// on its first registration -- the PyTorch DataLoader pattern.
TEST(JitPerfMapTest, ForkDuringConcurrentRegistrationDoesNotDeadlock) {
  enablePerfMap();
  constexpr int kForks = 100;
  // A healthy child exits in about a millisecond. The loop stops at the first
  // hang, so a regression fails in seconds instead of kForks * kTimeoutMs.
  constexpr int kTimeoutMs = 2000;

  std::atomic<bool> stop{false};
  std::thread writer([&stop] {
    while (!stop.load(std::memory_order_relaxed)) {
      registerJitCodeForProfiling(kAddr, kSize, "fbgemm::concurrent_kernel");
    }
  });

  int hung = 0;
  for (int i = 0; i < kForks; ++i) {
    const pid_t pid = fork();
    if (pid == 0) {
      registerJitCodeForProfiling(kAddr, kSize, kSymbol);
      _exit(0);
    }
    ASSERT_GT(pid, 0);

    bool exited = false;
    for (int waited = 0; waited < kTimeoutMs; ++waited) {
      int status = 0;
      if (waitpid(pid, &status, WNOHANG) == pid) {
        exited = true;
        break;
      }
      usleep(1000);
    }
    if (!exited) {
      ++hung;
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
    }
    unlink(mapPathFor(pid).c_str());
    if (hung > 0) {
      break;
    }
  }

  stop.store(true, std::memory_order_relaxed);
  writer.join();
  unlink(mapPathFor(getpid()).c_str());

  EXPECT_EQ(hung, 0)
      << "a child deadlocked on its first registration after forking while "
         "another thread was registering";
}

#endif // _MSC_VER

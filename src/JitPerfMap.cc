/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS

#include "./JitPerfMap.h" // @manual

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string_view>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _MSC_VER
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fbgemm {

namespace {

// This file deliberately takes no userspace lock. fbgemm JITs from worker
// threads, and a multithreaded process may fork at any point -- PyTorch's
// DataLoader does exactly that. fork() clones only the calling thread, so a
// mutex held by any other thread at that moment is inherited permanently
// locked and the child deadlocks on its first registration. Instead the
// descriptor is published through atomics and each record is handed to the
// kernel as one O_APPEND write(), which is what serialises concurrent writers.
// The one place threads do wait on each other -- electing which of them opens
// the map -- is keyed on the pid, so a child never waits on a claim its parent
// still holds.

int currentPid() {
#ifdef _MSC_VER
  return _getpid();
#else
  return static_cast<int>(getpid());
#endif
}

#ifndef _MSC_VER
// Seconds since the epoch at which this process started, or 0 when that cannot
// be determined. On Linux the mtime of /proc/self is exactly the process start
// time, which is what distinguishes a map this process wrote from one left by
// an earlier process that happened to hold the same pid.
//
// POSIX-only: the Windows path below has no staleness check, so defining this
// unconditionally leaves an unused static function that -Werror rejects.
std::time_t processStartTime() {
#ifdef __linux__
  struct stat st = {};
  if (::stat("/proc/self", &st) == 0) {
    return st.st_mtime;
  }
#endif
  return 0;
}
#endif // _MSC_VER

bool writeAll(int fd, const char* data, size_t len) {
  while (len > 0) {
#ifdef _MSC_VER
    const int written = _write(fd, data, static_cast<unsigned int>(len));
#else
    const ssize_t written = ::write(fd, data, len);
#endif
    if (written <= 0) {
      if (written < 0 && errno == EINTR) {
        continue;
      }
      return false;
    }
    data += written;
    len -= static_cast<size_t>(written);
  }
  return true;
}

// -1 before the first open. Written only by openPerfMap().
std::atomic<int> perfMapFd{-1};
std::atomic<int> perfMapPid{-1};
// Claimed by the single thread that opens the map for a given pid, so no
// descriptor is published while that thread may still be truncating.
std::atomic<int> perfMapOpeningPid{-1};

int openPerfMap(int pid) {
  char path[64];
  std::snprintf(path, sizeof(path), "/tmp/perf-%d.map", pid);

#ifdef _MSC_VER
  const int fd =
      _open(path, _O_WRONLY | _O_CREAT | _O_APPEND, _S_IREAD | _S_IWRITE);
  if (fd < 0) {
    return -1;
  }
#else
  // The path is predictable, so anyone able to create files in /tmp could
  // pre-place a symlink and capture what a more privileged process writes.
  // O_NOFOLLOW rejects a symlink at the final component; the fstat() below
  // then rejects anything that is not a regular file we own and that only we
  // can write, which covers a pre-created file or FIFO that O_NOFOLLOW allows.
  // O_CLOEXEC because the map belongs to this process and a child that execs
  // should not inherit the descriptor.
  const int fd = ::open(
      path,
      O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW | O_CLOEXEC,
      S_IRUSR | S_IWUSR);
  if (fd < 0) {
    return -1;
  }
  struct stat st = {};
  if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) ||
      st.st_uid != ::geteuid() || (st.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    ::close(fd);
    return -1;
  }

  // Pids are recycled and /tmp entries outlive the process that made them, so
  // an existing map may describe a dead process's address ranges. Appending to
  // it would attribute our samples to its symbols. Content older than this
  // process is therefore discarded; anything written since we started belongs
  // to another JIT writer in this same process and is left alone.
  const std::time_t startTime = processStartTime();
  if (st.st_size > 0 && startTime != 0 && st.st_mtime < startTime) {
    if (::ftruncate(fd, 0) != 0) {
      ::close(fd);
      return -1;
    }
  }
#endif
  return fd;
}

// The descriptor for this process's map, or -1 if it is unavailable.
//
// Reopened when the pid changes: a fork inherits both the descriptor and the
// parent's file name, so a child that JITs would otherwise append its symbols
// to /tmp/perf-<parent>.map and leave its own map empty. O_CLOEXEC covers
// exec but not fork, so the pid is rechecked on every record.
int perfMapDescriptor() {
  const int pid = currentPid();
  if (perfMapPid.load(std::memory_order_acquire) == pid) {
    // May be -1: a failed open is remembered rather than retried on every
    // kernel, so a hostile or unwritable /tmp costs one syscall, not one per
    // registration.
    return perfMapFd.load(std::memory_order_acquire);
  }

  // Exactly one thread opens. A stale map is discarded with ftruncate(), and
  // if a second thread had already been handed an O_APPEND descriptor it could
  // write a record that the truncation then threw away. Electing an opener
  // means no descriptor exists until truncation is done.
  int opening = perfMapOpeningPid.load(std::memory_order_acquire);
  if (opening != pid &&
      perfMapOpeningPid.compare_exchange_strong(
          opening, pid, std::memory_order_acq_rel)) {
    const int fd = openPerfMap(pid);
    perfMapFd.store(fd, std::memory_order_release);
    perfMapPid.store(pid, std::memory_order_release);
    // Any descriptor installed by a previous pid is deliberately not closed. A
    // concurrent writer may still be inside write() on it, and closing would
    // let the number be recycled under them. It is at most one descriptor per
    // fork that JITs, and O_CLOEXEC plus process exit reclaim it.
    return fd;
  }

  // Another thread is opening. Wait for it to publish rather than opening a
  // second descriptor onto the same file. This cannot be inherited stuck
  // across a fork: the child's pid differs, so it wins the election above
  // instead of waiting on a claim its parent still holds.
  while (perfMapPid.load(std::memory_order_acquire) != pid) {
    if (currentPid() != pid) {
      return -1;
    }
    std::this_thread::yield();
  }
  return perfMapFd.load(std::memory_order_acquire);
}

} // namespace

bool jitPerfMapEnabled() {
  // Cached in a constant-initialised atomic rather than a function-local
  // static: magic-static initialisation takes a guard lock, which is exactly
  // the kind of inherited lock a forked child can deadlock on. Racing threads
  // may both compute it, which is harmless -- they compute the same value.
  static std::atomic<int> cached{-1};
  int enabled = cached.load(std::memory_order_acquire);
  if (enabled < 0) {
    const char* value = std::getenv("FBGEMM_JIT_PERF_MAP");
    enabled =
        (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0)
        ? 1
        : 0;
    cached.store(enabled, std::memory_order_release);
  }
  return enabled != 0;
}

void registerJitCodeForProfiling(
    const void* addr,
    size_t size,
    const std::string& name) {
  if (!jitPerfMapEnabled() || addr == nullptr || size == 0) {
    return;
  }

  const int fd = perfMapDescriptor();
  if (fd < 0) {
    return;
  }

  char header[64];
  const int headerLen = std::snprintf(
      header,
      sizeof(header),
      "%llx %llx ",
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(addr)),
      static_cast<unsigned long long>(size));
  if (headerLen <= 0) {
    return;
  }

  // Assembled first and written once: an O_APPEND write is what keeps records
  // from other threads whole, and a record split across two writes could
  // interleave with theirs.
  std::string record;
  record.reserve(static_cast<size_t>(headerLen) + name.size() + 1);
  record.assign(header, static_cast<size_t>(headerLen));
  record += name;
  record += '\n';
  writeAll(fd, record.data(), record.size());
}

std::string embeddingKernelSymbol(
    std::string_view kernel,
    std::string_view shape,
    const EmbeddingKernelOptions& options) {
  std::string sym;
  sym.reserve(96);
  sym += "fbgemm::";
  sym += kernel;
  sym += "_";
  sym += shape;
  const auto append = [&sym](bool on, std::string_view flag) {
    if (on) {
      sym += "_";
      sym += flag;
    }
  };
  append(true, options.indices64 ? "idx64" : "idx32");
  append(true, options.offsets64 ? "off64" : "off32");
  append(options.thread_local_cache, "tls");
  append(true, options.avx512 ? "avx512" : "avx2");
  if (options.prefetch != 0) {
    sym += "_prefetch-";
    sym += std::to_string(options.prefetch);
  }
  append(options.weighted, "weighted");
  append(options.positional, "positional");
  append(options.normalize, "normalize");
  append(options.lengths, "lengths");
  append(options.rowwise_sparse, "rowwise_sparse");
  append(options.scale_bias_first, "scale_bias_first");
  append(options.fp16_out, "fp16_out");
  append(options.bf16_out, "bf16_out");
  sym += "_ostride-";
  sym += std::to_string(options.output_stride);
  sym += "_istride-";
  sym += std::to_string(options.input_stride);
  return sym;
}

} // namespace fbgemm

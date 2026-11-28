/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_inference_wrapper.h"
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <fb303/ServiceData.h>
#include <fmt/format.h>
#include <folly/container/F14Map.h>
#include <gflags/gflags.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include "deeplearning/fbgemm/fbgemm_gpu/include/fbgemm_gpu/embedding_common.h" // @manual=//deeplearning/fbgemm/fbgemm_gpu:fbgemm_gpu
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_inference_embedding.h"

DEFINE_int64(
    dram_kv_embedding_num_shards,
    32,
    "Number of shards for DRAM KV inference embedding");
DEFINE_bool(
    kv_embedding_async_get_set,
    true,
    "Whether to use async get/set for DRAM KV inference embedding."
    "This should be true for dram but might be different for other non-Dram backends.");
DEFINE_int32(
    kv_embedding_table_log_lookup_stats_interval_secs,
    0,
    "Periodic per-backend KV lookup statistics in INFO logs. Zero disables logging.");

namespace fbgemm_gpu {
namespace {

// Exports coverage as fb303 sums, keyed by device.
void export_kv_lookup_coverage(
    const c10::Device& device,
    const KVLookupCoverage& coverage) {
  const auto add = [&](std::string_view name, int64_t value) {
    if (value != 0) {
      facebook::fb303::fbData->addStatValue(
          fmt::format("fbgemm.kv_lookup.{}.{}", device.str(), name),
          value,
          facebook::fb303::SUM);
    }
  };
  add("keys", coverage.keys);
  add("zero_rows", coverage.zero_rows);
  add("backend_keys", coverage.backend_keys);
  add("hbm_unique_keys", coverage.hbm_unique_keys);
  add("hbm_unique_hits", coverage.hbm_unique_hits);
}

std::atomic<int64_t>& coverage_sample_every() {
  static std::atomic<int64_t> every{16};
  return every;
}

at::Tensor zero_row_count(const at::Tensor& rows, int64_t row_bytes) {
  return rows.narrow(1, 0, row_bytes).ne(0).any(1).logical_not().sum();
}

// Accumulates [zero rows, backend keys] on one device and exports deltas
// without blocking a lookup on the device.
class DeviceCoverage {
 public:
  explicit DeviceCoverage(const c10::Device& device)
      : device_{device},
        totals_{at::zeros({2}, at::TensorOptions(device).dtype(at::kLong))},
        positions_{at::arange(2, at::TensorOptions(device).dtype(at::kLong))},
        host_{at::zeros(
            {2},
            at::TensorOptions().dtype(at::kLong).pinned_memory(true))},
        copied_{device.type()} {
    // Lookups on other streams add once it is registered; the blocking copy
    // waits for the initialization queued on this stream.
    positions_.to(at::kCPU);
  }

  void flush() {
    // Adds may still be pending on other streams of this device.
    c10::impl::VirtualGuardImpl(device_.type())
        .synchronizeDevice(device_.index());
    std::scoped_lock lock(mutex_);
    for (int i = 0; i < 2; ++i) {
      if (copy_pending_) {
        copied_.synchronize();
      }
      last_copy_ = {};
      drain_locked();
    }
  }

  void add(const at::Tensor& counts) {
    // Atomic, since lookups on other streams add concurrently.
    totals_.index_add_(0, positions_, counts);
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (lock.owns_lock()) {
      drain_locked();
    }
  }

 private:
  void drain_locked() {
    if (copy_pending_) {
      if (!copied_.query()) {
        return;
      }
      copy_pending_ = false;
      const auto* totals = host_.const_data_ptr<int64_t>();
      export_kv_lookup_coverage(
          device_,
          {.zero_rows = totals[0] - reported_[0],
           .backend_keys = totals[1] - reported_[1]});
      reported_ = {totals[0], totals[1]};
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_copy_ < std::chrono::seconds(1)) {
      return;
    }
    last_copy_ = now;
    host_.copy_(totals_, /*non_blocking=*/true);
    copied_.record(
        c10::impl::VirtualGuardImpl(device_.type()).getStream(device_));
    copy_pending_ = true;
  }

  const c10::Device device_;
  const at::Tensor totals_;
  const at::Tensor positions_;
  const at::Tensor host_;
  std::mutex mutex_;
  c10::Event copied_;
  bool copy_pending_ = false;
  std::array<int64_t, 2> reported_{};
  std::chrono::steady_clock::time_point last_copy_;
};

// Leaked so totals outlive device teardown at exit.
std::mutex& device_coverages_mutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

folly::F14FastMap<c10::Device, DeviceCoverage*>& device_coverages() {
  static auto* coverages =
      new folly::F14FastMap<c10::Device, DeviceCoverage*>();
  return *coverages;
}

DeviceCoverage& device_coverage(const c10::Device& device) {
  std::scoped_lock lock(device_coverages_mutex());
  auto& coverage = device_coverages()[device];
  if (coverage == nullptr) {
    coverage = new DeviceCoverage(device);
  }
  return *coverage;
}

using SerializedSpec = DramKVEmbeddingInferenceWrapper::SerializedSepcType;

std::pair<int64_t, int64_t> legacy_row_sizes(
    const std::vector<SerializedSpec>& specs,
    const int64_t row_alignment,
    const int64_t scale_bias_size_in_bytes) {
  int64_t max_dim = 0;
  for (const auto& spec : specs) {
    max_dim = std::max(max_dim, std::get<1>(spec));
  }
  const auto row_bytes = nbit::padded_row_size_in_bytes(
      static_cast<int32_t>(max_dim),
      static_cast<fbgemm_gpu::SparseType>(std::get<2>(specs.front())),
      static_cast<int32_t>(row_alignment),
      static_cast<int32_t>(scale_bias_size_in_bytes));
  return {row_bytes, row_bytes};
}

std::pair<int64_t, int64_t> separate_row_sizes(
    const std::vector<SerializedSpec>& specs,
    const int64_t logical_row_alignment,
    const int64_t storage_row_alignment,
    const int64_t scale_bias_size_in_bytes) {
  std::optional<int64_t> expected_storage_row_bytes;
  std::optional<int64_t> expected_lookup_row_bytes;
  for (const auto& spec : specs) {
    const auto dim = std::get<1>(spec);
    const auto sparse_type =
        static_cast<fbgemm_gpu::SparseType>(std::get<2>(spec));
    const auto storage_row_bytes = nbit::padded_row_size_in_bytes(
        static_cast<int32_t>(dim),
        sparse_type,
        static_cast<int32_t>(storage_row_alignment),
        static_cast<int32_t>(scale_bias_size_in_bytes));
    const auto lookup_row_bytes = nbit::padded_row_size_in_bytes(
        static_cast<int32_t>(dim),
        sparse_type,
        static_cast<int32_t>(logical_row_alignment),
        static_cast<int32_t>(scale_bias_size_in_bytes));
    TORCH_CHECK(
        lookup_row_bytes <= storage_row_bytes,
        "Dram KV logical row width must not exceed storage row width");
    if (expected_lookup_row_bytes.has_value()) {
      TORCH_CHECK(
          lookup_row_bytes == expected_lookup_row_bytes.value(),
          "Dram KV embedding requires one logical row width across tables");
      TORCH_CHECK(
          storage_row_bytes == expected_storage_row_bytes.value(),
          "Dram KV embedding requires one storage row width across tables");
    }
    expected_lookup_row_bytes = lookup_row_bytes;
    expected_storage_row_bytes = storage_row_bytes;
  }
  return {
      expected_storage_row_bytes.value(), expected_lookup_row_bytes.value()};
}

constexpr int64_t kCudaRowAlignment = 16;
constexpr int64_t kHbmCacheAssoc = 32;
// Keys probed per step by cache maintenance (invalidation), so its device
// scratch, about 320 bytes per key, stays bounded however many keys a write
// or a snapshot names.
constexpr int64_t kHbmCacheProbeChunkKeys = 1 << 16;
// Batches of at most this many keys take the host admission path instead of
// the fused kernels.
constexpr int64_t kHbmCacheMinFusedAdmissionKeys = 32;

// How often lookups retry HBM caches disabled after a failed cache update;
// zero keeps them disabled.
std::atomic<int64_t>& hbm_cache_recover_interval_ms() {
  static std::atomic<int64_t> interval{60000};
  return interval;
}

// Whether publishing a served snapshot keeps lookups on the HBM caches
// instead of bypassing them until the snapshot's guard refreshes them.
std::atomic<bool>& hbm_cache_serve_during_publish() {
  static std::atomic<bool> enabled{true};
  return enabled;
}

int64_t round_up(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

} // namespace

namespace {

void export_disabled_hbm_backends(int64_t disabled_backends) {
  facebook::fb303::fbData->setCounter(
      "fbgemm.kv_hbm_cache.disabled_backends", disabled_backends);
}

// Unlike the gauge, which the next recovery resets, a sum shows every disable.
void export_hbm_cache_disable_event() {
  facebook::fb303::fbData->addStatValue(
      "fbgemm.kv_hbm_cache.disable_events", 1, facebook::fb303::SUM);
}

std::atomic<int64_t>& disabled_hbm_backends() {
  static std::atomic<int64_t> count{0};
  return count;
}

void report_disabled_hbm_backends(int64_t delta) {
  // Serializes the update with its report so the gauge ends at the latest
  // count when backends change state concurrently.
  static std::mutex mutex;
  std::scoped_lock lock(mutex);
  export_disabled_hbm_backends(
      disabled_hbm_backends().fetch_add(delta) + delta);
}

} // namespace

class KVBackendMutationState {
 public:
  ~KVBackendMutationState() {
    if (gauge_counted_) {
      report_disabled_hbm_backends(-1);
    }
  }

  void mark_has_hbm_caches() {
    std::scoped_lock lock(mutex_);
    has_hbm_caches.store(true, std::memory_order_release);
    update_gauge_locked();
  }

  std::optional<int64_t> read_generation() const {
    std::scoped_lock lock(mutex_);
    return writers_ || disabled_ || snapshot_failed_
        ? std::nullopt
        : std::make_optional(generation_);
  }

  bool can_admit(int64_t generation) const {
    std::scoped_lock lock(mutex_);
    return !writers_ && !disabled_ && !snapshot_failed_ && !publish_serving_ &&
        generation_ == generation;
  }

  void begin() {
    std::scoped_lock lock(mutex_);
    advance_locked();
    ++writers_;
  }

  // Rejects admissions of reads from before now without bypassing lookups.
  void advance() {
    std::scoped_lock lock(mutex_);
    advance_locked();
  }

  void end() {
    std::scoped_lock lock(mutex_);
    advance_locked();
    --writers_;
  }

  // A served snapshot keeps lookups on the caches while it is staged.
  int64_t begin_snapshot(bool served) {
    std::scoped_lock lock(mutex_);
    advance_locked();
    ++(served ? served_snapshots_ : writers_);
    return generation_;
  }

  void end_snapshot(int64_t begin_generation, bool completed, bool served) {
    std::scoped_lock lock(mutex_);
    advance_locked();
    --(served ? served_snapshots_ : writers_);
    // The guard of a served snapshot refreshes the caches after this, and the
    // publish bypass must outlast every overlapping refresh.
    refreshing_snapshots_ += served;
    if (!completed) {
      snapshot_failed_ = true;
      last_snapshot_failure_generation_ = generation_;
      // The guard clears every cache after this, and no snapshot still stages
      // to hide these keys' rows from later lookups.
      if (served_snapshots_ == 0) {
        clear_staged_writes_locked();
      }
    } else if (last_snapshot_failure_generation_ <= begin_generation) {
      snapshot_failed_ = false;
    }
    update_gauge_locked();
  }

  // Publishing a served snapshot changes rows in place: lookups bypass the
  // caches until the last served snapshot's guard has refreshed them. With
  // serve_cached, values never change per id, so lookups keep their cached
  // rows and only admissions stop; the caller first invalidates the keys of
  // delta rows the publish reveals. Returns whether this call began it.
  bool begin_publish(bool serve_cached = false) {
    std::scoped_lock lock(mutex_);
    if (served_snapshots_ == 0 || publishing_) {
      return false;
    }
    advance_locked();
    publishing_ = true;
    publish_serving_ = serve_cached;
    if (!serve_cached) {
      ++writers_;
    }
    return true;
  }

  // Called once by every snapshot guard; served is the guard's.
  void end_publish(bool served) {
    std::scoped_lock lock(mutex_);
    refreshing_snapshots_ -= served;
    if (publishing_ && served_snapshots_ == 0 && refreshing_snapshots_ == 0) {
      advance_locked();
      if (!publish_serving_) {
        --writers_;
      }
      publishing_ = false;
      publish_serving_ = false;
    }
  }

  // Rows written with a snapshot timestamp while one is served are staged and
  // invisible to lookups until it publishes.
  bool serving_snapshot() const {
    std::scoped_lock lock(mutex_);
    return served_snapshots_ > 0;
  }

  // Keys of delta rows written while a served snapshot stages. The backend
  // shows them only when the snapshot publishes or rolls back, and lookups in
  // between cache the keys' previous rows again.
  // Returns the clear count the keys were recorded under, if they were.
  std::optional<int64_t> record_staged_writes(const at::Tensor& keys) {
    std::scoped_lock lock(mutex_);
    if (served_snapshots_ == 0) {
      return std::nullopt;
    }
    staged_writes_.push_back(
        keys.to(at::kCPU, at::kLong).reshape({-1}).clone());
    return staged_writes_clears_;
  }

  // Records the keys again after their write if a served snapshot is staging
  // that the first record does not cover: none was staging then, or the keys
  // it recorded were taken since.
  void record_staged_writes_after_write(
      const at::Tensor& keys,
      std::optional<int64_t> first_record) {
    std::scoped_lock lock(mutex_);
    if (served_snapshots_ > 0 &&
        (!first_record || *first_record != staged_writes_clears_)) {
      staged_writes_.push_back(
          keys.to(at::kCPU, at::kLong).reshape({-1}).clone());
    }
  }

  // Keys stay recorded while another served snapshot still stages and hides
  // their rows, so that snapshot's end invalidates them again.
  std::optional<at::Tensor> take_staged_writes() {
    std::scoped_lock lock(mutex_);
    if (staged_writes_.empty()) {
      return std::nullopt;
    }
    auto keys = at::cat(staged_writes_);
    if (served_snapshots_ == 0) {
      clear_staged_writes_locked();
    }
    return keys;
  }

  void disable() {
    std::scoped_lock lock(mutex_);
    disable_locked();
  }

  bool disabled() const {
    std::scoped_lock lock(mutex_);
    return disabled_;
  }

  // Lookups bypass the caches until a later snapshot completes.
  bool snapshot_failed() const {
    std::scoped_lock lock(mutex_);
    return snapshot_failed_;
  }

  // Starts re-enabling disabled caches when nothing writes the backend: like a
  // writer, it rejects admissions of reads from before it while the caller
  // clears the caches.
  bool begin_recovery() {
    std::scoped_lock lock(mutex_);
    if (!disabled_ || writers_ > 0 || served_snapshots_ > 0 ||
        generation_ == std::numeric_limits<int64_t>::max()) {
      return false;
    }
    advance_locked();
    ++writers_;
    return true;
  }

  // Re-enables the caches if the caller cleared them.
  void end_recovery(bool cleared) {
    std::scoped_lock lock(mutex_);
    advance_locked();
    --writers_;
    if (cleared && generation_ != std::numeric_limits<int64_t>::max()) {
      disabled_ = false;
      update_gauge_locked();
    }
  }

  // Rate-limits recovery attempts from lookups.
  bool recovery_due(std::chrono::milliseconds interval) {
    const auto now =
        std::chrono::steady_clock::now().time_since_epoch().count();
    auto last = last_recovery_attempt_.load(std::memory_order_relaxed);
    return now - last >=
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            interval)
            .count() &&
        last_recovery_attempt_.compare_exchange_strong(last, now);
  }

  // Eviction state is reached only through these, each taking the lock that
  // lock_eviction() returns.
  [[nodiscard]] std::unique_lock<std::mutex> lock_eviction() {
    return std::unique_lock<std::mutex>(eviction_mutex_);
  }

  bool eviction_active(const std::unique_lock<std::mutex>& held) const {
    check_eviction_lock(held);
    return eviction_active_;
  }

  // Lookups bypass the caches from here until end_eviction.
  void begin_eviction(const std::unique_lock<std::mutex>& held) {
    check_eviction_lock(held);
    if (!eviction_active_) {
      begin();
      eviction_active_ = true;
    }
  }

  void end_eviction(const std::unique_lock<std::mutex>& held) {
    check_eviction_lock(held);
    if (eviction_active_) {
      end();
      eviction_active_ = false;
    }
  }

  void fail_eviction(const std::unique_lock<std::mutex>& held) {
    disable();
    end_eviction(held);
  }

  std::vector<int64_t> stats() const {
    std::scoped_lock lock(mutex_);
    return {generation_, writers_};
  }

  int64_t staged_write_keys() const {
    std::scoped_lock lock(mutex_);
    int64_t keys = 0;
    for (const auto& staged : staged_writes_) {
      keys += staged.numel();
    }
    return keys;
  }

 private:
  void check_eviction_lock(const std::unique_lock<std::mutex>& held) const {
    TORCH_CHECK(
        held.owns_lock() && held.mutex() == &eviction_mutex_,
        "KV eviction state needs the backend's eviction lock");
  }

  void advance_locked() {
    if (generation_ == std::numeric_limits<int64_t>::max()) {
      disable_locked();
    } else {
      ++generation_;
    }
  }

  void disable_locked() {
    if (!disabled_) {
      export_hbm_cache_disable_event();
    }
    disabled_ = true;
    update_gauge_locked();
  }

  bool off_locked() const {
    return disabled_ || snapshot_failed_;
  }

  // The disabled-backends gauge counts backends with HBM caches whose lookups
  // bypass them until a recovery or a later snapshot.
  void update_gauge_locked() {
    const bool counted =
        off_locked() && has_hbm_caches.load(std::memory_order_acquire);
    if (counted != gauge_counted_) {
      gauge_counted_ = counted;
      report_disabled_hbm_backends(counted ? 1 : -1);
    }
  }

  int64_t generation_ = 0;
  int64_t writers_ = 0;
  int64_t served_snapshots_ = 0;
  // Served snapshots that ended and whose guards are still refreshing.
  int64_t refreshing_snapshots_ = 0;
  std::atomic<std::chrono::steady_clock::rep> last_recovery_attempt_{0};
  int64_t last_snapshot_failure_generation_ = 0;
  void clear_staged_writes_locked() {
    staged_writes_.clear();
    ++staged_writes_clears_;
  }

  std::vector<at::Tensor> staged_writes_;
  int64_t staged_writes_clears_ = 0;
  std::mutex eviction_mutex_;
  mutable std::mutex mutex_;

 public:
  // Set once the backend has an HBM cache on any device.
  std::atomic<bool> has_hbm_caches{false};

 private:
  bool eviction_active_ = false; // Guarded by eviction_mutex_.
  bool publishing_ = false;
  // The current publish keeps lookups on the caches (begin_publish).
  bool publish_serving_ = false;
  bool disabled_ = false;
  bool snapshot_failed_ = false;
  bool gauge_counted_ = false;
};

// Set-associative LRU cache of KV rows in HBM, shared by every wrapper bound to
// the same backend on the same device. Lookups copy hit rows into a per-call
// buffer under the lock, so later evictions cannot change rows a caller has
// already taken. Every locked section ends with a stream sync, which orders
// cache mutations across the callers' streams.
class KVHbmRowCache {
 public:
  // Tag of empty and invalidated ways. Key -1 is never cached, so it cannot
  // match them.
  static constexpr int64_t kEmptyTag = -1;

  KVHbmRowCache(
      int64_t rows,
      int64_t assoc,
      int64_t row_bytes,
      const c10::Device& device,
      std::shared_ptr<KVBackendMutationState> mutation)
      : assoc_{assoc},
        num_sets_{std::max<int64_t>(rows / assoc, 1)},
        row_bytes_{row_bytes},
        device_{device},
        mutation_{std::move(mutation)} {
    const auto options = at::TensorOptions().device(device);
    tags_ = at::full({num_sets_, assoc_}, kEmptyTag, options.dtype(at::kLong));
    lru_ = at::zeros({num_sets_, assoc_}, options.dtype(at::kLong));
    rows_ =
        at::zeros({num_sets_ * assoc_, row_bytes_}, options.dtype(at::kByte));
    // Wrappers on other streams may probe the cache once it is registered.
    // The initialization above ran only on the current stream.
    sync();
    LOG(INFO) << "KV HBM row cache on " << device << ": " << num_sets_
              << " sets x " << assoc_ << " ways x " << row_bytes_
              << " B = " << (rows_.nbytes() >> 20) << " MiB";
  }

  // Lookups may have used the rows on any stream of the device, so freeing
  // them waits for the device first.
  ~KVHbmRowCache() {
    try {
      c10::impl::VirtualGuardImpl(device_.type())
          .synchronizeDevice(device_.index());
    } catch (const std::exception& e) {
      LOG(ERROR) << "Cannot synchronize " << device_
                 << " before freeing a KV HBM row cache: " << e.what();
    }
  }

  KVHbmRowCache(const KVHbmRowCache&) = delete;
  KVHbmRowCache& operator=(const KVHbmRowCache&) = delete;

  struct Lookup {
    int64_t generation;
    at::Tensor rows; // [U, row_bytes], hit rows filled
    at::Tensor inverse; // [N] position of each key in unique keys
    at::Tensor miss_positions; // [M] positions of misses in unique keys
    at::Tensor miss_keys; // [M] on device
    at::Tensor miss_keys_cpu; // [M]
  };

  struct DirectLease {
    explicit DirectLease(std::atomic<int64_t>& active) : active_{&active} {
      ++*active_;
    }

    ~DirectLease() {
      --*active_;
    }

    std::atomic<int64_t>* active_;
  };

  struct DirectLookup {
    int64_t generation;
    std::shared_ptr<DirectLease> lease;
    at::Tensor rows;
    at::Tensor miss_keys;
    at::Tensor miss_keys_cpu;
    at::Tensor miss_slots;
    at::Tensor miss_rank;
  };

  std::optional<DirectLookup> lookup_direct(
      const at::Tensor& keys,
      int64_t logical_row_bytes,
      bool fused_enabled) {
    if (!fused_enabled || !keys.is_cuda() || keys.dim() != 1 ||
        keys.numel() == 0 || keys.numel() > 512 * 1024 || assoc_ > 32 ||
        row_bytes_ != 80 || logical_row_bytes != 72) {
      ++direct_fallback_unsupported_;
      return std::nullopt;
    }
#ifdef USE_ROCM
    ++direct_fallback_unsupported_;
    return std::nullopt;
#endif
    // Bypassing is always safe, so lookups skip the cache lock while a writer
    // holds the backend; the check repeats under the lock.
    if (!mutation_->read_generation()) {
      ++generation_bypasses_;
      return std::nullopt;
    }
    std::scoped_lock lock(mutex_);
    const auto generation = mutation_->read_generation();
    if (!generation) {
      ++generation_bypasses_;
      return std::nullopt;
    }
    if (direct_inflight_.load() >= 4) {
      ++direct_fallback_capacity_;
      return std::nullopt;
    }
    auto lease = std::make_shared<DirectLease>(direct_inflight_);
    using Result =
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t>;
    using Signature = Result(
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        int64_t);
    static const auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::kv_hbm_cache_lookup_direct", "")
            .typed<Signature>();
    const auto next_tick = tick_ + 1;
    auto [rows, miss_keys, miss_slots, miss_rank, hits] =
        op.call(keys, tags_, rows_, lru_, next_tick);
    tick_ = next_tick;
    auto miss_keys_cpu = miss_keys.to(at::kCPU);
    // The miss key copy waits for the stream unless there is nothing to copy.
    if (miss_keys.numel() == 0) {
      sync();
    }
    hits_ += hits;
    misses_ += miss_keys.numel();
    lookups_ += keys.numel();
    ++direct_lookup_calls_;
    direct_sorted_miss_keys_ += miss_keys.numel();
    maybe_log_stats_locked();
    return DirectLookup{
        *generation,
        std::move(lease),
        std::move(rows),
        std::move(miss_keys),
        std::move(miss_keys_cpu),
        std::move(miss_slots),
        std::move(miss_rank)};
  }

  std::optional<Lookup> lookup(const at::Tensor& keys, bool fused_lookup) {
    // As in lookup_direct, before the sort as well.
    if (!mutation_->read_generation()) {
      ++generation_bypasses_;
      return std::nullopt;
    }
    auto unique = at::_unique2(keys, /*sorted=*/true, /*return_inverse=*/true);
    const auto& unique_keys = std::get<0>(unique);
    std::scoped_lock lock(mutex_);
    const auto generation = mutation_->read_generation();
    if (!generation) {
      ++generation_bypasses_;
      return std::nullopt;
    }
    ++tick_;
    at::Tensor rows;
    at::Tensor hit;
    if (fused_lookup && unique_keys.is_cuda() && assoc_ > 0 && assoc_ <= 32) {
      std::tie(rows, hit) = fused_probe_locked(unique_keys);
    } else {
      const auto probe = probe_locked(unique_keys);
      rows = at::empty({unique_keys.numel(), row_bytes_}, rows_.options());
      hit = probe.hit;
      const auto hit_positions = hit.nonzero().squeeze(1);
      const auto hit_slots = probe.slots.index_select(0, hit_positions);
      rows.index_copy_(0, hit_positions, rows_.index_select(0, hit_slots));
      lru_.view(-1).index_fill_(0, hit_slots, tick_);
    }
    auto result = finish_lookup(
        *generation, std::move(unique), std::move(rows), hit, keys.numel());
    // finish_lookup's nonzero and miss key copy already waited for the stream.
    maybe_log_stats_locked();
    return result;
  }

  // Inserts rows for keys that are still absent, replacing the least recently
  // used ways of each set. Keys beyond the associativity of their set in one
  // call are not admitted.
  void admit(
      const at::Tensor& keys,
      const at::Tensor& rows,
      bool fused_admission,
      int64_t generation) {
    std::scoped_lock lock(mutex_);
    if (!mutation_->can_admit(generation)) {
      generation_rejected_ += keys.numel();
      return;
    }
    auto admitted_keys = keys;
    auto admitted_rows = rows;
    if (const auto kept = nonzero_row_positions_locked(admitted_rows)) {
      if (kept->numel() == 0) {
        sync();
        return;
      }
      admitted_keys = admitted_keys.index_select(0, *kept);
      admitted_rows = admitted_rows.index_select(0, *kept);
    }
    if (fused_admission &&
        try_fused_admission_locked(admitted_keys, admitted_rows)) {
      return;
    }
    admit_locked(admitted_keys, admitted_rows);
  }

  void admit_direct(
      const DirectLookup& lookup,
      const at::Tensor& packed_rows,
      bool fused_admission) {
    std::scoped_lock lock(mutex_);
    if (!mutation_->can_admit(lookup.generation)) {
      generation_rejected_ += lookup.miss_keys.numel();
      pad_direct_output_locked(lookup, packed_rows);
      sync();
      return;
    }
    const auto kept = nonzero_row_positions_locked(packed_rows);
    if (!kept.has_value() && fused_admission &&
        try_fused_admission_locked(lookup.miss_keys, packed_rows, &lookup)) {
      return;
    }
    auto padded_rows = pad_direct_output_locked(lookup, packed_rows);
    if (!kept.has_value()) {
      admit_locked(lookup.miss_keys, padded_rows);
    } else if (kept->numel() > 0) {
      admit_locked(
          lookup.miss_keys.index_select(0, *kept),
          padded_rows.index_select(0, *kept));
    }
    sync();
  }

  std::vector<int64_t> direct_stats() const {
    return {
        direct_lookup_calls_.load(),
        direct_sorted_miss_keys_.load(),
        direct_fallback_unsupported_.load(),
        direct_fallback_capacity_.load()};
  }

  std::vector<int64_t> generation_stats() const {
    return {generation_bypasses_.load(), generation_rejected_.load()};
  }

  int64_t zero_rows_not_admitted() const {
    return zero_rows_not_admitted_.load();
  }

  void invalidate(const at::Tensor& keys) {
    std::scoped_lock lock(mutex_);
    // Snapshot loaders pass [N, 1] indices.
    const auto flat_keys = keys.reshape({-1});
    const auto num_keys = flat_keys.numel();
    for (int64_t start = 0; start < num_keys;
         start += kHbmCacheProbeChunkKeys) {
      const auto device_keys =
          flat_keys
              .narrow(
                  0, start, std::min(kHbmCacheProbeChunkKeys, num_keys - start))
              .to(tags_.device(), at::kLong)
              .contiguous();
      const auto probe = probe_locked(device_keys);
      const auto slots = probe.slots.masked_select(probe.hit);
      tags_.view(-1).index_fill_(0, slots, kEmptyTag);
      // Live rows have ticks of at least 1, so empty slots are replaced first.
      lru_.view(-1).index_fill_(0, slots, 0);
      invalidated_ += slots.numel();
    }
    sync();
  }

  void clear() {
    std::scoped_lock lock(mutex_);
    tags_.fill_(kEmptyTag);
    lru_.zero_();
    sync();
    invalidated_ += tags_.numel();
  }

  // Sorted CPU keys of the cached rows, selected and sorted on the host.
  at::Tensor resident_keys() {
    std::scoped_lock lock(mutex_);
    const auto tags = tags_.to(at::kCPU);
    return std::get<0>(tags.masked_select(tags.ne(kEmptyTag)).sort());
  }

  // Invalidates the cached keys `contains_keys` reports absent from the
  // backend's published snapshot. Without that answer, clears every row.
  void sweep_absent(kv_mem::KVInferenceEmbeddingInterface<uint8_t>& backend) {
    const auto start = std::chrono::steady_clock::now();
    const auto keys = resident_keys();
    const auto present = backend.contains_keys(keys);
    int64_t absent = 0;
    if (present) {
      const auto absent_keys = keys.masked_select(present->logical_not());
      absent = absent_keys.numel();
      if (absent > 0) {
        invalidate(absent_keys);
      }
    } else {
      absent = keys.numel();
      clear();
    }
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
    ++sweeps_;
    swept_keys_ += keys.numel();
    swept_absent_ += absent;
    last_sweep_ms_ = millis;
    LOG(INFO) << "KV HBM row cache on " << device_
              << " swept after a snapshot: " << keys.numel() << " keys, "
              << absent << " invalidated as absent"
              << (present ? "" : " (backend cannot check keys)") << ", "
              << millis << " ms";
  }

  // [sweeps, keys checked, keys invalidated, last sweep ms].
  std::vector<int64_t> sweep_stats() const {
    return {
        sweeps_.load(),
        swept_keys_.load(),
        swept_absent_.load(),
        last_sweep_ms_.load()};
  }

  int64_t slots() const {
    return num_sets_ * assoc_;
  }

  int64_t row_bytes() const {
    return row_bytes_;
  }

  const c10::Device& device() const {
    return device_;
  }

  const at::Tensor& rows() const {
    return rows_;
  }

  std::vector<int64_t> stats() const {
    return {
        hits_.load(),
        misses_.load(),
        admitted_.load(),
        rejected_.load(),
        invalidated_.load()};
  }

  std::vector<int64_t> fused_admission_stats() const {
    return {
        fused_admission_attempts_.load(),
        fused_admission_calls_.load(),
        fused_admission_rows_.load(),
        fallback_small_batch_.load(),
        fallback_set_overflow_.load(),
        fallback_generation_overflow_.load(),
        fallback_unsupported_.load()};
  }

 private:
  std::tuple<at::Tensor, at::Tensor> fused_probe_locked(
      const at::Tensor& unique_keys) {
    using Signature = std::tuple<at::Tensor, at::Tensor>(
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        int64_t);
    static const auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::kv_hbm_cache_lookup", "")
            .typed<Signature>();
    return op.call(unique_keys, tags_, rows_, lru_, tick_);
  }

  // Splits probed unique keys into misses for the backend and copies the miss
  // keys to the host. Touches only per-call tensors.
  Lookup finish_lookup(
      int64_t generation,
      std::tuple<at::Tensor, at::Tensor, at::Tensor> unique,
      at::Tensor rows,
      const at::Tensor& hit,
      int64_t num_keys) {
    const auto& unique_keys = std::get<0>(unique);
    const auto miss_positions = hit.logical_not().nonzero().squeeze(1);
    auto miss_keys = unique_keys.index_select(0, miss_positions);
    auto miss_keys_cpu = miss_keys.to(at::kCPU);
    hits_ += unique_keys.numel() - miss_positions.numel();
    misses_ += miss_positions.numel();
    lookups_ += num_keys;
    return Lookup{
        generation,
        std::move(rows),
        std::move(std::get<1>(unique)),
        miss_positions,
        std::move(miss_keys),
        std::move(miss_keys_cpu)};
  }

  // All-zero rows are what the backend returns for absent ids, so caching them
  // would spend slots on ids with no row. Returns the positions of the other
  // rows, or nullopt when every row is kept.
  std::optional<at::Tensor> nonzero_row_positions_locked(
      const at::Tensor& rows) {
    auto kept = rows.ne(0).any(1).nonzero().squeeze(1);
    const auto skipped = rows.size(0) - kept.numel();
    if (skipped == 0) {
      return std::nullopt;
    }
    zero_rows_not_admitted_ += skipped;
    return kept;
  }

  at::Tensor pad_direct_output_locked(
      const DirectLookup& lookup,
      const at::Tensor& packed_rows) {
    using Signature = at::Tensor(
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&);
    static const auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::kv_hbm_cache_pad_output", "")
            .typed<Signature>();
    return op.call(
        packed_rows, lookup.miss_slots, lookup.miss_rank, lookup.rows);
  }

  void admit_locked(const at::Tensor& keys, const at::Tensor& rows) {
    const auto probe = probe_locked(keys);
    const auto absent = probe.hit.logical_not()
                            .logical_and(keys.ne(kEmptyTag))
                            .nonzero()
                            .squeeze(1);
    if (absent.numel() == 0) {
      return;
    }
    const auto [source, slots] = place_absent(probe, absent, lru_);
    // Allocate before mutating any slot, so a failure leaves the cache as it
    // was; if a write itself fails, the slots may pair a key with another
    // key's row, so lookups bypass the caches until they are cleared.
    const auto selected_rows = rows.index_select(0, source);
    const auto selected_keys = keys.index_select(0, source);
    try {
      rows_.index_copy_(0, slots, selected_rows);
      tags_.view(-1).index_copy_(0, slots, selected_keys);
      lru_.view(-1).index_fill_(0, slots, tick_);
    } catch (...) {
      mutation_->disable();
      throw;
    }
    sync();
    admitted_ += source.numel();
    rejected_ += absent.numel() - source.numel();
  }

  struct Probe {
    at::Tensor sets;
    at::Tensor hit;
    at::Tensor slots;
  };

  // Picks the least recently used ways for the absent positions of a probe,
  // at most one per way of a set. Returns the placed positions and slots.
  static std::pair<at::Tensor, at::Tensor> place_absent(
      const Probe& probe,
      const at::Tensor& absent,
      const at::Tensor& lru) {
    const auto assoc = lru.size(1);
    auto sets = probe.sets.index_select(0, absent);
    const auto [sorted_sets, order] = sets.sort();
    const auto first_in_set = at::searchsorted(sorted_sets, sorted_sets);
    const auto rank =
        at::arange(sorted_sets.numel(), sorted_sets.options()) - first_in_set;
    const auto admitted = rank.lt(assoc).nonzero().squeeze(1);
    auto source = absent.index_select(0, order.index_select(0, admitted));
    sets = sorted_sets.index_select(0, admitted);
    const auto ways =
        lru.index_select(0, sets)
            .argsort(/*stable=*/false, /*dim=*/1, /*descending=*/false)
            .gather(1, rank.index_select(0, admitted).unsqueeze(1))
            .squeeze(1);
    return {std::move(source), sets * assoc + ways};
  }

  static Probe probe_tags(const at::Tensor& tags, const at::Tensor& keys) {
    // splitmix-style mixing; int64 multiplication wraps on both CPU and CUDA.
    auto mixed = keys.bitwise_xor(keys.bitwise_right_shift(31))
                     .mul(static_cast<int64_t>(0x9E3779B97F4A7C15ULL));
    mixed = mixed.bitwise_xor(mixed.bitwise_right_shift(29));
    auto sets = mixed.remainder(tags.size(0));
    const auto matches = tags.index_select(0, sets).eq(keys.unsqueeze(1));
    auto hit = matches.any(1).logical_and(keys.ne(kEmptyTag));
    auto slots = sets * tags.size(1) + matches.to(at::kByte).argmax(1);
    return {std::move(sets), std::move(hit), std::move(slots)};
  }

  bool try_fused_admission_locked(
      const at::Tensor& keys,
      const at::Tensor& rows,
      const DirectLookup* direct = nullptr) {
    ++fused_admission_attempts_;
    if (!keys.is_cuda() || assoc_ > 32 ||
        keys.numel() > std::numeric_limits<int32_t>::max() ||
        num_sets_ > std::numeric_limits<int32_t>::max()) {
      ++fallback_unsupported_;
      return false;
    }
    if (keys.numel() <= kHbmCacheMinFusedAdmissionKeys) {
      ++fallback_small_batch_;
      return false;
    }
    if (!admission_epochs_.defined()) {
      admission_epochs_ = at::zeros({num_sets_}, tags_.options());
      admission_heads_ =
          at::empty({num_sets_}, tags_.options().dtype(at::kInt));
      admission_counts_ =
          at::empty({num_sets_}, tags_.options().dtype(at::kInt));
    }
    const auto scratch_options = tags_.options().dtype(at::kInt);
    auto next = at::empty({keys.numel()}, scratch_options);
    auto touched = at::empty({keys.numel()}, scratch_options);
    if (admission_generation_ <= std::numeric_limits<int64_t>::max() / 2) {
      ++admission_generation_;
    }
    using Result = std::tuple<int64_t, int64_t, bool, bool, bool, bool>;
    using Signature = Result(
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        int64_t,
        int64_t);
    Result result;
    // A kernel fault after the write began can leave tags paired with other
    // keys' rows, so lookups bypass the caches until they are cleared.
    try {
      if (direct) {
        using DirectSignature = Result(
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            const at::Tensor&,
            int64_t,
            int64_t);
        static const auto op =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::kv_hbm_cache_admit_direct", "")
                .typed<DirectSignature>();
        result = op.call(
            keys,
            rows,
            tags_,
            rows_,
            lru_,
            admission_epochs_,
            admission_heads_,
            admission_counts_,
            next,
            touched,
            direct->miss_slots,
            direct->miss_rank,
            direct->rows,
            tick_,
            admission_generation_);
      } else {
        static const auto op =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::kv_hbm_cache_admit", "")
                .typed<Signature>();
        result = op.call(
            keys,
            rows,
            tags_,
            rows_,
            lru_,
            admission_epochs_,
            admission_heads_,
            admission_counts_,
            next,
            touched,
            tick_,
            admission_generation_);
      }
    } catch (...) {
      mutation_->disable();
      throw;
    }
    const auto [admitted, rejected, small, overflow, generation, unsupported] =
        result;
    if (small || overflow || generation || unsupported) {
      fallback_small_batch_ += small;
      fallback_set_overflow_ += overflow;
      fallback_generation_overflow_ += generation;
      fallback_unsupported_ += unsupported;
      return false;
    }
    ++fused_admission_calls_;
    fused_admission_rows_ += admitted;
    admitted_ += admitted;
    rejected_ += rejected;
    // The op synchronized the stream after its write kernel, so lookups on
    // other streams see its tags only with its rows.
    return true;
  }

  Probe probe_locked(const at::Tensor& keys) const {
    return probe_tags(tags_, keys);
  }

  void maybe_log_stats_locked() {
    const auto interval = std::chrono::seconds{
        FLAGS_kv_embedding_table_log_lookup_stats_interval_secs};
    const auto now = std::chrono::steady_clock::now();
    if (interval.count() <= 0 || now - last_log_time_ < interval) {
      return;
    }
    const auto current = stats();
    const auto window_hits = current[0] - last_logged_[0];
    const auto window_unique = window_hits + current[1] - last_logged_[1];
    const auto fused = fused_admission_stats();
    const auto direct = direct_stats();
    LOG(INFO) << "KV HBM row cache on " << device_
              << ": window unique-key hit rate "
              << (window_unique ? 100.0 * window_hits / window_unique : 0.0)
              << "% (" << window_hits << "/" << window_unique << "), lookups "
              << lookups_.load() - last_logged_lookups_
              << "; lifetime hits=" << current[0] << " misses=" << current[1]
              << " admitted=" << current[2] << " rejected=" << current[3]
              << " invalidated=" << current[4]
              << "; fused admission window attempts="
              << fused[0] - last_logged_fused_[0]
              << " calls=" << fused[1] - last_logged_fused_[1]
              << " rows=" << fused[2] - last_logged_fused_[2]
              << " fallback_small_batch=" << fused[3] - last_logged_fused_[3]
              << " fallback_set_overflow=" << fused[4] - last_logged_fused_[4]
              << " fallback_generation_overflow="
              << fused[5] - last_logged_fused_[5]
              << " fallback_unsupported=" << fused[6] - last_logged_fused_[6]
              << "; direct output window executions="
              << direct[0] - last_logged_direct_[0]
              << " miss_keys_sorted=" << direct[1] - last_logged_direct_[1]
              << " unsupported=" << direct[2] - last_logged_direct_[2]
              << " capacity_limit=" << direct[3] - last_logged_direct_[3]
              << "; backend generation=" << mutation_->stats()[0]
              << " generation_bypass=" << generation_bypasses_.load()
              << " generation_rejected_rows=" << generation_rejected_.load()
              << " zero_rows_not_admitted=" << zero_rows_not_admitted_.load();
    last_log_time_ = now;
    last_logged_ = current;
    last_logged_fused_ = fused;
    last_logged_direct_ = direct;
    last_logged_lookups_ = lookups_.load();
  }

  // Copying a cache value to the host waits for all prior work on the stream.
  void sync() const {
    tags_.view(-1).narrow(0, 0, 1).to(at::kCPU);
  }

  const int64_t assoc_;
  const int64_t num_sets_;
  const int64_t row_bytes_;
  const c10::Device device_;
  const std::shared_ptr<KVBackendMutationState> mutation_;
  std::mutex mutex_;
  int64_t tick_ = 0;
  at::Tensor tags_;
  at::Tensor lru_;
  at::Tensor rows_;
  at::Tensor admission_epochs_;
  at::Tensor admission_heads_;
  at::Tensor admission_counts_;
  int64_t admission_generation_ = 0;
  std::atomic<int64_t> hits_{0};
  std::atomic<int64_t> misses_{0};
  std::atomic<int64_t> admitted_{0};
  std::atomic<int64_t> rejected_{0};
  std::atomic<int64_t> invalidated_{0};
  std::atomic<int64_t> lookups_{0};
  std::atomic<int64_t> generation_bypasses_{0};
  std::atomic<int64_t> generation_rejected_{0};
  std::atomic<int64_t> zero_rows_not_admitted_{0};
  std::atomic<int64_t> fused_admission_attempts_{0};
  std::atomic<int64_t> fused_admission_calls_{0};
  std::atomic<int64_t> fused_admission_rows_{0};
  std::atomic<int64_t> fallback_small_batch_{0};
  std::atomic<int64_t> fallback_set_overflow_{0};
  std::atomic<int64_t> fallback_generation_overflow_{0};
  std::atomic<int64_t> fallback_unsupported_{0};
  std::atomic<int64_t> direct_inflight_{0};
  std::atomic<int64_t> direct_lookup_calls_{0};
  std::atomic<int64_t> direct_sorted_miss_keys_{0};
  std::atomic<int64_t> direct_fallback_unsupported_{0};
  std::atomic<int64_t> direct_fallback_capacity_{0};
  std::atomic<int64_t> sweeps_{0};
  std::atomic<int64_t> swept_keys_{0};
  std::atomic<int64_t> swept_absent_{0};
  std::atomic<int64_t> last_sweep_ms_{0};
  std::chrono::steady_clock::time_point last_log_time_{
      std::chrono::steady_clock::now()};
  std::vector<int64_t> last_logged_ = std::vector<int64_t>(5, 0);
  std::vector<int64_t> last_logged_fused_ = std::vector<int64_t>(7, 0);
  std::vector<int64_t> last_logged_direct_ = std::vector<int64_t>(4, 0);
  int64_t last_logged_lookups_ = 0;
};

namespace {

struct HbmCacheEntry {
  std::weak_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>> backend;
  c10::Device device;
  // Owned by the wrappers that use it (owned_hbm_caches_).
  std::weak_ptr<KVHbmRowCache> cache;
  // Bytes of each row the wrappers sharing the cache look up and admit.
  int64_t lookup_row_bytes;
  // Allocated at load for a KV leaf placed on this device: the backend may
  // have no cache on any other device.
  bool exclusive = false;
};

struct BackendMutationEntry {
  std::weak_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>> backend;
  std::shared_ptr<KVBackendMutationState> mutation;
};

std::mutex& hbm_cache_registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

// Leaked so it outlives the wrappers that free caches at exit.
std::vector<HbmCacheEntry>& hbm_cache_registry() {
  static auto* registry = new std::vector<HbmCacheEntry>();
  return *registry;
}

std::shared_ptr<KVBackendMutationState> get_or_create_backend_mutation(
    const std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>&
        backend) {
  std::scoped_lock lock(hbm_cache_registry_mutex());
  // Leaked like the cache registry: an entry may hold the last reference to a
  // mutation state, whose destructor updates the disabled-backends gauge.
  static auto& registry = *new std::vector<BackendMutationEntry>();
  std::erase_if(registry, [](const BackendMutationEntry& entry) {
    return entry.backend.expired();
  });
  for (const auto& entry : registry) {
    if (entry.backend.lock() == backend) {
      return entry.mutation;
    }
  }
  auto mutation = std::make_shared<KVBackendMutationState>();
  registry.push_back({backend, mutation});
  return mutation;
}

// Only wrappers with planned rows create caches. Writers of a backend without
// planned rows check has_hbm_caches without the registry lock, so every cache
// of a backend such a wrapper writes must exist before its first write: the
// predictor allocates planned caches before loading any row
// (applyPlannedKVHbmCacheRows) and refuses in-place loads that would mix
// planned and unplanned wrappers on one backend
// (checkInplaceKVHbmCacheReusable). See
// TableWithoutPlannedRowsGuardsASharedCachedBackend.
std::shared_ptr<KVHbmRowCache> get_or_create_hbm_cache(
    const std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>&
        backend,
    const std::shared_ptr<KVBackendMutationState>& mutation,
    const c10::Device& device,
    int64_t rows,
    int64_t row_bytes,
    int64_t lookup_row_bytes,
    bool exclusive = false) {
  std::scoped_lock lock(hbm_cache_registry_mutex());
  auto& registry = hbm_cache_registry();
  std::erase_if(registry, [](const HbmCacheEntry& entry) {
    return entry.backend.expired() || entry.cache.expired();
  });
  // Every entry of the backend is checked: an exclusive allocation may adopt a
  // cache created lazily on its device after others were created elsewhere.
  const bool any_exclusive =
      exclusive ||
      std::ranges::any_of(registry, [&](const HbmCacheEntry& entry) {
        return entry.exclusive && entry.backend.lock() == backend;
      });
  HbmCacheEntry* same_device = nullptr;
  for (auto& entry : registry) {
    if (entry.backend.lock() != backend) {
      continue;
    }
    TORCH_CHECK(
        entry.device == device || !any_exclusive,
        "KV HBM row cache is allocated on ",
        entry.device,
        " for this KV table, but it is used on ",
        device,
        "; the KV leaf must run on one device with one cache");
    if (entry.device == device) {
      same_device = &entry;
    }
  }
  auto cache = same_device != nullptr ? same_device->cache.lock() : nullptr;
  if (cache != nullptr) {
    TORCH_CHECK(cache->row_bytes() == row_bytes);
    // Rows admitted through a narrower wrapper end in zeros where a wider one
    // expects backend bytes.
    TORCH_CHECK(
        same_device->lookup_row_bytes == lookup_row_bytes,
        "KV HBM row cache is shared by wrappers that look up ",
        same_device->lookup_row_bytes,
        " and ",
        lookup_row_bytes,
        " bytes per row");
    same_device->exclusive = same_device->exclusive || exclusive;
    return cache;
  }
  cache = std::make_shared<KVHbmRowCache>(
      rows, kHbmCacheAssoc, row_bytes, device, mutation);
  registry.push_back({backend, device, cache, lookup_row_bytes, exclusive});
  mutation->mark_has_hbm_caches();
  return cache;
}

std::vector<std::shared_ptr<KVHbmRowCache>> hbm_caches_for_backend(
    const kv_mem::KVInferenceEmbeddingInterface<uint8_t>* backend) {
  std::scoped_lock lock(hbm_cache_registry_mutex());
  std::vector<std::shared_ptr<KVHbmRowCache>> caches;
  for (const auto& entry : hbm_cache_registry()) {
    if (entry.backend.lock().get() != backend) {
      continue;
    }
    if (auto cache = entry.cache.lock()) {
      caches.push_back(std::move(cache));
    }
  }
  return caches;
}

// Bytes per row that the wrappers sharing backend's cache on device look up.
std::optional<int64_t> hbm_cache_lookup_row_bytes(
    const kv_mem::KVInferenceEmbeddingInterface<uint8_t>* backend,
    const c10::Device& device) {
  std::scoped_lock lock(hbm_cache_registry_mutex());
  for (const auto& entry : hbm_cache_registry()) {
    if (entry.backend.lock().get() == backend && entry.device == device &&
        !entry.cache.expired()) {
      return entry.lookup_row_bytes;
    }
  }
  return std::nullopt;
}

void invalidate_hbm_caches(
    const kv_mem::KVInferenceEmbeddingInterface<uint8_t>* backend,
    const at::Tensor* indices) {
  for (const auto& cache : hbm_caches_for_backend(backend)) {
    if (indices) {
      cache->invalidate(*indices);
    } else {
      cache->clear();
    }
  }
}

class HbmCacheMutation {
 public:
  HbmCacheMutation(
      kv_mem::KVInferenceEmbeddingInterface<uint8_t>* backend,
      KVBackendMutationState* mutation)
      : backend_{backend}, mutation_{mutation} {
    mutation_->begin();
  }

  ~HbmCacheMutation() {
    if (!finished_) {
      try {
        invalidate_hbm_caches(backend_, nullptr);
      } catch (const std::exception& e) {
        LOG(ERROR) << "KV HBM cache invalidation after a failed write failed, "
                   << "disabling the caches: " << e.what();
        mutation_->disable();
      } catch (...) {
        LOG(ERROR) << "KV HBM cache invalidation after a failed write failed, "
                   << "disabling the caches";
        mutation_->disable();
      }
      mutation_->end();
    }
  }

  // The write succeeded, so a failed invalidation does not fail it: the
  // destructor then clears every cache, or disables them.
  void finish(const at::Tensor& indices) {
    try {
      invalidate_hbm_caches(backend_, &indices);
    } catch (const std::exception& e) {
      LOG(ERROR) << "KV HBM cache invalidation after a write failed, clearing "
                 << "the caches: " << e.what();
      return;
    } catch (...) {
      LOG(ERROR) << "KV HBM cache invalidation after a write failed, clearing "
                 << "the caches";
      return;
    }
    mutation_->end();
    finished_ = true;
  }

  HbmCacheMutation(const HbmCacheMutation&) = delete;
  HbmCacheMutation& operator=(const HbmCacheMutation&) = delete;

 private:
  kv_mem::KVInferenceEmbeddingInterface<uint8_t>* const backend_;
  KVBackendMutationState* const mutation_;
  bool finished_ = false;
};

// Snapshots can begin or end during a write, and a failed write may have
// staged some of its rows, so this runs whether or not the write returned. If
// the keys cannot be recorded, a served snapshot's end could miss them, so the
// caches are disabled.
void record_staged_writes_after_write(
    KVBackendMutationState& mutation,
    const at::Tensor& keys,
    std::optional<int64_t> first_record) {
  try {
    mutation.record_staged_writes_after_write(keys, first_record);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Cannot record KV delta keys staged under a served snapshot, "
               << "disabling the HBM caches: " << e.what();
    mutation.disable();
  } catch (...) {
    LOG(ERROR) << "Cannot record KV delta keys staged under a served snapshot, "
               << "disabling the HBM caches";
    mutation.disable();
  }
}

} // namespace

KVBackendSnapshotGuard::KVBackendSnapshotGuard(
    std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>> backend,
    std::shared_ptr<KVBackendMutationState> mutation,
    bool served)
    : backend_{std::move(backend)},
      mutation_{std::move(mutation)},
      served_{served},
      begin_generation_{mutation_->begin_snapshot(served_)} {}

KVBackendSnapshotGuard::~KVBackendSnapshotGuard() {
  if (served_ && completed_) {
    end_served();
    return;
  }
  const auto invalidate = [this] {
    try {
      invalidate_hbm_caches(backend_.get(), nullptr);
    } catch (const std::exception& e) {
      LOG(ERROR) << "KV HBM cache invalidation after a snapshot failed, "
                 << "disabling the caches: " << e.what();
      mutation_->disable();
    } catch (...) {
      LOG(ERROR) << "KV HBM cache invalidation after a snapshot failed, "
                 << "disabling the caches";
      mutation_->disable();
    }
  };
  // A failed snapshot bypasses the caches and rejects admissions of earlier
  // reads once it ends, so clearing them afterwards also drops rows that a
  // served snapshot's lookups admitted until then.
  if (!completed_) {
    mutation_->end_snapshot(begin_generation_, completed_, served_);
    invalidate();
  } else {
    invalidate();
    mutation_->end_snapshot(begin_generation_, completed_, served_);
  }
  mutation_->end_publish(served_);
}

// Ending the snapshot first rejects admissions of rows read before it
// published. Other cached rows stay valid, except those of keys written by
// deltas while it staged, and since values never change per id, except that
// cached keys may have been dropped.
void KVBackendSnapshotGuard::end_served() {
  mutation_->end_snapshot(begin_generation_, /*completed=*/true, served_);
  try {
    const auto staged_writes = mutation_->take_staged_writes();
    for (const auto& cache : hbm_caches_for_backend(backend_.get())) {
      if (staged_writes) {
        cache->invalidate(*staged_writes);
      }
      if (!rolled_back_) {
        cache->sweep_absent(*backend_);
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "KV HBM cache refresh after a snapshot failed, disabling "
               << "the caches: " << e.what();
    mutation_->disable();
  } catch (...) {
    LOG(ERROR) << "KV HBM cache refresh after a snapshot failed, disabling "
               << "the caches";
    mutation_->disable();
  }
  mutation_->end_publish(served_);
}

void KVBackendSnapshotGuard::complete() {
  completed_ = true;
}

void KVBackendSnapshotGuard::begin_rollback() {
  mutation_->begin_publish();
}

void KVBackendSnapshotGuard::rolled_back() {
  completed_ = true;
  rolled_back_ = true;
}

DramKVEmbeddingInferenceWrapper::DramKVEmbeddingInferenceWrapper(
    int64_t num_shards,
    double uniform_init_lower,
    double uniform_init_upper,
    bool disable_random_init)
    : num_shards_(num_shards),
      uniform_init_lower_(uniform_init_lower),
      uniform_init_upper_(uniform_init_upper),
      disable_random_init_(disable_random_init) {
  LOG(INFO) << "DramKVEmbeddingInferenceWrapper constructed with "
            << "disable_random_init = " << disable_random_init_
            << ", num_shards = " << num_shards_ << ", this = " << this
            << " (deserialize() may overwrite this if a 3+-entry payload is "
            << "present)";
}

void DramKVEmbeddingInferenceWrapper::init(
    const std::vector<SerializedSepcType>& specs,
    const int64_t row_alignment,
    const int64_t scale_bias_size_in_bytes,
    const std::optional<at::Tensor>& hash_size_cumsum) {
  init_impl(
      specs,
      row_alignment,
      row_alignment,
      scale_bias_size_in_bytes,
      hash_size_cumsum,
      InitMode::Legacy);
}

void DramKVEmbeddingInferenceWrapper::init_with_row_alignments(
    const std::vector<SerializedSepcType>& specs,
    const int64_t logical_row_alignment,
    const int64_t storage_row_alignment,
    const int64_t scale_bias_size_in_bytes,
    const std::optional<at::Tensor>& hash_size_cumsum) {
  init_impl(
      specs,
      logical_row_alignment,
      storage_row_alignment,
      scale_bias_size_in_bytes,
      hash_size_cumsum,
      InitMode::SeparateAlignments);
}

void DramKVEmbeddingInferenceWrapper::init_impl(
    const std::vector<SerializedSepcType>& specs,
    const int64_t logical_row_alignment,
    const int64_t storage_row_alignment,
    const int64_t scale_bias_size_in_bytes,
    const std::optional<at::Tensor>& hash_size_cumsum,
    const InitMode mode) {
  LOG(INFO) << "DramKVEmbeddingInferenceWrapper::init_impl() starts with "
            << "disable_random_init = " << disable_random_init_
            << ", backend already set = " << (kv_backend_ != nullptr)
            << ", this = " << this;
  TORCH_CHECK(!specs.empty(), "Dram KV embedding specs must not be empty");
  TORCH_CHECK(
      logical_row_alignment > 0,
      "Dram KV logical row alignment must be positive");
  TORCH_CHECK(
      storage_row_alignment > 0,
      "Dram KV storage row alignment must be positive");
  const auto [storage_max_row_bytes, lookup_max_row_bytes] =
      mode == InitMode::Legacy
      ? legacy_row_sizes(specs, storage_row_alignment, scale_bias_size_in_bytes)
      : separate_row_sizes(
            specs,
            logical_row_alignment,
            storage_row_alignment,
            scale_bias_size_in_bytes);
  LOG(INFO) << "Initialize dram_kv with logical_row_alignment: "
            << logical_row_alignment
            << ", storage_row_alignment: " << storage_row_alignment
            << ", scale_bias_size_in_bytes: " << scale_bias_size_in_bytes
            << ", storage_max_row_bytes: " << storage_max_row_bytes
            << ", lookup_max_row_bytes: " << lookup_max_row_bytes;
  if (initialized_) {
    TORCH_CHECK(
        storage_max_row_bytes == storage_max_row_bytes_,
        "Cannot change KV storage row width after backend initialization: "
        "existing width=",
        storage_max_row_bytes_,
        ", requested width=",
        storage_max_row_bytes);
    TORCH_CHECK(
        lookup_max_row_bytes == lookup_max_row_bytes_,
        "Cannot change KV lookup row width after backend initialization: "
        "existing width=",
        lookup_max_row_bytes_,
        ", requested width=",
        lookup_max_row_bytes);
  }
  if (backend_storage_row_bytes_.has_value()) {
    TORCH_CHECK(
        storage_max_row_bytes <= backend_storage_row_bytes_.value(),
        "KV storage row width exceeds backend capacity: backend capacity=",
        backend_storage_row_bytes_.value(),
        ", requested width=",
        storage_max_row_bytes);
  }
  storage_max_row_bytes_ = storage_max_row_bytes;
  lookup_max_row_bytes_ = lookup_max_row_bytes;
  initialized_ = true;
  if (kv_backend_ != nullptr) {
    return;
  }
  kv_backend_ = std::make_shared<kv_mem::DramKVInferenceEmbedding<uint8_t>>(
      storage_max_row_bytes_,
      uniform_init_lower_,
      uniform_init_upper_,
      c10::make_intrusive<kv_mem::FeatureEvictConfig>(
          3 /* EvictTriggerMode.MANUAL */,
          4 /* EvictTriggerStrategy::BY_TIMESTAMP_THRESHOLD */,
          0 /* trigger_step_intervals */,
          0 /* mem_util_threshold_in_GB */,
          std::nullopt /* ttls_in_mins */,
          std::nullopt /* counter_thresholds */,
          std::nullopt /* counter_decay_rates */,
          std::nullopt /* feature_score_counter_decay_rates */,
          std::nullopt /* training_id_eviction_trigger_count */,
          std::nullopt /* training_id_keep_count */,
          std::nullopt /* enable_eviction_for_feature_score_eviction_policy */,
          std::nullopt /* l2_weight_thresholds */,
          std::nullopt /* embedding_dims */,
          std::nullopt /* threshold_calculation_bucket_stride */,
          std::nullopt /* threshold_calculation_bucket_num */,
          0 /* interval for insufficient eviction s*/,
          0 /* interval for sufficient eviction s*/,
          0 /* interval_for_feature_statistics_decay_s_*/),
      num_shards_ /* num_shards */,
      num_shards_ /* num_threads */,
      8 /* row_storage_bitwidth */,
      false /* enable_async_update */,
      std::nullopt /* table_dims */,
      hash_size_cumsum.has_value() ? std::make_optional(hash_size_cumsum->cpu())
                                   : std::nullopt,
      disable_random_init_);
  uses_dram_backend_ = true;
  backend_storage_row_bytes_ = storage_max_row_bytes_;
  backend_mutation_ = get_or_create_backend_mutation(kv_backend_);
}

int64_t DramKVEmbeddingInferenceWrapper::get_max_row_bytes() const {
  return storage_max_row_bytes_;
}

int64_t DramKVEmbeddingInferenceWrapper::get_lookup_row_bytes() const {
  return lookup_max_row_bytes_;
}

std::unique_lock<std::mutex> DramKVEmbeddingInferenceWrapper::lock_if_planned()
    const {
  return planned_hbm_cache_.load(std::memory_order_relaxed)
      ? std::unique_lock<std::mutex>(hbm_cache_mutex_)
      : std::unique_lock<std::mutex>();
}

std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
DramKVEmbeddingInferenceWrapper::get_kv_backend() {
  const auto lock = lock_if_planned();
  return kv_backend_;
}

void DramKVEmbeddingInferenceWrapper::set_kv_backend(
    std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
        kv_backend) {
  TORCH_CHECK(kv_backend != nullptr, "KV backend must not be null");
  const auto lock = lock_if_planned();
  if (kv_backend == kv_backend_) {
    kv_backend_->set_disable_random_init(disable_random_init_);
    return;
  }
  const auto backend_storage_row_bytes = kv_backend->get_storage_row_bytes();
  if (initialized_ && backend_storage_row_bytes.has_value()) {
    TORCH_CHECK(
        storage_max_row_bytes_ <= backend_storage_row_bytes.value(),
        "KV storage row width exceeds backend capacity: backend capacity=",
        backend_storage_row_bytes.value(),
        ", requested width=",
        storage_max_row_bytes_);
  }
  kv_backend_ = std::move(kv_backend);
  if (kv_backend_ != nullptr) {
    kv_backend_->set_disable_random_init(disable_random_init_);
  }
  hbm_cache_.reset();
  owned_hbm_caches_.clear();
  backend_mutation_ = get_or_create_backend_mutation(kv_backend_);
  const auto* dram_backend =
      dynamic_cast<kv_mem::DramKVInferenceEmbedding<uint8_t>*>(
          kv_backend_.get());
  uses_dram_backend_ = dram_backend != nullptr;
  backend_storage_row_bytes_ = backend_storage_row_bytes;
}

std::unique_ptr<KVBackendSnapshotGuard>
DramKVEmbeddingInferenceWrapper::begin_snapshot_update() {
  const auto binding = backend_binding(/*owned=*/true);
  const bool served = [this] {
    std::scoped_lock lock(hbm_cache_mutex_);
    return values_immutable_;
  }();
  return std::unique_ptr<KVBackendSnapshotGuard>(new KVBackendSnapshotGuard(
      binding.backend_owner, binding.mutation_owner, served));
}

void DramKVEmbeddingInferenceWrapper::set_values_immutable(bool immutable) {
  std::scoped_lock lock(hbm_cache_mutex_);
  values_immutable_ = immutable;
}

bool DramKVEmbeddingInferenceWrapper::values_immutable() const {
  std::scoped_lock lock(hbm_cache_mutex_);
  return values_immutable_;
}

bool DramKVEmbeddingInferenceWrapper::is_initialized() const {
  const auto lock = lock_if_planned();
  return initialized_ && kv_backend_ != nullptr;
}

void DramKVEmbeddingInferenceWrapper::check_initialized() const {
  const auto lock = lock_if_planned();
  TORCH_CHECK(
      initialized_ && kv_backend_ != nullptr,
      "KV embedding cache must be initialized before use");
}

DramKVEmbeddingInferenceWrapper::BackendBinding
DramKVEmbeddingInferenceWrapper::backend_binding(bool owned) const {
  const auto lock = owned ? std::unique_lock<std::mutex>(hbm_cache_mutex_)
                          : lock_if_planned();
  TORCH_CHECK(
      initialized_ && kv_backend_ && backend_mutation_,
      "KV embedding cache must be initialized before use");
  BackendBinding binding{
      kv_backend_.get(),
      backend_mutation_.get(),
      nullptr,
      nullptr,
      backend_storage_row_bytes_,
      uses_dram_backend_};
  if (lock.owns_lock()) {
    binding.backend_owner = kv_backend_;
    binding.mutation_owner = backend_mutation_;
  }
  return binding;
}

void DramKVEmbeddingInferenceWrapper::set_embeddings(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<int64_t> inplace_update_ts_opt) {
  set_embeddings_impl(indices, weights, inplace_update_ts_opt, nullptr, 0, 0);
}

void DramKVEmbeddingInferenceWrapper::set_embeddings_with_value_versions(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<int64_t> inplace_update_ts_opt,
    const at::Tensor& value_versions,
    int64_t watermark,
    int64_t epoch) {
  TORCH_CHECK(
      watermark >= 0 && epoch >= 0, "Invalid KV value version metadata");
  set_embeddings_impl(
      indices,
      weights,
      inplace_update_ts_opt,
      epoch ? &value_versions : nullptr,
      watermark,
      epoch);
}

at::Tensor DramKVEmbeddingInferenceWrapper::stored_rows(
    const at::Tensor& weights,
    int64_t backend_row_bytes) const {
  // Loaders targeting CUDA pad rows to 16 bytes, including those of a KV-ZCH
  // GPU serving shard planned with no HBM cache rows; its all-zero padding is
  // not stored. Other wrappers reject the wider rows as on trunk.
  const bool cuda_padded = kv_gpu_serving_.load(std::memory_order_relaxed) &&
      weights.size(1) > storage_max_row_bytes_ &&
      weights.size(1) == round_up(storage_max_row_bytes_, kCudaRowAlignment);
  TORCH_CHECK(
      !cuda_padded ||
          !weights
               .narrow(
                   1,
                   storage_max_row_bytes_,
                   weights.size(1) - storage_max_row_bytes_)
               .ne(0)
               .any()
               .item<bool>(),
      "Embedding update rows carry non-zero bytes past the KV storage width");
  auto stored = cuda_padded
      ? weights.narrow(1, 0, storage_max_row_bytes_).contiguous()
      : weights;
  TORCH_CHECK(
      stored.size(1) <= storage_max_row_bytes_,
      "Embedding update row is wider than KV storage");
  TORCH_CHECK(
      stored.size(1) <= backend_row_bytes,
      "Embedding update row is wider than backend storage capacity");
  return stored;
}

int64_t DramKVEmbeddingInferenceWrapper::fill_new_embeddings(
    const at::Tensor& indices,
    const at::Tensor& weights) {
  TORCH_CHECK(
      indices.device().is_cpu() && weights.device().is_cpu(),
      "Fill-only KV writes take CPU indices and rows");
  TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == indices.numel(),
      "Fill-only KV writes need one row per index");
  if (indices.numel() == 0) {
    return 0;
  }
  // The write never touches an HBM cache: absent ids are never admitted to
  // one.
  const auto binding = backend_binding();
  const auto backend_row_bytes =
      binding.storage_row_bytes.value_or(storage_max_row_bytes_);
  const auto stored_weights = stored_rows(weights, backend_row_bytes);
  const auto flat_indices = indices.reshape({-1});
  // An all-zero row is an unfilled row, as for the HBM cache, which never
  // admits one: a backend that keeps such a row still counts it absent.
  at::Tensor absent;
  if (const auto present = binding.backend->contains_keys(flat_indices)) {
    absent = present->logical_not();
    const auto stored = present->nonzero().squeeze(1);
    if (stored.numel() > 0) {
      absent.index_copy_(
          0,
          stored,
          get_embeddings_cpu(
              flat_indices.index_select(0, stored),
              /*for_cuda=*/false,
              binding)
              .eq(0)
              .all(1));
    }
  } else {
    TORCH_CHECK(
        binding.backend->missing_rows_are_zero().value_or(disable_random_init_),
        "Fill-only KV writes need a backend that answers contains_keys or "
        "returns zero rows for absent ids");
    absent = get_embeddings_cpu(flat_indices, /*for_cuda=*/false, binding)
                 .eq(0)
                 .all(1);
  }
  auto positions = absent.nonzero().squeeze(1);
  if (positions.numel() > 1) {
    // One row per id: the first of an id repeated in the call.
    const auto [sorted_keys, order] =
        flat_indices.index_select(0, positions)
            .sort(/*stable=*/true, /*dim=*/0, /*descending=*/false);
    auto first = at::ones_like(sorted_keys, at::kBool);
    first.narrow(0, 1, first.numel() - 1)
        .copy_(sorted_keys.narrow(0, 1, sorted_keys.numel() - 1)
                   .ne(sorted_keys.narrow(0, 0, sorted_keys.numel() - 1)));
    positions = positions.index_select(0, order.masked_select(first));
  }
  const auto num_new = positions.numel();
  const auto existing = indices.numel() - num_new;
  fill_existing_rows_ += existing;
  facebook::fb303::fbData->addStatValue(
      "fbgemm.kv_fill.existing_rows", existing, facebook::fb303::SUM);
  if (num_new == 0) {
    return 0;
  }
  const auto keys = flat_indices.index_select(0, positions);
  auto rows = stored_weights.index_select(0, positions);
  if (!binding.uses_dram && rows.size(1) < backend_row_bytes) {
    auto padded = at::zeros({rows.size(0), backend_row_bytes}, rows.options());
    padded.slice(1, 0, rows.size(1)).copy_(rows);
    rows = padded;
  }
  folly::coro::blockingWait(binding.backend->inference_set_kv_db_async(
      keys, rows, at::tensor({num_new}, at::ScalarType::Long), std::nullopt));
  fill_written_rows_ += num_new;
  facebook::fb303::fbData->addStatValue(
      "fbgemm.kv_fill.written_rows", num_new, facebook::fb303::SUM);
  return num_new;
}

std::vector<int64_t> DramKVEmbeddingInferenceWrapper::get_fill_stats() const {
  return {fill_written_rows_.load(), fill_existing_rows_.load()};
}

void DramKVEmbeddingInferenceWrapper::set_embeddings_impl(
    const at::Tensor& indices,
    const at::Tensor& weights,
    std::optional<int64_t> inplace_update_ts_opt,
    const at::Tensor* value_versions,
    int64_t watermark,
    int64_t epoch) {
  const auto binding = backend_binding();
  TORCH_CHECK(
      weights.dim() == 2, "Embedding updates must be a two-dimensional tensor");
  const auto backend_row_bytes =
      binding.storage_row_bytes.value_or(storage_max_row_bytes_);
  const auto stored_weights = stored_rows(weights, backend_row_bytes);
  const auto count = at::tensor({indices.numel()}, at::ScalarType::Long);
  std::optional<uint32_t> inplacee_update_ts = std::nullopt;
  if (inplace_update_ts_opt.has_value()) {
    inplacee_update_ts =
        static_cast<std::uint32_t>(inplace_update_ts_opt.value());
  }

  auto backend_weights = stored_weights;
  if (!binding.uses_dram && stored_weights.size(1) < backend_row_bytes) {
    // External backends consume physical rows. A shared scratch tensor is
    // unsafe because updates can run concurrently.
    backend_weights = at::zeros(
        {stored_weights.size(0), backend_row_bytes}, weights.options());
    backend_weights.slice(1, 0, stored_weights.size(1)).copy_(stored_weights);
  }

  const auto write = [&] {
    if (value_versions) {
      TORCH_CHECK(
          value_versions->device().is_cpu() &&
              value_versions->scalar_type() == at::kLong &&
              value_versions->is_contiguous() &&
              value_versions->numel() == indices.numel(),
          "KV value versions must be row-aligned contiguous CPU int64 data");
      binding.backend->set_kv_db_sync_with_value_versions(
          indices,
          backend_weights,
          count,
          inplacee_update_ts,
          *value_versions,
          watermark,
          epoch);
    } else {
      folly::coro::blockingWait(binding.backend->inference_set_kv_db_async(
          indices, backend_weights, count, inplacee_update_ts));
    }
  };
  // A table without a planned HBM cache, e.g. every CPU-served KV table,
  // writes as on trunk: no cache bookkeeping and no registry lock. It still
  // guards the caches of a backend it shares with a table that has one, which
  // exist before its first write (see get_or_create_hbm_cache).
  if (!planned_hbm_cache_.load(std::memory_order_relaxed) &&
      !binding.mutation->has_hbm_caches.load(std::memory_order_acquire)) {
    write();
    return;
  }
  if (inplacee_update_ts && binding.mutation->serving_snapshot()) {
    write();
    return;
  }
  if (planned_hbm_cache_.load(std::memory_order_relaxed) &&
      !inplacee_update_ts) {
    const auto present = binding.backend->contains_keys(indices);
    if (present.has_value() && !present->any().item<bool>()) {
      // A key cannot be in a row cache before it has a row. Advancing rejects
      // admissions of reads from before the write.
      const auto staged = binding.mutation->record_staged_writes(indices);
      const auto invalidate = [&] {
        binding.mutation->advance();
        try {
          invalidate_hbm_caches(binding.backend, &indices);
        } catch (...) {
          binding.mutation->disable();
        }
      };
      try {
        write();
      } catch (...) {
        record_staged_writes_after_write(*binding.mutation, indices, staged);
        invalidate();
        throw;
      }
      record_staged_writes_after_write(*binding.mutation, indices, staged);
      invalidate();
      return;
    }
  }
  HbmCacheMutation mutation(binding.backend, binding.mutation);
  const auto staged = binding.mutation->record_staged_writes(indices);
  try {
    write();
  } catch (...) {
    record_staged_writes_after_write(*binding.mutation, indices, staged);
    throw;
  }
  record_staged_writes_after_write(*binding.mutation, indices, staged);
  mutation.finish(indices);
}

void DramKVEmbeddingInferenceWrapper::set_value_version_metadata(
    int64_t watermark,
    int64_t epoch) {
  TORCH_CHECK(watermark >= 0 && epoch > 0, "Invalid KV value version metadata");
  value_version_watermark_ = watermark;
  value_version_epoch_ = epoch;
}

int64_t DramKVEmbeddingInferenceWrapper::get_value_version_watermark() const {
  return value_version_watermark_;
}

int64_t DramKVEmbeddingInferenceWrapper::get_value_version_epoch() const {
  return value_version_epoch_.value_or(0);
}

void set_kv_hbm_cache_recover_interval_ms_for_testing(int64_t interval_ms) {
  hbm_cache_recover_interval_ms().store(interval_ms, std::memory_order_relaxed);
}

void set_kv_hbm_cache_serve_during_publish(bool enabled) {
  hbm_cache_serve_during_publish().store(enabled, std::memory_order_relaxed);
}

void set_kv_lookup_coverage_sample_every_for_testing(int64_t every) {
  TORCH_CHECK(every > 0, "coverage sample interval must be positive");
  coverage_sample_every().store(every, std::memory_order_relaxed);
}

void flush_kv_lookup_coverage() {
  std::scoped_lock lock(device_coverages_mutex());
  for (auto& [device, coverage] : device_coverages()) {
    coverage->flush();
  }
}

at::Tensor DramKVEmbeddingInferenceWrapper::get_embeddings(
    const at::Tensor& indices) {
  const auto binding = backend_binding();
  // Only KV-ZCH GPU serving tables, which have a planned HBM cache, report.
  const bool report = planned_hbm_cache_.load(std::memory_order_relaxed);
  KVLookupCoverage coverage{.keys = indices.numel()};
  if (!indices.is_cuda()) {
    auto rows = get_embeddings_cpu(indices, /*for_cuda=*/false, binding);
    if (report) {
      coverage.backend_keys = coverage.keys;
      coverage.zero_rows =
          zero_row_count(rows, lookup_max_row_bytes_).item<int64_t>();
      export_kv_lookup_coverage(indices.device(), coverage);
    }
    return rows;
  }
  CudaLookupCoverage cuda_coverage{
      .host = coverage,
      // One call in coverage_sample_every() per wrapper, so most lookups launch
      // no reductions and every table is sampled at the same rate.
      .device_counts = report &&
          (coverage_calls_.fetch_add(1, std::memory_order_relaxed) + 1) %
                  coverage_sample_every().load(std::memory_order_relaxed) ==
              0};
  auto rows =
      get_embeddings_cuda(indices, binding, report ? &cuda_coverage : nullptr);
  if (report) {
    export_kv_lookup_coverage(indices.device(), cuda_coverage.host);
  }
  if (cuda_coverage.device_counts) {
    const auto zero_rows = zero_row_count(rows, lookup_max_row_bytes_);
    const auto& backend_keys = cuda_coverage.device_backend_keys;
    device_coverage(rows.device())
        .add(
            at::stack({zero_rows,
                       backend_keys.defined() ? backend_keys
                                              : at::zeros_like(zero_rows)})
                .mul_(coverage_sample_every().load(std::memory_order_relaxed)));
  }
  return rows;
}

at::Tensor DramKVEmbeddingInferenceWrapper::get_embeddings_cpu(
    const at::Tensor& indices,
    bool for_cuda,
    const BackendBinding& binding) {
  const auto count = at::tensor({indices.numel()}, at::ScalarType::Long);
  const auto options =
      at::TensorOptions().dtype(at::kByte).pinned_memory(for_cuda);
  const std::array<int64_t, 2> shape{indices.numel(), lookup_max_row_bytes_};
  auto weights =
      for_cuda ? at::zeros(shape, options) : at::empty(shape, options);

  folly::coro::blockingWait(
      binding.backend->get_kv_db_async(indices, weights, count));
  return weights;
}

at::Tensor DramKVEmbeddingInferenceWrapper::get_embeddings_cuda(
    const at::Tensor& indices,
    const BackendBinding& binding,
    CudaLookupCoverage* coverage) {
  const auto keys = indices.to(at::kLong).contiguous();
  const auto uncached = [&] {
    if (coverage != nullptr) {
      coverage->host.backend_keys = keys.numel();
    }
    return get_embeddings_cuda_uncached(keys, binding);
  };
  const auto cache = hbm_cache_for(keys, binding);
  if (cache == nullptr) {
    return uncached();
  }
  const bool fused_lookup =
      hbm_cache_fused_lookup_.load(std::memory_order_relaxed);
  if (hbm_cache_direct_output_.load(std::memory_order_relaxed)) {
    const bool fused_admission =
        hbm_cache_fused_admission_.load(std::memory_order_relaxed);
    if (auto direct = cache->lookup_direct(
            keys, lookup_max_row_bytes_, fused_lookup && fused_admission)) {
      if (direct->miss_keys.numel() > 0) {
        auto packed_rows =
            get_embeddings_cpu(
                direct->miss_keys_cpu, /*for_cuda=*/true, binding)
                .to(keys.device(), /*non_blocking=*/true);
        cache->admit_direct(
            *direct,
            packed_rows,
            hbm_cache_fused_admission_.load(std::memory_order_relaxed));
      }
      if (coverage != nullptr) {
        coverage->host.backend_keys = direct->miss_keys.numel();
      }
      return direct->rows;
    }
  }
  auto lookup = cache->lookup(keys, fused_lookup);
  if (!lookup) {
    return uncached();
  }
  if (lookup->miss_keys.numel() > 0) {
    auto miss_rows = at::zeros(
        {lookup->miss_keys.numel(), cache->row_bytes()},
        lookup->rows.options());
    miss_rows.narrow(1, 0, lookup_max_row_bytes_)
        .copy_(get_embeddings_cpu(
                   lookup->miss_keys_cpu, /*for_cuda=*/true, binding)
                   .to(keys.device(), /*non_blocking=*/true));
    lookup->rows.index_copy_(0, lookup->miss_positions, miss_rows);
    cache->admit(
        lookup->miss_keys,
        miss_rows,
        hbm_cache_fused_admission_.load(std::memory_order_relaxed),
        lookup->generation);
  }
  auto rows = lookup->rows.index_select(0, lookup->inverse);
  if (coverage != nullptr) {
    const auto unique_keys = lookup->rows.size(0);
    if (coverage->device_counts) {
      coverage->device_backend_keys =
          at::zeros({unique_keys}, lookup->inverse.options().dtype(at::kBool))
              .index_fill_(0, lookup->miss_positions, true)
              .index_select(0, lookup->inverse)
              .sum();
    }
    coverage->host.hbm_unique_keys = unique_keys;
    coverage->host.hbm_unique_hits = unique_keys - lookup->miss_keys.numel();
  }
  return rows;
}

at::Tensor DramKVEmbeddingInferenceWrapper::get_embeddings_cuda_uncached(
    const at::Tensor& keys,
    const BackendBinding& binding) {
  // Batches repeat ids, so each distinct id is read from the backend once.
  const auto [unique_keys, inverse, counts] =
      at::_unique2(keys, /*sorted=*/false, /*return_inverse=*/true);
  auto rows = at::zeros(
      {unique_keys.numel(), get_cuda_row_bytes()},
      keys.options().dtype(at::kByte));
  rows.narrow(1, 0, lookup_max_row_bytes_)
      .copy_(get_embeddings_cpu(
          unique_keys.to(at::kCPU), /*for_cuda=*/true, binding));
  return rows.index_select(0, inverse.reshape({-1}));
}

int64_t DramKVEmbeddingInferenceWrapper::get_cuda_row_bytes() const {
  return round_up(lookup_max_row_bytes_, kCudaRowAlignment);
}

std::shared_ptr<KVHbmRowCache> DramKVEmbeddingInferenceWrapper::hbm_cache_for(
    const at::Tensor& indices,
    const BackendBinding& binding) {
  if (const auto interval =
          hbm_cache_recover_interval_ms().load(std::memory_order_relaxed);
      interval > 0 && binding.mutation->disabled() &&
      binding.mutation->recovery_due(std::chrono::milliseconds(interval))) {
    recover_hbm_caches(binding);
  }
  std::scoped_lock lock(hbm_cache_mutex_);
  const auto rows = hbm_cache_rows_.value_or(0);
  if (rows <= 0) {
    return nullptr;
  }
  // A backend that serves default rows for absent ids (random, or another
  // id's row) would make them sticky in HBM, so those lookups go to the
  // backend instead.
  if (!binding.backend->missing_rows_are_zero().value_or(
          disable_random_init_)) {
    ++random_init_bypasses_;
    LOG_FIRST_N(WARNING, 1)
        << "KV HBM row cache bypassed: the backend may return random rows "
           "for absent ids (set disable_random_init, and for SSD "
           "--kv_embedding_table_ssd_honor_disable_random_init)";
    return nullptr;
  }
  if (binding.backend == kv_backend_.get() && hbm_cache_ != nullptr &&
      hbm_cache_->device() == indices.device()) {
    return hbm_cache_;
  }
  // Taken before the rows were planned; this lookup goes uncached.
  if (binding.backend_owner == nullptr) {
    return nullptr;
  }
  auto cache = get_or_create_hbm_cache(
      binding.backend_owner,
      binding.mutation_owner,
      indices.device(),
      rows,
      get_cuda_row_bytes(),
      lookup_max_row_bytes_);
  if (binding.backend == kv_backend_.get()) {
    own_hbm_cache_locked(cache);
  }
  return cache;
}

int64_t DramKVEmbeddingInferenceWrapper::hbm_cache_footprint_bytes() const {
  std::scoped_lock lock(hbm_cache_mutex_);
  const auto rows = hbm_cache_rows_.value_or(0);
  if (rows <= 0) {
    return 0;
  }
  const auto slots =
      std::max<int64_t>(rows / kHbmCacheAssoc, 1) * kHbmCacheAssoc;
  constexpr int64_t kTagAndLruBytes = 2 * sizeof(int64_t);
  return slots * (get_cuda_row_bytes() + kTagAndLruBytes);
}

void DramKVEmbeddingInferenceWrapper::allocate_hbm_cache(
    const c10::Device& device) {
  TORCH_CHECK(device.is_cuda(), "KV HBM row cache needs a CUDA device");
  const auto binding = backend_binding(/*owned=*/true);
  std::scoped_lock lock(hbm_cache_mutex_);
  const auto rows = hbm_cache_rows_.value_or(0);
  if (rows <= 0 ||
      !binding.backend->missing_rows_are_zero().value_or(
          disable_random_init_)) {
    return;
  }
  auto cache = get_or_create_hbm_cache(
      binding.backend_owner,
      binding.mutation_owner,
      device,
      rows,
      get_cuda_row_bytes(),
      lookup_max_row_bytes_,
      /*exclusive=*/true);
  if (binding.backend == kv_backend_.get()) {
    own_hbm_cache_locked(cache);
  }
}

std::optional<std::string>
DramKVEmbeddingInferenceWrapper::hbm_cache_reuse_mismatch(
    const c10::Device& device) {
  const auto binding = backend_binding();
  const auto caches = hbm_caches_for_backend(binding.backend);
  const auto rows = [this] {
    std::scoped_lock lock(hbm_cache_mutex_);
    return hbm_cache_rows_.value_or(0);
  }();
  if (caches.empty()) {
    // Lookups of a backend that may return non-zero rows for absent ids
    // bypass the cache, so such a backend never has one.
    if (rows > 0 &&
        binding.backend->missing_rows_are_zero().value_or(
            disable_random_init_)) {
      return std::string("the plan has an HBM cache but the backend has none");
    }
    return std::nullopt;
  }
  if (rows <= 0) {
    return c10::str(
        "the backend has ", caches.size(), " HBM caches but the plan has none");
  }
  if (caches.size() != 1) {
    return c10::str(
        "the backend has ", caches.size(), " HBM caches but the plan has one");
  }
  const auto& cache = *caches.front();
  if (cache.device() != device) {
    return c10::str(
        "the HBM cache is on ",
        cache.device(),
        " but the plan places it on ",
        device);
  }
  const auto slots =
      std::max<int64_t>(rows / kHbmCacheAssoc, 1) * kHbmCacheAssoc;
  if (cache.slots() != slots) {
    return c10::str(
        "the HBM cache has ", cache.slots(), " slots but the plan has ", slots);
  }
  if (cache.row_bytes() != get_cuda_row_bytes()) {
    return c10::str(
        "the HBM cache holds ",
        cache.row_bytes(),
        "-byte rows but the plan has ",
        get_cuda_row_bytes(),
        "-byte rows");
  }
  // get_or_create_hbm_cache also refuses to share a cache across logical row
  // widths that pad to the same CUDA row.
  const auto shared_lookup_row_bytes =
      hbm_cache_lookup_row_bytes(binding.backend, device);
  if (shared_lookup_row_bytes.has_value() &&
      shared_lookup_row_bytes.value() != lookup_max_row_bytes_) {
    return c10::str(
        "the HBM cache is looked up with ",
        shared_lookup_row_bytes.value(),
        "-byte rows but the plan looks up ",
        lookup_max_row_bytes_,
        "-byte rows");
  }
  return std::nullopt;
}

bool DramKVEmbeddingInferenceWrapper::recover_hbm_caches() {
  return recover_hbm_caches(backend_binding());
}

bool DramKVEmbeddingInferenceWrapper::recover_hbm_caches(
    const BackendBinding& binding) {
  auto& mutation = *binding.mutation;
  if (!mutation.disabled()) {
    return !mutation.snapshot_failed();
  }
  if (!mutation.begin_recovery()) {
    return false;
  }
  bool cleared = false;
  try {
    invalidate_hbm_caches(binding.backend, nullptr);
    cleared = true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "KV HBM cache recovery failed to clear the caches: "
               << e.what();
  }
  mutation.end_recovery(cleared);
  LOG(WARNING) << "KV HBM caches of a disabled backend "
               << (cleared ? "were cleared and re-enabled" : "stay disabled");
  return cleared && !mutation.snapshot_failed();
}

bool DramKVEmbeddingInferenceWrapper::hbm_cache_disabled() {
  return backend_binding().mutation->disabled();
}

void DramKVEmbeddingInferenceWrapper::set_hbm_cache_rows(int64_t rows) {
  planned_hbm_cache_.store(rows > 0, std::memory_order_relaxed);
  kv_gpu_serving_.store(true, std::memory_order_relaxed);
  std::scoped_lock lock(hbm_cache_mutex_);
  hbm_cache_rows_ = rows;
  hbm_cache_.reset();
  owned_hbm_caches_.clear();
}

void DramKVEmbeddingInferenceWrapper::own_hbm_cache_locked(
    const std::shared_ptr<KVHbmRowCache>& cache) {
  hbm_cache_ = cache;
  if (std::ranges::find(owned_hbm_caches_, cache) == owned_hbm_caches_.end()) {
    owned_hbm_caches_.push_back(cache);
  }
}

void DramKVEmbeddingInferenceWrapper::set_hbm_cache_fused_lookup(bool enabled) {
  hbm_cache_fused_lookup_.store(enabled, std::memory_order_relaxed);
}

void DramKVEmbeddingInferenceWrapper::set_hbm_cache_fused_admission(
    bool enabled) {
  hbm_cache_fused_admission_.store(enabled, std::memory_order_relaxed);
}

void DramKVEmbeddingInferenceWrapper::set_hbm_cache_direct_output(
    bool enabled) {
  hbm_cache_direct_output_.store(enabled, std::memory_order_relaxed);
}

std::vector<int64_t>
DramKVEmbeddingInferenceWrapper::get_hbm_cache_direct_output_stats() {
  std::scoped_lock lock(hbm_cache_mutex_);
  return hbm_cache_ ? hbm_cache_->direct_stats() : std::vector<int64_t>(4, 0);
}

std::string DramKVEmbeddingInferenceWrapper::get_hbm_cache_device() {
  std::scoped_lock lock(hbm_cache_mutex_);
  return hbm_cache_ ? hbm_cache_->device().str() : "";
}

int64_t DramKVEmbeddingInferenceWrapper::get_hbm_cache_data_ptr() {
  std::scoped_lock lock(hbm_cache_mutex_);
  return hbm_cache_
      ? reinterpret_cast<int64_t>(hbm_cache_->rows().const_data_ptr())
      : 0;
}

void DramKVEmbeddingInferenceWrapper::share_kv_backend(
    const c10::intrusive_ptr<DramKVEmbeddingInferenceWrapper>& other) {
  set_kv_backend(other->get_kv_backend());
}

std::vector<int64_t> DramKVEmbeddingInferenceWrapper::get_hbm_cache_stats() {
  std::scoped_lock lock(hbm_cache_mutex_);
  return hbm_cache_ ? hbm_cache_->stats() : std::vector<int64_t>(5, 0);
}

std::vector<int64_t>
DramKVEmbeddingInferenceWrapper::get_hbm_cache_fused_admission_stats() {
  std::scoped_lock lock(hbm_cache_mutex_);
  return hbm_cache_ ? hbm_cache_->fused_admission_stats()
                    : std::vector<int64_t>(7, 0);
}

std::vector<int64_t>
DramKVEmbeddingInferenceWrapper::get_hbm_cache_admission_policy_stats() {
  std::scoped_lock lock(hbm_cache_mutex_);
  return {
      random_init_bypasses_.load(),
      hbm_cache_ ? hbm_cache_->zero_rows_not_admitted() : 0};
}

std::vector<int64_t>
DramKVEmbeddingInferenceWrapper::get_hbm_cache_snapshot_sweep_stats() {
  std::scoped_lock lock(hbm_cache_mutex_);
  return hbm_cache_ ? hbm_cache_->sweep_stats() : std::vector<int64_t>(4, 0);
}

std::vector<int64_t>
DramKVEmbeddingInferenceWrapper::get_hbm_cache_generation_stats() {
  std::scoped_lock lock(hbm_cache_mutex_);
  auto stats = backend_mutation_ ? backend_mutation_->stats()
                                 : std::vector<int64_t>(2, 0);
  const auto cache_stats =
      hbm_cache_ ? hbm_cache_->generation_stats() : std::vector<int64_t>(2, 0);
  stats.insert(stats.end(), cache_stats.begin(), cache_stats.end());
  stats.push_back(
      backend_mutation_ ? backend_mutation_->staged_write_keys() : 0);
  return stats;
}

int64_t DramKVEmbeddingInferenceWrapper::mark_existing_keys(
    const at::Tensor& indices,
    int64_t inplace_update_ts_64b) {
  const auto binding = backend_binding();
  const auto count = at::tensor({indices.numel()}, at::ScalarType::Long);
  return binding.backend->mark_existing_keys(
      indices, count, static_cast<std::uint32_t>(inplace_update_ts_64b));
}

void DramKVEmbeddingInferenceWrapper::log_inplace_update_stats() {
  backend_binding().backend->log_inplace_update_stats();
  const auto written = fill_written_rows_.load();
  const auto existing = fill_existing_rows_.load();
  if (written + existing > 0) {
    LOG(INFO) << "[KV fill] this = " << this << " written=" << written
              << " existing=" << existing;
  }
}

std::vector<int64_t>
DramKVEmbeddingInferenceWrapper::get_read_hit_rate_stats() {
  return backend_binding().backend->get_read_hit_rate_stats();
}

void DramKVEmbeddingInferenceWrapper::trigger_evict(
    int64_t inplace_update_ts_64b) {
  const auto binding = backend_binding();
  uint32_t inplace_update_ts_32b =
      static_cast<std::uint32_t>(inplace_update_ts_64b);
  // A backend without HBM caches evicts as before: its caches, if planned,
  // exist before its first row (get_or_create_hbm_cache).
  if (!binding.mutation->has_hbm_caches.load(std::memory_order_acquire)) {
    binding.backend->trigger_feature_evict(inplace_update_ts_32b);
    binding.backend->resume_ongoing_eviction();
    return;
  }
  const auto lock = binding.mutation->lock_eviction();
  // Publishes the served snapshot; its guard refreshes the caches after.
  if (binding.mutation->serving_snapshot() &&
      !binding.mutation->eviction_active(lock)) {
    const bool serve_cached =
        hbm_cache_serve_during_publish().load(std::memory_order_relaxed);
    if (binding.mutation->begin_publish(serve_cached) && serve_cached) {
      // The publish reveals the rows of keys that deltas wrote while the
      // snapshot staged; the guard invalidates them again when it ends.
      try {
        if (const auto staged = binding.mutation->take_staged_writes()) {
          for (const auto& cache : hbm_caches_for_backend(binding.backend)) {
            cache->invalidate(*staged);
          }
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << "KV HBM cache invalidation before a publish failed, "
                   << "disabling the caches: " << e.what();
        binding.mutation->disable();
      } catch (...) {
        LOG(ERROR) << "KV HBM cache invalidation before a publish failed, "
                   << "disabling the caches";
        binding.mutation->disable();
      }
    }
    binding.backend->trigger_feature_evict(inplace_update_ts_32b);
    binding.backend->resume_ongoing_eviction();
    return;
  }
  binding.mutation->begin_eviction(lock);
  try {
    binding.backend->trigger_feature_evict(inplace_update_ts_32b);
    binding.backend->resume_ongoing_eviction();
    invalidate_hbm_caches(binding.backend, nullptr);
  } catch (...) {
    binding.mutation->fail_eviction(lock);
    throw;
  }
}

void DramKVEmbeddingInferenceWrapper::wait_evict_completion() {
  const auto binding = backend_binding();
  if (!binding.mutation->has_hbm_caches.load(std::memory_order_acquire)) {
    binding.backend->wait_until_eviction_done();
    return;
  }
  const auto lock = binding.mutation->lock_eviction();
  try {
    binding.backend->wait_until_eviction_done();
    if (binding.mutation->eviction_active(lock)) {
      invalidate_hbm_caches(binding.backend, nullptr);
      binding.mutation->end_eviction(lock);
    }
  } catch (...) {
    binding.mutation->fail_eviction(lock);
    throw;
  }
}

c10::List<at::Tensor> DramKVEmbeddingInferenceWrapper::serialize() const {
  c10::List<at::Tensor> results;
  results.push_back(torch::tensor({num_shards_}, torch::kInt64));
  results.push_back(
      torch::tensor(
          {uniform_init_lower_, uniform_init_upper_}, torch::kDouble));
  results.push_back(
      torch::tensor(
          {static_cast<int64_t>(disable_random_init_)}, torch::kInt64));
  if (value_version_epoch_) {
    results.push_back(torch::tensor({value_version_watermark_}, torch::kInt64));
    results.push_back(torch::tensor({*value_version_epoch_}, torch::kInt64));
  }
  LOG(INFO) << "DramKVEmbeddingInferenceWrapper serialized " << results.size()
            << " entries with disable_random_init = " << disable_random_init_
            << ", this = " << this;
  return results;
}

void DramKVEmbeddingInferenceWrapper::deserialize(
    const c10::List<at::Tensor>& device_states) {
  value_version_watermark_ = 0;
  value_version_epoch_.reset();
  if (device_states.empty()) {
    return;
  }
  TORCH_CHECK(device_states.size() >= 2);
  // Loading with a CUDA map_location places the pickled states on the device.
  std::vector<at::Tensor> states;
  states.reserve(device_states.size());
  for (const at::Tensor& state : device_states) {
    states.push_back(state.cpu().contiguous());
  }

  const auto* intPtr = states[0].const_data_ptr<int64_t>();
  TORCH_CHECK(states[0].numel() >= 1)
  num_shards_ = intPtr[0];

  TORCH_CHECK(states[1].numel() >= 2)
  const auto* floatPtr = states[1].const_data_ptr<double>();
  uniform_init_lower_ = floatPtr[0];
  uniform_init_upper_ = floatPtr[1];

  // Payloads published before disable_random_init_ was serialized carry only
  // the two entries above; leave the constructor default in place for those.
  if (states.size() >= 3 && states[2].numel() >= 1) {
    disable_random_init_ = states[2].const_data_ptr<int64_t>()[0] != 0;
  }
  if (states.size() >= 5) {
    for (const auto i : {3, 4})
      TORCH_CHECK(
          states[i].device().is_cpu() && states[i].scalar_type() == at::kLong &&
              states[i].is_contiguous() && states[i].numel() == 1,
          "KV value version state must be CPU int64[1]");
    const auto watermark = states[3].item<int64_t>();
    const auto epoch = states[4].item<int64_t>();
    TORCH_CHECK(watermark >= 0 && epoch >= 0, "Invalid KV value version state");
    if (epoch)
      set_value_version_metadata(watermark, epoch);
  }
  if (kv_backend_ != nullptr) {
    kv_backend_->set_disable_random_init(disable_random_init_);
  }
  LOG(INFO) << "DramKVEmbeddingInferenceWrapper deserialized " << states.size()
            << " entries, final disable_random_init = " << disable_random_init_
            << ", num_shards = " << num_shards_ << ", this = " << this;
}

} // namespace fbgemm_gpu

static auto dram_kv_embedding_inference_wrapper =
    torch::class_<fbgemm_gpu::DramKVEmbeddingInferenceWrapper>(
        "fbgemm",
        "DramKVEmbeddingInferenceWrapper")
        .def(torch::init<int64_t, double, double, bool>())
        .def(
            "init",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::init,
            "",
            {
                torch::arg("specs"),
                torch::arg("row_alignment"),
                torch::arg("scale_bias_size_in_bytes"),
                torch::arg("hash_size_cumsum"),
            })
        .def(
            "init_with_row_alignments",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                init_with_row_alignments,
            "",
            {
                torch::arg("specs"),
                torch::arg("logical_row_alignment"),
                torch::arg("storage_row_alignment"),
                torch::arg("scale_bias_size_in_bytes"),
                torch::arg("hash_size_cumsum"),
            })
        .def(
            "set_embeddings",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::set_embeddings,
            "",
            {
                torch::arg("indices"),
                torch::arg("weights"),
                torch::arg("inplace_update_ts_opt") = std::nullopt,
            })
        .def(
            "fill_new_embeddings",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::fill_new_embeddings)
        .def(
            "get_fill_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_fill_stats)
        .def(
            "get_embeddings",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_embeddings)
        .def(
            "get_max_row_bytes",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_max_row_bytes)
        .def(
            "get_lookup_row_bytes",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_lookup_row_bytes)
        .def(
            "get_cuda_row_bytes",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_cuda_row_bytes)
        .def(
            "get_hbm_cache_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_hbm_cache_stats)
        .def(
            "set_hbm_cache_rows",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::set_hbm_cache_rows)
        .def(
            "set_hbm_cache_fused_lookup",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                set_hbm_cache_fused_lookup)
        .def(
            "set_hbm_cache_fused_admission",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                set_hbm_cache_fused_admission)
        .def(
            "get_hbm_cache_fused_admission_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                get_hbm_cache_fused_admission_stats)
        .def(
            "get_hbm_cache_admission_policy_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                get_hbm_cache_admission_policy_stats)
        .def(
            "get_hbm_cache_snapshot_sweep_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                get_hbm_cache_snapshot_sweep_stats)
        .def(
            "get_hbm_cache_generation_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                get_hbm_cache_generation_stats)
        .def(
            "set_hbm_cache_direct_output",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                set_hbm_cache_direct_output)
        .def(
            "get_hbm_cache_direct_output_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                get_hbm_cache_direct_output_stats)
        .def(
            "get_hbm_cache_device",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::get_hbm_cache_device)
        .def(
            "get_hbm_cache_data_ptr",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                get_hbm_cache_data_ptr)
        .def(
            "is_initialized",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::is_initialized)
        .def(
            "share_kv_backend",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::share_kv_backend)
        .def(
            "set_value_version_metadata",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                set_value_version_metadata)
        .def(
            "get_value_version_watermark",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                get_value_version_watermark)
        .def(
            "get_value_version_epoch",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                get_value_version_epoch)
        .def(
            "mark_existing_keys",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::mark_existing_keys,
            "",
            {
                torch::arg("indices"),
                torch::arg("inplace_update_ts_64b"),
            })
        .def(
            "trigger_evict",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::trigger_evict)
        .def(
            "wait_evict_completion",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::wait_evict_completion)
        .def(
            // Kept so that published snapshots calling it still load; the
            // message is not logged.
            "log_message",
            [](const c10::intrusive_ptr<
                   fbgemm_gpu::DramKVEmbeddingInferenceWrapper>&,
               const std::string& /* message */) {})
        .def(
            "log_inplace_update_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                log_inplace_update_stats)
        .def(
            "get_read_hit_rate_stats",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::
                get_read_hit_rate_stats)
        .def(
            "serialize",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::serialize)
        .def(
            "deserialize",
            &fbgemm_gpu::DramKVEmbeddingInferenceWrapper::deserialize)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<
                fbgemm_gpu::DramKVEmbeddingInferenceWrapper>& self)
                -> c10::List<at::Tensor> { return self->serialize(); },
            // __setstate__
            [](const c10::List<at::Tensor>& states) {
              auto ptr = c10::make_intrusive<
                  fbgemm_gpu::DramKVEmbeddingInferenceWrapper>();
              ptr->deserialize(states);
              return ptr;
            });

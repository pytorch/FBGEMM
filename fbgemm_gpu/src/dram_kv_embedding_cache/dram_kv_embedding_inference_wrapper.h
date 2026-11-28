/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gflags/gflags_declare.h>
#include <torch/custom_class.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/kv_inference_embedding_interface.h"

DECLARE_int64(dram_kv_embedding_num_shards);
DECLARE_bool(kv_embedding_async_get_set);

namespace fbgemm_gpu {

class KVHbmRowCache;
class KVBackendMutationState;

// Coverage of one get_embeddings call. Unique and HBM counts are zero when the
// call bypassed the HBM cache; backend keys are those served by the backend.
struct KVLookupCoverage {
  int64_t keys = 0;
  int64_t zero_rows = 0;
  int64_t backend_keys = 0;
  int64_t hbm_unique_keys = 0;
  int64_t hbm_unique_hits = 0;
};

// Tables with planned HBM cache rows export their lookup coverage as fb303
// sums fbgemm.kv_lookup.<device>.<count>: host-known counts after each lookup,
// and device-computed counts once their asynchronous device-to-host copy
// lands, at most about once a second per device. The fb303 counter
// fbgemm.kv_hbm_cache.disabled_backends counts KV backends with HBM caches
// whose caches are disabled.
// Blocks until every device count so far has been exported.
void flush_kv_lookup_coverage();

// Tests only: how often lookups retry HBM caches disabled after a failed cache
// update (60 s by default); zero keeps them disabled.
void set_kv_hbm_cache_recover_interval_ms_for_testing(int64_t interval_ms);

class KVBackendSnapshotGuard {
 public:
  ~KVBackendSnapshotGuard();
  KVBackendSnapshotGuard(const KVBackendSnapshotGuard&) = delete;
  KVBackendSnapshotGuard& operator=(const KVBackendSnapshotGuard&) = delete;

  void complete();
  // Call before the backend cancels the snapshot: cancelling exposes the delta
  // rows it staged, so lookups bypass the caches until this guard refreshes.
  void begin_rollback();
  // The backend dropped the snapshot and still serves its previous view, so
  // the HBM caches need not keep bypassing it.
  void rolled_back();

 private:
  friend class DramKVEmbeddingInferenceWrapper;
  KVBackendSnapshotGuard(
      std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>> backend,
      std::shared_ptr<KVBackendMutationState> mutation,
      bool served);

  void end_served();

  const std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
      backend_;
  const std::shared_ptr<KVBackendMutationState> mutation_;
  // Whether the caches keep serving while the snapshot stages: the table's
  // values were immutable per id when the snapshot began.
  const bool served_;
  const int64_t begin_generation_;
  bool completed_ = false;
  bool rolled_back_ = false;
};

class DramKVEmbeddingInferenceWrapper : public torch::jit::CustomClassHolder {
 public:
  DramKVEmbeddingInferenceWrapper(
      int64_t num_shards = FLAGS_dram_kv_embedding_num_shards,
      double uniform_init_lower = 0.0,
      double uniform_init_upper = 0.0,
      bool disable_random_init = false);

  using SerializedSepcType =
      std::tuple<int64_t, int64_t, int64_t>; // (rows, dime, sparse_type)

  void init(
      const std::vector<SerializedSepcType>& specs,
      const int64_t row_alignment,
      const int64_t scale_bias_size_in_bytes,
      const std::optional<at::Tensor>& hash_size_cumsum);

  void init_with_row_alignments(
      const std::vector<SerializedSepcType>& specs,
      const int64_t logical_row_alignment,
      const int64_t storage_row_alignment,
      const int64_t scale_bias_size_in_bytes,
      const std::optional<at::Tensor>& hash_size_cumsum);

  void set_embeddings(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<int64_t> inplace_update_ts_opt = std::nullopt);

  void set_embeddings_with_value_versions(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<int64_t> inplace_update_ts_opt,
      const at::Tensor& value_versions,
      int64_t watermark,
      int64_t epoch);

  void set_value_version_metadata(int64_t watermark, int64_t epoch);
  int64_t get_value_version_watermark() const;
  int64_t get_value_version_epoch() const;

  // CPU indices return [N, lookup_row_bytes] CPU rows. CUDA indices return
  // [N, round_up(lookup_row_bytes, 16)] rows on the same device, served from an
  // HBM row cache in front of the backend when set_hbm_cache_rows set rows.
  // Writes the rows of the CPU indices the backend has no row, or an all-zero
  // row, for, one per id, never touching an HBM cache, and returns how many it
  // wrote. For fill-only deltas, which never change an existing row and carry
  // the same row for an id: concurrent calls may both write an id. Exports
  // the written and dropped rows as fb303 sums fbgemm.kv_fill.written_rows and
  // fbgemm.kv_fill.existing_rows; log_inplace_update_stats logs their totals.
  int64_t fill_new_embeddings(
      const at::Tensor& indices,
      const at::Tensor& weights);

  // [rows written, rows dropped] by fill_new_embeddings so far.
  std::vector<int64_t> get_fill_stats() const;

  at::Tensor get_embeddings(const at::Tensor& indices);

  int64_t get_cuda_row_bytes() const;

  // [hits, misses, admitted, rejected, invalidated] of the HBM row cache.
  std::vector<int64_t> get_hbm_cache_stats();

  // Rows of this wrapper's HBM row cache, as planned at publish for a KV-ZCH
  // GPU serving table. Without rows (the default) there is no HBM cache, and
  // writes and lookups skip the GPU serving bookkeeping.
  void set_hbm_cache_rows(int64_t rows);
  // Device bytes of the HBM row cache; zero when off.
  int64_t hbm_cache_footprint_bytes() const;
  // Allocates the HBM row cache on device at load. It is then the only cache
  // of this KV table: lookups on any other device fail.
  void allocate_hbm_cache(const c10::Device& device);
  // Why the HBM caches of the bound backend, which an in-place snapshot
  // shares, do not match this wrapper's plan on device (cache count, device,
  // slots or row width); nullopt when they match or there are none.
  std::optional<std::string> hbm_cache_reuse_mismatch(
      const c10::Device& device);
  // Whether the HBM caches of the bound backend are disabled, as after a cache
  // update failed; lookups then bypass them.
  bool hbm_cache_disabled();
  // Clears and re-enables disabled HBM caches of the bound backend when no
  // writer or snapshot is in flight; returns whether they are enabled. After a
  // failed snapshot they stay bypassed until a later snapshot completes.
  bool recover_hbm_caches();

  // Fused lookup and admission are on and direct output is off by default.
  void set_hbm_cache_fused_lookup(bool enabled);

  void set_hbm_cache_fused_admission(bool enabled);

  void set_hbm_cache_direct_output(bool enabled);

  // [direct calls, sorted miss keys, unsupported fallbacks, capacity
  // fallbacks].
  std::vector<int64_t> get_hbm_cache_direct_output_stats();

  // [attempts, fused calls, fused rows, small-batch fallbacks, set-overflow
  // fallbacks, generation-overflow fallbacks, unsupported fallbacks].
  std::vector<int64_t> get_hbm_cache_fused_admission_stats();

  // [lookups that bypassed the HBM cache because absent ids may get random
  // rows, all-zero rows not admitted].
  std::vector<int64_t> get_hbm_cache_admission_policy_stats();

  // [post-snapshot sweeps, keys checked, keys invalidated as absent, last
  // sweep ms] of this wrapper's HBM cache.
  std::vector<int64_t> get_hbm_cache_snapshot_sweep_stats();

  // [backend generation, in-flight writers, bypassed lookups, rejected
  // admissions, keys of delta writes kept to invalidate when a served snapshot
  // publishes]. Generation counters are shared across methods and devices.
  std::vector<int64_t> get_hbm_cache_generation_stats();

  // Device of this wrapper's HBM row cache, or "" before its first CUDA lookup.
  std::string get_hbm_cache_device();

  // Address of this wrapper's HBM cache rows, or 0 before its first CUDA
  // lookup.
  int64_t get_hbm_cache_data_ptr();

  // True once init() ran and a backend is set.
  bool is_initialized() const;

  // Serves rows from `other`'s backend. HBM caches stay per device.
  void share_kv_backend(
      const c10::intrusive_ptr<DramKVEmbeddingInferenceWrapper>& other);

  // Stamps inplace_update_ts onto rows already present in the store, so a
  // subsequent trigger_evict() drops only the rows the incoming snapshot does
  // not contain. Returns how many rows were matched.
  int64_t mark_existing_keys(
      const at::Tensor& indices,
      int64_t inplace_update_ts_64b);

  void log_inplace_update_stats();

  std::vector<int64_t> get_read_hit_rate_stats();

  void trigger_evict(int64_t inplace_update_ts_64b);

  void wait_evict_completion();

  std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
  get_kv_backend();

  void set_kv_backend(
      std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
          kv_backend);

  std::unique_ptr<KVBackendSnapshotGuard> begin_snapshot_update();
  // Marks the table's values as never changing per id, which publish writes
  // for KV-ZCH enrichment tables (CU embeddings come from Laser and are
  // immutable per id). Only such tables keep their HBM caches serving while
  // an in-place snapshot stages, for backends whose staged rows stay
  // invisible until publication.
  void set_values_immutable(bool immutable);
  bool values_immutable() const;

  c10::List<at::Tensor> serialize() const;

  void deserialize(const c10::List<at::Tensor>& states);

  int64_t get_max_row_bytes() const;

  int64_t get_lookup_row_bytes() const;

 private:
  enum class InitMode {
    Legacy,
    SeparateAlignments,
  };

  void init_impl(
      const std::vector<SerializedSepcType>& specs,
      int64_t logical_row_alignment,
      int64_t storage_row_alignment,
      int64_t scale_bias_size_in_bytes,
      const std::optional<at::Tensor>& hash_size_cumsum,
      InitMode mode);

  void check_initialized() const;
  // A wrapper without a planned HBM cache reads its members without copying
  // shared_ptrs, as on trunk; the owners are set only for a planned wrapper,
  // whose backend can be rebound while a request is in flight.
  struct BackendBinding {
    kv_mem::KVInferenceEmbeddingInterface<uint8_t>* backend;
    KVBackendMutationState* mutation;
    std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>>
        backend_owner;
    std::shared_ptr<KVBackendMutationState> mutation_owner;
    std::optional<int64_t> storage_row_bytes;
    bool uses_dram;
  };
  BackendBinding backend_binding(bool owned = false) const;
  bool recover_hbm_caches(const BackendBinding& binding);
  // for_cuda: pinned rows, zeroed first because the backend skips negative
  // keys and the rows are copied to the GPU and may be cached there.
  at::Tensor get_embeddings_cpu(
      const at::Tensor& indices,
      bool for_cuda,
      const BackendBinding& binding);
  struct CudaLookupCoverage {
    KVLookupCoverage host;
    at::Tensor device_backend_keys; // Set when only the device knows it.
  };
  at::Tensor get_embeddings_cuda(
      const at::Tensor& indices,
      const BackendBinding& binding,
      CudaLookupCoverage* coverage);
  at::Tensor get_embeddings_cuda_uncached(
      const at::Tensor& keys,
      const BackendBinding& binding);
  // Requires hbm_cache_mutex_.
  void own_hbm_cache_locked(const std::shared_ptr<KVHbmRowCache>& cache);
  std::shared_ptr<KVHbmRowCache> hbm_cache_for(
      const at::Tensor& indices,
      const BackendBinding& binding);
  // Rows as stored: drops the all-zero padding of rows padded to 16 bytes for
  // a KV-ZCH GPU serving shard, and checks they fit the storage.
  at::Tensor stored_rows(const at::Tensor& weights, int64_t backend_row_bytes)
      const;
  void set_embeddings_impl(
      const at::Tensor& indices,
      const at::Tensor& weights,
      std::optional<int64_t> inplace_update_ts_opt,
      const at::Tensor* value_versions,
      int64_t watermark,
      int64_t epoch);

  int64_t num_shards_ = 32;
  double uniform_init_lower_ = 0.0;
  double uniform_init_upper_ = 0.0;
  bool disable_random_init_ = false;

  std::shared_ptr<kv_mem::KVInferenceEmbeddingInterface<uint8_t>> kv_backend_;
  std::shared_ptr<KVBackendMutationState> backend_mutation_;
  bool uses_dram_backend_ = false;
  bool initialized_ = false;
  std::optional<int64_t> backend_storage_row_bytes_;
  int64_t storage_max_row_bytes_ = 0;
  int64_t lookup_max_row_bytes_ = 0;
  int64_t value_version_watermark_ = 0;
  std::optional<int64_t> value_version_epoch_;
  mutable std::mutex hbm_cache_mutex_;
  // Holds hbm_cache_mutex_ for a wrapper with planned HBM cache rows. Other
  // wrappers keep the unlocked backend access of wrappers without caches.
  std::unique_lock<std::mutex> lock_if_planned() const;
  std::shared_ptr<KVHbmRowCache> hbm_cache_;
  // The caches this wrapper looked up on its backend, one per device. The
  // registry only refers to caches, so a cache is freed with the last wrapper
  // that uses it.
  std::vector<std::shared_ptr<KVHbmRowCache>> owned_hbm_caches_;
  std::optional<int64_t> hbm_cache_rows_;
  // hbm_cache_rows_ > 0, read without the lock on the lookup and write paths.
  std::atomic<bool> planned_hbm_cache_{false};
  // Set by set_hbm_cache_rows, 0 rows included: only a KV-ZCH GPU serving
  // shard is planned, so only it takes CUDA-padded rows.
  std::atomic<bool> kv_gpu_serving_{false};
  bool values_immutable_ = false;
  std::atomic<bool> hbm_cache_fused_lookup_{true};
  std::atomic<bool> hbm_cache_fused_admission_{true};
  std::atomic<bool> hbm_cache_direct_output_{false};
  std::atomic<int64_t> random_init_bypasses_{0};
  std::atomic<int64_t> fill_written_rows_{0};
  std::atomic<int64_t> fill_existing_rows_{0};
};

} // namespace fbgemm_gpu

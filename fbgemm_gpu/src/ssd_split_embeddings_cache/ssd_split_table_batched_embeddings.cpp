/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include <torch/custom_class.h>

#include "./ssd_table_batched_embeddings.h"
#include "fbgemm_gpu/utils/ops_utils.h"

using namespace at;

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
ssd_cache_populate_actions_cuda(
    Tensor linear_indices,
    int64_t total_hash_size,
    Tensor lxu_cache_state,
    int64_t time_stamp,
    int64_t prefetch_dist,
    Tensor lru_state,
    bool gather_cache_stats,
    std::optional<Tensor> ssd_cache_stats,
    const bool lock_cache_line,
    const c10::optional<Tensor>& lxu_cache_locking_counter);

/// @ingroup embedding-ssd
///
/// @brief Similar to `torch.Tensor.index_put` but ignore `indices < 0`
///
/// `masked_index_put_cuda` only supports 2D input `values`. It puts
/// `count` rows in `values` into `self` using the row indices that
/// are >= 0 in `indices`.
///
/// ```python
/// # Equivalent PyTorch Python code
/// indices = indices[:count]
/// filter_ = indices >= 0
/// indices_ = indices[filter_]
/// self[indices_] = values[filter_.nonzero().flatten()]
/// ```
///
/// @param self The 2D output tensor (the tensor that is indexed)
/// @param indices The 1D index tensor
/// @param values The 2D input tensor
/// @param count The tensor that contains the length of `indices` to
///            process
/// @param use_pipeline A flag that indicates that this kernel will
///            overlap with other kernels. If it is true, then use a
///            fraction of SMs to reduce resource competition
/// @param preferred_sms The number of preferred SMs for the kernel to
///            use when use_pipeline=true. This value is ignored when
///            use_pipeline=false.
///
/// @return The `self` tensor
Tensor masked_index_put_cuda(
    Tensor self,
    Tensor indices,
    Tensor values,
    Tensor count,
    const bool use_pipeline,
    const int64_t preferred_sms);

/// @ingroup embedding-ssd
///
/// @brief Similar to `torch.index_select` but ignore `indices < 0`
///
/// `masked_index_select_cuda` only supports 2D input `values`. It
/// puts `count` rows that are specified in `indices` (where `indices`
/// >= 0) from `values` into `self`
///
/// ```python
/// # Equivalent PyTorch Python code
/// indices = indices[:count]
/// filter_ = indices >= 0
/// indices_ = indices[filter_]
/// self[filter_.nonzero().flatten()] = values[indices_]
/// ```
///
/// @param self The 2D output tensor
/// @param indices The 1D index tensor
/// @param values The 2D input tensor (the tensor that is indexed)
/// @param count The tensor that contains the length of `indices` to
///            process
/// @param use_pipeline A flag that indicates that this kernel will
///            overlap with other kernels. If it is true, then use a
///            fraction of SMs to reduce resource competition
///// @param preferred_sms The number of preferred SMs for the kernel to
///            use when use_pipeline=true. This value is ignored when
///            use_pipeline=false.
///
/// @return The `self` tensor
Tensor masked_index_select_cuda(
    Tensor self,
    Tensor indices,
    Tensor values,
    Tensor count,
    const bool use_pipeline,
    const int64_t preferred_sms);

Tensor masked_index_put_byte_cuda(
    Tensor self,
    Tensor indices,
    Tensor values,
    Tensor count);

/// @ingroup embedding-ssd
///
/// @brief Generate memory addresses for SSD TBE data
///
/// The data retrieved from SSD can be stored in either a scratch pad
/// (HBM) or LXU cache (also HBM). `lxu_cache_locations` is used to
/// specify the location of the data. If the location is -1, the data
/// for the associated index is in the scratch pad; otherwise, it is
/// in the cache. To enable TBE kernels to access the data
/// conveniently, this operator generates memory addresses of the
/// first byte for each index. When accessing data, a TBE kernel only
/// needs to convert addresses into pointers.
///
/// Moreover, this operator also generate the list of post backward
/// evicted indices which are basically the indices that their data
/// is in the scratch pad.
///
/// @param lxu_cache_locations The tensor that contains cache slots
///                            where data is stored for the *full* list
///                            of indices. -1 is a sentinel value that
///                            indicates that data is not in cache.
/// @param assigned_cache_slots The tensor that contains cache slots
///                             for the *unique* list of indices. -1
///                             indicates that data is not in cache
/// @param linear_index_inverse_indices The tensor that contains
///                                     the original position of
///                                     linear indices before being
///                                     sorted
/// @param unique_indices_count_cumsum The tensor that contains the
///                                    the exclusive prefix sum
///                                    results of the counts of unique
///                                    indices
/// @param cache_set_inverse_indices The tensor that contains the
///                                  original positions of cache sets
///                                  before being sorted
/// @param lxu_cache_weights The LXU cache tensor
/// @param inserted_ssd_weights The scratch pad tensor
/// @param unique_indices_length The tensor that contains the number
///                              of unique indices (GPU tensor)
/// @param cache_set_sorted_unique_indices The tensor that contains
///                                        associated unique indices
///                                        for the sorted unique cache
///                                        sets
///
/// @return A tuple of tensors (the SSD row address tensor and the
///         post backward evicted index tensor)
std::tuple<Tensor, Tensor> ssd_generate_row_addrs_cuda(
    const Tensor& lxu_cache_locations,
    const Tensor& assigned_cache_slots,
    const Tensor& linear_index_inverse_indices,
    const Tensor& unique_indices_count_cumsum,
    const Tensor& cache_set_inverse_indices,
    const Tensor& lxu_cache_weights,
    const Tensor& inserted_ssd_weights,
    const Tensor& unique_indices_length,
    const Tensor& cache_set_sorted_unique_indices);

/// @ingroup embedding-ssd
///
/// @brief Update memory addresses for SSD TBE data
///
/// When pipeline prefetching is enabled, data in a scratch pad of the
/// current iteration can be moved to L1 or a scratch pad of the next
/// iteration during the prefetch step. This operator updates the
/// memory addresses of data that is relocated to the correct
/// location.
///
/// @param ssd_row_addrs_curr The tensor that contains the row address
///            of the current iteration
/// @param inserted_ssd_weights_curr_next_map The tensor that contains
///            mapping between the location of each index in the
///            current iteration in the scratch pad of the next
///            iteration. (-1 = the data has not been moved).
///            inserted_ssd_weights_curr_next_map[i] is the location
//             of index i in the next iteration's scratch pad.
/// @param lxu_cache_locations_curr The tensor that contains cache
///            slots where data is stored for the *full* list of
///            indices for the current iteration. -1 is a sentinel
///            value that indicates that data is not in cache.
/// @param linear_index_inverse_indices_curr The tensor that contains
///            the original position of linear indices before being
///            sorted for the current iteration
/// @param unique_indices_count_cumsum_curr The tensor that contains
///            the the exclusive prefix sum results of the counts of
///            unique indices for the current iteration
/// @param cache_set_inverse_indices_curr The tensor that contains the
///            original positions of cache sets before being sorted
///            for the current iteration
/// @param lxu_cache_weights The LXU cache tensor
/// @param inserted_ssd_weights_next The scratch pad tensor for the
///            next iteration
/// @param unique_indices_length_curr The tensor that contains the
///            number of unique indices (GPU tensor) for the current
///            iteration
///
/// @return None
void ssd_update_row_addrs_cuda(
    const Tensor& ssd_row_addrs_curr,
    const Tensor& inserted_ssd_weights_curr_next_map,
    const Tensor& lxu_cache_locations_curr,
    const Tensor& linear_index_inverse_indices_curr,
    const Tensor& unique_indices_count_cumsum_curr,
    const Tensor& cache_set_inverse_indices_curr,
    const Tensor& lxu_cache_weights,
    const Tensor& inserted_ssd_weights_next,
    const Tensor& unique_indices_length_curr);

namespace {
class EmbeddingRocksDBWrapper : public torch::jit::CustomClassHolder {
 public:
  EmbeddingRocksDBWrapper(
      std::string path,
      int64_t num_shards,
      int64_t num_threads,
      int64_t memtable_flush_period,
      int64_t memtable_flush_offset,
      int64_t l0_files_per_compact,
      int64_t max_D,
      int64_t rate_limit_mbps,
      int64_t size_ratio,
      int64_t compaction_ratio,
      int64_t write_buffer_size,
      int64_t max_write_buffer_num,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t row_storage_bitwidth = 32,
      int64_t cache_size = 0,
      bool use_passed_in_path = false,
      int64_t tbe_unique_id = 0,
      int64_t l2_cache_size_gb = 0)
      : impl_(std::make_shared<ssd::EmbeddingRocksDB>(
            path,
            num_shards,
            num_threads,
            memtable_flush_period,
            memtable_flush_offset,
            l0_files_per_compact,
            max_D,
            rate_limit_mbps,
            size_ratio,
            compaction_ratio,
            write_buffer_size,
            max_write_buffer_num,
            uniform_init_lower,
            uniform_init_upper,
            row_storage_bitwidth,
            cache_size,
            use_passed_in_path,
            tbe_unique_id,
            l2_cache_size_gb)) {}

  void set_cuda(
      Tensor indices,
      Tensor weights,
      Tensor count,
      int64_t timestep,
      bool is_bwd) {
    return impl_->set_cuda(indices, weights, count, timestep, is_bwd);
  }

  void get_cuda(Tensor indices, Tensor weights, Tensor count) {
    return impl_->get_cuda(indices, weights, count);
  }

  void set(Tensor indices, Tensor weights, Tensor count) {
    return impl_->set(indices, weights, count);
  }

  void get(Tensor indices, Tensor weights, Tensor count) {
    return impl_->get(indices, weights, count);
  }

  std::vector<int64_t> get_mem_usage() {
    return impl_->get_mem_usage();
  }

  std::vector<double> get_rocksdb_io_duration(
      const int64_t step,
      const int64_t interval) {
    return impl_->get_rocksdb_io_duration(step, interval);
  }

  std::vector<double> get_l2cache_perf(
      const int64_t step,
      const int64_t interval) {
    return impl_->get_l2cache_perf(step, interval);
  }

  void compact() {
    return impl_->compact();
  }

  void flush() {
    return impl_->flush();
  }

 private:
  // shared pointer since we use shared_from_this() in callbacks.
  std::shared_ptr<ssd::EmbeddingRocksDB> impl_;
};

static auto embedding_rocks_db_wrapper =
    torch::class_<EmbeddingRocksDBWrapper>("fbgemm", "EmbeddingRocksDBWrapper")
        .def(
            torch::init<
                std::string,
                int64_t,
                int64_t,
                int64_t,
                int64_t,
                int64_t,
                int64_t,
                int64_t,
                int64_t,
                int64_t,
                int64_t,
                int64_t,
                double,
                double,
                int64_t,
                int64_t,
                bool,
                int64_t,
                int64_t>(),
            "",
            {
                torch::arg("path"),
                torch::arg("num_shards"),
                torch::arg("num_threads"),
                torch::arg("memtable_flush_period"),
                torch::arg("memtable_flush_offset"),
                torch::arg("l0_files_per_compact"),
                torch::arg("max_D"),
                torch::arg("rate_limit_mbps"),
                torch::arg("size_ratio"),
                torch::arg("compaction_ratio"),
                torch::arg("write_buffer_size"),
                torch::arg("max_write_buffer_num"),
                torch::arg("uniform_init_lower"),
                torch::arg("uniform_init_upper"),
                torch::arg("row_storage_bitwidth"),
                torch::arg("cache_size"),
                torch::arg("use_passed_in_path") = true,
                torch::arg("tbe_unique_id") = 0,
                torch::arg("l2_cache_size_gb") = 0,
            })
        .def(
            "set_cuda",
            &EmbeddingRocksDBWrapper::set_cuda,
            "",
            {
                torch::arg("indices"),
                torch::arg("weights"),
                torch::arg("count"),
                torch::arg("timestep"),
                torch::arg("is_bwd") = false,
            })
        .def("get_cuda", &EmbeddingRocksDBWrapper::get_cuda)
        .def("compact", &EmbeddingRocksDBWrapper::compact)
        .def("flush", &EmbeddingRocksDBWrapper::flush)
        .def("get_mem_usage", &EmbeddingRocksDBWrapper::get_mem_usage)
        .def(
            "get_rocksdb_io_duration",
            &EmbeddingRocksDBWrapper::get_rocksdb_io_duration)
        .def("get_l2cache_perf", &EmbeddingRocksDBWrapper::get_l2cache_perf)
        .def("set", &EmbeddingRocksDBWrapper::set)
        .def("get", &EmbeddingRocksDBWrapper::get);

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "masked_index_put("
      "    Tensor self, "
      "    Tensor indices, "
      "    Tensor values, "
      "    Tensor count, "
      "    bool use_pipeline=False, "
      "    int preferred_sms=-1"
      ") -> Tensor");
  DISPATCH_TO_CUDA("masked_index_put", masked_index_put_cuda);
  m.def(
      "masked_index_select("
      "    Tensor self, "
      "    Tensor indices, "
      "    Tensor values, "
      "    Tensor count, "
      "    bool use_pipeline=False, "
      "    int preferred_sms=-1"
      ") -> Tensor");
  DISPATCH_TO_CUDA("masked_index_select", masked_index_select_cuda);
  m.def(
      "ssd_cache_populate_actions("
      "    Tensor linear_indices, "
      "    int total_hash_size, "
      "    Tensor lxu_cache_state, "
      "    int time_stamp, "
      "    int prefetch_dist, "
      "    Tensor lru_state, "
      "    bool gather_cache_stats=False, "
      "    Tensor? ssd_cache_stats=None, "
      "    bool lock_cache_line=False, "
      "    Tensor? lxu_cache_locking_counter=None"
      ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  DISPATCH_TO_CUDA(
      "ssd_cache_populate_actions", ssd_cache_populate_actions_cuda);
  m.def(
      "ssd_generate_row_addrs("
      "    Tensor lxu_cache_locations, "
      "    Tensor assigned_cache_slots, "
      "    Tensor linear_index_inverse_indices, "
      "    Tensor unique_indices_count_cumsum, "
      "    Tensor cache_set_inverse_indices, "
      "    Tensor lxu_cache_weights, "
      "    Tensor inserted_ssd_weights, "
      "    Tensor unique_indices_length, "
      "    Tensor cache_set_sorted_unique_indices"
      ") -> (Tensor, Tensor)");
  DISPATCH_TO_CUDA("ssd_generate_row_addrs", ssd_generate_row_addrs_cuda);
  m.def(
      "ssd_update_row_addrs("
      "    Tensor ssd_row_addrs_curr, "
      "    Tensor inserted_ssd_weights_curr_next_map, "
      "    Tensor lxu_cache_locations_curr, "
      "    Tensor linear_index_inverse_indices_curr, "
      "    Tensor unique_indices_count_cumsum_curr, "
      "    Tensor cache_set_inverse_indices_curr, "
      "    Tensor lxu_cache_weights, "
      "    Tensor inserted_ssd_weights_next, "
      "    Tensor unique_indices_length_curr"
      ") -> ()");
  DISPATCH_TO_CUDA("ssd_update_row_addrs", ssd_update_row_addrs_cuda);
}
} // namespace

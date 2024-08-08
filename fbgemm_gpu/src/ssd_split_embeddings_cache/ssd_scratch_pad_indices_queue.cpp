/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/custom_class.h>
#include <memory>
#include <queue>
#include <unordered_map>
#include "kv_db_utils.h"

using namespace at;

namespace {

/// @ingroup embedding-ssd
///
/// @brief A class for SSD scratch pad index queue.
///
/// It is for storing scratch pad indices (conflict missed indices) from
/// previous iterations. It is used during the L1 cache prefetching
/// step: instead of fetching the missing indices directly from SSD, TBE
/// will lookup the scatch pad index queue first to check whether the
/// missing data is in the scratch pad from the previous iteration.
///
/// Note that this class only handles scratch pad indices. Scratch pad
/// data is stored outside of this class.
class SSDScratchPadIndicesQueueImpl
    : public std::enable_shared_from_this<SSDScratchPadIndicesQueueImpl> {
  using map_t = std::unordered_map<int64_t, int32_t>;

 public:
  explicit SSDScratchPadIndicesQueueImpl(const int64_t sentinel_val)
      : sentinel_val_(sentinel_val) {}

  void insert_cuda(const Tensor& indices, const Tensor& count) {
    // take reference to self to avoid lifetime issues.
    auto self = shared_from_this();
    std::function<void()>* functor =
        new std::function<void()>([=]() { self->insert(indices, count); });
    AT_CUDA_CHECK(cudaStreamAddCallback(
        at::cuda::getCurrentCUDAStream(),
        kv_db_utils::cuda_callback_func,
        functor,
        0));
  }

  void lookup_mask_and_pop_front_cuda(
      const Tensor& scratch_pad_locations,
      const Tensor& scratch_pad_indices,
      const Tensor& ssd_indices,
      const Tensor& count) {
    // take reference to self to avoid lifetime issues.
    auto self = shared_from_this();
    std::function<void()>* functor = new std::function<void()>([=]() {
      self->lookup_mask_and_pop_front(
          scratch_pad_locations, scratch_pad_indices, ssd_indices, count);
    });
    AT_CUDA_CHECK(cudaStreamAddCallback(
        at::cuda::getCurrentCUDAStream(),
        kv_db_utils::cuda_callback_func,
        functor,
        0));
  }

  int64_t size() {
    return index_loc_map_queue.size();
  }

 private:
  /// Inserts indices from a scratch pad that are not sentinel values
  /// into a hash map and then pushes the hash map into a queue. The
  /// key and value of each item in the hash map are an index and its
  /// location in the scratch pad, respectively.
  ///
  /// @param indices The 1D scratch pad index tensor
  /// @param count The tensor that contains the number of indices to
  ///              be processed
  ///
  /// @return None
  void insert(const Tensor& indices, const Tensor& count) {
    const auto count_ = count.item<int64_t>();
    TORCH_CHECK(indices.numel() >= count_);
    std::unique_ptr<map_t> map = std::make_unique<map_t>();
    const auto acc = indices.accessor<int64_t, 1>();
    for (const auto i : c10::irange(0, count_)) {
      const auto val = acc[i];
      if (val != sentinel_val_) {
        // val = index, i = location of the index
        map->insert({val, i});
      }
    }
    // Push hash map into the queue
    index_loc_map_queue.push(std::move(map));
  }

  /// Looks up `ssd_indices` in the front hash map in the queue. This
  /// is equivalent to looking up indices in the front scratch pad.
  ///
  /// If an index is found:
  ///
  /// - Sets the corresponding `scratch_pad_locations` to the location
  /// of the index in the scratch pad (the value in the hash map)
  ///
  /// - Sets the corresponding `ssd_indices` to the sentinel value
  /// (this is to prevent looking up this index from SSD)
  ///
  /// - Sets the `scratch_pad_indices` to the sentinel value (this is
  /// to prevent evicting the corresponding row from the scratch pad).
  ///
  /// Else: Sets the corresponding `scratch_pad_locations` to the
  /// sentinel value (to indicate that the index is not found in the
  /// scratch pad)
  ///
  /// Once the process above is done, pop the hash map from the queue.
  ///
  /// @param scratch_pad_locations The 1D output tensor that has the
  ///                              same size as `ssd_indices`. It
  ///                              contains locations of the
  ///                              corresponding indices in the
  ///                              scratch pad if they are found or
  ///                              sentinel values
  /// @param scratch_pad_indices The 1D tensor that contains scratch
  ///                            pad indices, i.e., conflict missed
  ///                            indices from the previous iteration.
  ///                            The indices and their locations must
  ///                            match the keys and values in the
  ///                            front hash map. After this function,
  ///                            the indices that are found will be
  ///                            set to sentinel values to prevent
  ///                            them from getting evicted
  /// @param ssd_indices The 1D tensor that contains indices that are
  ///                    missed from the L1 cache, i.e., all missed
  ///                    indices (including conflict misses). After
  ///                    this function, the indices that are found
  ///                    will be set to sentinel values to prevent
  ///                    them from being looked up in from SSD
  /// @param count The tensor that contains the number of indices to
  ///              be processed
  ///
  /// @return Outputs are passed by reference.
  void lookup_mask_and_pop_front(
      const Tensor& scratch_pad_locations,
      const Tensor& scratch_pad_indices,
      const Tensor& ssd_indices,
      const Tensor& count) {
    TORCH_CHECK(
        index_loc_map_queue.size() > 0,
        "index_loc_map_queue must not be empty");

    const auto count_ = count.item<int64_t>();
    TORCH_CHECK(ssd_indices.numel() >= count_);
    TORCH_CHECK(scratch_pad_locations.numel() == ssd_indices.numel());

    auto& map = index_loc_map_queue.front();

    auto loc_acc = scratch_pad_locations.accessor<int64_t, 1>();
    auto sp_acc = scratch_pad_indices.accessor<int64_t, 1>();
    auto ssd_acc = ssd_indices.accessor<int64_t, 1>();

    // Concurrent lookup is OK since it is read-only
    at::parallel_for(0, count_, 1, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        const auto val = ssd_acc[i];
        const auto val_loc = map->find(val);

        // If index is found in the map
        if (val_loc != map->end()) {
          const auto loc = val_loc->second;
          // Store the location in the scratch pad
          loc_acc[i] = loc;
          // Set the scratch pad index as the sentinel value to
          // prevent it from being evicted
          sp_acc[loc] = sentinel_val_;
          // Set the SSD index as the sentinel value to prevent it
          // from being looked up in SSD
          ssd_acc[i] = sentinel_val_;
        } else {
          // Set the location to the sentinel value to indicate that
          // the index is not found in the scratch pad
          loc_acc[i] = sentinel_val_;
        }
      }
    });

    // Remove the front hash map from the queue
    index_loc_map_queue.pop();
  }

  std::queue<std::unique_ptr<map_t>> index_loc_map_queue;
  const int64_t sentinel_val_;
};

class SSDScratchPadIndicesQueue : public torch::jit::CustomClassHolder {
 public:
  SSDScratchPadIndicesQueue(const int64_t sentinel_val)
      : impl_(std::make_shared<SSDScratchPadIndicesQueueImpl>(sentinel_val)) {}

  void insert_cuda(const Tensor& indices, const Tensor& count) {
    impl_->insert_cuda(indices, count);
  }

  void lookup_mask_and_pop_front_cuda(
      const Tensor& scratch_pad_locations,
      const Tensor& scratch_pad_indices,
      const Tensor& ssd_indices,
      const Tensor& count) {
    impl_->lookup_mask_and_pop_front_cuda(
        scratch_pad_locations, scratch_pad_indices, ssd_indices, count);
  }

  int64_t size() {
    return impl_->size();
  }

 private:
  std::shared_ptr<SSDScratchPadIndicesQueueImpl> impl_;
};

static auto index_loc_map_queue =
    torch::class_<SSDScratchPadIndicesQueue>(
        "fbgemm",
        "SSDScratchPadIndicesQueue")
        .def(torch::init<const int64_t>())
        .def("insert_cuda", &SSDScratchPadIndicesQueue::insert_cuda)
        .def(
            "lookup_mask_and_pop_front_cuda",
            &SSDScratchPadIndicesQueue::lookup_mask_and_pop_front_cuda)
        .def("size", &SSDScratchPadIndicesQueue::size);

} // namespace

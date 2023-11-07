/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/PeerToPeerAccess.h>

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/irange.h>
#include <torch/library.h>
#include <algorithm>
#include <tuple>
#include "fbgemm_gpu/merge_pooled_embeddings.h"

#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/topology_utils.h"

using Tensor = at::Tensor;

namespace {
struct DirectConnectedPeer {
  int64_t num_peer_links;
  int64_t peer_id;
  // number of transfers from peer
  int32_t peer_transfers;
};

struct TwoHopTransferContainer {
  Tensor intermediate_tensor;
  uint64_t output_idx;
  std::unique_ptr<at::cuda::CUDAEvent> transfer_cuda_event;
};

AdjacencyMatrix<Node> get_intermediate_node(
    const AdjacencyMatrix<Links>& links) {
  const auto world_size = at::cuda::getNumGPUs();
  std::vector<Node> link_vec(static_cast<size_t>(world_size * world_size));
  for (const auto i : c10::irange(world_size)) {
    for (const auto j : c10::irange(world_size)) {
      link_vec[i * world_size + j] = links(i, j);
    }
  }
  auto link_tensor = at::from_blob(
      link_vec.data(),
      {world_size, world_size},
      at::TensorOptions().dtype(at::kLong));
  LOG(INFO) << "NVLink Topology Matrix: \n" << link_tensor;
  std::vector<Node> assignments(
      static_cast<size_t>(world_size * world_size), -1);
  for (const auto dst_rank_id : c10::irange(world_size)) {
    std::vector<int> non_direct_src_ids;
    non_direct_src_ids.reserve(world_size);
    std::vector<DirectConnectedPeer> direct_connected_peers;
    direct_connected_peers.reserve(world_size);
    for (const auto src_rank_id : c10::irange(world_size)) {
      if (dst_rank_id == src_rank_id) {
        continue;
      }

      const auto num_peer_links = links(dst_rank_id, src_rank_id);
      if (num_peer_links > 0) {
        direct_connected_peers.push_back(
            {.num_peer_links = num_peer_links,
             .peer_id = src_rank_id,
             .peer_transfers = 1});
      } else {
        non_direct_src_ids.push_back(src_rank_id);
      }
    }

    // Assign intermediate hop ranks for non-directly connected peers.
    // Assigns intermediate hops based on the number of links from the
    //  potential intermediate rank to target rank, as well as
    //  the number of two_hop connections already assigned to the
    //  intermediate rank.
    for (const auto i : c10::irange(non_direct_src_ids.size())) {
      std::sort(
          direct_connected_peers.begin(),
          direct_connected_peers.end(),
          [](const auto& a, const auto& b) {
            if (a.num_peer_links > b.num_peer_links) {
              return true;
            } else if (a.num_peer_links == b.num_peer_links) {
              return a.peer_transfers < b.peer_transfers;
            } else {
              return false;
            }
          });
      const auto non_direct_src_id = non_direct_src_ids.at(i);
      for (auto& j : direct_connected_peers) {
        const auto potential_hop_id = j.peer_id;
        const auto potential_hop_peer_links =
            links(potential_hop_id, non_direct_src_id);
        if (potential_hop_peer_links > 0) {
          assignments[dst_rank_id * world_size + non_direct_src_id] =
              potential_hop_id;
          j.peer_transfers += 1;
          break;
        }
      }
    }
  }
  if (std::any_of(assignments.begin(), assignments.end(), [](Node n) {
        return n != -1;
      })) {
    auto tensor = at::from_blob(
        assignments.data(),
        {world_size, world_size},
        at::TensorOptions().dtype(at::kLong));
    LOG(INFO) << "Detected a multi-hop NVLink configuration: \n" << tensor;
    return
        [=](Node src, Node dst) { return assignments[dst * world_size + src]; };
  } else {
    return [](Node, Node) { return -1; };
  }
}

// Tensors in `output_tensors` should all be on target_device. We copy the
// tensor in the same index from `input_tensors` to `output_tensors`. If the
// tensor in `input_tensors` is already in the `target_device`, we will skip
// copy it if `skip_if_same_device` is true.
void all_to_one(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors,
    at::Device target_device,
    bool skip_if_same_device) {
  auto num_gpus = at::cuda::getNumGPUs();
  // Create static thread local CUDAEvent as creating/destroying CUDAEvents
  // can be expensive. In one call to this function, we use events created on
  // the target device. Since target device can be different across each call,
  // we store all the events in a 2-dimentionsal vector.
  using PerDeviceEventList = std::vector<at::cuda::CUDAEvent>;
  static thread_local std::vector<PerDeviceEventList> copy_begin_events;
  static thread_local std::vector<PerDeviceEventList> copy_completion_events;
  static thread_local std::once_flag flag1;
  std::call_once(flag1, [num_gpus]() {
    for (auto i = 0; i < num_gpus; i++) {
      copy_begin_events.push_back(PerDeviceEventList(num_gpus));
      copy_completion_events.push_back(PerDeviceEventList(num_gpus));
    }
  });

  auto target_device_index = target_device.index();
  TORCH_CHECK(
      target_device_index != -1,
      "target_device.index() is -1. Please pass target_device with device "
      "index, e.g., torch.device(\"cuda:0\")")

  TORCH_CHECK(target_device_index < num_gpus);

  std::vector<TwoHopTransferContainer> two_hop_transfers;
  two_hop_transfers.reserve(input_tensors.size());
  std::vector<bool> is_two_hop_transfer;
  is_two_hop_transfer.reserve(input_tensors.size());

  static auto intermediate_nodes =
      get_intermediate_node(fbgemm_gpu::get_nvlink_matrix());
  for (const auto i : c10::irange(input_tensors.size())) {
    const auto& src = input_tensors.at(i);
    Node src_device_id = src.get_device();
    auto intermediate_node =
        intermediate_nodes(src_device_id, target_device_index);
    if (intermediate_node != -1) {
      two_hop_transfers.push_back(
          {.intermediate_tensor = at::empty(
               src.sizes(),
               src.options().device(at::Device(at::kCUDA, intermediate_node))),
           .output_idx = i,
           .transfer_cuda_event =
               std::make_unique<at::cuda::CUDAEvent>(cudaEventDisableTiming)});
      auto& dst = two_hop_transfers.back().intermediate_tensor;
      at::cuda::CUDAStream copy_stream =
          at::cuda::getCurrentCUDAStream(src_device_id);
      AT_CUDA_CHECK(cudaMemcpy2DAsync(
          dst.data_ptr(),
          dst.stride(0) * dst.element_size(),
          src.data_ptr(),
          src.stride(0) * src.element_size(),
          src.size(1) * src.element_size(),
          src.size(0),
          cudaMemcpyDeviceToDevice,
          copy_stream));
      two_hop_transfers.back().transfer_cuda_event->record(copy_stream);
      is_two_hop_transfer.push_back(true);
    } else {
      is_two_hop_transfer.push_back(false);
    }
  }

  // For each source device directly connected to the destination device, we
  // sync its current stream and launch all the copies that are from that
  // device.
  for (const auto device_id : c10::irange(num_gpus)) {
    auto src_device = at::Device(at::kCUDA, device_id);
    if (src_device == target_device) {
      continue;
    }

    // synchronize source streams and launch copies on source stream.
    at::cuda::CUDAGuard device_guard(src_device);
    // We always perform the copy on the source device, using the current
    // stream on the source device, and we fully synchronize on both src and
    // dst's current streams for completion of the copy. We have to explicitly
    // do this for non-contig copies. This mimics the behavior of cross-device
    // cudaMemcpyAsync on the default stream.

    at::cuda::CUDAStream copy_stream =
        at::cuda::getCurrentCUDAStream(device_id);
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)
    auto& dst_ready = copy_begin_events[target_device_index][device_id];
    device_guard.set_device(target_device);
    dst_ready.record(at::cuda::getCurrentCUDAStream(target_device_index));
    device_guard.set_device(src_device);
    dst_ready.block(copy_stream);
    for (const auto i : c10::irange(input_tensors.size())) {
      const auto metadata = is_two_hop_transfer.at(i);
      // Initiate all transfer for tensors with direct
      // NVLink connection to target rank
      if (metadata) {
        continue;
      }

      auto& src = input_tensors[i];
      if (src.device() != src_device) {
        continue;
      }

      auto& dst = output_tensors[i];
      // on source device, launch memcpy.
      AT_CUDA_CHECK(cudaMemcpy2DAsync(
          dst.data_ptr(),
          dst.stride(0) * dst.element_size(),
          src.data_ptr(),
          src.stride(0) * src.element_size(),
          src.size(1) * src.element_size(),
          src.size(0),
          cudaMemcpyDeviceToDevice,
          copy_stream));
    }
  }

  // Complete 2-hop transfers to target rank
  for (auto& two_hop_transfer : two_hop_transfers) {
    const auto& src = two_hop_transfer.intermediate_tensor;
    const auto src_device_id = src.get_device();
    const auto src_device = at::Device(at::kCUDA, src_device_id);
    if (src_device == target_device) {
      continue;
    }

    // intermediate rank
    at::cuda::CUDAGuard device_guard(src_device);
    // intermediate rank stream
    at::cuda::CUDAStream copy_stream =
        at::cuda::getCurrentCUDAStream(src_device_id);
    // wait on first hop transfer
    two_hop_transfer.transfer_cuda_event->block(copy_stream);
    // synchronize with target rank
    auto& dst_ready = copy_begin_events[target_device_index][src_device_id];
    device_guard.set_device(target_device);
    dst_ready.record(at::cuda::getCurrentCUDAStream(target_device_index));
    device_guard.set_device(src_device);
    dst_ready.block(copy_stream);
    // originating tensor output position
    const auto output_index = two_hop_transfer.output_idx;
    auto& dst = output_tensors.at(output_index);
    // on source device, launch memcpy.
    AT_CUDA_CHECK(cudaMemcpy2DAsync(
        dst.data_ptr(),
        dst.stride(0) * dst.element_size(),
        src.data_ptr(),
        src.stride(0) * src.element_size(),
        src.size(1) * src.element_size(),
        src.size(0),
        cudaMemcpyDeviceToDevice,
        copy_stream));
  }

  // Do the same-GPU cases.
  if (!skip_if_same_device) {
    for (const auto i : c10::irange(input_tensors.size())) {
      auto& src = input_tensors[i];
      if (src.device() == target_device) {
        auto& dst = output_tensors[i];
        // single device memcpy, not that src_device == dst_device.
        at::cuda::CUDAStream copy_stream =
            at::cuda::getCurrentCUDAStream(target_device_index);
        AT_CUDA_CHECK(cudaMemcpy2DAsync(
            dst.data_ptr(),
            dst.stride(0) * dst.element_size(),
            src.data_ptr(),
            src.stride(0) * src.element_size(),
            src.size(1) * src.element_size(),
            src.size(0),
            cudaMemcpyDeviceToDevice,
            copy_stream));
      }
    }
  }

  // wait for cross-device copies to complete.
  for (const auto device_id : c10::irange(num_gpus)) {
    if (device_id != target_device_index) {
      auto src_device = at::Device(at::kCUDA, device_id);
      // Still on src_device, record stream event
      at::cuda::CUDAGuard device_guard(src_device);
      at::cuda::CUDAStream copy_stream =
          at::cuda::getCurrentCUDAStream(device_id);

      auto& src_ready = copy_completion_events[target_device_index][device_id];
      src_ready.record(copy_stream);

      device_guard.set_device(target_device);
      src_ready.block(at::cuda::getCurrentCUDAStream(target_device_index));
    }
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

Tensor sum_reduce_to_one(
    std::vector<Tensor> input_tensors,
    at::Device target_device) {
  auto num_gpus = at::cuda::getNumGPUs();
  // Create static thread local CUDAEvent as creating/destroying CUDAEvents
  // can be expensive. In one call to this function, we use events created on
  // the target device. Since target device can be different across each call,
  // we store all the events in a 2-dimentionsal vector.
  using PerDeviceEventList = std::vector<at::cuda::CUDAEvent>;
  static thread_local std::vector<PerDeviceEventList> copy_completion_events;
  static thread_local std::once_flag flag1;
  std::call_once(flag1, [num_gpus]() {
    for (auto i = 0; i < num_gpus; i++) {
      copy_completion_events.push_back(PerDeviceEventList(num_gpus));
    }
  });

  auto target_device_index = target_device.index();
  TORCH_CHECK(target_device_index < num_gpus && target_device_index >= 0);

  // Local reduction for tensors residing the same GPU.
  // And if there's a tensor already in target device, use it for output tensor.
  Tensor output_tensor;
  for (const auto i : c10::irange(input_tensors.size())) {
    auto& ten = input_tensors[i];
    if (!ten.has_storage()) {
      continue;
    }
    TENSOR_ON_CUDA_GPU(ten);
    if (ten.device() == target_device && !output_tensor.has_storage()) {
      output_tensor = ten;
    }
    for (auto j = i + 1; j < input_tensors.size(); ++j) {
      if (input_tensors[j].has_storage() &&
          ten.device() == input_tensors[j].device()) {
        ten.add_(input_tensors[j]);
        // Replace with a dummy tensor without storage to mark reduced away
        input_tensors[j] = Tensor();
      }
    }
  }

  // First copy from GPUs that are in 2-hop distance to their intermediate
  // GPUs.
  static auto intermediate_nodes =
      get_intermediate_node(fbgemm_gpu::get_nvlink_matrix());
  std::vector<Tensor> copied_tensors(input_tensors.size());
  for (const auto i : c10::irange(input_tensors.size())) {
    auto& src = input_tensors[i];
    if (!src.has_storage()) {
      continue;
    }
    auto intermediate_node =
        intermediate_nodes(src.get_device(), target_device_index);
    if (intermediate_node == -1) {
      continue;
    }
    auto intermediate_device = at::Device(at::kCUDA, intermediate_node);
    Tensor dst = at::empty_like(src, intermediate_device);

    // This is a cross-device copy on the src current stream and dst current
    // stream.
    // Unlike all_to_one case, we don't need to wait for dst ready to worry
    // about write-after-write and write-after-read dependencies because we're
    // creating a temp tensor, dst.

    at::cuda::CUDAGuard device_guard(src.device());
    // on source device, launch memcpy.
    AT_CUDA_CHECK(cudaMemcpy2DAsync(
        dst.data_ptr(),
        dst.stride(0) * dst.element_size(),
        src.data_ptr(),
        src.stride(0) * src.element_size(),
        src.size(1) * src.element_size(),
        src.size(0),
        cudaMemcpyDeviceToDevice,
        at::cuda::getCurrentCUDAStream(src.get_device())));
    copied_tensors[i] = dst;
  }

  // Wait for cross-device copies to complete, then reduce
  for (const auto device_id : c10::irange(num_gpus)) {
    auto intermediate_node = intermediate_nodes(device_id, target_device_index);
    if (intermediate_node == -1) {
      continue;
    }
    auto intermediate_device = at::Device(at::kCUDA, intermediate_node);

    auto src_device = at::Device(at::kCUDA, device_id);
    // Still on src_device, record stream event
    at::cuda::CUDAGuard device_guard(src_device);
    at::cuda::CUDAStream copy_stream =
        at::cuda::getCurrentCUDAStream(device_id);

    auto& src_ready = copy_completion_events[target_device_index][device_id];
    src_ready.record(copy_stream);

    device_guard.set_device(intermediate_device);
    src_ready.block(at::cuda::getCurrentCUDAStream(intermediate_node));

    // Find any tensor in the intermediate GPU to reduce to.
    Tensor ten_at_intermediate_node;
    for (const auto i : c10::irange(input_tensors.size())) {
      if (input_tensors[i].has_storage() &&
          input_tensors[i].device() == intermediate_device) {
        ten_at_intermediate_node = input_tensors[i];
        break;
      }
    }

    for (const auto i : c10::irange(copied_tensors.size())) {
      auto& ten = copied_tensors[i];
      if (!ten.has_storage() || ten.device() != intermediate_device ||
          !input_tensors[i].has_storage() ||
          input_tensors[i].device() != src_device) {
        continue;
      }
      if (ten_at_intermediate_node.has_storage()) {
        ten_at_intermediate_node.add_(ten);
        input_tensors[i] = Tensor();
      } else {
        // No tensor to reduce to, so we just replace input_tensors[i] with
        // the version copied to the intermediate GPU.
        input_tensors[i] = ten;
      }
    }
  }

  // Final hop.
  for (const auto i : c10::irange(input_tensors.size())) {
    auto& src = input_tensors[i];
    if (!src.has_storage() || src.device() == target_device) {
      continue;
    }

    Tensor dst = at::empty_like(src, target_device);

    at::cuda::CUDAGuard device_guard(src.device());
    AT_CUDA_CHECK(cudaMemcpy2DAsync(
        dst.data_ptr(),
        dst.stride(0) * dst.element_size(),
        src.data_ptr(),
        src.stride(0) * src.element_size(),
        src.size(1) * src.element_size(),
        src.size(0),
        cudaMemcpyDeviceToDevice,
        at::cuda::getCurrentCUDAStream(src.get_device())));
    copied_tensors[i] = dst;
  }

  // Wait for cross-device copies to complete, then reduce
  for (const auto device_id : c10::irange(num_gpus)) {
    if (device_id != target_device_index) {
      auto src_device = at::Device(at::kCUDA, device_id);
      // Still on src_device, record stream event
      at::cuda::CUDAGuard device_guard(src_device);
      at::cuda::CUDAStream copy_stream =
          at::cuda::getCurrentCUDAStream(device_id);

      auto& src_ready = copy_completion_events[target_device_index][device_id];
      src_ready.record(copy_stream);

      device_guard.set_device(target_device);
      src_ready.block(at::cuda::getCurrentCUDAStream(target_device_index));

      for (const auto i : c10::irange(input_tensors.size())) {
        auto& src = input_tensors[i];
        if (!src.has_storage() || src.device() != src_device) {
          continue;
        }

        if (output_tensor.has_storage()) {
          output_tensor.add_(copied_tensors[i]);
        } else {
          // Very first reduction at the target device is just a shallow copy.
          output_tensor = copied_tensors[i];
        }
      }
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output_tensor;
}

std::tuple<std::array<int64_t, 2>, std::vector<int64_t>, int64_t>
cat_dim_2d_output_shape(
    std::vector<Tensor>& tensors,
    int64_t uncat_dim_size,
    int64_t cat_dim) {
  TORCH_CHECK(!tensors.empty());

  // only support 2d tensor concatenation.
  TORCH_CHECK(cat_dim >= 0 && cat_dim <= 1);

  int64_t total_cat_dim = 0;
  std::vector<int64_t> cumulative_dims;
  cumulative_dims.push_back(0);
  for (const auto& t : tensors) {
    TORCH_CHECK(t.dim() == 2);
    // only support two-dimension tensors.
    TORCH_CHECK(t.size(1 - cat_dim) == uncat_dim_size);
    total_cat_dim += t.size(cat_dim);
    cumulative_dims.push_back(total_cat_dim);
  }

  // default shape for concatenating on dim 1
  std::array<int64_t, 2> output_shape;
  if (cat_dim == 0) {
    output_shape = {total_cat_dim, uncat_dim_size};
  } else {
    output_shape = {uncat_dim_size, total_cat_dim};
  }

  return std::make_tuple(output_shape, cumulative_dims, total_cat_dim);
}

Tensor cat_dim_2d(
    std::vector<Tensor>& tensors,
    int64_t uncat_dim_size,
    at::Device output_device,
    int64_t cat_dim = 1) {
  if (tensors.size() == 0) {
    return at::empty({0}, at::TensorOptions().device(output_device));
  }
  // only support 2d tensor concatenation.
  auto [output_shape, cumulative_dims, total_cat_dim] =
      cat_dim_2d_output_shape(tensors, uncat_dim_size, cat_dim);

  auto* prop = at::cuda::getCurrentDeviceProperties();
  auto output =
      at::empty(output_shape, tensors.front().options().device(output_device));
  TORCH_CHECK(
      output.stride(0) * output.element_size() <=
      static_cast<int64_t>(prop->memPitch));
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(tensors.size());

  for (const auto i : c10::irange(tensors.size())) {
    output_tensors.push_back(
        output.slice(cat_dim, cumulative_dims[i], cumulative_dims[i + 1]));
  }
  all_to_one(
      tensors, output_tensors, output_device, /* skip_if_same_device */ false);

  return output;
}

void init_p2p_access() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    for (const auto i : c10::irange(at::cuda::getNumGPUs())) {
      for (const auto j : c10::irange(at::cuda::getNumGPUs())) {
        if (i != j) {
          AT_ASSERT(at::cuda::get_p2p_access(i, j));
        }
      }
    }
  });
}

} // namespace

namespace fbgemm_gpu {

Tensor merge_pooled_embeddings(
    std::vector<Tensor> pooled_embeddings,
    int64_t uncat_dim_size,
    at::Device target_device,
    int64_t cat_dim = 1) {
  init_p2p_access();
  at::cuda::CUDAGuard g(target_device);

  TORCH_CHECK(!pooled_embeddings.empty());
  return cat_dim_2d(pooled_embeddings, uncat_dim_size, target_device, cat_dim);
}

std::vector<Tensor> all_to_one_device(
    std::vector<Tensor> input_tensors,
    at::Device target_device) {
  init_p2p_access();
  at::cuda::CUDAGuard g(target_device);

  std::vector<Tensor> output_tensors;
  output_tensors.reserve(input_tensors.size());

  for (const auto& tensor : input_tensors) {
    TORCH_CHECK(tensor.is_cuda());
    output_tensors.push_back(
        tensor.device() != target_device
            ? at::empty(tensor.sizes(), tensor.options().device(target_device))
            : tensor);
  }
  all_to_one(
      input_tensors,
      output_tensors,
      target_device,
      /* skip_if_same_device */ true);
  return output_tensors;
}

Tensor sum_reduce_to_one_device(
    std::vector<Tensor> input_tensors,
    at::Device target_device) {
  TORCH_CHECK(input_tensors.size() > 0, "reducing no tensor is undefined");

  init_p2p_access();

  return sum_reduce_to_one(input_tensors, target_device);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CUDA(
      "merge_pooled_embeddings", fbgemm_gpu::merge_pooled_embeddings);
  DISPATCH_TO_CUDA("all_to_one_device", fbgemm_gpu::all_to_one_device);
  DISPATCH_TO_CUDA("sum_reduce_to_one", fbgemm_gpu::sum_reduce_to_one_device);
}

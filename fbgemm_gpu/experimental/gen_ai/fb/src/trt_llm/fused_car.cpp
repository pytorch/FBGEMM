// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime_api.h>

#if !defined(USE_ROCM)
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 12040)
#define ENABLE_TRT_LLM_CAR
#endif
#endif

#include <ATen/ATen.h>
#include <torch/library.h>
#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

#ifdef ENABLE_TRT_LLM_CAR
#include <NvInferRuntime.h>
#include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"

namespace tk = tensorrt_llm::kernels;
namespace tu = tensorrt_llm::utils;

constexpr int MAX_RANKS_PER_NODE = 8;
constexpr size_t FLAGS_SIZE =
    (tk::MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t);
constexpr size_t MAX_ALL_REDUCE_BUFF_SIZE = 8192 * 8192 * sizeof(float);

enum AllReduceBuff {
  DATA_0 = 0,
  BARRIER_0 = 1,
  BARRIER_1 = 2,
  LAMPORT_0 = 3,
  LAMPORT_1 = 4,
  LAMPORT_2 = 5,
  NUM_BUFFS = 6,
};
#endif // ifdef ENABLE_TRT_LLM_CAR

namespace fbgemm_gpu {

#ifdef ENABLE_TRT_LLM_CAR
namespace {

struct TrtLlmCustomAllReduceState {
  std::vector<std::vector<at::Tensor>> buffers;

  tk::AllReduceParams params;
};

TrtLlmCustomAllReduceState* get_car_state() {
  static auto* r = new TrtLlmCustomAllReduceState();
  return r;
}

constexpr size_t align_size(size_t size, size_t to) {
  if ((size % to) != 0U) {
    size += to - size % to;
  }
  return size;
}

bool can_access_peer(int rank, int world_size) {
  const auto src_device = at::cuda::current_device();
  TORCH_CHECK(rank == src_device);
  for (auto dst_device = 0; dst_device < world_size; dst_device++) {
    if (src_device == dst_device) {
      continue;
    }

    int can_access_peer = 0;
    TLLM_CUDA_CHECK(
        cudaDeviceCanAccessPeer(&can_access_peer, rank, dst_device));
    if (can_access_peer == 0) {
      return false;
    }
  }
  return true;
}

void set_params(
    tk::AllReduceParams& params,
    int world_size,
    const std::vector<std::vector<at::Tensor>>& buffers) {
  for (int i = 0; i < world_size; ++i) {
    params.peer_comm_buffer_ptrs[i] = buffers[i][DATA_0].data_ptr();
    params.fusion_params.lamport_peer_comm_buffer_ptrs[i] =
        buffers[i][LAMPORT_0].data_ptr();
    params.fusion_params.lamport_peer_comm_buffer_ptrs[i + MAX_RANKS_PER_NODE] =
        buffers[i][LAMPORT_1].data_ptr();
    params.fusion_params
        .lamport_peer_comm_buffer_ptrs[i + MAX_RANKS_PER_NODE * 2] =
        buffers[i][LAMPORT_2].data_ptr();
  }
  for (int i = 0; i < world_size; ++i) {
    params.peer_barrier_ptrs_in[i] =
        reinterpret_cast<uint32_t*>(buffers[i][BARRIER_0].data_ptr());
  }
  for (int i = 0; i < world_size; ++i) {
    params.peer_barrier_ptrs_out[i] =
        reinterpret_cast<uint32_t*>(buffers[i][BARRIER_1].data_ptr());
  }
}

} // namespace
#endif // ifdef ENABLE_TRT_LLM_CAR

std::tuple<at::Tensor, at::Tensor, at::Tensor>
fused_one_shot_allreduce_residual_rms_norm_allocate_buffers(
    at::Device device,
    int64_t world_size,
    int64_t max_num_seqs,
    int64_t hidden_size) {
#ifdef ENABLE_TRT_LLM_CAR
  const auto local_buffer_size =
      static_cast<std::size_t>(max_num_seqs) * hidden_size * sizeof(float);

  TORCH_CHECK(
      local_buffer_size <= MAX_ALL_REDUCE_BUFF_SIZE,
      "The buffer size (",
      local_buffer_size,
      ") is larger than the max allowed buffer size (",
      MAX_ALL_REDUCE_BUFF_SIZE,
      ")");

  const auto buffer_size = world_size * local_buffer_size;
  size_t real_hidden_size = world_size * hidden_size;

  // PUSH_MODE need TP_SIZE times the activation tensor size
  // kLamportTokenNumThreshold = 16
  auto const lamport_buffer_size = world_size *
      tk::reduce_fusion::details::kLamportTokenNumThreshold * real_hidden_size *
      sizeof(half);

  // FLAGS_SIZE = (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t)
  // MAX_ALL_REDUCE_BLOCKS = 24
  auto const flags_size = FLAGS_SIZE * world_size * 2;

  const int64_t total_ipc_buffer_size =
      buffer_size + (flags_size * 2) + (lamport_buffer_size * 3);
  const int64_t total_ipc_aligned_buffer_size =
      align_size(total_ipc_buffer_size, 1LU << 21);

  // Use cudaMalloc to avoid caching allocator handing out wrong base poitner
  void* buffer_ptr{nullptr};
  C10_CUDA_CHECK(cudaMalloc(&buffer_ptr, total_ipc_aligned_buffer_size));
  auto buffer = at::from_blob(
      buffer_ptr,
      {total_ipc_aligned_buffer_size},
      at::TensorOptions().dtype(at::kByte).device(device));

  // Initialize buffer with zeros
  buffer.fill_(0);

  // Get IPC handle
  auto buffer_handle = at::empty(
      {sizeof(cudaIpcMemHandle_t)},
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  cudaIpcMemHandle_t handle;
  TLLM_CUDA_CHECK(cudaIpcGetMemHandle(&handle, buffer.data_ptr()));
  std::memcpy(buffer_handle.data_ptr(), &handle, sizeof(cudaIpcMemHandle_t));

  // Compute offsets
  auto buffer_offsets = at::empty(
      {NUM_BUFFS + 1}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  auto* buffer_offsets_ptr = buffer_offsets.data_ptr<int64_t>();
  int64_t offset = 0;
  int i = 0;
  for (auto size :
       {buffer_size,
        flags_size,
        flags_size,
        lamport_buffer_size,
        lamport_buffer_size,
        lamport_buffer_size}) {
    buffer_offsets_ptr[i++] = offset;
    offset += size;
  }
  buffer_offsets_ptr[i] = offset;
  return {buffer, buffer_offsets, buffer_handle};
#else
  TORCH_CHECK(
      false,
      "fused_one_shot_allreduce_residual_rms_norm_allocate_buffers does not "
      "support CUDA version < 12.4");
  return {at::empty({1}), at::empty({1}), at::empty({1})};
#endif
}

void fused_one_shot_allreduce_residual_rms_norm_init(
    int64_t rank,
    int64_t world_size,
    int64_t hidden_size,
    const at::Tensor& local_buffer,
    const std::vector<at::Tensor>& all_buffer_handles,
    const at::Tensor& buffer_offsets) {
#ifdef ENABLE_TRT_LLM_CAR
  TORCH_CHECK(can_access_peer(rank, world_size));
  TORCH_CHECK(all_buffer_handles.size() == world_size);

  auto* state = get_car_state();
  auto& all_buffers = state->buffers;
  auto* buff_offsets = buffer_offsets.data_ptr<int64_t>();
  for (auto r = 0; r < world_size; r++) {
    uint8_t* buff{nullptr};
    if (r == rank) {
      buff = reinterpret_cast<uint8_t*>(local_buffer.data_ptr());
    } else {
      cudaIpcMemHandle_t handle;
      std::memcpy(
          &handle,
          all_buffer_handles[r].data_ptr(),
          sizeof(cudaIpcMemHandle_t));
      TLLM_CUDA_CHECK(cudaIpcOpenMemHandle(
          reinterpret_cast<void**>(&buff),
          handle,
          cudaIpcMemLazyEnablePeerAccess));
    }
    std::vector<at::Tensor> buffers;
    for (int i = 0; i < buffer_offsets.numel() - 1; i++) {
      auto ipc_tensor = at::from_blob(
          buff + buff_offsets[i],
          {buff_offsets[i + 1] - buff_offsets[i]},
          local_buffer.options());
      buffers.push_back(std::move(ipc_tensor));
    }
    all_buffers.push_back(std::move(buffers));
  }

  all_buffers[rank][BARRIER_0].view(at::kInt).fill_(0);
  all_buffers[rank][BARRIER_1].view(at::kInt).fill_(0);

  for (int i = LAMPORT_0; i <= LAMPORT_2; i++) {
    // Lamport-style kernel only supports FP16 and BF16
    tk::lamportInitialize(
        all_buffers[rank][i].data_ptr(),
        all_buffers[rank][i].numel() / sizeof(__nv_bfloat16),
        nvinfer1::DataType::kBF16,
        at::cuda::getCurrentCUDAStream());
  }

  auto& params = state->params;
  set_params(params, world_size, all_buffers);
  params.barrier_flag = 0;
  params.ranks_per_node = world_size;
  params.local_rank = rank;
  params.fusion_params.hidden_size = hidden_size;
#else
  TORCH_CHECK(
      false,
      "fused_one_shot_allreduce_residual_rms_norm_init does not support CUDA "
      "version < 12.4");
#endif // ifdef ENABLE_TRT_LLM_CAR
}

at::Tensor fused_one_shot_allreduce_residual_rms_norm(
    const at::Tensor& input,
    const at::Tensor& residual,
    double eps,
    const std::optional<at::Tensor>& affine) {
#ifdef ENABLE_TRT_LLM_CAR
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input, residual, affine);

  TENSORS_HAVE_SAME_TYPE(input, residual);
  const auto ndims = input.ndimension();
  TENSOR_NDIM_EQUALS(residual, ndims);
  for (auto d = 0; d < ndims; d++) {
    const auto idim = input.size(d);
    TORCH_CHECK(residual.size(d) == idim);
  }

  auto state = get_car_state();
  auto& params = state->params;

  if (affine.has_value()) {
    auto affine_ = affine.value();
    TENSORS_HAVE_SAME_TYPE(input, affine_);
    TORCH_CHECK(affine_.numel() == params.fusion_params.hidden_size);
  }

  const int message_size = input.numel();
  TORCH_CHECK(message_size % params.fusion_params.hidden_size == 0);

  auto output = at::empty_like(input);
  auto inter = at::empty_like(input);

  params.local_output_buffer_ptr = output.data_ptr();
  params.local_input_buffer_ptr = input.data_ptr();
  params.elts_total = message_size;
  params.fusion_params.bias_buffer = nullptr;
  params.fusion_params.residual_buffer = residual.data_ptr();
  params.fusion_params.weight_buffer =
      affine.has_value() ? affine->data_ptr() : nullptr;
  params.fusion_params.intermediate_buffer = inter.data_ptr();
  params.fusion_params.eps = eps;

  // Increment the barrier flag (important for correctness)
  params.barrier_flag += 1;

  const auto torch_dtype = input.scalar_type();
  TORCH_CHECK(
      torch_dtype == at::ScalarType::BFloat16 ||
      torch_dtype == at::ScalarType::Half ||
      torch_dtype == at::ScalarType::Float);

  nvinfer1::DataType dtype;
  if (torch_dtype == at::ScalarType::BFloat16) {
    dtype = nvinfer1::DataType::kBF16;
  } else if (torch_dtype == at::ScalarType::Half) {
    dtype = nvinfer1::DataType::kHALF;
  } else {
    dtype = nvinfer1::DataType::kFLOAT;
  }

  constexpr auto st = tk::AllReduceStrategyType::ONESHOT;
  constexpr auto st_config = tk::AllReduceStrategyConfig::PUSH_MODE;
  constexpr auto fusion_op = tk::AllReduceFusionOp::RESIDUAL_RMS_NORM;

  tk::customAllReduce(
      params,
      dtype,
      st,
      st_config,
      fusion_op,
      at::cuda::getCurrentCUDAStream());

  return output;
#else
  TORCH_CHECK(
      false,
      "fused_one_shot_allreduce_residual_rms_norm does not support CUDA version "
      "< 12.4");
  return at::empty({1});
#endif // ENABLE_TRT_LLM_CAR
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "_fused_one_shot_allreduce_residual_rms_norm("
      "  Tensor input, "
      "  Tensor residual, "
      "  float eps, "
      "  Tensor? affine=None"
      ") -> Tensor");
  m.def(
      "_fused_one_shot_allreduce_residual_rms_norm_allocate_buffers("
      "  Device device, "
      "  int world_size, "
      "  int max_num_seqs, "
      "  int hidden_size"
      ") -> (Tensor, Tensor, Tensor)");
  m.impl(
      "_fused_one_shot_allreduce_residual_rms_norm_allocate_buffers",
      fused_one_shot_allreduce_residual_rms_norm_allocate_buffers);
  m.def(
      "_fused_one_shot_allreduce_residual_rms_norm_init("
      "  int rank, "
      "  int world_size, "
      "  int hidden_size, "
      "  Tensor local_buffer, "
      "  Tensor[] all_buffer_handles, "
      "  Tensor buffer_offsets"
      ") -> ()");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl(
      "_fused_one_shot_allreduce_residual_rms_norm",
      fused_one_shot_allreduce_residual_rms_norm);
  m.impl(
      "_fused_one_shot_allreduce_residual_rms_norm_init",
      fused_one_shot_allreduce_residual_rms_norm_init);
}

} // namespace fbgemm_gpu

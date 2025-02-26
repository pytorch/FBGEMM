/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/library.h>

#include "c10/core/ScalarType.h"
#include "c10/cuda/CUDADeviceAssertionHost.h"
#include "c10/cuda/CUDAFunctions.h"
#include "c10/cuda/CUDAStream.h"
#include "c10/util/Optional.h"

#include <ATen/cuda/CUDAEvent.h>
#include <sys/stat.h>
#include <torch/csrc/cuda/nccl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <vector>
#include "c10/cuda/CUDAException.h"
#include "c10/util/Exception.h"

namespace fbgemm_gpu {

namespace {

static_assert(sizeof(cudaIpcMemHandle_t) == 64, "");

constexpr size_t kMaxNumNcclComms = 5;

static ncclComm_t* get_nccl_comm(int64_t comm_idx) {
  static ncclComm_t comms[kMaxNumNcclComms];

  TORCH_CHECK_GE(comm_idx, 0);
  TORCH_CHECK_LT(comm_idx, kMaxNumNcclComms);
  return &comms[comm_idx];
}

void nccl_init(
    int64_t rank,
    int64_t world_size,
    std::string rendevouz,
    int64_t comm_idx) {
  using namespace c10d;
  ncclUniqueId id;
  if (rank == 0) {
    C10D_NCCL_CHECK(ncclGetUniqueId(&id), "ncclGetUniqueId");
    auto* f = fopen(rendevouz.c_str(), "w");
    fwrite(&id, sizeof(id), 1, f);
    fclose(f);
  } else {
    auto check_size = [&]() {
      struct stat s;
      memset(&s, 0, sizeof(s));
      stat(rendevouz.c_str(), &s);
      return s.st_size;
    };
    while (static_cast<unsigned long>(check_size()) < sizeof(ncclUniqueId)) {
      usleep(1000);
    }
    auto* f = fopen(rendevouz.c_str(), "r");
    fread(&id, sizeof(id), 1, f);
    fclose(f);
  }
  C10D_NCCL_CHECK(
      ncclCommInitRank(get_nccl_comm(comm_idx), world_size, id, rank),
      "ncclCommInitRank");
  return;
}

at::Tensor nccl_get_unique_id() {
  using namespace c10d;
  ncclUniqueId id;
  static_assert(sizeof(ncclUniqueId) == 128, "");
  C10D_NCCL_CHECK(ncclGetUniqueId(&id), "ncclGetUniqueId");
  auto id_ = at::empty({128}, at::TensorOptions().dtype(at::kChar));
  std::memcpy(id_.data_ptr(), &id, sizeof(id));
  return id_;
}

void nccl_comm_init_rank(
    int64_t world_size,
    int64_t rank,
    at::Tensor id_,
    int64_t comm_idx) {
  using namespace c10d;
  ncclUniqueId id;
  static_assert(sizeof(ncclUniqueId) == 128, "");
  std::memcpy(&id, id_.data_ptr(), sizeof(id));
  C10D_NCCL_CHECK(
      ncclCommInitRank(get_nccl_comm(comm_idx), world_size, id, rank),
      "ncclCommInitRank");
}

ncclDataType_t to_nccl_data_type(c10::ScalarType type) {
  switch (type) {
    case at::kFloat:
      return ncclDataType_t::ncclFloat;
    case at::kHalf:
      return ncclDataType_t::ncclHalf;
    case at::kDouble:
      return ncclDataType_t::ncclDouble;
    case at::kLong:
      return ncclDataType_t::ncclInt64;
    case at::kInt:
      return ncclDataType_t::ncclInt;
    case at::kChar:
      return ncclDataType_t::ncclChar;
    case at::kByte:
      return ncclDataType_t::ncclUint8;
    case at::kBool:
      return ncclDataType_t::ncclUint8;
#if defined(USE_ROCM)
    case at::kFloat8_e4m3fnuz:
      return ncclDataType_t::ncclUint8;
    case at::kFloat8_e5m2fnuz:
      return ncclDataType_t::ncclUint8;
#else
    case at::kFloat8_e4m3fn:
      return ncclDataType_t::ncclUint8;
    case at::kFloat8_e5m2:
      return ncclDataType_t::ncclUint8;
#endif
    case at::kBFloat16:
      return ncclDataType_t::ncclBfloat16;
    default:
      TORCH_CHECK(false, "Unconvertible NCCL type ", type);
  }
}

void nccl_allgather(at::Tensor dst, at::Tensor src, int64_t comm_idx) {
  using namespace c10d;
  TORCH_CHECK(src.is_contiguous());
  TORCH_CHECK(dst.is_contiguous());
  TORCH_CHECK(
      src.dtype() == dst.dtype(),
      "dst and src tensors must have the same dtype.");
  ncclDataType_t type = to_nccl_data_type(src.scalar_type());
  C10D_NCCL_CHECK(
      ncclAllGather(
          src.data_ptr(),
          dst.data_ptr(),
          src.numel(),
          type,
          *get_nccl_comm(comm_idx),
          at::cuda::getCurrentCUDAStream()),
      "ncclAllGather");
}

void nccl_alltoall_single(
    at::Tensor dst,
    at::Tensor src,
    int64_t world_size,
    int64_t comm_idx) {
  TORCH_CHECK(src.is_contiguous());
  TORCH_CHECK(dst.is_contiguous());

  auto stream = at::cuda::getCurrentCUDAStream();
  torch::cuda::nccl::all2all_single_equal_split(
      src, dst, world_size, *get_nccl_comm(comm_idx), stream);
}

void nccl_alltoall(
    std::vector<at::Tensor> dsts,
    std::vector<at::Tensor> srcs,
    int64_t comm_idx) {
  auto stream = at::cuda::getCurrentCUDAStream();
  torch::cuda::nccl::all2all(dsts, srcs, *get_nccl_comm(comm_idx), stream);
}

void nccl_reducescatter(at::Tensor dst, at::Tensor src, int64_t comm_idx) {
  using namespace c10d;
  TORCH_CHECK(src.is_contiguous());
  TORCH_CHECK(dst.is_contiguous());
  TORCH_CHECK(src.dtype() == at::ScalarType::BFloat16);
  TORCH_CHECK(dst.dtype() == at::ScalarType::BFloat16);

  C10D_NCCL_CHECK(
      ncclReduceScatter(
          src.data_ptr(),
          dst.data_ptr(),
          dst.numel(),
          ncclDataType_t::ncclBfloat16,
          ncclSum,
          *get_nccl_comm(comm_idx),
          at::cuda::getCurrentCUDAStream()),
      "ncclReduceScatter");
}

void nccl_allreduce(
    at::Tensor dst,
    at::Tensor src,
    std::optional<at::Tensor> bias,
    int64_t comm_idx) {
  using namespace c10d;
  TORCH_CHECK(src.is_contiguous());
  TORCH_CHECK(dst.is_contiguous());
  TORCH_CHECK(dst.dtype() == src.dtype());
  ncclDataType_t type;
  switch (src.scalar_type()) {
    case at::kFloat:
      type = ncclDataType_t::ncclFloat;
      break;
    case at::kHalf:
      type = ncclDataType_t::ncclHalf;
      break;
#ifdef IS_NCCLX_MSCCL
    case at::kFloat8_e4m3fn:
      type = ncclDataType_t::ncclFp8E4M3;
      break;
#endif
    case at::kBFloat16:
      type = ncclDataType_t::ncclBfloat16;
      break;
    default:
      TORCH_CHECK(false, "unsupported type: ", src.scalar_type());
  }
  C10D_NCCL_CHECK(
      ncclAllReduce(
          src.data_ptr(),
          dst.data_ptr(),
          src.numel(),
          type,
          ncclSum,
          *get_nccl_comm(comm_idx),
          at::cuda::getCurrentCUDAStream()),
      "ncclAllReduce");
  if (bias) {
    dst.add_(*bias);
  }
}

} // namespace

at::Tensor car_ipc_handle(at::Tensor x);
void car_init(
    int64_t rank,
    int64_t world_size,
    at::Tensor local_barrier,
    std::vector<at::Tensor> all_barrier_handles,
    at::Tensor local_buffer,
    std::vector<at::Tensor> all_buffer_handles);
void one_shot_car_allreduce(
    at::Tensor dst,
    at::Tensor src,
    std::optional<at::Tensor> bias,
    int64_t comm_idx);
void two_shot_car_allreduce(
    at::Tensor dst,
    at::Tensor src,
    std::optional<at::Tensor> bias,
    int64_t comm_idx);
void car_reducescatter(
    at::Tensor dst,
    at::Tensor src,
    bool split_last_dim,
    int64_t comm_idx);

at::Tensor car_tensor();

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.comm_ops");

  m.def(
      "nccl_init(int rank, int world_size, str rendevouz, int comm_idx=0) -> ()");
  m.impl("nccl_init", nccl_init);

  m.def("nccl_get_unique_id() -> Tensor");
  m.impl("nccl_get_unique_id", nccl_get_unique_id);

  m.def(
      "nccl_comm_init_rank(int world_size, int rank, Tensor id_, int comm_idx=0) -> ()");
  m.impl("nccl_comm_init_rank", nccl_comm_init_rank);

  m.def("nccl_allgather(Tensor(a!) dst, Tensor src, int comm_idx=0) -> ()");

  m.def(
      "nccl_alltoall_single(Tensor(a!) dst, Tensor src, int world_size, int comm_idx=0) -> ()");
  m.def("nccl_alltoall(Tensor(a!)[] dst, Tensor[] src, int comm_idx=0) -> ()");

  m.def("nccl_reducescatter(Tensor(a!) dst, Tensor src, int comm_idx=0) -> ()");

  m.def(
      "nccl_allreduce(Tensor(a!) dst, Tensor src, Tensor? bias=None, int comm_idx=0) -> ()");
  // car: customized all reduce
  m.def("car_tensor() -> Tensor");
  m.impl("car_tensor", car_tensor);

  m.def("car_ipc_handle(Tensor buffer) -> Tensor");
  m.impl("car_ipc_handle", car_ipc_handle);

  m.def(
      "car_init(int rank, int world_size, Tensor local_barrier, Tensor[] all_barrier_handles, Tensor local_buffer, Tensor[] all_buffer_handles) -> ()");
  m.impl("car_init", car_init);

  m.def(
      "one_shot_car_allreduce(Tensor(a!) dst, Tensor src, Tensor? bias=None, int comm_idx=0) -> ()");

  m.def(
      "two_shot_car_allreduce(Tensor(a!) dst, Tensor src, Tensor? bias=None, int comm_idx=0) -> ()");

  m.def(
      "car_reducescatter(Tensor(a!) dst, Tensor src, bool split_last_dim=False, int comm_idx=0) -> ()");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("nccl_allreduce", nccl_allreduce);
  m.impl("nccl_allgather", nccl_allgather);
  m.impl("nccl_alltoall_single", nccl_alltoall_single);
  m.impl("nccl_alltoall", nccl_alltoall);
  m.impl("nccl_reducescatter", nccl_reducescatter);
  m.impl("one_shot_car_allreduce", one_shot_car_allreduce);
  m.impl("two_shot_car_allreduce", two_shot_car_allreduce);
  m.impl("car_reducescatter", car_reducescatter);
}

// Though it shouldnt be used, it is useful to define these functions for CPU to
// accomodate model creation.
TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl("nccl_allreduce", nccl_allreduce);
  m.impl("nccl_allgather", nccl_allgather);
  m.impl("nccl_alltoall_single", nccl_alltoall_single);
  m.impl("nccl_alltoall", nccl_alltoall);
  m.impl("nccl_reducescatter", nccl_reducescatter);
  m.impl("one_shot_car_allreduce", one_shot_car_allreduce);
  m.impl("two_shot_car_allreduce", two_shot_car_allreduce);
  m.impl("car_reducescatter", car_reducescatter);
}

// Shape registration functions for car operators.
void nccl_allreduce_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    std::optional<at::Tensor> /* bias */,
    int64_t /* comm_idx */) {
  return;
}

void nccl_allgather_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    int64_t /* comm_idx */) {
  return;
}

void nccl_alltoall_single_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    int64_t /* world_size */,
    int64_t /* comm_idx */) {
  return;
}

void nccl_alltoall_meta(
    std::vector<at::Tensor> /* dsts */,
    std::vector<at::Tensor> /* srcs */,
    int64_t /* comm_idx */) {
  return;
}

void nccl_reducescatter_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    int64_t /* comm_idx */) {
  return;
}

void one_shot_car_allreduce_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    std::optional<at::Tensor> /* bias */,
    int64_t /* comm_idx */) {
  return;
}

void two_shot_car_allreduce_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    std::optional<at::Tensor> /* bias */,
    int64_t /* comm_idx */) {
  return;
}

void car_reducescatter_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    bool /* split_last_dim */,
    int64_t /* comm_idx */) {
  return;
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("nccl_allreduce", nccl_allreduce_meta);
  m.impl("nccl_allgather", nccl_allgather_meta);
  m.impl("nccl_alltoall_single", nccl_alltoall_single_meta);
  m.impl("nccl_alltoall", nccl_alltoall_meta);
  m.impl("nccl_reducescatter", nccl_reducescatter_meta);
  m.impl("one_shot_car_allreduce", one_shot_car_allreduce_meta);
  m.impl("two_shot_car_allreduce", two_shot_car_allreduce_meta);
  m.impl("car_reducescatter", car_reducescatter_meta);
}

} // namespace fbgemm_gpu

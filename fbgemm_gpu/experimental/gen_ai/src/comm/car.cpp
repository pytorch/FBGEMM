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
#include "folly/futures/Future.h"

#include <ATen/cuda/CUDAEvent.h>
#include <folly/experimental/symbolizer/SignalHandler.h>
#include <sys/stat.h>
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

constexpr size_t kMaxNumNcclComms = 3;

static ncclComm_t* get_nccl_comm(int64_t comm_idx) {
  static ncclComm_t comms[kMaxNumNcclComms];

  CHECK_GE(comm_idx, 0);
  CHECK_LT(comm_idx, kMaxNumNcclComms);
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

void nccl_allgather(at::Tensor y_allgather, at::Tensor y, int64_t comm_idx) {
  using namespace c10d;
  TORCH_CHECK(y.is_contiguous());
  TORCH_CHECK(y_allgather.is_contiguous());
  ncclDataType_t type;
  switch (y.scalar_type()) {
    case at::kFloat:
      type = ncclDataType_t::ncclFloat;
      break;
    case at::kHalf:
      type = ncclDataType_t::ncclHalf;
      break;
    case at::kBFloat16:
      type = ncclDataType_t::ncclBfloat16;
      break;
    default:
      TORCH_CHECK(false, "unsupported type: ", y.scalar_type());
  }
  C10D_NCCL_CHECK(
      ncclAllGather(
          y.data_ptr(),
          y_allgather.data_ptr(),
          y.numel(),
          type,
          *get_nccl_comm(comm_idx),
          at::cuda::getCurrentCUDAStream()),
      "ncclAllGather");
}

void nccl_reducescatter(
    at::Tensor y_reducescatter,
    at::Tensor y,
    int64_t comm_idx) {
  using namespace c10d;
  TORCH_CHECK(y.is_contiguous());
  TORCH_CHECK(y_reducescatter.is_contiguous());
  TORCH_CHECK(y.dtype() == at::ScalarType::BFloat16);
  TORCH_CHECK(y_reducescatter.dtype() == at::ScalarType::BFloat16);

  C10D_NCCL_CHECK(
      ncclReduceScatter(
          y.data_ptr(),
          y_reducescatter.data_ptr(),
          y_reducescatter.numel(),
          ncclDataType_t::ncclBfloat16,
          ncclSum,
          *get_nccl_comm(comm_idx),
          at::cuda::getCurrentCUDAStream()),
      "ncclReduceScatter");
}

void nccl_allreduce(
    at::Tensor y_allreduce,
    at::Tensor y,
    std::optional<at::Tensor> z,
    int64_t comm_idx) {
  using namespace c10d;
  TORCH_CHECK(y.is_contiguous());
  TORCH_CHECK(y_allreduce.is_contiguous());
  TORCH_CHECK(y_allreduce.dtype() == y.dtype());
  ncclDataType_t type;
  switch (y.scalar_type()) {
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
      TORCH_CHECK(false, "unsupported type: ", y.scalar_type());
  }
  C10D_NCCL_CHECK(
      ncclAllReduce(
          y.data_ptr(),
          y_allreduce.data_ptr(),
          y.numel(),
          type,
          ncclSum,
          *get_nccl_comm(comm_idx),
          at::cuda::getCurrentCUDAStream()),
      "ncclAllReduce");
  if (z) {
    y_allreduce.add_(*z);
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
    at::Tensor y_allreduce,
    at::Tensor y,
    std::optional<at::Tensor> z,
    int64_t comm_idx);
void two_shot_car_allreduce(
    at::Tensor y_allreduce,
    at::Tensor y,
    std::optional<at::Tensor> z,
    int64_t comm_idx);

at::Tensor car_tensor();

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "nccl_init(int rank, int world_size, str rendevouz, int comm_idx=0) -> ()");
  m.impl("nccl_init", nccl_init);

  m.def("nccl_get_unique_id() -> Tensor");
  m.impl("nccl_get_unique_id", nccl_get_unique_id);

  m.def(
      "nccl_comm_init_rank(int world_size, int rank, Tensor id_, int comm_idx=0) -> ()");
  m.impl("nccl_comm_init_rank", nccl_comm_init_rank);

  m.def("nccl_allgather(Tensor y_allgather, Tensor y, int comm_idx=0) -> ()");
  m.impl("nccl_allgather", nccl_allgather);

  m.def(
      "nccl_reducescatter(Tensor y_reducescatter, Tensor y, int comm_idx=0) -> ()");
  m.impl("nccl_reducescatter", nccl_reducescatter);

  m.def(
      "nccl_allreduce(Tensor y_allreduce, Tensor y, Tensor? z=None, int comm_idx=0) -> ()");
  m.impl("nccl_allreduce", nccl_allreduce);

  // car: customized all reduce
  m.def("car_tensor() -> Tensor");
  m.impl("car_tensor", car_tensor);

  m.def("car_ipc_handle(Tensor buffer) -> Tensor");
  m.impl("car_ipc_handle", car_ipc_handle);

  m.def(
      "car_init(int rank, int world_size, Tensor local_barrier, Tensor[] all_barrier_handles, Tensor local_buffer, Tensor[] all_buffer_handles) -> ()");
  m.impl("car_init", car_init);

  m.def(
      "one_shot_car_allreduce(Tensor y_allreduce, Tensor y, Tensor? z=None, int comm_idx=0) -> ()");
  m.impl("one_shot_car_allreduce", one_shot_car_allreduce);

  m.def(
      "two_shot_car_allreduce(Tensor y_allreduce, Tensor y, Tensor? z=None, int comm_idx=0) -> ()");
  m.impl("two_shot_car_allreduce", two_shot_car_allreduce);
}

} // namespace fbgemm_gpu

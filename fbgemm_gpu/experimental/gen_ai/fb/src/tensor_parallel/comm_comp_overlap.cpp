// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nccl.h>
#include <torch/cuda.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <torch/types.h>
#include <cassert>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <tuple>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                 \
  do {                                   \
    cudaError_t status_ = call;          \
    if (status_ != cudaSuccess) {        \
      fprintf(                           \
          stderr,                        \
          "CUDA Error at line %d: %s\n", \
          __LINE__,                      \
          cudaGetErrorString(status_));  \
      exit(1);                           \
    }                                    \
  } while (0)
#endif

#ifndef CHECK_NCCL
#define CHECK_NCCL(cmd)                      \
  do {                                       \
    ncclResult_t e = cmd;                    \
    if (e != ncclSuccess) {                  \
      printf(                                \
          "Failed: NCCL error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          ncclGetErrorString(e));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)
#endif

namespace fbgemm_gpu {
namespace gen_ai {
namespace fb {
namespace tensor_parallel {

// Class definition for fused communication and computation overlap.
// This declaration can be further migrated into a separate header file in the
// future.
class FusedCommComp : public torch::CustomClassHolder {
 private:
  const int tpID_, tpSize_, tpLocalID_, tpLocalSize_;
  const size_t numElements_;
  const at::ScalarType dtype_;

  size_t chunkElements_, buffBytes_;

  int nodeID_, numNodes_;
  bool shouldDestroyComm_, avoidInCastCongestion_;
  ncclComm_t comm_;
  ncclWin_t win_;

  ncclDataType_t ncclDtype_;
  size_t dtypeBytes_;

  void* ncclBufPtr_ = nullptr;
  void* barrierBufPtr_ = nullptr;

  cudaStream_t putStream_, waitStream_, ctrlStream_;
  cudaEvent_t waitEvent_;

  size_t getDataTypebytes_(const at::ScalarType& type);
  ncclDataType_t getNcclDataType_(const at::ScalarType& type);

  void internalBarrier_(cudaStream_t stream);

  bool checkInputTensor_(const at::Tensor& inputTensor);

 public:
  FusedCommComp(
      int64_t tpID,
      int64_t tpSize,
      int64_t tpLocalID,
      int64_t tpLocalSize,
      int64_t rank,
      int64_t worldSize,
      int64_t numElements,
      at::ScalarType dtype,
      int64_t externalCommPtr);
  ~FusedCommComp();

  // Get an internal buffer like tensor A from offset elements
  at::Tensor internalBufferLike(int64_t offsetElements, at::Tensor A);

  // Ask the main stream to wait for internal comm streams
  void waitForCommStream();

  // Use remote-memory-access (RMA) based all-gather to collect tensors into
  // internal buffer (ncclBufPtr_).
  at::Tensor rmaWinAllGather(at::Tensor A);

  // Use remote-memory-access (RMA) based scatter to scatter tensors into
  // internal buffer (ncclBufPtr_). This function togther with a local reduce
  // is equivalent to a reduce-scatter.
  void rmaWinScatter(at::Tensor A);

  void avoidInCastCongestion(bool flag);

  // Local reduce results from the internal buffer into the final output tensor
  void localReduceIntoTensor(at::Tensor output, at::Tensor input);

  std::tuple<at::Tensor, at::Tensor> splitOverlapAllGather(
      at::Tensor A,
      bool transa,
      at::Tensor B,
      bool transb,
      at::Tensor D,
      bool inplaceA,
      bool barrier);

  at::Tensor splitOverlapReduceScatter(
      at::Tensor A,
      bool transa,
      at::Tensor B,
      bool transb,
      at::Tensor D,
      bool barrier);
};

FusedCommComp::FusedCommComp(
    int64_t tpID,
    int64_t tpSize,
    int64_t tpLocalID,
    int64_t tpLocalSize,
    int64_t rank,
    int64_t worldSize,
    int64_t numElements,
    at::ScalarType dtype,
    int64_t externalCommPtr)
    : tpID_(tpID),
      tpSize_(tpSize),
      tpLocalID_(tpLocalID),
      tpLocalSize_(tpLocalSize),
      numElements_(numElements),
      dtype_(dtype) {
  assert(tpSize_ <= 8); // We only support single node now

  // TP toplogy info checks.
  assert(tpID_ < tpSize_);
  assert(tpLocalID_ < tpLocalSize_);
  assert(tpLocalSize_ <= tpSize_);
  assert(tpSize_ % tpLocalSize_ == 0);
  numNodes_ = (tpSize_ / tpLocalSize_);
  nodeID_ = (tpID_ / tpLocalSize_);
  assert((nodeID_ > 0) && (nodeID_ < numNodes_));

  // Create a NCCL communicator
  ncclComm_t externalComm = reinterpret_cast<ncclComm_t>(externalCommPtr);
  if (externalComm != nullptr) {
    // This communicator is allocated from the caller end, thus the destructor
    // of this class should not handle the commDestroy process
    shouldDestroyComm_ = false;
    comm_ = externalComm;
    fprintf(
        stderr,
        "[TP Rank=%d] Use the external communicator %p\n",
        tpID_,
        externalComm);
  } else {
    assert(worldSize % tpSize_ == 0);
    assert(rank < worldSize);
    int commColor = rank / tpSize_;
    CHECK_NCCL(ncclCommSplit(NCCL_COMM_WORLD, commColor, tpID_, &comm_, NULL));
    shouldDestroyComm_ = true;
    fprintf(
        stderr, "[TP Rank=%d] Allocated a new communicator %p\n", tpID_, comm_);
  }

  // Collectively allocate a nccl window of a data buffer
  dtypeBytes_ = getDataTypebytes_(dtype_);
  ncclDtype_ = getNcclDataType_(dtype_);
  assert((numElements_ % tpSize_) == 0);
  chunkElements_ = (numElements_ / tpSize);
  buffBytes_ =
      (numElements_ * dtypeBytes_ * 2 // 2x for send recv at the same time
       + chunkElements_ * dtypeBytes_ * numNodes_);
  CHECK_NCCL(ncclWinAllocate(buffBytes_, comm_, &ncclBufPtr_, &win_));
  CHECK_CUDA(cudaMalloc(&barrierBufPtr_, tpSize_ * sizeof(int) * 2));

  const size_t totalBufBytes = buffBytes_ + tpSize_ * sizeof(int) * 2;
  fprintf(
      stderr,
      "[TP Rank=%d] Allocated %zu bytes for FusedCommComp\n",
      tpID_,
      totalBufBytes);

  // Create internal cuda streams and events
  CHECK_CUDA(
      cudaStreamCreateWithPriority(&putStream_, cudaStreamNonBlocking, -1));
  CHECK_CUDA(
      cudaStreamCreateWithPriority(&waitStream_, cudaStreamNonBlocking, -1));
  CHECK_CUDA(
      cudaStreamCreateWithPriority(&ctrlStream_, cudaStreamNonBlocking, -1));
  CHECK_CUDA(cudaEventCreate(&waitEvent_));

  // Avoid in-cast congestion can improve the memcpy bandwidth. However, it also
  // brings additional syncs between ranks. Thus this flag needs to be
  // explicitly turned on.
  avoidInCastCongestion_ = false;
};

FusedCommComp::~FusedCommComp() {
  CHECK_CUDA(cudaEventDestroy(waitEvent_));
  CHECK_CUDA(cudaStreamDestroy(putStream_));
  CHECK_CUDA(cudaStreamDestroy(waitStream_));
  CHECK_CUDA(cudaStreamDestroy(ctrlStream_));
  CHECK_CUDA(cudaFree(barrierBufPtr_));
  CHECK_NCCL(ncclWinFree(comm_, win_));
  if (shouldDestroyComm_) {
    CHECK_NCCL(ncclCommDestroy(comm_));
  }
}

void FusedCommComp::internalBarrier_(cudaStream_t stream) {
  // Launch a barrier on stream
  cudaStream_t mainStream = ((cudaStream_t)at::cuda::getDefaultCUDAStream());
  CHECK_CUDA(cudaEventRecord(waitEvent_, mainStream));
  CHECK_CUDA(cudaStreamWaitEvent(stream, waitEvent_));

  CHECK_NCCL(ncclAllGather(
      barrierBufPtr_,
      reinterpret_cast<char*>(barrierBufPtr_) + 2,
      1,
      ncclBfloat16,
      comm_,
      stream));
  return;
}

bool FusedCommComp::checkInputTensor_(const at::Tensor& inputTensor) {
  assert(inputTensor.options.dtype() == dtype_);
  assert(inputTensor.dim() == 2);
  return true;
}

at::Tensor FusedCommComp::internalBufferLike(
    int64_t offsetElements,
    at::Tensor A) {
  assert(checkInputTensor_(A));
  const size_t offsetBytes = offsetElements * A.element_size();
  const size_t dataBytes = A.numel() * A.element_size();
  // Out of range check
  assert((offsetBytes + dataBytes) <= buffBytes_);

  char* outputPtr = reinterpret_cast<char*>(ncclBufPtr_) + offsetBytes;
  return torch::from_blob(outputPtr, A.sizes(), A.options());
}

std::tuple<at::Tensor, at::Tensor> FusedCommComp::splitOverlapAllGather(
    at::Tensor A,
    bool transa,
    at::Tensor B,
    bool transb,
    at::Tensor D,
    bool inplaceA,
    bool barrier) {
  assert(!transa); // Current impl assumes A layout is not transposed
  if (transb) {
    B = B.t();
  }
  assert(checkInputTensor_(A) && checkInputTensor_(B) && checkInputTensor_(D));

  cudaStream_t mainStream = ((cudaStream_t)at::cuda::getDefaultCUDAStream());
  const size_t bytesTensorA = A.numel() * A.element_size();
  const size_t chunkBytesTensorD = ((D.numel() * D.element_size()) / tpSize_);
  if (!inplaceA) {
    CHECK_CUDA(cudaMemcpyAsync(
        reinterpret_cast<char*>(ncclBufPtr_) + tpID_ * bytesTensorA,
        A.data_ptr(),
        bytesTensorA,
        cudaMemcpyDeviceToDevice,
        mainStream));
  }

  CHECK_CUDA(cudaEventRecord(waitEvent_, mainStream));
  CHECK_CUDA(cudaStreamWaitEvent(putStream_, waitEvent_));
  CHECK_CUDA(cudaStreamWaitEvent(waitStream_, waitEvent_));

  // Ask all ranks to sync before putting data into the buffer of other ranks
  if (barrier) {
    internalBarrier_(putStream_);
  }

  const int64_t prevRankID = (tpID_ - 1 + tpSize_) % tpSize_;
  const int64_t nextRankID = (tpID_ + 1) % tpSize_;
  for (int step = 0; step < tpSize_; step++) {
    if (step != 0) {
      // We need to wait for the data before comp and put
      CHECK_CUDA(cudaEventRecord(waitEvent_, waitStream_));
      CHECK_CUDA(cudaStreamWaitEvent(putStream_, waitEvent_));
      CHECK_CUDA(cudaStreamWaitEvent(mainStream, waitEvent_));
    }

    const int chunkID = (tpID_ + step) % tpSize_;
    char* chunkAPtr =
        (reinterpret_cast<char*>(ncclBufPtr_) + chunkID * bytesTensorA);
    at::Tensor chunkTensorA =
        torch::from_blob(chunkAPtr, {A.size(0), A.size(1)}, A.options());

    char* chunkDPtr =
        (reinterpret_cast<char*>(D.data_ptr()) + chunkID * chunkBytesTensorD);
    at::Tensor chunkTensorD = torch::from_blob(
        chunkDPtr, {D.size(0) / tpSize_, D.size(1)}, D.options());

    at::matmul_out(chunkTensorD, chunkTensorA, B);

    if (step != (tpSize_ - 1)) {
      CHECK_NCCL(ncclPutSignal(
          chunkAPtr,
          A.numel(),
          ncclDtype_,
          prevRankID,
          (chunkID * bytesTensorA / dtypeBytes_),
          win_,
          putStream_));
      CHECK_NCCL(ncclWaitSignal(nextRankID, win_, waitStream_));
    }
  }

  at::Tensor allGatheredA = torch::from_blob(
      ncclBufPtr_, {tpSize_ * A.size(0), A.size(1)}, A.options());
  return {allGatheredA, D};
}

at::Tensor FusedCommComp::splitOverlapReduceScatter(
    at::Tensor A,
    bool transa,
    at::Tensor B,
    bool transb,
    at::Tensor D,
    bool barrier) {
  assert(!transa); // Current impl assumes A layout is not transposed
  if (transb) {
    B = B.t();
  }
  assert(checkInputTensor_(A) && checkInputTensor_(B) && checkInputTensor_(D));
  cudaStream_t mainStream = ((cudaStream_t)at::cuda::getDefaultCUDAStream());

  const size_t bytesTensorA = A.numel() * A.element_size();
  const size_t chunkBytesTensorA = (bytesTensorA / tpSize_);
  const size_t bytesTensorD = D.numel() * D.element_size();

  const int64_t prevRankID = (tpID_ - 1 + tpSize_) % tpSize_;
  const int64_t nextRankID = (tpID_ + 1) % tpSize_;
  for (int step = 1; step < tpSize_; step++) {
    const int chunkID = (tpID_ + step) % tpSize_;
    const int srcRankID = (tpID_ - step + tpSize_) % tpSize_;

    // GEMM to compute outputs
    char* chunkAPtr =
        (reinterpret_cast<char*>(A.data_ptr()) + chunkID * chunkBytesTensorA);
    at::Tensor chunkATensor = torch::from_blob(
        chunkAPtr, {A.size(0) / tpSize_, A.size(1)}, A.options());
    char* chunkDPtr =
        (reinterpret_cast<char*>(ncclBufPtr_) + chunkID * bytesTensorD);
    at::Tensor chunkDTensor =
        torch::from_blob(chunkDPtr, {D.size(0), D.size(1)}, D.options());
    at::matmul_out(chunkDTensor, chunkATensor, B);

    CHECK_CUDA(cudaEventRecord(waitEvent_, mainStream));
    CHECK_CUDA(cudaStreamWaitEvent(putStream_, waitEvent_));
    CHECK_CUDA(cudaStreamWaitEvent(waitStream_, waitEvent_));

    CHECK_NCCL(ncclPutSignal(
        chunkDPtr,
        D.numel(),
        ncclDtype_,
        chunkID,
        ((tpSize_ + step) * bytesTensorD) / dtypeBytes_,
        win_,
        putStream_));
    CHECK_NCCL(ncclWaitSignal(srcRankID, win_, waitStream_));

    if ((step != (tpSize_ - 1)) && avoidInCastCongestion_) {
      CHECK_NCCL(ncclPutSignal(
          chunkDPtr,
          0, // send count is 0 to launch kernelNotify
          ncclDtype_,
          prevRankID,
          0, // a dummy offset
          win_,
          putStream_));
      // Wait the signal from next rank to proceed next PUT
      CHECK_NCCL(ncclWaitSignal(nextRankID, win_, ctrlStream_));
      CHECK_CUDA(cudaEventRecord(waitEvent_, ctrlStream_));
      CHECK_CUDA(cudaStreamWaitEvent(putStream_, waitEvent_));
    }
  }

  at::Tensor lastChunkTensorA = torch::from_blob(
      reinterpret_cast<char*>(A.data_ptr()) + tpID_ * chunkBytesTensorA,
      {A.size(0) / tpSize_, A.size(1)},
      A.options());
  at::Tensor lastChunkTensorD = torch::from_blob(
      reinterpret_cast<char*>(ncclBufPtr_) + tpSize_ * bytesTensorD,
      {D.size(0), D.size(1)},
      D.options());
  at::matmul_out(lastChunkTensorD, lastChunkTensorA, B);

  CHECK_CUDA(cudaEventRecord(waitEvent_, waitStream_));
  CHECK_CUDA(cudaStreamWaitEvent(mainStream, waitEvent_));

  at::Tensor partialD = torch::from_blob(
      reinterpret_cast<char*>(ncclBufPtr_) + tpSize_ * bytesTensorD,
      {tpSize_, D.size(0), D.size(1)},
      D.options());
  at::sum_out(D, partialD, /*dim=*/{0});
  return D;
}

size_t FusedCommComp::getDataTypebytes_(const at::ScalarType& type) {
  switch (type) {
    case at::ScalarType::BFloat16:
      return 2;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

ncclDataType_t FusedCommComp::getNcclDataType_(const at::ScalarType& type) {
  switch (type) {
    case at::ScalarType::BFloat16:
      return ncclBfloat16;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

at::Tensor FusedCommComp::rmaWinAllGather(at::Tensor A) {
  assert(tpSize_ <= 8); // We only support single node now
  assert(checkInputTensor_(A));

  const size_t bytesTensorA = A.numel() * A.element_size();
  assert(bytesTensorA * tpSize_ * 2 <= buffBytes_);

  cudaStream_t mainStream = ((cudaStream_t)at::cuda::getDefaultCUDAStream());
  CHECK_CUDA(cudaMemcpyAsync(
      reinterpret_cast<char*>(ncclBufPtr_) + tpID_ * bytesTensorA,
      A.data_ptr(),
      bytesTensorA,
      cudaMemcpyDeviceToDevice,
      mainStream));

  // Adding dependencies for comm streams to wait for compute data
  internalBarrier_(putStream_);
  CHECK_CUDA(cudaEventRecord(waitEvent_, mainStream));
  CHECK_CUDA(cudaStreamWaitEvent(waitStream_, waitEvent_));

  const int64_t prevRankID = (tpID_ - 1 + tpSize_) % tpSize_;
  const int64_t nextRankID = (tpID_ + 1) % tpSize_;
  // Put and wait (tp size - 1) chunks of data
  for (int step = 0; step < (tpSize_ - 1); step++) {
    if (step != 0) {
      // The data to put is not a local chunk of data
      CHECK_CUDA(cudaEventRecord(waitEvent_, waitStream_));
      CHECK_CUDA(cudaStreamWaitEvent(putStream_, waitEvent_));
    }

    const int64_t chunkID = (tpID_ + step) % tpSize_;
    const size_t bufOffset = chunkID * bytesTensorA;
    CHECK_NCCL(ncclPutSignal(
        reinterpret_cast<char*>(ncclBufPtr_) + bufOffset,
        A.numel(),
        ncclDtype_,
        prevRankID,
        (bufOffset / dtypeBytes_),
        win_,
        putStream_));
    CHECK_NCCL(ncclWaitSignal(nextRankID, win_, waitStream_));
  }

  at::Tensor allGatheredA = torch::from_blob(
      ncclBufPtr_, {tpSize_ * A.size(0), A.size(1)}, A.options());
  return allGatheredA;
}

void FusedCommComp::waitForCommStream() {
  cudaStream_t stream = ((cudaStream_t)at::cuda::getDefaultCUDAStream());
  CHECK_CUDA(cudaEventRecord(waitEvent_, putStream_));
  CHECK_CUDA(cudaStreamWaitEvent(stream, waitEvent_));
  CHECK_CUDA(cudaEventRecord(waitEvent_, waitStream_));
  CHECK_CUDA(cudaStreamWaitEvent(stream, waitEvent_));
  return;
}

void FusedCommComp::avoidInCastCongestion(bool flag) {
  avoidInCastCongestion_ = flag;
}

void FusedCommComp::rmaWinScatter(at::Tensor A) {
  assert(tpSize_ <= 8); // We only support single node now
  assert(checkInputTensor_(A));

  const size_t bytesTensorA = A.numel() * A.element_size();
  const size_t chunkBytesTensorA = (bytesTensorA / tpSize_);
  assert(bytesTensorA * tpSize_ * 2 <= buffBytes_);
  cudaStream_t mainStream = ((cudaStream_t)at::cuda::getDefaultCUDAStream());

  // Adding dependencies for comm streams to wait for compute data
  internalBarrier_(putStream_);
  CHECK_CUDA(cudaEventRecord(waitEvent_, mainStream));
  CHECK_CUDA(cudaStreamWaitEvent(waitStream_, waitEvent_));
  if (avoidInCastCongestion_) {
    CHECK_CUDA(cudaStreamWaitEvent(ctrlStream_, waitEvent_));
  }

  const int64_t prevRankID = (tpID_ - 1 + tpSize_) % tpSize_;
  const int64_t nextRankID = (tpID_ + 1) % tpSize_;
  for (int step = 1; step < tpSize_; ++step) {
    const int64_t chunkID = (tpID_ + step) % tpSize_;
    const int64_t srcRankID = (tpID_ - step + tpSize_) % tpSize_;

    // Scatter results to the destination rank
    CHECK_NCCL(ncclPutSignal(
        reinterpret_cast<char*>(A.data_ptr()) + chunkID * chunkBytesTensorA,
        (A.numel() / tpSize_),
        ncclDtype_,
        chunkID,
        (bytesTensorA + step * chunkBytesTensorA) / dtypeBytes_,
        win_,
        putStream_));
    CHECK_NCCL(ncclWaitSignal(srcRankID, win_, waitStream_));

    if ((step != (tpSize_ - 1)) && (avoidInCastCongestion_)) {
      // Notify the prev rank the completion to avoid the congestion
      CHECK_NCCL(ncclPutSignal(
          A.data_ptr(),
          0, // send count is 0 to launch kernelNotify
          ncclDtype_,
          prevRankID,
          0, // a dummy offset
          win_,
          putStream_));
      // Wait the signal from next rank to proceed next PUT
      CHECK_NCCL(ncclWaitSignal(nextRankID, win_, ctrlStream_));
      CHECK_CUDA(cudaEventRecord(waitEvent_, ctrlStream_));
      CHECK_CUDA(cudaStreamWaitEvent(putStream_, waitEvent_));
    }
  }
}

void FusedCommComp::localReduceIntoTensor(at::Tensor out, at::Tensor inp) {
  assert(tpSize_ <= 8); // We only support single node now
  assert(checkInputTensor_(inp));

  const size_t bytesTensorInp = inp.numel() * inp.element_size();
  const size_t chunkBytesTensorInp = (bytesTensorInp / tpSize_);
  assert(bytesTensorInp * tpSize_ * 2 <= buffBytes_);

  waitForCommStream();
  cudaStream_t mainStream = ((cudaStream_t)at::cuda::getDefaultCUDAStream());
  CHECK_CUDA(cudaMemcpyAsync(
      reinterpret_cast<char*>(ncclBufPtr_) + bytesTensorInp,
      reinterpret_cast<char*>(inp.data_ptr()) + tpID_ * chunkBytesTensorInp,
      chunkBytesTensorInp,
      cudaMemcpyDeviceToDevice,
      mainStream));

  at::Tensor partialOut = torch::from_blob(
      reinterpret_cast<char*>(ncclBufPtr_) + bytesTensorInp,
      {tpSize_, out.size(0), out.size(1)},
      out.options());

  at::sum_out(out, partialOut, /*dim=*/{0});
  return;
}

} // namespace tensor_parallel
} // namespace fb
} // namespace gen_ai
} // namespace fbgemm_gpu

using namespace fbgemm_gpu::gen_ai::fb::tensor_parallel;

TORCH_LIBRARY(fbgemm, m) {
  m.class_<FusedCommComp>("FusedCommComp")
      .def(torch::init<
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           at::ScalarType,
           int64_t>())
      .def("internal_buffer_like", &FusedCommComp::internalBufferLike)
      .def("rma_win_all_gather", &FusedCommComp::rmaWinAllGather)
      .def("wait_for_comm_stream", &FusedCommComp::waitForCommStream)
      .def("ram_win_scatter", &FusedCommComp::rmaWinScatter)
      .def("avoid_incast_congestion", &FusedCommComp::avoidInCastCongestion)
      .def("local_reduce_into_tensor", &FusedCommComp::localReduceIntoTensor)
      .def("split_overlap_ag", &FusedCommComp::splitOverlapAllGather)
      .def("split_overlap_rs", &FusedCommComp::splitOverlapReduceScatter);
}

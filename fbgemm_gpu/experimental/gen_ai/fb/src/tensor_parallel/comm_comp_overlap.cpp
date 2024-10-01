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

// TODO: Separate the following FP8 quantization and rescale related ops into
// a different file
at::Tensor row_col_rescale(
    at::Tensor input,
    at::Tensor row_scale,
    at::Tensor col_scale,
    std::optional<at::Tensor> output);

// Class definition for fused communication and computation overlap.
// This declaration can be further migrated into a separate header file in the
// future.
class FusedCommComp : public torch::CustomClassHolder {
 private:
  const int tpID_, tpSize_, tpLocalID_, tpLocalSize_;
  const size_t numElements_;

  size_t chunkElements_, buffBytes_;

  int nodeID_, numNodes_;
  bool shouldDestroyComm_, avoidInCastCongestion_;
  ncclComm_t comm_;
  ncclWin_t win_;

  void* ncclBufPtr_ = nullptr;
  void* barrierBufPtr_ = nullptr;

  cudaStream_t putStream_, waitStream_, ctrlStream_;
  cudaEvent_t waitEvent_;

  at::Tensor oneTensor_;

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

  // An internal GEMM function to handle both BF16 and FP8 GEMMs
  void gemm_(at::Tensor output, at::Tensor A, at::Tensor B);

  // Get an internal buffer like tensor A from offset elements
  at::Tensor internalBufferLike(int64_t offsetElements, at::Tensor A);

  // Get an internal buffer of length elements from an offset
  at::Tensor getInternalBuffer(
      int64_t offsetElements,
      int64_t lengthElements,
      at::Tensor A);

  // Ask the main stream to wait for internal comm streams
  void waitForCommStream();

  // Use remote-memory-access (RMA) based all-gather to collect tensors into
  // internal buffer (ncclBufPtr_).
  at::Tensor rmaWinAllGather(at::Tensor A, int64_t offsetElements);
  at::Tensor rmaWinAllGatherTree(at::Tensor A, int64_t offsetElements);

  // Use remote-memory-access (RMA) based scatter to scatter tensors into
  // internal buffer (ncclBufPtr_). This function togther with a local reduce
  // is equivalent to a reduce-scatter.
  void rmaWinScatter(at::Tensor A, int64_t offsetElements);

  void avoidInCastCongestion(bool flag);

  // Local reduce results from the internal buffer into the final output tensor
  void localReduceIntoTensor(
      at::Tensor output,
      at::Tensor input,
      int64_t offsetElements);

  std::tuple<at::Tensor, at::Tensor> splitOverlapAllGather(
      at::Tensor A,
      bool transa,
      at::Tensor B,
      bool transb,
      at::Tensor D,
      bool inplaceA,
      bool barrier);

  std::tuple<at::Tensor, at::Tensor> splitOverlapAllGatherTree(
      at::Tensor A,
      bool transa,
      at::Tensor B,
      bool transb,
      at::Tensor D,
      bool inplaceA,
      bool barrier,
      c10::optional<at::Tensor> localRowScale,
      c10::optional<at::Tensor> rowScale);

  at::Tensor splitOverlapReduceScatter(
      at::Tensor A,
      bool transa,
      at::Tensor B,
      bool transb,
      at::Tensor D,
      bool barrier,
      bool skipReduce,
      c10::optional<at::Tensor> rowScale,
      c10::optional<at::Tensor> colScale);

  // Public member variables
  bool fp8GEMMFastAccum;
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
      numElements_(numElements) {
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
  const size_t dtypeBytes = getDataTypebytes_(dtype);
  assert((numElements_ % tpSize_) == 0);
  chunkElements_ = (numElements_ / tpSize);
  buffBytes_ =
      (numElements_ * dtypeBytes * 2 // 2x for send recv at the same time
       + chunkElements_ * dtypeBytes * numNodes_);
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

  // Member variables for FP8 GEMM:
  //  - fp8GEMMFastAccum: use fast accumulation for FP8 GEMM
  //  - oneTensor_: a tensor of 1.0 used as the scale for FP8 GEMM
  fp8GEMMFastAccum = true;
  auto options = torch::TensorOptions()
                     .device(torch::kCUDA, at::cuda::current_device())
                     .dtype(torch::kFloat32);
  oneTensor_ = at::ones({1}, options);
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

void FusedCommComp::gemm_(at::Tensor output, at::Tensor A, at::Tensor B) {
  assert(A.scalar_type() == B.scalar_type());
  assert(output.scalar_type() == at::ScalarType::BFloat16);

  // BF16 GEMM
  if (A.scalar_type() == at::ScalarType::BFloat16) {
    at::matmul_out(output, A, B);
  } else if (A.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    at::_scaled_mm_out(
        output,
        A,
        B,
        oneTensor_,
        oneTensor_,
        std::nullopt,
        std::nullopt,
        at::ScalarType::BFloat16,
        fp8GEMMFastAccum);
  } else {
    throw std::runtime_error("Unsupported data type for GEMM");
  }
  return;
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

at::Tensor FusedCommComp::getInternalBuffer(
    int64_t offsetElements,
    int64_t lengthElements,
    at::Tensor A) {
  const size_t offsetBytes = offsetElements * A.element_size();
  const size_t lengthBytes = lengthElements * A.element_size();
  // Out of range check
  assert((offsetBytes + lengthBytes) <= buffBytes_);

  char* outputPtr = reinterpret_cast<char*>(ncclBufPtr_) + offsetBytes;
  return torch::from_blob(outputPtr, {lengthElements}, A.options());
}

std::tuple<at::Tensor, at::Tensor> FusedCommComp::splitOverlapAllGatherTree(
    at::Tensor A,
    bool transa,
    at::Tensor B,
    bool transb,
    at::Tensor D,
    bool inplaceA,
    bool barrier,
    c10::optional<at::Tensor> localRowScale,
    c10::optional<at::Tensor> rowScale) {
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

  int64_t gemmChunkID = tpID_;
  int64_t gemmBlockSize = 1;
  const ncclDataType_t ncclDtype = getNcclDataType_(A.scalar_type());
  for (int blockSize = 1; blockSize <= tpSize_; blockSize *= 2) {
    if (blockSize != 1) {
      // We need to wait for the data before comp and put
      CHECK_CUDA(cudaEventRecord(waitEvent_, waitStream_));
      CHECK_CUDA(cudaStreamWaitEvent(putStream_, waitEvent_));
      CHECK_CUDA(cudaStreamWaitEvent(mainStream, waitEvent_));
    }

    char* chunkAPtr =
        (reinterpret_cast<char*>(ncclBufPtr_) +
         gemmChunkID * gemmBlockSize * bytesTensorA);
    at::Tensor chunkTensorA = torch::from_blob(
        chunkAPtr, {A.size(0) * gemmBlockSize, A.size(1)}, A.options());

    char* chunkDPtr =
        (reinterpret_cast<char*>(D.data_ptr()) +
         gemmChunkID * gemmBlockSize * chunkBytesTensorD);
    at::Tensor chunkTensorD = torch::from_blob(
        chunkDPtr, {A.size(0) * gemmBlockSize, D.size(1)}, D.options());

    gemm_(chunkTensorD, chunkTensorA, B);

    if (blockSize < tpSize_) {
      const int64_t peerRankID = (tpID_ ^ blockSize);
      const int64_t chunkID = (tpID_ / blockSize);
      const size_t bufOffset = chunkID * blockSize * bytesTensorA;
      CHECK_NCCL(ncclPutSignal(
          reinterpret_cast<char*>(ncclBufPtr_) + bufOffset,
          A.numel() * blockSize,
          ncclDtype,
          peerRankID,
          bufOffset / A.element_size(),
          win_,
          putStream_));
      CHECK_NCCL(ncclWaitSignal(peerRankID, win_, waitStream_));

      // Next chunk of GEMM computation
      gemmChunkID = (peerRankID / blockSize);
      gemmBlockSize = blockSize;
    }
  }

  if (localRowScale.has_value() && rowScale.has_value()) {
    CHECK_NCCL(ncclAllGather(
        localRowScale.value().data_ptr(),
        rowScale.value().data_ptr(),
        localRowScale.value().numel(),
        getNcclDataType_(rowScale.value().scalar_type()),
        comm_,
        putStream_));
    CHECK_CUDA(cudaEventRecord(waitEvent_, putStream_));
    CHECK_CUDA(cudaStreamWaitEvent(mainStream, waitEvent_));
  }

  at::Tensor allGatheredA = torch::from_blob(
      ncclBufPtr_, {tpSize_ * A.size(0), A.size(1)}, A.options());
  return {allGatheredA, D};
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
  const ncclDataType_t ncclDtype = getNcclDataType_(A.scalar_type());
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

    gemm_(chunkTensorD, chunkTensorA, B);

    if (step != (tpSize_ - 1)) {
      CHECK_NCCL(ncclPutSignal(
          chunkAPtr,
          A.numel(),
          ncclDtype,
          prevRankID,
          chunkID * A.numel(),
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
    bool barrier,
    bool skipReduce,
    c10::optional<at::Tensor> rowScale,
    c10::optional<at::Tensor> colScale) {
  assert(!transa); // Current impl assumes A layout is not transposed
  const bool hasScale = (rowScale.has_value() || colScale.has_value());
  if (hasScale) {
    assert(rowScale.has_value() && colScale.has_value());
    assert((D.size(0) * tpSize_) == rowScale.value().numel());
    assert(D.size(1) == colScale.value().numel());
  }
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
  const ncclDataType_t ncclDtype = getNcclDataType_(D.scalar_type());
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
    gemm_(chunkDTensor, chunkATensor, B);
    if (hasScale) {
      char* chunkRowScalePtr =
          (reinterpret_cast<char*>(rowScale.value().data_ptr()) +
           chunkID * D.size(0) * rowScale.value().element_size());
      at::Tensor chunkRowScaleTensor = torch::from_blob(
          chunkRowScalePtr, {D.size(0)}, rowScale.value().options());
      chunkDTensor = row_col_rescale(
          chunkDTensor, chunkRowScaleTensor, colScale.value(), chunkDTensor);
    }

    CHECK_CUDA(cudaEventRecord(waitEvent_, mainStream));
    CHECK_CUDA(cudaStreamWaitEvent(putStream_, waitEvent_));
    CHECK_CUDA(cudaStreamWaitEvent(waitStream_, waitEvent_));

    CHECK_NCCL(ncclPutSignal(
        chunkDPtr,
        D.numel(),
        ncclDtype,
        chunkID,
        (tpSize_ + step) * D.numel(),
        win_,
        putStream_));
    CHECK_NCCL(ncclWaitSignal(srcRankID, win_, waitStream_));

    if ((step != (tpSize_ - 1)) && avoidInCastCongestion_) {
      CHECK_NCCL(ncclPutSignal(
          chunkDPtr,
          0, // send count is 0 to launch kernelNotify
          ncclDtype,
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
  gemm_(lastChunkTensorD, lastChunkTensorA, B);
  if (hasScale) {
    char* chunkRowScalePtr =
        (reinterpret_cast<char*>(rowScale.value().data_ptr()) +
         tpID_ * D.size(0) * rowScale.value().element_size());
    at::Tensor chunkRowScaleTensor = torch::from_blob(
        chunkRowScalePtr, {D.size(0)}, rowScale.value().options());
    lastChunkTensorD = row_col_rescale(
        lastChunkTensorD,
        chunkRowScaleTensor,
        colScale.value(),
        lastChunkTensorD);
  }

  if (!skipReduce) {
    CHECK_CUDA(cudaEventRecord(waitEvent_, waitStream_));
    CHECK_CUDA(cudaStreamWaitEvent(mainStream, waitEvent_));

    at::Tensor partialD = torch::from_blob(
        reinterpret_cast<char*>(ncclBufPtr_) + tpSize_ * bytesTensorD,
        {tpSize_, D.size(0), D.size(1)},
        D.options());
    at::sum_out(D, partialD, /*dim=*/{0});
  }
  return D;
}

size_t FusedCommComp::getDataTypebytes_(const at::ScalarType& type) {
  switch (type) {
    case at::ScalarType::Float8_e4m3fn:
      return 1;
    case at::ScalarType::BFloat16:
      return 2;
    case at::ScalarType::Float:
      return 4;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

ncclDataType_t FusedCommComp::getNcclDataType_(const at::ScalarType& type) {
  switch (type) {
    case at::ScalarType::Float8_e4m3fn:
      return ncclUint8;
    case at::ScalarType::BFloat16:
      return ncclBfloat16;
    case at::ScalarType::Float:
      return ncclFloat32;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

at::Tensor FusedCommComp::rmaWinAllGatherTree(
    at::Tensor A,
    int64_t offsetElements) {
  assert(tpSize_ <= 8); // We only support single node now
  assert(checkInputTensor_(A));

  const size_t bytesTensorA = A.numel() * A.element_size();
  const size_t offsetBytes = offsetElements * A.element_size();
  assert((offsetBytes + bytesTensorA) <= buffBytes_);
  char* internalBufPtr = reinterpret_cast<char*>(ncclBufPtr_) + offsetBytes;

  cudaStream_t mainStream = ((cudaStream_t)at::cuda::getDefaultCUDAStream());
  CHECK_CUDA(cudaMemcpyAsync(
      internalBufPtr + tpID_ * bytesTensorA,
      A.data_ptr(),
      bytesTensorA,
      cudaMemcpyDeviceToDevice,
      mainStream));

  // Adding dependencies for comm streams to wait for compute data
  internalBarrier_(putStream_);
  CHECK_CUDA(cudaEventRecord(waitEvent_, mainStream));
  CHECK_CUDA(cudaStreamWaitEvent(waitStream_, waitEvent_));

  const ncclDataType_t ncclDtype = getNcclDataType_(A.scalar_type());
  for (int blockSize = 1; blockSize < tpSize_; blockSize *= 2) {
    if (blockSize != 1) {
      // Wait for the peer data chunk to arrive in the last step
      CHECK_CUDA(cudaEventRecord(waitEvent_, waitStream_));
      CHECK_CUDA(cudaStreamWaitEvent(putStream_, waitEvent_));
    }

    const int64_t peerRankID = (tpID_ ^ blockSize);
    const int64_t chunkID = (tpID_ / blockSize);
    const size_t bufOffset = chunkID * blockSize * bytesTensorA;
    CHECK_NCCL(ncclPutSignal(
        internalBufPtr + bufOffset,
        A.numel() * blockSize,
        ncclDtype,
        peerRankID,
        // NCCL window target display offset is an absolute offset in elements
        (bufOffset + offsetBytes) / A.element_size(),
        win_,
        putStream_));
    CHECK_NCCL(ncclWaitSignal(peerRankID, win_, waitStream_));
  }

  at::Tensor allGatheredA = torch::from_blob(
      internalBufPtr, {tpSize_ * A.size(0), A.size(1)}, A.options());
  return allGatheredA;
}

at::Tensor FusedCommComp::rmaWinAllGather(
    at::Tensor A,
    int64_t offsetElements) {
  assert(tpSize_ <= 8); // We only support single node now
  assert(checkInputTensor_(A));

  const size_t bytesTensorA = A.numel() * A.element_size();
  const size_t offsetBytes = offsetElements * A.element_size();
  assert((offsetBytes + bytesTensorA) <= buffBytes_);
  char* internalBufPtr = reinterpret_cast<char*>(ncclBufPtr_) + offsetBytes;

  cudaStream_t mainStream = ((cudaStream_t)at::cuda::getDefaultCUDAStream());
  CHECK_CUDA(cudaMemcpyAsync(
      internalBufPtr + tpID_ * bytesTensorA,
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
  const ncclDataType_t ncclDtype = getNcclDataType_(A.scalar_type());
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
        internalBufPtr + bufOffset,
        A.numel(),
        ncclDtype,
        prevRankID,
        // NCCL window target display offset is an absolute offset in elements
        (bufOffset + offsetBytes) / A.element_size(),
        win_,
        putStream_));
    CHECK_NCCL(ncclWaitSignal(nextRankID, win_, waitStream_));
  }

  at::Tensor allGatheredA = torch::from_blob(
      internalBufPtr, {tpSize_ * A.size(0), A.size(1)}, A.options());
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

void FusedCommComp::rmaWinScatter(at::Tensor A, int64_t offsetElements) {
  assert(tpSize_ <= 8); // We only support single node now
  assert(checkInputTensor_(A));

  const size_t bytesTensorA = A.numel() * A.element_size();
  const size_t offsetBytes = offsetElements * A.element_size();
  assert((offsetBytes + bytesTensorA) <= buffBytes_);

  const size_t chunkBytesTensorA = (bytesTensorA / tpSize_);
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
  const ncclDataType_t ncclDtype = getNcclDataType_(A.scalar_type());
  for (int step = 1; step < tpSize_; ++step) {
    const int64_t chunkID = (tpID_ + step) % tpSize_;
    const int64_t srcRankID = (tpID_ - step + tpSize_) % tpSize_;

    // Scatter results to the destination rank
    CHECK_NCCL(ncclPutSignal(
        reinterpret_cast<char*>(A.data_ptr()) + chunkID * chunkBytesTensorA,
        (A.numel() / tpSize_),
        ncclDtype,
        chunkID,
        (offsetBytes + step * chunkBytesTensorA) / A.element_size(),
        win_,
        putStream_));
    CHECK_NCCL(ncclWaitSignal(srcRankID, win_, waitStream_));

    if ((step != (tpSize_ - 1)) && (avoidInCastCongestion_)) {
      // Notify the prev rank the completion to avoid the congestion
      CHECK_NCCL(ncclPutSignal(
          A.data_ptr(),
          0, // send count is 0 to launch kernelNotify
          ncclDtype,
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

void FusedCommComp::localReduceIntoTensor(
    at::Tensor output,
    at::Tensor input,
    int64_t offsetElements) {
  assert(tpSize_ <= 8); // We only support single node now
  assert(checkInputTensor_(input));

  const size_t bytesTensorInp = input.numel() * input.element_size();
  const size_t offsetBytes = offsetElements * input.element_size();
  assert((offsetBytes + bytesTensorInp) <= buffBytes_);

  const size_t chunkBytesTensorInp = (bytesTensorInp / tpSize_);

  waitForCommStream();
  cudaStream_t mainStream = ((cudaStream_t)at::cuda::getDefaultCUDAStream());
  CHECK_CUDA(cudaMemcpyAsync(
      reinterpret_cast<char*>(ncclBufPtr_) + offsetBytes,
      reinterpret_cast<char*>(input.data_ptr()) + tpID_ * chunkBytesTensorInp,
      chunkBytesTensorInp,
      cudaMemcpyDeviceToDevice,
      mainStream));

  at::Tensor partialOut = torch::from_blob(
      reinterpret_cast<char*>(ncclBufPtr_) + offsetBytes,
      {tpSize_, output.size(0), output.size(1)},
      output.options());

  at::sum_out(output, partialOut, /*dim=*/{0});
  return;
}

} // namespace tensor_parallel
} // namespace fb
} // namespace gen_ai
} // namespace fbgemm_gpu

using namespace fbgemm_gpu::gen_ai::fb::tensor_parallel;

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "row_col_rescale(Tensor input, Tensor row_scale, Tensor col_scale, Tensor? output=None) -> Tensor");
  m.impl("row_col_rescale", row_col_rescale);
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("row_col_rescale", row_col_rescale);
}

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
      .def_readwrite("fp8_gemm_fast_accum", &FusedCommComp::fp8GEMMFastAccum)
      .def("_gemm", &FusedCommComp::gemm_)
      .def("internal_buffer_like", &FusedCommComp::internalBufferLike)
      .def("get_internal_buffer", &FusedCommComp::getInternalBuffer)
      .def(
          "rma_win_all_gather",
          &FusedCommComp::rmaWinAllGather,
          "",
          {torch::arg("A"), torch::arg("offsetElements") = 0})
      .def(
          "rma_win_all_gather_tree",
          &FusedCommComp::rmaWinAllGatherTree,
          "",
          {torch::arg("A"), torch::arg("offsetElements") = 0})
      .def("wait_for_comm_stream", &FusedCommComp::waitForCommStream)
      .def(
          "rma_win_scatter",
          &FusedCommComp::rmaWinScatter,
          "",
          {torch::arg("A"), torch::arg("offsetElements") = 0})
      .def("avoid_incast_congestion", &FusedCommComp::avoidInCastCongestion)
      .def(
          "local_reduce_into_tensor",
          &FusedCommComp::localReduceIntoTensor,
          "",
          {torch::arg("output"),
           torch::arg("input"),
           torch::arg("offsetElements") = 0})
      .def("split_overlap_ag", &FusedCommComp::splitOverlapAllGather)
      .def(
          "split_overlap_ag_tree",
          &FusedCommComp::splitOverlapAllGatherTree,
          "",
          {torch::arg("A"),
           torch::arg("transa"),
           torch::arg("B"),
           torch::arg("transb"),
           torch::arg("D"),
           torch::arg("inplaceA"),
           torch::arg("barrier"),
           torch::arg("localRowScale") = c10::nullopt,
           torch::arg("rowScale") = c10::nullopt})
      .def(
          "split_overlap_rs",
          &FusedCommComp::splitOverlapReduceScatter,
          "",
          {torch::arg("A"),
           torch::arg("transa"),
           torch::arg("B"),
           torch::arg("transb"),
           torch::arg("D"),
           torch::arg("barrier"),
           torch::arg("skipReduce") = false,
           torch::arg("rowScale") = c10::nullopt,
           torch::arg("colScale") = c10::nullopt});
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                     \
  do {                                                                      \
    const hipError_t error = (expr);                                        \
    if (error != hipSuccess) {                                              \
      throw std::runtime_error(std::string(#expr) + ": " +                \
                               hipGetErrorString(error));                    \
    }                                                                       \
  } while (false)

namespace {

constexpr int kBatchSize = 384;
constexpr int kNumFeatures = 2;
constexpr int kLengthsSize = kBatchSize * kNumFeatures;
constexpr int kWorldSize = 160;
constexpr int64_t kTotalNumBlocks = 3'991'680;
constexpr int64_t kNumIndices = 2'807'700;
constexpr int64_t kTargetIndices = 1'619;
constexpr int64_t kNumEmbeddings = 1'125'899'902'955'520;
constexpr int kLegacyThreads = 256;

template <int WeightDim>
__global__ void legacyKernel(
    int lengthsSize,
    int batchSize,
    int mySize,
    const int64_t* blockSizes,
    const int64_t* totalNumBlocks,
    const int64_t* offsets,
    const int64_t* indices,
    const float* weights,
    int64_t* newOffsets,
    int64_t* newIndices,
    float* newWeights,
    int64_t* unbucketizePermute) {
  for (int segment = blockIdx.x * blockDim.x + threadIdx.x;
       segment < lengthsSize;
       segment += blockDim.x * gridDim.x) {
    const int feature = segment / batchSize;
    const uint64_t blockSize = blockSizes[feature];
    const uint64_t globalNumBlocks = totalNumBlocks[feature];
    const uint64_t localNumBlocks = globalNumBlocks / mySize;
    const uint64_t globalIndexSize = blockSize * globalNumBlocks;
    const uint64_t localIndexSize = blockSize * localNumBlocks;
    const int64_t rowStart = segment == 0 ? 0 : offsets[segment - 1];
    const int64_t rowEnd = offsets[segment];
    for (int64_t i = rowStart; i < rowEnd; ++i) {
      const uint64_t index = static_cast<uint64_t>(indices[i]);
      const uint64_t bucket = index < globalIndexSize
          ? index / localIndexSize
          : (index % globalNumBlocks) / localNumBlocks;
      const uint64_t newIndex = index < globalIndexSize
          ? index % localIndexSize
          : index / globalNumBlocks;
      const uint64_t position = newOffsets[bucket * lengthsSize + segment]++;
      newIndices[position] = newIndex;
      unbucketizePermute[i] = position;
#pragma unroll
      for (int d = 0; d < WeightDim; ++d) {
        newWeights[position * WeightDim + d] = weights[i * WeightDim + d];
      }
    }
  }
}

template <int WeightDim>
__global__ void waveKernel(
    int lengthsSize,
    int batchSize,
    int mySize,
    const int64_t* blockSizes,
    const int64_t* totalNumBlocks,
    const int64_t* offsets,
    const int64_t* indices,
    const float* weights,
    const int64_t* newOffsets,
    int64_t* newIndices,
    float* newWeights,
    int64_t* unbucketizePermute) {
  extern __shared__ int64_t bucketOffsets[];
  const int lane = threadIdx.x;
  const unsigned long long laneMaskLt =
      lane == 0 ? 0 : ((uint64_t{1} << lane) - 1);
  const unsigned long long activeMask = __activemask();

  for (int segment = blockIdx.x; segment < lengthsSize;
       segment += gridDim.x) {
    for (int bucket = lane; bucket < mySize; bucket += warpSize) {
      bucketOffsets[bucket] = newOffsets[bucket * lengthsSize + segment];
    }
    __syncthreads();

    const int feature = segment / batchSize;
    const uint64_t blockSize = blockSizes[feature];
    const uint64_t globalNumBlocks = totalNumBlocks[feature];
    const uint64_t localNumBlocks = globalNumBlocks / mySize;
    const uint64_t globalIndexSize = blockSize * globalNumBlocks;
    const uint64_t localIndexSize = blockSize * localNumBlocks;
    const int64_t rowStart = segment == 0 ? 0 : offsets[segment - 1];
    const int64_t rowEnd = offsets[segment];
    for (int64_t chunk = rowStart; chunk < rowEnd; chunk += warpSize) {
      const int64_t i = chunk + lane;
      const bool valid = i < rowEnd;
      const uint64_t index = valid ? static_cast<uint64_t>(indices[i]) : 0;
      const uint64_t bucket = valid
          ? (index < globalIndexSize
                 ? index / localIndexSize
                 : (index % globalNumBlocks) / localNumBlocks)
          : mySize;
      const uint64_t newIndex = index < globalIndexSize
          ? index % localIndexSize
          : index / globalNumBlocks;
      const unsigned long long peers = __match_any_sync(activeMask, bucket);
      const int leader = __ffsll(peers) - 1;
      const int rank = __popcll(peers & laneMaskLt);
      int64_t base = 0;
      if (valid && lane == leader) {
        base = bucketOffsets[bucket];
        bucketOffsets[bucket] += __popcll(peers);
      }
      base = __shfl(base, leader);
      if (valid) {
        const int64_t position = base + rank;
        newIndices[position] = newIndex;
        unbucketizePermute[i] = position;
#pragma unroll
        for (int d = 0; d < WeightDim; ++d) {
          newWeights[position * WeightDim + d] = weights[i * WeightDim + d];
        }
      }
    }
    __syncthreads();
  }
}

template <int WeightDim>
__global__ void waveGlobalCursorKernel(
    int lengthsSize,
    int batchSize,
    int mySize,
    const int64_t* blockSizes,
    const int64_t* totalNumBlocks,
    const int64_t* offsets,
    const int64_t* indices,
    const float* weights,
    int64_t* newOffsets,
    int64_t* newIndices,
    float* newWeights,
    int64_t* unbucketizePermute) {
  const int lane = threadIdx.x;
  const unsigned long long laneMaskLt =
      lane == 0 ? 0 : ((uint64_t{1} << lane) - 1);
  const unsigned long long activeMask = __activemask();
  for (int segment = blockIdx.x; segment < lengthsSize;
       segment += gridDim.x) {
    const int feature = segment / batchSize;
    const uint64_t blockSize = blockSizes[feature];
    const uint64_t globalNumBlocks = totalNumBlocks[feature];
    const uint64_t localNumBlocks = globalNumBlocks / mySize;
    const uint64_t globalIndexSize = blockSize * globalNumBlocks;
    const uint64_t localIndexSize = blockSize * localNumBlocks;
    const int64_t rowStart = segment == 0 ? 0 : offsets[segment - 1];
    const int64_t rowEnd = offsets[segment];
    for (int64_t chunk = rowStart; chunk < rowEnd; chunk += warpSize) {
      const int64_t i = chunk + lane;
      const bool valid = i < rowEnd;
      const uint64_t index = valid ? static_cast<uint64_t>(indices[i]) : 0;
      const uint64_t bucket = valid
          ? (index < globalIndexSize
                 ? index / localIndexSize
                 : (index % globalNumBlocks) / localNumBlocks)
          : mySize;
      const uint64_t newIndex = index < globalIndexSize
          ? index % localIndexSize
          : index / globalNumBlocks;
      const unsigned long long peers = __match_any_sync(activeMask, bucket);
      const int leader = __ffsll(peers) - 1;
      const int rank = __popcll(peers & laneMaskLt);
      int64_t base = 0;
      if (valid && lane == leader) {
        const int64_t offsetIndex = bucket * lengthsSize + segment;
        base = newOffsets[offsetIndex];
        newOffsets[offsetIndex] += __popcll(peers);
      }
      base = __shfl(base, leader);
      if (valid) {
        const int64_t position = base + rank;
        newIndices[position] = newIndex;
        unbucketizePermute[i] = position;
#pragma unroll
        for (int d = 0; d < WeightDim; ++d) {
          newWeights[position * WeightDim + d] = weights[i * WeightDim + d];
        }
      }
    }
  }
}

std::vector<int64_t> balancedLengths(int rows, int64_t values) {
  std::vector<int64_t> result(rows, values / rows);
  for (int64_t i = 0; i < values % rows; ++i) {
    ++result[i];
  }
  return result;
}

template <int WeightDim, typename Launch>
float timeKernel(Launch launch, int warmups, int iterations) {
  for (int i = 0; i < warmups; ++i) {
    launch();
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    hipEvent_t start;
    hipEvent_t end;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(start));
    launch();
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));
    float elapsed = 0;
    HIP_CHECK(hipEventElapsedTime(&elapsed, start, end));
    samples.push_back(elapsed);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(end));
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

template <typename T>
T* copyToDevice(const std::vector<T>& host) {
  T* device = nullptr;
  HIP_CHECK(hipMalloc(&device, host.size() * sizeof(T)));
  HIP_CHECK(hipMemcpy(
      device,
      host.data(),
      host.size() * sizeof(T),
      hipMemcpyHostToDevice));
  return device;
}

template <int WeightDim>
void runCase(
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& indices,
    const std::vector<float>& weights,
    const std::vector<int64_t>& initialNewOffsets,
    const int64_t* deviceBlockSizes,
    const int64_t* deviceTotalNumBlocks,
    const int64_t* deviceOffsets,
    const int64_t* deviceIndices,
    const float* deviceWeights,
    int warmups,
    int iterations) {
  const size_t indexBytes = indices.size() * sizeof(int64_t);
  const size_t weightBytes = weights.size() * sizeof(float);
  const size_t offsetBytes = initialNewOffsets.size() * sizeof(int64_t);
  int64_t* deviceInitialNewOffsets = copyToDevice(initialNewOffsets);
  int64_t* legacyOffsets = nullptr;
  int64_t* waveOffsets = nullptr;
  int64_t* waveGlobalOffsets = nullptr;
  int64_t* legacyIndices = nullptr;
  int64_t* waveIndices = nullptr;
  int64_t* waveGlobalIndices = nullptr;
  int64_t* legacyPermute = nullptr;
  int64_t* wavePermute = nullptr;
  int64_t* waveGlobalPermute = nullptr;
  float* legacyWeights = nullptr;
  float* waveWeights = nullptr;
  float* waveGlobalWeights = nullptr;
  HIP_CHECK(hipMalloc(&legacyOffsets, offsetBytes));
  HIP_CHECK(hipMalloc(&waveOffsets, offsetBytes));
  HIP_CHECK(hipMalloc(&waveGlobalOffsets, offsetBytes));
  HIP_CHECK(hipMalloc(&legacyIndices, indexBytes));
  HIP_CHECK(hipMalloc(&waveIndices, indexBytes));
  HIP_CHECK(hipMalloc(&waveGlobalIndices, indexBytes));
  HIP_CHECK(hipMalloc(&legacyPermute, indexBytes));
  HIP_CHECK(hipMalloc(&wavePermute, indexBytes));
  HIP_CHECK(hipMalloc(&waveGlobalPermute, indexBytes));
  HIP_CHECK(hipMalloc(&legacyWeights, weightBytes));
  HIP_CHECK(hipMalloc(&waveWeights, weightBytes));
  HIP_CHECK(hipMalloc(&waveGlobalWeights, weightBytes));

  auto launchLegacy = [&]() {
    HIP_CHECK(hipMemcpyAsync(
        legacyOffsets,
        deviceInitialNewOffsets,
        offsetBytes,
        hipMemcpyDeviceToDevice));
    hipLaunchKernelGGL(
        (legacyKernel<WeightDim>),
        dim3((kLengthsSize + kLegacyThreads - 1) / kLegacyThreads),
        dim3(kLegacyThreads),
        0,
        0,
        kLengthsSize,
        kBatchSize,
        kWorldSize,
        deviceBlockSizes,
        deviceTotalNumBlocks,
        deviceOffsets,
        deviceIndices,
        deviceWeights,
        legacyOffsets,
        legacyIndices,
        legacyWeights,
        legacyPermute);
    HIP_CHECK(hipGetLastError());
  };
  auto launchWave = [&]() {
    HIP_CHECK(hipMemcpyAsync(
        waveOffsets,
        deviceInitialNewOffsets,
        offsetBytes,
        hipMemcpyDeviceToDevice));
    hipLaunchKernelGGL(
        (waveKernel<WeightDim>),
        dim3(kLengthsSize),
        dim3(64),
        kWorldSize * sizeof(int64_t),
        0,
        kLengthsSize,
        kBatchSize,
        kWorldSize,
        deviceBlockSizes,
        deviceTotalNumBlocks,
        deviceOffsets,
        deviceIndices,
        deviceWeights,
        waveOffsets,
        waveIndices,
        waveWeights,
        wavePermute);
    HIP_CHECK(hipGetLastError());
  };
  auto launchWaveGlobal = [&]() {
    HIP_CHECK(hipMemcpyAsync(
        waveGlobalOffsets,
        deviceInitialNewOffsets,
        offsetBytes,
        hipMemcpyDeviceToDevice));
    hipLaunchKernelGGL(
        (waveGlobalCursorKernel<WeightDim>),
        dim3(kLengthsSize),
        dim3(64),
        0,
        0,
        kLengthsSize,
        kBatchSize,
        kWorldSize,
        deviceBlockSizes,
        deviceTotalNumBlocks,
        deviceOffsets,
        deviceIndices,
        deviceWeights,
        waveGlobalOffsets,
        waveGlobalIndices,
        waveGlobalWeights,
        waveGlobalPermute);
    HIP_CHECK(hipGetLastError());
  };

  launchLegacy();
  launchWave();
  launchWaveGlobal();
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<int64_t> legacyIndicesHost(indices.size());
  std::vector<int64_t> waveIndicesHost(indices.size());
  std::vector<int64_t> waveGlobalIndicesHost(indices.size());
  std::vector<int64_t> legacyPermuteHost(indices.size());
  std::vector<int64_t> wavePermuteHost(indices.size());
  std::vector<int64_t> waveGlobalPermuteHost(indices.size());
  std::vector<float> legacyWeightsHost(weights.size());
  std::vector<float> waveWeightsHost(weights.size());
  std::vector<float> waveGlobalWeightsHost(weights.size());
  HIP_CHECK(hipMemcpy(
      legacyIndicesHost.data(), legacyIndices, indexBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(
      waveIndicesHost.data(), waveIndices, indexBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(
      waveGlobalIndicesHost.data(),
      waveGlobalIndices,
      indexBytes,
      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(
      legacyPermuteHost.data(), legacyPermute, indexBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(
      wavePermuteHost.data(), wavePermute, indexBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(
      waveGlobalPermuteHost.data(),
      waveGlobalPermute,
      indexBytes,
      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(
      legacyWeightsHost.data(), legacyWeights, weightBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(
      waveWeightsHost.data(), waveWeights, weightBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(
      waveGlobalWeightsHost.data(),
      waveGlobalWeights,
      weightBytes,
      hipMemcpyDeviceToHost));
  if (legacyIndicesHost != waveIndicesHost ||
      legacyIndicesHost != waveGlobalIndicesHost ||
      legacyPermuteHost != wavePermuteHost ||
      legacyPermuteHost != waveGlobalPermuteHost ||
      legacyWeightsHost != waveWeightsHost) {
    throw std::runtime_error("wave kernel output differs from legacy output");
  }
  if (legacyWeightsHost != waveGlobalWeightsHost) {
    throw std::runtime_error("wave-global kernel output differs from legacy output");
  }

  const float legacyMs =
      timeKernel<WeightDim>(launchLegacy, warmups, iterations);
  const float waveMs = timeKernel<WeightDim>(launchWave, warmups, iterations);
  const float waveGlobalMs =
      timeKernel<WeightDim>(launchWaveGlobal, warmups, iterations);
  std::cout << "weights_dim=" << WeightDim << " legacy_ms=" << legacyMs
            << " wave_global_ms=" << waveGlobalMs
            << " wave_shared_ms=" << waveMs
            << " shared_speedup=" << legacyMs / waveMs << "x" << std::endl;

  HIP_CHECK(hipFree(deviceInitialNewOffsets));
  HIP_CHECK(hipFree(legacyOffsets));
  HIP_CHECK(hipFree(waveOffsets));
  HIP_CHECK(hipFree(waveGlobalOffsets));
  HIP_CHECK(hipFree(legacyIndices));
  HIP_CHECK(hipFree(waveIndices));
  HIP_CHECK(hipFree(waveGlobalIndices));
  HIP_CHECK(hipFree(legacyPermute));
  HIP_CHECK(hipFree(wavePermute));
  HIP_CHECK(hipFree(waveGlobalPermute));
  HIP_CHECK(hipFree(legacyWeights));
  HIP_CHECK(hipFree(waveWeights));
  HIP_CHECK(hipFree(waveGlobalWeights));
}

} // namespace

int main(int argc, char** argv) {
  const int warmups = argc > 1 ? std::atoi(argv[1]) : 5;
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
  hipDeviceProp_t properties;
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  std::cout << "device=" << properties.name << " warp_size=" << properties.warpSize
            << " compute_units=" << properties.multiProcessorCount << std::endl;
  if (properties.warpSize != 64) {
    throw std::runtime_error("This production-shape benchmark expects wave64");
  }

  std::vector<int64_t> lengths =
      balancedLengths(kBatchSize, kNumIndices - kTargetIndices);
  const auto targetLengths = balancedLengths(kBatchSize, kTargetIndices);
  lengths.insert(lengths.end(), targetLengths.begin(), targetLengths.end());
  std::vector<int64_t> offsets(kLengthsSize);
  std::partial_sum(lengths.begin(), lengths.end(), offsets.begin());

  std::mt19937_64 generator(20260928);
  std::uniform_int_distribution<int64_t> distribution(0, kNumEmbeddings - 1);
  std::vector<int64_t> indices(kNumIndices);
  std::vector<float> weights2d(kNumIndices * 2);
  for (int64_t i = 0; i < kNumIndices; ++i) {
    indices[i] = distribution(generator);
    const uint64_t raw = generator();
    weights2d[2 * i] = static_cast<float>(raw & 0xffff);
    weights2d[2 * i + 1] = static_cast<float>(raw >> 32);
  }
  const int64_t blockSize =
      (kNumEmbeddings + kTotalNumBlocks - 1) / kTotalNumBlocks;
  const std::vector<int64_t> blockSizes(kNumFeatures, blockSize);
  const std::vector<int64_t> totalNumBlocks(
      kNumFeatures, kTotalNumBlocks);

  std::vector<int64_t> counts(kWorldSize * kLengthsSize, 0);
  for (int segment = 0; segment < kLengthsSize; ++segment) {
    const int64_t rowStart = segment == 0 ? 0 : offsets[segment - 1];
    const int64_t rowEnd = offsets[segment];
    for (int64_t i = rowStart; i < rowEnd; ++i) {
      const int64_t localIndexSize = blockSize * (kTotalNumBlocks / kWorldSize);
      const int64_t bucket = indices[i] / localIndexSize;
      ++counts[bucket * kLengthsSize + segment];
    }
  }
  std::vector<int64_t> newOffsets(counts.size());
  std::exclusive_scan(counts.begin(), counts.end(), newOffsets.begin(), 0LL);

  int64_t* deviceBlockSizes = copyToDevice(blockSizes);
  int64_t* deviceTotalNumBlocks = copyToDevice(totalNumBlocks);
  int64_t* deviceOffsets = copyToDevice(offsets);
  int64_t* deviceIndices = copyToDevice(indices);
  float* deviceWeights = copyToDevice(weights2d);

  std::cout << "segments=" << kLengthsSize << " indices=" << kNumIndices
            << " world_size=" << kWorldSize
            << " total_num_blocks=" << kTotalNumBlocks
            << " legacy_grid=3x256 wave_grid=768x64" << std::endl;
  runCase<1>(
      offsets,
      indices,
      std::vector<float>(weights2d.begin(), weights2d.begin() + kNumIndices),
      newOffsets,
      deviceBlockSizes,
      deviceTotalNumBlocks,
      deviceOffsets,
      deviceIndices,
      deviceWeights,
      warmups,
      iterations);
  runCase<2>(
      offsets,
      indices,
      weights2d,
      newOffsets,
      deviceBlockSizes,
      deviceTotalNumBlocks,
      deviceOffsets,
      deviceIndices,
      deviceWeights,
      warmups,
      iterations);

  HIP_CHECK(hipFree(deviceBlockSizes));
  HIP_CHECK(hipFree(deviceTotalNumBlocks));
  HIP_CHECK(hipFree(deviceOffsets));
  HIP_CHECK(hipFree(deviceIndices));
  HIP_CHECK(hipFree(deviceWeights));
  return 0;
}

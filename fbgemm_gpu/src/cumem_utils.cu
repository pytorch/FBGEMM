/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"

using namespace at;
using namespace fbgemm_gpu;

// Registered CUDA managed memory (UVM) deleter
static void CUDAManagedDeleter(void* ptr) {
  AT_CUDA_CHECK(cudaFree(ptr));
}

// Wrapper for CUDA managed memory (UVM) allocator
struct CUDAManagedAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void* ptr;
    AT_CUDA_CHECK(cudaMallocManaged(&ptr, size));
    // User hints with "preferred location": Here the kernel will page fault
    // and generate direct mapping to data on the CPU.
    AT_CUDA_CHECK(cudaMemAdvise(
        ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    // User hints with "accessed by": GPU will establish direct mapping of data
    // in CPU memory, no page faults will be generated
    AT_CUDA_CHECK(cudaMemAdvise(
        ptr, size, cudaMemAdviseSetAccessedBy, at::cuda::current_device()));
    return {ptr,
            ptr,
            &CUDAManagedDeleter,
            {at::DeviceType::CUDA, at::cuda::current_device()}};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &CUDAManagedDeleter;
  }
};

// Registered CUDA host-mapped memory deleter
static void CUDAHostMappedDeleter(void* ptr) {
  AT_CUDA_CHECK(cudaFreeHost(ptr));
}

// Wrapper for CUDA host-mapped memory allocator
struct CUDAHostMappedAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void* ptr;
    AT_CUDA_CHECK(cudaHostAlloc(
        &ptr, size, cudaHostAllocWriteCombined | cudaHostAllocMapped));

    void* dev_ptr;
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, ptr, 0));
    return {dev_ptr,
            ptr,
            &CUDAHostMappedDeleter,
            {at::DeviceType::CUDA, at::cuda::current_device()}};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &CUDAHostMappedDeleter;
  }
};

static CUDAManagedAllocator g_managed_allocator;
static CUDAHostMappedAllocator g_host_mapped_allocator;

// Get the default strides from the input Tensor dimensions
std::vector<int64_t> defaultStrides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (size_t i = sizes.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= sizes[i - 1];
  }
  return strides;
}

// Allocate the ATen Tensor with unified managed memory (UVM)
Tensor new_managed_tensor(Tensor self, std::vector<std::int64_t> sizes) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(self.get_device());

  auto strides = defaultStrides(sizes);
  auto storage = Storage(
      Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(sizes, strides, self.dtype().itemsize()),
      &g_managed_allocator,
      /*resizable=*/false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, sizes, strides);
  return tensor;
}

// Allocate the ATen Tensor with host-mapped memory
Tensor new_host_mapped_tensor(Tensor self, std::vector<std::int64_t> sizes) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(self.get_device());

  auto strides = defaultStrides(sizes);
  auto storage = Storage(
      Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(sizes, strides, self.dtype().itemsize()),
      &g_host_mapped_allocator,
      /*resizable=*/false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, sizes, strides);
  return tensor;
}

// Check if a tensor is allocated with UVM or host-mapped memory
bool is_uvm_tensor(Tensor t) {
  if (t.device().is_cpu()) {
    return false;
  }
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t.get_device());

  return t.storage().allocator() == &g_managed_allocator ||
      t.storage().allocator() == &g_host_mapped_allocator;
}

// Convert a UVM tensor to a CPU tensor
Tensor uvm_to_cpu(Tensor t) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t.get_device());

  TORCH_CHECK(is_uvm_tensor(t));
  // not copy the storage
  return at::from_blob(
      t.data_ptr(), t.sizes(), t.strides(), t.options().device(kCPU));
}

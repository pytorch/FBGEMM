/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

#include "cumem_utils.h"
#include "fbgemm_gpu/enum_utils.h"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Freeing host/uvm memory with cudaFree[Host] requires a cuda context.
// If a uvm tensor is released from an arbitrary thread without a context
// then cuda helpfully create a new default context on the default device.
// If we have not used the default device before in this process cuda
// needs to also allocate a device context. However creating a device
// context requires device resources and may fail with out of memory error
// causing  cudaFree[Host] to fail with out of memory error.
// The solution is simply to remember the device from the allocation context
// and set the correct device in the thread before calling cudaFree[Host]

namespace {
struct CUDAHostMappedContext {
  void* ptr_;
  int cuda_device_;

  CUDAHostMappedContext(void* ptr, int cuda_device)
      : ptr_(ptr), cuda_device_(cuda_device){};

  ~CUDAHostMappedContext() {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(cuda_device_);
    AT_CUDA_CHECK(cudaHostUnregister(ptr_));
    free(ptr_);
  }

  static void release(void* ptr) {
    delete static_cast<CUDAHostMappedContext*>(ptr);
  }
};

struct CUDAManagedContext {
  void* ptr_;
  int cuda_device_;

  CUDAManagedContext(void* ptr, int cuda_device)
      : ptr_(ptr), cuda_device_(cuda_device){};

  ~CUDAManagedContext() {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(cuda_device_);
    AT_CUDA_CHECK(cudaFree(ptr_));
  }

  static void release(void* ptr) {
    delete static_cast<CUDAManagedContext*>(ptr);
  }
};

// Keep a reference to the UVM memory allocation from the associated
// CPU Tensor to prevent lifetime issues (use after free)
struct CUDAManagedIndirectContext {
  Storage storage_;

  CUDAManagedIndirectContext(Storage storage) : storage_(std::move(storage)){};

  static void release(void* ptr) {
    delete static_cast<CUDAManagedIndirectContext*>(ptr);
  }
};

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
Tensor new_managed_tensor_internal(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(self.get_device());

  auto strides = defaultStrides(sizes);
  size_t size_bytes =
      at::detail::computeStorageNbytes(sizes, strides, self.dtype().itemsize());
  void* ptr;
  AT_CUDA_CHECK(cudaMallocManaged(&ptr, size_bytes));

  // The memory allocated above can be accessed from CUDA and CPU
  // However Storage requires a specific device and we need to retain the cuda
  // device for releasing the memory (see "Freeing host/uvm memory with cudaFree
  // .." above. To access the memory from devices other then the one used for
  // allocation we need a new Storage object (with the new device) referring to
  // the original storage object. We force this indirection even for newly
  // allocated Tensors for code unification.

  auto real_storage = Storage(
      Storage::use_byte_size_t(),
      size_bytes,
      at::DataPtr(
          ptr,
          new CUDAManagedContext(ptr, self.get_device()),
          &CUDAManagedContext::release,
          {at::DeviceType::CUDA, self.device().index()}),
      nullptr, /* allocator */
      /*resizable=*/false);

  auto indirect_storage = Storage(
      Storage::use_byte_size_t(),
      size_bytes,
      at::DataPtr(
          ptr,
          new CUDAManagedIndirectContext(real_storage),
          &CUDAManagedIndirectContext::release,
          {at::DeviceType::CUDA, self.device().index()}),
      nullptr, /* allocator */
      /*resizable=*/false);

  return at::empty({0}, self.options())
      .set_(indirect_storage, 0, sizes, strides);
}

std::tuple<void*, size_t> adjust_to_page_boundaries(void* ptr, size_t size) {
  static uint64_t page_mask = ([]() -> uint64_t {
    uint64_t page_size = (uint64_t)sysconf(_SC_PAGESIZE);
    return (page_size - 1);
  })();

  uint64_t raw_ptr = (uint64_t)ptr;
  uint64_t raw_ptr_adjusted = raw_ptr & ~page_mask;
  uint64_t raw_ptr_end_adjusted = (raw_ptr + size + page_mask) & ~page_mask;
  uint64_t size_adjusted = raw_ptr_end_adjusted - raw_ptr_adjusted;

  return std::make_tuple((void*)raw_ptr_adjusted, (size_t)size_adjusted);
}

} // namespace

// Allocate a cuda Tensor with unified managed memory (UVM)
// Then set the preferred data location to CPU (host memory)
// And establish mappings on the cuda device to the host memory
Tensor new_managed_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(self.get_device());

  Tensor t = new_managed_tensor_internal(self, sizes);

  void* ptr = t.data_ptr();
  size_t size_bytes = t.storage().nbytes();

  // Set preferred memory location to host memory
  AT_CUDA_CHECK(cudaMemAdvise(
      ptr, size_bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
  // User hints with "accessed by": GPU will establish direct mapping of data
  // in CPU memory, no page faults will be generated
  AT_CUDA_CHECK(cudaMemAdvise(
      ptr, size_bytes, cudaMemAdviseSetAccessedBy, at::cuda::current_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Work around fork issue - see uvm_mem_advice_dont_fork for details
  auto adjusted = adjust_to_page_boundaries(ptr, size_bytes);
  int result =
      madvise(std::get<0>(adjusted), std::get<1>(adjusted), MADV_DONTFORK);
  TORCH_CHECK(result == 0)

  return t;
}

// Allocate a cuda Tensor with unified managed memory (UVM) without the
// additional steps taked by new_managed_tensor above
Tensor new_vanilla_managed_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(self.get_device());

  return new_managed_tensor_internal(self, sizes);
}

// Allocate the ATen Tensor with host-mapped memory
Tensor new_host_mapped_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(self.get_device());

  auto strides = defaultStrides(sizes);
  size_t size_bytes =
      at::detail::computeStorageNbytes(sizes, strides, self.dtype().itemsize());

  // When using cudaHostAlloc for large allocations, we found that it can
  // potentially take a global lock and lock out CUDA APIs from other processes.
  // The main cost in cudaHostAlloc is faulting/mapping the pages. So, instead
  // of using this cuda API, we can do regular malloc, pre-fault the pages, and
  // then do cudaHostRegister with GPU mapping flags to lock the pages, so we
  // can minimize the cost while holding this global lock.
  void* const ptr = malloc(size_bytes);

  // advise the kernel to allocate large 2M pages
  madvise(ptr, size_bytes, MADV_HUGEPAGE);

  // pre-fault/map the pages by setting the first byte of the page
  size_t pageSize = (1 << 21);
  uintptr_t alignedPtr = (((uintptr_t)ptr + pageSize - 1) & ~(pageSize - 1));
  for (uintptr_t p = alignedPtr; p < ((uintptr_t)ptr + size_bytes);
       p += pageSize) {
    memset((void*)p, 0, 1);
  }

  AT_CUDA_CHECK(cudaHostRegister(
      ptr, size_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
  void* dev_ptr;
  AT_CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, ptr, 0));

  auto storage = Storage(
      Storage::use_byte_size_t(),
      size_bytes,
      at::DataPtr(
          dev_ptr,
          new CUDAHostMappedContext(ptr, self.get_device()),
          &CUDAHostMappedContext::release,
          {at::DeviceType::CUDA, self.device().index()}),
      nullptr, /* allocator */
      /*resizable=*/false);
  return at::empty({0}, self.options())
      .set_(std::move(storage), 0, sizes, strides);
}

// Allocate the ATen Tensor with UVM or host-mapped memory
Tensor new_unified_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes,
    bool is_host_mapped) {
  if (is_host_mapped) {
    VLOG(2) << "Allocate the ATen Tensor with cudaHostAlloc";
    return new_host_mapped_tensor(self, sizes);
  } else {
    VLOG(2) << "Allocate the ATen Tensor with cudaMallocManaged";
    return new_managed_tensor(self, sizes);
  }
}

// Check if a tensor is allocated with UVM (CPU or GPU Tensor)
bool uvm_storage(const Tensor& t) {
  auto deleter = t.storage().data_ptr().get_deleter();
  return deleter == &CUDAManagedIndirectContext::release ||
      deleter == &CUDAHostMappedContext::release;
}

// Check if a tensor is allocated with UVM but is not a CPU Tensor
bool is_uvm_tensor(const Tensor& t) {
  if (t.device().is_cpu()) {
    return false;
  }
  return uvm_storage(t);
}

// Convert a UVM tensor to a CPU tensor
Tensor uvm_to_cpu(const Tensor& t) {
  TORCH_CHECK(is_uvm_tensor(t));
  // Don't copy the storage - just keep a reference to the original storage
  auto* tcontext =
      t.storage().data_ptr().cast_context<CUDAManagedIndirectContext>(
          &CUDAManagedIndirectContext::release);
  TORCH_CHECK(tcontext != nullptr)
  auto* ocontext =
      tcontext->storage_.data_ptr().cast_context<CUDAManagedContext>(
          &CUDAManagedContext::release);
  auto storage = Storage(
      Storage::use_byte_size_t(),
      t.storage().nbytes(),
      at::DataPtr(
          ocontext->ptr_,
          new CUDAManagedIndirectContext(tcontext->storage_),
          &CUDAManagedIndirectContext::release,
          {at::DeviceType::CPU}),
      nullptr, /* allocator */
      /*resizable=*/false);
  return at::empty({0}, t.options().device(Device::Type::CPU))
      .set_(std::move(storage), t.storage_offset(), t.sizes(), t.strides());
}

// Create a new UVM tensor sharing storage with t on the same device as
// prototype
Tensor uvm_to_device(const Tensor& t, const Tensor& prototype) {
  TORCH_CHECK(is_uvm_tensor(t));
  // Don't copy the storage - just keep a reference to the original storage
  auto* tcontext =
      t.storage().data_ptr().cast_context<CUDAManagedIndirectContext>(
          &CUDAManagedIndirectContext::release);
  TORCH_CHECK(tcontext != nullptr)

  auto device = prototype.device();
  auto* ocontext =
      tcontext->storage_.data_ptr().cast_context<CUDAManagedContext>(
          &CUDAManagedContext::release);
  auto storage = Storage(
      Storage::use_byte_size_t(),
      t.storage().nbytes(),
      at::DataPtr(
          ocontext->ptr_,
          new CUDAManagedIndirectContext(tcontext->storage_),
          &CUDAManagedIndirectContext::release,
          device),
      nullptr, /* allocator */
      /*resizable=*/false);
  return at::empty({0}, t.options().device(device))
      .set_(std::move(storage), t.storage_offset(), t.sizes(), t.strides());
}

namespace {
int64_t uvm_get_guard_index(const Tensor& t) {
  TORCH_CHECK(uvm_storage(t));
  int cuda_device_index;
  if (t.is_cpu()) {
    auto* tcontext =
        t.storage().data_ptr().cast_context<CUDAManagedIndirectContext>(
            &CUDAManagedIndirectContext::release);
    TORCH_CHECK(tcontext != nullptr)
    auto* ocontext =
        tcontext->storage_.data_ptr().cast_context<CUDAManagedContext>(
            &CUDAManagedContext::release);
    TORCH_CHECK(ocontext != nullptr)
    cuda_device_index = static_cast<int64_t>(ocontext->cuda_device_);
  } else {
    TORCH_CHECK(t.is_cuda());
    cuda_device_index = t.get_device();
  }
  return cuda_device_index;
}
} // namespace

void uvm_cuda_mem_advise(const Tensor& t, int64_t cuda_memory_advise) {
  // Call cudaMemAdvise on vm tensor
  // See cudaMemoryAdvise enum (automatically exported to python fbgemm_gpu.uvm
  // namespace) for valid values and interface stub.
  at::cuda::OptionalCUDAGuard device_guard;
  int64_t cuda_device_index = uvm_get_guard_index(t);
  int hint_device;
  if (t.is_cpu()) {
    hint_device = cudaCpuDeviceId;
  } else {
    TORCH_CHECK(t.is_cuda());
    hint_device = static_cast<int>(cuda_device_index);
  }

  void* ptr = t.data_ptr();
  size_t size_bytes = at::detail::computeStorageNbytes(
      t.sizes(), t.strides(), t.dtype().itemsize());

  device_guard.set_index(cuda_device_index);

  // FIXME: some advanced "cudaMemAdvise" flags are not supported by HIP.
  AT_CUDA_CHECK(cudaMemAdvise(
      ptr,
      size_bytes,
      static_cast<enum cudaMemoryAdvise>(cuda_memory_advise),
      hint_device));
  return;
}

void uvm_cuda_mem_prefetch_async(
    const Tensor& t,
    c10::optional<Tensor> device_t) {
  // Call cudaMemPrefetchAsync on Tensor
  at::cuda::OptionalCUDAGuard device_guard;
  TORCH_CHECK(uvm_storage(t));
  TORCH_CHECK(t.is_cuda() || (t.is_cpu() && device_t.has_value()));
  TORCH_CHECK(!device_t.has_value() || device_t.value().is_cuda());

  int prefetch_device =
      (t.is_cpu()) ? cudaCpuDeviceId : static_cast<int>(t.get_device());

  const Tensor& context_t = device_t.has_value() ? device_t.value() : t;

  void* ptr = t.data_ptr();
  size_t size_bytes = at::detail::computeStorageNbytes(
      t.sizes(), t.strides(), t.dtype().itemsize());

  device_guard.set_index(context_t.get_device());

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_CUDA_CHECK(cudaMemPrefetchAsync(ptr, size_bytes, prefetch_device, stream));

  return;
}

void uvm_mem_advice_dont_fork(const Tensor& t) {
  // During fork() the uvm driver is called to copy VMA for UVM space.
  // The uvm driver then removes pmap entries for both child and parent.
  // Re-establishing the mappings for the paretn is slow.
  // This works around the issue by setting the UVM VMA to not be copied
  // into the child.
  TORCH_CHECK(uvm_storage(t));

  void* ptr = t.data_ptr();
  size_t size_bytes = at::detail::computeStorageNbytes(
      t.sizes(), t.strides(), t.dtype().itemsize());

  auto adjusted = adjust_to_page_boundaries(ptr, size_bytes);

  int result =
      madvise(std::get<0>(adjusted), std::get<1>(adjusted), MADV_DONTFORK);

  TORCH_CHECK(result == 0);

  return;
}

Tensor uvm_to_cpu_clone(const Tensor& t) {
  TORCH_CHECK(uvm_storage(t));
  TORCH_CHECK(t.is_contiguous());

  Tensor cpu_clone = at::empty_like(t, t.options().device(kCPU));

  size_t size_bytes = at::detail::computeStorageNbytes(
      t.sizes(), t.strides(), t.dtype().itemsize());

  memcpy(cpu_clone.data_ptr(), t.data_ptr(), size_bytes);

  return cpu_clone;
}

FBGEMM_GPU_ENUM_GLOGAL(uvm)

FBGEMM_GPU_ENUM_REGISTER_START(uvm, cudaMemory, Advise){
    FBGEMM_GPU_ENUM_ITEM(
        cudaMem,
        AdviseSetReadMostly,
        cudaMemAdviseSetReadMostly),
    FBGEMM_GPU_ENUM_ITEM(
        cudaMem,
        AdviseUnsetReadMostly,
        cudaMemAdviseUnsetReadMostly),
    FBGEMM_GPU_ENUM_ITEM(
        cudaMem,
        AdviseSetPreferredLocation,
        cudaMemAdviseSetPreferredLocation),
    FBGEMM_GPU_ENUM_ITEM(
        cudaMem,
        AdviseUnsetPreferredLocation,
        cudaMemAdviseUnsetPreferredLocation),
    FBGEMM_GPU_ENUM_ITEM(
        cudaMem,
        AdviseSetAccessedBy,
        cudaMemAdviseSetAccessedBy),
    FBGEMM_GPU_ENUM_ITEM(
        cudaMem,
        AdviseUnsetAccessedBy,
        cudaMemAdviseUnsetAccessedBy),
} FBGEMM_GPU_ENUM_REGISTER_END

} // namespace fbgemm_gpu

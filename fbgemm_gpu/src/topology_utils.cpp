/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/utils/topology_utils.h"
#include <ATen/cuda/CUDAContext.h> // IWYU pragma: keep
#include <c10/core/Device.h> // IWYU pragma: keep
#include <c10/cuda/CUDAException.h>
#include <c10/util/Logging.h> // IWYU pragma: keep
#include <algorithm>

#ifdef USE_ROCM
#include <inttypes.h>
#include "amd_smi/amdsmi.h"
#include "hip/hip_runtime.h"

#define AMDSMI_CHECK(fn)                          \
  do {                                            \
    amdsmi_status_t ret = (fn);                   \
    TORCH_CHECK_EQ((ret), AMDSMI_STATUS_SUCCESS); \
  } while (0)

#define AMDSMI_DEVICE_PCI_BUS_ID_BUFFER_SIZE 16

namespace fbgemm_gpu {
AdjacencyMatrix<Links> get_nvlink_matrix() {
  auto world_size = at::cuda::getNumGPUs();
  AMDSMI_CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS));

  // Note that AMD SMI uses a different numbering method to ROCm runtime,
  // so we need to learn the mapping by using the bus ID.

  // Get all sockets, then collect all GPU processor handles across sockets.
  uint32_t socket_count = 0;
  AMDSMI_CHECK(amdsmi_get_socket_handles(&socket_count, nullptr));
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  AMDSMI_CHECK(amdsmi_get_socket_handles(&socket_count, sockets.data()));

  std::vector<amdsmi_processor_handle> processor_handles;
  for (uint32_t s = 0; s < socket_count; s++) {
    uint32_t device_count = 0;
    AMDSMI_CHECK(
        amdsmi_get_processor_handles(sockets[s], &device_count, nullptr));
    std::vector<amdsmi_processor_handle> socket_handles(device_count);
    AMDSMI_CHECK(amdsmi_get_processor_handles(
        sockets[s], &device_count, socket_handles.data()));
    processor_handles.insert(
        processor_handles.end(), socket_handles.begin(), socket_handles.end());
  }

  std::unordered_map<Node, amdsmi_processor_handle> hip_device_to_handle;

  for (const auto& handle : processor_handles) {
    uint64_t pci_info;
    AMDSMI_CHECK(amdsmi_get_gpu_bdf_id(handle, &pci_info));
    uint64_t domain, bus, device, function;
    domain = (pci_info >> 32) & 0xffffffff;
    bus = (pci_info >> 8) & 0xff;
    device = (pci_info >> 3) & 0x1f;
    function = pci_info & 0x7;
    // Different from CUDA, we do not get the PCI BUS ID as a char* and we need
    // to reconstruct it.
    char pci_bus_id_str[AMDSMI_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    sprintf(
        pci_bus_id_str,
        "%04" PRIu64 ":%02" PRIu64 ":%02" PRIu64 ".%0" PRIu64,
        domain,
        bus,
        device,
        function);

    std::array<char, AMDSMI_DEVICE_PCI_BUS_ID_BUFFER_SIZE> pci_bus_id;
    std::copy(
        &pci_bus_id_str[0],
        &pci_bus_id_str[AMDSMI_DEVICE_PCI_BUS_ID_BUFFER_SIZE],
        pci_bus_id.data());
    int32_t node = 0;
    auto err = hipDeviceGetByPCIBusId(&node, pci_bus_id.data());
    if (err == hipSuccess) {
      hip_device_to_handle.insert({node, handle});
    } else {
      // flush the last error - this can occur when e.g. we set
      // HIP_VISIBLE_DEVICES to a subset of the available GPUs in the system.
      std::ignore = hipGetLastError();
    }
  }

  std::vector<Links> links(world_size * world_size);
  for (const auto i : c10::irange(world_size)) {
    auto src = hip_device_to_handle.find(i);
    if (src != hip_device_to_handle.end()) {
      for (const auto j : c10::irange(world_size)) {
        auto dst = hip_device_to_handle.find(j);
        if (dst != hip_device_to_handle.end()) {
          bool is_active;
          AMDSMI_CHECK(
              amdsmi_is_P2P_accessible(src->second, dst->second, &is_active));
          if (is_active) {
            links[i * world_size + j] += 1;
          }
        }
      }
    }
  }
  AMDSMI_CHECK(amdsmi_shut_down());
  return [=](Node i, Node j) {
    TORCH_CHECK_LT(i, world_size);
    TORCH_CHECK_LT(j, world_size);
    return links[i * world_size + j];
  };
}
} // namespace fbgemm_gpu

#else // CUDA

#include <nvml.h>

#define NVML_CHECK(fn)                   \
  do {                                   \
    nvmlReturn_t ret = (fn);             \
    TORCH_CHECK_EQ((ret), NVML_SUCCESS); \
  } while (0)

namespace fbgemm_gpu {
AdjacencyMatrix<Links> get_nvlink_matrix() {
  auto world_size = at::cuda::getNumGPUs();
  NVML_CHECK(nvmlInit());

  // Note that NVML uses a different numbering method to CUDA runtime,
  // so we need to learn the mapping by using the bus ID.
  uint32_t device_count = 0;
  NVML_CHECK(nvmlDeviceGetCount(&device_count));

  std::map<std::array<char, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE>, Node>
      pci_bus_ids;
  std::unordered_map<Node, uint32_t> cuda_device_to_nvml_device;

  for (const auto i : c10::irange(device_count)) {
    nvmlDevice_t handle = nullptr;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(i, &handle));
    nvmlPciInfo_t pci_info;
    NVML_CHECK(nvmlDeviceGetPciInfo(handle, &pci_info));
    std::array<char, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE> pci_bus_id{};
    std::copy(
        &pci_info.busId[0],
        &pci_info.busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE],
        pci_bus_id.data());
    int32_t node = 0;
    auto err = C10_CUDA_ERROR_HANDLED(
        cudaDeviceGetByPCIBusId(&node, pci_bus_id.data()));
    if (err == cudaSuccess) {
      pci_bus_ids.insert({pci_bus_id, node});
      cuda_device_to_nvml_device.insert({node, i});
    } else {
      // flush the last error - this can occur when e.g. we set
      // CUDA_VISIBLE_DEVICES to a subset of the available GPUs in the system.
      C10_CUDA_CLEAR_ERROR();
    }
  }

  std::vector<Links> links(world_size * world_size);
  for (const auto i : c10::irange(world_size)) {
    nvmlDevice_t handle = nullptr;
    NVML_CHECK(
        nvmlDeviceGetHandleByIndex(cuda_device_to_nvml_device[i], &handle));
    for (const auto link : c10::irange(NVML_NVLINK_MAX_LINKS)) {
      nvmlEnableState_t is_active{NVML_FEATURE_DISABLED};
      auto nvmlRet = nvmlDeviceGetNvLinkState(handle, link, &is_active);
      if (nvmlRet == NVML_ERROR_INVALID_ARGUMENT ||
          nvmlRet == NVML_ERROR_NOT_SUPPORTED) {
        continue;
      }
      if (is_active != NVML_FEATURE_ENABLED) {
        continue;
      }
      nvmlPciInfo_t pci_info;
      NVML_CHECK(nvmlDeviceGetNvLinkRemotePciInfo(handle, link, &pci_info));
      std::array<char, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE> pci_bus_id{};
      std::copy(
          &pci_info.busId[0],
          &pci_info.busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE],
          pci_bus_id.data());
      auto dst = pci_bus_ids.find(pci_bus_id);
      if (dst != pci_bus_ids.end()) {
        auto j = dst->second;
        links[i * world_size + j] += 1;
      }
    }
  }

  return [=](Node i, Node j) {
    TORCH_CHECK_LT(i, world_size);
    TORCH_CHECK_LT(j, world_size);
    return links[i * world_size + j];
  };
}
} // namespace fbgemm_gpu

#endif // USE_ROCM

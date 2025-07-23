/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cfloat>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

#include <ATen/ATen.h>

// In OSS hipification of the include is not working, so we hipify it manually.
#ifdef USE_ROCM
#include <ATen/hip/HIPEvent.h> // @manual
#include <ATen/hip/HIPGraph.h> // @manual
#include <hip/hip_runtime.h>
#define GPUStream at::hip::HIPStreamMasqueradingAsCUDA
#define GPUStreamGuard at::hip::HIPStreamGuardMasqueradingAsCUDA
#define getStreamFromPool at::hip::getStreamFromPoolMasqueradingAsCUDA
#define gpuStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define gpuEventDefault hipEventDefault
#else
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <cuda_runtime.h>
#define GPUStream at::cuda::CUDAStream
#define GPUStreamGuard at::cuda::CUDAStreamGuard
#define getStreamFromPool at::cuda::getStreamFromPool
#define gpuStreamCaptureModeRelaxed cudaStreamCaptureModeRelaxed
#define gpuEventDefault cudaEventDefault
#endif

#include <ostream>

/**
 * Tuning cache for kernels. This class is responsible for evaluating new
 * problem shapes (keyed by a string) against a predefined set of kernels, and
 * caching the best kernel found.
 */
class TuningCache final {
 public:
  // kernelName should be unique for each type of kernel, as it is used to
  // construct the filename.
  explicit TuningCache(const std::string& kernelName)
      : useCudaGraph_(std::getenv("FBGEMM_AUTOTUNE_USE_CUDA_GRAPH") != nullptr),
        cacheDirectory_(getCacheDirectory()),
        cacheFilename_(getCacheFilename(kernelName)),
        detailedFilename_(getDetailedFilename(kernelName)) {
    std::cout << "Using cache file at " << cacheFilename_ << std::endl;

    createCacheDirectory();
    loadCache();
  }

  TuningCache(const TuningCache&) = delete;
  TuningCache& operator=(const TuningCache&) = delete;
  TuningCache(TuningCache&&) = delete;
  TuningCache& operator=(TuningCache&&) = delete;

  ~TuningCache() {
    saveCache();
  }

  template <typename Kernel, typename... Args>
  Kernel findBestKernelMaybeAutotune(
      const std::string& cache_key,
      const std::unordered_map<std::string, Kernel>& kernels,
      Args&&... args) {
    TORCH_CHECK(!kernels.empty(), "Kernels to tune over is empty.");

    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
      return getKernel(it->second, kernels);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    auto kernel_key =
        findBestKernel(cache_key, kernels, std::forward<Args>(args)...);
    if (kernel_key.empty()) {
      throw std::runtime_error("Failed to tune a kernel for key: " + cache_key);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tuned " << kernel_key << " for key " << cache_key << " in "
              << elapsed.count() << " ms." << std::endl;

    cache_.insert({cache_key, kernel_key});
    return getKernel(kernel_key, kernels);
  }

 private:
  template <typename Kernel>
  Kernel getKernel(
      const std::string& kernel_key,
      const std::unordered_map<std::string, Kernel>& kernels) {
    auto it = kernels.find(kernel_key);
    TORCH_CHECK(
        it != kernels.end(),
        "Failed to find kernel keyed by " + kernel_key +
            ". Consider deleting your fbgemm cache (~/.fbgemm).");
    return it->second;
  }

  std::string getCacheDirectory() {
    // If the environment variable is set, use that instead of the default
    const char* cache_dir = std::getenv("FBGEMM_CACHE_DIR");
    if (cache_dir) {
      return cache_dir;
    }

    return std::string(std::getenv("HOME")) + "/" +
        std::string(FBGEMM_CACHE_DIR);
  }

  std::string getCacheFilename(const std::string& kernel_name) {
    return getCacheDirectory() + "/" + kernel_name + ".txt";
  }

  std::string getDetailedFilename(const std::string& kernel_name) {
    return getCacheDirectory() + "/" + kernel_name + "_detailed.txt";
  }

  bool cacheDirExists() {
    return std::filesystem::exists(cacheDirectory_) &&
        std::filesystem::is_directory(cacheDirectory_);
  }

  void createCacheDirectory() {
    if (!cacheDirExists()) {
      // Try to create the directory, multiple caches/processes may attempt
      // this, and only one would succeed.
      std::string error;
      try {
        if (std::filesystem::create_directory(cacheDirectory_)) {
          return;
        }
      } catch (const std::filesystem::filesystem_error& e) {
        error = e.what();
      }

      // If the directory still doesn't exist, error out
      TORCH_CHECK(
          cacheDirExists(),
          "FBGEMM cache directory creation at " + cacheDirectory_ +
              " failed: " + error);
    }
  }

  void loadCache() {
    std::ifstream file(cacheFilename_);
    if (!file.is_open()) {
      // Create a new cache file if it doesn't exist
      std::ofstream newFile(cacheFilename_);
      newFile.close();
    } else {
      std::string line;
      while (std::getline(file, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
          std::string key = line.substr(0, pos);
          std::string value = line.substr(pos + 1);
          cache_.insert_or_assign(key, value);
        }
      }
      file.close();
    }
  }

  void saveCache() {
    // Only one rank needs to save the cache. This is fine as the cache
    // should be largely equivalent across ranks.
    if (at::cuda::current_device() != 0) {
      return;
    }

    std::ofstream file(cacheFilename_);
    if (file.is_open()) {
      for (const auto& pair : cache_) {
        file << pair.first << "=" << pair.second << std::endl;
      }
      file.close();
    }

    if (!detailedTuningInfo_.empty()) {
      std::ofstream detailed_file(detailedFilename_, std::ios_base::app);
      if (detailed_file.is_open()) {
        for (auto& [cache_key, kernels] : detailedTuningInfo_) {
          // Sort for convenience in descending order of time_ms
          std::sort(
              kernels.begin(), kernels.end(), [](const auto& a, const auto& b) {
                return a.second < b.second;
              });
          for (const auto& [kernel_name, time_ms] : kernels) {
            detailed_file << cache_key << "," << kernel_name << "," << time_ms
                          << std::endl;
          }
        }

        detailed_file.close();
      }
    }
  }

  template <typename Kernel, typename... Args>
  float
  benchmark(const std::string& kernel_name, Kernel kernel, Args&&... args) {
    // Warmup iteration
    try {
      kernel(std::forward<Args>(args)...);
    } catch (const std::exception& e) {
      std::cout << "Warmup iteration failed for " << kernel_name
                << " it will be skipped." << '\n';
      return FLT_MAX;
    }

    // Estimate the number of iterations needed to run for 10 ms. This
    // helps with stability for fast kernels.
    start_.record();
    kernel(std::forward<Args>(args)...);
    stop_.record();
    stop_.synchronize();
    const auto estimated_time_ms = start_.elapsed_time(stop_);
    const int num_iters = std::max(1, int(10 / estimated_time_ms));

    if (useCudaGraph_) {
      at::cuda::CUDAGraph graph;
      {
        // CUDAGraph capture must happen on non-default stream
        GPUStream stream = getStreamFromPool(true);
        GPUStreamGuard streamGuard(stream);

        // For flexibility, we use cudaStreamCaptureModeRelaxed.
        // - cudaStreamCaptureModeGlobal prevents other threads from calling
        // certain CUDA APIs such as cudaEventQuery. This can conflict with
        // things like ProcessGroupNCCL.
        // - cudaStreamCaptureModeThreadLocal prevents CCA from freeing memory.
        // Since CUDA graph is preferred for offline benchmark this should be
        // fine.
        graph.capture_begin({0, 0}, gpuStreamCaptureModeRelaxed);
        for (int i = 0; i < num_iters; ++i) {
          kernel(std::forward<Args>(args)...);
        }
        graph.capture_end();
      }

      // Time execution of graph
      start_.record();
      graph.replay();
      stop_.record();
      stop_.synchronize();
      const auto graph_time_ms = start_.elapsed_time(stop_);

      return graph_time_ms / num_iters;
    } else {
      // Time execution of kernels
      start_.record();
      for (int i = 0; i < num_iters; ++i) {
        kernel(std::forward<Args>(args)...);
      }
      stop_.record();
      stop_.synchronize();
      const auto kernels_time_ms = start_.elapsed_time(stop_);

      return kernels_time_ms / num_iters;
    }
  }

  template <typename Kernel, typename... Args>
  std::string findBestKernel(
      const std::string& cache_key,
      const std::unordered_map<std::string, Kernel>& kernels,
      Args&&... args) {
    std::string best_kernel;
    float best_time = FLT_MAX;

    for (const auto& [kernel_name, kernel] : kernels) {
      const float time =
          benchmark(kernel_name, kernel, std::forward<Args>(args)...);
      if (time < best_time) {
        best_time = time;
        best_kernel = kernel_name;
      }
      if (std::getenv("FBGEMM_AUTOTUNE_COLLECT_STATS")) {
        detailedTuningInfo_[cache_key].push_back({kernel_name, time});
      }
    }

    return best_kernel;
  }

  constexpr static std::string_view FBGEMM_CACHE_DIR = ".fbgemm";

  at::cuda::CUDAEvent start_ = at::cuda::CUDAEvent(gpuEventDefault);
  at::cuda::CUDAEvent stop_ = at::cuda::CUDAEvent(gpuEventDefault);

  // If FBGEMM_AUTOTUNE_USE_CUDA_GRAPH is set, use CUDA graph for benchmarking.
  // CUDA graphs use a separate memory pool to do allocation in PyTorch
  // CUDACachingAllocator to ensure the memory is valid throughout the graph,
  // which can memory fragmentation (and higher chance of CUDA OOM). We can
  // prefer to use CUDA graph for offline benchmarking, but not for online
  // serving.
  bool useCudaGraph_;
  // Absolute path of the cache directory
  std::string cacheDirectory_;
  // Absolute path of the cache file for the kernel
  std::string cacheFilename_;
  // Absolute path of the detailed tuning info
  std::string detailedFilename_;
  // (cache key, best kernel)
  std::unordered_map<std::string, std::string> cache_;
  // If FBGEMM_AUTOTUNE_COLLECT_STATS is set, we will log the timing for each
  // kernel for each problem shape. This is useful to distill the best kernels
  // into a smaller set.
  std::unordered_map<std::string, std::vector<std::pair<std::string, float>>>
      detailedTuningInfo_;
};

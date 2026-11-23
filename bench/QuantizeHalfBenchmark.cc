/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cpuinfo.h>

#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/FloatConversion.h"
#include "fbgemm/QuantUtils.h"
#include "fbgemm/Types.h"
#include "fbgemm/Utils.h"

using namespace fbgemm;

namespace {

enum class Method { FloatInput, Scratch64KiB, DirectHalf };

const char* methodName(Method method) {
  switch (method) {
    case Method::FloatInput:
      return "fp32_input";
    case Method::Scratch64KiB:
      return "scratch_64k";
    case Method::DirectHalf:
      return "direct_half";
  }
  throw std::logic_error("Unknown benchmark method");
}

template <FbgemmHalfType InputType>
void convert(const InputType* source, float* destination, size_t count) {
  if constexpr (std::is_same_v<InputType, float16>) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
    if (cpuinfo_initialize() && fbgemmHasAvx2Support() &&
        cpuinfo_has_x86_f16c()) {
      Float16ToFloat_simd(source, destination, count);
      return;
    }
#endif
    for (size_t i = 0; i < count; ++i) {
      destination[i] = cpu_half2float_ref(source[i]);
    }
  } else {
    Bfloat16ToFloat_simd(source, destination, count);
  }
}

template <typename T, FbgemmHalfType InputType>
void runKernel(
    Method method,
    const std::vector<InputType>& source,
    const std::vector<float>& widened,
    std::vector<T>& output,
    const TensorQuantizationParams& qparams,
    int thread_id,
    int threads) {
  if (method == Method::FloatInput) {
    Quantize<T>(
        widened.data(),
        output.data(),
        source.size(),
        qparams,
        thread_id,
        threads);
  } else if (method == Method::DirectHalf) {
    Quantize<T>(
        source.data(),
        output.data(),
        source.size(),
        qparams,
        thread_id,
        threads);
  } else {
    int64_t begin = 0, end = 0;
    fbgemmPartition1D(thread_id, threads, source.size(), begin, end);
    std::vector<float> scratch(16384);
    for (int64_t start = begin; start < end; start += scratch.size()) {
      const auto count = std::min<int64_t>(scratch.size(), end - start);
      convert(source.data() + start, scratch.data(), count);
      Quantize<T>(scratch.data(), output.data() + start, count, qparams);
    }
  }
}

template <typename T, FbgemmHalfType InputType>
void run(
    Method method,
    const std::vector<InputType>& source,
    const std::vector<float>& widened,
    std::vector<T>& output,
    const TensorQuantizationParams& qparams,
    int threads) {
  if (threads == 1) {
    runKernel(method, source, widened, output, qparams, 0, 1);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (int id = 0; id < threads; ++id) {
    workers.emplace_back([&, id] {
      runKernel(method, source, widened, output, qparams, id, threads);
    });
  }
  for (auto& worker : workers)
    worker.join();
}

template <typename T, FbgemmHalfType InputType>
void benchmark(
    const char* dtype,
    const char* output_type,
    int64_t elements,
    int repetitions) {
  std::mt19937 generator(172);
  std::uniform_real_distribution<float> distribution(-4.f, 4.f);
  std::vector<InputType> source(elements);
  for (auto& value : source)
    value = convert_ref<InputType>(distribution(generator));
  const auto original = source;
  std::vector<float> widened(elements);
  convert(source.data(), widened.data(), elements);
  const TensorQuantizationParams qparams{
      0.03125f,
      std::is_signed_v<T> ? 0 : (1 << (sizeof(T) * 8 - 1)),
      static_cast<int>(sizeof(T) * 8)};
  std::vector<T> expected(elements), output(elements);
  Quantize<T>(widened.data(), expected.data(), elements, qparams);
  for (const int threads : {1, 8}) {
    std::vector<Method> order{
        Method::FloatInput, Method::Scratch64KiB, Method::DirectHalf};
    for (const auto method : order) {
      run(method, source, widened, output, qparams, threads);
      if (output != expected)
        throw std::runtime_error("Warmup parity failed");
    }
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      for (const auto method : order) {
        const auto start = std::chrono::steady_clock::now();
        run(method, source, widened, output, qparams, threads);
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if (output != expected)
          throw std::runtime_error("Timed result parity failed");
        std::cout << dtype << ',' << output_type << ',' << threads << ','
                  << elements << ',' << methodName(method) << ',' << repetition
                  << ',' << seconds << '\n';
      }
      std::ranges::reverse(order);
    }
  }
  if (source != original)
    throw std::runtime_error("Input mutated");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const int64_t elements = argc > 1 ? std::stoll(argv[1]) : 1048576LL * 128;
    const int repetitions = argc > 2 ? std::stoi(argv[2]) : 5;
    if (elements <= 0 || repetitions < 5 || argc > 3) {
      throw std::invalid_argument(
          "Usage: QuantizeHalfBenchmark [positive-elements] [repetitions>=5]");
    }
    std::cout
        << "dtype,output_type,threads,elements,method,repetition,seconds\n";
    benchmark<int8_t, float16>("fp16", "int8", elements, repetitions);
    benchmark<int8_t, bfloat16>("bf16", "int8", elements, repetitions);
    benchmark<uint8_t, float16>("fp16", "uint8", elements, repetitions);
    benchmark<uint8_t, bfloat16>("bf16", "uint8", elements, repetitions);
    benchmark<int16_t, float16>("fp16", "int16", elements, repetitions);
    benchmark<int16_t, bfloat16>("bf16", "int16", elements, repetitions);
    benchmark<uint16_t, float16>("fp16", "uint16", elements, repetitions);
    benchmark<uint16_t, bfloat16>("bf16", "uint16", elements, repetitions);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

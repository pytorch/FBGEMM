/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cuda.h>
#include <fenv.h>
#include <getopt.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "fbgemm_gpu/batched_unary_embedding_wrappers.cuh"
#include "fbgemm_gpu/bench_utils.cuh"
#include "fbgemm_gpu/cuda_utils.cuh"

void generate_auxiliary_tensors(
    int batch_size,
    std::vector<int>& hash_sizes,
    std::vector<long>& table_offsets,
    std::vector<long>& lengths,
    std::vector<long>& offsets,
    std::vector<long>& indices) {
  // generate lengths and indices
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  fesetround(FE_TONEAREST);
  for (int h = 0; h < hash_sizes.size(); h++) {
    for (int i = 0; i < batch_size; i++) {
      long n_indices = 1;
      indices.push_back(
          std::lrintf(distribution(generator) * (hash_sizes[h] - 1)));
      lengths.push_back(n_indices);
    }
  }

  // generate offsets
  offsets.push_back(0);
  long inc_sum = 0;
  for (auto const& item : lengths) {
    offsets.push_back(inc_sum += item);
  }

  // generate table_offsets
  long inc_table_hash_sum = 0;
  table_offsets.push_back(0);
  for (auto const& item : hash_sizes) {
    table_offsets.push_back(inc_table_hash_sum += item);
  }
}

void parse_commandline(
    int argc,
    char* argv[],
    int* batch_size,
    int* num_tables,
    int* num_tasks,
    int* iters) {
  static struct option longopts[] = {
      {"batch-size", required_argument, NULL, 'b'},
      {"num_tables", required_argument, NULL, 't'},
      {"num_tasks", required_argument, NULL, 'p'},
      {"iters", required_argument, NULL, 'i'}};

  int opt;
  while ((opt = getopt_long(argc, argv, "b:t:p:i", longopts, NULL)) != -1) {
    switch (opt) {
      case 'b':
        *batch_size = atoi(optarg);
        break;
      case 't':
        *num_tables = atoi(optarg);
        break;
      case 'p':
        *num_tasks = atoi(optarg);
        break;
      case 'i':
        *iters = atoi(optarg);
        break;
    }
  }
  std::cout << "batch size: " << *batch_size << std::endl;
  std::cout << "number of tables: " << *num_tables << std::endl;
  std::cout << "number of tasks: " << *num_tasks << std::endl;
  std::cout << "iteration: " << *iters << std::endl;
}

int main(int argc, char* argv[]) {
  int batch_size = 512;
  int num_tables = 2;
  int num_tasks = 3;
  int iters = 100;
  parse_commandline(argc, argv, &batch_size, &num_tables, &num_tasks, &iters);

  // generate hash_sizes
  std::vector<int> hash_sizes;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(50, 250);
  for (int i = 0; i < num_tables; i++) {
    hash_sizes.push_back(distribution(generator));
  }
  std::cout << "table rows: ";
  for (auto const& hash_size : hash_sizes) {
    std::cout << hash_size << ",";
  }
  std::cout << std::endl;

  // the auxilary tensors
  std::vector<long> table_offsets;
  std::vector<long> lengths;
  std::vector<long> offsets;
  std::vector<long> indices;

  generate_auxiliary_tensors(
      batch_size, hash_sizes, table_offsets, lengths, offsets, indices);

  // cache flush utility
  // gpu ptrs
  float* embedding_table_ptr;
  long* table_offsets_ptr;
  long* offsets_ptr;
  long* indices_ptr;
  float* output_ptr;
  float* grad_ptr;
  float* grad_weight_ptr;

  int embedding_rows = 0;
  for (auto const& h : hash_sizes) {
    embedding_rows += h;
  }
  CUDA_CHECK(cudaMalloc(
      &embedding_table_ptr, embedding_rows * num_tasks * sizeof(float)));
  // generate embedding table random numbers
  generate_random_table(embedding_table_ptr, embedding_rows * num_tasks);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(
      cudaMalloc(&table_offsets_ptr, table_offsets.size() * sizeof(long)));
  CUDA_CHECK(cudaMalloc(&offsets_ptr, offsets.size() * sizeof(long)));
  CUDA_CHECK(cudaMalloc(&indices_ptr, indices.size() * sizeof(long)));
  CUDA_CHECK(cudaMalloc(
      &output_ptr, batch_size * num_tables * num_tasks * sizeof(float)));
  CUDA_CHECK(cudaGetLastError());

  // memcpy
  CUDA_CHECK(cudaMemcpy(
      table_offsets_ptr,
      table_offsets.data(),
      table_offsets.size() * sizeof(long),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      offsets_ptr,
      offsets.data(),
      offsets.size() * sizeof(long),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      indices_ptr,
      indices.data(),
      indices.size() * sizeof(long),
      cudaMemcpyHostToDevice));

  // forward
  float forward_time = benchmark_function(iters, [&]() {
    fbgemm_gpu_test::batched_unary_embeddings_forward(
        num_tasks,
        batch_size,
        num_tables,
        embedding_table_ptr,
        table_offsets_ptr,
        offsets_ptr,
        indices_ptr,
        output_ptr);
  });

  // free forward-only gpu ptrs
  cudaFree(output_ptr);

  // backward
  cudaMalloc(&grad_ptr, batch_size * num_tables * num_tasks * sizeof(float));
  generate_random_table(grad_ptr, batch_size * num_tables * num_tasks);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaMalloc(&grad_weight_ptr, embedding_rows * num_tasks * sizeof(float));
  float backward_time = benchmark_function(iters, [&]() {
    fbgemm_gpu_test::batched_unary_embeddings_backward(
        num_tasks,
        batch_size,
        num_tables,
        grad_ptr,
        table_offsets_ptr,
        offsets_ptr,
        indices_ptr,
        grad_weight_ptr);
  });

  // free backward-only gpu ptrs
  cudaFree(grad_ptr);
  cudaFree(grad_weight_ptr);

  // free other gpu ptrs;
  cudaFree(embedding_table_ptr);
  cudaFree(table_offsets_ptr);
  cudaFree(offsets_ptr);
  cudaFree(indices_ptr);
  cudaFree(table_offsets_ptr);
  cudaFree(table_offsets_ptr);

  std::cout << "Average Forward Pass Execution time per iteration: "
            << forward_time << " ms" << std::endl;
  std::cout << "Forward Pass Memory Bandwidth: "
            << (num_tasks * num_tables * batch_size *
                (5 * sizeof(long) + 2 * sizeof(float))) /
          (forward_time * 1e-3) / 1e9
            << " GB/s" << std::endl;
  std::cout << "Average Backward Pass Execution time per iteration: "
            << backward_time << " ms" << std::endl;
  std::cout << "Backward Pass Memory Bandwidth: "
            << (num_tasks * num_tables * batch_size *
                (5 * sizeof(long) + 2 * sizeof(float))) /
          (backward_time * 1e-3) / 1e9
            << " GB/s" << std::endl;
}

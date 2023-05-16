/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

template <
    typename scalar_t,
    typename index_t,
    typename acc_t,
    int NUM_THREADS_PER_BLOCK,
    int MAX_ENTRIES_PER_BLOCK>
__global__ void index_select_scalar_cumsum_kernel(
    scalar_t* output,
    acc_t* output_cumsum,
    const scalar_t* __restrict__ input,
    const index_t* __restrict__ indices,
    const int num_batches,
    const int input_batch_size,
    const int output_batch_size,
    const int last_block_num_entries,
    int* block_flags,
    acc_t* block_sums) {
  typedef cub::BlockScan<acc_t, NUM_THREADS_PER_BLOCK> BlockScan;
  __shared__ typename BlockScan::TempStorage bs_temp_storage;
  __shared__ acc_t smem[MAX_ENTRIES_PER_BLOCK];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = tid / output_batch_size;
  const int num_entries_per_block = blockIdx.x == gridDim.x - 1
      ? last_block_num_entries
      : MAX_ENTRIES_PER_BLOCK;

  // Load data
  acc_t local_data[1];
  if (tid < num_batches * output_batch_size) {
    *local_data =
        input[bid * input_batch_size + indices[tid % output_batch_size]];
    output[tid] = *local_data;
  } else {
    *local_data = 0;
  }

  // Cumsum
  inclusive_sum_scan_kernel<acc_t, 1, NUM_THREADS_PER_BLOCK>(
      local_data,
      bs_temp_storage,
      block_flags,
      block_sums,
      &smem[0],
      num_entries_per_block,
      blockIdx.x,
      gridDim.x > 1,
      1);

  // Store data
  if (tid < num_batches * output_batch_size) {
    output_cumsum[tid] = *local_data;
  }
}

template <
    typename scalar_t,
    typename index_t,
    typename offset_t,
    typename weight_t,
    bool has_weights>
__global__ void keyed_jagged_index_select_dim1_kernel(
    scalar_t* output,
    weight_t* output_weights,
    const scalar_t* input,
    const weight_t* weights,
    const offset_t* input_offsets,
    const index_t* indices,
    const offset_t* output_offsets,
    const int num_batches,
    const int input_batch_size,
    const int output_batch_size,
    const int64_t num_outputs) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_outputs) {
    // Each thread searches index position
    int index_pos;
    binary_search_range(
        &index_pos,
        output_offsets,
        (offset_t)tid,
        num_batches * output_batch_size);

    const offset_t rel_index =
        tid - (index_pos == 0 ? 0 : output_offsets[index_pos - 1]);

    // indices are the same for all batches
    const index_t index = indices[index_pos % output_batch_size];
    const int bid = index_pos / output_batch_size;
    const offset_t input_offset =
        (index == 0 && bid == 0
             ? 0
             : input_offsets[bid * input_batch_size + index - 1]) +
        rel_index;

    // Store data
    output[tid] = input[input_offset];
    if (has_weights) {
      output_weights[tid] = weights[input_offset];
    }
  }
}

template <typename scalar_t, typename index_t, typename offset_t>
__global__ void keyed_jagged_index_add_dim1_kernel(
    scalar_t* output,
    const scalar_t* input,
    const offset_t* input_offsets,
    const index_t* indices,
    const offset_t* output_offsets,
    const int num_batches,
    const int input_batch_size,
    const int output_batch_size,
    const int64_t num_inputs) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_inputs) {
    // Each thread searches index position
    int index_pos;
    binary_search_range(
        &index_pos,
        input_offsets,
        (offset_t)tid,
        num_batches * input_batch_size);

    const offset_t rel_index =
        tid - (index_pos == 0 ? 0 : input_offsets[index_pos - 1]);

    // indices are the same for all batches
    const index_t index = indices[index_pos % input_batch_size];
    const int bid = index_pos / input_batch_size;
    const offset_t output_offset =
        (index == 0 && bid == 0
             ? 0
             : output_offsets[bid * output_batch_size + index - 1]) +
        rel_index;

    // Store data
    gpuAtomicAdd(&output[output_offset], input[tid]);
  }
}

namespace {

class KeyedJaggedIndexSelectDim1GPUOp
    : public torch::autograd::Function<KeyedJaggedIndexSelectDim1GPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& values,
      const Tensor& lengths,
      const Tensor& offsets,
      const Tensor& indices, // select same indices for all batches
      const int batch_size,
      const c10::optional<Tensor>& weights) {
    // TODO: Add weights support
    TENSOR_ON_CUDA_GPU(lengths);
    TENSOR_ON_CUDA_GPU(offsets);
    TENSOR_ON_CUDA_GPU(values);
    TENSOR_ON_CUDA_GPU(indices);
    TENSORS_ON_SAME_DEVICE(lengths, indices);
    TENSORS_ON_SAME_DEVICE(offsets, indices);
    TENSORS_ON_SAME_DEVICE(values, indices);
    TORCH_CHECK(values.dim() == 1, "values must be a 1D tensor");
    TORCH_CHECK(lengths.dim() == 1, "lengths must be a 1D tensor");
    TORCH_CHECK(offsets.dim() == 1, "offsets must be a 1D tensor");
    TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor");
    TORCH_CHECK(
        lengths.numel() + 1 == offsets.numel(),
        "offsets size must be lengths size + 1");
    TORCH_CHECK(lengths.numel() % batch_size == 0, "lengths");

    if (weights.has_value()) {
      const Tensor& pos_weights = weights.value();
      TENSOR_ON_CUDA_GPU(pos_weights);
      TENSORS_ON_SAME_DEVICE(pos_weights, indices);
      TORCH_CHECK(pos_weights.dim() == 1, "weights must be a 1D tensor");
      TORCH_CHECK(
          pos_weights.numel() == values.numel(),
          "weights size and values size must be the same");
    }

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(values.get_device());

    const int num_batches = lengths.numel() / batch_size;
    const int num_output_lengths = num_batches * indices.numel();
    const int MAX_CUMSUM_ENTRIES_PER_BLOCK = 256;
    auto grid_size = cuda_calc_xblock_count(
        num_output_lengths, MAX_CUMSUM_ENTRIES_PER_BLOCK);

    Tensor output_offsets =
        at::empty({num_batches * indices.numel()}, offsets.options());
    Tensor output_lengths =
        at::empty({num_batches * indices.numel()}, lengths.options());

    Tensor block_flags, block_sums;
    if (grid_size > 1) {
      block_flags = at::zeros({grid_size}, lengths.options().dtype(at::kInt));
      block_sums = at::empty({grid_size}, output_offsets.options());
    }
    // Do index select and cumsum
    AT_DISPATCH_INDEX_TYPES(
        lengths.scalar_type(), "index_select_scalar_cumsum_wrapper_1", [&] {
          using length_t = index_t;
          AT_DISPATCH_INDEX_TYPES(
              offsets.scalar_type(),
              "index_select_scalar_cumsum_wrapper_2",
              [&] {
                using offset_t = index_t;
                AT_DISPATCH_INDEX_TYPES(
                    indices.scalar_type(),
                    "index_select_scalar_cumsum_wrapper_3",
                    [&] {
                      index_select_scalar_cumsum_kernel<
                          length_t,
                          index_t,
                          offset_t,
                          MAX_CUMSUM_ENTRIES_PER_BLOCK,
                          MAX_CUMSUM_ENTRIES_PER_BLOCK>
                          <<<grid_size,
                             MAX_CUMSUM_ENTRIES_PER_BLOCK,
                             0,
                             at::cuda::getCurrentCUDAStream()>>>(
                              output_lengths.data_ptr<length_t>(),
                              output_offsets.data_ptr<offset_t>(),
                              lengths.data_ptr<length_t>(),
                              indices.data_ptr<index_t>(),
                              num_batches,
                              batch_size,
                              indices.numel(),
                              num_output_lengths -
                                  MAX_CUMSUM_ENTRIES_PER_BLOCK *
                                      (grid_size - 1),
                              grid_size > 1 ? block_flags.data_ptr<int>()
                                            : nullptr,
                              grid_size > 1 ? block_sums.data_ptr<offset_t>()
                                            : nullptr);
                      C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
              });
        });

    // TODO: Try to not do D->H transfer
    const int64_t num_outputs =
        output_offsets[output_offsets.numel() - 1].item<int64_t>();
    Tensor output = at::empty({num_outputs}, values.options());
    Tensor output_weights;
    if (weights.has_value()) {
      output_weights = at::empty({num_outputs}, weights.value().options());
    }
    grid_size = cuda_calc_xblock_count(num_outputs, kMaxThreads);

    if (grid_size != 0) {
#define LAUNCH_KERNEL(WEIGHTED, WEIGHT_TYPE, OUTPUT_WEIGHTS, WEIGHTS)      \
  {                                                                        \
    keyed_jagged_index_select_dim1_kernel<                                 \
        value_t,                                                           \
        index_t,                                                           \
        offset_t,                                                          \
        WEIGHT_TYPE,                                                       \
        WEIGHTED>                                                          \
        <<<grid_size, kMaxThreads, 0, at::cuda::getCurrentCUDAStream()>>>( \
            output.data_ptr<value_t>(),                                    \
            OUTPUT_WEIGHTS,                                                \
            values.data_ptr<value_t>(),                                    \
            WEIGHTS,                                                       \
            offsets.data_ptr<offset_t>() + 1,                              \
            indices.data_ptr<index_t>(),                                   \
            output_offsets.data_ptr<offset_t>(),                           \
            num_batches,                                                   \
            batch_size,                                                    \
            indices.numel(),                                               \
            num_outputs);                                                  \
  }
      AT_DISPATCH_ALL_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          values.scalar_type(),
          "keyed_jagged_index_select_dim1_warpper_1",
          [&] {
            using value_t = scalar_t;
            AT_DISPATCH_INDEX_TYPES(
                offsets.scalar_type(),
                "keyed_jagged_index_select_dim1_warpper_2",
                [&] {
                  using offset_t = index_t;
                  AT_DISPATCH_INDEX_TYPES(
                      indices.scalar_type(),
                      "keyed_jagged_index_select_dim1_warpper_3",
                      [&] {
                        if (weights.has_value()) {
                          AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                              weights.value().scalar_type(),
                              "keyed_jagged_index_select_dim1_warpper_4",
                              [&] {
                                using weight_t = scalar_t;
                                LAUNCH_KERNEL(
                                    true,
                                    weight_t,
                                    output_weights.data_ptr<weight_t>(),
                                    weights.value().data_ptr<weight_t>())
                              });
                        } else {
                          LAUNCH_KERNEL(false, scalar_t, nullptr, nullptr)
                        }
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      });
                });
          });
    }

#undef LAUNCH_KERNEL

    ctx->save_for_backward({indices, output_offsets, offsets});
    ctx->saved_data["num_outputs"] = num_outputs;
    ctx->saved_data["num_inputs"] = values.numel();
    ctx->saved_data["batch_size"] = batch_size;
    ctx->saved_data["num_batches"] = num_batches;
    ctx->saved_data["has_weights"] = weights.has_value();

    if (weights.has_value()) {
      return {output, output_lengths, output_weights};
    }
    return {output, output_lengths};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    bool has_weights = ctx->saved_data["has_weights"].toBool();
    TORCH_CHECK(
        (has_weights && grad_outputs.size() == 3) || grad_outputs.size() == 2);

    const Tensor& grad = grad_outputs[0];
    TENSOR_ON_CUDA_GPU(grad_outputs[0]);

    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    const Tensor& indices = *savedItr++;
    const Tensor& grad_offsets = *savedItr++;
    const Tensor& output_offsets = *savedItr++;

    TENSORS_ON_SAME_DEVICE(grad, indices);

    int64_t num_outputs = ctx->saved_data["num_inputs"].toInt();
    int64_t output_batch_size = ctx->saved_data["batch_size"].toInt();
    int64_t num_batches = ctx->saved_data["num_batches"].toInt();

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(grad.get_device());

    Tensor grad_input = at::zeros({num_outputs}, grad.options());
    auto grid_size = cuda_calc_xblock_count(grad.numel(), kMaxThreads);

    if (grid_size != 0) {
      AT_DISPATCH_ALL_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          grad.scalar_type(),
          "keyed_jagged_index_add_dim1_wrapper_1",
          [&] {
            AT_DISPATCH_INDEX_TYPES(
                grad_offsets.scalar_type(),
                "keyed_jagged_index_add_dim1_wrapper_2",
                [&] {
                  using offset_t = index_t;
                  AT_DISPATCH_INDEX_TYPES(
                      indices.scalar_type(),
                      "keyed_jagged_index_add_dim1_wrapper_3",
                      [&] {
                        keyed_jagged_index_add_dim1_kernel<<<
                            grid_size,
                            kMaxThreads,
                            0,
                            at::cuda::getCurrentCUDAStream()>>>(
                            grad_input.data_ptr<scalar_t>(),
                            grad.data_ptr<scalar_t>(),
                            grad_offsets.data_ptr<offset_t>(),
                            indices.data_ptr<index_t>(),
                            output_offsets.data_ptr<offset_t>() +
                                1, // shift it to make it inclusive cumsum
                            num_batches,
                            indices.numel(),
                            output_batch_size,
                            grad.numel());
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      });
                });
          });
    }

    return {
        grad_input,
        torch::autograd::Variable(), // lengths
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable(), // indices
        torch::autograd::Variable(), // batch_size
        torch::autograd::Variable() // weights
    };
  }
};
} // namespace

std::vector<Tensor> keyed_jagged_index_select_dim_1_gpu(
    const Tensor& values,
    const Tensor& lengths,
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t batch_size,
    const c10::optional<Tensor>& weights) {
  return KeyedJaggedIndexSelectDim1GPUOp::apply(
      values, lengths, offsets, indices, batch_size, weights);
}

} // namespace fbgemm_gpu

JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "keyed_jagged_index_select_dim1",
    fbgemm_gpu::keyed_jagged_index_select_dim_1_gpu);

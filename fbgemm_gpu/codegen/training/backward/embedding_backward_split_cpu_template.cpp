/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include <map>
#include <tuple>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

#include "fbgemm_gpu/embedding_forward_split_cpu.h"
#include "fbgemm/FbgemmEmbedding.h"
#include "fbgemm/Types.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/cpu_utils.h"
#include "fbgemm_gpu/utils/ops_utils.h"

#if FBGEMM_GPU_MEMCHECK
#define FBGEMM_MEM_CHECK_ONLY
#else
#define FBGEMM_MEM_CHECK_ONLY maybe_unused
#endif

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

namespace internal {
template <typename T>
struct half2float16 {
  using type = T;
};

template <>
struct half2float16<at::Half> {
  using type = fbgemm::float16;
};
} // namespace internal

namespace {
template <typename index_t, typename scalar_t, typename grad_t>
void split_embedding_backward_exact_cpu_kernel(
    Tensor grad_output,
    Tensor host_weights,
    const at::TensorAccessor<int64_t, 1> weights_offsets_data,
    const at::TensorAccessor<int, 1> D_offsets_data,
    Tensor hash_size_cumsum,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int num_tables,
    int B,
    const int* table_to_feature_offset,
    {% if "momentum1_offsets" in args.split_function_arg_names %}
    const at::TensorAccessor<int64_t, 1> momentum1_offsets_data,
    {% endif %}
    {% if "momentum2_offsets" in args.split_function_arg_names %}
    const at::TensorAccessor<int64_t, 1> momentum2_offsets_data,
    {% endif %}
    {{ args.split_cpu_kernel_args | join(", ") }}) {
  const grad_t* grad_output_data = grad_output.data_ptr<grad_t>();
  auto host_weights_data = host_weights.accessor<scalar_t, 1>();
  const auto hash_size_cumsum_data = hash_size_cumsum.accessor<int64_t, 1>();

  auto grad_stride = grad_output.size(1);

  std::vector<::internal::HyperCompressedSparseColumn> cscs(num_tables);

  auto get_hash_size = [&hash_size_cumsum_data](int feature_begin) {
    int64_t hash_size;
    int t_temp = feature_begin + 1;
    do {
      hash_size =
          hash_size_cumsum_data[t_temp] - hash_size_cumsum_data[feature_begin];
      ++t_temp;
    } while (hash_size == 0);
    TORCH_CHECK(
        hash_size < ((1L << 31) - 1),
        "CPU exact rowwise adagrad currently doesn't support embedding tables "
        "with more than 2B rows");
    return hash_size;
  };
for (const auto t : c10::irange(num_tables)) {
    int feature_begin = table_to_feature_offset[t];
    int64_t hash_size = get_hash_size(feature_begin);

#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name = "::internal::csr2csc";
#endif
    using weight_t = at::acc_type<scalar_t, true>;
    ::internal::csr2csc(
        cscs[t],
        B,
        MAKE_TA_WITH_NAME(func_name, offsets, index_t, 1),
        MAKE_TA_WITH_NAME(func_name, indices, index_t, 1),
        MAKE_TA_WITH_NAME(func_name, indice_weights, weight_t, 1),
        pooling_mode,
        table_to_feature_offset + t,
        hash_size);
  }
for (const auto t : c10::irange(num_tables)) {
    int feature_begin = table_to_feature_offset[t];

    int num_non_zero_columns = cscs[t].num_non_zero_columns;
    int* col_segment_ptr = cscs[t].column_segment_ptr;
    int* col_segment_indices = cscs[t].column_segment_indices;

    const auto D_begin = D_offsets_data[feature_begin];
    const auto D =
        D_offsets_data[feature_begin + 1] - D_offsets_data[feature_begin];
    const auto table_begin = weights_offsets_data[feature_begin];
    bool is_shared_table =
        table_to_feature_offset[t + 1] > table_to_feature_offset[t] + 1;

    {% if optimizer == "rowwise_adagrad" %}
    const auto hash_size = get_hash_size(feature_begin);
    constexpr bool use_fbgemm = std::is_same<scalar_t, float>::value
                                && std::is_same<scalar_t, grad_t>::value;
    // || std::is_same<scalar_t, at::Half>::value;
    if (use_fbgemm && !is_shared_table) {
      // fbgemm handles common case of no shared table
      using fbgemm_weight_t = typename ::internal::half2float16<scalar_t>::type;
      auto spmdm_kernel = fbgemm::GenerateEmbeddingSpMDMWithStrides<
          fbgemm_weight_t,
          /*IndexType=*/int32_t,
          /*OffsetType=*/int32_t>(
          D,
          cscs[t].weights != nullptr,
          /*normalize_by_lengths=*/false,
          /*prefetch=*/16,
          /*is_weight_positional=*/false,
          /*use_offsets=*/true,
          /*output_stride=*/-1,
          /*input_stride=*/grad_stride);
      auto rowwise_adagrad_kernel =
          fbgemm::GenerateSparseAdaGrad</*IndexType=*/int>(D, /*rowwise=*/true);

      constexpr int C_BLOCK = 64;
      at::parallel_for(0, num_non_zero_columns, C_BLOCK, [&](int64_t c0, int64_t c1) {
        grad_t grad_blocked_buffer[C_BLOCK * D];
        for (int64_t c = c0; c < c1; c += C_BLOCK) {
          const int* offsets_begin_ptr = col_segment_ptr + c;
          int64_t c_block_end = std::min(c + C_BLOCK, c1);
          bool success = spmdm_kernel(
              c_block_end - c,
              col_segment_ptr[c_block_end] - *offsets_begin_ptr,
              B,
              reinterpret_cast<const fbgemm_weight_t*>(
                  grad_output_data + D_begin),
              cscs[t].row_indices + *offsets_begin_ptr,
              offsets_begin_ptr,
              cscs[t].weights == nullptr
                  ? nullptr
                  : cscs[t].weights + *offsets_begin_ptr,
              reinterpret_cast<float*>(grad_blocked_buffer));

          if (!success) {
            fbgemm_gpu::report_embedding_error(
              t,
              B,
              c,
              c_block_end,
              col_segment_ptr,
              cscs[t].row_indices,
              hash_size,
              /*allow_minus_one=*/false);
          }
          int num_rows_processed = rowwise_adagrad_kernel(
              c_block_end - c,
              hash_size * D,
              reinterpret_cast<float*>(&host_weights_data[table_begin]),
              reinterpret_cast<const float*>(grad_blocked_buffer),
              reinterpret_cast<float*>(
                  &momentum1_host[momentum1_offsets_data[feature_begin]]),
              col_segment_indices + c,
              eps,
              -learning_rate,
              /*weight_decay=*/0,
              /*counter=*/nullptr,
              /*counter_halflife=*/0);

          TORCH_CHECK(num_rows_processed == c_block_end - c,
              "num of rows processed by adagrad: ",
              num_rows_processed,
              "does not match c_block size: ",
              c_block_end - c);
        } // for each c
      }); // parallel for
    } else
    {% endif %}
    {
      // no fbgemm
      // TODO: to parallelize, we should easily identify segments belong to
      // the same column.
      at::acc_type<grad_t, true> grad_buffer[D];
      for (const auto c : c10::irange(num_non_zero_columns)) {
        int64_t idx = col_segment_indices[c];
        if (c == 0 || col_segment_indices[c - 1] != idx) {
          memset(grad_buffer, 0, D * sizeof(at::acc_type<grad_t, true>));
        }
        [[maybe_unused]] const int64_t embedding_begin = table_begin + idx * D;
        
        for (int r = col_segment_ptr[c]; r < col_segment_ptr[c + 1]; ++r) {
          int D_offset = D_begin;
          if (is_shared_table) {
            D_offset += cscs[t].column_segment_ids[r] * D;
          }
          int b = cscs[t].row_indices[r];
          
          for (const auto d : c10::irange(D)) {
            if (cscs[t].weights != nullptr) {
              grad_buffer[d] += grad_output_data[b * grad_stride + D_offset + d] *
                    cscs[t].weights[r];
            } else {
              grad_buffer[d] += grad_output_data[b * grad_stride + D_offset + d];
            }
          }
        }
        if (c == num_non_zero_columns - 1 || col_segment_indices[c + 1] != idx) {
          {{ split_weight_update_cpu }}
        }
      } // for each c
    } // no fbgemm
  } // for each table
}

template <typename index_t, typename scalar_t, typename grad_t>
void split_embedding_nobag_backward_exact_cpu_kernel(
    const Tensor& grad_output,
    const Tensor& host_weights,
    const Tensor& weights_offsets,
    int64_t D,
    const Tensor& hash_size_cumsum,
    const Tensor& indices,
    const Tensor& offsets,
    int num_tables,
    int B,
    const int* table_to_feature_offset,
    {% if "momentum1_offsets" in args.split_function_arg_names %}
    const at::TensorAccessor<int64_t, 1> momentum1_offsets_data,
    {% endif %}
    {% if "momentum2_offsets" in args.split_function_arg_names %}
    const at::TensorAccessor<int64_t, 1> momentum2_offsets_data,
    {% endif %}
    {{ args.split_cpu_kernel_args | join(", ") }}) {
      const grad_t* grad_output_data = grad_output.data_ptr<grad_t>();
      auto host_weights_data = host_weights.accessor<scalar_t, 1>();
      const auto hash_size_cumsum_data = hash_size_cumsum.accessor<int64_t, 1>();
      const auto indices_data = indices.data_ptr<index_t>();
      const auto offsets_data = offsets.data_ptr<index_t>();
      const auto weights_offsets_data = weights_offsets.data_ptr<int64_t>();
     
      typedef std::unordered_map<int64_t, std::vector<at::acc_type<grad_t, true>>> tb_grad_buffer_map_t;
      typedef std::unordered_map<int64_t, int64_t> tb_fb_map_t;

      std::unordered_map<index_t, tb_grad_buffer_map_t> idx_tb_grad_buffer;
      std::unordered_map<index_t, tb_fb_map_t> idx_tb_fb;
      
      int64_t T = weights_offsets.size(0);
      for (const auto t : c10::irange(T)) {
        int64_t hash_size = 0;
        int64_t t_temp = static_cast<int64_t>(t) + 1;
        do {
          hash_size = hash_size_cumsum_data[t_temp] - hash_size_cumsum_data[t];
          ++t_temp;
        } while (hash_size == 0);
        
        [[maybe_unused]] const auto feature_begin = t;
        [[maybe_unused]] const auto table_begin = weights_offsets_data[feature_begin];
        bool success = true;
        
        for (const auto b : c10::irange(B)) {
          const auto indices_start = offsets_data[t * B + b];
          const auto indices_end = offsets_data[t * B + b + 1];
          
          for (auto i = indices_start; i < indices_end; ++i) {
            const auto idx = indices_data[i];
            if (idx < 0 || idx >= hash_size) {
              success = false;
              continue;
            }
            
            auto& grad_buffer = idx_tb_grad_buffer[idx][table_begin];
            idx_tb_fb[idx][table_begin] = feature_begin;
            if (grad_buffer.empty()) {
              grad_buffer.resize(D, 0);
            }
            
            for (const auto d : c10::irange(D)) {
              grad_buffer[d] += grad_output_data[i * D + d];
            }
          }
        }
        
        if (!success) {
          fbgemm_gpu::report_embedding_error(
            t, B, 0, B, offsets_data, indices_data, hash_size);
        }
      }
    
      std::vector<index_t> idx_vec;
      idx_vec.reserve(idx_tb_grad_buffer.size());
      for (const auto& [idx, _] : idx_tb_grad_buffer) {
        idx_vec.push_back(idx);
      }
    
      at::parallel_for(0, idx_vec.size(), 0, [&](int64_t start_idx, int64_t end_idx) {
        for (int64_t i = start_idx; i < end_idx; ++i) {
          const auto idx = idx_vec[i];
          const auto& tb_grad_buffer = idx_tb_grad_buffer[idx];
          
          for (const auto& [table_begin, grad_buffer] : tb_grad_buffer) {
            [[maybe_unused]] const auto feature_begin = idx_tb_fb[idx][table_begin];
            [[maybe_unused]] const int64_t embedding_begin = table_begin + idx * D;
            {{ split_weight_update_cpu }}
          }
        }
      });
}

template <typename index_t, typename scalar_t>
void split_embedding_backward_exact_cpu_dense_kernel(
    Tensor grad,
    Tensor grad_output,
    const at::TensorAccessor<int64_t, 1> weights_offsets_data,
    const at::TensorAccessor<int, 1> D_offsets_data,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int num_tables,
    int B,
    const int* table_to_feature_offset) {
  auto grad_data = grad.data_ptr<scalar_t>();

  auto grad_output_data = grad_output.accessor<scalar_t, 2>();

  [[FBGEMM_MEM_CHECK_ONLY]] const auto func_name = "split_embedding_backward_exact_cpu_dense_kernel";

  const auto indices_data = MAKE_TA_WITH_NAME(func_name, indices, index_t, 1);
  const auto offsets_data = MAKE_TA_WITH_NAME(func_name, offsets, index_t, 1);
  const auto indice_weights_data = indice_weights.defined()
      ?
      // If indice_weights are not defined, then this accessor won't be
      // used
      indice_weights.accessor<scalar_t, 1>()
      : grad.accessor<scalar_t, 1>(); // this is just to make compiler
                                      // happy

  at::parallel_for(0, num_tables, 0, [&](int64_t t_begin, int64_t t_end) {
    for (int64_t t = table_to_feature_offset[t_begin];
         t < table_to_feature_offset[t_end];
         ++t) {
      const auto D_begin = D_offsets_data[t];
      const auto D = D_offsets_data[t + 1] - D_offsets_data[t];
      const auto table_begin = weights_offsets_data[t];
for (const auto b : c10::irange(B)) {
        const auto pool_begin = offsets_data[t * B + b];
        const auto pool_end = offsets_data[t * B + b + 1];
        const auto L = pool_end - pool_begin;
        const scalar_t scale_factor =
            // NOTE: MEAN pooling will not work with indice_weights!
            (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && !indice_weights.defined() && L > 0)
            ? 1.0 / L
            : 1.0;
for (const auto p : c10::irange(pool_begin,pool_end)) {
          const int64_t embedding_begin = table_begin + indices_data[p] * D;
          const scalar_t v = indice_weights.defined()
              ? (indice_weights_data[p] * scale_factor)
              : scale_factor;
for (const auto d : c10::irange(D)) {
            grad_data[embedding_begin + d] +=
                grad_output_data[b][D_begin + d] * v;
          }
        }
      }
    }
  }); // parallel_for
}
} // namespace

{%- for nobag in ([False] if (dense) else [True, False]) %}
{%- set ndesc = "_nobag" if nobag else "" %}
// The template for exact optimizers
{{ "void" if not dense else "Tensor" }}  split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_cpu(
    Tensor grad_output,
    Tensor host_weights,
    {% if not dense %}
    Tensor weights_placements,
    {% endif %}
    Tensor weights_offsets,
    {% if nobag %}
    int64_t D,
    {% else %}
    Tensor D_offsets,
    int64_t max_D,
    {% endif %}
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    {% if not nobag %}
    int64_t pooling_mode,
    Tensor indice_weights,
    {% endif %}
    {% if not dense %}
    bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }}
    {%- if not nobag %}
    , int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)
    {%- endif %}
    {% else %}
    {{ args.split_function_args | join(", ") }}
    {% endif %}
) {
  {% if not nobag %}
  int64_t T = D_offsets.numel() - 1;
  const auto D_offsets_data = D_offsets.accessor<int, 1>();
  {% else %}
  int64_t T = weights_offsets.size(0);
  {% endif %}
  TORCH_CHECK_GT(T, 0);
  // offsets = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK_GE(B, 0);
  
  {%- if "learning_rate" in args.split_cpu_kernel_arg_constructors %}
  // convert `learning rate` to float since `learning rate` is float in kernels
  const float learning_rate = learning_rate_tensor.item<float>();
  {%- endif %}
  
  {%- if not nobag %}
  const auto weights_offsets_data = weights_offsets.accessor<int64_t, 1>();
  {%- endif %}

  const auto hash_size_cumsum_data = hash_size_cumsum.accessor<int64_t, 1>();

  int num_tables = 0; // # of physical tables
  int table_to_feature_offset[T + 1];
  table_to_feature_offset[0] = 0;
  for (int feature = 0; feature < T - 1; ++feature) {
    if (hash_size_cumsum_data[feature + 1] != hash_size_cumsum_data[feature]) {
      ++num_tables;
      table_to_feature_offset[num_tables] = feature + 1;
    }
  }
  ++num_tables;
  table_to_feature_offset[num_tables] = T;

  TORCH_CHECK_EQ(host_weights.dim(), 1);

  {% if not dense %}
  {% if "momentum1_offsets" in args.split_function_arg_names %}
  const auto momentum1_offsets_data = momentum1_offsets.accessor<int64_t, 1>();
  {% endif %}
  {% if "momentum2_offsets" in args.split_function_arg_names %}
  const auto momentum2_offsets_data = momentum2_offsets.accessor<int64_t, 1>();
  {% endif %}

  grad_output = grad_output.contiguous();

  AT_DISPATCH_INDEX_TYPES(
    indices.scalar_type(), 
    "split_embedding_backward_exact_cpu_kernel_1", [&] {
      
    FBGEMM_DISPATCH_FLOAT_AND_HALF(
      grad_output.scalar_type(),
      "split_embedding_backward_exact_cpu_kernel_2", [&] {
      using grad_t = scalar_t;

      FBGEMM_DISPATCH_FLOAT_AND_HALF(
        host_weights.scalar_type(), 
        "split_embedding_backward_exact_cpu_kernel_3", [&] {

          split_embedding{{ ndesc }}_backward_exact_cpu_kernel<index_t, scalar_t, grad_t>(
              grad_output,
              host_weights,
              {% if nobag %}
              weights_offsets,
              D,
              {% else %}
              weights_offsets_data,
              D_offsets_data,
              {% endif %}
              hash_size_cumsum,
              indices,
              offsets,
              {% if not nobag %}
              pooling_mode,
              indice_weights,
              {% endif %}
              num_tables,
              B,
              table_to_feature_offset,
              {% if "momentum1_offsets" in args.split_function_arg_names %}
              momentum1_offsets_data,
              {% endif %}
              {% if "momentum2_offsets" in args.split_function_arg_names %}
              momentum2_offsets_data,
              {% endif %}
              {{ args.split_cpu_kernel_arg_constructors | join(", ") }});
        });
      });
    });

  return;

  {% else %}

  // When input is dense enough, avoid sorting and just treat as dense.
  auto grad = zeros_like(host_weights, grad_output.dtype());
  AT_DISPATCH_INDEX_TYPES(
    indices.scalar_type(), 
    "split_embedding_backward_exact_cpu_dense_kernel", [&] {

    FBGEMM_DISPATCH_FLOAT_AND_HALF(
      grad_output.scalar_type(), 
      "split_embedding_backward_exact_cpu", [&] {

        split_embedding_backward_exact_cpu_dense_kernel<index_t, scalar_t>(
            grad,
            grad_output,
            weights_offsets_data,
            D_offsets_data,
            indices,
            offsets,
            pooling_mode,
            indice_weights,
            num_tables,
            B,
            table_to_feature_offset);
      });
    });

  return grad;
  {% endif %}
}
{%- endfor %} {#-/*for nobag*/#}   

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  {%- for nobag in ([False] if (dense) else [True, False]) %}
  {%- set ndesc = "_nobag" if nobag else "" %}

  {% if not dense %}
  m.def("split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_cpu("
    "Tensor grad_output, "
    "Tensor(a!) host_weights, "
    "Tensor weights_placements, "
    "Tensor weights_offsets, "
    {%- if nobag %}
    "int D, "
    {%- else %}
    "Tensor D_offsets, "
    "int max_D, "
    {%- endif %}
    "Tensor hash_size_cumsum, "
    "int total_hash_size_bits, "
    "Tensor indices, "
    "Tensor offsets, "
    {%- if not nobag %}
    "int pooling_mode, "
    "Tensor indice_weights, "
    {%- endif %}
    "bool stochastic_rounding, "
    "{{ (args.split_function_args | join(", ")).replace("double", "float").replace("int64_t", "int").replace("Tensor momentum1_host", "Tensor(b!) momentum1_host")}}"
    {%- if not nobag %}
    ", int output_dtype = 0"
    {%- endif %}
    ") -> ()");
  {% else %}
  m.def("split_embedding_backward_codegen_{{ optimizer }}_cpu(Tensor grad_output, Tensor(a!) host_weights, Tensor weights_offsets, Tensor D_offsets, int max_D, Tensor hash_size_cumsum, int total_hash_size_bits, Tensor indices, Tensor offsets,int pooling_mode, Tensor indice_weights, {{ (args.split_function_args | join(", ")).replace("double", "float").replace("int64_t", "int").replace("Tensor momentum1_host", "Tensor(b!) momentum1_host")}}) -> Tensor");
  {% endif %}
  DISPATCH_TO_CPU("split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_cpu", split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_cpu);

  {%- endfor %} {#-/*for nobag*/#}   
}

// clang-format on

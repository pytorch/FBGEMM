/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/nn/functional.h>

#include <optional>

#include "hstu.h"
#include "static_switch.h"

namespace fbgemm_gpu::hstu {

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                           \
  TORCH_CHECK(                                        \
      x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
      #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(
    Hstu_fwd_params& params,
    // sizes
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t target_group_size,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t h_rab,
    const size_t d,
    const float alpha,
    const int quant_mode,
    // device pointers
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor q_descale,
    const at::Tensor k_descale,
    const at::Tensor v_descale,
    const at::Tensor rab,
    at::Tensor out,
    void* num_contexts_d,
    void* cu_seqlens_q_d,
    void* cu_seqlens_k_d,
    void* cu_seqlens_vt_descale_d,
    void* cu_seqlens_q_block_descale,
    void* cu_seqlens_kv_block_descale,
    void* num_targets_d,
    bool has_rab,
    const at::Tensor& func,
    int window_size_left,
    int window_size_right,
    const at::Tensor vt,
    const at::Tensor vt_descale) {
  // Reset the parameters
  params = {};

  params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 +
      at::cuda::getCurrentDeviceProperties()->minor;
  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;
#ifdef HSTU_DISABLE_BF16
  TORCH_CHECK(
      !params.is_bf16, "This hstu attention build does not support bf16.");
#endif
#ifdef HSTU_DISABLE_FP16
  TORCH_CHECK(
      q.dtype() != torch::kFloat16,
      "This hstu attention build does not support fp16.");
#endif
#ifdef HSTU_DISABLE_FP8
  TORCH_CHECK(
      !params.is_e4m3, "This hstu attention build does not support fp8_e4m3.");
#endif

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  if (vt.numel() > 0) {
    params.vt_ptr = vt.data_ptr();
  } else {
    params.vt_ptr = nullptr;
  }
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  if (vt.numel() > 0) {
    params.vt_row_stride = vt.stride(-1);
    params.vt_head_stride = vt.stride(-2);
  }

  if (out.numel() > 0) {
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);
  }

  params.has_rab = has_rab;
#ifdef HSTU_DISABLE_RAB
  TORCH_CHECK(!has_rab, "This hstu attention build does not support has_rab.");
#endif
  if (has_rab) {
    params.rab_ptr = rab.data_ptr();
    params.rab_batch_stride = rab.stride(0);
    params.rab_row_stride = rab.stride(-2);
    params.rab_head_stride = rab.stride(-3);
    params.h_rab = h_rab;
  } else {
    params.rab_ptr = nullptr;
    params.rab_batch_stride = 0;
    params.rab_row_stride = 0;
    params.rab_head_stride = 0;
    params.h_rab = h;
  }

  if (q_descale.numel() > 0) {
    params.descale_q_ptr = q_descale.data_ptr<float>();
    params.descale_q_head_stride = q_descale.stride(0);
  } else {
    params.descale_q_ptr = nullptr;
    params.descale_q_head_stride = 0;
  }
  if (k_descale.numel() > 0) {
    params.descale_k_ptr = k_descale.data_ptr<float>();
    params.descale_k_head_stride = k_descale.stride(0);
  } else {
    params.descale_k_ptr = nullptr;
    params.descale_k_head_stride = 0;
  }
  if (v_descale.numel() > 0) {
    params.descale_v_ptr = v_descale.data_ptr<float>();
    params.descale_v_head_stride = v_descale.stride(0);
  } else {
    params.descale_v_ptr = nullptr;
    params.descale_v_head_stride = 0;
  }
  if (vt_descale.numel() > 0) {
    params.descale_vt_ptr = vt_descale.data_ptr<float>();
    params.descale_vt_row_stride = vt_descale.stride(-3);
    params.descale_vt_head_stride = vt_descale.stride(-2);
    params.cu_seqlens_vt_descale = static_cast<int*>(cu_seqlens_vt_descale_d);
  } else {
    params.descale_vt_ptr = nullptr;
    params.descale_vt_row_stride = 0;
    params.descale_vt_head_stride = 0;
    params.cu_seqlens_vt_descale = nullptr;
  }

  if (quant_mode == 2) {
    params.q_block_descale_head_stride = q_descale.stride(-2);
    params.kv_block_descale_head_stride = k_descale.stride(-2);
    params.cu_seqlens_q_block_descale = static_cast<int*>(cu_seqlens_q_block_descale);
    params.cu_seqlens_kv_block_descale = static_cast<int*>(cu_seqlens_kv_block_descale);
  }

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);

  // Set num SM
  params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.alpha = alpha;
  // 1: 1xDIM&128x1 quantization, 2: per-block quantization, 3: per-head quantization, 4: per-batch quantization, 5: per-tensor quantization.
  params.quant_mode = quant_mode;

  // Set the masks.
  params.is_target = num_targets_d != nullptr;
  params.num_targets = static_cast<int*>(num_targets_d);
#ifdef HSTU_DISABLE_TARGET
  TORCH_CHECK(
      !params.is_target, "This hstu attention build does not support target.");
#endif
  params.target_group_size = target_group_size;
  if (params.is_target) {
    TORCH_CHECK(
        target_group_size > 0,
        "target_group_size must be greater than 0 when target is True");
  }
  params.is_context = num_contexts_d != nullptr;
  params.num_contexts = static_cast<int*>(num_contexts_d);
#ifdef HSTU_DISABLE_CONTEXT
  TORCH_CHECK(
      !params.is_context,
      "This hstu attention build does not support context mask.");
#endif

  if (window_size_left < 0 || window_size_left > (int)seqlen_k) {
    window_size_left = seqlen_k;
  }
  if (window_size_right < 0 || window_size_right > (int)seqlen_k) {
    window_size_right = seqlen_k;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

  params.is_causal =
      window_size_left == (int)seqlen_k && window_size_right == 0;
#ifdef HSTU_DISABLE_CAUSAL
  TORCH_CHECK(
      !params.is_causal, "This hstu attention build does not support causal.");
#endif
  TORCH_CHECK(
      !(!params.is_causal && params.is_target),
      "Target mask is True, but causal mask is False, this is undefined behavior.");
  TORCH_CHECK(
      !(!params.is_causal && params.is_context),
      "Context mask is True, but causal mask is False, this is undefined behavior.");
  params.is_local =
      (window_size_left < (int)seqlen_k || window_size_right < (int)seqlen_k) &&
      !params.is_causal;
#ifdef HSTU_DISABLE_LOCAL
  TORCH_CHECK(
      !params.is_local,
      "This hstu attention build does not support local mask.");
#endif

  params.is_arbitrary_mask = func.defined();
#ifdef HSTU_DISABLE_ARBITRARY
  TORCH_CHECK(!params.is_arbitrary_mask, "This hstu attention build does not support arbitrary mask.");
#endif
  if (params.is_arbitrary_mask) {
    TORCH_CHECK(func.dtype() == torch::kInt32, "func must have dtype int32");
    CHECK_DEVICE(func);

    params.func_ptr = func.data_ptr();
    params.func_ids_stride = func.stride(-2);
    params.func_head_stride = func.stride(-3);
    params.n_func = func.size(-2);
    TORCH_CHECK(params.n_func == HSTU_ARBITRARY_NFUNC, "n_func must be equal to HSTU_ARBITRARY_NFUNC");
  }
}

void set_params_dgrad(
    Hstu_bwd_params& params,
    // sizes
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t target_group_size,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t h_rab,
    const size_t d,
    const float alpha,
    const int quant_mode,
    // device pointers
    const at::Tensor q,
    const at::Tensor qt,
    const at::Tensor k,
    const at::Tensor kt,
    const at::Tensor v,
    const at::Tensor dout,
    const at::Tensor dout_t,
    const at::Tensor rab,
    const at::Tensor do_descale,
    const at::Tensor q_descale,
    const at::Tensor k_descale,
    const at::Tensor v_descale,
    const at::Tensor dot_descale,
    const at::Tensor qt_descale,
    const at::Tensor kt_descale,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor drab,
    void* num_contexts_d,
    void* cu_seqlens_q_d,
    void* cu_seqlens_k_d,
    void* cu_seqlens_qt_descale_d,
    void* cu_seqlens_kt_descale_d,
    void* cu_seqlens_q_block_descale,
    void* cu_seqlens_kv_block_descale,
    void* num_targets_d,
    void* dq_accum_d,
    const at::Tensor& func,
    bool has_rab,
    bool has_drab,
    int window_size_left,
    int window_size_right,
    bool deterministic) {
  set_params_fprop(
      params,
      b,
      seqlen_q,
      seqlen_k,
      target_group_size,
      seqlen_q_rounded,
      seqlen_k_rounded,
      h,
      h_k,
      h_rab,
      d,
      alpha,
      quant_mode,
      q,
      k,
      v,
      q_descale,
      k_descale,
      v_descale,
      rab,
      /*out=*/torch::Tensor(),
      num_contexts_d,
      cu_seqlens_q_d,
      cu_seqlens_k_d,
      /*cu_seqlens_vt_descale_d=*/nullptr,
      cu_seqlens_q_block_descale,
      cu_seqlens_kv_block_descale,
      num_targets_d,
      has_rab,
      func,
      window_size_left,
      window_size_right,
      /*vt=*/torch::Tensor(),
      /*vt_descale=*/torch::Tensor());

  params.is_e5m2 = q.dtype() == torch::kFloat8_e5m2;
  params.has_drab = has_drab;
#ifdef HSTU_DISABLE_DRAB
  TORCH_CHECK(
      !has_drab, "This hstu attention build does not support has_drab.");
#endif
  if (has_drab) {
    params.drab_ptr = drab.data_ptr();
    params.drab_batch_stride = drab.stride(0);
    params.drab_row_stride = drab.stride(-2);
    params.drab_head_stride = drab.stride(-3);
  } else {
    params.drab_ptr = nullptr;
    params.drab_batch_stride = 0;
    params.drab_row_stride = 0;
    params.drab_head_stride = 0;
  }
  // Set the pointers and strides.
  params.do_ptr = dout.data_ptr();
  params.do_row_stride = dout.stride(-3);
  params.do_head_stride = dout.stride(-2);
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();
  params.dq_row_stride = dq.stride(-3);
  params.dk_row_stride = dk.stride(-3);
  params.dv_row_stride = dv.stride(-3);
  params.dq_head_stride = dq.stride(-2);
  params.dk_head_stride = dk.stride(-2);
  params.dv_head_stride = dv.stride(-2);
  if (qt.numel() > 0) {
    params.qt_ptr = qt.data_ptr();
    params.qt_row_stride = qt.stride(-1);
    params.qt_head_stride = qt.stride(-2);
  } else {
    params.qt_ptr = nullptr;
    params.qt_row_stride = 0;
    params.qt_head_stride = 0;
  }
  if (kt.numel() > 0) {
    params.kt_ptr = kt.data_ptr();
    params.kt_row_stride = kt.stride(-1);
    params.kt_head_stride = kt.stride(-2);
  } else {
    params.kt_ptr = nullptr;
    params.kt_row_stride = 0;
    params.kt_head_stride = 0;
  }
  if (dout_t.numel() > 0) {
    params.dot_ptr = dout_t.data_ptr();
    params.dot_row_stride = dout_t.stride(-1);
    params.dot_head_stride = dout_t.stride(-2);
  } else {
    params.dot_ptr = nullptr;
    params.dot_row_stride = 0;
    params.dot_head_stride = 0;
  }
  if (qt_descale.numel() > 0) {
  params.descale_qt_ptr = qt_descale.data_ptr<float>();
  params.descale_qt_head_stride = qt_descale.stride(-2);
  params.descale_qt_row_stride = qt_descale.stride(-3);
  } else {
    params.descale_qt_ptr = nullptr;
    params.descale_qt_head_stride = 0;
    params.descale_qt_row_stride = 0;
  }
  if (kt_descale.numel() > 0) {
    params.descale_kt_ptr = kt_descale.data_ptr<float>();
    params.descale_kt_head_stride = kt_descale.stride(-2);
    params.descale_kt_row_stride = kt_descale.stride(-3);
  } else {
    params.descale_kt_ptr = nullptr;
    params.descale_kt_head_stride = 0;
    params.descale_kt_row_stride = 0;
  }
  if (do_descale.numel() > 0) {
    params.descale_do_ptr = do_descale.data_ptr<float>();
    params.descale_do_head_stride = do_descale.stride(0);
  } else {
    params.descale_do_ptr = nullptr;
    params.descale_do_head_stride = 0;
  }
  if (dot_descale.numel() > 0) {
    params.descale_dot_ptr = dot_descale.data_ptr<float>();
    params.descale_dot_head_stride = dot_descale.stride(-2);
    params.descale_dot_row_stride = dot_descale.stride(-3);
  } else {
    params.descale_dot_ptr = nullptr;
    params.descale_dot_head_stride = 0;
    params.descale_dot_row_stride = 0;
  }

  params.cu_seqlens_descale_qt_ptr = static_cast<int*>(cu_seqlens_qt_descale_d);
  params.cu_seqlens_descale_kt_ptr = static_cast<int*>(cu_seqlens_kt_descale_d);

  params.dq_accum_ptr = dq_accum_d;

  params.deterministic = deterministic;
}

template <typename Dtype, int Headdim>
void run_hstu_fwd_mask(Hstu_fwd_params& params, cudaStream_t stream) {
  RAB_SWITCH(params.has_rab, Has_rab, [&] {
#ifndef HSTU_DISABLE_ARBITRARY
    if (params.is_arbitrary_mask) {
      run_hstu_fwd_<
          90,
          Dtype,
          Headdim,
          Has_rab,
          false,
          false,
          false,
          false,
          true,
          HSTU_ARBITRARY_NFUNC>(params, stream);
      return;
    }
#endif
#ifndef HSTU_DISABLE_LOCAL
    if (params.is_local) {
      run_hstu_fwd_<
          90,
          Dtype,
          Headdim,
          Has_rab,
          true,
          false,
          false,
          false,
          false,
          0>(params, stream);
      return;
    }
#endif
    if (!params.is_causal) {
      run_hstu_fwd_<
          90,
          Dtype,
          Headdim,
          Has_rab,
          false,
          false,
          false,
          false,
          false,
          0>(params, stream);
      return;
    } else {
#ifndef HSTU_DISABLE_CAUSAL
      CONTEXT_SWITCH(params.is_context, Is_context, [&] {
        TARGET_SWITCH(params.is_target, Is_target, [&] {
          run_hstu_fwd_<
              90,
              Dtype,
              Headdim,
              Has_rab,
              false,
              true,
              Is_context,
              Is_target,
              false,
              0>(params, stream);
        });
      });
#endif
    }
  });
}

void run_hstu_fwd_hopper(Hstu_fwd_params& params, cudaStream_t stream) {
  if (params.is_bf16) {
#ifndef HSTU_DISABLE_BF16
#ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) {
      run_hstu_fwd_mask<cutlass::bfloat16_t, 32>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) {
      run_hstu_fwd_mask<cutlass::bfloat16_t, 64>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) {
      run_hstu_fwd_mask<cutlass::bfloat16_t, 128>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) {
      run_hstu_fwd_mask<cutlass::bfloat16_t, 256>(params, stream);
    }
#endif
#else
    TORCH_CHECK(false, "This flash attention build does not support BF16.");
#endif
  } else if (params.is_e4m3) {
#ifndef HSTU_DISABLE_FP8
#ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) {
      std::cerr << "Not support dim = 32 and dtype = float_e4m3_t for now."
                << std::endl;
    }
#endif
#ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) {
      run_hstu_fwd_mask<cutlass::float_e4m3_t, 64>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) {
      run_hstu_fwd_mask<cutlass::float_e4m3_t, 128>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) {
      run_hstu_fwd_mask<cutlass::float_e4m3_t, 256>(params, stream);
    }
#endif
#else
    TORCH_CHECK(false, "This flash attention build does not support FP8.");
#endif
  } else {
#ifndef HSTU_DISABLE_FP16
#ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) {
      run_hstu_fwd_mask<cutlass::half_t, 32>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) {
      run_hstu_fwd_mask<cutlass::half_t, 64>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) {
      run_hstu_fwd_mask<cutlass::half_t, 128>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) {
      run_hstu_fwd_mask<cutlass::half_t, 256>(params, stream);
    }
#endif
#else
    TORCH_CHECK(false, "This flash attention build does not support FP16.");
#endif
  }
}

std::tuple<at::Tensor, at::Tensor> hstu_varlen_fwd_90(
    const at::Tensor&
        q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor&
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor&
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q, // b+1
    const at::Tensor& cu_seqlens_k, // b+1
    const int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const std::optional<at::Tensor>& num_contexts, // b
    const std::optional<at::Tensor>& num_targets, // b
    const int64_t target_group_size,
    int64_t window_size_left,
    int64_t window_size_right,
    const double alpha,
    std::optional<at::Tensor> rab,
    const std::optional<at::Tensor>& func,
    const int64_t quant_mode,
    const std::optional<at::Tensor>& vt,
    const std::optional<at::Tensor>& cu_seqlens_vt_descale,
    const std::optional<at::Tensor>& q_descale,
    const std::optional<at::Tensor>& k_descale,
    const std::optional<at::Tensor>& v_descale,
    const std::optional<at::Tensor>& vt_descale,
    const std::optional<at::Tensor>& cu_seqlens_q_block_descale,
    const std::optional<at::Tensor>& cu_seqlens_kv_block_descale) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(dprops->major >= 8, "HSTU only supports Ampere GPUs or newer.");

  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 ||
          q_type == at::ScalarType::Float8_e4m3fn,
      "HSTU only supports fp16, bf16, and fp8_e4m3 data type");
  if (dprops->major < 9) {
    TORCH_CHECK(
        q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
        "HSTU on Ampere/Ada cards only supports fp16 and bf16 data type");
  }
  TORCH_CHECK(
      k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(
      v.scalar_type() == q_type, "query and value must have the same dtype");
  TORCH_CHECK(
      cu_seqlens_q.dtype() == at::kInt, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(
      cu_seqlens_k.dtype() == at::kInt, "cu_seqlens_k must have dtype int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(cu_seqlens_k);
  TORCH_CHECK(
      q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const int batch_size = cu_seqlens_q.numel() - 1;
  const int total_q = q.size(0);
  const int num_heads = q.size(1);
  const int head_size = q.size(2);
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);

  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      num_heads == num_heads_k,
      "Number of heads in key/value and query must be equal");
  if (q_type == at::ScalarType::Float8_e4m3fn) {
    TORCH_CHECK(
        head_size == 64 || head_size == 128 || head_size == 256,
        "For fp8, HSTU forward only supports head dimension 64, 128, or 256");
  } else {
    TORCH_CHECK(
        head_size == 32 || head_size == 64 || head_size == 128 ||
            head_size == 256,
        "HSTU forward only supports head dimension 32, 64, 128, or 256");
  }

  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  if (num_contexts.has_value()) {
    TORCH_CHECK(
        num_contexts.value().dtype() == at::kInt,
        "num_contexts must have dtype int32");
    CHECK_DEVICE(num_contexts.value());
    CHECK_CONTIGUOUS(num_contexts.value());
    CHECK_SHAPE(num_contexts.value(), batch_size);
  }
  if (num_targets.has_value()) {
    TORCH_CHECK(
        num_targets.value().dtype() == at::kInt,
        "num_targets must have dtype int32");
    CHECK_DEVICE(num_targets.value());
    CHECK_CONTIGUOUS(num_targets.value());
    CHECK_SHAPE(num_targets.value(), batch_size);
  }

  auto out_type =
      q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::Half : q_type;
  at::Tensor out = torch::empty_like(q, out_type);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int seqlen_q_rounded = round_multiple(
      max_seqlen_q, sizeof(cutlass::uint128_t) / sizeof(q.dtype()));
  const int seqlen_k_rounded = round_multiple(
      max_seqlen_k, sizeof(cutlass::uint128_t) / sizeof(q.dtype()));

  // Otherwise the kernel will be launched from cuda:0 device
  at::cuda::CUDAGuard device_guard{q.get_device()};

  auto opts = q.options();
  bool has_rab = rab.has_value();
  int num_heads_rab = num_heads;
  if (has_rab) {
    num_heads_rab = rab.value().size(1);
    CHECK_DEVICE(rab.value());
    TORCH_CHECK(
        rab.value().stride(-1) == 1,
        "Input tensor must have contiguous last dimension");
    TORCH_CHECK(
        num_heads == num_heads_rab || num_heads_rab == 1,
        "Number of heads in rab must be 1 or equal to number of heads in query");
    CHECK_SHAPE(
        rab.value(), batch_size, num_heads_rab, max_seqlen_k, max_seqlen_k);
    if (seqlen_k_rounded != max_seqlen_k) {
      rab = torch::nn::functional::pad(
          rab.value(),
          torch::nn::functional::PadFuncOptions(
              {0, seqlen_k_rounded - max_seqlen_k}));
    }
  }
  bool is_fp8 = q_type == at::ScalarType::Float8_e4m3fn;
  if (is_fp8) {
    TORCH_CHECK(quant_mode >= 0 && quant_mode <= 5, "quant_mode must be 0, 1, 2, 3, 4, or 5 when dtype is float8_e4m3fn");
    TORCH_CHECK(q_descale.has_value() && k_descale.has_value(),
                "q_descale and k_descale must be provided when dtype is float8_e4m3fn");
    CHECK_DEVICE(q_descale.value());
    CHECK_DEVICE(k_descale.value());
    if (quant_mode == 1) {
      TORCH_CHECK(vt.has_value() && vt_descale.has_value() && cu_seqlens_vt_descale.has_value(),
                  "vt, vt_descale and cu_seqlens_vt_descale must be provided when dtype is float8_e4m3fn and quant_mode is 1");
      CHECK_DEVICE(vt.value());
      CHECK_DEVICE(vt_descale.value());
      CHECK_DEVICE(cu_seqlens_vt_descale.value());
      CHECK_SHAPE(vt.value(), total_k, num_heads_k, head_size);
      // Add 128 to the total_q and total_k to avoid out of bounds access
      CHECK_SHAPE(q_descale.value(), num_heads, total_q + 128);
      CHECK_SHAPE(k_descale.value(), num_heads, total_k + 128);
      CHECK_SHAPE(cu_seqlens_vt_descale.value(), batch_size + 1);
    } else if (quant_mode == 2) {
      TORCH_CHECK(v_descale.has_value() && cu_seqlens_q_block_descale.has_value() && cu_seqlens_kv_block_descale.has_value(),
                  "v_descale, cu_seqlens_q_block_descale and cu_seqlens_kv_block_descale must be provided when dtype is float8_e4m3fn and quant_mode is 2");
      CHECK_DEVICE(v_descale.value());
      CHECK_DEVICE(cu_seqlens_q_block_descale.value());
      CHECK_DEVICE(cu_seqlens_kv_block_descale.value());
      CHECK_SHAPE(cu_seqlens_q_block_descale.value(), batch_size + 1);
      CHECK_SHAPE(cu_seqlens_kv_block_descale.value(), batch_size + 1);
    } else if (quant_mode == 3) {
      TORCH_CHECK(v_descale.has_value(), "v_descale must be provided when dtype is float8_e4m3fn and quant_mode is 3");
      CHECK_DEVICE(v_descale.value());
      CHECK_SHAPE(q_descale.value(), batch_size, num_heads);
      CHECK_SHAPE(k_descale.value(), batch_size, num_heads_k);
      CHECK_SHAPE(v_descale.value(), batch_size, num_heads_k);
    } else if (quant_mode == 4) {
      TORCH_CHECK(v_descale.has_value(), "v_descale must be provided when dtype is float8_e4m3fn and quant_mode is 4");
      CHECK_DEVICE(v_descale.value());
      CHECK_SHAPE(q_descale.value(), batch_size);
      CHECK_SHAPE(k_descale.value(), batch_size);
      CHECK_SHAPE(v_descale.value(), batch_size);
    } else if (quant_mode == 5) {
      TORCH_CHECK(v_descale.has_value(), "v_descale must be provided when dtype is float8_e4m3fn and quant_mode is 5");
      CHECK_DEVICE(v_descale.value());
      CHECK_SHAPE(q_descale.value(), 1);
      CHECK_SHAPE(k_descale.value(), 1);
      CHECK_SHAPE(v_descale.value(), 1);
    }
  }

  Hstu_fwd_params params;
  set_params_fprop(
      params,
      batch_size,
      max_seqlen_q,
      max_seqlen_k,
      target_group_size,
      seqlen_q_rounded,
      seqlen_k_rounded,
      num_heads,
      num_heads_k,
      num_heads_rab,
      head_size,
      alpha,
      quant_mode,
      q,
      k,
      v,
      is_fp8 ? q_descale.value() : torch::Tensor(),
      is_fp8 ? k_descale.value() : torch::Tensor(),
      is_fp8 ? v_descale.value() : torch::Tensor(),
      has_rab ? rab.value() : torch::Tensor(),
      out,
      num_contexts.has_value() ? num_contexts.value().data_ptr() : nullptr,
      cu_seqlens_q.data_ptr(),
      cu_seqlens_k.data_ptr(),
      cu_seqlens_vt_descale.has_value() ? cu_seqlens_vt_descale.value().data_ptr() : nullptr,
      cu_seqlens_q_block_descale.has_value() ? cu_seqlens_q_block_descale.value().data_ptr() : nullptr,
      cu_seqlens_kv_block_descale.has_value() ? cu_seqlens_kv_block_descale.value().data_ptr() : nullptr,
      num_targets.has_value() ? num_targets.value().data_ptr() : nullptr,
      has_rab,
      func.has_value() ? func.value() : torch::Tensor(),
      window_size_left,
      window_size_right,
      vt.has_value() ? vt.value() : torch::Tensor(),
      vt_descale.has_value() ? vt_descale.value() : torch::Tensor());

  auto tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));
  params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

  params.total_q = total_q;
  params.total_k = total_k;

  if (total_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_hstu_fwd_hopper(params, stream);
  } else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output
    // to 0.
    out.zero_();
  }

  return {out, has_rab ? rab.value() : torch::Tensor()};
}

template <typename Dtype, int Headdim>
void run_hstu_bwd_mask(Hstu_bwd_params& params, cudaStream_t stream) {
  RAB_DRAB_SWITCH(params.has_rab, params.has_drab, Has_rab, Has_drab, [&] {
#ifndef HSTU_DISABLE_ARBITRARY
    if (params.is_arbitrary_mask) {
      run_hstu_bwd_<
          90,
          Dtype,
          Headdim,
          Has_rab,
          Has_drab,
          false,
          false,
          false,
          false,
          true,
          HSTU_ARBITRARY_NFUNC>(params, stream);
      return;
    }
#endif
#ifndef HSTU_DISABLE_LOCAL
    if (params.is_local) {
      run_hstu_bwd_<
          90,
          Dtype,
          Headdim,
          Has_rab,
          Has_drab,
          true,
          false,
          false,
          false,
          false,
          0>(params, stream);
      return;
    }
#endif
    if (!params.is_causal) {
      run_hstu_bwd_<
          90,
          Dtype,
          Headdim,
          Has_rab,
          Has_drab,
          false,
          false,
          false,
          false,
          false,
          0>(params, stream);
      return;
    } else {
#ifndef HSTU_DISABLE_CAUSAL
      CONTEXT_SWITCH(params.is_context, Is_context, [&] {
        TARGET_SWITCH(params.is_target, Is_target, [&] {
          run_hstu_bwd_<
              90,
              Dtype,
              Headdim,
              Has_rab,
              Has_drab,
              false,
              true,
              Is_context,
              Is_target,
              false,
              0>(params, stream);
        });
      });
#endif
    }
  });
}

void run_hstu_bwd_hopper(Hstu_bwd_params& params, cudaStream_t stream) {
#ifndef HSTU_DISABLE_BACKWARD
  if (params.is_bf16) {
#ifndef HSTU_DISABLE_BF16
#ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) {
      run_hstu_bwd_mask<cutlass::bfloat16_t, 32>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) {
      run_hstu_bwd_mask<cutlass::bfloat16_t, 64>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) {
      run_hstu_bwd_mask<cutlass::bfloat16_t, 128>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) {
      run_hstu_bwd_mask<cutlass::bfloat16_t, 256>(params, stream);
    }
#endif
#else
    TORCH_CHECK(false, "This flash attention build does not support BF16.");
#endif
  } else if (params.is_e4m3) {
#ifndef HSTU_DISABLE_FP8
#ifndef HSTU_USE_E5M2_BWD
#ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) {
      std::cerr << "Not support dim = 32 and dtype = float_e4m3_t for now." << std::endl;
    }
#endif
#ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) {
      run_hstu_bwd_mask<cutlass::float_e4m3_t, 64>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) {
      run_hstu_bwd_mask<cutlass::float_e4m3_t, 128>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) {
      run_hstu_bwd_mask<cutlass::float_e4m3_t, 256>(params, stream);
    }
#endif
#else
    TORCH_CHECK(false, "This flash attention build does not support e4m3 bwd.");
#endif
#else
    TORCH_CHECK(false, "This flash attention build does not support FP8.");
#endif
  } else if (params.is_e5m2) {
#ifndef HSTU_DISABLE_FP8
#ifdef HSTU_USE_E5M2_BWD
#ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) {
      std::cerr << "Not support dim = 32 and dtype = float_e5m2_t for now." << std::endl;
    }
#endif
#ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) {
      run_hstu_bwd_mask<cutlass::float_e5m2_t, 64>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) {
      run_hstu_bwd_mask<cutlass::float_e5m2_t, 128>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) {
      run_hstu_bwd_mask<cutlass::float_e5m2_t, 256>(params, stream);
    }
#endif
#else
    TORCH_CHECK(false, "This flash attention build does not support e5m2 bwd.");
#endif
#else
    TORCH_CHECK(false, "This flash attention build does not support FP8.");
#endif
  } else {
#ifndef HSTU_DISABLE_FP16
#ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) {
      run_hstu_bwd_mask<cutlass::half_t, 32>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) {
      run_hstu_bwd_mask<cutlass::half_t, 64>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) {
      run_hstu_bwd_mask<cutlass::half_t, 128>(params, stream);
    }
#endif
#ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) {
      run_hstu_bwd_mask<cutlass::half_t, 256>(params, stream);
    }
#endif
#else
    TORCH_CHECK(false, "This flash attention build does not support FP16.");
#endif
  }
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_varlen_bwd_90(
    const at::Tensor&
        dout, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const std::optional<at::Tensor>&
        dout_t, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor&
        q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const std::optional<at::Tensor>&
        q_t, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor&
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const std::optional<at::Tensor>&
        k_t, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor&
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q, // b+1
    const at::Tensor& cu_seqlens_k, // b+1
    const int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const std::optional<at::Tensor>& dq_,
    const std::optional<at::Tensor>& dk_,
    const std::optional<at::Tensor>& dv_,
    const std::optional<at::Tensor>& num_contexts, // b
    const std::optional<at::Tensor>& num_targets, // b
    const int64_t target_group_size,
    int64_t window_size_left,
    int64_t window_size_right,
    const double alpha,
    const int64_t quant_mode,
    const std::optional<at::Tensor>& rab,
    const bool has_drab,
    const std::optional<at::Tensor>& func,
    const std::optional<at::Tensor>& q_descale,
    const std::optional<at::Tensor>& qt_descale,
    const std::optional<at::Tensor>& k_descale,
    const std::optional<at::Tensor>& kt_descale,
    const std::optional<at::Tensor>& v_descale,
    const std::optional<at::Tensor>& do_descale,
    const std::optional<at::Tensor>& dot_descale,
    const std::optional<at::Tensor>& cu_seqlens_qt_descale,
    const std::optional<at::Tensor>& cu_seqlens_kt_descale,
    const std::optional<at::Tensor>& cu_seqlens_q_block_descale,
    const std::optional<at::Tensor>& cu_seqlens_kv_block_descale,
    const bool deterministic) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(dprops->major >= 8, "HSTU only supports Ampere GPUs or newer.");
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto q_dtype = q.dtype();
  TORCH_CHECK(
      q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16 ||
      q_dtype == at::ScalarType::Float8_e4m3fn || q_dtype == at::ScalarType::Float8_e5m2,
      "HSTU bwd only support fp16, bf16, fp8_e4m3, and fp8_e5m2 data type");
  if (dprops->major < 9) {
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "HSTU on Ampere/Ada cards only supports fp16 and bf16 data type");
  }
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(
      dout.dtype() == q_dtype, "query and dout must have the same dtype");
  TORCH_CHECK(
      cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(
      cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(dout);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(cu_seqlens_k);
  TORCH_CHECK(
      q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const int batch_size = cu_seqlens_q.numel() - 1;
  const int total_q = q.size(0);
  const int num_heads = q.size(1);
  const int head_size = q.size(2);
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);

  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size);
  CHECK_SHAPE(dout, total_q, num_heads, head_size);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      head_size == 32 || head_size == 64 || head_size == 128 ||
          head_size == 256,
      "head_size should be 32, 64, 128, or 256");
  TORCH_CHECK(
      num_heads == num_heads_k,
      "Number of heads in key/value and query must be equal");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  // This should match the kernel configs
  const int kBlockM = head_size <= 64 ? 128 : 64;
  const int kBlockN = head_size <= 128 ? 128 : 64;
  const int seqlen_q_rounded = round_multiple(
      max_seqlen_q, sizeof(cutlass::uint128_t) / sizeof(q_dtype));
  const int seqlen_k_rounded = round_multiple(
      max_seqlen_k, sizeof(cutlass::uint128_t) / sizeof(q_dtype));
  int const total_q_padded_rounded =
      round_multiple(total_q + batch_size * kBlockN, kBlockN);

  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  if (num_contexts.has_value()) {
    TORCH_CHECK(
        num_contexts.value().dtype() == torch::kInt32,
        "num_contexts must have dtype int32");
    CHECK_DEVICE(num_contexts.value());
    CHECK_CONTIGUOUS(num_contexts.value());
    CHECK_SHAPE(num_contexts.value(), batch_size);
  }
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  if (num_targets.has_value()) {
    TORCH_CHECK(
        num_targets.value().dtype() == torch::kInt32,
        "num_targets must have dtype int32");
    CHECK_DEVICE(num_targets.value());
    CHECK_CONTIGUOUS(num_targets.value());
    CHECK_SHAPE(num_targets.value(), batch_size);
  }

  bool has_rab = rab.has_value();
  int num_heads_rab = num_heads;
  if (has_rab) {
    num_heads_rab = rab.value().size(1);
    CHECK_DEVICE(rab.value());
    TORCH_CHECK(
        rab.value().stride(-1) == 1,
        "Input tensor must have contiguous last dimension");
    TORCH_CHECK(
        num_heads == num_heads_rab || num_heads_rab == 1,
        "Number of heads in rab must be 1 or equal to number of heads in query");
    CHECK_SHAPE(
        rab.value(), batch_size, num_heads_rab, max_seqlen_k, seqlen_k_rounded);
  }

  at::Tensor dq = dq_.has_value() ? dq_.value() : torch::empty_like(q);
  at::Tensor dk = dk_.has_value() ? dk_.value() : torch::empty_like(k);
  at::Tensor dv = dv_.has_value() ? dv_.value() : torch::empty_like(v);

  if (q_dtype == at::ScalarType::Float8_e4m3fn) {
    dq = dq.to(torch::kFloat16);
    dk = dk.to(torch::kFloat16);
    dv = dv.to(torch::kFloat16);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  at::cuda::CUDAGuard device_guard{q.get_device()};

  auto opts = q.options();
  at::Tensor dq_accum;
  dq_accum = torch::empty(
      {num_heads, total_q_padded_rounded, head_size}, opts.dtype(at::kFloat));
  dq_accum.zero_();

  at::Tensor drab;
  TORCH_CHECK(
      !(!has_rab && has_drab), "has_rab must be True when has_drab=True");
  if (has_drab) {
    drab = torch::zeros_like(rab.value());
  } else {
    drab = torch::empty({seqlen_k_rounded}, opts);
  }

  at::Tensor dot_value = dout_t.has_value() ? dout_t.value() : torch::Tensor();
  at::Tensor qt_value = q_t.has_value() ? q_t.value() : torch::Tensor();
  at::Tensor kt_value = k_t.has_value() ? k_t.value() : torch::Tensor();
  if (q_dtype == at::ScalarType::Float8_e4m3fn || q_dtype == at::ScalarType::Float8_e5m2) {
    TORCH_CHECK(q_descale.has_value() && k_descale.has_value() && v_descale.has_value() && do_descale.has_value(),
                "q_descale, k_descale, v_descale, and do_descale must be provided when dtype is float8");
    CHECK_DEVICE(q_descale.value());
    CHECK_DEVICE(k_descale.value());
    CHECK_DEVICE(v_descale.value());
    CHECK_DEVICE(do_descale.value());
    if (quant_mode == 1) {
      TORCH_CHECK(q_t.has_value() && k_t.has_value(),
                  "q_t and k_t must be provided when dtype is float8 and dout_t is provided");
      TORCH_CHECK(qt_descale.has_value() && kt_descale.has_value() && dot_descale.has_value(),
                  "qt_descale, kt_descale, and dot_descale must be provided when dtype is float8");
      TORCH_CHECK(cu_seqlens_qt_descale.has_value() && cu_seqlens_kt_descale.has_value(),
                  "cu_seqlens_qt_descale and cu_seqlens_kt_descale must be provided when dtype is float8");
      CHECK_DEVICE(qt_value);
      CHECK_DEVICE(kt_value);
      CHECK_DEVICE(dot_value);
      CHECK_DEVICE(qt_descale.value());
      CHECK_DEVICE(kt_descale.value());
      CHECK_DEVICE(dot_descale.value());
      CHECK_DEVICE(cu_seqlens_qt_descale.value());
      CHECK_DEVICE(cu_seqlens_kt_descale.value());
      CHECK_SHAPE(dot_value, total_q, num_heads, head_size);
      CHECK_SHAPE(qt_value, total_q, num_heads, head_size);
      CHECK_SHAPE(kt_value, total_k, num_heads_k, head_size);
      CHECK_SHAPE(cu_seqlens_qt_descale.value(), batch_size + 1);
      CHECK_SHAPE(cu_seqlens_kt_descale.value(), batch_size + 1);
      // Add 128 to the total_q and total_k to avoid out of bounds access
      CHECK_SHAPE(q_descale.value(), num_heads, total_q + 128);
      CHECK_SHAPE(k_descale.value(), num_heads_k, total_k + 128);
      CHECK_SHAPE(v_descale.value(), num_heads_k, total_k + 128);
      CHECK_SHAPE(do_descale.value(), num_heads, total_q + 128);
    } else if (quant_mode == 2) {
      TORCH_CHECK(cu_seqlens_q_block_descale.has_value() && cu_seqlens_kv_block_descale.has_value(),
                  "cu_seqlens_q_block_descale and cu_seqlens_kv_block_descale must be provided when dtype is float8 and quant_mode is 2");
      CHECK_DEVICE(cu_seqlens_q_block_descale.value());
      CHECK_DEVICE(cu_seqlens_kv_block_descale.value());
      CHECK_SHAPE(cu_seqlens_q_block_descale.value(), batch_size + 1);
      CHECK_SHAPE(cu_seqlens_kv_block_descale.value(), batch_size + 1);
    } else if (quant_mode == 3) {
      CHECK_SHAPE(q_descale.value(), batch_size, num_heads);
      CHECK_SHAPE(k_descale.value(), batch_size, num_heads_k);
      CHECK_SHAPE(v_descale.value(), batch_size, num_heads_k);
    } else if (quant_mode == 4) {
      CHECK_SHAPE(q_descale.value(), batch_size);
      CHECK_SHAPE(k_descale.value(), batch_size);
      CHECK_SHAPE(v_descale.value(), batch_size);
    } else if (quant_mode == 5) {
      CHECK_SHAPE(q_descale.value(), 1);
      CHECK_SHAPE(k_descale.value(), 1);
      CHECK_SHAPE(v_descale.value(), 1);
    }
  }

  Hstu_bwd_params params;

  set_params_dgrad(
      params,
      batch_size,
      max_seqlen_q,
      max_seqlen_k,
      target_group_size,
      seqlen_q_rounded,
      seqlen_k_rounded,
      num_heads,
      num_heads_k,
      num_heads_rab,
      head_size,
      alpha,
      quant_mode,
      q,
      qt_value,
      k,
      kt_value,
      v,
      dout,
      dot_value,
      has_rab ? rab.value() : torch::Tensor(),
      do_descale.has_value() ? do_descale.value() : torch::Tensor(),
      q_descale.has_value() ? q_descale.value() : torch::Tensor(),
      k_descale.has_value() ? k_descale.value() : torch::Tensor(),
      v_descale.has_value() ? v_descale.value() : torch::Tensor(),
      dot_descale.has_value() ? dot_descale.value() : torch::Tensor(),
      qt_descale.has_value() ? qt_descale.value() : torch::Tensor(),
      kt_descale.has_value() ? kt_descale.value() : torch::Tensor(),
      dq,
      dk,
      dv,
      drab,
      num_contexts.has_value() ? num_contexts.value().data_ptr() : nullptr,
      cu_seqlens_q.data_ptr(),
      cu_seqlens_k.data_ptr(),
      cu_seqlens_qt_descale.has_value() ? cu_seqlens_qt_descale.value().data_ptr() : cu_seqlens_q.data_ptr(),
      cu_seqlens_kt_descale.has_value() ? cu_seqlens_kt_descale.value().data_ptr() : cu_seqlens_k.data_ptr(),
      cu_seqlens_q_block_descale.has_value() ? cu_seqlens_q_block_descale.value().data_ptr() : nullptr,
      cu_seqlens_kv_block_descale.has_value() ? cu_seqlens_kv_block_descale.value().data_ptr() : nullptr,
      num_targets.has_value() ? num_targets.value().data_ptr() : nullptr,
      dq_accum.data_ptr(),
      func.has_value() ? func.value() : torch::Tensor(),
      has_rab,
      has_drab,
      window_size_left,
      window_size_right,
      deterministic);
  params.total_q = total_q;
  params.total_k = total_k;

  // Will be zero'ed out in the backward preprocess kernel
  at::Tensor dq_semaphore = torch::empty(
      {(max_seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads},
      opts.dtype(torch::kInt32));
  dq_semaphore.zero_();
  params.dq_semaphore = dq_semaphore.data_ptr<int>();

  if (total_q > 0) {
    run_hstu_bwd_hopper(params, stream);
  } else {
    // If max_seqlen_q == 0, then we have an empty tensor. We need to set the
    // output to 0.
    dk.zero_();
    dv.zero_();
    drab.zero_();
  }

  if (has_drab && seqlen_k_rounded != max_seqlen_k) {
    drab = drab.index(
        {"...", torch::indexing::Slice(torch::indexing::None, max_seqlen_k)});
  }

  return {dq, dk, dv, drab};
}

} // namespace fbgemm_gpu::hstu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.hstu.hstu_ops_gpu");
  m.def(
      "hstu_varlen_fwd_90("
      "    Tensor q, "
      "    Tensor k, "
      "    Tensor v, "
      "    Tensor cu_seqlens_q, "
      "    Tensor cu_seqlens_k, "
      "    int max_seqlen_q, "
      "    int max_seqlen_k, "
      "    Tensor? num_contexts=None, "
      "    Tensor? num_targets=None, "
      "    int target_group_size=1, "
      "    int window_size_left=-1, "
      "    int window_size_right=-1, "
      "    float alpha=1.0, "
      "    Tensor? rab=None, "
      "    Tensor? func=None, "
      "    int quant_mode=-1, "
      "    Tensor? vt=None, "
      "    Tensor? cu_seqlens_vt_descale=None, "
      "    Tensor? q_descale=None, "
      "    Tensor? k_descale=None, "
      "    Tensor? v_descale=None, "
      "    Tensor? vt_descale=None, "
      "    Tensor? cu_seqlens_q_block_descale=None, "
      "    Tensor? cu_seqlens_kv_block_descale=None"
      ") -> (Tensor, Tensor)");
  m.def(
      "hstu_varlen_bwd_90("
      "    Tensor dout, "
      "    Tensor? dout_t, "
      "    Tensor q, "
      "    Tensor? q_t, "
      "    Tensor k, "
      "    Tensor? k_t, "
      "    Tensor v, "
      "    Tensor cu_seqlens_q, "
      "    Tensor cu_seqlens_k, "
      "    int max_seqlen_q, "
      "    int max_seqlen_k, "
      "    Tensor? dq_=None, "
      "    Tensor? dk_=None, "
      "    Tensor? dv_=None, "
      "    Tensor? num_contexts=None, "
      "    Tensor? num_targets=None, "
      "    int target_group_size=1, "
      "    int window_size_left=-1, "
      "    int window_size_right=-1, "
      "    float alpha=1.0, "
      "    int quant_mode=-1, "
      "    Tensor? rab=None, "
      "    bool has_drab=False, "
      "    Tensor? func=None, "
      "    Tensor? q_descale=None, "
      "    Tensor? qt_descale=None, "
      "    Tensor? k_descale=None, "
      "    Tensor? kt_descale=None, "
      "    Tensor? v_descale=None, "
      "    Tensor? do_descale=None, "
      "    Tensor? dot_descale=None, "
      "    Tensor? cu_seqlens_qt_descale=None, "
      "    Tensor? cu_seqlens_kt_descale=None, "
      "    Tensor? cu_seqlens_q_block_descale=None, "
      "    Tensor? cu_seqlens_kv_block_descale=None, "
      "    bool deterministic=False"
      ") -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl(
      "hstu_varlen_fwd_90",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(fbgemm_gpu::hstu::hstu_varlen_fwd_90)));
  m.impl(
      "hstu_varlen_bwd_90",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(fbgemm_gpu::hstu::hstu_varlen_bwd_90)));
}

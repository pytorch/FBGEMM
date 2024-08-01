#include <array>
#include <iostream>
#include <memory>

#include <cudnn.h>
#include <cudnn_frontend.h>

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "fmha.h"

#define MAX_MHA_DIM 3

#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define SOFTMAX_STATS_UID 5
#define D_Q_UID 101
#define D_K_UID 102
#define D_V_UID 103
#define D_O_UID 104
#define ATTN_SCALE_UID 201
#define DROPOUT_SEED_UID 202
#define DROPOUT_OFFSET_UID 203
#define SEQ_LEN_Q_UID 301
#define SEQ_LEN_KV_UID 302
#define SEQ_OFFSET_Q_UID 303
#define SEQ_OFFSET_KV_UID 304

namespace fe = cudnn_frontend;

namespace fbgemm_gpu::gen_ai::attention {

struct MHAParams {
  c10::DeviceIndex device_id;
  fe::DataType_t dtype;
  int64_t b;
  int64_t h;
  int64_t max_seq_len_q;
  int64_t max_seq_len_kv;
  int64_t d;
  float attention_scale;
  double dropout_p;
  bool is_causal;
  bool return_softmax_stats;
  std::array<int64_t, MAX_MHA_DIM> q_dim;
  std::array<int64_t, MAX_MHA_DIM> k_dim;
  std::array<int64_t, MAX_MHA_DIM> v_dim;
  std::array<int64_t, MAX_MHA_DIM> q_stride;
  std::array<int64_t, MAX_MHA_DIM> k_stride;
  std::array<int64_t, MAX_MHA_DIM> v_stride;

  MHAParams(){};

  MHAParams(
      int64_t b,
      int64_t h,
      int64_t max_seq_len_q,
      int64_t max_seq_len_kv,
      int64_t d,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      float attention_scale,
      double dropout_p,
      bool is_causal,
      bool return_softmax_stats)
      : device_id(at::cuda::current_device()),
        b(b),
        h(h),
        max_seq_len_q(max_seq_len_q),
        max_seq_len_kv(max_seq_len_kv),
        d(d),
        attention_scale(attention_scale),
        dropout_p(dropout_p),
        is_causal(is_causal),
        return_softmax_stats(return_softmax_stats) {
    if (q.scalar_type() == at::kBFloat16) {
      dtype = fe::DataType_t::BFLOAT16;
    } else if (q.scalar_type() == at::kHalf) {
      dtype = fe::DataType_t::HALF;
    } else {
      TORCH_CHECK(false, "Only fp16 and bf16 are supported!");
    }
    TORCH_CHECK(q.dim() == MAX_MHA_DIM);
    TORCH_CHECK(q.strides().size() == MAX_MHA_DIM);
    TORCH_CHECK(k.sizes().size() == MAX_MHA_DIM);
    TORCH_CHECK(k.strides().size() == MAX_MHA_DIM);
    TORCH_CHECK(v.sizes().size() == MAX_MHA_DIM);
    TORCH_CHECK(v.strides().size() == MAX_MHA_DIM);
    std::copy(q.sizes().begin(), q.sizes().end(), q_dim.begin());
    std::copy(q.strides().begin(), q.strides().end(), q_stride.begin());
    std::copy(k.sizes().begin(), k.sizes().end(), k_dim.begin());
    std::copy(k.strides().begin(), k.strides().end(), k_stride.begin());
    std::copy(v.sizes().begin(), v.sizes().end(), v_dim.begin());
    std::copy(v.strides().begin(), v.strides().end(), v_stride.begin());
  }
};

struct MHACacheKeyWrapper : at::native::ParamsWrapper<MHAParams> {
  MHACacheKeyWrapper(
      int64_t b,
      int64_t h,
      int64_t max_seq_len_q,
      int64_t max_seq_len_kv,
      int64_t d,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      float attention_scale,
      double dropout_p,
      bool is_causal,
      bool return_softmax_stats) {
    this->pod = MHAParams(
        b,
        h,
        max_seq_len_q,
        max_seq_len_kv,
        d,
        q,
        k,
        v,
        attention_scale,
        dropout_p,
        is_causal,
        return_softmax_stats);
  }
};

struct MHAFwdGraph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> q;
  std::shared_ptr<fe::graph::Tensor_attributes> k;
  std::shared_ptr<fe::graph::Tensor_attributes> v;
  std::shared_ptr<fe::graph::Tensor_attributes> attention_scale;
  std::shared_ptr<fe::graph::Tensor_attributes> seq_len_q;
  std::shared_ptr<fe::graph::Tensor_attributes> seq_len_kv;
  std::shared_ptr<fe::graph::Tensor_attributes> seq_offset_q;
  std::shared_ptr<fe::graph::Tensor_attributes> seq_offset_kv;
  std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed;
  std::shared_ptr<fe::graph::Tensor_attributes> dropout_offset;
  std::shared_ptr<fe::graph::Tensor_attributes> o;
  std::shared_ptr<fe::graph::Tensor_attributes> softmax_stats;
};

struct MHABwdGraph {
  std::shared_ptr<fe::graph::Graph> graph;
  std::shared_ptr<fe::graph::Tensor_attributes> q;
  std::shared_ptr<fe::graph::Tensor_attributes> k;
  std::shared_ptr<fe::graph::Tensor_attributes> v;
  std::shared_ptr<fe::graph::Tensor_attributes> attention_scale;
  std::shared_ptr<fe::graph::Tensor_attributes> seq_len_q;
  std::shared_ptr<fe::graph::Tensor_attributes> seq_len_kv;
  std::shared_ptr<fe::graph::Tensor_attributes> seq_offset_q;
  std::shared_ptr<fe::graph::Tensor_attributes> seq_offset_kv;
  std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed;
  std::shared_ptr<fe::graph::Tensor_attributes> dropout_offset;
  std::shared_ptr<fe::graph::Tensor_attributes> o;
  std::shared_ptr<fe::graph::Tensor_attributes> softmax_stats;
  std::shared_ptr<fe::graph::Tensor_attributes> d_o;
  std::shared_ptr<fe::graph::Tensor_attributes> d_q;
  std::shared_ptr<fe::graph::Tensor_attributes> d_k;
  std::shared_ptr<fe::graph::Tensor_attributes> d_v;
};

template <typename KeyType, typename T>
struct MHAGraphCache {
  std::unordered_map<KeyType, T, at::native::ParamsWrapperHash<KeyType>>
      engine_cache;

  // no mutexes here as caches are now thread local.
  T* find(const KeyType& key) {
    auto it = engine_cache.find(key);
    if (it == engine_cache.end()) {
      return nullptr;
    }
    return &(it->second);
  }

  void update(const KeyType& key, T& results) {
    engine_cache.erase(key);
    engine_cache.emplace(key, std::move(results));
  }
};

// Use thread local caches as cuDNN Execution Plans are not guaranteed to
// be thread safe across all engines see Limitations in
// https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html
thread_local MHAGraphCache<MHACacheKeyWrapper, MHAFwdGraph> mha_fwd_graph_cache;
thread_local MHAGraphCache<MHACacheKeyWrapper, MHABwdGraph> mha_bwd_graph_cache;

void set_thd_dim_and_stride(
    std::shared_ptr<fe::graph::Tensor_attributes> tensor,
    const int64_t b,
    const int64_t s,
    const std::array<int64_t, 3>& thd_dim,
    const std::array<int64_t, 3>& thd_stride,
    std::shared_ptr<fe::graph::Tensor_attributes> ragged_offset) {
  const int64_t h = thd_dim[1];
  const int64_t d = thd_dim[2];
  const int64_t t_stride = thd_stride[0];
  const int64_t h_stride = thd_stride[1];
  const int64_t d_stride = thd_stride[2];
  TORCH_CHECK(
      ragged_offset->get_dim().size() == 4 &&
      ragged_offset->get_dim()[0] == b + 1);
  tensor->set_dim({b, h, s, d})
      .set_stride({t_stride * s, h_stride, t_stride, d_stride})
      .set_ragged_offset(ragged_offset);
}

template <typename MHAGraph>
MHAGraph build_mha_graph(
    /*Inputs*/
    const MHAParams& params,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& seq_len_q,
    const Tensor& seq_len_kv,
    const Tensor& seq_offset_q,
    const Tensor& seq_offset_kv) {
  // TORCH_CHECK(cudnnGetVersion() >= 9000);

  const fe::DataType_t dtype = params.dtype;

  const int64_t b = params.b;
  const int64_t h = params.h;
  const int64_t max_seq_len_q = params.max_seq_len_q;
  const int64_t max_seq_len_kv = params.max_seq_len_kv;
  const int64_t d = params.d;
  const float attention_scale = params.attention_scale;
  const double dropout_p = params.dropout_p;
  const bool is_causal = params.is_causal;
  const bool return_softmax_stats = params.return_softmax_stats;

  auto mha_graph = std::make_shared<fe::graph::Graph>();
  // We're baking in float accumulation and scale types
  // in theory the graph may support other types, but they
  // have not been tested
  mha_graph->set_io_data_type(dtype)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  auto fe_seq_offset_q =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("offset_q")
                            .set_uid(SEQ_OFFSET_Q_UID)
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));
  auto fe_seq_offset_kv =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("offset_kv")
                            .set_uid(SEQ_OFFSET_KV_UID)
                            .set_dim({b + 1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));

  auto fe_q = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_name("q").set_uid(Q_UID));
  set_thd_dim_and_stride(
      fe_q, b, max_seq_len_q, params.q_dim, params.q_stride, fe_seq_offset_q);

  auto fe_k = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_name("k").set_uid(K_UID));
  set_thd_dim_and_stride(
      fe_k, b, max_seq_len_kv, params.k_dim, params.k_stride, fe_seq_offset_kv);

  auto fe_v = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_name("v").set_uid(V_UID));
  set_thd_dim_and_stride(
      fe_v, b, max_seq_len_kv, params.v_dim, params.v_stride, fe_seq_offset_kv);

  auto fe_attention_scale =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("attention_scale")
                            .set_uid(ATTN_SCALE_UID)
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));

  auto fe_dropout_seed =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("dropout_seed")
                            .set_uid(DROPOUT_SEED_UID)
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT64));
  auto fe_dropout_offset =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("dropout_offset")
                            .set_uid(DROPOUT_OFFSET_UID)
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT64));

  auto fe_seq_len_q =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("seq_len_q")
                            .set_uid(SEQ_LEN_Q_UID)
                            .set_dim({b, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));

  auto fe_seq_len_kv =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("seq_len_kv")
                            .set_uid(SEQ_LEN_KV_UID)
                            .set_dim({b, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::INT32));

  return {
      .graph = std::move(mha_graph),
      .q = std::move(fe_q),
      .k = std::move(fe_k),
      .v = std::move(fe_v),
      .attention_scale = std::move(fe_attention_scale),
      .seq_len_q = std::move(fe_seq_len_q),
      .seq_len_kv = std::move(fe_seq_len_kv),
      .seq_offset_q = std::move(fe_seq_offset_q),
      .seq_offset_kv = std::move(fe_seq_offset_kv),
      .dropout_seed = std::move(fe_dropout_seed),
      .dropout_offset = std::move(fe_dropout_offset),
  };
}

MHAFwdGraph build_mha_fwd_graph(
    /*Inputs*/
    const MHAParams& params,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& seq_len_q,
    const Tensor& seq_len_kv,
    const Tensor& seq_offset_q,
    const Tensor& seq_offset_kv,
    /*Outputs*/
    Tensor& softmax_stats,
    Tensor& o,
    Tensor& dropout_seed,
    Tensor& dropout_offset,
    cudnnHandle_t& handle) {
  TORCH_CHECK(o.sizes() == q.sizes() && o.strides() == q.strides());

  MHAFwdGraph mha = build_mha_graph<MHAFwdGraph>(
      params,
      q,
      k,
      v,
      seq_offset_q,
      seq_offset_kv,
      seq_offset_q,
      seq_offset_kv);

  auto sdpa_options =
      fe::graph::SDPA_attributes()
          .set_name("cudnn_spda")
          .set_is_inference(params.return_softmax_stats == false)
          .set_causal_mask(params.is_causal)
          .set_attn_scale(mha.attention_scale)
          .set_padding_mask(true)
          .set_seq_len_q(mha.seq_len_q)
          .set_seq_len_kv(mha.seq_len_kv);

  if (params.dropout_p != 0.0f) {
    sdpa_options.set_dropout(
        params.dropout_p, mha.dropout_seed, mha.dropout_offset);
  }

  auto& mha_graph = mha.graph;
  auto [fe_o, fe_softmax_stats] =
      mha_graph->sdpa(mha.q, mha.k, mha.v, sdpa_options);

  const int64_t b = params.b;
  const int64_t h = params.h;
  const int64_t max_seq_len_q = params.max_seq_len_q;
  const int64_t d = params.d;

  fe_o->set_output(true).set_name("o").set_uid(O_UID);
  set_thd_dim_and_stride(
      fe_o, b, max_seq_len_q, params.q_dim, params.q_stride, mha.seq_offset_q);
  mha.o = std::move(fe_o);

  if (fe_softmax_stats) {
    fe_softmax_stats->set_output(true)
        .set_name("softmax_stats")
        .set_uid(SOFTMAX_STATS_UID)
        .set_dim({b, h, max_seq_len_q, 1})
        .set_stride({h * max_seq_len_q, max_seq_len_q, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT);
  }
  mha.softmax_stats = std::move(fe_softmax_stats);

  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));

  return mha;
}

MHABwdGraph build_mha_bwd_graph(
    /*Inputs*/
    const MHAParams& params,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& seq_len_q,
    const Tensor& seq_len_kv,
    const Tensor& seq_offset_q,
    const Tensor& seq_offset_kv,
    const Tensor& o,
    const Tensor& softmax_stats,
    const Tensor& dropout_seed,
    const Tensor& dropout_offset,
    const Tensor& d_o,
    /*Outputs*/
    Tensor& d_q,
    Tensor& d_k,
    Tensor& d_v,
    cudnnHandle_t& handle) {
  TORCH_CHECK(o.sizes() == q.sizes() && o.strides() == q.strides());
  TORCH_CHECK(d_o.sizes() == q.sizes() && d_o.strides() == q.strides());
  TORCH_CHECK(d_q.sizes() == q.sizes() && d_q.strides() == q.strides());
  TORCH_CHECK(d_k.sizes() == k.sizes() && d_k.strides() == k.strides());
  TORCH_CHECK(d_v.sizes() == v.sizes() && d_v.strides() == v.strides());

  MHABwdGraph mha = build_mha_graph<MHABwdGraph>(
      params,
      q,
      k,
      v,
      seq_offset_q,
      seq_offset_kv,
      seq_offset_q,
      seq_offset_kv);

  auto sdpa_options = fe::graph::SDPA_backward_attributes()
                          .set_name("cudnn_spda")
                          .set_causal_mask(params.is_causal)
                          .set_attn_scale(mha.attention_scale)
                          .set_padding_mask(true)
                          .set_seq_len_q(mha.seq_len_q)
                          .set_seq_len_kv(mha.seq_len_kv);

  if (params.dropout_p != 0.0f) {
    sdpa_options.set_dropout(
        params.dropout_p, mha.dropout_seed, mha.dropout_offset);
  }

  auto& mha_graph = mha.graph;

  const int64_t b = params.b;
  const int64_t h = params.h;
  const int64_t max_seq_len_q = params.max_seq_len_q;
  const int64_t max_seq_len_kv = params.max_seq_len_kv;
  const int64_t d = params.d;

  mha.o = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_name("o").set_uid(O_UID));
  set_thd_dim_and_stride(
      mha.o, b, max_seq_len_q, params.q_dim, params.q_stride, mha.seq_offset_q);

  mha.softmax_stats = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("softmax_stats")
          .set_uid(SOFTMAX_STATS_UID)
          .set_dim({b, h, max_seq_len_q, 1})
          .set_stride({h * max_seq_len_q, max_seq_len_q, 1, 1})
          .set_data_type(fe::DataType_t::FLOAT));

  mha.d_o = mha_graph->tensor(
      fe::graph::Tensor_attributes().set_name("d_o").set_uid(D_O_UID));
  set_thd_dim_and_stride(
      mha.d_o,
      b,
      max_seq_len_q,
      params.q_dim,
      params.q_stride,
      mha.seq_offset_q);

  auto [fe_d_q, fe_d_k, fe_d_v] = mha_graph->sdpa_backward(
      mha.q, mha.k, mha.v, mha.o, mha.d_o, mha.softmax_stats, sdpa_options);

  fe_d_q->set_output(true).set_name("d_q").set_uid(D_Q_UID);
  set_thd_dim_and_stride(
      fe_d_q,
      b,
      max_seq_len_q,
      params.q_dim,
      params.q_stride,
      mha.seq_offset_q);
  mha.d_q = std::move(fe_d_q);

  fe_d_k->set_output(true).set_name("d_k").set_uid(D_K_UID);
  set_thd_dim_and_stride(
      fe_d_k,
      b,
      max_seq_len_kv,
      params.k_dim,
      params.k_stride,
      mha.seq_offset_kv);
  mha.d_k = std::move(fe_d_k);

  fe_d_v->set_output(true).set_name("d_v").set_uid(D_V_UID);
  set_thd_dim_and_stride(
      fe_d_v,
      b,
      max_seq_len_kv,
      params.v_dim,
      params.v_stride,
      mha.seq_offset_kv);
  mha.d_v = std::move(fe_d_v);

  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));

  return mha;
}

void run_cudnn_sdpa_fprop(
    int64_t b,
    int64_t h,
    int64_t max_seq_len_q,
    int64_t max_seq_len_kv,
    int64_t d,
    float attention_scale,
    double dropout_p,
    bool is_causal,
    bool return_softmax_stats,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& seq_len_q,
    const Tensor& seq_len_kv,
    const Tensor& seq_offset_q,
    const Tensor& seq_offset_kv,
    Tensor& o,
    Tensor& softmax_stats,
    Tensor& dropout_seed,
    Tensor& dropout_offset) {
  cudnnHandle_t handle = at::native::getCudnnHandle();

  auto key = MHACacheKeyWrapper(
      b,
      h,
      max_seq_len_q,
      max_seq_len_kv,
      d,
      q,
      k,
      v,
      attention_scale,
      dropout_p,
      is_causal,
      return_softmax_stats);
  auto mha_ptr = mha_fwd_graph_cache.find(key);
  MHAFwdGraph mha;
  if (mha_ptr) {
    mha = *mha_ptr;
  } else {
    mha = build_mha_fwd_graph(
        key.pod,
        q,
        k,
        v,
        seq_len_q,
        seq_len_kv,
        seq_offset_q,
        seq_offset_kv,
        softmax_stats,
        o,
        dropout_seed,
        dropout_offset,
        handle);
  }
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
      variant_pack = {
          {mha.q, q.data_ptr()},
          {mha.k, k.data_ptr()},
          {mha.v, v.data_ptr()},
          {mha.attention_scale, &attention_scale},
          {mha.seq_len_q, seq_len_q.data_ptr()},
          {mha.seq_len_kv, seq_len_kv.data_ptr()},
          {mha.seq_offset_q, seq_offset_q.data_ptr()},
          {mha.seq_offset_kv, seq_offset_kv.data_ptr()},
          {mha.o, o.data_ptr()}};
  if (dropout_p != 0.0f) {
    variant_pack[mha.dropout_seed] = dropout_seed.data_ptr();
    variant_pack[mha.dropout_offset] = dropout_offset.data_ptr();
  }
  if (return_softmax_stats) {
    variant_pack[mha.softmax_stats] = softmax_stats.data_ptr();
  }
  auto workspace_size = mha.graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  // AT_CUDNN_FRONTEND_CHECK(mha.graph->validate());
  auto status = mha.graph->execute(handle, variant_pack, workspace_ptr.get());
  if (!status.is_good()) {
    std::cerr << "CuDNN Error: " << status.get_message() << std::endl;
  }
  TORCH_CHECK(status.is_good());
  if (!mha_ptr) {
    mha_fwd_graph_cache.update(key, mha);
  }
}

void run_cudnn_sdpa_bprop(
    int64_t b,
    int64_t h,
    int64_t max_seq_len_q,
    int64_t max_seq_len_kv,
    int64_t d,
    float attention_scale,
    double dropout_p,
    bool is_causal,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& seq_len_q,
    const Tensor& seq_len_kv,
    const Tensor& seq_offset_q,
    const Tensor& seq_offset_kv,
    const Tensor& o,
    const Tensor& softmax_stats,
    const Tensor& dO,
    Tensor& dQ,
    Tensor& dK,
    Tensor& dV,
    const Tensor& dropout_seed,
    const Tensor& dropout_offset) {
  cudnnHandle_t handle = at::native::getCudnnHandle();
  auto key = MHACacheKeyWrapper(
      b,
      h,
      max_seq_len_q,
      max_seq_len_kv,
      d,
      q,
      k,
      v,
      attention_scale,
      dropout_p,
      is_causal,
      /*return_softmax_stats=*/true);
  auto mha_ptr = mha_bwd_graph_cache.find(key);
  MHABwdGraph mha;
  if (mha_ptr) {
    mha = *mha_ptr;
  } else {
    mha = build_mha_bwd_graph(
        key.pod,
        q,
        k,
        v,
        seq_len_q,
        seq_len_kv,
        seq_offset_q,
        seq_offset_kv,
        o,
        softmax_stats,
        dropout_seed,
        dropout_offset,
        dO,
        dQ,
        dK,
        dV,
        handle);
  }
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
      variant_pack = {
          // inputs
          {mha.q, q.data_ptr()},
          {mha.k, k.data_ptr()},
          {mha.v, v.data_ptr()},
          {mha.attention_scale, &attention_scale},
          {mha.seq_len_q, seq_len_q.data_ptr()},
          {mha.seq_len_kv, seq_len_kv.data_ptr()},
          {mha.seq_offset_q, seq_offset_q.data_ptr()},
          {mha.seq_offset_kv, seq_offset_kv.data_ptr()},
          {mha.dropout_seed, dropout_seed.data_ptr()},
          {mha.dropout_offset, dropout_offset.data_ptr()},
          {mha.o, o.data_ptr()},
          {mha.softmax_stats, softmax_stats.data_ptr()},
          {mha.d_o, dO.data_ptr()},
          // Outputs
          {mha.d_q, dQ.data_ptr()},
          {mha.d_k, dK.data_ptr()},
          {mha.d_v, dV.data_ptr()},
      };
  if (dropout_p != 0.0f) {
    variant_pack[mha.dropout_seed] = dropout_seed.data_ptr();
    variant_pack[mha.dropout_offset] = dropout_offset.data_ptr();
  }
  auto workspace_size = mha.graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(!workspace_size || workspace_ptr.get());
  auto status = mha.graph->execute(handle, variant_pack, workspace_ptr.get());
  if (!status.is_good()) {
    std::cerr << "CuDNN Error: " << status.get_message() << std::endl;
  }
  TORCH_CHECK(status.is_good());
  mha_bwd_graph_cache.update(key, mha);
}

} // namespace fbgemm_gpu::gen_ai::attention

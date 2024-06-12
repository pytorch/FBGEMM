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

namespace fe = cudnn_frontend;

namespace fbgemm_gpu::gen_ai::attention {

using graph_and_tensors =
    std::tuple<std::shared_ptr<fe::graph::Graph>,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Q,
               std::shared_ptr<fe::graph::Tensor_attributes>, // K,
               std::shared_ptr<fe::graph::Tensor_attributes>, // V,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Attn_scale,
               // std::shared_ptr<fe::graph::Tensor_attributes>, // Bias,
               std::shared_ptr<fe::graph::Tensor_attributes>, // SEQ_LEN_Q,
               std::shared_ptr<fe::graph::Tensor_attributes>, // SEQ_LEN_KV,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Seed,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Offset,
               // std::shared_ptr<fe::graph::Tensor_attributes>, //
               // Dropout_mask, std::shared_ptr<fe::graph::Tensor_attributes>,
               // // Dropout_scale
               std::shared_ptr<fe::graph::Tensor_attributes>, // O
               std::shared_ptr<fe::graph::Tensor_attributes>  // Stats
               >;

using graph_and_tensors_backward =
    std::tuple<std::shared_ptr<fe::graph::Graph>,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Q,
               std::shared_ptr<fe::graph::Tensor_attributes>, // K,
               std::shared_ptr<fe::graph::Tensor_attributes>, // V,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Attn_scale
               std::shared_ptr<fe::graph::Tensor_attributes>, // SEQ_LEN_Q,
               std::shared_ptr<fe::graph::Tensor_attributes>, // SEQ_LEN_KV,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Seed,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Offset,
               std::shared_ptr<fe::graph::Tensor_attributes>, // O,
               std::shared_ptr<fe::graph::Tensor_attributes>, // dO,
               std::shared_ptr<fe::graph::Tensor_attributes>, // stats,
               std::shared_ptr<fe::graph::Tensor_attributes>, // dQ,
               std::shared_ptr<fe::graph::Tensor_attributes>, // dK,,
               std::shared_ptr<fe::graph::Tensor_attributes>  // dV,
               >;

#define MAX_MHA_DIM 4

struct MHAParams {
  c10::DeviceIndex device_id;
  fe::DataType_t dataType;
  std::array<int, MAX_MHA_DIM> q_dim;
  std::array<int, MAX_MHA_DIM> k_dim;
  std::array<int, MAX_MHA_DIM> v_dim;
  std::array<int, MAX_MHA_DIM> q_stride;
  std::array<int, MAX_MHA_DIM> k_stride;
  std::array<int, MAX_MHA_DIM> v_stride;
  int64_t b;
  int64_t h;
  int64_t s_q;
  int64_t s_kv;
  int64_t d;
  double dropout_probability;
  bool is_causal;
  bool return_softmaxstats;
};

void setMHAParams(MHAParams &params, int64_t b, int64_t h, int64_t s_q,
                  int64_t s_kv, int64_t d, const Tensor &q, const Tensor &k,
                  const Tensor &v, double dropout_probability, bool is_causal,
                  bool return_softmaxstats) {
  memset(&params, 0, sizeof(MHAParams));
  params.device_id = at::cuda::current_device();
  params.dataType = fe::DataType_t::HALF;
  if (q.scalar_type() == at::kBFloat16) {
    params.dataType = fe::DataType_t::BFLOAT16;
  }
  params.b = b;
  params.h = h;
  params.d = d;
  params.s_q = s_q;
  params.s_kv = s_kv;
  params.dropout_probability = dropout_probability;
  params.is_causal = is_causal;
  params.return_softmaxstats = return_softmaxstats;
  TORCH_INTERNAL_ASSERT(q.sizes().size() == MAX_MHA_DIM,
                        "Q tensor has unexpected number of dims, please report "
                        "a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(q.strides().size() == MAX_MHA_DIM,
                        "Q tensor has unexpected number of dims, please report "
                        "a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(k.sizes().size() == MAX_MHA_DIM,
                        "K tensor has unexpected number of dims, please report "
                        "a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(k.strides().size() == MAX_MHA_DIM,
                        "K tensor has unexpected number of dims, please report "
                        "a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(v.sizes().size() == MAX_MHA_DIM,
                        "V tensor has unexpected number of dims, please report "
                        "a bug to PyTorch.");
  TORCH_INTERNAL_ASSERT(v.strides().size() == MAX_MHA_DIM,
                        "V tensor has unexpected number of dims, please report "
                        "a bug to PyTorch.");
  std::copy(q.sizes().begin(), q.sizes().end(), params.q_dim.begin());
  std::copy(q.strides().begin(), q.strides().end(), params.q_stride.begin());
  std::copy(k.sizes().begin(), k.sizes().end(), params.k_dim.begin());
  std::copy(k.strides().begin(), k.strides().end(), params.k_stride.begin());
  std::copy(v.sizes().begin(), v.sizes().end(), params.v_dim.begin());
  std::copy(v.strides().begin(), v.strides().end(), params.v_stride.begin());
}

struct MHACacheKeyWrapper : at::native::ParamsWrapper<MHAParams> {
  MHACacheKeyWrapper(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                     const Tensor &q, const Tensor &k, const Tensor &v,
                     double dropout_probability, bool is_causal,
                     bool return_softmaxstats) {
    setMHAParams(this->pod, b, h, s_q, s_kv, d, q, k, v, dropout_probability,
                 is_causal, return_softmaxstats);
  }
};

template <typename T, typename KeyType> struct MHAGraphCache {
  std::unordered_map<KeyType, T, at::native::ParamsWrapperHash<KeyType>>
      engine_cache;

  // no mutexes here as caches are now thread local for v8, can also return a
  // pointer to the Execution Plan if we know it will not be invalidated by
  // another thread
  T *find(const KeyType &key) {
    auto it = engine_cache.find(key);
    if (it == engine_cache.end()) {
      return nullptr;
    }
    return &(it->second);
  }

  void update(const KeyType &key, T &results) {
    engine_cache.erase(key);
    engine_cache.emplace(key, std::move(results));
  }
};

// @eqy: use thread local caches as cuDNN Execution Plans are not guaranteed to
// be thread safe across all engines see Limitations in
// https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html
thread_local MHAGraphCache<graph_and_tensors, MHACacheKeyWrapper> mhagraphcache;
thread_local MHAGraphCache<graph_and_tensors_backward, MHACacheKeyWrapper>
    mhagraphbackwardcache;

auto build_graph_and_tensors(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
                             int64_t d, float scaling_factor,
                             bool return_softmaxstats, bool is_causal,
                             double dropout_probability, const Tensor &q,
                             const Tensor &k, const Tensor &v,
                             Tensor &softmaxstats, Tensor &o,
                             Tensor &dropoutseed, Tensor &dropoutoffset,
                             cudnnHandle_t &handle, MHAParams &params) {
  auto dtype = fe::DataType_t::HALF;
  if (q.scalar_type() == at::kBFloat16) {
    dtype = fe::DataType_t::BFLOAT16;
  }
  auto mha_graph = std::make_shared<fe::graph::Graph>();
  // We're baking in float accumulation and scale types
  // in theory the graph may support other types, but they
  // have not been tested
  mha_graph->set_io_data_type(dtype)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto Q = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("Q")
          .set_dim(
              std::vector<int64_t>(params.q_dim.begin(), params.q_dim.end()))
          .set_stride(std::vector<int64_t>(params.q_stride.begin(),
                                           params.q_stride.end())));
  auto K = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("K")
          .set_dim(
              std::vector<int64_t>(params.k_dim.begin(), params.k_dim.end()))
          .set_stride(std::vector<int64_t>(params.k_stride.begin(),
                                           params.k_stride.end())));
  auto V = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("V")
          .set_dim(
              std::vector<int64_t>(params.v_dim.begin(), params.v_dim.end()))
          .set_stride(std::vector<int64_t>(params.v_stride.begin(),
                                           params.v_stride.end())));
  auto attn_scale =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("Attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));

  // TODO(shikaili): support bias in the future in a follow-up PR
  // auto bias = mha_graph->tensor(fe::graph::Tensor_attributes()
  //                         .set_name("bias")
  //                         .set_dim({b, 1, s_q, s_kv})
  //                         .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
  auto seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("Seed")
                                    .set_dim({1, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));
  auto offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("Offset")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));

  auto seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("Seq_q")
                                     .set_dim({b, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(fe::DataType_t::INT32));

  auto seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("Seq_kv")
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));
  auto sdpa_forward_options =
      fe::graph::SDPA_attributes()
          .set_name("CUDNN_SDPA")
          .set_is_inference(return_softmaxstats == false)
          .set_causal_mask(is_causal)
          .set_attn_scale(attn_scale)
          .set_dropout(dropout_probability, seed, offset);
  // Optional bias in flash attention is only supported 8.9.3 onwards
  if (cudnnGetVersion() >= 8904) {
    // scaled_dot_product_flash_attention_options.set_alibi_mask(true);
  }
  if (cudnnGetVersion() >= 8903) {
    sdpa_forward_options.set_padding_mask(true)
        .set_seq_len_q(seq_q)
        .set_seq_len_kv(seq_kv);
  }

  auto [O, Stats] = mha_graph->sdpa(Q, K, V, sdpa_forward_options);
  O->set_output(true)
      .set_dim(std::vector<int64_t>(o.sizes().data(),
                                    o.sizes().data() + o.sizes().size()))
      .set_stride(std::vector<int64_t>(
          o.strides().data(), o.strides().data() + o.strides().size()));

  if (Stats) {
    Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
  }

  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));

  /*
  std::cerr << "Q uid=" << Q->get_uid() << std::endl;
  std::cerr << "K uid=" << K->get_uid() << std::endl;
  std::cerr << "V uid=" << V->get_uid() << std::endl;
  std::cerr << "seq_q uid=" << seq_q->get_uid() << std::endl;
  std::cerr << "seq_kv uid=" << seq_kv->get_uid() << std::endl;
  std::cerr << "attn_scale uid=" << attn_scale->get_uid() << std::endl;
  std::cerr << "seed uid=" << seed->get_uid() << std::endl;
  std::cerr << "offset uid=" << offset->get_uid() << std::endl;
  std::cerr << "O uid=" << O->get_uid() << std::endl;
  if (Stats) {
    std::cerr << "Stats uid=" << Stats->get_uid() << std::endl;
  }
  */

  return std::make_tuple(std::move(mha_graph), std::move(Q), std::move(K),
                         std::move(V), std::move(seq_q), std::move(seq_kv),
                         std::move(attn_scale), std::move(seed),
                         std::move(offset), std::move(O), std::move(Stats));
}

auto build_graph_and_tensors_backward(
    int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
    float scaling_factor, bool is_causal, float dropout_probability,
    const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &o,
    const Tensor &dO, const Tensor &softmaxstats, Tensor &dQ, Tensor &dK,
    Tensor &dV, const Tensor &dropoutseed, const Tensor &dropoutoffset,
    cudnnHandle_t &handle, MHAParams &params) {
  auto dtype = fe::DataType_t::HALF;
  if (q.scalar_type() == at::kBFloat16) {
    dtype = fe::DataType_t::BFLOAT16;
  }
  auto mha_graph = std::make_shared<fe::graph::Graph>();
  // We're baking in float accumulation and scale types
  // in theory the graph may support other types, but they
  // have not been tested
  mha_graph->set_io_data_type(dtype)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto Q = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("Q")
          .set_dim(std::vector<int64_t>(q.sizes().begin(), q.sizes().end()))
          .set_stride(
              std::vector<int64_t>(q.strides().begin(), q.strides().end())));
  auto K = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("K")
          .set_dim(std::vector<int64_t>(k.sizes().begin(), k.sizes().end()))
          .set_stride(
              std::vector<int64_t>(k.strides().begin(), k.strides().end())));
  auto V = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("V")
          .set_dim(std::vector<int64_t>(v.sizes().begin(), v.sizes().end()))
          .set_stride(
              std::vector<int64_t>(v.strides().begin(), v.strides().end())));
  auto attn_scale =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("Attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));

  auto Seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("Seed")
                                    .set_dim({1, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));
  auto Offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("Offset")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));
  auto seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("Seq_q")
                                     .set_dim({b, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(fe::DataType_t::INT32));
  auto seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("Seq_kv")
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));
  auto O = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("O")
          .set_dim(std::vector<int64_t>(o.sizes().begin(), o.sizes().end()))
          .set_stride(
              std::vector<int64_t>(o.strides().begin(), o.strides().end())));
  auto STATS = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("Stats")
          .set_dim(std::vector<int64_t>(softmaxstats.sizes().begin(),
                                        softmaxstats.sizes().end()))
          .set_stride(std::vector<int64_t>(softmaxstats.strides().begin(),
                                           softmaxstats.strides().end()))
          .set_data_type(fe::DataType_t::FLOAT));
  auto DO = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("DO")
          .set_dim(std::vector<int64_t>(dO.sizes().begin(), dO.sizes().end()))
          .set_stride(
              std::vector<int64_t>(dO.strides().begin(), dO.strides().end())));
  auto sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                   .set_name("CUDNN_SDPA_BACKWARD")
                                   .set_causal_mask(is_causal)
                                   .set_attn_scale(attn_scale);
  if (dropout_probability != 0.0f) {
    sdpa_backward_options.set_dropout(dropout_probability, Seed, Offset);
  }
  if (cudnnGetVersion() >= 8903) {
    sdpa_backward_options.set_padding_mask(true)
        .set_seq_len_q(seq_q)
        .set_seq_len_kv(seq_kv);
  }
  auto [DQ, DK, DV] =
      mha_graph->sdpa_backward(Q, K, V, O, DO, STATS, sdpa_backward_options);
  DQ->set_output(true)
      .set_dim(std::vector<int64_t>(dQ.sizes().begin(), dQ.sizes().end()))
      .set_stride(
          std::vector<int64_t>(dQ.strides().begin(), dQ.strides().end()));
  DK->set_output(true)
      .set_dim(std::vector<int64_t>(dK.sizes().begin(), dK.sizes().end()))
      .set_stride(
          std::vector<int64_t>(dK.strides().begin(), dK.strides().end()));
  DV->set_output(true)
      .set_dim(std::vector<int64_t>(dV.sizes().begin(), dV.sizes().end()))
      .set_stride(
          std::vector<int64_t>(dV.strides().begin(), dV.strides().end()));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->validate());
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_operation_graph(handle));
  AT_CUDNN_FRONTEND_CHECK(
      mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->check_support(handle));
  AT_CUDNN_FRONTEND_CHECK(mha_graph->build_plans(handle));
  return std::make_tuple(std::move(mha_graph), std::move(Q), std::move(K),
                         std::move(V), std::move(attn_scale), std::move(seq_q),
                         std::move(seq_kv), std::move(Seed), std::move(Offset),
                         std::move(O), std::move(DO), std::move(STATS),
                         std::move(DQ), std::move(DK), std::move(DV));
}

void run_cudnn_sdpa_fprop(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
                          int64_t d, float scaling_factor,
                          bool return_softmaxstats, bool is_causal,
                          double dropout_probability, const Tensor &q,
                          const Tensor &k, const Tensor &v, const Tensor &seq_q,
                          const Tensor &seq_kv, Tensor &softmaxstats, Tensor &o,
                          Tensor &dropoutseed, Tensor &dropoutoffset) {
  cudnnHandle_t handle = at::native::getCudnnHandle();
  o = at::empty_strided({b, h, s_q, d}, {s_q * h * d, d, h * d, 1},
                        q.options());
  if (return_softmaxstats) {
    // TODO(shikaili): verify that this is correct
    softmaxstats = at::empty({b, h, s_q}, q.options().dtype(at::kFloat));
  }

  auto key =
      MHACacheKeyWrapper(b, h, s_q, s_kv, d, q, k, v, dropout_probability,
                         is_causal, return_softmaxstats);
  auto graph_and_tensors_ptr = mhagraphcache.find(key);
  graph_and_tensors graph_and_tensors_values;
  if (graph_and_tensors_ptr) {
    graph_and_tensors_values = *graph_and_tensors_ptr;
  } else {
    graph_and_tensors_values = build_graph_and_tensors(
        b, h, s_q, s_kv, d, scaling_factor, return_softmaxstats, is_causal,
        dropout_probability, q, k, v, softmaxstats, o, dropoutseed,
        dropoutoffset, handle, key.pod);
  }
  auto [mha_graph, Q, K, V, attn_scale, seq_q_cudnn, seq_kv_cudnn, seed, offset,
        O, Stats] = graph_and_tensors_values;
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
      variant_pack = {{Q, q.data_ptr()},
                      {K, k.data_ptr()},
                      {V, v.data_ptr()},
                      {attn_scale, &scaling_factor},
                      //{bias, bias.data_ptr()},
                      {seq_q_cudnn, seq_q.data_ptr()},
                      {seq_kv_cudnn, seq_kv.data_ptr()},
                      {seed, dropoutseed.data_ptr()},
                      {offset, dropoutoffset.data_ptr()},
                      {O, o.data_ptr()}};
  if (return_softmaxstats) {
    variant_pack[Stats] = softmaxstats.data_ptr();
  }
  auto workspace_size = mha_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  std::cerr << "Launch!\n";
  auto status = mha_graph->execute(handle, variant_pack, workspace_ptr.get());
  if (!status.is_good()) {
    std::cerr << "CuDNN Error: " << status.get_message() << std::endl;
  }
  std::cerr << "Success!\n";
  TORCH_CHECK(status.is_good());
  mhagraphcache.update(key, graph_and_tensors_values);
}

void run_cudnn_sdpa_bprop(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
                          int64_t d, float scaling_factor, bool is_causal,
                          float dropout_probability, const Tensor &q,
                          const Tensor &k, const Tensor &v, const Tensor &seq_q,
                          const Tensor &seq_kv, const Tensor &o,
                          const Tensor &dO, const Tensor &softmaxstats,
                          Tensor &dQ, Tensor &dK, Tensor &dV,
                          const Tensor &dropoutseed,
                          const Tensor &dropoutoffset) {
  cudnnHandle_t handle = at::native::getCudnnHandle();
  auto key = MHACacheKeyWrapper(b, h, s_q, s_kv, d, q, k, v,
                                dropout_probability, is_causal, true);
  auto graph_and_tensors_backward_ptr = mhagraphbackwardcache.find(key);
  graph_and_tensors_backward graph_and_tensors_backward_values;
  if (graph_and_tensors_backward_ptr) {
    graph_and_tensors_backward_values = *graph_and_tensors_backward_ptr;
  } else {
    graph_and_tensors_backward_values = build_graph_and_tensors_backward(
        b, h, s_q, s_kv, d, scaling_factor, is_causal, dropout_probability, q,
        k, v, o, dO, softmaxstats, dQ, dK, dV, dropoutseed, dropoutoffset,
        handle, key.pod);
  }
  auto [mha_graph, Q, K, V, attn_scale, seq_q_cudnn, seq_kv_cudnn, Seed, Offset,
        O, Do, Stats, Dq, Dk, Dv] = graph_and_tensors_backward_values;
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
      variant_pack = {// inputs
                      {Q, q.data_ptr()},
                      {K, k.data_ptr()},
                      {V, v.data_ptr()},
                      {O, o.data_ptr()},
                      {Do, dO.data_ptr()},
                      {Stats, softmaxstats.data_ptr()},
                      // outputs
                      {Dq, dQ.data_ptr()},
                      {Dk, dK.data_ptr()},
                      {Dv, dV.data_ptr()},
                      // pass by value
                      {attn_scale, &scaling_factor},
                      {seq_q_cudnn, seq_q.data_ptr()},
                      {seq_kv_cudnn, seq_kv.data_ptr()}};
  if (dropout_probability != 0.0f) {
    variant_pack[Seed] = dropoutseed.data_ptr();
    variant_pack[Offset] = dropoutoffset.data_ptr();
  }
  auto workspace_size = mha_graph->get_workspace_size();
  auto workspace_ptr =
      c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);
  TORCH_CHECK(!workspace_size || workspace_ptr.get());
  auto status = mha_graph->execute(handle, variant_pack, workspace_ptr.get());
  if (!status.is_good()) {
    std::cerr << "CuDNN Error: " << status.get_message() << std::endl;
  }
  TORCH_CHECK(status.is_good());
  mhagraphbackwardcache.update(key, graph_and_tensors_backward_values);
}

} // namespace fbgemm
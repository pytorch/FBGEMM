/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstring>
#include <set>
#include <unordered_set>
#include <vector>

#include "ck_tile/host.hpp"
#include "fused_moe.hpp"

// different threshold for different dtype
template <typename DataType>
auto get_elimit() {
  double rtol = 1e-2;
  double atol = 1e-2;
  return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>() {
  double rtol = 1e-2;
  double atol = 1e-2;
  return ck_tile::make_tuple(rtol, atol);
}

// mfma_type, 0:32x32, 1:16x16
// TODO: padding?
template <typename T>
auto shuffle_moe_weight(
    const ck_tile::HostTensor<T>& t,
    std::string mfma_dtype,
    int mfma_type = 0) {
  assert(t.get_lengths().size() == 3);
  int b_ = t.get_lengths()[0];
  int n_ = t.get_lengths()[1];
  int k_ = t.get_lengths()[2];
  if ((mfma_dtype == "bf16" || mfma_dtype == "fp16") && mfma_type == 0) {
    ck_tile::HostTensor<T> t_view({b_, n_ / 32, 32, k_ / 16, 2, 8});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 1, 3, 4, 2, 5});
  } else if ((mfma_dtype == "bf16" || mfma_dtype == "fp16") && mfma_type == 1) {
    ck_tile::HostTensor<T> t_view({b_, n_ / 16, 16, k_ / 32, 4, 8});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 1, 3, 4, 2, 5});
  } else if ((mfma_dtype == "int8" || mfma_dtype == "fp8") && mfma_type == 0) {
    ck_tile::HostTensor<T> t_view({b_, n_ / 32, 32, k_ / 32, 2, 16});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 1, 3, 4, 2, 5});
  } else if ((mfma_dtype == "int8" || mfma_dtype == "fp8") && mfma_type == 1) {
    ck_tile::HostTensor<T> t_view({b_, n_ / 16, 16, k_ / 64, 4, 16});
    std::copy(t.begin(), t.end(), t_view.begin());
    return ck_tile::reference_permute(t_view, {0, 1, 3, 4, 2, 5});
  }
  return t;
}

template <typename IndexType>
void topid_unique_gen(
    std::vector<IndexType>& host_tensor,
    int tokens,
    int topk,
    int num_expert,
    int seed) {
  size_t total_size = topk * tokens;
  std::srand(seed);
  std::set<IndexType> unique_set;
  IndexType current_v;
  for (size_t i = 0; i < total_size; i++) {
    if (i % topk == 0) {
      unique_set.clear();
    }
    current_v = std::rand() % num_expert;
    while (unique_set.find(current_v) != unique_set.end()) {
      current_v = std::rand() % num_expert;
    }
    unique_set.insert(current_v);
    host_tensor[i] = current_v;
  }
}

auto create_args(int argc, char* argv[]) {
  ck_tile::ArgParser arg_parser;
  arg_parser.insert("t", "128", "num input tokens")
      .insert("e", "32", "num of experts")
      .insert("k", "5", "topk")
      .insert("h", "8192", "hidden_size of this model")
      .insert("i", "8192", "intermediate_size between 2 gemms of FFN")
      .insert("stride", "-1", "stride per row, if -1 then equal to hidden_size")
      .insert("bm", "32", "blocking factor for sorted tokens")
      .insert("tp", "8", "tensor parallel size")
      .insert("v", "1", "cpu validation or not")
      .insert("kname", "1", "print kernel name or not")
      .insert("prec_i", "bf16", "input precision")
      .insert("prec_w", "bf16", "weight precision")
      .insert("prec_o", "bf16", "output precision")
      .insert("prec_st", "auto", "token scale data type. auto will set to fp32")
      .insert(
          "prec_sw", "auto", "weight scale data type. auto will set to fp32")
      .insert(
          "prec_sq",
          "auto",
          "(dynamic) smooth quant data type. auto will set to fp32")
      .insert("prec_kw", "auto", "topk-weight data type. auto will set to fp32")
      .insert(
          "fquant",
          "0",
          "fused-quant, 0:no, 1:smooth-dynamic-quant, 2:dynamic-quant")
      .insert(
          "gate_only",
          "1",
          "w0(gate/up) style, 0:gate+up will double interm size, 1:only gate")
      .insert(
          "api",
          "0",
          "benchmark api set: 0:fused-moe(moe-gemm+moe-sorting), 1:moe-gemm")
      .insert(
          "balance",
          "0",
          "if set to 1, will try balance the expert in topk-ids(convenient for testing)")
      .insert(
          "init",
          "2",
          "init method. 0:random stepped float(fast). 1: random uniform, 2:rand normalized"
          "normalized(slow)")
      .insert("seed", "11939", "seed used to do random")
      .insert("warmup", "5", "cold iter")
      .insert("repeat", "20", "hot iter");

  bool result = arg_parser.parse(argc, argv);
  return std::make_tuple(result, arg_parser);
}

// I:input-type, W:weight-type, O:output-type, ST:toke-scale-tpye,
// SW:weight-scale-type, SQ:smooth-quant-type, KW:topk-weight-type
template <
    typename I,
    typename W,
    typename O,
    typename ST,
    typename SW,
    typename SQ,
    typename KW>
bool run(const ck_tile::ArgParser& arg_parser) {
  ck_tile::index_t tokens = arg_parser.get_int("t");
  ck_tile::index_t experts = arg_parser.get_int("e");
  ck_tile::index_t topk = arg_parser.get_int("k");
  ck_tile::index_t hidden_size = arg_parser.get_int("h");
  ck_tile::index_t intermediate_size = arg_parser.get_int("i");
  ck_tile::index_t stride = arg_parser.get_int("stride");
  ck_tile::index_t block_m = arg_parser.get_int("bm");
  if (stride < 0)
    stride = hidden_size;
  std::string prec_i = arg_parser.get_str("prec_i");
  std::string prec_w = arg_parser.get_str("prec_w");
  std::string prec_o = arg_parser.get_str("prec_o");
  std::string prec_st = arg_parser.get_str("prec_st");
  std::string prec_sw = arg_parser.get_str("prec_sw");
  std::string prec_sq = arg_parser.get_str("prec_sq");
  std::string prec_kw = arg_parser.get_str("prec_kw");
  prec_st = (prec_st == "auto") ? "fp32" : prec_st;
  prec_sw = (prec_sw == "auto") ? "fp32" : prec_sw;
  prec_sq = (prec_sq == "auto") ? "fp32" : prec_sq;
  prec_kw = (prec_kw == "auto") ? "fp32" : prec_kw;
  int kname = arg_parser.get_int("kname");
  int do_validation = arg_parser.get_int("v");
  int warmup = arg_parser.get_int("warmup");
  int repeat = arg_parser.get_int("repeat");
  int fused_quant = arg_parser.get_int("fquant");
  int gate_only = arg_parser.get_int("gate_only");
  int api = arg_parser.get_int("api");
  int balance = arg_parser.get_int("balance");
  int tp = arg_parser.get_int("tp");
  int init = arg_parser.get_int("init");
  uint32_t seed = arg_parser.get_uint32("seed");

  // w0 (Gate+Up or Gate only, N size)
  ck_tile::index_t shared_intermediate_size_0 =
      intermediate_size * (gate_only ? 1 : 2) / tp;
  // w1 (Down, N size)
  ck_tile::index_t shared_intermediate_size_1 = intermediate_size / tp;

  auto prec_str = [&]() {
    auto base_str = prec_i;
    if (prec_i != prec_w)
      base_str += "x" + prec_w;
    if (prec_i != prec_o)
      base_str += "=" + prec_o;
    if (fused_quant != 0) {
      base_str +=
          std::string("(") + prec_st + "|" + prec_sw + "|" + prec_sq + ")";
    }
    return base_str;
  }();
  auto api_str = [&]() {
    if (api == 0)
      return std::string("fmoe");
    else if (api == 1)
      return std::string("moeg");
    else if (api == 2)
      return std::string("moes");
    return std::string("");
  }();

  auto stride_str = [&]() {
    if (stride == hidden_size)
      return std::string("");
    else
      return std::string(", st:") + std::to_string(stride);
  }();

  std::cout << "[" << api_str << "|" << prec_str << "]" << " t:" << tokens
            << ", e:" << experts << ", k:" << topk << stride_str
            << ", hidden:" << hidden_size << ", interm:" << intermediate_size
            << ", tp:" << tp << ", shrd_interm:" << shared_intermediate_size_0
            << "|" << shared_intermediate_size_1 << ", go:" << gate_only
            << ", q:" << fused_quant << std::flush;

  using TypeConfig = FusedMoeGemmTypeConfig<I, W, O, ST, SW, SQ, KW>;
  using ADataType = typename TypeConfig::ADataType;
  using GDataType = typename TypeConfig::GDataType;
  using DDataType = typename TypeConfig::DDataType;
  using AccDataType = typename TypeConfig::AccDataType;
  using ODataType = typename TypeConfig::ODataType;
  using AScaleDataType = typename TypeConfig::AScaleDataType;
  using GScaleDataType = typename TypeConfig::GScaleDataType;
  using DScaleDataType = typename TypeConfig::DScaleDataType;
  using YSmoothScaleDataType = typename TypeConfig::YSmoothScaleDataType;
  using TopkWeightDataType = typename TypeConfig::TopkWeightDataType;
  using IndexDataType = typename TypeConfig::IndexDataType;

  // host verify
  ck_tile::HostTensor<ADataType> a_host({tokens, hidden_size}, {stride, 1});
  ck_tile::HostTensor<GDataType> g_host(
      {experts, shared_intermediate_size_0, hidden_size});
  ck_tile::HostTensor<DDataType> d_host(
      {experts, hidden_size, shared_intermediate_size_1});
  ck_tile::HostTensor<ODataType> o_host({tokens, hidden_size}, {stride, 1});
  ck_tile::HostTensor<AScaleDataType> sa_host({tokens});
  ck_tile::HostTensor<GScaleDataType> sg_host({shared_intermediate_size_0});
  ck_tile::HostTensor<DScaleDataType> sd_host({shared_intermediate_size_1});
  ck_tile::HostTensor<YSmoothScaleDataType> sy_host(
      {shared_intermediate_size_1}); // smooth-quant
  ck_tile::HostTensor<IndexDataType> topk_ids_host(
      {tokens, topk}); // to be sort
  ck_tile::HostTensor<TopkWeightDataType> topk_weight_host(
      {tokens, topk}); // to be sort

  int max_num_tokens_padded = topk * tokens + experts * block_m - topk;
  ck_tile::HostTensor<IndexDataType> sorted_token_ids_host(
      {max_num_tokens_padded});
  ck_tile::HostTensor<TopkWeightDataType> sorted_weight_host(
      {max_num_tokens_padded});
  ck_tile::HostTensor<IndexDataType> sorted_expert_ids_host(
      {(max_num_tokens_padded + block_m - 1) / block_m});
  ck_tile::HostTensor<IndexDataType> num_sorted_tiles_host({1});

  if (init == 0) {
    ck_tile::FillStepRange<ADataType>{-.5f, .5f, 0.01f}(a_host);
    ck_tile::FillStepRange<GDataType>{-.5f, .5f, 0.01f}(g_host);
    ck_tile::FillStepRange<DDataType, false>{.5f, -.5f, -0.01f}(d_host);
    ck_tile::FillStepRange<AScaleDataType>{0.f, 1.f, 0.01f}(sa_host);
    ck_tile::FillStepRange<GScaleDataType>{0.f, 1.f, 0.01f}(sg_host);
    ck_tile::FillStepRange<DScaleDataType>{0.f, 1.f, 0.01f}(sd_host);
    ck_tile::FillStepRange<YSmoothScaleDataType>{0.f, 1.f, 0.01f}(sy_host);
    ck_tile::FillStepRange<TopkWeightDataType>{
        -.5f, .5f, 0.01f}(topk_weight_host);
  } else if (init == 1) {
    ck_tile::FillUniformDistribution<ADataType>{-.5f, .5f, seed, true}(a_host);
    ck_tile::FillUniformDistribution<GDataType>{-.5f, .5f, seed, true}(g_host);
    ck_tile::FillUniformDistribution<DDataType>{-.5f, .5f, seed, true}(d_host);
    ck_tile::FillUniformDistribution<AScaleDataType>{
        -.5f, .5f, seed, true}(sa_host);
    ck_tile::FillUniformDistribution<GScaleDataType>{
        -.5f, .5f, seed, true}(sg_host);
    ck_tile::FillUniformDistribution<DScaleDataType>{
        -.5f, .5f, seed, true}(sd_host);
    ck_tile::FillUniformDistribution<YSmoothScaleDataType>{
        -.5f, .5f, seed, true}(sy_host);
    ck_tile::FillUniformDistribution<TopkWeightDataType>{
        -.5f, .5f, seed, true}(topk_weight_host);
  } else if (init == 2) {
    ck_tile::FillNormalDistribution<ADataType>{0.f, 1.f, seed, true}(a_host);
    ck_tile::FillNormalDistribution<GDataType>{0.f, 1.f, seed, true}(g_host);
    ck_tile::FillNormalDistribution<DDataType>{0.f, 1.f, seed, true}(d_host);
    ck_tile::FillNormalDistribution<AScaleDataType>{
        0.f, 1.f, seed, true}(sa_host);
    ck_tile::FillNormalDistribution<GScaleDataType>{
        0.f, 1.f, seed, true}(sg_host);
    ck_tile::FillNormalDistribution<DScaleDataType>{
        0.f, 1.f, seed, true}(sd_host);
    ck_tile::FillNormalDistribution<YSmoothScaleDataType>{
        0.f, 1.f, seed, true}(sy_host);
    ck_tile::FillNormalDistribution<TopkWeightDataType>{
        0.f, 1.f, seed, true}(topk_weight_host);
  }

  // permute weight
  ck_tile::HostTensor<GDataType> g_perm_host =
      shuffle_moe_weight(g_host, prec_w, 1);
  ck_tile::HostTensor<DDataType> d_perm_host =
      shuffle_moe_weight(d_host, prec_w, 1);

  // do moe sorting
  if (balance) {
    int e_cnt = 0;
    for (int i = 0; i < static_cast<int>(topk_ids_host.mData.size()); i++) {
      topk_ids_host.mData[i] = e_cnt;
      e_cnt++;
      if (e_cnt >= experts)
        e_cnt = 0;
    }
  } else {
    topid_unique_gen<IndexDataType>(
        topk_ids_host.mData, tokens, topk, experts, 11913);
  }

// leave it here for future debug purpose
#if 0
    a_host.loadtxt("../../ater/input_torch.txt");

    topk_ids_host.loadtxt("../../ater/topk_ids_torch.txt", "int");
    // topk_ids_host.savetxt("topk_ids_2.txt");
    topk_weight_host.loadtxt("../../ater/topk_weights_torch.txt", "float");
    std::cout << "------- @@@ " << __LINE__ << std::flush << std::endl;

    g_host.loadtxt("../../ater/w1_torch.txt", "float");
    std::cout << "------- @@@ " << __LINE__ << std::flush << std::endl;
    d_host.loadtxt("../../ater/w2_torch.txt", "float");
    std::cout << "------- @@@ " << __LINE__ << std::flush << std::endl;

    ck_tile::HostTensor<GDataType> g_perm_host = shuffle_moe_weight(g_host, prec_w, 1);
    std::cout << "------- @@@ " << __LINE__ << std::flush << std::endl;
    ck_tile::HostTensor<DDataType> d_perm_host = shuffle_moe_weight(d_host, prec_w, 1);
    std::cout << "------- @@@ " << __LINE__ << std::flush << std::endl;
#endif

#if 0
    std::cout << "sorted_token_ids_host:" << sorted_token_ids_host << std::endl;
    std::cout << "num_sorted_tiles_host:" << num_sorted_tiles_host << std::endl;
    std::cout << "sorted_expert_ids_host:" << sorted_expert_ids_host << std::endl;
    std::cout << "topk_weight_host:" << topk_weight_host << std::endl;
    std::cout << "sorted_weight_host:" << sorted_weight_host << std::endl;
#endif
  auto cal_tflops = [&](auto ms) {
    double flop_gemm_0 = 2 * static_cast<double>(tokens) * topk *
        shared_intermediate_size_0 * hidden_size;
    double flop_gemm_1 = 2 * static_cast<double>(tokens) * topk *
        shared_intermediate_size_1 * hidden_size;
    return (flop_gemm_0 + flop_gemm_1) / (static_cast<double>(ms) * 1e-3) /
        1e12;
  };

  // TODO: this method we use expert-by-expert view, just for reference
  auto cal_tbps = [&](auto ms) {
    double token_bytes = static_cast<double>(tokens) * topk / experts *
        hidden_size * sizeof(ADataType);
    double w0_bytes = static_cast<double>(shared_intermediate_size_0) *
        experts * hidden_size * sizeof(GDataType);
    double w1_bytes = static_cast<double>(shared_intermediate_size_1) *
        experts * hidden_size * sizeof(DDataType);
    double o_bytes = static_cast<double>(tokens) * topk / experts *
        hidden_size * sizeof(ODataType);
    double topk_weights_bytes =
        static_cast<double>(tokens) * topk * sizeof(TopkWeightDataType);
    // ignore index, they are too small

    return (token_bytes + w0_bytes + w1_bytes + o_bytes + topk_weights_bytes) /
        (static_cast<double>(ms) * 1e-3) / 1e12;
  };

  if (api == 0) {
    ck_tile::DeviceMem a_buf(a_host);
    ck_tile::DeviceMem g_perm_buf(g_perm_host);
    ck_tile::DeviceMem d_perm_buf(d_perm_host);
    ck_tile::DeviceMem sa_buf(sa_host);
    ck_tile::DeviceMem sg_buf(sg_host);
    ck_tile::DeviceMem sd_buf(sd_host);
    ck_tile::DeviceMem sy_buf(sy_host);
    ck_tile::DeviceMem o_buf(o_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem topk_ids_buf(topk_ids_host);
    ck_tile::DeviceMem topk_weight_buf(topk_weight_host);

    ck_tile::DeviceMem sorted_token_ids_buf(
        sorted_token_ids_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sorted_weight_buf(
        sorted_weight_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sorted_expert_ids_buf(
        sorted_expert_ids_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem num_sorted_tiles_buf(
        num_sorted_tiles_host.get_element_space_size_in_bytes());

    fused_moe_traits traits{
        prec_i,
        prec_w,
        prec_o,
        prec_st,
        prec_sw,
        prec_sq,
        prec_kw,
        block_m,
        gate_only,
        fused_quant};

    fused_moe_args args{
        a_buf.GetDeviceBuffer(),
        fused_quant != 0 ? sa_buf.GetDeviceBuffer() : nullptr,
        g_perm_buf.GetDeviceBuffer(),
        d_perm_buf.GetDeviceBuffer(),
        fused_quant != 0 ? sg_buf.GetDeviceBuffer() : nullptr,
        fused_quant != 0 ? sd_buf.GetDeviceBuffer() : nullptr,
        fused_quant == 1 ? sy_buf.GetDeviceBuffer() : nullptr,
        o_buf.GetDeviceBuffer(),
        topk_ids_buf.GetDeviceBuffer(),
        topk_weight_buf.GetDeviceBuffer(),
        sorted_token_ids_buf.GetDeviceBuffer(),
        sorted_weight_buf.GetDeviceBuffer(),
        sorted_expert_ids_buf.GetDeviceBuffer(),
        num_sorted_tiles_buf.GetDeviceBuffer(),
        block_m,
        hidden_size,
        shared_intermediate_size_0,
        tokens,
        experts,
        topk,
        stride};
    float ave_time = fused_moe(
        traits,
        args,
        ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    if (ave_time < 0) {
      std::cout << " not supported!" << std::endl << std::flush;
      return false;
    }

    // float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << ", " << ave_time * 1.E3 << " us, " << cal_tflops(ave_time)
              << " tflops, " << cal_tbps(ave_time) << " TB/s" << std::flush;
    bool pass = true;

    if (do_validation) {
      ck_tile::reference_moe_sorting<TopkWeightDataType, IndexDataType>(
          topk_ids_host,
          topk_weight_host,
          sorted_token_ids_host,
          sorted_weight_host,
          sorted_expert_ids_host,
          num_sorted_tiles_host.mData[0],
          experts,
          block_m);

      ck_tile::reference_fused_moe<AccDataType, ck_tile::element_wise::Gelu>(
          a_host,
          g_host,
          d_host,
          sa_host,
          sg_host,
          sd_host,
          sy_host,
          o_host,
          sorted_token_ids_host,
          sorted_weight_host,
          sorted_expert_ids_host,
          num_sorted_tiles_host,
          topk_ids_host,
          block_m,
          tokens,
          experts,
          hidden_size,
          shared_intermediate_size_0,
          topk,
          gate_only);

      auto o_dev = o_buf.ToHost<ODataType>();
      // o_dev.savetxt("gpu-out.txt", "float");
      auto [rtol, atol] = get_elimit<ADataType>();
      pass &= ck_tile::check_err(
          o_dev,
          o_host,
          std::string("OUT Error: Incorrect results!"),
          rtol,
          atol);
      std::cout << ", valid:" << (pass ? "y" : "n") << std::flush;
    }
    std::cout << std::flush << std::endl;
    return pass;
  } else if (api == 1) {
    ck_tile::reference_moe_sorting<TopkWeightDataType, IndexDataType>(
        topk_ids_host,
        topk_weight_host,
        sorted_token_ids_host,
        sorted_weight_host,
        sorted_expert_ids_host,
        num_sorted_tiles_host.mData[0],
        experts,
        block_m);

    // done, preparing GPU buffer
    ck_tile::DeviceMem a_buf(a_host);
    ck_tile::DeviceMem g_perm_buf(g_perm_host);
    ck_tile::DeviceMem d_perm_buf(d_perm_host);
    ck_tile::DeviceMem sa_buf(sa_host);
    ck_tile::DeviceMem sg_buf(sg_host);
    ck_tile::DeviceMem sd_buf(sd_host);
    ck_tile::DeviceMem sy_buf(sy_host);
    ck_tile::DeviceMem o_buf(o_host);

    // manually clear output buffer for atomic
    o_buf.SetZero();
    //

    ck_tile::DeviceMem sorted_token_ids_buf(sorted_token_ids_host);
    ck_tile::DeviceMem sorted_weight_buf(sorted_weight_host);
    ck_tile::DeviceMem sorted_expert_ids_buf(sorted_expert_ids_host);
    ck_tile::DeviceMem num_sorted_tiles_buf(num_sorted_tiles_host);

    fused_moegemm_traits traits{
        prec_i,
        prec_w,
        prec_o,
        prec_st,
        prec_sw,
        prec_sq,
        prec_kw,
        block_m,
        gate_only,
        fused_quant};

    fused_moegemm_args args{
        a_buf.GetDeviceBuffer(),
        fused_quant != 0 ? sa_buf.GetDeviceBuffer() : nullptr,
        g_perm_buf.GetDeviceBuffer(),
        d_perm_buf.GetDeviceBuffer(),
        fused_quant != 0 ? sg_buf.GetDeviceBuffer() : nullptr,
        fused_quant != 0 ? sd_buf.GetDeviceBuffer() : nullptr,
        fused_quant == 1 ? sy_buf.GetDeviceBuffer() : nullptr,
        o_buf.GetDeviceBuffer(),
        sorted_token_ids_buf.GetDeviceBuffer(),
        sorted_weight_buf.GetDeviceBuffer(),
        sorted_expert_ids_buf.GetDeviceBuffer(),
        num_sorted_tiles_buf.GetDeviceBuffer(),
        hidden_size,
        shared_intermediate_size_0,
        tokens,
        experts,
        topk,
        stride};

    float ave_time = fused_moegemm(
        traits,
        args,
        ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    if (ave_time < 0) {
      std::cout << " not supported!" << std::endl << std::flush;
      return false;
    }

    // float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << ", " << ave_time * 1.E3 << " us, " << cal_tflops(ave_time)
              << " tflops, " << cal_tbps(ave_time) << " TB/s" << std::flush;
    bool pass = true;

    if (do_validation) {
      ck_tile::reference_fused_moe<AccDataType, ck_tile::element_wise::Gelu>(
          a_host,
          g_host,
          d_host,
          sa_host,
          sg_host,
          sd_host,
          sy_host,
          o_host,
          sorted_token_ids_host,
          sorted_weight_host,
          sorted_expert_ids_host,
          num_sorted_tiles_host,
          topk_ids_host,
          block_m,
          tokens,
          experts,
          hidden_size,
          shared_intermediate_size_0,
          topk,
          gate_only);

      auto o_dev = o_buf.ToHost<ODataType>();
      // o_dev.savetxt("gpu-out.txt", "float");
      auto [rtol, atol] = get_elimit<ADataType>();
      pass &= ck_tile::check_err(
          o_dev,
          o_host,
          std::string("OUT Error: Incorrect results!"),
          rtol,
          atol);
      std::cout << ", valid:" << (pass ? "y" : "n") << std::flush;
    }
    std::cout << std::flush << std::endl;

    return pass;
  }
  return false;
}

int main(int argc, char* argv[]) {
  auto [result, arg_parser] = create_args(argc, argv);
  if (!result)
    return -1;

  std::string prec_i = arg_parser.get_str("prec_i");
  std::string prec_w = arg_parser.get_str("prec_w");
  std::string prec_o = arg_parser.get_str("prec_o");
  std::string prec_st = arg_parser.get_str("prec_st");
  std::string prec_sw = arg_parser.get_str("prec_sw");
  std::string prec_sq = arg_parser.get_str("prec_sq");
  std::string prec_kw = arg_parser.get_str("prec_kw");
  prec_st = (prec_st == "auto") ? "fp32" : prec_st;
  prec_sw = (prec_sw == "auto") ? "fp32" : prec_sw;
  prec_sq = (prec_sq == "auto") ? "fp32" : prec_sq;
  prec_kw = (prec_kw == "auto") ? "fp32" : prec_kw;

  // no dynamic quant case
  if (prec_i == "bf16" && prec_w == "bf16" && prec_o == "bf16" &&
      prec_kw == "fp32") {
    return run<ck_tile::bf16_t,
               ck_tile::bf16_t,
               ck_tile::bf16_t,
               float,
               float,
               float,
               float>(arg_parser)
        ? 0
        : -2;
  } else if (
      prec_i == "fp16" && prec_w == "fp16" && prec_o == "fp16" &&
      prec_kw == "fp32") {
    return run<ck_tile::fp16_t,
               ck_tile::fp16_t,
               ck_tile::fp16_t,
               float,
               float,
               float,
               float>(arg_parser)
        ? 0
        : -2;
  }

  return -3;
}

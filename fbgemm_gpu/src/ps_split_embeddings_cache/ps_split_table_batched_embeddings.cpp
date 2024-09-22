/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./ps_table_batched_embeddings.h"

#include <torch/custom_class.h>

using namespace at;
using namespace ps;

namespace {
class EmbeddingParameterServerWrapper : public torch::jit::CustomClassHolder {
 public:
  EmbeddingParameterServerWrapper(
      const std::vector<std::string>& tps_ips,
      const std::vector<int64_t>& tps_ports,
      int64_t tbe_id,
      int64_t maxLocalIndexLength = 54,
      int64_t num_threads = 32,
      int64_t maxKeysPerRequest = 500,
      int64_t l2_cache_size_gb = 0,
      int64_t max_D = 0) {
    TORCH_CHECK(
        tps_ips.size() == tps_ports.size(),
        "tps_ips and tps_ports must have the same size");
    std::vector<std::pair<std::string, int>> tpsHosts = {};
    for (int i = 0; i < tps_ips.size(); i++) {
      tpsHosts.push_back(std::make_pair(tps_ips[i], tps_ports[i]));
    }

    impl_ = std::make_shared<ps::EmbeddingParameterServer>(
        std::move(tpsHosts),
        tbe_id,
        maxLocalIndexLength,
        num_threads,
        maxKeysPerRequest,
        l2_cache_size_gb,
        max_D);
  }

  void set_cuda(
      Tensor indices,
      Tensor weights,
      Tensor count,
      int64_t timestep,
      bool is_bwd = false) {
    return impl_->set_cuda(indices, weights, count, timestep, is_bwd);
  }

  void get_cuda(Tensor indices, Tensor weights, Tensor count) {
    return impl_->get_cuda(indices, weights, count);
  }

  void set(Tensor indices, Tensor weights, Tensor count, bool is_bwd = false) {
    return impl_->set(indices, weights, count, is_bwd);
  }

  void get(Tensor indices, Tensor weights, Tensor count) {
    return impl_->get(indices, weights, count);
  }

  void compact() {
    return impl_->compact();
  }

  void flush() {
    return impl_->flush();
  }

  void cleanup() {
    return impl_->cleanup();
  }

 private:
  // shared pointer since we use shared_from_this() in callbacks.
  std::shared_ptr<EmbeddingParameterServer> impl_;
};

static auto embedding_parameter_server_wrapper =
    torch::class_<EmbeddingParameterServerWrapper>(
        "fbgemm",
        "EmbeddingParameterServerWrapper")
        .def(torch::init<
             const std::vector<std::string>,
             const std::vector<int64_t>,
             int64_t,
             int64_t,
             int64_t,
             int64_t,
             int64_t,
             int64_t>())
        .def("set_cuda", &EmbeddingParameterServerWrapper::set_cuda)
        .def("get_cuda", &EmbeddingParameterServerWrapper::get_cuda)
        .def("compact", &EmbeddingParameterServerWrapper::compact)
        .def("flush", &EmbeddingParameterServerWrapper::flush)
        .def("set", &EmbeddingParameterServerWrapper::set)
        .def("get", &EmbeddingParameterServerWrapper::get)
        .def("cleanup", &EmbeddingParameterServerWrapper::cleanup);
} // namespace

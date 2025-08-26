/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "deeplearning/fbgemm/fbgemm_gpu/include/fbgemm_gpu/split_embeddings_cache/raw_embedding_streamer.h" // @manual=//deeplearning/fbgemm/fbgemm_gpu/src/split_embeddings_cache:raw_embedding_streamer
#ifdef FBGEMM_FBCODE
#include <folly/experimental/coro/GmockHelpers.h>
#include "aiplatform/gmpp/experimental/training_ps/gen-cpp2/TrainingParameterServerService.h"
#include "servicerouter/client/cpp2/mocks/MockSRClientFactory.h"
#include "thrift/lib/cpp2/util/ScopedServerInterfaceThread.h"
#endif

using namespace ::testing;
using namespace fbgemm_gpu;
constexpr int64_t EMBEDDING_DIMENSION = 8;

#ifdef FBGEMM_FBCODE
class MockTrainingParameterServerService
    : public ::apache::thrift::ServiceHandler<
          aiplatform::gmpp::experimental::training_ps::
              TrainingParameterServerService> {
 public:
  MOCK_METHOD(
      folly::coro::Task<std::unique_ptr<
          aiplatform::gmpp::experimental::training_ps::SetEmbeddingsResponse>>,
      co_setEmbeddings,
      (std::unique_ptr<
          aiplatform::gmpp::experimental::training_ps::SetEmbeddingsRequest>));
};
#endif

static std::unique_ptr<fbgemm_gpu::RawEmbeddingStreamer>
getRawEmbeddingStreamer(
    const std::string& unique_id,
    bool enable_raw_embedding_streaming = false,
    const std::vector<std::string>& table_names = {},
    const std::vector<int64_t>& table_offsets = {},
    const std::vector<int64_t>& table_sizes = {}) {
  return std::make_unique<fbgemm_gpu::RawEmbeddingStreamer>(
      unique_id,
      enable_raw_embedding_streaming,
      3, // res_store_shards
      0, // res_server_port
      table_names,
      table_offsets,
      table_sizes);
}

TEST(RawEmbeddingStreamerTest, TestConstructorAndDestructor) {
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  auto streamer = getRawEmbeddingStreamer(
      "test_constructor", false, table_names, table_offsets, table_sizes);
  EXPECT_NE(streamer, nullptr);
}

TEST(RawEmbeddingStreamerTest, TestStreamWithoutStreaming) {
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  auto streamer = getRawEmbeddingStreamer(
      "test_no_streaming", false, table_names, table_offsets, table_sizes);

  auto indices = at::tensor(
      {10, 2, 1, 150, 170, 230, 280},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = at::randn(
      {indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));
  auto count = at::tensor(
      {indices.size(0)}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));

  // Should not crash when streaming is disabled
  streamer->stream(indices, weights, std::nullopt, count, true, true);
}

#ifdef FBGEMM_FBCODE
TEST(RawEmbeddingStreamerTest, TestTensorStream) {
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  auto streamer = getRawEmbeddingStreamer(
      "test_tensor_stream", true, table_names, table_offsets, table_sizes);

  // Mock TrainingParameterServerService
  auto mock_service = std::make_shared<MockTrainingParameterServerService>();
  auto mock_server =
      std::make_shared<apache::thrift::ScopedServerInterfaceThread>(
          mock_service,
          "::1",
          0,
          facebook::services::TLSConfig::applyDefaultsToThriftServer);
  auto& mock_client_factory =
      facebook::servicerouter::getMockSRClientFactory(false /* strict */);
  mock_client_factory.registerMockService(
      "realtime.delta.publish.esr", mock_server);

  // Test with invalid indices - should not call service
  auto invalid_indices = at::tensor(
      {300, 301, 999}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = at::randn(
      {invalid_indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));
  EXPECT_CALL(*mock_service, co_setEmbeddings(_)).Times(0);
  folly::coro::blockingWait(
      streamer->tensor_stream(invalid_indices, weights, std::nullopt));

  // Test with valid indices - should call service
  auto valid_indices = at::tensor(
      {10, 2, 1, 150, 170, 230, 280},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  weights = at::randn(
      {valid_indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));
  EXPECT_CALL(*mock_service, co_setEmbeddings(_))
      .Times(3) // 3 shards with consistent hashing
      .WillRepeatedly(folly::coro::gmock_helpers::CoInvoke(
          [](std::unique_ptr<
              aiplatform::gmpp::experimental::training_ps::SetEmbeddingsRequest>
                 request)
              -> folly::coro::Task<
                  std::unique_ptr<aiplatform::gmpp::experimental::training_ps::
                                      SetEmbeddingsResponse>> {
            co_return std::make_unique<
                aiplatform::gmpp::experimental::training_ps::
                    SetEmbeddingsResponse>();
          }));
  folly::coro::blockingWait(
      streamer->tensor_stream(valid_indices, weights, std::nullopt));
}

TEST(RawEmbeddingStreamerTest, TestStreamWithCopy) {
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  auto streamer = getRawEmbeddingStreamer(
      "test_stream_copy", true, table_names, table_offsets, table_sizes);

  // Mock TrainingParameterServerService
  auto mock_service = std::make_shared<MockTrainingParameterServerService>();
  auto mock_server =
      std::make_shared<apache::thrift::ScopedServerInterfaceThread>(
          mock_service,
          "::1",
          0,
          facebook::services::TLSConfig::applyDefaultsToThriftServer);
  auto& mock_client_factory =
      facebook::servicerouter::getMockSRClientFactory(false /* strict */);
  mock_client_factory.registerMockService(
      "realtime.delta.publish.esr", mock_server);

  auto indices = at::tensor(
      {10, 2, 1, 150, 170, 230, 280},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = at::randn(
      {indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));
  auto count = at::tensor(
      {indices.size(0)}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));

  // Stop the dequeue thread to get accurate queue size
  streamer->join_weights_stream_thread();

  // Test blocking tensor copy
  streamer->stream(indices, weights, std::nullopt, count, true, true);
  EXPECT_EQ(streamer->get_weights_to_stream_queue_size(), 1);

  // Test non-blocking tensor copy
  streamer->stream(indices, weights, std::nullopt, count, true, false);
  EXPECT_EQ(streamer->get_weights_to_stream_queue_size(), 1);
  streamer->join_stream_tensor_copy_thread();
  EXPECT_EQ(streamer->get_weights_to_stream_queue_size(), 2);
}

TEST(RawEmbeddingStreamerTest, TestStreamE2E) {
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  // Mock TrainingParameterServerService
  auto mock_service = std::make_shared<MockTrainingParameterServerService>();
  auto mock_server =
      std::make_shared<apache::thrift::ScopedServerInterfaceThread>(
          mock_service,
          "::1",
          0,
          facebook::services::TLSConfig::applyDefaultsToThriftServer);
  auto& mock_client_factory =
      facebook::servicerouter::getMockSRClientFactory(false /* strict */);
  mock_client_factory.registerMockService(
      "realtime.delta.publish.esr", mock_server);

  auto default_response =
      [](std::unique_ptr<
          aiplatform::gmpp::experimental::training_ps::SetEmbeddingsRequest>
             request)
      -> folly::coro::Task<std::unique_ptr<
          aiplatform::gmpp::experimental::training_ps::SetEmbeddingsResponse>> {
    co_return std::make_unique<
        aiplatform::gmpp::experimental::training_ps::SetEmbeddingsResponse>();
  };

  EXPECT_CALL(*mock_service, co_setEmbeddings(_))
      .Times(3) // 3 shards with consistent hashing
      .WillRepeatedly(folly::coro::gmock_helpers::CoInvoke(default_response));

  auto streamer = getRawEmbeddingStreamer(
      "test_stream_e2e", true, table_names, table_offsets, table_sizes);

  auto indices = at::tensor(
      {10, 2, 1, 150, 170, 230, 280},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = at::randn(
      {indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));
  auto count = at::tensor(
      {indices.size(0)}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));

  streamer->stream(indices, weights, std::nullopt, count, true, true);
  // Make sure dequeue finished
  std::this_thread::sleep_for(std::chrono::seconds(1));
  streamer->join_weights_stream_thread();
}

TEST(RawEmbeddingStreamerTest, TestMismatchedIndicesWeights) {
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  auto streamer = getRawEmbeddingStreamer(
      "test_mismatch", true, table_names, table_offsets, table_sizes);

  // Mock TrainingParameterServerService
  auto mock_service = std::make_shared<MockTrainingParameterServerService>();
  auto mock_server =
      std::make_shared<apache::thrift::ScopedServerInterfaceThread>(
          mock_service,
          "::1",
          0,
          facebook::services::TLSConfig::applyDefaultsToThriftServer);
  auto& mock_client_factory =
      facebook::servicerouter::getMockSRClientFactory(false /* strict */);
  mock_client_factory.registerMockService(
      "realtime.delta.publish.esr", mock_server);

  // Test with mismatched sizes - should not call service
  auto indices = at::tensor(
      {10, 2, 1}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = at::randn(
      {5, EMBEDDING_DIMENSION}, // Different size than indices
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));

  EXPECT_CALL(*mock_service, co_setEmbeddings(_)).Times(0);
  folly::coro::blockingWait(
      streamer->tensor_stream(indices, weights, std::nullopt));
}

TEST(RawEmbeddingStreamerTest, TestStreamWithIdentities) {
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  auto streamer = getRawEmbeddingStreamer(
      "test_stream_identities", true, table_names, table_offsets, table_sizes);

  // Mock TrainingParameterServerService
  auto mock_service = std::make_shared<MockTrainingParameterServerService>();
  auto mock_server =
      std::make_shared<apache::thrift::ScopedServerInterfaceThread>(
          mock_service,
          "::1",
          0,
          facebook::services::TLSConfig::applyDefaultsToThriftServer);
  auto& mock_client_factory =
      facebook::servicerouter::getMockSRClientFactory(false /* strict */);
  mock_client_factory.registerMockService(
      "realtime.delta.publish.esr", mock_server);

  auto indices = at::tensor(
      {10, 2, 1, 150, 170, 230, 280},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = at::randn(
      {indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));
  auto identities = at::tensor(
      {1001, 1002, 1003, 1004, 1005, 1006, 1007},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto count = at::tensor(
      {indices.size(0)}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));

  // Test that identities are properly handled in tensor_stream
  EXPECT_CALL(*mock_service, co_setEmbeddings(_))
      .Times(3) // 3 shards with consistent hashing
      .WillRepeatedly(folly::coro::gmock_helpers::CoInvoke(
          [](std::unique_ptr<
              aiplatform::gmpp::experimental::training_ps::SetEmbeddingsRequest>
                 request)
              -> folly::coro::Task<
                  std::unique_ptr<aiplatform::gmpp::experimental::training_ps::
                                      SetEmbeddingsResponse>> {
            // Verify that the request is properly formed
            EXPECT_GT(request->fqns()->size(), 0);
            co_return std::make_unique<
                aiplatform::gmpp::experimental::training_ps::
                    SetEmbeddingsResponse>();
          }));
  folly::coro::blockingWait(
      streamer->tensor_stream(indices, weights, identities));

  // Test streaming with identities using the stream method
  streamer->join_weights_stream_thread(); // Stop dequeue thread for testing
  streamer->stream(indices, weights, identities, count, true, true);
  EXPECT_EQ(streamer->get_weights_to_stream_queue_size(), 1);
}
#endif

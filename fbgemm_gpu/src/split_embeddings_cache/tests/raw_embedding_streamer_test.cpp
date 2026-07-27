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
#include <folly/coro/GmockHelpers.h>
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
      500000, // res_chunk_size
      8, // res_num_consumers
      4, // res_num_copy_threads
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
  streamer->stream(
      indices, weights, std::nullopt, std::nullopt, count, true, true);
}

namespace {
// Row-major tensor with distinct, predictable values so a sliced copy is
// unambiguous: value at [r, c] == r * dim + c.
at::Tensor makeRowMajor(int64_t num_rows, int64_t dim, at::ScalarType dtype) {
  return at::arange(
             num_rows * dim, at::TensorOptions().device(at::kCPU).dtype(dtype))
      .reshape({num_rows, dim});
}
} // namespace

// tensor_copy_chunk is build-agnostic (defined outside FBGEMM_FBCODE). Expected
// tensors are constructed independently via at::slice, never copied from impl
// output.
TEST(RawEmbeddingStreamerTest, TensorCopyChunkFullRange) {
  constexpr int64_t kNumRows = 5;
  auto indices = at::tensor(
      {10, 20, 30, 40, 50},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = makeRowMajor(kNumRows, EMBEDDING_DIMENSION, c10::kFloat);

  auto item = tensor_copy_chunk(
      indices, weights, std::nullopt, std::nullopt, /*start_row=*/0, kNumRows);

  EXPECT_TRUE(at::equal(item.indices, indices));
  EXPECT_TRUE(at::equal(item.weights, weights));
  EXPECT_FALSE(item.identities.has_value());
  EXPECT_FALSE(item.runtime_meta.has_value());
  const auto expected_count = at::tensor(
      {kNumRows}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  EXPECT_TRUE(at::equal(item.count, expected_count));
}

TEST(RawEmbeddingStreamerTest, TensorCopyChunkNonZeroStartSlicesCorrectSlice) {
  // Guards the start_row*dim / end_row*dim offset arithmetic in the copy.
  constexpr int64_t kNumRows = 6;
  constexpr int64_t kStart = 2;
  constexpr int64_t kEnd = 5; // n == 3
  auto indices = at::tensor(
      {10, 20, 30, 40, 50, 60},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = makeRowMajor(kNumRows, EMBEDDING_DIMENSION, c10::kFloat);

  auto item = tensor_copy_chunk(
      indices, weights, std::nullopt, std::nullopt, kStart, kEnd);

  EXPECT_TRUE(at::equal(item.indices, indices.slice(0, kStart, kEnd)));
  EXPECT_TRUE(at::equal(item.weights, weights.slice(0, kStart, kEnd)));
  const auto expected_count = at::tensor(
      {kEnd - kStart}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  EXPECT_TRUE(at::equal(item.count, expected_count));
}

TEST(RawEmbeddingStreamerTest, TensorCopyChunkCopiesIdentitiesAndRuntimeMeta) {
  constexpr int64_t kNumRows = 5;
  constexpr int64_t kStart = 1;
  constexpr int64_t kEnd = 4; // n == 3
  auto indices = at::tensor(
      {10, 20, 30, 40, 50},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = makeRowMajor(kNumRows, EMBEDDING_DIMENSION, c10::kFloat);
  auto identities = makeRowMajor(kNumRows, /*dim=*/2, at::kLong);
  auto runtime_meta = makeRowMajor(kNumRows, /*dim=*/1, at::kLong);

  auto item = tensor_copy_chunk(
      indices, weights, identities, runtime_meta, kStart, kEnd);

  ASSERT_TRUE(item.identities.has_value());
  ASSERT_TRUE(item.runtime_meta.has_value());
  EXPECT_TRUE(at::equal(*item.identities, identities.slice(0, kStart, kEnd)));
  EXPECT_TRUE(
      at::equal(*item.runtime_meta, runtime_meta.slice(0, kStart, kEnd)));
  EXPECT_TRUE(at::equal(item.weights, weights.slice(0, kStart, kEnd)));
}

TEST(RawEmbeddingStreamerTest, TensorCopyChunkAbsentOptionalsStayNullopt) {
  constexpr int64_t kNumRows = 4;
  auto indices = at::tensor(
      {10, 20, 30, 40}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = makeRowMajor(kNumRows, EMBEDDING_DIMENSION, c10::kFloat);

  auto item = tensor_copy_chunk(
      indices, weights, std::nullopt, std::nullopt, /*start_row=*/0, kNumRows);

  EXPECT_FALSE(item.identities.has_value());
  EXPECT_FALSE(item.runtime_meta.has_value());
}

TEST(RawEmbeddingStreamerTest, TensorCopyChunkInt32IndicesDtype) {
  // Coverage for the integral-index dispatch branch with int32 indices.
  constexpr int64_t kNumRows = 5;
  constexpr int64_t kStart = 1;
  constexpr int64_t kEnd = 4;
  auto indices = at::tensor(
      {10, 20, 30, 40, 50},
      at::TensorOptions().device(at::kCPU).dtype(at::kInt));
  auto weights = makeRowMajor(kNumRows, EMBEDDING_DIMENSION, c10::kFloat);

  auto item = tensor_copy_chunk(
      indices, weights, std::nullopt, std::nullopt, kStart, kEnd);

  EXPECT_EQ(item.indices.scalar_type(), at::kInt);
  EXPECT_TRUE(at::equal(item.indices, indices.slice(0, kStart, kEnd)));
  EXPECT_TRUE(at::equal(item.weights, weights.slice(0, kStart, kEnd)));
}

TEST(RawEmbeddingStreamerTest, TensorCopyChunkFloatRuntimeMeta) {
  // tensor_copy_chunk must dispatch runtime_meta over all dtypes
  // (FBGEMM_DISPATCH_ALL_TYPES): a float runtime_meta would throw if it were
  // dispatched integral-only.
  constexpr int64_t kNumRows = 5;
  constexpr int64_t kStart = 1;
  constexpr int64_t kEnd = 4;
  auto indices = at::tensor(
      {10, 20, 30, 40, 50},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = makeRowMajor(kNumRows, EMBEDDING_DIMENSION, c10::kFloat);
  auto runtime_meta = makeRowMajor(kNumRows, /*dim=*/2, c10::kFloat);

  auto item = tensor_copy_chunk(
      indices, weights, std::nullopt, runtime_meta, kStart, kEnd);

  ASSERT_TRUE(item.runtime_meta.has_value());
  EXPECT_TRUE(
      at::equal(*item.runtime_meta, runtime_meta.slice(0, kStart, kEnd)));
}

TEST(RawEmbeddingStreamerTest, TensorCopyChunkHalfWeights) {
  // Coverage for the half (fp16) weights dispatch branch of
  // FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE -- a common quantized serving dtype.
  constexpr int64_t kNumRows = 5;
  constexpr int64_t kStart = 1;
  constexpr int64_t kEnd = 4;
  auto indices = at::tensor(
      {10, 20, 30, 40, 50},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = makeRowMajor(kNumRows, EMBEDDING_DIMENSION, c10::kHalf);

  auto item = tensor_copy_chunk(
      indices, weights, std::nullopt, std::nullopt, kStart, kEnd);

  EXPECT_EQ(item.weights.scalar_type(), at::kHalf);
  EXPECT_TRUE(at::equal(item.weights, weights.slice(0, kStart, kEnd)));
}

TEST(RawEmbeddingStreamerTest, TensorCopyChunkByteWeights) {
  // Coverage for the byte (int8) weights dispatch branch of
  // FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE -- the int8-quantized serving dtype.
  constexpr int64_t kNumRows = 5;
  constexpr int64_t kStart = 1;
  constexpr int64_t kEnd = 4;
  auto indices = at::tensor(
      {10, 20, 30, 40, 50},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = makeRowMajor(kNumRows, EMBEDDING_DIMENSION, c10::kByte);

  auto item = tensor_copy_chunk(
      indices, weights, std::nullopt, std::nullopt, kStart, kEnd);

  EXPECT_EQ(item.weights.scalar_type(), at::kByte);
  EXPECT_TRUE(at::equal(item.weights, weights.slice(0, kStart, kEnd)));
}

namespace {
// Structural invariants computeChunks must always satisfy: chunks are
// contiguous
// + non-overlapping starting at 0, cover exactly [0, num_rows), and every chunk
// is non-empty and no larger than chunk_size. An off-by-one in the tiling
// arithmetic breaks at least one of these.
void expectValidChunks(
    const std::vector<std::pair<int64_t, int64_t>>& chunks,
    int64_t num_rows,
    int64_t chunk_size) {
  int64_t cursor = 0;
  for (const auto& [start, end] : chunks) {
    EXPECT_EQ(start, cursor) << "chunks must be contiguous and non-overlapping";
    EXPECT_GT(end, start) << "no empty chunks";
    EXPECT_LE(end - start, chunk_size) << "each chunk must be <= chunk_size";
    cursor = end;
  }
  EXPECT_EQ(cursor, num_rows) << "chunks must cover exactly [0, num_rows)";
}
} // namespace

// computeChunks (like tensor_copy_chunk) is build-agnostic. Expected chunks are
// constructed independently.
TEST(RawEmbeddingStreamerTest, ComputeChunksExactMultiple) {
  const auto chunks = computeChunks(/*num_rows=*/8, /*chunk_size=*/4);
  const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 4}, {4, 8}};
  EXPECT_EQ(chunks, expected);
  expectValidChunks(chunks, /*num_rows=*/8, /*chunk_size=*/4);
}

TEST(RawEmbeddingStreamerTest, ComputeChunksRemainderChunk) {
  // Last chunk carries the remainder when num_rows isn't a multiple of
  // chunk_size.
  const auto chunks = computeChunks(/*num_rows=*/10, /*chunk_size=*/4);
  const std::vector<std::pair<int64_t, int64_t>> expected = {
      {0, 4}, {4, 8}, {8, 10}};
  EXPECT_EQ(chunks, expected);
  expectValidChunks(chunks, /*num_rows=*/10, /*chunk_size=*/4);
}

TEST(RawEmbeddingStreamerTest, ComputeChunksCountLessThanChunkSize) {
  // num_rows < chunk_size collapses to a single partial chunk.
  const auto chunks = computeChunks(/*num_rows=*/3, /*chunk_size=*/10);
  const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 3}};
  EXPECT_EQ(chunks, expected);
  expectValidChunks(chunks, /*num_rows=*/3, /*chunk_size=*/10);
}

TEST(RawEmbeddingStreamerTest, ComputeChunksSingleChunk) {
  // num_rows == chunk_size is exactly one full chunk (no empty trailing chunk).
  const auto chunks = computeChunks(/*num_rows=*/5, /*chunk_size=*/5);
  const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 5}};
  EXPECT_EQ(chunks, expected);
  expectValidChunks(chunks, /*num_rows=*/5, /*chunk_size=*/5);
}

TEST(RawEmbeddingStreamerTest, ComputeChunksChunkSizeOne) {
  const auto chunks = computeChunks(/*num_rows=*/4, /*chunk_size=*/1);
  const std::vector<std::pair<int64_t, int64_t>> expected = {
      {0, 1}, {1, 2}, {2, 3}, {3, 4}};
  EXPECT_EQ(chunks, expected);
  expectValidChunks(chunks, /*num_rows=*/4, /*chunk_size=*/1);
}

TEST(RawEmbeddingStreamerTest, ComputeChunksOneOverChunk) {
  // num_rows == chunk_size + 1: the +1 spills into a second, single-row chunk.
  const auto chunks = computeChunks(/*num_rows=*/5, /*chunk_size=*/4);
  const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 4}, {4, 5}};
  EXPECT_EQ(chunks, expected);
  expectValidChunks(chunks, /*num_rows=*/5, /*chunk_size=*/4);
}

TEST(RawEmbeddingStreamerTest, ComputeChunksZeroRowsIsEmpty) {
  EXPECT_TRUE(computeChunks(/*num_rows=*/0, /*chunk_size=*/4).empty());
}

TEST(RawEmbeddingStreamerTest, ComputeChunksZeroChunkSizeIsEmpty) {
  // Defensive guard: chunk_size==0 would loop forever (s += 0), so it must
  // short-circuit to an empty result.
  EXPECT_TRUE(computeChunks(/*num_rows=*/5, /*chunk_size=*/0).empty());
}

#ifdef FBGEMM_FBCODE
TEST(RawEmbeddingStreamerTest, CtorRejectsZeroKnob) {
  // A 0-valued RES knob would silently disable streaming (0 consumers never
  // drain the queue; res_chunk_size/res_num_copy_threads=0 make chunk ranges
  // empty), so the ctor must reject it loudly. The TORCH_CHECK fires before any
  // thrift client is created, so no mock server is needed.
  EXPECT_ANY_THROW(
      fbgemm_gpu::RawEmbeddingStreamer(
          "test_zero_knob",
          /*enable_raw_embedding_streaming=*/true,
          /*res_store_shards=*/3,
          /*res_server_port=*/0,
          /*res_chunk_size=*/0,
          /*res_num_consumers=*/8,
          /*res_num_copy_threads=*/4,
          /*table_names=*/{},
          /*table_offsets=*/{},
          /*table_sizes=*/{}));
}

TEST(RawEmbeddingStreamerTest, TestMultiChunkFanOutShipsEveryChunk) {
  // The stream()/tensor_stream tests all run at the default res_chunk_size (one
  // chunk), so the chunked_copy_and_enqueue -> computeChunks -> copy_executor_
  // fan-out -> collectAllRange -> per-chunk submit_stream_item composition is
  // otherwise never exercised with >1 chunk. Here res_chunk_size=4 over 10 rows
  // tiles into ceil(10/4)=3 chunks; with a single shard each chunk ships
  // exactly one co_setEmbeddings, so RPC count == chunk count proves every
  // chunk is copied and shipped -- a dropped/duplicated chunk future or an
  // off-by-one in the fan-out would change the count. computeChunks' partition
  // correctness (no gap/dup/overflow) is covered by the ComputeChunks* unit
  // tests above.
  std::vector<std::string> table_names = {"tb1"};
  std::vector<int64_t> table_offsets = {0};
  std::vector<int64_t> table_sizes = {0, 300};

  // Static storage duration so the co_setEmbeddings coroutine mock can read it
  // WITHOUT capturing (avoids the capturing-lambda-coroutine UAF lint).
  static std::atomic<int> rpc_count;
  rpc_count.store(0);
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

  auto counting_response =
      [](std::unique_ptr<
          aiplatform::gmpp::experimental::training_ps::SetEmbeddingsRequest>)
      -> folly::coro::Task<std::unique_ptr<
          aiplatform::gmpp::experimental::training_ps::SetEmbeddingsResponse>> {
    rpc_count.fetch_add(1);
    co_return std::make_unique<
        aiplatform::gmpp::experimental::training_ps::SetEmbeddingsResponse>();
  };
  EXPECT_CALL(*mock_service, co_setEmbeddings(_))
      .WillRepeatedly(folly::coro::gmock_helpers::CoInvoke(counting_response));

  // res_chunk_size=4 (not the 500000 default) so 10 rows fan out into 3 chunks
  // across the 3-worker copy pool; res_store_shards=1 so each chunk ships once.
  auto streamer = std::make_unique<fbgemm_gpu::RawEmbeddingStreamer>(
      "test_multi_chunk_fanout",
      /*enable_raw_embedding_streaming=*/true,
      /*res_store_shards=*/1,
      /*res_server_port=*/0,
      /*res_chunk_size=*/4,
      /*res_num_consumers=*/2,
      /*res_num_copy_threads=*/3,
      table_names,
      table_offsets,
      table_sizes);

  constexpr int64_t kNumRows = 10;
  auto indices = at::arange(
      kNumRows, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = makeRowMajor(kNumRows, EMBEDDING_DIMENSION, c10::kFloat);
  auto count = at::tensor(
      {kNumRows}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));

  streamer->stream(
      indices,
      weights,
      std::nullopt,
      std::nullopt,
      count,
      /*require_tensor_copy=*/true,
      /*blocking_tensor_copy=*/true);

  // Wait on the RPC count (not queue size) so no in-flight ship is missed.
  constexpr int kExpectedChunks = 3; // ceil(10 / 4)
  for (int i = 0; i < 1000 && rpc_count.load() < kExpectedChunks; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_EQ(rpc_count.load(), kExpectedChunks);
  streamer->join_weights_stream_thread();
}

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
  folly::coro::blockingWait(streamer->tensor_stream(
      invalid_indices, weights, std::nullopt, std::nullopt));

  // Test with valid indices - should call service
  auto valid_indices = at::tensor(
      {10, 2, 1, 150, 170, 230, 280},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  weights = at::randn(
      {valid_indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));
  EXPECT_CALL(*mock_service, co_setEmbeddings(_))
      .Times(3) // 3 shards with consistent hashing
      .WillRepeatedly(
          folly::coro::gmock_helpers::CoInvoke(
              [](std::unique_ptr<aiplatform::gmpp::experimental::training_ps::
                                     SetEmbeddingsRequest> request)
                  -> folly::coro::Task<
                      std::unique_ptr<aiplatform::gmpp::experimental::
                                          training_ps::SetEmbeddingsResponse>> {
                co_return std::make_unique<
                    aiplatform::gmpp::experimental::training_ps::
                        SetEmbeddingsResponse>();
              }));
  folly::coro::blockingWait(streamer->tensor_stream(
      valid_indices, weights, std::nullopt, std::nullopt));
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
  streamer->stream(
      indices, weights, std::nullopt, std::nullopt, count, true, true);
  EXPECT_EQ(streamer->get_weights_to_stream_queue_size(), 1);

  // Test non-blocking tensor copy. The copy runs on the dispatcher, so we must
  // join_dispatch() before checking the queue -- asserting the size
  // before the join would race the background copy.
  streamer->stream(
      indices, weights, std::nullopt, std::nullopt, count, true, false);
  streamer->join_dispatch();
  EXPECT_EQ(streamer->get_weights_to_stream_queue_size(), 2);
}

TEST(RawEmbeddingStreamerTest, TestStreamWithCopyZeroCountEnqueuesNothing) {
  // count <= 0 drives num_rows == 0, so chunked_copy_and_enqueue early-returns
  // and nothing is enqueued.
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  auto streamer = getRawEmbeddingStreamer(
      "test_zero_count", true, table_names, table_offsets, table_sizes);

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
  auto count =
      at::tensor({0}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));

  // Stop the consumer threads so the queue size is stable to read.
  streamer->join_weights_stream_thread();

  streamer->stream(
      indices,
      weights,
      std::nullopt,
      std::nullopt,
      count,
      /*require_tensor_copy=*/true,
      /*blocking_tensor_copy=*/true);
  EXPECT_EQ(streamer->get_weights_to_stream_queue_size(), 0);
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

  streamer->stream(
      indices, weights, std::nullopt, std::nullopt, count, true, true);
  // Bounded wait for the consumer to drain the enqueued item (so
  // co_setEmbeddings has run) before stopping the thread -- avoids a
  // fixed-sleep flake and the stop_-between-peek-and-process race.
  for (int i = 0; i < 1000 && streamer->get_weights_to_stream_queue_size() > 0;
       ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  streamer->join_weights_stream_thread();
}

TEST(RawEmbeddingStreamerTest, TestNoCopyMultiItemConsumerDrain) {
  // require_tensor_copy=false enqueues one raw item per stream() call (no D2H
  // copy, no chunking). Enqueue several and let the N-thread consumer pool
  // drain them concurrently through the UMPMC queue: every item must be
  // shipped, so co_setEmbeddings runs (kNumItems * shards-per-item) times.
  // Exercises the raw-enqueue (require_tensor_copy=false) branch and multi-item
  // concurrent drain. Wait on the RPC count BEFORE stopping so no in-flight
  // item is dropped.
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  // Static storage duration so the co_setEmbeddings coroutine mock below can
  // read it WITHOUT capturing -- a capturing coroutine lambda risks
  // use-after-free once its closure is destroyed
  // (cppcoreguidelines-avoid-capturing-lambda-coroutines). Reset per run.
  static std::atomic<int> rpc_count;
  rpc_count.store(0);
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

  auto counting_response =
      [](std::unique_ptr<
          aiplatform::gmpp::experimental::training_ps::SetEmbeddingsRequest>)
      -> folly::coro::Task<std::unique_ptr<
          aiplatform::gmpp::experimental::training_ps::SetEmbeddingsResponse>> {
    rpc_count.fetch_add(1);
    co_return std::make_unique<
        aiplatform::gmpp::experimental::training_ps::SetEmbeddingsResponse>();
  };
  EXPECT_CALL(*mock_service, co_setEmbeddings(_))
      .WillRepeatedly(folly::coro::gmock_helpers::CoInvoke(counting_response));

  auto streamer = getRawEmbeddingStreamer(
      "test_nocopy_multi_item", true, table_names, table_offsets, table_sizes);

  auto indices = at::tensor(
      {10, 2, 1, 150, 170, 230, 280},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = at::randn(
      {indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));
  auto count = at::tensor(
      {indices.size(0)}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));

  constexpr int kNumItems = 3;
  // These 7 indices span the 3 tables, so each item ships 3 shards (matches the
  // Times(3) in TestStreamE2E for a single item).
  constexpr int kShardsPerItem = 3;
  for (int i = 0; i < kNumItems; ++i) {
    streamer->stream(
        indices,
        weights,
        std::nullopt,
        std::nullopt,
        count,
        /*require_tensor_copy=*/false);
  }
  // Bounded wait until every item has been shipped, then stop -- waiting on the
  // count (not queue size) guarantees no dequeued-but-unprocessed item is
  // dropped by stop_.
  for (int i = 0; i < 1000 && rpc_count.load() < kNumItems * kShardsPerItem;
       ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_EQ(rpc_count.load(), kNumItems * kShardsPerItem);
  streamer->join_weights_stream_thread();
}

TEST(RawEmbeddingStreamerTest, TestCoSetEmbeddingsFailureIsSwallowed) {
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  auto streamer = getRawEmbeddingStreamer(
      "test_set_embeddings_throw",
      true,
      table_names,
      table_offsets,
      table_sizes);

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

  // Every shard RPC fails.
  EXPECT_CALL(*mock_service, co_setEmbeddings(_))
      .Times(3) // still attempts all 3 shards despite each failing
      .WillRepeatedly(
          folly::coro::gmock_helpers::CoInvoke(
              [](std::unique_ptr<aiplatform::gmpp::experimental::training_ps::
                                     SetEmbeddingsRequest>)
                  -> folly::coro::Task<
                      std::unique_ptr<aiplatform::gmpp::experimental::
                                          training_ps::SetEmbeddingsResponse>> {
                throw std::runtime_error("simulated RPC failure");
                co_return std::make_unique<
                    aiplatform::gmpp::experimental::training_ps::
                        SetEmbeddingsResponse>();
              }));

  auto indices = at::tensor(
      {10, 2, 1, 150, 170, 230, 280},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = at::randn(
      {indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));

  // A per-shard RPC failure must NOT propagate out of tensor_stream: it is
  // caught, logged, and counted (set_embeddings_rpc_failure), and the coroutine
  // completes normally so it can never escape a consumer thread and
  // std::terminate the trainer. The counter bump is not unit-asserted here (the
  // OBC logger has no in-test sink), so we verify the swallow behavior: all 3
  // shards are attempted and no exception escapes.
  EXPECT_NO_THROW(
      folly::coro::blockingWait(streamer->tensor_stream(
          indices, weights, std::nullopt, std::nullopt)));
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

  // Test with mismatched sizes - should not call service. The
  // shard_size_mismatch counter is bumped in this path, but OBC
  // counters are not in-process readable, so we verify the behavior (no RPC)
  // rather than the counter value.
  auto indices = at::tensor(
      {10, 2, 1}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = at::randn(
      {5, EMBEDDING_DIMENSION}, // Different size than indices
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));

  EXPECT_CALL(*mock_service, co_setEmbeddings(_)).Times(0);
  folly::coro::blockingWait(
      streamer->tensor_stream(indices, weights, std::nullopt, std::nullopt));
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
                        at::TensorOptions().device(at::kCPU).dtype(at::kLong))
                        .reshape({7, 1});
  auto runtime_meta = at::tensor(
                          {101, 102, 103, 104, 105, 106, 107},
                          at::TensorOptions().device(at::kCPU).dtype(at::kLong))
                          .reshape({7, 1});
  auto count = at::tensor(
      {indices.size(0)}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));

  // Test that identities and runtime_meta are properly handled in tensor_stream
  EXPECT_CALL(*mock_service, co_setEmbeddings(_))
      .Times(3) // 3 shards with consistent hashing
      .WillRepeatedly(
          folly::coro::gmock_helpers::CoInvoke(
              [](std::unique_ptr<aiplatform::gmpp::experimental::training_ps::
                                     SetEmbeddingsRequest> request)
                  -> folly::coro::Task<
                      std::unique_ptr<aiplatform::gmpp::experimental::
                                          training_ps::SetEmbeddingsResponse>> {
                // Verify that the request is properly formed
                EXPECT_GT(request->fqns()->size(), 0);
                co_return std::make_unique<
                    aiplatform::gmpp::experimental::training_ps::
                        SetEmbeddingsResponse>();
              }));
  folly::coro::blockingWait(
      streamer->tensor_stream(indices, weights, identities, runtime_meta));

  // Test streaming with identities and runtime_meta using the stream method
  streamer->join_weights_stream_thread(); // Stop dequeue thread for testing
  streamer->stream(
      indices, weights, identities, runtime_meta, count, true, true);
  EXPECT_EQ(streamer->get_weights_to_stream_queue_size(), 1);
}

TEST(RawEmbeddingStreamerTest, TestStreamWithCopyDoneFlagNonBlockingCopy) {
  std::vector<std::string> table_names = {"tb1", "tb2", "tb3"};
  std::vector<int64_t> table_offsets = {0, 100, 300};
  std::vector<int64_t> table_sizes = {0, 50, 200, 300};

  auto streamer = getRawEmbeddingStreamer(
      "test_flag_nonblocking", true, table_names, table_offsets, table_sizes);

  // Create CPU tensors
  auto indices = at::tensor(
      {10, 2, 1, 150, 170, 230, 280},
      at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto weights = at::randn(
      {indices.size(0), EMBEDDING_DIMENSION},
      at::TensorOptions().device(at::kCPU).dtype(c10::kFloat));
  auto count = at::tensor(
      {indices.size(0)}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));

  // Create a CPU int32 tensor with value 1 (already "done")
  auto copy_done_flag =
      at::ones({1}, at::TensorOptions().device(at::kCPU).dtype(at::kInt));

  // Stop the dequeue thread to get accurate queue size
  streamer->join_weights_stream_thread();

  // Test with copy_done_flag in non-blocking mode
  streamer->stream(
      indices,
      weights,
      std::nullopt,
      std::nullopt,
      count,
      true,
      false,
      copy_done_flag);

  // Wait for the async thread to complete
  streamer->join_dispatch();
  EXPECT_EQ(streamer->get_weights_to_stream_queue_size(), 1);
  // poll_flag() must have observed the flag (1) and reset it to 0; without the
  // reset the next iteration would stream before the D2H copy finished.
  EXPECT_EQ(copy_done_flag.item<int32_t>(), 0);
}
#endif

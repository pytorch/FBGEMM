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
// computeChunkRanges groups chunks per thread (outer index = thread). Flatten
// to the in-order chunk list so the coverage/contiguity invariants can be
// checked across the whole range regardless of the thread grouping.
std::vector<std::pair<int64_t, int64_t>> flatten(
    const std::vector<std::vector<std::pair<int64_t, int64_t>>>&
        thread_chunks) {
  std::vector<std::pair<int64_t, int64_t>> ranges;
  for (const auto& chunks : thread_chunks) {
    ranges.insert(ranges.end(), chunks.begin(), chunks.end());
  }
  return ranges;
}

// Structural invariants computeChunkRanges must always satisfy: ranges are
// contiguous + non-overlapping starting at 0, cover exactly [0, num_rows), and
// every chunk is non-empty and no larger than chunk_size. An off-by-one in the
// tiling arithmetic breaks at least one of these.
void expectValidChunkRanges(
    const std::vector<std::pair<int64_t, int64_t>>& ranges,
    int64_t num_rows,
    int64_t chunk_size) {
  int64_t cursor = 0;
  for (const auto& [start, end] : ranges) {
    EXPECT_EQ(start, cursor) << "ranges must be contiguous and non-overlapping";
    EXPECT_GT(end, start) << "no empty ranges";
    EXPECT_LE(end - start, chunk_size) << "each chunk must be <= chunk_size";
    cursor = end;
  }
  EXPECT_EQ(cursor, num_rows) << "ranges must cover exactly [0, num_rows)";
}
} // namespace

// computeChunkRanges (like tensor_copy_chunk) is build-agnostic. Expected
// ranges are constructed independently.
TEST(RawEmbeddingStreamerTest, ComputeChunkRangesExactMultiple) {
  const auto ranges = flatten(
      computeChunkRanges(/*num_rows=*/8, /*chunk_size=*/4, /*num_threads=*/2));
  const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 4}, {4, 8}};
  EXPECT_EQ(ranges, expected);
  expectValidChunkRanges(ranges, /*num_rows=*/8, /*chunk_size=*/4);
}

TEST(RawEmbeddingStreamerTest, ComputeChunkRangesRemainderChunk) {
  // Single thread => pure chunking; last chunk carries the remainder.
  const auto ranges = flatten(
      computeChunkRanges(/*num_rows=*/10, /*chunk_size=*/4, /*num_threads=*/1));
  const std::vector<std::pair<int64_t, int64_t>> expected = {
      {0, 4}, {4, 8}, {8, 10}};
  EXPECT_EQ(ranges, expected);
  expectValidChunkRanges(ranges, /*num_rows=*/10, /*chunk_size=*/4);
}

TEST(RawEmbeddingStreamerTest, ComputeChunkRangesCountLessThanChunkSize) {
  const auto ranges = flatten(
      computeChunkRanges(/*num_rows=*/3, /*chunk_size=*/10, /*num_threads=*/4));
  const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 3}};
  EXPECT_EQ(ranges, expected);
  expectValidChunkRanges(ranges, /*num_rows=*/3, /*chunk_size=*/10);
}

TEST(RawEmbeddingStreamerTest, ComputeChunkRangesSingleChunk) {
  const auto ranges = flatten(
      computeChunkRanges(/*num_rows=*/5, /*chunk_size=*/5, /*num_threads=*/4));
  const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 5}};
  EXPECT_EQ(ranges, expected);
  expectValidChunkRanges(ranges, /*num_rows=*/5, /*chunk_size=*/5);
}

TEST(RawEmbeddingStreamerTest, ComputeChunkRangesChunkSizeOne) {
  const auto ranges = flatten(
      computeChunkRanges(/*num_rows=*/4, /*chunk_size=*/1, /*num_threads=*/1));
  const std::vector<std::pair<int64_t, int64_t>> expected = {
      {0, 1}, {1, 2}, {2, 3}, {3, 4}};
  EXPECT_EQ(ranges, expected);
  expectValidChunkRanges(ranges, /*num_rows=*/4, /*chunk_size=*/1);
}

TEST(RawEmbeddingStreamerTest, ComputeChunkRangesZeroRowsIsEmpty) {
  EXPECT_TRUE(
      computeChunkRanges(/*num_rows=*/0, /*chunk_size=*/4, /*num_threads=*/4)
          .empty());
}

TEST(RawEmbeddingStreamerTest, ComputeChunkRangesNumThreadsExceedsNumChunks) {
  // n_threads is clamped to n_chunks, so exactly n_chunks groups are emitted,
  // one chunk each, with no empty group.
  const auto thread_chunks =
      computeChunkRanges(/*num_rows=*/6, /*chunk_size=*/3, /*num_threads=*/10);
  EXPECT_EQ(thread_chunks.size(), 2u) << "one group per chunk";
  const auto ranges = flatten(thread_chunks);
  const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 3}, {3, 6}};
  EXPECT_EQ(ranges, expected);
  expectValidChunkRanges(ranges, /*num_rows=*/6, /*chunk_size=*/3);
}

TEST(RawEmbeddingStreamerTest, ComputeChunkRangesThreadSplitThenChunk) {
  // num_threads < num_chunks and rows_per_thread not a multiple of chunk_size:
  // rows are pre-split into 2 per-thread bands ([0,50), [50,100)) and each band
  // is then chunked by 30, so boundaries land at the thread split (50), not at
  // 60. This locks the tiling to the original inline behavior.
  const auto thread_chunks = computeChunkRanges(
      /*num_rows=*/100, /*chunk_size=*/30, /*num_threads=*/2);
  EXPECT_EQ(thread_chunks.size(), 2u) << "one group per thread band";
  const std::vector<std::vector<std::pair<int64_t, int64_t>>> expected_groups =
      {{{0, 30}, {30, 50}}, {{50, 80}, {80, 100}}};
  EXPECT_EQ(thread_chunks, expected_groups);
  expectValidChunkRanges(
      flatten(thread_chunks), /*num_rows=*/100, /*chunk_size=*/30);
}

TEST(RawEmbeddingStreamerTest, ComputeChunkRangesNoEmptyTailRange) {
  // rows_per_thread rounds up (ceil(5/4)=2) so the 4th thread's band would be
  // [6,5); that empty band must be dropped, leaving 3 non-empty groups.
  const auto thread_chunks =
      computeChunkRanges(/*num_rows=*/5, /*chunk_size=*/1, /*num_threads=*/4);
  EXPECT_EQ(thread_chunks.size(), 3u) << "empty trailing band is dropped";
  const auto ranges = flatten(thread_chunks);
  const std::vector<std::pair<int64_t, int64_t>> expected = {
      {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}};
  EXPECT_EQ(ranges, expected);
  expectValidChunkRanges(ranges, /*num_rows=*/5, /*chunk_size=*/1);
}

TEST(RawEmbeddingStreamerTest, ComputeChunkRangesOneOverChunk) {
  // num_rows == chunk_size + 1: the +1 spills into a second, single-row chunk.
  const auto ranges = flatten(
      computeChunkRanges(/*num_rows=*/5, /*chunk_size=*/4, /*num_threads=*/1));
  const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 4}, {4, 5}};
  EXPECT_EQ(ranges, expected);
  expectValidChunkRanges(ranges, /*num_rows=*/5, /*chunk_size=*/4);
}

TEST(RawEmbeddingStreamerTest, ComputeChunkRangesZeroThreadsOrChunkSizeEmpty) {
  // Defensive guards: chunk_size==0 and num_threads==0 would divide by zero in
  // the ceil-div tiling, so both must short-circuit to an empty result.
  EXPECT_TRUE(
      computeChunkRanges(/*num_rows=*/5, /*chunk_size=*/0, /*num_threads=*/4)
          .empty());
  EXPECT_TRUE(
      computeChunkRanges(/*num_rows=*/5, /*chunk_size=*/4, /*num_threads=*/0)
          .empty());
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
  // join_dispatch_thread() before checking the queue -- asserting the size
  // before the join would race the background copy.
  streamer->stream(
      indices, weights, std::nullopt, std::nullopt, count, true, false);
  streamer->join_dispatch_thread();
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

TEST(RawEmbeddingStreamerTest, TestCoSetEmbeddingsThrowPropagates) {
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

  EXPECT_CALL(*mock_service, co_setEmbeddings(_))
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

  // Thrift repackages server-side exceptions into TApplicationException, so
  // assert that the exception still propagates (preserved contract). The
  // catch block bumps set_embeddings_rpc via the OBC logger before
  // rethrowing; OBC counters are not in-process readable, so the bump itself
  // is not unit-asserted here (the logger has no in-test sink).
  EXPECT_ANY_THROW(
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
  streamer->join_dispatch_thread();
  EXPECT_EQ(streamer->get_weights_to_stream_queue_size(), 1);
  // poll_flag() must have observed the flag (1) and reset it to 0; without the
  // reset the next iteration would stream before the D2H copy finished.
  EXPECT_EQ(copy_done_flag.item<int32_t>(), 0);
}
#endif

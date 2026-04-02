// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "allocator.h"
#include "client_service.h"
#include "default_config.h"
#include "test_server_helpers.h"
#include "types.h"
#include "utils.h"

DEFINE_string(protocol, "tcp", "Transfer protocol: rdma|tcp");
DEFINE_string(device_name, "", "Device name to use, valid if protocol=rdma");

namespace mooncake {
namespace testing {
namespace {
using ScatterRange = std::tuple<size_t, size_t, size_t>;
using ScatterKeyRanges =
    std::vector<std::pair<Replica::Descriptor, std::vector<ScatterRange>>>;

struct ChunkSpan {
    size_t dest_offset;
    size_t src_offset;
    size_t size;
};

struct ScatterFixtureData {
    std::string key_a;
    std::string key_b;
    size_t size_a;
    size_t size_b;
    ScatterKeyRanges key_ranges;
    std::vector<std::vector<ChunkSpan>> chunk_spans;
    std::vector<std::string> cleanup_keys;
    size_t total_dest_size;
};
}  // namespace

class ScatterStreamingTest : public ::testing::Test {
   protected:
    void SetUp() override { SetTransferTaskTestHooks(&hooks_); }

    void TearDown() override { SetTransferTaskTestHooks(nullptr); }

    static void SetUpTestSuite() {
        google::InitGoogleLogging("ScatterStreamingTest");
        FLAGS_logtostderr = 1;

        if (getenv("PROTOCOL")) FLAGS_protocol = getenv("PROTOCOL");
        if (getenv("DEVICE_NAME")) FLAGS_device_name = getenv("DEVICE_NAME");

        ASSERT_TRUE(master_.Start(InProcMasterConfigBuilder().build()));
        master_address_ = master_.master_address();

        auto provider =
            Client::Create("localhost:17830", "P2PHANDSHAKE", FLAGS_protocol,
                           std::nullopt, master_address_);
        ASSERT_TRUE(provider.has_value());
        segment_provider_ = provider.value();

        segment_ptr_ = allocate_buffer_allocator_memory(kSegmentSize);
        ASSERT_NE(segment_ptr_, nullptr);
        auto mount = segment_provider_->MountSegment(segment_ptr_, kSegmentSize,
                                                     FLAGS_protocol);
        ASSERT_TRUE(mount.has_value()) << toString(mount.error());

        auto client =
            Client::Create("localhost:17831", "P2PHANDSHAKE", FLAGS_protocol,
                           std::nullopt, master_address_);
        ASSERT_TRUE(client.has_value());
        test_client_ = client.value();

        client_buf_alloc_ = std::make_unique<SimpleAllocator>(kClientBufSize);
        auto reg = test_client_->RegisterLocalMemory(
            client_buf_alloc_->getBase(), kClientBufSize, "cpu:0", false,
            false);
        ASSERT_TRUE(reg.has_value()) << toString(reg.error());
    }

    static void TearDownTestSuite() {
        test_client_.reset();
        if (segment_provider_ && segment_ptr_) {
            segment_provider_->UnmountSegment(segment_ptr_, kSegmentSize);
        }
        segment_provider_.reset();
        if (segment_ptr_) free(segment_ptr_);
        master_.Stop();
        google::ShutdownGoogleLogging();
    }

    void PutPattern(const std::string& key, size_t size) {
        void* buf = client_buf_alloc_->allocate(size);
        ASSERT_NE(buf, nullptr);
        FillPattern(buf, size, key);

        std::vector<Slice> slices{{buf, size}};
        ReplicateConfig cfg;
        cfg.replica_num = 1;
        auto r = test_client_->Put(key, slices, cfg);
        ASSERT_TRUE(r.has_value()) << toString(r.error());
        client_buf_alloc_->deallocate(buf, size);
    }

    static void FillPattern(void* buf, size_t size, const std::string& seed) {
        auto* p = static_cast<uint8_t*>(buf);
        uint32_t h = 0;
        for (char c : seed) h = h * 31 + static_cast<uint8_t>(c);
        for (size_t i = 0; i < size; ++i) {
            h ^= (h << 13);
            h ^= (h >> 17);
            h ^= (h << 5);
            p[i] = static_cast<uint8_t>(h);
        }
    }

    Replica::Descriptor LookupReplica(const std::string& key) {
        auto query = test_client_->Query(key);
        EXPECT_TRUE(query.has_value()) << toString(query.error());
        EXPECT_FALSE(query->replicas.empty());
        return query->replicas.front();
    }

    bool ShouldUseLogicalGrouping() const { return FLAGS_protocol == "rdma"; }

    ScatterFixtureData BuildTwoKeyFixture() {
        ScatterFixtureData data;
        data.key_a = "scatter_streaming_key_a";
        data.key_b = "scatter_streaming_key_b";
        data.size_a = 512 * 1024;
        data.size_b = 512 * 1024;

        PutPattern(data.key_a, data.size_a);
        PutPattern(data.key_b, data.size_b);

        std::vector<ScatterRange> ranges_a{{0, 32 * 1024, 48 * 1024},
                                           {96 * 1024, 160 * 1024, 32 * 1024}};
        std::vector<ScatterRange> ranges_b{{192 * 1024, 64 * 1024, 64 * 1024},
                                           {320 * 1024, 256 * 1024, 48 * 1024}};

        data.key_ranges = {{LookupReplica(data.key_a), ranges_a},
                           {LookupReplica(data.key_b), ranges_b}};
        data.cleanup_keys = {data.key_a, data.key_b};
        data.total_dest_size = 512 * 1024;

        if (ShouldUseLogicalGrouping()) {
            data.chunk_spans = {
                {{0, 32 * 1024, 48 * 1024}, {96 * 1024, 160 * 1024, 32 * 1024}},
                {{192 * 1024, 64 * 1024, 64 * 1024},
                 {320 * 1024, 256 * 1024, 48 * 1024}}};
        } else {
            data.chunk_spans = {{{0, 32 * 1024, 48 * 1024}},
                                {{96 * 1024, 160 * 1024, 32 * 1024}},
                                {{192 * 1024, 64 * 1024, 64 * 1024}},
                                {{320 * 1024, 256 * 1024, 48 * 1024}}};
        }

        return data;
    }

    ScatterFixtureData BuildSingleLogicalChunkFixture() {
        ScatterFixtureData data;
        data.key_a = "scatter_streaming_single_group";
        data.size_a = 512 * 1024;
        data.total_dest_size = 256 * 1024;

        PutPattern(data.key_a, data.size_a);
        std::vector<ScatterRange> ranges{{0, 0, 64 * 1024},
                                         {80 * 1024, 96 * 1024, 64 * 1024},
                                         {160 * 1024, 224 * 1024, 32 * 1024}};
        data.key_ranges = {{LookupReplica(data.key_a), ranges}};
        data.cleanup_keys = {data.key_a};
        data.chunk_spans = {{{0, 0, 64 * 1024},
                             {80 * 1024, 96 * 1024, 64 * 1024},
                             {160 * 1024, 224 * 1024, 32 * 1024}}};
        return data;
    }

    ScatterFixtureData BuildMultiWindowFixture() {
        ScatterFixtureData data;
        data.total_dest_size = 2 * 1024 * 1024;
        const size_t object_size = 2 * 1024 * 1024;
        const size_t group_count = 10;
        const size_t ranges_per_group = 2;
        const size_t range_size = 32 * 1024;

        for (size_t group = 0; group < group_count; ++group) {
            std::string key = "scatter_streaming_window_" +
                              std::to_string(group % 2) + "_" +
                              std::to_string(group);
            PutPattern(key, object_size);
            data.cleanup_keys.push_back(key);

            std::vector<ScatterRange> ranges;
            std::vector<ChunkSpan> spans;
            for (size_t r = 0; r < ranges_per_group; ++r) {
                size_t dest_offset =
                    (group * ranges_per_group + r) * range_size;
                size_t src_offset = (group * 96 * 1024) + r * 40 * 1024;
                ranges.emplace_back(dest_offset, src_offset, range_size);
                spans.push_back({dest_offset, src_offset, range_size});
            }

            data.key_ranges.emplace_back(LookupReplica(key), ranges);
            data.chunk_spans.push_back(std::move(spans));
        }
        return data;
    }

    void ExpectChunkMatchesPattern(const std::vector<ChunkSpan>& spans,
                                   const std::string& seed,
                                   const void* dest_buffer) {
        std::vector<uint8_t> expected;
        size_t max_end = 0;
        for (const auto& span : spans) {
            max_end = std::max(max_end, span.src_offset + span.size);
        }
        expected.resize(max_end);
        FillPattern(expected.data(), expected.size(), seed);

        for (const auto& span : spans) {
            EXPECT_EQ(std::memcmp(static_cast<const char*>(dest_buffer) +
                                      span.dest_offset,
                                  expected.data() + span.src_offset, span.size),
                      0);
        }
    }

    void CleanupFixture(const ScatterFixtureData& data) {
        for (const auto& key : data.cleanup_keys) {
            test_client_->Remove(key);
        }
    }

    static constexpr size_t kSegmentSize = 2ULL * 1024 * 1024 * 1024;
    static constexpr size_t kClientBufSize = 128 * 1024 * 1024;

    static InProcMaster master_;
    static std::string master_address_;
    static std::shared_ptr<Client> segment_provider_;
    static std::shared_ptr<Client> test_client_;
    static void* segment_ptr_;
    static std::unique_ptr<SimpleAllocator> client_buf_alloc_;

    TransferTaskTestHooks hooks_;
};

InProcMaster ScatterStreamingTest::master_;
std::string ScatterStreamingTest::master_address_;
std::shared_ptr<Client> ScatterStreamingTest::segment_provider_ = nullptr;
std::shared_ptr<Client> ScatterStreamingTest::test_client_ = nullptr;
void* ScatterStreamingTest::segment_ptr_ = nullptr;
std::unique_ptr<SimpleAllocator> ScatterStreamingTest::client_buf_alloc_ =
    nullptr;

TEST_F(ScatterStreamingTest, DataMatchesBlockingScatterRead) {
    auto data = BuildTwoKeyFixture();
    void* blocking_buf = client_buf_alloc_->allocate(data.total_dest_size);
    void* streaming_buf = client_buf_alloc_->allocate(data.total_dest_size);
    ASSERT_NE(blocking_buf, nullptr);
    ASSERT_NE(streaming_buf, nullptr);
    std::memset(blocking_buf, 0, data.total_dest_size);
    std::memset(streaming_buf, 0, data.total_dest_size);

    EXPECT_EQ(
        test_client_->BatchTransferReadRanges(blocking_buf, data.key_ranges),
        ErrorCode::OK);

    auto handle = test_client_->StreamingBatchTransferReadRanges(
        streaming_buf, data.key_ranges);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(handle->wait_all(), ErrorCode::OK);
    EXPECT_EQ(std::memcmp(blocking_buf, streaming_buf, data.total_dest_size),
              0);

    client_buf_alloc_->deallocate(blocking_buf, data.total_dest_size);
    client_buf_alloc_->deallocate(streaming_buf, data.total_dest_size);
    CleanupFixture(data);
}

TEST_F(ScatterStreamingTest, WaitChunkPerLogicalChunk) {
    auto data = BuildTwoKeyFixture();
    void* buf = client_buf_alloc_->allocate(data.total_dest_size);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, 0, data.total_dest_size);

    auto handle =
        test_client_->StreamingBatchTransferReadRanges(buf, data.key_ranges);
    ASSERT_TRUE(handle.has_value());
    ASSERT_EQ(handle->num_chunks(), data.chunk_spans.size());

    std::vector<std::string> chunk_keys;
    if (ShouldUseLogicalGrouping()) {
        chunk_keys = {data.key_a, data.key_b};
    } else {
        chunk_keys = {data.key_a, data.key_a, data.key_b, data.key_b};
    }

    for (size_t i = 0; i < data.chunk_spans.size(); ++i) {
        EXPECT_EQ(handle->wait_chunk(i), ErrorCode::OK);
        EXPECT_TRUE(handle->is_chunk_ready(i));
        ExpectChunkMatchesPattern(data.chunk_spans[i], chunk_keys[i], buf);
    }

    EXPECT_EQ(handle->completed_count(), handle->num_chunks());
    client_buf_alloc_->deallocate(buf, data.total_dest_size);
    CleanupFixture(data);
}

TEST_F(ScatterStreamingTest, CompletedCountIsMonotonic) {
    auto data = BuildMultiWindowFixture();
    void* buf = client_buf_alloc_->allocate(data.total_dest_size);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, 0, data.total_dest_size);

    auto handle =
        test_client_->StreamingBatchTransferReadRanges(buf, data.key_ranges);
    ASSERT_TRUE(handle.has_value());

    size_t prev = 0;
    for (int i = 0; i < 8; ++i) {
        size_t current = handle->completed_count();
        EXPECT_GE(current, prev);
        EXPECT_LE(current, handle->num_chunks());
        prev = current;
    }

    EXPECT_EQ(handle->wait_all(), ErrorCode::OK);
    EXPECT_EQ(handle->completed_count(), handle->num_chunks());

    client_buf_alloc_->deallocate(buf, data.total_dest_size);
    CleanupFixture(data);
}

TEST_F(ScatterStreamingTest, InvalidChunkIndex) {
    auto data = BuildTwoKeyFixture();
    void* buf = client_buf_alloc_->allocate(data.total_dest_size);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, 0, data.total_dest_size);

    auto handle =
        test_client_->StreamingBatchTransferReadRanges(buf, data.key_ranges);
    ASSERT_TRUE(handle.has_value());

    EXPECT_EQ(handle->wait_chunk(handle->num_chunks() + 1),
              ErrorCode::INVALID_PARAMS);
    EXPECT_FALSE(handle->is_chunk_ready(handle->num_chunks() + 1));
    EXPECT_EQ(handle->wait_all(), ErrorCode::OK);

    client_buf_alloc_->deallocate(buf, data.total_dest_size);
    CleanupFixture(data);
}

TEST_F(ScatterStreamingTest, SingleLogicalChunk) {
    if (!ShouldUseLogicalGrouping()) {
        GTEST_SKIP()
            << "single logical chunk semantics require grouped scatter";
    }

    auto data = BuildSingleLogicalChunkFixture();
    void* buf = client_buf_alloc_->allocate(data.total_dest_size);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, 0, data.total_dest_size);

    auto handle =
        test_client_->StreamingBatchTransferReadRanges(buf, data.key_ranges);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(handle->num_chunks(), 1u);
    EXPECT_EQ(handle->wait_chunk(0), ErrorCode::OK);
    ExpectChunkMatchesPattern(data.chunk_spans[0], data.key_a, buf);

    client_buf_alloc_->deallocate(buf, data.total_dest_size);
    CleanupFixture(data);
}

TEST_F(ScatterStreamingTest, WindowBoundaryDoesNotSplitGroupedTasks) {
    if (!ShouldUseLogicalGrouping()) {
        GTEST_SKIP() << "window-boundary grouping semantics require rdma mode";
    }

    auto data = BuildMultiWindowFixture();
    void* buf = client_buf_alloc_->allocate(data.total_dest_size);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, 0, data.total_dest_size);

    auto handle =
        test_client_->StreamingBatchTransferReadRanges(buf, data.key_ranges);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(handle->num_chunks(), data.key_ranges.size());

    for (size_t i = 0; i < handle->num_chunks(); ++i) {
        ASSERT_EQ(handle->wait_chunk(i), ErrorCode::OK);
        ExpectChunkMatchesPattern(data.chunk_spans[i],
                                  "scatter_streaming_window_" +
                                      std::to_string(i % 2) + "_" +
                                      std::to_string(i),
                                  buf);
    }

    client_buf_alloc_->deallocate(buf, data.total_dest_size);
    CleanupFixture(data);
}

TEST_F(ScatterStreamingTest, InitialSubmitFailureReturnsNulloptAndFreesBatch) {
    std::atomic<int> alloc_count{0};
    std::atomic<int> free_count{0};
    bool fail_once = true;
    hooks_.fail_next_streaming_scatter_submit = [&]() {
        bool should_fail = fail_once;
        fail_once = false;
        return should_fail;
    };
    hooks_.on_batch_allocated = [&](BatchID) { ++alloc_count; };
    hooks_.on_batch_freed = [&](BatchID) { ++free_count; };

    auto data = BuildTwoKeyFixture();
    void* buf = client_buf_alloc_->allocate(data.total_dest_size);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, 0, data.total_dest_size);

    auto handle =
        test_client_->StreamingBatchTransferReadRanges(buf, data.key_ranges);
    EXPECT_FALSE(handle.has_value());
    EXPECT_EQ(alloc_count.load(), 1);
    EXPECT_EQ(free_count.load(), 1);

    client_buf_alloc_->deallocate(buf, data.total_dest_size);
    CleanupFixture(data);
}

TEST_F(ScatterStreamingTest, DestructorFreesBatchWithoutExplicitWaitAll) {
    std::atomic<int> alloc_count{0};
    std::atomic<int> free_count{0};
    hooks_.on_batch_allocated = [&](BatchID) { ++alloc_count; };
    hooks_.on_batch_freed = [&](BatchID) { ++free_count; };

    auto data = BuildMultiWindowFixture();
    void* buf = client_buf_alloc_->allocate(data.total_dest_size);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, 0, data.total_dest_size);

    {
        auto handle = test_client_->StreamingBatchTransferReadRanges(
            buf, data.key_ranges);
        ASSERT_TRUE(handle.has_value());
        EXPECT_EQ(alloc_count.load(), 1);
    }

    EXPECT_EQ(free_count.load(), 1);
    client_buf_alloc_->deallocate(buf, data.total_dest_size);
    CleanupFixture(data);
}

TEST_F(ScatterStreamingTest, MixedPollingAndWaitSequence) {
    auto data = BuildMultiWindowFixture();
    void* buf = client_buf_alloc_->allocate(data.total_dest_size);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, 0, data.total_dest_size);

    auto handle =
        test_client_->StreamingBatchTransferReadRanges(buf, data.key_ranges);
    ASSERT_TRUE(handle.has_value());
    ASSERT_GT(handle->num_chunks(), 1u);

    size_t before = handle->completed_count();
    EXPECT_LE(before, handle->num_chunks());

    bool chunk0_ready = handle->is_chunk_ready(0);
    EXPECT_EQ(handle->wait_chunk(0), ErrorCode::OK);
    EXPECT_TRUE(chunk0_ready || handle->is_chunk_ready(0));

    size_t after_first = handle->completed_count();
    EXPECT_GE(after_first, 1u);
    EXPECT_LE(after_first, handle->num_chunks());

    const size_t tail_chunk = handle->num_chunks() - 1;
    EXPECT_FALSE(handle->is_chunk_ready(tail_chunk));
    EXPECT_EQ(handle->wait_chunk(tail_chunk), ErrorCode::OK);
    EXPECT_TRUE(handle->is_chunk_ready(tail_chunk));

    EXPECT_EQ(handle->wait_all(), ErrorCode::OK);
    EXPECT_EQ(handle->completed_count(), handle->num_chunks());

    client_buf_alloc_->deallocate(buf, data.total_dest_size);
    CleanupFixture(data);
}

}  // namespace testing
}  // namespace mooncake

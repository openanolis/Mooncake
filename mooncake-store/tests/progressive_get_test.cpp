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

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
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

// ============================================================================
// Test fixture — mirrors ClientIntegrationTest setup
// ============================================================================

class ProgressiveGetTest : public ::testing::Test {
   protected:
    void SetUp() override { SetTransferTaskTestHooks(&hooks_); }

    void TearDown() override { SetTransferTaskTestHooks(nullptr); }

    static void SetUpTestSuite() {
        google::InitGoogleLogging("ProgressiveGetTest");
        FLAGS_logtostderr = 1;

        if (getenv("PROTOCOL")) FLAGS_protocol = getenv("PROTOCOL");
        if (getenv("DEVICE_NAME")) FLAGS_device_name = getenv("DEVICE_NAME");

        ASSERT_TRUE(master_.Start(InProcMasterConfigBuilder().build()));
        master_address_ = master_.master_address();

        // Segment provider: hosts the data objects
        auto provider =
            Client::Create("localhost:17820", "P2PHANDSHAKE", FLAGS_protocol,
                           std::nullopt, master_address_);
        ASSERT_TRUE(provider.has_value());
        segment_provider_ = provider.value();

        segment_ptr_ = allocate_buffer_allocator_memory(kSegmentSize);
        ASSERT_NE(segment_ptr_, nullptr);
        auto mount = segment_provider_->MountSegment(segment_ptr_, kSegmentSize,
                                                     FLAGS_protocol);
        ASSERT_TRUE(mount.has_value()) << toString(mount.error());

        // Test client: performs Get / ProgressiveGet
        auto client =
            Client::Create("localhost:17821", "P2PHANDSHAKE", FLAGS_protocol,
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

    // ---- helpers ----

    // Store `size` bytes filled with a deterministic pattern keyed on `key`.
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

    // Fill buffer with a reproducible byte pattern derived from `seed`.
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

    // Verify that `buf` contains the expected pattern.
    static bool VerifyPattern(const void* buf, size_t size,
                              const std::string& seed) {
        std::vector<uint8_t> expected(size);
        FillPattern(expected.data(), size, seed);
        return std::memcmp(buf, expected.data(), size) == 0;
    }

    static constexpr size_t kSegmentSize = 2ULL * 1024 * 1024 * 1024;  // 2 GB
    static constexpr size_t kClientBufSize = 128 * 1024 * 1024;        // 128 MB

    static InProcMaster master_;
    static std::string master_address_;
    static std::shared_ptr<Client> segment_provider_;
    static std::shared_ptr<Client> test_client_;
    static void* segment_ptr_;
    static std::unique_ptr<SimpleAllocator> client_buf_alloc_;

    TransferTaskTestHooks hooks_;
};

// Static members
InProcMaster ProgressiveGetTest::master_;
std::string ProgressiveGetTest::master_address_;
std::shared_ptr<Client> ProgressiveGetTest::segment_provider_ = nullptr;
std::shared_ptr<Client> ProgressiveGetTest::test_client_ = nullptr;
void* ProgressiveGetTest::segment_ptr_ = nullptr;
std::unique_ptr<SimpleAllocator> ProgressiveGetTest::client_buf_alloc_ =
    nullptr;

// ============================================================================
// Correctness tests
// ============================================================================

// Basic: ProgressiveGet retrieves data identical to regular Get.
TEST_F(ProgressiveGetTest, DataMatchesRegularGet) {
    const std::string key = "progressive_correctness_1";
    constexpr size_t kObjSize = 1 * 1024 * 1024;  // 1 MB
    constexpr size_t kChunkSize = 256 * 1024;     // 256 KB → 4 chunks

    PutPattern(key, kObjSize);

    // --- regular Get ---
    void* get_buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(get_buf, nullptr);
    {
        std::vector<Slice> slices{{get_buf, kObjSize}};
        auto r = test_client_->Get(key, slices);
        ASSERT_TRUE(r.has_value()) << toString(r.error());
    }

    // --- ProgressiveGet ---
    void* prog_buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(prog_buf, nullptr);
    {
        auto handle =
            test_client_->ProgressiveGet(key, prog_buf, kObjSize, kChunkSize);
        ASSERT_TRUE(handle.has_value());
        EXPECT_EQ(handle->num_chunks(),
                  (kObjSize + kChunkSize - 1) / kChunkSize);

        ErrorCode ec = handle->wait_all();
        EXPECT_EQ(ec, ErrorCode::OK);
    }

    // Byte-for-byte comparison
    EXPECT_EQ(std::memcmp(get_buf, prog_buf, kObjSize), 0)
        << "ProgressiveGet data differs from regular Get";
    EXPECT_TRUE(VerifyPattern(prog_buf, kObjSize, key))
        << "ProgressiveGet data doesn't match expected pattern";

    client_buf_alloc_->deallocate(get_buf, kObjSize);
    client_buf_alloc_->deallocate(prog_buf, kObjSize);

    // Cleanup
    test_client_->Remove(key);
}

// Per-chunk wait: wait_chunk returns correct data per chunk.
TEST_F(ProgressiveGetTest, WaitChunkPerChunk) {
    const std::string key = "progressive_per_chunk";
    constexpr size_t kObjSize = 512 * 1024;    // 512 KB
    constexpr size_t kChunkSize = 128 * 1024;  // 128 KB → 4 chunks

    PutPattern(key, kObjSize);

    void* buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(buf, nullptr);

    auto handle = test_client_->ProgressiveGet(key, buf, kObjSize, kChunkSize);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(handle->num_chunks(), 4u);

    // Wait for each chunk individually and verify it immediately
    std::vector<uint8_t> expected(kObjSize);
    FillPattern(expected.data(), kObjSize, key);

    for (size_t i = 0; i < handle->num_chunks(); ++i) {
        ErrorCode ec = handle->wait_chunk(i);
        EXPECT_EQ(ec, ErrorCode::OK) << "chunk " << i << " failed";

        size_t offset = i * kChunkSize;
        size_t len = std::min(kChunkSize, kObjSize - offset);
        EXPECT_EQ(std::memcmp(static_cast<char*>(buf) + offset,
                              expected.data() + offset, len),
                  0)
            << "chunk " << i << " data mismatch";

        // After waiting, is_chunk_ready must be true
        EXPECT_TRUE(handle->is_chunk_ready(i));
    }

    EXPECT_EQ(handle->completed_count(), handle->num_chunks());

    client_buf_alloc_->deallocate(buf, kObjSize);
    test_client_->Remove(key);
}

// Edge case: object size not a multiple of chunk_size (last chunk smaller).
TEST_F(ProgressiveGetTest, UnevenLastChunk) {
    const std::string key = "progressive_uneven";
    constexpr size_t kObjSize = 300 * 1024;    // 300 KB
    constexpr size_t kChunkSize = 128 * 1024;  // 128 KB → 3 chunks (128+128+44)

    PutPattern(key, kObjSize);

    void* buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(buf, nullptr);

    auto handle = test_client_->ProgressiveGet(key, buf, kObjSize, kChunkSize);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(handle->num_chunks(), 3u);

    ErrorCode ec = handle->wait_all();
    EXPECT_EQ(ec, ErrorCode::OK);
    EXPECT_TRUE(VerifyPattern(buf, kObjSize, key));

    client_buf_alloc_->deallocate(buf, kObjSize);
    test_client_->Remove(key);
}

// Single chunk: chunk_size >= object_size → 1 chunk.
TEST_F(ProgressiveGetTest, SingleChunk) {
    const std::string key = "progressive_single_chunk";
    constexpr size_t kObjSize = 64 * 1024;     // 64 KB
    constexpr size_t kChunkSize = 256 * 1024;  // 256 KB > object

    PutPattern(key, kObjSize);

    void* buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(buf, nullptr);

    auto handle = test_client_->ProgressiveGet(key, buf, kObjSize, kChunkSize);
    ASSERT_TRUE(handle.has_value());
    EXPECT_EQ(handle->num_chunks(), 1u);

    EXPECT_EQ(handle->wait_chunk(0), ErrorCode::OK);
    EXPECT_TRUE(VerifyPattern(buf, kObjSize, key));

    client_buf_alloc_->deallocate(buf, kObjSize);
    test_client_->Remove(key);
}

// is_chunk_ready / completed_count polling works before any wait.
TEST_F(ProgressiveGetTest, PollingBeforeWait) {
    const std::string key = "progressive_polling";
    constexpr size_t kObjSize = 256 * 1024;
    constexpr size_t kChunkSize = 64 * 1024;  // 4 chunks

    PutPattern(key, kObjSize);

    void* buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(buf, nullptr);

    auto handle = test_client_->ProgressiveGet(key, buf, kObjSize, kChunkSize);
    ASSERT_TRUE(handle.has_value());

    // completed_count() should be in [0, num_chunks]
    size_t cc = handle->completed_count();
    EXPECT_LE(cc, handle->num_chunks());

    // Now wait for all and verify
    EXPECT_EQ(handle->wait_all(), ErrorCode::OK);
    EXPECT_EQ(handle->completed_count(), handle->num_chunks());
    EXPECT_TRUE(VerifyPattern(buf, kObjSize, key));

    client_buf_alloc_->deallocate(buf, kObjSize);
    test_client_->Remove(key);
}

// wait_chunk with invalid index returns INVALID_PARAMS.
TEST_F(ProgressiveGetTest, InvalidChunkIndex) {
    const std::string key = "progressive_invalid_idx";
    constexpr size_t kObjSize = 64 * 1024;
    constexpr size_t kChunkSize = 64 * 1024;

    PutPattern(key, kObjSize);

    void* buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(buf, nullptr);

    auto handle = test_client_->ProgressiveGet(key, buf, kObjSize, kChunkSize);
    ASSERT_TRUE(handle.has_value());

    EXPECT_EQ(handle->wait_chunk(999), ErrorCode::INVALID_PARAMS);
    EXPECT_FALSE(handle->is_chunk_ready(999));

    // Normal operation still works
    EXPECT_EQ(handle->wait_all(), ErrorCode::OK);

    client_buf_alloc_->deallocate(buf, kObjSize);
    test_client_->Remove(key);
}

TEST_F(ProgressiveGetTest, CompletedCountIsMonotonic) {
    const std::string key = "progressive_completed_count_monotonic";
    constexpr size_t kObjSize = 2 * 1024 * 1024;
    constexpr size_t kChunkSize = 64 * 1024;

    PutPattern(key, kObjSize);

    void* buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(buf, nullptr);

    auto handle = test_client_->ProgressiveGet(key, buf, kObjSize, kChunkSize);
    ASSERT_TRUE(handle.has_value());

    size_t prev = 0;
    for (int i = 0; i < 8; ++i) {
        size_t current = handle->completed_count();
        EXPECT_GE(current, prev);
        EXPECT_LE(current, handle->num_chunks());
        prev = current;
    }

    EXPECT_FALSE(handle->is_chunk_ready(handle->num_chunks() - 1));

    EXPECT_EQ(handle->wait_all(), ErrorCode::OK);
    size_t final_count = handle->completed_count();
    EXPECT_EQ(final_count, handle->num_chunks());
    EXPECT_GE(final_count, prev);
    EXPECT_TRUE(VerifyPattern(buf, kObjSize, key));

    client_buf_alloc_->deallocate(buf, kObjSize);
    test_client_->Remove(key);
}

TEST_F(ProgressiveGetTest, MixedPollingAndWaitSequence) {
    const std::string key = "progressive_mixed_sequence";
    constexpr size_t kObjSize = 1024 * 1024;
    constexpr size_t kChunkSize = 128 * 1024;

    PutPattern(key, kObjSize);

    void* buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(buf, nullptr);

    auto handle = test_client_->ProgressiveGet(key, buf, kObjSize, kChunkSize);
    ASSERT_TRUE(handle.has_value());
    ASSERT_GT(handle->num_chunks(), 1u);

    size_t before = handle->completed_count();
    EXPECT_LE(before, handle->num_chunks());

    bool chunk0_ready = handle->is_chunk_ready(0);
    ErrorCode ec0 = handle->wait_chunk(0);
    EXPECT_EQ(ec0, ErrorCode::OK);
    EXPECT_TRUE(handle->is_chunk_ready(0));
    EXPECT_TRUE(chunk0_ready || handle->is_chunk_ready(0));

    size_t after_first = handle->completed_count();
    EXPECT_GE(after_first, 1u);
    EXPECT_LE(after_first, handle->num_chunks());

    const size_t tail_chunk = handle->num_chunks() - 1;
    EXPECT_FALSE(handle->is_chunk_ready(tail_chunk));
    EXPECT_EQ(handle->wait_chunk(tail_chunk), ErrorCode::OK);
    EXPECT_TRUE(handle->is_chunk_ready(tail_chunk));

    ErrorCode all_ec = handle->wait_all();
    EXPECT_EQ(all_ec, ErrorCode::OK);
    EXPECT_EQ(handle->completed_count(), handle->num_chunks());
    EXPECT_TRUE(VerifyPattern(buf, kObjSize, key));

    client_buf_alloc_->deallocate(buf, kObjSize);
    test_client_->Remove(key);
}

TEST_F(ProgressiveGetTest, WaitAllLargeChunkCount) {
    const std::string key = "progressive_wait_all_many_chunks";
    constexpr size_t kObjSize = 4 * 1024 * 1024;
    constexpr size_t kChunkSize = 32 * 1024;

    PutPattern(key, kObjSize);

    void* buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(buf, nullptr);

    auto handle = test_client_->ProgressiveGet(key, buf, kObjSize, kChunkSize);
    ASSERT_TRUE(handle.has_value());
    EXPECT_GT(handle->num_chunks(), 100u);
    EXPECT_FALSE(handle->is_chunk_ready(handle->num_chunks() - 1));

    EXPECT_EQ(handle->wait_all(), ErrorCode::OK);
    EXPECT_EQ(handle->completed_count(), handle->num_chunks());
    for (size_t i = 0; i < handle->num_chunks(); ++i) {
        EXPECT_TRUE(handle->is_chunk_ready(i));
    }
    EXPECT_TRUE(VerifyPattern(buf, kObjSize, key));

    client_buf_alloc_->deallocate(buf, kObjSize);
    test_client_->Remove(key);
}

TEST_F(ProgressiveGetTest, InitialSubmitFailureReturnsNulloptAndFreesBatch) {
    std::atomic<int> alloc_count{0};
    std::atomic<int> free_count{0};
    bool fail_once = true;
    hooks_.fail_next_progressive_submit = [&]() {
        bool should_fail = fail_once;
        fail_once = false;
        return should_fail;
    };
    hooks_.on_batch_allocated = [&](BatchID) { ++alloc_count; };
    hooks_.on_batch_freed = [&](BatchID) { ++free_count; };

    const std::string key = "progressive_submit_failure";
    constexpr size_t kObjSize = 256 * 1024;
    constexpr size_t kChunkSize = 64 * 1024;
    PutPattern(key, kObjSize);

    void* buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(buf, nullptr);

    auto handle = test_client_->ProgressiveGet(key, buf, kObjSize, kChunkSize);
    EXPECT_FALSE(handle.has_value());
    EXPECT_EQ(alloc_count.load(), 1);
    EXPECT_EQ(free_count.load(), 1);

    client_buf_alloc_->deallocate(buf, kObjSize);
    test_client_->Remove(key);
}

TEST_F(ProgressiveGetTest, DestructorFreesBatchWithoutExplicitWaitAll) {
    std::atomic<int> alloc_count{0};
    std::atomic<int> free_count{0};
    hooks_.on_batch_allocated = [&](BatchID) { ++alloc_count; };
    hooks_.on_batch_freed = [&](BatchID) { ++free_count; };

    const std::string key = "progressive_destructor_cleanup";
    constexpr size_t kObjSize = 2 * 1024 * 1024;
    constexpr size_t kChunkSize = 64 * 1024;
    PutPattern(key, kObjSize);

    void* buf = client_buf_alloc_->allocate(kObjSize);
    ASSERT_NE(buf, nullptr);

    {
        auto handle =
            test_client_->ProgressiveGet(key, buf, kObjSize, kChunkSize);
        ASSERT_TRUE(handle.has_value());
        EXPECT_EQ(alloc_count.load(), 1);
    }

    EXPECT_EQ(free_count.load(), 1);
    client_buf_alloc_->deallocate(buf, kObjSize);
    test_client_->Remove(key);
}

// ============================================================================
// Comprehensive benchmark: small / medium / large objects x chunk sizes
//
// Uses malloc + RegisterLocalMemory for large buffers (bypasses cachelib
// slab size limits). For each object size tier, measures:
//   - Regular Get (baseline)
//   - ProgressiveGet wait_all
//   - ProgressiveGet sequential wait_chunk
//   - First-chunk latency
// ============================================================================

TEST_F(ProgressiveGetTest, ComprehensiveBenchmark) {
    struct ObjTier {
        const char* label;
        size_t obj_size;
        std::vector<size_t> chunk_sizes;
        int warmup;
        int iters;
    };

    const std::vector<ObjTier> tiers = {
        {"SMALL(4MB)",
         4ULL * 1024 * 1024,
         {64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024},
         2,
         10},
        {"MEDIUM(64MB)",
         64ULL * 1024 * 1024,
         {256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024},
         2,
         5},
        {"LARGE(1GB)",
         1ULL * 1024 * 1024 * 1024,
         {1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024},
         1,
         3},
    };

    auto med = [](std::vector<double>& v) -> double {
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return n % 2 == 0 ? (v[n / 2 - 1] + v[n / 2]) / 2.0 : v[n / 2];
    };

    auto fmt_size = [](size_t bytes) -> std::string {
        if (bytes >= 1024 * 1024 * 1024)
            return std::to_string(bytes / (1024 * 1024 * 1024)) + "GB";
        if (bytes >= 1024 * 1024)
            return std::to_string(bytes / (1024 * 1024)) + "MB";
        return std::to_string(bytes / 1024) + "KB";
    };

    // Allocate a single large buffer for the biggest tier and register it.
    // Reuse for all tiers (smaller objects simply use a prefix).
    size_t max_obj = std::max_element(tiers.begin(), tiers.end(),
                                      [](const ObjTier& a, const ObjTier& b) {
                                          return a.obj_size < b.obj_size;
                                      })
                         ->obj_size;

    void* bench_buf = std::malloc(max_obj);
    ASSERT_NE(bench_buf, nullptr) << "malloc " << fmt_size(max_obj);

    auto reg = test_client_->RegisterLocalMemory(bench_buf, max_obj, "cpu:0",
                                                 false, false);
    ASSERT_TRUE(reg.has_value()) << toString(reg.error());

    LOG(INFO) << "=== Comprehensive ProgressiveGet Benchmark ===";
    LOG(INFO) << "protocol=" << FLAGS_protocol;

    for (const auto& tier : tiers) {
        LOG(INFO) << "";
        LOG(INFO) << "--- " << tier.label
                  << " (obj_size=" << fmt_size(tier.obj_size) << ") ---";

        // Put object using the registered bench_buf
        const std::string key = "bench_" + std::to_string(tier.obj_size);
        FillPattern(bench_buf, tier.obj_size, key);
        {
            std::vector<Slice> slices{{bench_buf, tier.obj_size}};
            ReplicateConfig cfg;
            cfg.replica_num = 1;
            auto r = test_client_->Put(key, slices, cfg);
            ASSERT_TRUE(r.has_value()) << toString(r.error());
        }

        // --- Baseline: Regular Get ---
        std::vector<double> get_us;
        for (int iter = 0; iter < tier.warmup + tier.iters; ++iter) {
            std::vector<Slice> slices{{bench_buf, tier.obj_size}};
            auto t0 = std::chrono::steady_clock::now();
            auto r = test_client_->Get(key, slices);
            auto t1 = std::chrono::steady_clock::now();
            ASSERT_TRUE(r.has_value()) << toString(r.error());
            if (iter >= tier.warmup) {
                get_us.push_back(
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 -
                                                                          t0)
                        .count());
            }
        }
        double get_med = med(get_us);
        LOG(INFO) << "  baseline Get: " << get_med << " us";

        // --- Per chunk-size benchmarks ---
        for (size_t chunk_size : tier.chunk_sizes) {
            size_t num_chunks = (tier.obj_size + chunk_size - 1) / chunk_size;
            std::vector<double> wa_us, sc_us, fc_us;

            for (int iter = 0; iter < tier.warmup + tier.iters; ++iter) {
                // wait_all
                {
                    auto t0 = std::chrono::steady_clock::now();
                    auto h = test_client_->ProgressiveGet(
                        key, bench_buf, tier.obj_size, chunk_size);
                    ASSERT_TRUE(h.has_value());
                    ErrorCode ec = h->wait_all();
                    auto t1 = std::chrono::steady_clock::now();
                    ASSERT_EQ(ec, ErrorCode::OK);
                    if (iter >= tier.warmup)
                        wa_us.push_back(std::chrono::duration_cast<
                                            std::chrono::microseconds>(t1 - t0)
                                            .count());
                }
                // sequential wait_chunk
                {
                    auto t0 = std::chrono::steady_clock::now();
                    auto h = test_client_->ProgressiveGet(
                        key, bench_buf, tier.obj_size, chunk_size);
                    ASSERT_TRUE(h.has_value());
                    for (size_t c = 0; c < h->num_chunks(); ++c) {
                        ErrorCode ec = h->wait_chunk(c);
                        ASSERT_EQ(ec, ErrorCode::OK);
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    if (iter >= tier.warmup)
                        sc_us.push_back(std::chrono::duration_cast<
                                            std::chrono::microseconds>(t1 - t0)
                                            .count());
                }
                // first-chunk latency
                {
                    auto t0 = std::chrono::steady_clock::now();
                    auto h = test_client_->ProgressiveGet(
                        key, bench_buf, tier.obj_size, chunk_size);
                    ASSERT_TRUE(h.has_value());
                    ErrorCode ec0 = h->wait_chunk(0);
                    auto t1 = std::chrono::steady_clock::now();
                    ASSERT_EQ(ec0, ErrorCode::OK);
                    h->wait_all();
                    if (iter >= tier.warmup)
                        fc_us.push_back(std::chrono::duration_cast<
                                            std::chrono::microseconds>(t1 - t0)
                                            .count());
                }
            }

            double wa_med = med(wa_us);
            double sc_med = med(sc_us);
            double fc_med = med(fc_us);

            LOG(INFO) << "  chunk=" << fmt_size(chunk_size)
                      << "  chunks=" << num_chunks << "  | wait_all=" << wa_med
                      << "us"
                      << " ("
                      << (get_med > 0 ? (wa_med - get_med) / get_med * 100.0
                                      : 0.0)
                      << "%)"
                      << "  | seq_chunk=" << sc_med << "us"
                      << " ("
                      << (get_med > 0 ? (sc_med - get_med) / get_med * 100.0
                                      : 0.0)
                      << "%)"
                      << "  | first_chunk=" << fc_med << "us"
                      << " (" << (get_med > 0 ? fc_med / get_med * 100.0 : 0.0)
                      << "% of Get)";
        }

        test_client_->Remove(key);
    }

    std::free(bench_buf);
    LOG(INFO) << "=== Comprehensive Benchmark complete ===";
}

}  // namespace testing
}  // namespace mooncake

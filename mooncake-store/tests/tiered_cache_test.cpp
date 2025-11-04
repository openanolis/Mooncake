#include "gtest/gtest.h"
#include "tiered_cache/tiered_backend.h"
#include "tiered_cache/vram_cache_tier.h"
#include "config_helper.h"
#include <algorithm>
#include <json/value.h>
#include <cuda_runtime.h>

static bool parseJsonString(const std::string &json_str, Json::Value &value,
                            std::string *error_msg = nullptr) {
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    std::string errs;

    bool success = reader->parse(
        json_str.data(), json_str.data() + json_str.size(), &value, &errs);
    if (!success && error_msg) {
        *error_msg = errs;
    }
    return success;
}

namespace mooncake {
// Forward declarations to avoid dependency on transfer_engine.h for this test
class TransferEngine;

// Test fixture for setting up tiered cache components
class TieredCacheTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("TieredBackendTest");
        FLAGS_logtostderr = 1;
        backend = std::make_unique<TieredBackend>();

        // 1. Create a JSON configuration for the tiers.
        // This JSON defines two DRAM tiers with different capacities and priorities.
        const std::string json_config_str = R"({
            "tiers": [
                {
                    "id": 1,
                    "type": "DRAM",
                    "capacity": 1024,
                    "tags": ["numa-0", "fast"],
                    "priority": 0,
                    "numa": 0,
                },
                {
                    "id": 2,
                    "type": "DRAM",
                    "capacity": 2048,
                    "tags": ["slow"],
                    "priority": 1,
                }
            ]
        })";

        Json::Value config;
        ASSERT_TRUE(parseJsonString(json_config_str, config));

        // 2. Initialize the TieredBackend.
        // We pass nullptr for the TransferEngine as it's not needed for these unit tests.
        ASSERT_TRUE(backend->Init(config, nullptr));
    }

    void TearDown() override { google::ShutdownGoogleLogging(); }

    std::unique_ptr<TieredBackend> backend;
};

// Test case for DataCopier (accessed via TieredBackend)
TEST_F(TieredCacheTest, DataCopier) {
    const auto& copier = backend->GetDataCopier();

    // Allocate source and destination memory on DRAM
    const size_t size = 128;
    std::vector<char> src_data(size, 'a');
    std::vector<char> dst_data(size, 0);

    DataSource src{src_data.data(), size, MemoryType::DRAM};

    // Perform the copy
    ASSERT_TRUE(copier.Copy(src, MemoryType::DRAM, dst_data.data()));

    // Verify the content
    ASSERT_EQ(src_data, dst_data);
    LOG(INFO) << "Test DataCopier Successfully";
}

// Test case for TieredBackend basic operations.
// Covers Put, Get, FindKey, MoveData, and Delete operations.
TEST_F(TieredCacheTest, TieredBackendBasicOps) {
    const std::string key = "backend_key";
    const std::string value = "backend_value";
    DataSource src{value.data(), value.size(), MemoryType::DRAM};

    // Test Put to a specific tier.
    ASSERT_TRUE(backend->Put(key, 1, src));
    auto location = backend->FindKey(key);
    ASSERT_TRUE(location.has_value());
    ASSERT_EQ(location.value(), 1);
    LOG(INFO) << "Test Put Successfully";

    // Test Get.
    void* data_ptr = nullptr;
    size_t read_size = 0;
    ASSERT_TRUE(backend->Get(key, data_ptr, read_size));
    ASSERT_NE(data_ptr, nullptr);
    ASSERT_EQ(read_size, value.size());
    ASSERT_EQ(std::string(static_cast<char*>(data_ptr), read_size), value);
    LOG(INFO) << "Test Get Successfully";

    // Test MoveData between tiers.
    ASSERT_TRUE(backend->MoveData(key, 1, 2));
    location = backend->FindKey(key);
    ASSERT_TRUE(location.has_value());
    ASSERT_EQ(location.value(), 2);
    LOG(INFO) << "Moved data from tier 1 to tier 2";

    // Verify data is now in tier 2 by getting it again.
    void* moved_data_ptr = nullptr;
    size_t moved_read_size = 0;
    ASSERT_TRUE(backend->Get(key, moved_data_ptr, moved_read_size));
    ASSERT_EQ(std::string(static_cast<char*>(moved_data_ptr), moved_read_size), value);
    LOG(INFO) << "Test MoveData Successfully";

    // Verify data is no longer in tier 1.
    ASSERT_NE(backend->FindKey(key).value_or(0), 1);

    // Test Delete.
    ASSERT_TRUE(backend->Delete(key));
    ASSERT_FALSE(backend->FindKey(key).has_value());
    LOG(INFO) << "Test Delete Successfully";
}

// Test case for TieredBackend edge cases.
// Covers operations on non-existent keys and full tiers.
TEST_F(TieredCacheTest, TieredBackendEdgeCases) {
    const std::string non_existent_key = "non_existent_key";
    void* data_ptr = nullptr;
    size_t read_size = 0;

    // Operations on a non-existent key should fail gracefully.
    ASSERT_FALSE(backend->Get(non_existent_key, data_ptr, read_size));
    ASSERT_FALSE(backend->MoveData(non_existent_key, 1, 2));
    ASSERT_FALSE(backend->Delete(non_existent_key));
    ASSERT_FALSE(backend->FindKey(non_existent_key).has_value());
    LOG(INFO) << "Test operations on non-existent key successfully.";

    // Test putting data that exceeds the capacity of the first tier.
    // The first tier has a capacity of 1024 bytes.
    const std::string large_value(1025, 'b');
    const std::string large_key = "large_key";
    DataSource large_src{large_value.data(), large_value.size(), MemoryType::DRAM};

    // This should fail because the data is larger than the tier's capacity.
    ASSERT_FALSE(backend->Put(large_key, 1, large_src));
    LOG(INFO) << "Test Put with value larger than tier capacity successfully.";

    // Test filling up a tier and then attempting to add more.
    const std::string value1(512, 'c');
    DataSource src1{value1.data(), value1.size(), MemoryType::DRAM};
    ASSERT_TRUE(backend->Put("key1", 1, src1));

    const std::string value2(512, 'd');
    DataSource src2{value2.data(), value2.size(), MemoryType::DRAM};
    ASSERT_TRUE(backend->Put("key2", 1, src2));
    // Tier 1 is now full with OffsetBufferAllocator

    const std::string value3(10, 'e');
    DataSource src3{value3.data(), value3.size(), MemoryType::DRAM};
    // This Put should fail because tier 1 is full.
    ASSERT_FALSE(backend->Put("key3", 1, src3));
    LOG(INFO) << "Test Put to a full tier successfully.";
}

//
// Test Fixture for VRAM tests (DRAM + 2 VRAM Tiers)
//
class TieredCacheVramTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("TieredBackendVramTest");
        FLAGS_logtostderr = 1;

        // Check for CUDA device availability before proceeding
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            // Don't fail the test, skip it instead
            GTEST_SKIP() << "Skipping VRAM test: No CUDA devices found or CUDA error: "
                       << (err == cudaSuccess ? "No devices" : cudaGetErrorString(err));
        }

        backend = std::make_unique<TieredBackend>();

        // 1. Create a JSON config with DRAM and *two* VRAM tiers.
        //    VRAM (id: 3) has priority 0 (highest)
        //    VRAM (id: 4) has priority 1
        //    DRAM (id: 1) has priority 2
        const std::string json_config_str = R"({
            "tiers": [
                {
                    "id": 1,
                    "type": "DRAM",
                    "capacity": 1024,
                    "priority": 2
                },
                {
                    "id": 3,
                    "type": "VRAM",
                    "capacity": 2048,
                    "priority": 0,
                    "gpu_id": 0
                },
                {
                    "id": 4,
                    "type": "VRAM",
                    "capacity": 2048,
                    "priority": 1,
                    "gpu_id": 1
                }
            ]
        })";

        Json::Value config;
        ASSERT_TRUE(parseJsonString(json_config_str, config));

        // 2. Initialize the TieredBackend.
        // This will be a valid test failure if cudaMalloc fails (e.g., OOM).
        ASSERT_TRUE(backend->Init(config, nullptr));
    }

    void TearDown() override { google::ShutdownGoogleLogging(); }

    std::unique_ptr<TieredBackend> backend;
};

// Test case for VRAM Tier operations
// Covers Put (DRAM->VRAM), Move (VRAM->VRAM), Move (VRAM->DRAM), Get, and Delete.
TEST_F(TieredCacheVramTest, VramBasicOps) {
    const std::string key = "vram_key";
    const std::string value(256, 'v'); // Use 256 bytes of 'v'

    // Source data is on DRAM
    DataSource src{value.data(), value.size(), MemoryType::DRAM};

    // --- 1. Test Put to VRAM tier (ID 3) ---
    ASSERT_TRUE(backend->Put(key, 3, src));
    auto location = backend->FindKey(key);
    ASSERT_TRUE(location.has_value());
    ASSERT_EQ(location.value(), 3);
    LOG(INFO) << "Test Put to VRAM (3) Successfully";

    // --- 2. Test Get from VRAM (ID 3) ---
    void* data_ptr = nullptr;
    size_t read_size = 0;
    ASSERT_TRUE(backend->Get(key, data_ptr, read_size));
    ASSERT_NE(data_ptr, nullptr);
    ASSERT_EQ(read_size, value.size());

    // Verify content. Must copy from VRAM (device) to DRAM (host) for checking.
    std::vector<char> host_buffer(read_size);
    cudaError_t cpy_err = cudaMemcpy(host_buffer.data(), data_ptr, read_size, cudaMemcpyDeviceToHost);
    ASSERT_EQ(cpy_err, cudaSuccess) << "Failed to copy data from VRAM 3 to host for verification.";

    ASSERT_EQ(std::string(host_buffer.begin(), host_buffer.end()), value);
    LOG(INFO) << "Test Get from VRAM (3) Successfully";

    // --- 3. Test MoveData from VRAM (3) to VRAM (4) ---
    // This relies on the DataCopier change (cudaMemcpyDeviceToDevice)
    ASSERT_TRUE(backend->MoveData(key, 3, 4));
    location = backend->FindKey(key);
    ASSERT_TRUE(location.has_value());
    ASSERT_EQ(location.value(), 4);
    LOG(INFO) << "Moved data from VRAM (3) to VRAM (4)";

    // --- 4. Test Get from VRAM (ID 4) ---
    void* vram4_data_ptr = nullptr;
    size_t vram4_read_size = 0;
    ASSERT_TRUE(backend->Get(key, vram4_data_ptr, vram4_read_size));
    ASSERT_NE(vram4_data_ptr, nullptr);
    ASSERT_EQ(vram4_read_size, value.size());

    // Verify content again from the new VRAM location
    std::vector<char> vram4_host_buffer(vram4_read_size);
    cpy_err = cudaMemcpy(vram4_host_buffer.data(), vram4_data_ptr, vram4_read_size, cudaMemcpyDeviceToHost);
    ASSERT_EQ(cpy_err, cudaSuccess) << "Failed to copy data from VRAM 4 to host for verification.";

    ASSERT_EQ(std::string(vram4_host_buffer.begin(), vram4_host_buffer.end()), value);
    LOG(INFO) << "Test Get from VRAM (4) after VRAM-VRAM move Successfully";

    // --- 5. Test MoveData from VRAM (4) to DRAM (1) ---
    ASSERT_TRUE(backend->MoveData(key, 4, 1));
    location = backend->FindKey(key);
    ASSERT_TRUE(location.has_value());
    ASSERT_EQ(location.value(), 1);
    LOG(INFO) << "Moved data from VRAM (4) to DRAM (1)";

    // --- 6. Get again, verifying data is now in DRAM tier 1 ---
    void* moved_data_ptr = nullptr;
    size_t moved_read_size = 0;
    ASSERT_TRUE(backend->Get(key, moved_data_ptr, moved_read_size));
    // Pointer is now to DRAM, so we can cast directly.
    ASSERT_EQ(std::string(static_cast<char*>(moved_data_ptr), moved_read_size), value);
    LOG(INFO) << "Test MoveData VRAM->DRAM Successfully";

    // --- 7. Test Delete ---
    ASSERT_TRUE(backend->Delete(key));
    ASSERT_FALSE(backend->FindKey(key).has_value());
    LOG(INFO) << "Test Delete after VRAM->DRAM move Successfully";
}

}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
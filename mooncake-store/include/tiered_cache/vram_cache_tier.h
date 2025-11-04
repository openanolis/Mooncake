#pragma once

#include "tiered_cache/cache_tier.h"
#include "allocator.h"
#include "transfer_engine.h"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>

namespace mooncake {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            LOG(ERROR) << "CUDA Error in " << __FILE__ << ":" << __LINE__ \
                       << ": " << cudaGetErrorString(err); \
            return false; \
        } \
    } while (0)

#define CUDA_CHECK_DTOR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            LOG(ERROR) << "CUDA Error in VramCacheTier Destructor: " \
                       << cudaGetErrorString(err); \
        } \
    } while (0)

class VramCacheTier : public CacheTier {
public:
    VramCacheTier(uint64_t tier_id, size_t capacity,
                  const std::vector<std::string>& tags,
                  int gpu_id,
                  BufferAllocatorType allocator_type = BufferAllocatorType::OFFSET);
    ~VramCacheTier() override;

    bool Init(TieredBackend* backend, TransferEngine* engine) override;
    bool Get(const std::string& key, void*& data, size_t& size) override;
    bool Put(const std::string& key, const DataSource& source) override;
    bool Delete(const std::string& key) override;
    bool Contains(const std::string& key) const override;
    DataSource AsDataSource(const std::string& key) override;

    uint64_t GetTierId() const override { return tier_id_; }
    size_t GetCapacity() const override { return capacity_; }
    size_t GetUsage() const override;
    const std::vector<std::string>& GetTags() const override { return tags_; }
    MemoryType GetMemoryType() const override { return MemoryType::VRAM; }
    int GetGpuId() const { return gpu_id_; }

private:
    uint64_t tier_id_;
    size_t capacity_;
    size_t current_usage_;
    std::vector<std::string> tags_;
    int gpu_id_;
    BufferAllocatorType allocator_type_;
    std::shared_ptr<BufferAllocatorBase> allocator_;
    std::unordered_map<std::string, std::unique_ptr<AllocatedBuffer>> cache_entries_;
    TransferEngine* engine_ = nullptr;
    std::unique_ptr<char, std::function<void(char*)>> memory_buffer_{nullptr, [](char*){}};
};

} // namespace mooncake
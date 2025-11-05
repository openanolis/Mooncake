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

class DramCacheTier : public CacheTier {
public:
    DramCacheTier(uint64_t tier_id, size_t capacity,
                  const std::vector<std::string>& tags,
                  std::optional<int> numa_node = std::nullopt,
                  BufferAllocatorType allocator_type = BufferAllocatorType::OFFSET);
    ~DramCacheTier() override;

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
    MemoryType GetMemoryType() const override { return MemoryType::DRAM; }

private:
    uint64_t tier_id_;
    size_t capacity_;
    size_t current_usage_;
    std::vector<std::string> tags_;
    std::optional<int> numa_node_;
    BufferAllocatorType allocator_type_;
    std::shared_ptr<BufferAllocatorBase> allocator_;
    std::unordered_map<std::string, std::unique_ptr<AllocatedBuffer>> cache_entries_;
    TransferEngine* engine_;
    std::unique_ptr<char[], void(*)(char*)> memory_buffer_{nullptr, [](char* p){ delete[] p; }};
};

} // namespace mooncake
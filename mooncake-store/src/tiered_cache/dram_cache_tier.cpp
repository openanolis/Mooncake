#include "tiered_cache/dram_cache_tier.h"
#include "tiered_cache/tiered_backend.h"
#include "tiered_cache/copier_registry.h"
#include "transfer_engine.h"

#include <glog/logging.h>
#include <numa.h>

namespace mooncake {

DramCacheTier::DramCacheTier(uint64_t tier_id, size_t capacity,
                           const std::vector<std::string>& tags,
                           std::optional<int> numa_node,
                           BufferAllocatorType allocator_type)
    : tier_id_(tier_id), capacity_(capacity), tags_(tags), numa_node_(numa_node), allocator_type_(allocator_type) {}

DramCacheTier::~DramCacheTier() {
    cache_entries_.clear();
    allocator_.reset();

    if (engine_ != nullptr && memory_buffer_ != nullptr) {
        LOG(INFO) << "unregistering memory for DramCacheTier " << tier_id_;
        int rc = engine_->unregisterLocalMemory(memory_buffer_.get());
        if (rc != 0) {
            LOG(ERROR) << "Failed to unregister memory for DramCacheTier " << tier_id_
            << ", engine ret is " << rc;
        }
    }
}

bool DramCacheTier::Init(TieredBackend* backend, TransferEngine* engine) {
    int node = -1;
    std::string location;

    backend_ = backend;
    if (engine != nullptr)
        engine_ = engine;

    // Allocate a contiguous memory block.
    if (numa_node_.has_value()) {
        if (numa_available() < 0) {
            LOG(ERROR) << "NUMA not available on this system.";
            return false;
        }
        node = numa_node_.value();
        if (node < 0 || node > numa_max_node()) {
            LOG(ERROR) << "Invalid NUMA node " << node;
            return false;
        }
        char* mem_ptr = static_cast<char*>(numa_alloc_onnode(capacity_, node));
        if (!mem_ptr) {
            LOG(ERROR) << "Failed to allocate " << capacity_ << " bytes from NUMA node " << node
                       << " for DramCacheTier " << tier_id_;
            return false;
        }
        memory_buffer_ = std::unique_ptr<char[], void(*)(char*)>(mem_ptr, [](char* p){ numa_free(p, 0); });
        LOG(INFO) << "Allocated " << capacity_ << " bytes from NUMA node " << node
                  << " for DramCacheTier " << tier_id_;
    } else {
        try {
            memory_buffer_ = std::unique_ptr<char[], void(*)(char*)>(new char[capacity_], [](char* p){ delete[] p; });
        } catch (const std::bad_alloc& e) {
            LOG(ERROR) << "Failed to allocate " << capacity_ << " bytes for DramCacheTier "
                       << tier_id_ << ": " << e.what();
            return false;
        }
        LOG(INFO) << "Allocated " << capacity_ << " bytes for DramCacheTier " << tier_id_;
    }
    char* mem_ptr = memory_buffer_.get();

    // Register this newly allocated memory with the TransferEngine.
    if (engine_) {
        if (numa_node_.has_value()) {
            location = "cpu:" + std::to_string(node);
        } else {
            location = "*";
        }
        int rc = engine_->registerLocalMemory(mem_ptr, capacity_, location);
        if (rc != 0) {
            LOG(ERROR) << "Failed to register memory with TransferEngine for DramCacheTier "
                    << tier_id_ << ", engine ret is " << rc;
            return false;
        } else {
            LOG(INFO) << "registered memory with TransferEngine for DramCacheTier "
                    << tier_id_ << " at " << static_cast<void*>(mem_ptr);
        }
    }

    // Use the address of this registered block as the base_address for the allocator.
    const uintptr_t base_address = reinterpret_cast<uintptr_t>(mem_ptr);
    std::string segment_name = "dram_tier_" + std::to_string(tier_id_);

    switch (allocator_type_) {
        case BufferAllocatorType::OFFSET:
            allocator_ = std::make_shared<OffsetBufferAllocator>(
                segment_name, base_address, capacity_, segment_name);
            break;
        case BufferAllocatorType::CACHELIB:
            allocator_ = std::make_shared<CachelibBufferAllocator>(
                segment_name, base_address, capacity_, segment_name);
            break;
        default:
            LOG(ERROR) << "Unsupported allocator type for DramCacheTier";
            if (engine_) {
                engine_->unregisterLocalMemory(mem_ptr, capacity_);
            }
            return false;
    }

    LOG(INFO) << "DramCacheTier " << tier_id_ << " initialized and registered " << capacity_
              << " bytes at base address 0x" << std::hex << base_address;
    return true;
}

bool DramCacheTier::Get(const std::string& key, void*& data, size_t& size) {
    auto it = cache_entries_.find(key);
    if (it == cache_entries_.end()) {
        return false;
    }
    AllocatedBuffer* buf_handle = it->second.get();
    data = buf_handle->data();
    size = buf_handle->get_descriptor().size_;
    return true;
}

bool DramCacheTier::Put(const std::string& key, const DataSource& source) {
    if (current_usage_ + source.size > capacity_) {
        return false;
    }

    if (Contains(key)) {
        Delete(key);
    }

    auto buffer = allocator_->allocate(source.size);
    if (!buffer) {
        LOG(WARNING) << "DramCacheTier " << tier_id_ << " failed to allocate " << source.size << " bytes.";
        return false;
    }

    if (!backend_->GetDataCopier().Copy(source, GetMemoryType(), buffer->data())) {
        LOG(ERROR) << "Failed to copy data from " << static_cast<int>(source.type)
                   << " to DRAM for key " << key;
        return false;
    }

    const size_t allocated_size = buffer->get_descriptor().size_;
    cache_entries_[key] = std::move(buffer);
    current_usage_ += allocated_size;
    return true;
}

bool DramCacheTier::Delete(const std::string& key) {
    auto it = cache_entries_.find(key);
    if (it != cache_entries_.end()) {
        const size_t allocated_size = it->second->get_descriptor().size_;
        current_usage_ -= allocated_size;
        cache_entries_.erase(it);
        return true;
    }
    return false;
}

bool DramCacheTier::Contains(const std::string& key) const {
    return cache_entries_.count(key);
}

DataSource DramCacheTier::AsDataSource(const std::string& key) {
    auto it = cache_entries_.find(key);
    if (it == cache_entries_.end()) {
        return {nullptr, 0, MemoryType::DRAM};
    }
    return {it->second->data(), it->second->get_descriptor().size_, MemoryType::DRAM};
}

size_t DramCacheTier::GetUsage() const {
    return current_usage_;
}

} // namespace mooncake
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace mooncake {

class RegisteredPinnedMemoryManager;

class RegisteredPinnedRegion {
   public:
    RegisteredPinnedRegion(const RegisteredPinnedRegion&) = delete;
    RegisteredPinnedRegion& operator=(const RegisteredPinnedRegion&) = delete;
    ~RegisteredPinnedRegion();

    void* addr() const { return addr_; }
    size_t size() const { return size_; }
    const std::string& owner() const { return owner_; }

   private:
    friend class RegisteredPinnedMemoryManager;

    RegisteredPinnedRegion(void* addr, size_t size, std::string owner)
        : addr_(addr), size_(size), owner_(std::move(owner)) {}

    void* addr_ = nullptr;
    size_t size_ = 0;
    std::string owner_;
};

class RegisteredPinnedMemoryManager {
   public:
    static RegisteredPinnedMemoryManager& instance();

    bool enabled() const { return enabled_; }
    uint64_t limit_bytes() const { return limit_bytes_; }

    uint64_t pinned_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pinned_bytes_;
    }

    std::shared_ptr<RegisteredPinnedRegion> try_pin(void* addr, size_t size,
                                                    const std::string& owner);

   private:
    friend class RegisteredPinnedRegion;

    struct RegionKey {
        void* addr = nullptr;
        size_t size = 0;

        bool operator==(const RegionKey& other) const {
            return addr == other.addr && size == other.size;
        }
    };

    struct RegionKeyHash {
        size_t operator()(const RegionKey& key) const {
            return std::hash<void*>{}(key.addr) ^
                   (std::hash<size_t>{}(key.size) << 1);
        }
    };

    RegisteredPinnedMemoryManager();

    void release(RegisteredPinnedRegion* region);

    const bool enabled_;
    const uint64_t limit_bytes_;

    mutable std::mutex mutex_;
    uint64_t pinned_bytes_ = 0;
    std::unordered_map<RegionKey, std::weak_ptr<RegisteredPinnedRegion>,
                       RegionKeyHash>
        regions_;
};

}  // namespace mooncake

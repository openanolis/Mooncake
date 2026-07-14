#include "registered_pinned_memory.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <limits>

#include <glog/logging.h>

#if defined(USE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace mooncake {
namespace {

bool ParsePinnedMemoryEnabled() {
    const char* value = std::getenv("MC_STORE_PIN_MEMORY");
    if (!value) return true;

    std::string normalized(value);
    std::transform(
        normalized.begin(), normalized.end(), normalized.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return !(normalized == "0" || normalized == "false" ||
             normalized == "off" || normalized == "no");
}

uint64_t ParsePinnedMemoryLimit() {
    const char* value = std::getenv("MC_STORE_PIN_MEMORY_MAX_BYTES");
    if (!value || value[0] == '\0') return 0;
    const char* number = value;
    while (std::isspace(static_cast<unsigned char>(*number))) {
        ++number;
    }
    if (*number == '-') {
        LOG(WARNING) << "Invalid MC_STORE_PIN_MEMORY_MAX_BYTES='" << value
                     << "', treating it as unlimited";
        return 0;
    }

    char* end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(number, &end, 10);
    if (end == number || (end && *end != '\0') || errno == ERANGE ||
        parsed > std::numeric_limits<uint64_t>::max()) {
        LOG(WARNING) << "Invalid MC_STORE_PIN_MEMORY_MAX_BYTES='" << value
                     << "', treating it as unlimited";
        return 0;
    }
    return static_cast<uint64_t>(parsed);
}

}  // namespace

RegisteredPinnedRegion::~RegisteredPinnedRegion() {
    RegisteredPinnedMemoryManager::instance().release(this);
}

RegisteredPinnedMemoryManager& RegisteredPinnedMemoryManager::instance() {
    static RegisteredPinnedMemoryManager manager;
    return manager;
}

RegisteredPinnedMemoryManager::RegisteredPinnedMemoryManager()
    : enabled_(ParsePinnedMemoryEnabled()),
      limit_bytes_(ParsePinnedMemoryLimit()) {
#if defined(USE_CUDA)
    LOG(INFO) << "Store registered pinned memory is "
              << (enabled_ ? "enabled" : "disabled")
              << ", max_bytes=" << limit_bytes_;
#else
    if (enabled_) {
        LOG(INFO) << "Store registered pinned memory requested but this build "
                     "has no CUDA runtime support";
    }
#endif
}

std::shared_ptr<RegisteredPinnedRegion> RegisteredPinnedMemoryManager::try_pin(
    void* addr, size_t size, const std::string& owner) {
    if (!addr || size == 0 || !enabled_) return nullptr;

#if !defined(USE_CUDA)
    (void)owner;
    return nullptr;
#else
    std::lock_guard<std::mutex> lock(mutex_);
    RegionKey key{addr, size};
    auto existing = regions_.find(key);
    if (existing != regions_.end()) {
        if (auto region = existing->second.lock()) return region;
        regions_.erase(existing);
    }

    const auto start = reinterpret_cast<uintptr_t>(addr);
    const auto end = start + size;
    if (end < start) {
        LOG(WARNING) << "Skip cudaHostRegister for " << owner
                     << ": address range overflow, size=" << size;
        return nullptr;
    }

    for (auto it = regions_.begin(); it != regions_.end();) {
        auto region = it->second.lock();
        if (!region) {
            it = regions_.erase(it);
            continue;
        }

        const auto region_start = reinterpret_cast<uintptr_t>(region->addr());
        const auto region_end = region_start + region->size();
        if (start >= region_start && end <= region_end) {
            return region;
        }
        const bool overlaps = start < region_end && end > region_start;
        if (overlaps) {
            LOG(WARNING) << "Skip cudaHostRegister for " << owner
                         << ": overlaps an existing pinned region, size="
                         << size;
            return nullptr;
        }
        ++it;
    }

    if (limit_bytes_ > 0) {
        if (size > limit_bytes_ || pinned_bytes_ > limit_bytes_ - size) {
            LOG(WARNING) << "Skip cudaHostRegister for " << owner
                         << ": quota exceeded, requested=" << size
                         << ", pinned=" << pinned_bytes_
                         << ", limit=" << limit_bytes_;
            return nullptr;
        }
    }

    cudaError_t err = cudaHostRegister(addr, size, cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        LOG(WARNING) << "cudaHostRegister failed for " << owner
                     << ", size=" << size
                     << ", error=" << cudaGetErrorString(err)
                     << ". Continue with pageable host memory.";
        cudaGetLastError();
        return nullptr;
    }

    auto region = std::shared_ptr<RegisteredPinnedRegion>(
        new RegisteredPinnedRegion(addr, size, owner));
    regions_[key] = region;
    pinned_bytes_ += size;
    LOG(INFO) << "cudaHostRegister succeeded for " << owner << ", size=" << size
              << ", pinned=" << pinned_bytes_ << ", limit=" << limit_bytes_;
    return region;
#endif
}

void RegisteredPinnedMemoryManager::release(RegisteredPinnedRegion* region) {
    if (!region || !region->addr_ || region->size_ == 0) return;

    std::lock_guard<std::mutex> lock(mutex_);
    RegionKey key{region->addr_, region->size_};
    auto it = regions_.find(key);

#if defined(USE_CUDA)
    cudaError_t err = cudaHostUnregister(region->addr_);
    if (err != cudaSuccess) {
        LOG(WARNING) << "cudaHostUnregister failed for " << region->owner_
                     << ", size=" << region->size_
                     << ", error=" << cudaGetErrorString(err);
        cudaGetLastError();
        return;
    }
#endif

    if (it != regions_.end()) regions_.erase(it);
    if (pinned_bytes_ >= region->size_) {
        pinned_bytes_ -= region->size_;
    } else {
        pinned_bytes_ = 0;
    }
}

}  // namespace mooncake

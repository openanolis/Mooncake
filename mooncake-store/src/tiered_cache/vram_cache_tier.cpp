#include "tiered_cache/vram_cache_tier.h"
#include "tiered_cache/tiered_backend.h"
#include "tiered_cache/copier_registry.h"
#include "transfer_engine.h"

#include <glog/logging.h>
#include <cuda_runtime.h>

namespace mooncake {

static bool CopyVramToDram(const DataSource& src, void* dest) {
    try {
        CUDA_CHECK(cudaMemcpy(dest, src.ptr, src.size, cudaMemcpyDeviceToHost));
        return true;
    } catch (const std::runtime_error& e) {
        LOG(ERROR) << e.what();
        return false;
    }
}

static bool CopyDramToVram(const DataSource& src, void* dest) {
    try {
        CUDA_CHECK(cudaMemcpy(dest, src.ptr, src.size, cudaMemcpyHostToDevice));
        return true;
    } catch (const std::runtime_error& e) {
        LOG(ERROR) << e.what();
        return false;
    }
}

static bool CopyVramToVram(const DataSource& src, void* dest) {
    try {
        CUDA_CHECK(cudaMemcpy(dest, src.ptr, src.size, cudaMemcpyDeviceToDevice));
        return true;
    } catch (const std::runtime_error& e) {
        LOG(ERROR) << e.what();
        return false;
    }
}

// Register the copier function
static CopierRegistrar vram_registrar = [] {
    CopierRegistrar reg(MemoryType::VRAM, CopyVramToDram, CopyDramToVram);
    reg.RegisterDirectPath(MemoryType::VRAM, MemoryType::VRAM, CopyVramToVram);
    return reg;
}();

VramCacheTier::VramCacheTier(uint64_t tier_id, size_t capacity,
                             const std::vector<std::string>& tags,
                             int gpu_id,
                             BufferAllocatorType allocator_type)
    : tier_id_(tier_id), capacity_(capacity), current_usage_(0), tags_(tags),
      gpu_id_(gpu_id), allocator_type_(allocator_type) {}

VramCacheTier::~VramCacheTier() {
    cache_entries_.clear();
    allocator_.reset();

    if (memory_buffer_ != nullptr && engine_ != nullptr) {
        LOG(INFO) << "unregistering memory for VramCacheTier " << tier_id_;
        int rc = engine_->unregisterLocalMemory(memory_buffer_.get());
        if (rc != 0) {
            LOG(ERROR) << "Failed to unregister memory for VramCacheTier " << tier_id_
                        << ", engine ret is " << rc;
        }
    }
}

bool VramCacheTier::Init(TieredBackend* backend, TransferEngine* engine) {
    backend_ = backend;
    if (engine != nullptr) {
        engine_ = engine;
    }

    // Set the active GPU device for CUDA operations
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device != gpu_id_) {
        CUDA_CHECK(cudaSetDevice(gpu_id_));
    }

    // Allocate a contiguous memory block on the specified GPU
    char* mem_ptr = nullptr;
    LOG(INFO) << "Attempting to allocate " << capacity_ << " bytes on GPU " << gpu_id_
              << " for VramCacheTier " << tier_id_;

    cudaError_t alloc_err = cudaMalloc(&mem_ptr, capacity_);
    if (alloc_err != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate " << capacity_ << " bytes on GPU " << gpu_id_
                   << " for VramCacheTier " << tier_id_ << ": " << cudaGetErrorString(alloc_err);
        // Reset device if we changed it
        if (current_device != gpu_id_) { cudaSetDevice(current_device); }
        return false;
    }

    // Assign to unique_ptr with a custom deleter
    int captured_gpu_id = gpu_id_;
    memory_buffer_ = std::unique_ptr<char, std::function<void(char*)>>(
        mem_ptr, [captured_gpu_id](char* p) {
            if (p) {
                // Set device context for correct freeing
                int original_device;
                cudaGetDevice(&original_device);
                cudaSetDevice(captured_gpu_id);

                LOG(INFO) << "Freeing VRAM for VramCacheTier on GPU " << captured_gpu_id;
                CUDA_CHECK_DTOR(cudaFree(p));

                // Restore original device
                if (original_device != captured_gpu_id) {
                    cudaSetDevice(original_device);
                }
            }
        });

    LOG(INFO) << "Allocated " << capacity_ << " bytes on GPU " << gpu_id_
              << " for VramCacheTier " << tier_id_ << " at address " << static_cast<void*>(mem_ptr);

    // Register this newly allocated memory with the TransferEngine.
    if (engine_) {
        std::string location = "cuda:" + std::to_string(gpu_id_);
        int rc = engine_->registerLocalMemory(mem_ptr, capacity_, location);
        if (rc != 0) {
            LOG(ERROR) << "Failed to register VRAM with TransferEngine for VramCacheTier "
                       << tier_id_ << ", engine ret is " << rc;
            if (current_device != gpu_id_) { cudaSetDevice(current_device); }
            return false;
        } else {
            LOG(INFO) << "Registered VRAM with TransferEngine for VramCacheTier "
                       << tier_id_ << " at " << static_cast<void*>(mem_ptr);
        }
    }

    // Use the address of this registered block as the base_address for the allocator.
    const uintptr_t base_address = reinterpret_cast<uintptr_t>(mem_ptr);
    std::string segment_name = "vram_tier_" + std::to_string(tier_id_);

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
            LOG(ERROR) << "Unsupported allocator type for VramCacheTier";
            if (engine_) {
                engine_->unregisterLocalMemory(mem_ptr);
            }
            if (current_device != gpu_id_) { cudaSetDevice(current_device); }
            return false;
    }

    LOG(INFO) << "VramCacheTier " << tier_id_ << " initialized and registered " << capacity_
              << " bytes on GPU " << gpu_id_ << " at base address 0x" << std::hex << base_address;

    if (current_device != gpu_id_) {
        CUDA_CHECK(cudaSetDevice(current_device));
    }
    return true;
}

bool VramCacheTier::Get(const std::string& key, void*& data, size_t& size) {
    auto it = cache_entries_.find(key);
    if (it == cache_entries_.end()) {
        return false;
    }
    AllocatedBuffer* buf_handle = it->second.get();
    data = buf_handle->data();
    size = buf_handle->get_descriptor().size_;
    return true;
}

bool VramCacheTier::Put(const std::string& key, const DataSource& source) {
    if (current_usage_ + source.size > capacity_) {
        LOG(WARNING) << "VramCacheTier " << tier_id_ << " is full. Cannot allocate "
                     << source.size << " bytes. Current usage: " << current_usage_
                     << ", Capacity: " << capacity_;
        return false;
    }

    if (Contains(key)) {
        Delete(key);
    }

    auto buffer = allocator_->allocate(source.size);
    if (!buffer) {
        LOG(WARNING) << "VramCacheTier " << tier_id_ << " failed to allocate " << source.size
                     << " bytes from its VRAM allocator.";
        return false;
    }

    if (!backend_->GetDataCopier().Copy(source, GetMemoryType(), buffer->data())) {
        LOG(ERROR) << "Failed to copy data from source (type " << static_cast<int>(source.type)
                   << ") to VRAM for key " << key;
        return false;
    }

    const size_t allocated_size = buffer->get_descriptor().size_;
    cache_entries_[key] = std::move(buffer);
    current_usage_ += allocated_size;
    return true;
}

bool VramCacheTier::Delete(const std::string& key) {
    auto it = cache_entries_.find(key);
    if (it != cache_entries_.end()) {
        const size_t allocated_size = it->second->get_descriptor().size_;
        current_usage_ -= allocated_size;
        cache_entries_.erase(it);
        return true;
    }
    return false;
}

bool VramCacheTier::Contains(const std::string& key) const {
    return cache_entries_.count(key);
}

DataSource VramCacheTier::AsDataSource(const std::string& key) {
    auto it = cache_entries_.find(key);
    if (it == cache_entries_.end()) {
        return {nullptr, 0, GetMemoryType()};
    }
    return {it->second->data(), it->second->get_descriptor().size_, GetMemoryType()};
}

size_t VramCacheTier::GetUsage() const {
    return current_usage_;
}

} // namespace mooncake
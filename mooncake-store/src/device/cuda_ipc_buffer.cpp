#include "device/cuda_ipc_buffer.h"

#include <cstring>
#include <limits>

#if defined(USE_CUDA)
#include "cuda_alike.h"
#endif

namespace mooncake {
namespace device {
namespace {

tl::expected<CudaIpcBufferHandle, ErrorCode> UnsupportedCudaIpc() {
    return tl::unexpected(ErrorCode::INVALID_PARAMS);
}

#if defined(USE_CUDA)
bool AddOverflows(uint64_t a, uint64_t b) {
    return a > std::numeric_limits<uint64_t>::max() - b;
}

void ClearCudaError() { cudaGetLastError(); }
#endif

}  // namespace

tl::expected<CudaIpcBufferHandle, ErrorCode> ExportCudaIpcBuffer(
    const void *ptr, size_t size) {
#if defined(USE_CUDA)
    static_assert(sizeof(cudaIpcMemHandle_t) == kCudaIpcHandleSize,
                  "Unexpected CUDA IPC handle size");

    if (ptr == nullptr || size == 0) return UnsupportedCudaIpc();

    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess ||
        attr.type != cudaMemoryTypeDevice || attr.devicePointer == nullptr) {
        ClearCudaError();
        return UnsupportedCudaIpc();
    }

    const auto ptr_addr = reinterpret_cast<uintptr_t>(ptr);
    const auto base_addr = reinterpret_cast<uintptr_t>(attr.devicePointer);
    if (ptr_addr < base_addr) return UnsupportedCudaIpc();

    const uint64_t offset = static_cast<uint64_t>(ptr_addr - base_addr);
    const uint64_t payload_size = static_cast<uint64_t>(size);
    if (AddOverflows(offset, payload_size)) return UnsupportedCudaIpc();

    int current_device = -1;
    cudaGetDevice(&current_device);
    if (cudaSetDevice(attr.device) != cudaSuccess) {
        ClearCudaError();
        return tl::unexpected(ErrorCode::INTERNAL_ERROR);
    }

    cudaIpcMemHandle_t ipc_handle{};
    cudaError_t ret = cudaIpcGetMemHandle(&ipc_handle, attr.devicePointer);
    if (current_device >= 0) cudaSetDevice(current_device);
    if (ret != cudaSuccess) {
        ClearCudaError();
        return UnsupportedCudaIpc();
    }

    CudaIpcBufferHandle exported;
    std::memcpy(exported.handle.data(), &ipc_handle, sizeof(ipc_handle));
    exported.offset = offset;
    exported.size = payload_size;
    exported.device_id = attr.device;
    return exported;
#else
    (void)ptr;
    (void)size;
    return UnsupportedCudaIpc();
#endif
}

CudaIpcBufferMapping::~CudaIpcBufferMapping() { Close(); }

CudaIpcBufferMapping::CudaIpcBufferMapping(
    CudaIpcBufferMapping &&other) noexcept {
    base_ = other.base_;
    ptr_ = other.ptr_;
    other.base_ = nullptr;
    other.ptr_ = nullptr;
}

CudaIpcBufferMapping &CudaIpcBufferMapping::operator=(
    CudaIpcBufferMapping &&other) noexcept {
    if (this != &other) {
        Close();
        base_ = other.base_;
        ptr_ = other.ptr_;
        other.base_ = nullptr;
        other.ptr_ = nullptr;
    }
    return *this;
}

tl::expected<CudaIpcBufferMapping, ErrorCode> CudaIpcBufferMapping::Open(
    const CudaIpcBufferHandle &handle) {
#if defined(USE_CUDA)
    static_assert(sizeof(cudaIpcMemHandle_t) == kCudaIpcHandleSize,
                  "Unexpected CUDA IPC handle size");

    if (handle.size == 0 || handle.device_id < 0 ||
        AddOverflows(handle.offset, handle.size)) {
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }

    int current_device = -1;
    cudaGetDevice(&current_device);
    if (cudaSetDevice(handle.device_id) != cudaSuccess) {
        ClearCudaError();
        return tl::unexpected(ErrorCode::INTERNAL_ERROR);
    }

    cudaIpcMemHandle_t ipc_handle{};
    std::memcpy(&ipc_handle, handle.handle.data(), sizeof(ipc_handle));

    void *base = nullptr;
    cudaError_t ret =
        cudaIpcOpenMemHandle(&base, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
    if (current_device >= 0) cudaSetDevice(current_device);
    if (ret != cudaSuccess || base == nullptr) {
        ClearCudaError();
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }

    auto *ptr = static_cast<char *>(base) + handle.offset;
    return CudaIpcBufferMapping(base, ptr);
#else
    (void)handle;
    return tl::unexpected(ErrorCode::INVALID_PARAMS);
#endif
}

void CudaIpcBufferMapping::Close() {
#if defined(USE_CUDA)
    if (base_ != nullptr) {
        cudaIpcCloseMemHandle(base_);
    }
#endif
    base_ = nullptr;
    ptr_ = nullptr;
}

}  // namespace device
}  // namespace mooncake

#include "nvme_kv_connector.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <utility>

#include <glog/logging.h>

#include "nvme_kv_executor.h"
#include "storage_backend.h"
#include "utils.h"

namespace mooncake {
namespace {

bool IsSafeDeviceId(const std::string& device_id) {
    return !device_id.empty() &&
           std::ranges::all_of(device_id,
                               [](unsigned char ch) {
                                   return std::isalnum(ch) || ch == '_' ||
                                          ch == '-' || ch == '.';
                               }) &&
           device_id != "." && device_id != "..";
}

}  // namespace

NvmeKvConnector::NvmeKvConnector(const FileStorageConfig& file_storage_config,
                                 std::string device_id)
    : storage_root_(file_storage_config.storage_filepath),
      device_id_(std::move(device_id)),
      storage_path_(std::filesystem::path(storage_root_) / "nvme_kv_blobs" /
                    device_id_) {}

tl::expected<void, ErrorCode> NvmeKvConnector::Init() {
    if (executor_ != nullptr || !IsSafeDeviceId(device_id_)) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    auto real_res = InitRealExecutor();
    if (!real_res) {
        metadata_.last_error = "real executor init failed";
        metadata_.health_state = "error";
    }
    return real_res;
}

tl::expected<void, ErrorCode> NvmeKvConnector::InitRealExecutor() {
    const auto device_path = ResolveConfiguredDevicePath();
    if (device_path.empty()) {
        return tl::make_unexpected(ErrorCode::OBJECT_NOT_FOUND);
    }

    metadata_.device_path = device_path;
    metadata_.uuid = device_path;
    metadata_.nsid = GetEnvOr<uint32_t>("MOONCAKE_NVME_KV_NSID", 1);

    tl::expected<Capabilities, ErrorCode> capabilities =
        tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    auto executor = CreateNvmeKvIoctlExecutor(
        device_id_, storage_path_, device_path, metadata_.nsid, capabilities);
    if (executor == nullptr || !capabilities) {
        return tl::make_unexpected(capabilities ? ErrorCode::INTERNAL_ERROR
                                                : capabilities.error());
    }

    executor_ = std::move(executor);
    metadata_.backend_type = executor_->GetBackendType();
    metadata_.probed_capabilities = capabilities->probed;
    metadata_.health_state = "healthy";
    return {};
}

std::string NvmeKvConnector::ResolveConfiguredDevicePath() const {
    const std::string direct_key = "MOONCAKE_NVME_KV_DEVICE_PATH_" + device_id_;
    auto direct = GetEnvStringOr(direct_key.c_str(), "");
    if (!direct.empty()) {
        return direct;
    }

    return GetEnvStringOr("MOONCAKE_NVME_KV_DEVICE_PATH", "");
}

tl::expected<void, ErrorCode> NvmeKvConnector::Store(const PhysicalKey& key,
                                                     std::string value) {
    if (executor_ == nullptr) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    auto result = executor_->Store(key, std::move(value));
    if (!result) {
        metadata_.health_state = "error";
        metadata_.last_error = "store failed";
    }
    return result;
}

tl::expected<std::string, ErrorCode> NvmeKvConnector::Retrieve(
    const PhysicalKey& key) const {
    if (executor_ == nullptr) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return executor_->Retrieve(key);
}

tl::expected<bool, ErrorCode> NvmeKvConnector::Exists(
    const PhysicalKey& key) const {
    if (executor_ == nullptr) {
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return executor_->Exists(key);
}

const NvmeKvConnector::Capabilities& NvmeKvConnector::GetCapabilities() const {
    static const Capabilities kDefaultCapabilities{};
    if (executor_ == nullptr) {
        return kDefaultCapabilities;
    }
    return executor_->GetCapabilities();
}

}  // namespace mooncake

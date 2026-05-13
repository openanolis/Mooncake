#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

#include <ylt/util/tl/expected.hpp>

#include "nvme_kv_executor.h"
#include "types.h"

namespace mooncake {

struct FileStorageConfig;

class NvmeKvConnector {
   public:
    using PhysicalKey = NvmeKvCommandExecutor::PhysicalKey;
    using Capabilities = NvmeKvCommandExecutor::Capabilities;

    struct DeviceMetadata {
        std::string backend_type;
        std::string uuid;
        std::string device_path;
        uint32_t nsid = 0;
        bool probed_capabilities = false;
        std::string health_state = "unknown";
        std::string last_error;
    };

    explicit NvmeKvConnector(const FileStorageConfig& file_storage_config,
                             std::string device_id = "default");

    tl::expected<void, ErrorCode> Init();
    tl::expected<void, ErrorCode> Store(const PhysicalKey& key,
                                        std::string value);
    tl::expected<std::string, ErrorCode> Retrieve(const PhysicalKey& key) const;
    tl::expected<bool, ErrorCode> Exists(const PhysicalKey& key) const;
    const Capabilities& GetCapabilities() const;
    const DeviceMetadata& GetMetadata() const { return metadata_; }
    const std::string& GetDeviceId() const { return device_id_; }
    const std::filesystem::path& GetStoragePath() const {
        return storage_path_;
    }

   private:
    tl::expected<void, ErrorCode> InitRealExecutor();
    std::string ResolveConfiguredDevicePath() const;

    std::string storage_root_;
    std::string device_id_;
    std::filesystem::path storage_path_;
    DeviceMetadata metadata_;
    std::unique_ptr<NvmeKvCommandExecutor> executor_;
};

}  // namespace mooncake

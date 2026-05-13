#include "nvme_kv_executor.h"

#include "nvme_kv_key_codec.h"

#include <fcntl.h>
#include <linux/nvme_ioctl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

namespace mooncake {
namespace {

constexpr uint32_t kDefaultValueSizeLimit = 128 * 1024;
constexpr uint32_t kDefaultTimeoutMs = 30000;
constexpr uint32_t kKvMaxKeySize = 16;
constexpr uint32_t kNvmeIdentifyDataSize = 4096;
constexpr uint8_t kNvmeAdminIdentifyOpcode = 0x06;
constexpr uint8_t kNvmeIdentifyCsiNamespace = 0x05;
constexpr uint8_t kKvStoreOpcode = 0x01;
constexpr uint8_t kKvRetrieveOpcode = 0x02;
constexpr uint8_t kKvExistOpcode = 0x14;
constexpr uint8_t kKvCommandSetIndicator = 0x01;
constexpr uint32_t kCdw11KeyLengthMask = 0xFFu;
constexpr uint32_t kCdw14PayloadMask = 0x00FFFFFFu;

uint32_t SetCdw14WithCsi(uint32_t key_hi_low32) {
    return (static_cast<uint32_t>(kKvCommandSetIndicator) << 24) |
           (key_hi_low32 & kCdw14PayloadMask);
}

uint32_t BuildKeyLengthField(size_t key_length) {
    return static_cast<uint32_t>(key_length) & kCdw11KeyLengthMask;
}

uint16_t ReadLe16(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) |
           (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t ReadLe32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

tl::expected<uint32_t, ErrorCode> ComputeKvBlockCountMinusOne(uint32_t bytes) {
    if (bytes == 0) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    return (bytes + 511u) / 512u - 1u;
}

bool ShouldTraceCommands() {
    static const bool enabled = [] {
        const char* value = std::getenv("MOONCAKE_NVME_KV_TRACE_COMMANDS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void EncodeKeyIntoCommand(const NvmeKvCommandExecutor::PhysicalKey& key,
                          nvme_passthru_cmd64& cmd) {
    uint64_t key_buf[2] = {0, 0};
    std::memcpy(key_buf, key.data(), key.size());
    cmd.cdw2 = static_cast<uint32_t>(key_buf[0]);
    cmd.cdw3 = static_cast<uint32_t>(key_buf[0] >> 32);
    cmd.cdw14 = SetCdw14WithCsi(static_cast<uint32_t>(key_buf[1]));
    cmd.cdw15 = static_cast<uint32_t>(key_buf[1] >> 32);
}

void TraceCommandBuild(const char* op_name,
                       const NvmeKvCommandExecutor::PhysicalKey& key,
                       const nvme_passthru_cmd64& cmd) {
    if (!ShouldTraceCommands()) {
        return;
    }
    LOG(INFO) << "[NVME-KV-TRACE] submit"
              << " transport=ioctl"
              << " op=" << op_name << " key=" << NvmeKvPhysicalKeyToHex(key)
              << " opcode=0x" << std::hex << static_cast<uint32_t>(cmd.opcode)
              << " nsid=" << std::dec << cmd.nsid
              << " data_len=" << cmd.data_len
              << " timeout_ms=" << cmd.timeout_ms << " cdw2=0x" << std::hex
              << cmd.cdw2 << " cdw3=0x" << cmd.cdw3 << " cdw10=0x" << cmd.cdw10
              << " cdw11=0x" << cmd.cdw11 << " cdw12=0x" << cmd.cdw12
              << " cdw13=0x" << cmd.cdw13 << " cdw14=0x" << cmd.cdw14
              << " cdw15=0x" << cmd.cdw15;
}

void TraceCommandResult(const char* op_name,
                        const NvmeKvCommandExecutor::PhysicalKey& key,
                        const nvme_passthru_cmd64& cmd, int raw_res,
                        ErrorCode mapped_error, bool success,
                        uint64_t result_word0) {
    if (!ShouldTraceCommands()) {
        return;
    }
    LOG(INFO) << "[NVME-KV-TRACE] complete"
              << " transport=ioctl"
              << " op=" << op_name << " key=" << NvmeKvPhysicalKeyToHex(key)
              << " success=" << (success ? "true" : "false")
              << " raw_res=" << raw_res
              << " mapped_error=" << static_cast<int>(mapped_error)
              << " result=0x" << std::hex << result_word0 << " cdw2=0x"
              << cmd.cdw2 << " cdw3=0x" << cmd.cdw3 << " cdw10=0x" << cmd.cdw10
              << " cdw11=0x" << cmd.cdw11 << " cdw12=0x" << cmd.cdw12
              << " cdw13=0x" << cmd.cdw13 << " cdw14=0x" << cmd.cdw14
              << " cdw15=0x" << cmd.cdw15;
}

std::optional<NvmeKvCommandExecutor::Capabilities> ParseKvIdentifyNamespace(
    const std::vector<uint8_t>& data) {
    constexpr size_t kNamespaceKvfcOffset = 28;
    constexpr size_t kFormatTableOffset = 72;
    constexpr size_t kFormatSize = 16;
    constexpr size_t kMaxFormatCount = 16;

    if (data.size() < kNvmeIdentifyDataSize) {
        return std::nullopt;
    }

    const uint8_t active_format = data[kNamespaceKvfcOffset] & 0x0F;
    if (active_format >= kMaxFormatCount) {
        return std::nullopt;
    }

    const size_t format_offset =
        kFormatTableOffset + active_format * kFormatSize;
    const uint32_t max_key_size = ReadLe16(data.data() + format_offset);
    const uint32_t max_value_size = ReadLe32(data.data() + format_offset + 4);
    if (max_key_size == 0 || max_value_size == 0) {
        return std::nullopt;
    }

    NvmeKvCommandExecutor::Capabilities caps;
    caps.max_key_size = max_key_size;
    caps.max_value_size = max_value_size;
    caps.fallback_value_size_limit = kDefaultValueSizeLimit;
    caps.effective_max_value_size = max_value_size;
    caps.probed = true;
    return caps;
}

ErrorCode MapTransportError(int err, bool is_write) {
    switch (err) {
        case ENOENT:
            return ErrorCode::OBJECT_NOT_FOUND;
        case EINVAL:
            return ErrorCode::INVALID_PARAMS;
        case ENOMEM:
            return ErrorCode::BUFFER_OVERFLOW;
        default:
            return is_write ? ErrorCode::FILE_WRITE_FAIL
                            : ErrorCode::FILE_READ_FAIL;
    }
}

class NvmeKvIoctlExecutor : public NvmeKvCommandExecutor {
   public:
    NvmeKvIoctlExecutor(std::string device_path, uint32_t nsid,
                        Capabilities capabilities)
        : device_path_(std::move(device_path)),
          nsid_(nsid),
          capabilities_(capabilities) {}

    tl::expected<void, ErrorCode> Init() {
        fd_ = ::open(device_path_.c_str(), O_RDWR | O_CLOEXEC);
        if (fd_ < 0) {
            LOG(ERROR) << "[NvmeKvIoctlExecutor] open failed for "
                       << device_path_ << ": " << strerror(errno);
            return tl::make_unexpected(ErrorCode::FILE_OPEN_FAIL);
        }
        return {};
    }

    std::optional<Capabilities> ProbeCapabilities() const {
        std::vector<uint8_t> data(kNvmeIdentifyDataSize);
        nvme_passthru_cmd64 cmd{};
        cmd.opcode = kNvmeAdminIdentifyOpcode;
        cmd.nsid = nsid_;
        cmd.addr = reinterpret_cast<uint64_t>(data.data());
        cmd.data_len = static_cast<uint32_t>(data.size());
        cmd.cdw10 = kNvmeIdentifyCsiNamespace;
        cmd.cdw11 = static_cast<uint32_t>(kKvCommandSetIndicator) << 24;
        cmd.timeout_ms = kDefaultTimeoutMs;

        const int ret = ::ioctl(fd_, NVME_IOCTL_ADMIN64_CMD, &cmd);
        if (ret != 0) {
            LOG(WARNING) << "[NvmeKvIoctlExecutor] capability probe failed for "
                         << device_path_ << ", using conservative fallback: "
                         << (ret < 0 ? strerror(errno) : "command failed");
            return std::nullopt;
        }
        auto caps = ParseKvIdentifyNamespace(data);
        if (!caps) {
            LOG(WARNING) << "[NvmeKvIoctlExecutor] capability probe returned "
                            "invalid data for "
                         << device_path_ << ", using conservative fallback";
        }
        return caps;
    }

    void SetCapabilities(Capabilities capabilities) {
        capabilities_ = capabilities;
    }

    ~NvmeKvIoctlExecutor() override {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    tl::expected<void, ErrorCode> Store(const PhysicalKey& key,
                                        std::string value) override {
        if (value.size() > capabilities_.effective_max_value_size) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }

        nvme_passthru_cmd64 cmd{};
        cmd.opcode = kKvStoreOpcode;
        cmd.nsid = nsid_;
        cmd.addr = reinterpret_cast<uint64_t>(value.data());
        cmd.data_len = static_cast<uint32_t>(value.size());
        cmd.cdw10 = static_cast<uint32_t>(value.size());
        cmd.cdw11 = BuildKeyLengthField(key.size());
        auto block_count =
            ComputeKvBlockCountMinusOne(static_cast<uint32_t>(value.size()));
        if (!block_count) {
            return tl::make_unexpected(block_count.error());
        }
        cmd.cdw12 = block_count.value();
        cmd.cdw13 = 0;
        cmd.timeout_ms = kDefaultTimeoutMs;
        EncodeKeyIntoCommand(key, cmd);

        auto result = Submit(cmd, true, "store", key);
        if (!result) {
            return tl::make_unexpected(result.error());
        }
        return {};
    }

    tl::expected<std::string, ErrorCode> Retrieve(
        const PhysicalKey& key) const override {
        std::vector<char> buffer(capabilities_.effective_max_value_size);
        nvme_passthru_cmd64 cmd{};
        cmd.opcode = kKvRetrieveOpcode;
        cmd.nsid = nsid_;
        cmd.addr = reinterpret_cast<uint64_t>(buffer.data());
        cmd.data_len = static_cast<uint32_t>(buffer.size());
        cmd.cdw10 = static_cast<uint32_t>(buffer.size());
        cmd.cdw11 = BuildKeyLengthField(key.size());
        auto block_count =
            ComputeKvBlockCountMinusOne(static_cast<uint32_t>(buffer.size()));
        if (!block_count) {
            return tl::make_unexpected(block_count.error());
        }
        cmd.cdw12 = block_count.value();
        cmd.cdw13 = 0;
        cmd.timeout_ms = kDefaultTimeoutMs;
        EncodeKeyIntoCommand(key, cmd);

        auto result = Submit(cmd, false, "retrieve", key);
        if (!result) {
            return tl::make_unexpected(result.error());
        }

        const uint32_t actual_size = result.value();
        if (actual_size > buffer.size()) {
            return tl::make_unexpected(ErrorCode::FILE_READ_FAIL);
        }
        return std::string(buffer.data(), buffer.data() + actual_size);
    }

    tl::expected<bool, ErrorCode> Exists(
        const PhysicalKey& key) const override {
        nvme_passthru_cmd64 cmd{};
        cmd.opcode = kKvExistOpcode;
        cmd.nsid = nsid_;
        cmd.cdw10 = 0;
        cmd.cdw11 = BuildKeyLengthField(key.size());
        cmd.cdw12 = 0;
        cmd.cdw13 = 0;
        cmd.timeout_ms = kDefaultTimeoutMs;
        EncodeKeyIntoCommand(key, cmd);

        auto result = Submit(cmd, false, "exists", key);
        if (!result) {
            if (result.error() == ErrorCode::OBJECT_NOT_FOUND) {
                return false;
            }
            return tl::make_unexpected(result.error());
        }
        return true;
    }

    const Capabilities& GetCapabilities() const override {
        return capabilities_;
    }

    std::string GetBackendType() const override { return "ioctl"; }

   private:
    tl::expected<uint32_t, ErrorCode> Submit(nvme_passthru_cmd64& cmd,
                                             bool is_write, const char* op_name,
                                             const PhysicalKey& key) const {
        TraceCommandBuild(op_name, key, cmd);
        const int ret = ::ioctl(fd_, NVME_IOCTL_IO64_CMD, &cmd);
        if (ret < 0) {
            const int err = errno;
            const ErrorCode mapped = MapTransportError(err, is_write);
            TraceCommandResult(op_name, key, cmd, -err, mapped, false,
                               cmd.result);
            return tl::make_unexpected(mapped);
        }
        if (ret > 0) {
            const ErrorCode mapped = MapTransportError(ret, is_write);
            TraceCommandResult(op_name, key, cmd, -ret, mapped, false,
                               cmd.result);
            return tl::make_unexpected(mapped);
        }
        TraceCommandResult(op_name, key, cmd, 0, ErrorCode::OK, true,
                           cmd.result);
        return static_cast<uint32_t>(cmd.result);
    }

    std::string device_path_;
    uint32_t nsid_ = 1;
    Capabilities capabilities_;
    int fd_ = -1;
};

}  // namespace

std::unique_ptr<NvmeKvCommandExecutor> CreateNvmeKvIoctlExecutor(
    const std::string& /*device_id*/, std::filesystem::path /*storage_path*/,
    std::string device_path, uint32_t nsid,
    tl::expected<NvmeKvCommandExecutor::Capabilities, ErrorCode>&
        capabilities) {
    constexpr uint32_t fallback_limit = kDefaultValueSizeLimit;

    NvmeKvCommandExecutor::Capabilities fallback_caps;
    fallback_caps.max_key_size = kKvMaxKeySize;
    fallback_caps.max_value_size = fallback_limit;
    fallback_caps.fallback_value_size_limit = fallback_limit;
    fallback_caps.effective_max_value_size = fallback_limit;

    auto executor = std::make_unique<NvmeKvIoctlExecutor>(
        std::move(device_path), nsid, fallback_caps);
    auto init_res = executor->Init();
    if (!init_res) {
        capabilities = tl::make_unexpected(init_res.error());
        return nullptr;
    }

    auto probed_caps = executor->ProbeCapabilities();
    NvmeKvCommandExecutor::Capabilities caps =
        probed_caps.value_or(fallback_caps);
    if (caps.max_key_size < kKvMaxKeySize) {
        LOG(ERROR) << "[NvmeKvIoctlExecutor] device max key size "
                   << caps.max_key_size << " is smaller than Mooncake key size "
                   << kKvMaxKeySize;
        capabilities = tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        return nullptr;
    }
    executor->SetCapabilities(caps);
    capabilities = caps;
    return executor;
}

}  // namespace mooncake

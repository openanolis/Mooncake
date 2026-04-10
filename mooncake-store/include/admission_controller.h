#pragma once

#include <memory>
#include <string>

#include <ylt/util/tl/expected.hpp>

#include "master_config.h"
#include "master_metric_manager.h"
#include "types.h"

namespace mooncake {

struct AdmissionRequestContext {
    enum class Operation {
        PUT_START = 0,
        UPSERT_START,
    };

    Operation operation;
    std::string key;
    uint64_t slice_length;
    uint64_t effective_requested_bytes;
    size_t replica_num;
    std::string tenant_id;
    std::string domain_id;
    std::string object_set;
    std::string sharing_scope;
    std::string qos_tier;
    std::string logical_key;
    std::string canonical_key;
};

class AdmissionController {
   public:
    virtual ~AdmissionController() = default;

    virtual tl::expected<void, ErrorCode> Admit(
        const AdmissionRequestContext& context) const = 0;
};

class NoopAdmissionController : public AdmissionController {
   public:
    tl::expected<void, ErrorCode> Admit(
        const AdmissionRequestContext& context) const override;
};

class QuotaAdmissionController : public AdmissionController {
   public:
    explicit QuotaAdmissionController(uint64_t quota_bytes)
        : quota_bytes_(quota_bytes) {}

    tl::expected<void, ErrorCode> Admit(
        const AdmissionRequestContext& context) const override;

   private:
    uint64_t quota_bytes_;
};

std::shared_ptr<AdmissionController> CreateAdmissionController(
    AdmissionStrategyType type, uint64_t quota_bytes);

}  // namespace mooncake

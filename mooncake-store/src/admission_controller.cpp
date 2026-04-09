#include "admission_controller.h"

#include <memory>

#include <glog/logging.h>

namespace mooncake {

tl::expected<void, ErrorCode> NoopAdmissionController::Admit(
    const AdmissionRequestContext& context) const {
    (void)context;
    return {};
}

tl::expected<void, ErrorCode> QuotaAdmissionController::Admit(
    const AdmissionRequestContext& context) const {
    if (quota_bytes_ == 0) {
        return {};
    }

    const auto current_live_bytes = MasterMetricManager::instance()
                                        .get_labeled_live_bytes(
                                            context.tenant_id, context.domain_id,
                                            context.object_set);
    const auto requested_bytes = context.effective_requested_bytes;
    if (static_cast<uint64_t>(current_live_bytes) + requested_bytes <=
        quota_bytes_) {
        return {};
    }

    LOG(INFO) << "action=admission_reject"
              << ", reason=quota_exceeded"
              << ", key=" << context.key << ", tenant_id="
              << context.tenant_id << ", domain_id=" << context.domain_id
              << ", object_set=" << context.object_set
              << ", sharing_scope=" << context.sharing_scope
              << ", canonical_key=" << context.canonical_key
              << ", requested_bytes=" << requested_bytes
              << ", slice_length=" << context.slice_length
              << ", current_live_bytes=" << current_live_bytes
              << ", quota_bytes=" << quota_bytes_;
    return tl::make_unexpected(ErrorCode::QUOTA_EXCEEDED);
}

std::shared_ptr<AdmissionController> CreateAdmissionController(
    AdmissionStrategyType type, uint64_t quota_bytes) {
    switch (type) {
        case AdmissionStrategyType::NOOP:
            return std::make_shared<NoopAdmissionController>();
        case AdmissionStrategyType::QUOTA:
            return std::make_shared<QuotaAdmissionController>(quota_bytes);
        default:
            return std::make_shared<NoopAdmissionController>();
    }
}

}  // namespace mooncake

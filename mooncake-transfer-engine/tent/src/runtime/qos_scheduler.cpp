// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tent/runtime/qos_scheduler.h"

#include <algorithm>
#include <deque>
#include <unordered_map>
#include <utility>

namespace mooncake {
namespace tent {
namespace {

std::string tenantKeyFor(const Request& request,
                         const QosSchedulerConfig& config) {
    if (!request.qos_context.tenant_id.empty()) {
        return request.qos_context.tenant_id;
    }
    return config.default_tenant_id;
}

uint32_t sharesFor(const Request& request, const QosSchedulerConfig& config) {
    if (request.qos_context.tenant_shares != 0) {
        return request.qos_context.tenant_shares;
    }
    if (!request.qos_context.qos_tier.empty()) {
        auto it = config.tier_shares.find(request.qos_context.qos_tier);
        if (it != config.tier_shares.end()) {
            return it->second;
        }
    }
    return config.default_tenant_shares;
}

Status loadTierShares(const Config& config, QosSchedulerConfig& qos_config) {
    if (!config.contains("qos/tier_shares")) {
        return Status::OK();
    }

    auto tier_shares = config.get<json>("qos/tier_shares", json::object());
    if (!tier_shares.is_object()) {
        return Status::InvalidArgument(
            "qos/tier_shares must be a JSON object" LOC_MARK);
    }

    qos_config.tier_shares.clear();
    for (const auto& [tier_name, share_value] : tier_shares.items()) {
        if (tier_name.empty()) {
            return Status::InvalidArgument(
                "qos/tier_shares keys must not be empty" LOC_MARK);
        }
        if (!share_value.is_number_unsigned() && !share_value.is_number_integer()) {
            return Status::InvalidArgument(
                "qos/tier_shares values must be integers" LOC_MARK);
        }

        auto shares = share_value.get<long long>();
        if (shares <= 0) {
            return Status::InvalidArgument(
                "qos/tier_shares values must be greater than 0" LOC_MARK);
        }
        qos_config.tier_shares.emplace(tier_name,
                                       static_cast<uint32_t>(shares));
    }

    return Status::OK();
}

Status loadTierRateLimits(const Config& config,
                          QosSchedulerConfig& qos_config) {
    if (!config.contains("qos/tier_rate_limits")) {
        return Status::OK();
    }

    auto tier_rate_limits =
        config.get<json>("qos/tier_rate_limits", json::object());
    if (!tier_rate_limits.is_object()) {
        return Status::InvalidArgument(
            "qos/tier_rate_limits must be a JSON object" LOC_MARK);
    }

    qos_config.tier_rate_limits.clear();
    for (const auto& [tier_name, rate_value] : tier_rate_limits.items()) {
        if (tier_name.empty()) {
            return Status::InvalidArgument(
                "qos/tier_rate_limits keys must not be empty" LOC_MARK);
        }
        if (!rate_value.is_number_unsigned() && !rate_value.is_number_integer()) {
            return Status::InvalidArgument(
                "qos/tier_rate_limits values must be integers" LOC_MARK);
        }

        auto rate_limit = rate_value.get<long long>();
        if (rate_limit <= 0) {
            return Status::InvalidArgument(
                "qos/tier_rate_limits values must be greater than 0" LOC_MARK);
        }
        qos_config.tier_rate_limits.emplace(
            tier_name, static_cast<uint64_t>(rate_limit));
    }

    return Status::OK();
}

}  // namespace

Status LoadQosSchedulerConfig(const Config& config,
                              QosSchedulerConfig& qos_config) {
    qos_config.enabled = config.get("qos/enabled", false);
    qos_config.default_tenant_id =
        config.get("qos/default_tenant_id", std::string("default"));
    qos_config.default_tenant_shares =
        config.get("qos/default_tenant_shares", 1024u);
    CHECK_STATUS(loadTierShares(config, qos_config));
    qos_config.scheduler_type =
        config.get("qos/scheduler/type", std::string("weighted_fair"));
    qos_config.dispatch_quantum_bytes =
        config.get("qos/scheduler/dispatch_quantum_bytes", 64ull * 1024ull);
    qos_config.max_inflight_per_tenant =
        config.get("qos/scheduler/max_inflight_per_tenant", size_t{64});
    qos_config.bandwidth_shaping_enabled =
        config.get("qos/scheduler/bandwidth_shaping_enabled", false);
    qos_config.default_tenant_rate_limit_bytes_per_sec = config.get(
        "qos/default_tenant_rate_limit_bytes_per_sec", uint64_t{0});
    CHECK_STATUS(loadTierRateLimits(config, qos_config));
    qos_config.bandwidth_burst_bytes = config.get(
        "qos/scheduler/bandwidth_burst_bytes", 4ull * 1024ull * 1024ull);
    qos_config.target_interval_us =
        config.get("qos/scheduler/target_interval_us", 2000ull);
    qos_config.min_chunk_bytes =
        config.get("qos/scheduler/min_chunk_bytes", size_t{256ull * 1024ull});
    qos_config.max_chunk_bytes =
        config.get("qos/scheduler/max_chunk_bytes", size_t{1024ull * 1024ull});
    qos_config.adaptive_shaping_enabled =
        config.get("qos/scheduler/adaptive_shaping_enabled", false);
    qos_config.capacity_estimation_enabled =
        config.get("qos/scheduler/capacity_estimation_enabled", false);
    qos_config.initial_estimated_bandwidth_bytes_per_sec = config.get(
        "qos/scheduler/initial_estimated_bandwidth_bytes_per_sec", uint64_t{0});
    qos_config.min_estimated_bandwidth_bytes_per_sec = config.get(
        "qos/scheduler/min_estimated_bandwidth_bytes_per_sec", uint64_t{0});
    qos_config.max_estimated_bandwidth_bytes_per_sec = config.get(
        "qos/scheduler/max_estimated_bandwidth_bytes_per_sec", uint64_t{0});
    qos_config.control_interval_us =
        config.get("qos/scheduler/control_interval_us", 5000ull);
    qos_config.throughput_ema_alpha =
        config.get("qos/scheduler/throughput_ema_alpha", 0.2);
    qos_config.capacity_increase_ratio =
        config.get("qos/scheduler/capacity_increase_ratio", 1.1);
    qos_config.capacity_decrease_ratio =
        config.get("qos/scheduler/capacity_decrease_ratio", 0.9);
    qos_config.transport_pacing_enabled =
        config.get("qos/scheduler/transport_pacing_enabled", false);
    qos_config.rdma_pacing_quantum_bytes =
        config.get("qos/scheduler/rdma_pacing_quantum_bytes", size_t{256ull * 1024ull});
    qos_config.tcp_pacing_quantum_bytes =
        config.get("qos/scheduler/tcp_pacing_quantum_bytes", size_t{256ull * 1024ull});
    qos_config.hierarchical_shaping_enabled =
        config.get("qos/scheduler/hierarchical_shaping_enabled", false);
    qos_config.domain_hierarchy_weight =
        config.get("qos/scheduler/domain_hierarchy_weight", 1.0);
    qos_config.object_set_hierarchy_weight =
        config.get("qos/scheduler/object_set_hierarchy_weight", 1.0);
    qos_config.closed_loop_control_enabled =
        config.get("qos/scheduler/closed_loop_control_enabled", false);
    qos_config.fairness_error_tolerance =
        config.get("qos/scheduler/fairness_error_tolerance", 0.1);
    qos_config.idle_capacity_recovery_ratio =
        config.get("qos/scheduler/idle_capacity_recovery_ratio", 1.05);

    if (qos_config.default_tenant_id.empty()) {
        return Status::InvalidArgument(
            "qos/default_tenant_id must not be empty" LOC_MARK);
    }
    if (qos_config.default_tenant_shares == 0) {
        return Status::InvalidArgument(
            "qos/default_tenant_shares must be greater than 0" LOC_MARK);
    }
    if (qos_config.scheduler_type != "weighted_fair") {
        return Status::InvalidArgument(
            "qos/scheduler/type must be weighted_fair" LOC_MARK);
    }
    if (qos_config.dispatch_quantum_bytes == 0) {
        return Status::InvalidArgument(
            "qos/scheduler/dispatch_quantum_bytes must be greater than 0" LOC_MARK);
    }
    if (qos_config.max_inflight_per_tenant == 0) {
        return Status::InvalidArgument(
            "qos/scheduler/max_inflight_per_tenant must be greater than 0" LOC_MARK);
    }
    if (qos_config.bandwidth_shaping_enabled) {
        if (qos_config.default_tenant_rate_limit_bytes_per_sec == 0 &&
            qos_config.tier_rate_limits.empty()) {
            return Status::InvalidArgument(
                "bandwidth shaping requires a default rate limit or tier rate limits" LOC_MARK);
        }
        if (qos_config.bandwidth_burst_bytes == 0) {
            return Status::InvalidArgument(
                "qos/scheduler/bandwidth_burst_bytes must be greater than 0" LOC_MARK);
        }
        if (qos_config.target_interval_us == 0) {
            return Status::InvalidArgument(
                "qos/scheduler/target_interval_us must be greater than 0" LOC_MARK);
        }
        if (qos_config.min_chunk_bytes == 0) {
            return Status::InvalidArgument(
                "qos/scheduler/min_chunk_bytes must be greater than 0" LOC_MARK);
        }
        if (qos_config.max_chunk_bytes == 0) {
            return Status::InvalidArgument(
                "qos/scheduler/max_chunk_bytes must be greater than 0" LOC_MARK);
        }
        if (qos_config.min_chunk_bytes > qos_config.max_chunk_bytes) {
            return Status::InvalidArgument(
                "qos/scheduler/min_chunk_bytes must be less than or equal to max_chunk_bytes" LOC_MARK);
        }
        if (qos_config.bandwidth_burst_bytes < qos_config.min_chunk_bytes) {
            return Status::InvalidArgument(
                "qos/scheduler/bandwidth_burst_bytes must be at least min_chunk_bytes" LOC_MARK);
        }
    }
    if (qos_config.adaptive_shaping_enabled ||
        qos_config.capacity_estimation_enabled ||
        qos_config.closed_loop_control_enabled) {
        if (!qos_config.bandwidth_shaping_enabled) {
            return Status::InvalidArgument(
                "adaptive shaping and closed-loop control require bandwidth shaping to be enabled" LOC_MARK);
        }
        if (qos_config.control_interval_us == 0) {
            return Status::InvalidArgument(
                "qos/scheduler/control_interval_us must be greater than 0" LOC_MARK);
        }
        if (qos_config.throughput_ema_alpha <= 0.0 ||
            qos_config.throughput_ema_alpha > 1.0) {
            return Status::InvalidArgument(
                "qos/scheduler/throughput_ema_alpha must be in (0, 1]" LOC_MARK);
        }
        if (qos_config.capacity_increase_ratio < 1.0) {
            return Status::InvalidArgument(
                "qos/scheduler/capacity_increase_ratio must be at least 1.0" LOC_MARK);
        }
        if (qos_config.capacity_decrease_ratio <= 0.0 ||
            qos_config.capacity_decrease_ratio > 1.0) {
            return Status::InvalidArgument(
                "qos/scheduler/capacity_decrease_ratio must be in (0, 1]" LOC_MARK);
        }
        if (qos_config.initial_estimated_bandwidth_bytes_per_sec == 0) {
            qos_config.initial_estimated_bandwidth_bytes_per_sec =
                qos_config.default_tenant_rate_limit_bytes_per_sec;
        }
        if (qos_config.min_estimated_bandwidth_bytes_per_sec == 0) {
            qos_config.min_estimated_bandwidth_bytes_per_sec =
                std::max<uint64_t>(1,
                    qos_config.initial_estimated_bandwidth_bytes_per_sec / 4);
        }
        if (qos_config.max_estimated_bandwidth_bytes_per_sec == 0) {
            qos_config.max_estimated_bandwidth_bytes_per_sec =
                std::max<uint64_t>(qos_config.initial_estimated_bandwidth_bytes_per_sec,
                                   qos_config.default_tenant_rate_limit_bytes_per_sec);
        }
        if (qos_config.min_estimated_bandwidth_bytes_per_sec >
            qos_config.initial_estimated_bandwidth_bytes_per_sec) {
            return Status::InvalidArgument(
                "min estimated bandwidth must be less than or equal to initial estimate" LOC_MARK);
        }
        if (qos_config.initial_estimated_bandwidth_bytes_per_sec >
            qos_config.max_estimated_bandwidth_bytes_per_sec) {
            return Status::InvalidArgument(
                "initial estimated bandwidth must be less than or equal to max estimate" LOC_MARK);
        }
    }
    if (qos_config.transport_pacing_enabled) {
        if (qos_config.rdma_pacing_quantum_bytes == 0 ||
            qos_config.tcp_pacing_quantum_bytes == 0) {
            return Status::InvalidArgument(
                "transport pacing requires non-zero pacing quanta" LOC_MARK);
        }
    }
    if (qos_config.hierarchical_shaping_enabled) {
        if (qos_config.domain_hierarchy_weight <= 0.0 ||
            qos_config.object_set_hierarchy_weight <= 0.0) {
            return Status::InvalidArgument(
                "hierarchical weights must be greater than 0" LOC_MARK);
        }
    }
    if (qos_config.closed_loop_control_enabled) {
        if (qos_config.fairness_error_tolerance < 0.0 ||
            qos_config.fairness_error_tolerance > 1.0) {
            return Status::InvalidArgument(
                "fairness error tolerance must be in [0, 1]" LOC_MARK);
        }
        if (qos_config.idle_capacity_recovery_ratio < 1.0) {
            return Status::InvalidArgument(
                "idle capacity recovery ratio must be at least 1.0" LOC_MARK);
        }
    }
    return Status::OK();
}

void QosScheduler::NormalizeRequests(std::vector<Request>& requests) const {
    for (auto& request : requests) {
        if (request.qos_context.tenant_id.empty()) {
            request.qos_context.tenant_id = config_.default_tenant_id;
        }
        if (request.qos_context.domain_id.empty()) {
            request.qos_context.domain_id = "default";
        }
        if (request.qos_context.object_set.empty()) {
            request.qos_context.object_set = "default";
        }
        if (request.qos_context.qos_tier.empty()) {
            request.qos_context.qos_tier = "default";
        }
        if (request.qos_context.tenant_shares == 0) {
            request.qos_context.tenant_shares = sharesFor(request, config_);
        }
    }
}

std::vector<Request> QosScheduler::OrderRequests(
    const std::vector<Request>& requests) const {
    if (!config_.enabled || requests.size() <= 1) {
        return requests;
    }

    struct TenantQueue {
        std::deque<Request> requests;
        uint64_t deficit{0};
        uint64_t quantum{0};
        size_t first_index{0};
    };

    std::unordered_map<std::string, TenantQueue> tenant_queues;
    std::vector<std::string> tenant_order;
    tenant_order.reserve(requests.size());

    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        auto tenant_key = tenantKeyFor(request, config_);
        auto [it, inserted] = tenant_queues.try_emplace(tenant_key);
        if (inserted) {
            tenant_order.push_back(tenant_key);
            it->second.first_index = i;
            uint64_t shares = sharesFor(request, config_);
            it->second.quantum =
                std::max<uint64_t>(1, config_.dispatch_quantum_bytes * shares /
                                          config_.default_tenant_shares);
        }
        it->second.requests.push_back(request);
    }

    std::stable_sort(tenant_order.begin(), tenant_order.end(),
                     [&](const std::string& lhs, const std::string& rhs) {
                         return tenant_queues[lhs].first_index <
                                tenant_queues[rhs].first_index;
                     });

    std::vector<Request> ordered;
    ordered.reserve(requests.size());

    while (ordered.size() < requests.size()) {
        bool made_progress = false;
        for (const auto& tenant_key : tenant_order) {
            auto& queue = tenant_queues[tenant_key];
            if (queue.requests.empty()) {
                continue;
            }

            queue.deficit += queue.quantum;
            while (!queue.requests.empty() &&
                   queue.requests.front().length <= queue.deficit) {
                queue.deficit -= queue.requests.front().length;
                ordered.push_back(queue.requests.front());
                queue.requests.pop_front();
                made_progress = true;
            }
        }

        if (made_progress) {
            continue;
        }

        for (const auto& tenant_key : tenant_order) {
            auto& queue = tenant_queues[tenant_key];
            if (queue.requests.empty()) {
                continue;
            }
            ordered.push_back(queue.requests.front());
            queue.requests.pop_front();
        }
    }

    return ordered;
}

}  // namespace tent
}  // namespace mooncake

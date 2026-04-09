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

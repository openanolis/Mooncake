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

#ifndef TENT_QOS_SCHEDULER_H_
#define TENT_QOS_SCHEDULER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tent/common/config.h"
#include "tent/common/status.h"
#include "tent/common/types.h"

namespace mooncake {
namespace tent {

struct QosSchedulerConfig {
    bool enabled{false};
    std::string default_tenant_id{"default"};
    uint32_t default_tenant_shares{1024};
    std::unordered_map<std::string, uint32_t> tier_shares{};
    std::string scheduler_type{"weighted_fair"};
    uint64_t dispatch_quantum_bytes{64 * 1024};
    size_t max_inflight_per_tenant{64};
};

Status LoadQosSchedulerConfig(const Config& config,
                              QosSchedulerConfig& qos_config);

class QosScheduler {
   public:
    explicit QosScheduler(QosSchedulerConfig config)
        : config_(std::move(config)) {}

    const QosSchedulerConfig& config() const { return config_; }

    void NormalizeRequests(std::vector<Request>& requests) const;

    std::vector<Request> OrderRequests(
        const std::vector<Request>& requests) const;

   private:
    QosSchedulerConfig config_;
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_QOS_SCHEDULER_H_

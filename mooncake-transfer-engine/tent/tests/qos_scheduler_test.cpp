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

#include <gtest/gtest.h>

#include <vector>

#include "tent/common/config.h"
#include "tent/runtime/qos_scheduler.h"

namespace mooncake {
namespace tent {
namespace {

Request makeRequest(const std::string& tenant_id, size_t length) {
    Request request;
    request.opcode = Request::READ;
    request.source = nullptr;
    request.target_id = 1;
    request.target_offset = 0;
    request.length = length;
    request.qos_context.tenant_id = tenant_id;
    return request;
}

TEST(QosSchedulerTest, LoadConfigUsesDefaults) {
    Config config;
    QosSchedulerConfig qos_config;
    auto status = LoadQosSchedulerConfig(config, qos_config);
    ASSERT_TRUE(status.ok());
    EXPECT_FALSE(qos_config.enabled);
    EXPECT_EQ(qos_config.default_tenant_id, "default");
    EXPECT_EQ(qos_config.default_tenant_shares, 1024u);
    EXPECT_EQ(qos_config.scheduler_type, "weighted_fair");
    EXPECT_EQ(qos_config.dispatch_quantum_bytes, 64u * 1024u);
    EXPECT_EQ(qos_config.max_inflight_per_tenant, 64u);
}

TEST(QosSchedulerTest, LoadConfigRejectsZeroDefaultShares) {
    Config config;
    config.set("qos/default_tenant_shares", 0u);

    QosSchedulerConfig qos_config;
    auto status = LoadQosSchedulerConfig(config, qos_config);
    EXPECT_FALSE(status.ok());
}

TEST(QosSchedulerTest, LoadConfigReadsTierShares) {
    Config config;
    config.set("qos/tier_shares/gold", 4096u);
    config.set("qos/tier_shares/silver", 2048u);

    QosSchedulerConfig qos_config;
    auto status = LoadQosSchedulerConfig(config, qos_config);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(qos_config.tier_shares.size(), 2u);
    EXPECT_EQ(qos_config.tier_shares.at("gold"), 4096u);
    EXPECT_EQ(qos_config.tier_shares.at("silver"), 2048u);
}

TEST(QosSchedulerTest, LoadConfigRejectsInvalidTierShares) {
    Config config;
    config.set("qos/tier_shares/bronze", 0u);

    QosSchedulerConfig qos_config;
    auto status = LoadQosSchedulerConfig(config, qos_config);

    EXPECT_FALSE(status.ok());
}

TEST(QosSchedulerTest, NormalizeRequestsFillsDefaults) {
    QosScheduler scheduler(QosSchedulerConfig{.enabled = true});
    std::vector<Request> requests(1);
    requests[0].length = 4096;

    scheduler.NormalizeRequests(requests);

    EXPECT_EQ(requests[0].qos_context.tenant_id, "default");
    EXPECT_EQ(requests[0].qos_context.domain_id, "default");
    EXPECT_EQ(requests[0].qos_context.object_set, "default");
    EXPECT_EQ(requests[0].qos_context.qos_tier, "default");
    EXPECT_EQ(requests[0].qos_context.tenant_shares, 1024u);
}

TEST(QosSchedulerTest, NormalizeRequestsUsesTierSharesWhenPresent) {
    QosScheduler scheduler(QosSchedulerConfig{
        .enabled = true,
        .default_tenant_shares = 1024,
        .tier_shares = {{"gold", 4096}, {"silver", 2048}},
    });
    std::vector<Request> requests{makeRequest("tenant-a", 1024)};
    requests[0].qos_context.qos_tier = "gold";
    requests[0].qos_context.tenant_shares = 0;

    scheduler.NormalizeRequests(requests);

    EXPECT_EQ(requests[0].qos_context.tenant_shares, 4096u);
}

TEST(QosSchedulerTest, ExplicitTenantSharesOverrideTierShares) {
    QosScheduler scheduler(QosSchedulerConfig{
        .enabled = true,
        .default_tenant_shares = 1024,
        .tier_shares = {{"gold", 4096}},
    });
    std::vector<Request> requests{makeRequest("tenant-a", 1024)};
    requests[0].qos_context.qos_tier = "gold";
    requests[0].qos_context.tenant_shares = 512;

    scheduler.NormalizeRequests(requests);

    EXPECT_EQ(requests[0].qos_context.tenant_shares, 512u);
}

TEST(QosSchedulerTest, DisabledSchedulerPreservesOrder) {
    QosScheduler scheduler(QosSchedulerConfig{.enabled = false});
    std::vector<Request> requests{makeRequest("tenant-a", 1024),
                                  makeRequest("tenant-b", 1024)};

    auto ordered = scheduler.OrderRequests(requests);

    ASSERT_EQ(ordered.size(), 2u);
    EXPECT_EQ(ordered[0].qos_context.tenant_id, "tenant-a");
    EXPECT_EQ(ordered[1].qos_context.tenant_id, "tenant-b");
}

TEST(QosSchedulerTest, WeightedFairOrderingHonorsShares) {
    QosScheduler scheduler(QosSchedulerConfig{.enabled = true,
                                              .default_tenant_shares = 1024,
                                              .dispatch_quantum_bytes = 1024});
    std::vector<Request> requests{makeRequest("tenant-a", 1024),
                                  makeRequest("tenant-a", 1024),
                                  makeRequest("tenant-b", 1024),
                                  makeRequest("tenant-b", 1024)};
    requests[0].qos_context.tenant_shares = 2048;
    requests[1].qos_context.tenant_shares = 2048;
    requests[2].qos_context.tenant_shares = 1024;
    requests[3].qos_context.tenant_shares = 1024;

    auto ordered = scheduler.OrderRequests(requests);

    ASSERT_EQ(ordered.size(), 4u);
    EXPECT_EQ(ordered[0].qos_context.tenant_id, "tenant-a");
    EXPECT_EQ(ordered[1].qos_context.tenant_id, "tenant-a");
    EXPECT_EQ(ordered[2].qos_context.tenant_id, "tenant-b");
    EXPECT_EQ(ordered[3].qos_context.tenant_id, "tenant-b");
}

TEST(QosSchedulerTest, EqualSharesRoundRobinAcrossTenants) {
    QosScheduler scheduler(QosSchedulerConfig{.enabled = true,
                                              .default_tenant_shares = 1024,
                                              .dispatch_quantum_bytes = 1024});
    std::vector<Request> requests{makeRequest("tenant-a", 1024),
                                  makeRequest("tenant-b", 1024),
                                  makeRequest("tenant-a", 1024),
                                  makeRequest("tenant-b", 1024)};
    for (auto& request : requests) {
        request.qos_context.tenant_shares = 1024;
    }

    auto ordered = scheduler.OrderRequests(requests);

    ASSERT_EQ(ordered.size(), 4u);
    EXPECT_EQ(ordered[0].qos_context.tenant_id, "tenant-a");
    EXPECT_EQ(ordered[1].qos_context.tenant_id, "tenant-b");
    EXPECT_EQ(ordered[2].qos_context.tenant_id, "tenant-a");
    EXPECT_EQ(ordered[3].qos_context.tenant_id, "tenant-b");
}

}  // namespace
}  // namespace tent
}  // namespace mooncake

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
    EXPECT_FALSE(qos_config.bandwidth_shaping_enabled);
    EXPECT_EQ(qos_config.default_tenant_rate_limit_bytes_per_sec, 0u);
    EXPECT_TRUE(qos_config.tier_rate_limits.empty());
    EXPECT_EQ(qos_config.bandwidth_burst_bytes, 4u * 1024u * 1024u);
    EXPECT_EQ(qos_config.target_interval_us, 2000u);
    EXPECT_EQ(qos_config.min_chunk_bytes, 256u * 1024u);
    EXPECT_EQ(qos_config.max_chunk_bytes, 1024u * 1024u);
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

TEST(QosSchedulerTest, LoadConfigReadsBandwidthShapingSettings) {
    Config config;
    config.set("qos/scheduler/bandwidth_shaping_enabled", true);
    config.set("qos/default_tenant_rate_limit_bytes_per_sec", 128u * 1024u * 1024u);
    config.set("qos/tier_rate_limits/gold", 256u * 1024u * 1024u);
    config.set("qos/scheduler/bandwidth_burst_bytes", 2u * 1024u * 1024u);
    config.set("qos/scheduler/target_interval_us", 1500u);
    config.set("qos/scheduler/min_chunk_bytes", 128u * 1024u);
    config.set("qos/scheduler/max_chunk_bytes", 512u * 1024u);
    config.set("qos/scheduler/adaptive_shaping_enabled", true);
    config.set("qos/scheduler/capacity_estimation_enabled", true);
    config.set("qos/scheduler/initial_estimated_bandwidth_bytes_per_sec",
               256u * 1024u * 1024u);
    config.set("qos/scheduler/min_estimated_bandwidth_bytes_per_sec",
               64u * 1024u * 1024u);
    config.set("qos/scheduler/max_estimated_bandwidth_bytes_per_sec",
               512u * 1024u * 1024u);
    config.set("qos/scheduler/control_interval_us", 4000u);
    config.set("qos/scheduler/throughput_ema_alpha", 0.25);
    config.set("qos/scheduler/capacity_increase_ratio", 1.2);
    config.set("qos/scheduler/capacity_decrease_ratio", 0.8);
    config.set("qos/scheduler/transport_pacing_enabled", true);
    config.set("qos/scheduler/rdma_pacing_quantum_bytes", 128u * 1024u);
    config.set("qos/scheduler/tcp_pacing_quantum_bytes", 64u * 1024u);
    config.set("qos/scheduler/hierarchical_shaping_enabled", true);
    config.set("qos/scheduler/domain_hierarchy_weight", 0.5);
    config.set("qos/scheduler/object_set_hierarchy_weight", 0.25);
    config.set("qos/scheduler/closed_loop_control_enabled", true);
    config.set("qos/scheduler/fairness_error_tolerance", 0.15);
    config.set("qos/scheduler/idle_capacity_recovery_ratio", 1.1);

    QosSchedulerConfig qos_config;
    auto status = LoadQosSchedulerConfig(config, qos_config);

    ASSERT_TRUE(status.ok());
    EXPECT_TRUE(qos_config.bandwidth_shaping_enabled);
    EXPECT_EQ(qos_config.default_tenant_rate_limit_bytes_per_sec,
              128u * 1024u * 1024u);
    ASSERT_EQ(qos_config.tier_rate_limits.size(), 1u);
    EXPECT_EQ(qos_config.tier_rate_limits.at("gold"), 256u * 1024u * 1024u);
    EXPECT_EQ(qos_config.bandwidth_burst_bytes, 2u * 1024u * 1024u);
    EXPECT_EQ(qos_config.target_interval_us, 1500u);
    EXPECT_EQ(qos_config.min_chunk_bytes, 128u * 1024u);
    EXPECT_EQ(qos_config.max_chunk_bytes, 512u * 1024u);
    EXPECT_TRUE(qos_config.adaptive_shaping_enabled);
    EXPECT_TRUE(qos_config.capacity_estimation_enabled);
    EXPECT_EQ(qos_config.initial_estimated_bandwidth_bytes_per_sec,
              256u * 1024u * 1024u);
    EXPECT_EQ(qos_config.min_estimated_bandwidth_bytes_per_sec,
              64u * 1024u * 1024u);
    EXPECT_EQ(qos_config.max_estimated_bandwidth_bytes_per_sec,
              512u * 1024u * 1024u);
    EXPECT_EQ(qos_config.control_interval_us, 4000u);
    EXPECT_DOUBLE_EQ(qos_config.throughput_ema_alpha, 0.25);
    EXPECT_DOUBLE_EQ(qos_config.capacity_increase_ratio, 1.2);
    EXPECT_DOUBLE_EQ(qos_config.capacity_decrease_ratio, 0.8);
    EXPECT_TRUE(qos_config.transport_pacing_enabled);
    EXPECT_EQ(qos_config.rdma_pacing_quantum_bytes, 128u * 1024u);
    EXPECT_EQ(qos_config.tcp_pacing_quantum_bytes, 64u * 1024u);
    EXPECT_TRUE(qos_config.hierarchical_shaping_enabled);
    EXPECT_DOUBLE_EQ(qos_config.domain_hierarchy_weight, 0.5);
    EXPECT_DOUBLE_EQ(qos_config.object_set_hierarchy_weight, 0.25);
    EXPECT_TRUE(qos_config.closed_loop_control_enabled);
    EXPECT_DOUBLE_EQ(qos_config.fairness_error_tolerance, 0.15);
    EXPECT_DOUBLE_EQ(qos_config.idle_capacity_recovery_ratio, 1.1);
}

TEST(QosSchedulerTest, LoadConfigRejectsInvalidBandwidthShapingSettings) {
    Config config;
    config.set("qos/scheduler/bandwidth_shaping_enabled", true);
    config.set("qos/default_tenant_rate_limit_bytes_per_sec", 0u);
    config.set("qos/scheduler/bandwidth_burst_bytes", 64u * 1024u);
    config.set("qos/scheduler/min_chunk_bytes", 128u * 1024u);
    config.set("qos/scheduler/max_chunk_bytes", 64u * 1024u);

    QosSchedulerConfig qos_config;
    auto status = LoadQosSchedulerConfig(config, qos_config);

    EXPECT_FALSE(status.ok());
}

TEST(QosSchedulerTest, LoadConfigRejectsInvalidAdaptiveAndClosedLoopSettings) {
    Config config;
    config.set("qos/scheduler/bandwidth_shaping_enabled", true);
    config.set("qos/default_tenant_rate_limit_bytes_per_sec", 128u * 1024u * 1024u);
    config.set("qos/scheduler/adaptive_shaping_enabled", true);
    config.set("qos/scheduler/throughput_ema_alpha", 1.5);
    config.set("qos/scheduler/capacity_increase_ratio", 0.9);
    config.set("qos/scheduler/capacity_decrease_ratio", 1.1);
    config.set("qos/scheduler/transport_pacing_enabled", true);
    config.set("qos/scheduler/rdma_pacing_quantum_bytes", 0u);
    config.set("qos/scheduler/tcp_pacing_quantum_bytes", 0u);

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

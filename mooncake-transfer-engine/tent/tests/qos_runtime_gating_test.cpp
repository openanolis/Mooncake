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

#include "tent/common/types.h"
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

TEST(QosRuntimeGatingTest, OrderedRequestsStillPreserveTenantLocalFifoUnderLimit) {
    QosScheduler scheduler(QosSchedulerConfig{.enabled = true,
                                              .default_tenant_shares = 1024,
                                              .dispatch_quantum_bytes = 1024,
                                              .max_inflight_per_tenant = 1});
    std::vector<Request> requests{makeRequest("tenant-a", 1024),
                                  makeRequest("tenant-a", 2048),
                                  makeRequest("tenant-b", 1024)};

    scheduler.NormalizeRequests(requests);
    auto ordered = scheduler.OrderRequests(requests);

    ASSERT_EQ(ordered.size(), 3u);
    EXPECT_EQ(ordered[0].qos_context.tenant_id, "tenant-a");
    EXPECT_EQ(ordered[1].qos_context.tenant_id, "tenant-b");
    EXPECT_EQ(ordered[2].qos_context.tenant_id, "tenant-a");
    EXPECT_EQ(ordered[0].length, 1024u);
    EXPECT_EQ(ordered[2].length, 2048u);
}

}  // namespace
}  // namespace tent
}  // namespace mooncake

// Copyright 2024 KVCache.AI
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

#include <map>
#include <memory>
#include <string>

#include "transport_isolation_strategy.h"

namespace mooncake {
namespace {

using TransferRequest = Transport::TransferRequest;

TransferRequest MakeRequest(Transport::SegmentID target_id) {
    return TransferRequest{TransferRequest::READ, nullptr, target_id, 0, 4096};
}

TransferMetadata::SegmentDesc MakeSegment(const std::string &protocol) {
    TransferMetadata::SegmentDesc segment;
    segment.name = "segment";
    segment.protocol = protocol;
    return segment;
}

TransportSelectionContext MakeContext(
    const TransferRequest &request, const TransferMetadata::SegmentDesc &segment,
    const std::map<std::string, std::shared_ptr<Transport>> &installed) {
    return TransportSelectionContext{request, segment, installed};
}

TEST(TransportIsolationStrategyTest, DefaultStrategyUsesSegmentProtocol) {
    DefaultTransportIsolationStrategy strategy;
    auto request = MakeRequest(7);
    auto segment = MakeSegment("rdma");
    std::map<std::string, std::shared_ptr<Transport>> installed;
    installed.emplace("rdma", nullptr);

    std::string protocol;
    auto status = strategy.selectProtocol(MakeContext(request, segment, installed),
                                          protocol);

    ASSERT_TRUE(status.ok());
#ifdef USE_ASCEND_HETEROGENEOUS
    ASSERT_EQ(protocol, "ascend");
#else
    ASSERT_EQ(protocol, "rdma");
#endif
}

TEST(TransportIsolationStrategyTest, DefaultStrategyRejectsMissingTransport) {
    DefaultTransportIsolationStrategy strategy;
    auto request = MakeRequest(9);
    auto segment = MakeSegment("tcp");
    std::map<std::string, std::shared_ptr<Transport>> installed;

    std::string protocol;
    auto status = strategy.selectProtocol(MakeContext(request, segment, installed),
                                          protocol);

    ASSERT_FALSE(status.ok());
#ifdef USE_ASCEND_HETEROGENEOUS
    ASSERT_EQ(protocol, "tcp");
#else
    ASSERT_EQ(protocol, "tcp");
#endif
    ASSERT_NE(status.ToString().find("Transport tcp not installed"),
              std::string::npos);
}

TEST(TransportIsolationStrategyTest, StrategyFactoryBuildsDefaultStrategy) {
    auto strategy = CreateTransportIsolationStrategy(
        TransportIsolationStrategyType::DEFAULT);

    ASSERT_NE(strategy, nullptr);
    ASSERT_NE(dynamic_cast<DefaultTransportIsolationStrategy *>(strategy.get()),
              nullptr);
}

}  // namespace
}  // namespace mooncake

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

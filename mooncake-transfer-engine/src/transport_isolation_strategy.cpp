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

#include "transport_isolation_strategy.h"

#include <memory>
#include <string>

namespace mooncake {

Status DefaultTransportIsolationStrategy::selectProtocol(
    const TransportSelectionContext &context, std::string &protocol) const {
    (void)context.request;

    protocol = context.target_segment.protocol;
#ifdef USE_ASCEND_HETEROGENEOUS
    // When USE_ASCEND_HETEROGENEOUS is enabled:
    // - Target side directly reuses RDMA Transport
    // - Initiator side uses heterogeneous_rdma_transport
    if (context.target_segment.protocol == "rdma") {
        protocol = "ascend";
    }
#endif

    if (!context.installed_transports.count(protocol)) {
        return Status::NotSupportedTransport("Transport " + protocol +
                                             " not installed");
    }
    return Status::OK();
}

std::unique_ptr<TransportIsolationStrategy> CreateTransportIsolationStrategy(
    TransportIsolationStrategyType type) {
    switch (type) {
        case TransportIsolationStrategyType::DEFAULT:
            return std::make_unique<DefaultTransportIsolationStrategy>();
    }
    return std::make_unique<DefaultTransportIsolationStrategy>();
}

}  // namespace mooncake

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

#ifndef TRANSPORT_ISOLATION_STRATEGY_H_
#define TRANSPORT_ISOLATION_STRATEGY_H_

#include <map>
#include <memory>
#include <string>

#include "config.h"
#include "transport/transport.h"

namespace mooncake {

struct TransportSelectionContext {
    const Transport::TransferRequest &request;
    const TransferMetadata::SegmentDesc &target_segment;
    const std::map<std::string, std::shared_ptr<Transport>> &installed_transports;
};

class TransportIsolationStrategy {
   public:
    virtual ~TransportIsolationStrategy() = default;

    virtual Status selectProtocol(const TransportSelectionContext &context,
                                  std::string &protocol) const = 0;
};

class DefaultTransportIsolationStrategy : public TransportIsolationStrategy {
   public:
    Status selectProtocol(const TransportSelectionContext &context,
                          std::string &protocol) const override;
};

std::unique_ptr<TransportIsolationStrategy> CreateTransportIsolationStrategy(
    TransportIsolationStrategyType type);

}  // namespace mooncake

#endif  // TRANSPORT_ISOLATION_STRATEGY_H_

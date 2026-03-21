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

#include <memory>
#include <string>
#include <vector>

#define private public
#include "multi_transport.h"
#undef private

#include "transfer_engine.h"
#include "transfer_metadata.h"
#include "transport/rdma_transport/rdma_transport.h"

namespace mooncake {
namespace {

class FakeRdmaTransport : public RdmaTransport {
   public:
    explicit FakeRdmaTransport(std::shared_ptr<TransferMetadata> metadata) {
        metadata_ = std::move(metadata);
        local_server_name_ = "fake-rdma-transport";
    }

    Status submitTransferTask(
        const std::vector<TransferTask*>& task_list) override {
        last_submitted_tasks = task_list;
        return Status::OK();
    }

    std::vector<TransferTask*> last_submitted_tasks;
};

std::shared_ptr<TransferMetadata::SegmentDesc> make_segment_desc(
    const std::string& name) {
    auto desc = std::make_shared<TransferMetadata::SegmentDesc>();
    desc->name = name;
    desc->protocol = "rdma";
    return desc;
}

}  // namespace

TEST(MultiTransportGroupingTest, BatchSizeCountsGroupedScatterTasks) {
    auto metadata = std::make_shared<TransferMetadata>(P2PHANDSHAKE);
    std::string local_server_name = "grouping-test-local";
    MultiTransport multi_transport(metadata, local_server_name);

    auto fake_rdma = std::make_shared<FakeRdmaTransport>(metadata);
    multi_transport.transport_map_["rdma"] = fake_rdma;

    ASSERT_EQ(
        metadata->addLocalSegment(1, "peer-a", make_segment_desc("peer-a")), 0);
    ASSERT_EQ(
        metadata->addLocalSegment(2, "peer-b", make_segment_desc("peer-b")), 0);

    std::vector<char> destination(128, 0);
    std::vector<Transport::TransferRequest> requests = {
        {Transport::TransferRequest::READ, destination.data(), 1, 100, 16, 0,
         7},
        {Transport::TransferRequest::READ, destination.data() + 16, 1, 116, 24,
         0, 7},
        {Transport::TransferRequest::READ, destination.data() + 40, 2, 200, 8,
         0, 8},
        {Transport::TransferRequest::READ, destination.data() + 48, 2, 208, 8,
         0, 8},
        {Transport::TransferRequest::READ, destination.data() + 56, 1, 300, 8,
         0, Transport::TransferRequest::kNoTaskGroup},
    };

    // The batch is sized by logical tasks, not raw request count:
    // [7,7], [8,8], [ungrouped].
    BatchID batch_id = multi_transport.allocateBatchID(3);
    ASSERT_NE(batch_id, INVALID_BATCH_ID);

    Status status = multi_transport.submitTransfer(batch_id, requests);
    ASSERT_TRUE(status.ok()) << status.ToString();

    ASSERT_EQ(fake_rdma->last_submitted_tasks.size(), 3U);

    const auto& batch_desc = Transport::toBatchDesc(batch_id);
    ASSERT_EQ(batch_desc.task_list.size(), 3U);

    EXPECT_EQ(batch_desc.task_list[0].request_group.size(), 2U);
    EXPECT_EQ(batch_desc.task_list[0].request_group[0].task_group_id, 7U);
    EXPECT_EQ(batch_desc.task_list[0].request_group[1].task_group_id, 7U);

    EXPECT_EQ(batch_desc.task_list[1].request_group.size(), 2U);
    EXPECT_EQ(batch_desc.task_list[1].request_group[0].task_group_id, 8U);
    EXPECT_EQ(batch_desc.task_list[1].request_group[1].task_group_id, 8U);

    EXPECT_TRUE(batch_desc.task_list[2].request_group.empty());
    ASSERT_NE(batch_desc.task_list[2].request, nullptr);
    EXPECT_EQ(batch_desc.task_list[2].request->task_group_id,
              Transport::TransferRequest::kNoTaskGroup);

    for (auto& task : Transport::toBatchDesc(batch_id).task_list) {
        task.is_finished = true;
    }
    EXPECT_TRUE(multi_transport.freeBatchID(batch_id).ok());
}

TEST(MultiTransportGroupingTest, GroupingRequiresAdjacentEntries) {
    auto metadata = std::make_shared<TransferMetadata>(P2PHANDSHAKE);
    std::string local_server_name = "grouping-test-local";
    MultiTransport multi_transport(metadata, local_server_name);

    auto fake_rdma = std::make_shared<FakeRdmaTransport>(metadata);
    multi_transport.transport_map_["rdma"] = fake_rdma;

    ASSERT_EQ(
        metadata->addLocalSegment(1, "peer-a", make_segment_desc("peer-a")), 0);

    std::vector<char> destination(96, 0);
    std::vector<Transport::TransferRequest> requests = {
        {Transport::TransferRequest::READ, destination.data(), 1, 0, 16, 0, 9},
        {Transport::TransferRequest::READ, destination.data() + 16, 1, 16, 16,
         0, Transport::TransferRequest::kNoTaskGroup},
        {Transport::TransferRequest::READ, destination.data() + 32, 1, 32, 16,
         0, 9},
        {Transport::TransferRequest::READ, destination.data() + 48, 1, 48, 16,
         0, 10},
        {Transport::TransferRequest::READ, destination.data() + 64, 1, 64, 16,
         0, 10},
    };

    // Only adjacent entries may collapse into one logical task. The two
    // requests tagged with task_group_id=9 stay separate because an ungrouped
    // request sits between them. The two task_group_id=10 entries are merged.
    BatchID batch_id = multi_transport.allocateBatchID(4);
    ASSERT_NE(batch_id, INVALID_BATCH_ID);

    Status status = multi_transport.submitTransfer(batch_id, requests);
    ASSERT_TRUE(status.ok()) << status.ToString();

    ASSERT_EQ(fake_rdma->last_submitted_tasks.size(), 4U);

    const auto& batch_desc = Transport::toBatchDesc(batch_id);
    ASSERT_EQ(batch_desc.task_list.size(), 4U);

    EXPECT_EQ(batch_desc.task_list[0].request_group.size(), 1U);
    EXPECT_EQ(batch_desc.task_list[0].request_group[0].task_group_id, 9U);

    EXPECT_TRUE(batch_desc.task_list[1].request_group.empty());
    ASSERT_NE(batch_desc.task_list[1].request, nullptr);
    EXPECT_EQ(batch_desc.task_list[1].request->task_group_id,
              Transport::TransferRequest::kNoTaskGroup);

    EXPECT_EQ(batch_desc.task_list[2].request_group.size(), 1U);
    EXPECT_EQ(batch_desc.task_list[2].request_group[0].task_group_id, 9U);

    EXPECT_EQ(batch_desc.task_list[3].request_group.size(), 2U);
    EXPECT_EQ(batch_desc.task_list[3].request_group[0].task_group_id, 10U);
    EXPECT_EQ(batch_desc.task_list[3].request_group[1].task_group_id, 10U);

    for (auto& task : Transport::toBatchDesc(batch_id).task_list) {
        task.is_finished = true;
    }
    EXPECT_TRUE(multi_transport.freeBatchID(batch_id).ok());
}

}  // namespace mooncake

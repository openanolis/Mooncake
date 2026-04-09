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

#ifndef TRANSFER_ENGINE_IMPL_H_
#define TRANSFER_ENGINE_IMPL_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tent/common/config.h"
#include "tent/common/status.h"
#include "tent/common/types.h"
#include "tent/common/concurrent/thread_local_storage.h"
#include "tent/runtime/qos_scheduler.h"

namespace mooncake {
namespace tent {

class Batch;
class BatchSet;
class Topology;
class Transport;
class SegmentDesc;
class AllocatedMemory;
class ControlService;
class SegmentTracker;
class Platform;
class ProxyManager;

struct TaskInfo {
    TransportType type{UNSPEC};
    int sub_task_id{-1};
    bool derived{false};  // merged by other tasks
    int xport_priority{0};
    Request request;
    Request active_request;
    std::string hierarchy_key;
    uint32_t effective_shares{1024};
    size_t pacing_quantum_bytes{0};
    bool staging{false};
    bool qos_admitted{false};
    bool qos_queued{false};
    bool pending_bandwidth{false};
    bool fragment_inflight{false};
    size_t logical_length{0};
    size_t submitted_bytes{0};
    size_t completed_bytes{0};
    TransferStatusEnum status{TransferStatusEnum::PENDING};
    volatile TransferStatusEnum staging_status{TransferStatusEnum::PENDING};
    std::chrono::steady_clock::time_point start_time{};  // For latency tracking
};

class TransferEngineImpl {
    friend class ProxyManager;

   public:
    TransferEngineImpl();

    TransferEngineImpl(std::shared_ptr<Config> config);

    ~TransferEngineImpl();

    TransferEngineImpl(const TransferEngineImpl&) = delete;

    TransferEngineImpl& operator=(const TransferEngineImpl&) = delete;

   public:
    bool available() const { return available_; }

    const std::string getSegmentName() const;

    const std::string getRpcServerAddress() const;

    uint16_t getRpcServerPort() const;

   public:
    Status exportLocalSegment(std::string& shared_handle);

    Status importRemoteSegment(SegmentID& handle,
                               const std::string& shared_handle);

    Status openSegment(SegmentID& handle, const std::string& segment_name);

    Status closeSegment(SegmentID handle);

    Status getSegmentInfo(SegmentID handle, SegmentInfo& info);

   public:
    Status allocateLocalMemory(void** addr, size_t size,
                               Location location = kWildcardLocation);

    Status allocateLocalMemory(void** addr, size_t size, Location location,
                               bool internal);

    Status freeLocalMemory(void* addr);

    Status registerLocalMemory(void* addr, size_t size,
                               Permission permission = kGlobalReadWrite);

    Status registerLocalMemory(std::vector<void*> addr_list,
                               std::vector<size_t> size_list,
                               Permission permission = kGlobalReadWrite);

    Status unregisterLocalMemory(void* addr, size_t size = 0);

    Status unregisterLocalMemory(std::vector<void*> addr_list,
                                 std::vector<size_t> size_list = {});

    // advanced buffer allocate function
    Status allocateLocalMemory(void** addr, size_t size,
                               MemoryOptions& options);

    // advanced buffer register function
    Status registerLocalMemory(std::vector<void*> addr_list,
                               std::vector<size_t> size_list,
                               MemoryOptions& options);

   public:
    BatchID allocateBatch(size_t batch_size);

    Status freeBatch(BatchID batch_id);

    Status submitTransfer(BatchID batch_id,
                          const std::vector<Request>& request_list);

    Status submitTransfer(BatchID batch_id,
                          const std::vector<Request>& request_list,
                          const Notification& notifi);

    Status sendNotification(SegmentID target_id, const Notification& notifi);

    Status receiveNotification(std::vector<Notification>& notifi_list);

    Status probePeerAliveByID(SegmentID target_id);

    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status);

    Status getTransferStatus(BatchID batch_id,
                             std::vector<TransferStatus>& status_list);

    Status getTransferStatus(BatchID batch_id, TransferStatus& overall_status);

    Status waitTransferCompletion(BatchID batch_id);

    Status transferSync(const std::vector<Request>& request_list);

    uint64_t lockStageBuffer(const std::string& location);

    Status unlockStageBuffer(uint64_t addr);

   private:
    Status construct();

    Status deconstruct();

    Status setupLocalSegment();

    Status lazyFreeBatch();

    TransportType getTransportType(const Request& request, int priority = 0);

    std::vector<TransportType> getSupportedTransports(
        TransportType request_type);

    Status resubmitTransferTask(Batch* batch, size_t task_id);

    TransportType resolveTransport(const Request& req, int priority,
                                   bool invalidate_on_fail = true);

    Status loadTransports();

    void findStagingPolicy(const Request& req,
                           std::vector<std::string>& policy);

    Status maybeFireSubmitHooks(Batch* batch, bool check = true);

    void recordTaskCompletionMetrics(TaskInfo& task,
                                     TransferStatusEnum prev_status,
                                     TransferStatusEnum new_status);

    std::string tenantKeyForRequest(const Request& request) const;

    bool isTerminalStatus(TransferStatusEnum status) const;

    bool tryAcquireQosSlot(TaskInfo& task);

    void releaseQosSlotIfNeeded(TaskInfo& task, TransferStatusEnum new_status);

    Status drainPendingRequestsForTenant(const std::string& hierarchy_key);

    uint64_t rateLimitForRequest(const Request& request) const;

    uint64_t effectiveRateLimitForTask(const TaskInfo& task) const;

    uint32_t effectiveSharesForRequest(const Request& request) const;

    uint32_t hierarchicalSharesForRequest(const Request& request) const;

    std::string hierarchyKeyForRequest(const Request& request) const;

    size_t pacingQuantumForTask(const TaskInfo& task) const;

    void refreshAdaptiveControlLocked(std::chrono::steady_clock::time_point now);

    void recordBandwidthSampleLocked(const std::string& hierarchy_key,
                                     size_t bytes_completed,
                                     std::chrono::steady_clock::time_point now);

    size_t computeChunkBytesForRate(uint64_t rate_limit_bytes_per_sec) const;

    bool shouldBypassBandwidthShaping(const TaskInfo& task) const;

    bool tryAcquireBandwidthBudget(TaskInfo& task);

    void replenishBandwidthTokensLocked(const std::string& hierarchy_key,
                                        std::chrono::steady_clock::time_point now);

    size_t countActiveTenantsWithBacklogLocked(
        const std::string& include_hierarchy_key) const;

    void enqueuePendingTaskLocked(const std::string& hierarchy_key, Batch* batch,
                                  size_t task_id);

    void buildActiveRequest(TaskInfo& task);

    Status classifyAndQueueTask(Batch* batch, size_t task_id, TaskInfo& task,
                                std::vector<Request> classified_request_list[],
                                std::vector<size_t> task_id_list[],
                                std::unordered_map<TransportType, size_t>& next_sub_task_id);

    Status flushQueuedTasks(Batch* batch,
                            std::vector<Request> classified_request_list[],
                            std::vector<size_t> task_id_list[]);

    void removePendingTasksForBatch(Batch* batch);

   private:
    struct AllocatedMemory {
        void* addr;
        size_t size;
        Transport* transport;
        MemoryOptions options;
    };

    struct BatchSet {
        std::unordered_set<Batch*> active;
        std::vector<Batch*> freelist;
    };

   private:
    std::shared_ptr<Config> conf_;
    std::shared_ptr<ControlService> metadata_;
    std::shared_ptr<Topology> topology_;
    bool available_;

    std::array<std::shared_ptr<Transport>, kSupportedTransportTypes>
        transport_list_;
    std::unique_ptr<SegmentTracker> local_segment_tracker_;

    ThreadLocalStorage<BatchSet> batch_set_;

    std::vector<AllocatedMemory> allocated_memory_;
    std::mutex mutex_;

    std::string hostname_;
    uint16_t port_;
    bool ipv6_;
    std::string local_segment_name_;

    std::unique_ptr<ProxyManager> staging_proxy_;
    bool merge_requests_;
    QosSchedulerConfig qos_scheduler_config_{};
    std::unique_ptr<QosScheduler> qos_scheduler_;
    struct PendingTaskRef {
        Batch* batch{nullptr};
        size_t task_id{0};
    };

    struct BandwidthShapingState {
        uint64_t tokens{0};
        uint64_t configured_rate_limit_bytes_per_sec{0};
        uint64_t adaptive_rate_limit_bytes_per_sec{0};
        uint64_t observed_throughput_ema_bytes_per_sec{0};
        uint64_t estimated_capacity_bytes_per_sec{0};
        uint64_t completed_bytes_since_control{0};
        uint64_t burst_bytes{0};
        std::chrono::steady_clock::time_point last_refill{};
        std::chrono::steady_clock::time_point last_control{};
        std::chrono::steady_clock::time_point last_sample{};
    };

    std::unordered_map<std::string, size_t> tenant_inflight_counts_;
    std::unordered_map<std::string, std::deque<PendingTaskRef>> tenant_pending_task_ids_;
    std::unordered_map<std::string, BandwidthShapingState> tenant_bandwidth_states_;
    std::chrono::steady_clock::time_point last_global_control_{};
    std::mutex qos_mutex_;
};
}  // namespace tent
}  // namespace mooncake

#endif
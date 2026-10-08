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

#include <algorithm>
#include <chrono>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <future>
#include <limits>
#include <thread>
#include <unordered_map>

#ifndef USE_TENT
#include "transfer_engine.h"
#include "show_links.h"
#include "transfer_engine_impl.h"
#include "graceful_shutdown.h"
#include <mutex>
#include <utility>

namespace mooncake {
namespace {

class TransferEngineShutdownToken : public ShutdownToken {
   public:
    explicit TransferEngineShutdownToken(TransferEngine* engine)
        : engine_(engine) {}

    void shutdown() override {
        TransferEngine* engine = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            engine = engine_;
            engine_ = nullptr;
        }
        if (engine) engine->freeEngine();
    }

    void detach() override {
        std::lock_guard<std::mutex> lock(mutex_);
        engine_ = nullptr;
    }

   private:
    std::mutex mutex_;
    TransferEngine* engine_;
};

std::shared_ptr<ShutdownToken> registerTransferEngineShutdownToken(
    TransferEngine* engine) {
    auto token = std::make_shared<TransferEngineShutdownToken>(engine);
    registerTokenForShutdown(token);
    return token;
}

void detachShutdownToken(std::shared_ptr<ShutdownToken>& token) {
    if (!token) return;
    token->detach();
    token.reset();
}

void writeLittleEndian32(char*& output, uint32_t value) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(output, &value, sizeof(value));
    output += sizeof(value);
#else
    for (size_t i = 0; i < sizeof(value); ++i)
        *output++ = static_cast<char>(value >> (i * 8));
#endif
}

uint32_t loadLittleEndian32(const char* input) {
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint32_t>(static_cast<unsigned char>(input[i]))
                 << (i * 8);
    }
    return value;
}

class PlannerValidationWorker {
   public:
    PlannerValidationWorker() {
        try {
            worker_ = std::thread([this] { workerLoop(); });
        } catch (...) {
            // Planning remains correct when thread creation is unavailable;
            // callers fall back to validating inline.
        }
    }

    ~PlannerValidationWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    template <typename Function>
    bool start(Function& function) {
        if (!worker_.joinable()) return false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            DCHECK_EQ(callback_, nullptr);
            callback_ = [](const void* context) {
                return (*static_cast<const Function*>(context))();
            };
            context_ = &function;
            completed_task_ = false;
            task_exception_ = nullptr;
            ++generation_;
        }
        ready_.notify_one();
        return true;
    }

    bool wait() {
        std::exception_ptr task_exception;
        bool result = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            completed_.wait(lock, [this] { return completed_task_; });
            task_exception = task_exception_;
            result = task_result_;
            callback_ = nullptr;
            context_ = nullptr;
        }
        if (task_exception) std::rethrow_exception(task_exception);
        return result;
    }

   private:
    using Callback = bool (*)(const void*);

    void workerLoop() {
        uint64_t seen_generation = 0;
        while (true) {
            Callback callback = nullptr;
            const void* context = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this, &seen_generation] {
                    return stopping_ || generation_ != seen_generation;
                });
                if (stopping_) return;
                seen_generation = generation_;
                callback = callback_;
                context = context_;
            }

            bool result = false;
            std::exception_ptr task_exception;
            try {
                result = callback(context);
            } catch (...) {
                task_exception = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                task_result_ = result;
                task_exception_ = task_exception;
                completed_task_ = true;
            }
            completed_.notify_one();
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable completed_;
    std::thread worker_;
    Callback callback_ = nullptr;
    const void* context_ = nullptr;
    std::exception_ptr task_exception_;
    uint64_t generation_ = 0;
    bool task_result_ = false;
    bool completed_task_ = false;
    bool stopping_ = false;
};

PlannerValidationWorker& plannerValidationWorker() {
    thread_local PlannerValidationWorker worker;
    return worker;
}

}  // namespace

TransferEngine::TransferEngine(bool auto_discover)
    : impl_(std::make_shared<TransferEngineImpl>(auto_discover)) {}

TransferEngine::TransferEngine(bool auto_discover,
                               const std::vector<std::string>& filter)
    : impl_(std::make_shared<TransferEngineImpl>(auto_discover, filter)) {}

TransferEngine::TransferEngine(TransferEngine&& other) noexcept
    : impl_(std::move(other.impl_)),
      impl_tent_(std::move(other.impl_tent_)),
      use_tent_(other.use_tent_) {
    const bool shutdown_enabled = static_cast<bool>(other.shutdown_token_);
    detachShutdownToken(other.shutdown_token_);
    if (shutdown_enabled) {
        shutdown_token_ = registerTransferEngineShutdownToken(this);
        installGracefulShutdownHandlers();
    }
}

TransferEngine& TransferEngine::operator=(TransferEngine&& other) noexcept {
    if (this == &other) return *this;
    freeEngine();
    impl_ = std::move(other.impl_);
    impl_tent_ = std::move(other.impl_tent_);
    tent_device_filter_ = std::move(other.tent_device_filter_);
    use_tent_ = other.use_tent_;
    const bool shutdown_enabled = static_cast<bool>(other.shutdown_token_);
    detachShutdownToken(other.shutdown_token_);
    if (shutdown_enabled) {
        shutdown_token_ = registerTransferEngineShutdownToken(this);
        installGracefulShutdownHandlers();
    }
    return *this;
}

TransferEngine::~TransferEngine() { freeEngine(); }

int TransferEngine::init(const std::string& metadata_conn_string,
                         const std::string& local_server_name,
                         const std::string& ip_or_host_name,
                         uint64_t rpc_port) {
    return impl_->init(metadata_conn_string, local_server_name, ip_or_host_name,
                       rpc_port);
}

int TransferEngine::init(const std::string& metadata_conn_string,
                         const std::string& local_server_name,
                         const std::string& ip_or_host_name, uint64_t rpc_port,
                         const std::string& protocol) {
    (void)protocol;
    return init(metadata_conn_string, local_server_name, ip_or_host_name,
                rpc_port);
}

int TransferEngine::freeEngine() {
    detachShutdownToken(shutdown_token_);
    if (impl_) {
        if (impl_.use_count() == 1) impl_->freeEngine();
        impl_.reset();
    }
    return 0;
}

Transport* TransferEngine::installTransport(const std::string& proto,
                                            void** args) {
    return impl_->installTransport(proto, args);
}

int TransferEngine::uninstallTransport(const std::string& proto) {
    return impl_->uninstallTransport(proto);
}

std::string TransferEngine::getLocalIpAndPort() {
    return impl_->getLocalIpAndPort();
}

int TransferEngine::getRpcPort() { return impl_->getRpcPort(); }

SegmentHandle TransferEngine::openSegment(const std::string& segment_name) {
    return impl_->openSegment(segment_name);
}

Status TransferEngine::CheckSegmentStatus(SegmentID sid) {
    return impl_->CheckSegmentStatus(sid);
}

int TransferEngine::closeSegment(SegmentHandle handle) {
    return impl_->closeSegment(handle);
}

int TransferEngine::removeLocalSegment(const std::string& segment_name) {
    return impl_->removeLocalSegment(segment_name);
}

int TransferEngine::registerLocalMemory(void* addr, size_t length,
                                        const std::string& location,
                                        bool remote_accessible,
                                        bool update_metadata) {
    return impl_->registerLocalMemory(addr, length, location, remote_accessible,
                                      update_metadata);
}

int TransferEngine::unregisterLocalMemory(void* addr, bool update_metadata) {
    return impl_->unregisterLocalMemory(addr, update_metadata);
}

void* TransferEngine::allocateSharedMemory(size_t length) {
    return impl_->allocateSharedMemory(length);
}

int TransferEngine::freeSharedMemory(void* addr) {
    return impl_->freeSharedMemory(addr);
}

Status TransferEngine::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest>& entries) {
    return impl_->submitTransfer(batch_id, entries);
}

Status TransferEngine::submitTransferWithNotify(
    BatchID batch_id, const std::vector<TransferRequest>& entries,
    TransferMetadata::NotifyDesc notify_msg) {
    return impl_->submitTransferWithNotify(batch_id, entries, notify_msg);
}

#ifdef ENABLE_MULTI_PROTOCOL
// Multi-protocol API (only available when ENABLE_MULTI_PROTOCOL is defined)
int TransferEngine::mp_registerLocalMemory(
    std::unordered_map<std::string, std::vector<RegisteredBuffer>>&
        buffer_map) {
    return impl_->mp_registerLocalMemory(buffer_map);
}

int TransferEngine::mp_unregisterLocalMemory(
    std::unordered_map<std::string, std::vector<RegisteredBuffer>>&
        buffer_map) {
    return impl_->mp_unregisterLocalMemory(buffer_map);
}

Status TransferEngine::mp_submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest>& entries,
    std::string& proto) {
    return impl_->mp_submitTransfer(batch_id, entries, proto);
}

Status TransferEngine::mp_submitTransferWithNotify(
    BatchID batch_id, const std::vector<TransferRequest>& entries,
    TransferMetadata::NotifyDesc notify_msg, std::string& proto) {
    return impl_->mp_submitTransferWithNotify(batch_id, entries, notify_msg,
                                              proto);
}
#endif

int TransferEngine::registerLocalMemoryBatch(
    const std::vector<BufferEntry>& buffer_list, const std::string& location) {
    return impl_->registerLocalMemoryBatch(buffer_list, location);
}

int TransferEngine::unregisterLocalMemoryBatch(
    const std::vector<void*>& addr_list) {
    return impl_->unregisterLocalMemoryBatch(addr_list);
}

BatchID TransferEngine::allocateBatchID(size_t batch_size) {
    return impl_->allocateBatchID(batch_size);
}

Status TransferEngine::freeBatchID(BatchID batch_id) {
    return impl_->freeBatchID(batch_id);
}

int TransferEngine::getNotifies(
    std::vector<TransferMetadata::NotifyDesc>& notifies) {
    return impl_->getNotifies(notifies);
}

int TransferEngine::sendNotifyByID(SegmentID target_id,
                                   TransferMetadata::NotifyDesc notify_msg) {
    return impl_->sendNotifyByID(target_id, notify_msg);
}

int TransferEngine::sendNotifyByName(std::string remote_agent,
                                     TransferMetadata::NotifyDesc notify_msg) {
    return impl_->sendNotifyByName(std::move(remote_agent), notify_msg);
}

PeerLiveness TransferEngine::probePeerAliveByID(SegmentID target_id) {
    return impl_->probePeerAliveByID(target_id) == 0
               ? PeerLiveness::Alive
               : PeerLiveness::Unreachable;
}

Status TransferEngine::getTransferStatus(BatchID batch_id, size_t task_id,
                                         TransferStatus& status) {
    return impl_->getTransferStatus(batch_id, task_id, status);
}

Status TransferEngine::getBatchTransferStatus(BatchID batch_id,
                                              TransferStatus& status) {
    return impl_->getBatchTransferStatus(batch_id, status);
}

Status TransferEngine::getNicLoadStats(std::vector<NicLoadStats>& stats) const {
    stats.clear();
    return Status::OK();
}

Transport* TransferEngine::getTransport(const std::string& proto) {
    return impl_->getTransport(proto);
}

#if (defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_MACA)) && \
    !defined(USE_CXI)
device::P2pTransport* TransferEngine::getOrCreateP2pTransport(int num_ranks) {
    return impl_->getOrCreateP2pTransport(num_ranks);
}

device::RdmaTransport* TransferEngine::getOrCreateRdmaTransport(
    const std::vector<std::string>& device_filter) {
    return impl_->getOrCreateRdmaTransport(device_filter);
}
#endif

#ifdef USE_NCCL_DEVICE
device::NcclTransport* TransferEngine::getOrCreateNcclTransport() {
    return impl_->getOrCreateNcclTransport();
}
#endif

bool TransferEngine::isTcpOnly() const { return impl_->isTcpOnly(); }

int TransferEngine::syncSegmentCache(const std::string& segment_name) {
    return impl_->syncSegmentCache(segment_name);
}

std::shared_ptr<TransferMetadata> TransferEngine::getMetadata() {
    return impl_->getMetadata();
}

bool TransferEngine::checkOverlap(void* addr, uint64_t length) {
    return impl_->checkOverlap(addr, length);
}

void TransferEngine::setAutoDiscover(bool auto_discover) {
    impl_->setAutoDiscover(auto_discover);
}

void TransferEngine::setAutoDiscover(const AutoDiscoverConfig& config) {
    impl_->setAutoDiscover(config);
}

void* TransferEngine::getBaseAddr() { return impl_->getBaseAddr(); }

void TransferEngine::setWhitelistFilters(std::vector<std::string>&& filters) {
    impl_->setWhitelistFilters(std::move(filters));
}

int TransferEngine::numContexts() const { return impl_->numContexts(); }

std::shared_ptr<Topology> TransferEngine::getLocalTopology() {
    return impl_->getLocalTopology();
}

std::string TransferEngine::getLocalTopologyString() {
    auto topo = impl_->getLocalTopology();
    return topo ? topo->toString() : "{}";
}

void TransferEngine::enableGracefulShutdown() {
    if (!shutdown_token_) {
        shutdown_token_ = registerTransferEngineShutdownToken(this);
    }
    installGracefulShutdownHandlers();
}

std::string TransferEngine::showLinks(bool json) const {
    if (!impl_) return "{}";
    return json ? buildShowLinksJson(impl_.get())
                : buildShowLinksReadable(impl_.get());
}

}  // namespace mooncake
#else
#include "transfer_engine.h"
#include "transfer_engine_impl.h"
#include "tent/transfer_engine.h"
#include "tent/common/config.h"
#include "tent/common/types.h"
#include "tent/runtime/topology.h"
#include "topology.h"
#if defined(USE_ASCEND) || defined(USE_ASCEND_DIRECT)
#include "config.h"
#endif

#include <mutex>
#include <utility>
#include "graceful_shutdown.h"
#include "show_links.h"

namespace mooncake {
namespace {

class TransferEngineShutdownToken : public ShutdownToken {
   public:
    explicit TransferEngineShutdownToken(TransferEngine* engine)
        : engine_(engine) {}

    void shutdown() override {
        TransferEngine* engine = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            engine = engine_;
            engine_ = nullptr;
        }
        if (engine) engine->freeEngine();
    }

    void detach() override {
        std::lock_guard<std::mutex> lock(mutex_);
        engine_ = nullptr;
    }

   private:
    std::mutex mutex_;
    TransferEngine* engine_;
};

std::shared_ptr<ShutdownToken> registerTransferEngineShutdownToken(
    TransferEngine* engine) {
    auto token = std::make_shared<TransferEngineShutdownToken>(engine);
    registerTokenForShutdown(token);
    return token;
}

void detachShutdownToken(std::shared_ptr<ShutdownToken>& token) {
    if (!token) return;
    token->detach();
    token.reset();
}

}  // namespace

TransferEngine::TransferEngine(bool auto_discover) {
    if (getenv("MC_USE_TENT") || getenv("MC_USE_TEV1")) {
        use_tent_ = true;
    }
    if (!use_tent_) {
        impl_ = std::make_shared<TransferEngineImpl>(auto_discover);
    }
}

TransferEngine::TransferEngine(bool auto_discover,
                               const std::vector<std::string>& filter) {
    if (getenv("MC_USE_TENT") || getenv("MC_USE_TEV1")) {
        use_tent_ = true;
    }
    if (use_tent_) {
        tent_device_filter_ = filter;
    } else {
        impl_ = std::make_shared<TransferEngineImpl>(auto_discover, filter);
    }
}

TransferEngine::TransferEngine(TransferEngine&& other) noexcept
    : impl_(std::move(other.impl_)),
      impl_tent_(std::move(other.impl_tent_)),
      shutdown_token_(nullptr),
      tent_device_filter_(std::move(other.tent_device_filter_)),
      use_tent_(other.use_tent_) {
    const bool shutdown_enabled = static_cast<bool>(other.shutdown_token_);
    detachShutdownToken(other.shutdown_token_);
    if (shutdown_enabled) {
        shutdown_token_ = registerTransferEngineShutdownToken(this);
        installGracefulShutdownHandlers();
    }
}

TransferEngine& TransferEngine::operator=(TransferEngine&& other) noexcept {
    if (this == &other) return *this;
    freeEngine();
    impl_ = std::move(other.impl_);
    impl_tent_ = std::move(other.impl_tent_);
    tent_device_filter_ = std::move(other.tent_device_filter_);
    use_tent_ = other.use_tent_;
    const bool shutdown_enabled = static_cast<bool>(other.shutdown_token_);
    detachShutdownToken(other.shutdown_token_);
    if (shutdown_enabled) {
        shutdown_token_ = registerTransferEngineShutdownToken(this);
        installGracefulShutdownHandlers();
    }
    return *this;
}

TransferEngine::~TransferEngine() { freeEngine(); }

static std::pair<std::string, std::string> parseConnectionStringInternal(
    const std::string& conn_string) {
    std::pair<std::string, std::string> result;
    std::string proto = "etcd";
    std::string domain;
    std::size_t pos = conn_string.find("://");

    if (pos != std::string::npos) {
        proto = conn_string.substr(0, pos);
        domain = conn_string.substr(pos + 3);
    } else if (conn_string == P2PHANDSHAKE) {
        proto = "";
        domain = P2PHANDSHAKE;
    } else {
        domain = conn_string;
    }

    result.first = proto;
    result.second = domain;
    return result;
}

std::shared_ptr<mooncake::tent::Config> TransferEngine::buildTentConfig(
    const std::string& metadata_conn_string,
    const std::string& local_server_name) const {
    auto config = std::make_shared<mooncake::tent::Config>();
    if (!local_server_name.empty())
        config->set("local_segment_name", local_server_name);
    if (metadata_conn_string == P2PHANDSHAKE) {
        config->set("metadata_type", "p2p");
    } else {
        auto [type, servers] =
            parseConnectionStringInternal(metadata_conn_string);
        if (!type.empty()) config->set("metadata_type", type);
        if (!servers.empty()) config->set("metadata_servers", servers);
    }
    if (!tent_device_filter_.empty()) {
        config->set("topology/rdma_whitelist", tent_device_filter_);
    }
    return config;
}

int TransferEngine::init(const std::string& metadata_conn_string,
                         const std::string& local_server_name,
                         const std::string& ip_or_host_name,
                         uint64_t rpc_port) {
    return init(metadata_conn_string, local_server_name, ip_or_host_name,
                rpc_port, "");
}

int TransferEngine::init(const std::string& metadata_conn_string,
                         const std::string& local_server_name,
                         const std::string& ip_or_host_name, uint64_t rpc_port,
                         const std::string& protocol) {
    if (!use_tent_) {
        return impl_->init(metadata_conn_string, local_server_name,
                           ip_or_host_name, rpc_port);
    } else {
        auto config = buildTentConfig(metadata_conn_string, local_server_name);
        if (protocol == "tcp") {
            mooncake::tent::ConfigHelper::forceTcp(*config);
            if (!std::getenv("MC_FORCE_TCP")) {
                LOG(INFO) << "protocol=tcp, forcing TENT memory transfers to "
                             "use TCP";
            }
        }
#if defined(USE_ASCEND) || defined(USE_ASCEND_DIRECT)
        // Store still constructs TENT through this classic init() shim, not
        // tent::TransferEngine(Config) directly. Copy Dummy-real flags until
        // Store creates the native engine itself.
        if (globalConfig().ascend_agent_mode) {
            config->set("transports/ascend_direct/agent_mode", true);
        }
        if (globalConfig().ascend_store_te_init) {
            config->set("transports/ascend_direct/store_te_init", true);
            if (globalConfig().ascend_use_fabric_mem) {
                config->set("transports/ascend_direct/fabric_mem", true);
            }
        }
#endif
        impl_tent_ = std::make_shared<mooncake::tent::TransferEngine>(config);
        return impl_tent_->available() ? 0 : 1;
    }
}

int TransferEngine::freeEngine() {
    detachShutdownToken(shutdown_token_);
    if (!use_tent_ && impl_) {
        if (impl_.use_count() == 1) impl_->freeEngine();
        impl_.reset();
    } else {
        impl_tent_.reset();
    }
    return 0;
}

Transport* TransferEngine::installTransport(const std::string& proto,
                                            void** args) {
    if (use_tent_) {
        static bool g_present = false;
        if (!g_present) {
            LOG(INFO) << "installTransport not used by TENT";
            g_present = true;
        }
        return nullptr;
    } else {
        return impl_->installTransport(proto, args);
    }
}

int TransferEngine::uninstallTransport(const std::string& proto) {
    if (use_tent_)
        return 0;
    else
        return impl_->uninstallTransport(proto);
}

std::string TransferEngine::getLocalIpAndPort() {
    if (use_tent_) {
        // Store handshake and openSegment must use the advertised segment
        // name, not a reconstructed host:port that can disagree with it.
        auto name = impl_tent_->getSegmentName();
        if (!name.empty()) {
            return name;
        }
        return impl_tent_->getRpcServerAddress() + ":" +
               std::to_string(impl_tent_->getRpcServerPort());
    } else
        return impl_->getLocalIpAndPort();
}

int TransferEngine::getRpcPort() {
    if (use_tent_) {
        return impl_tent_->getRpcServerPort();
    } else
        return impl_->getRpcPort();
}

SegmentHandle TransferEngine::openSegment(const std::string& segment_name) {
    if (use_tent_) {
        SegmentHandle handle;
        auto status = impl_tent_->openSegment(handle, segment_name);
        if (!status.ok())
            return static_cast<SegmentHandle>(ERR_INVALID_ARGUMENT);
        return handle;
    } else
        return impl_->openSegment(segment_name);
}

Status TransferEngine::CheckSegmentStatus(SegmentID sid) {
    if (use_tent_) {
        // TENT owns its segment cache, so actively probe the peer instead of
        // reporting OK unconditionally. Returning OK here left classic callers
        // (e.g. the Python wrapper's handle_map_) holding a dead peer's cached
        // handle forever, because they only evict the handle on a non-OK
        // status. A stale/unknown handle or an unreachable peer makes
        // probePeerAliveByID fail (invalid handle, empty RPC address, or RPC
        // error); surface that as a non-OK status so the caller closes and
        // re-opens the segment on the next transfer. Refs #3995
        // (P0-stale-handle).
        auto probe_status = impl_tent_->probePeerAliveByID(sid);
        if (probe_status.ok()) return Status::OK();
        return Status::Endpoint(std::string(probe_status.message()));
    } else
        return impl_->CheckSegmentStatus(sid);
}

int TransferEngine::closeSegment(SegmentHandle handle) {
    if (use_tent_) {
        auto status = impl_tent_->closeSegment(handle);
        return (int)status.code();
    } else
        return impl_->closeSegment(handle);
}

int TransferEngine::removeLocalSegment(const std::string& segment_name) {
    if (use_tent_)
        return 0;
    else
        return impl_->removeLocalSegment(segment_name);
}

int TransferEngine::registerLocalMemory(void* addr, size_t length,
                                        const std::string& location,
                                        bool remote_accessible,
                                        bool update_metadata) {
    if (use_tent_) {
        mooncake::tent::MemoryOptions option;
        if (!location.empty() && location != kWildcardLocation)
            option.location = location;
        auto status = impl_tent_->registerLocalMemory(addr, length, option);
        return (int)status.code();
    } else
        return impl_->registerLocalMemory(addr, length, location,
                                          remote_accessible, update_metadata);
}

int TransferEngine::unregisterLocalMemory(void* addr, bool update_metadata) {
    if (use_tent_) {
        auto status = impl_tent_->unregisterLocalMemory(addr);
        return (int)status.code();
    } else
        return impl_->unregisterLocalMemory(addr, update_metadata);
}

void* TransferEngine::allocateSharedMemory(size_t length) {
    if (use_tent_) {
        LOG(WARNING) << "allocateSharedMemory is classic TE only; use TENT "
                        "allocateLocalMemory with SHM enabled";
        return nullptr;
    }
    return impl_->allocateSharedMemory(length);
}

int TransferEngine::freeSharedMemory(void* addr) {
    if (use_tent_) return ERR_NOT_IMPLEMENTED;
    return impl_->freeSharedMemory(addr);
}

int TransferEngine::registerLocalMemoryBatch(
    const std::vector<BufferEntry>& buffer_list, const std::string& location) {
    if (use_tent_) {
        mooncake::tent::MemoryOptions option;
        if (!location.empty() && location != kWildcardLocation)
            option.location = location;
        std::vector<void*> addr_list;
        std::vector<size_t> size_list;
        for (auto& buffer : buffer_list) {
            addr_list.push_back(buffer.addr);
            size_list.push_back(buffer.length);
        }
        auto status =
            impl_tent_->registerLocalMemory(addr_list, size_list, option);
        return (int)status.code();
    } else {
        return impl_->registerLocalMemoryBatch(buffer_list, location);
    }
}

int TransferEngine::unregisterLocalMemoryBatch(
    const std::vector<void*>& addr_list) {
    if (use_tent_) {
        auto status = impl_tent_->unregisterLocalMemory(addr_list);
        return (int)status.code();
    } else {
        return impl_->unregisterLocalMemoryBatch(addr_list);
    }
}

BatchID TransferEngine::allocateBatchID(size_t batch_size) {
    if (use_tent_) {
        const auto batch_id = impl_tent_->allocateBatch(batch_size);
        return batch_id == 0 ? INVALID_BATCH_ID : batch_id;
    } else {
        return impl_->allocateBatchID(batch_size);
    }
}

Status TransferEngine::freeBatchID(BatchID batch_id) {
    if (use_tent_) {
        auto status = impl_tent_->freeBatch(batch_id);
        if (!status.ok())
            return Status::Context(status.ToString());
        else
            return Status::OK();
    } else {
        return impl_->freeBatchID(batch_id);
    }
}

Status TransferEngine::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest>& entries) {
    if (use_tent_) {
        std::vector<mooncake::tent::Request> requests;
        for (auto& item : entries) {
            mooncake::tent::Request req;
            req.opcode = (mooncake::tent::Request::OpCode)(int)item.opcode;
            req.length = item.length;
            req.source = item.source;
            req.target_id = item.target_id;
            req.target_offset = item.target_offset;
            req.transport_hint =
                mooncake::tent::c_to_transport_hint(item.transport_hint);
            requests.push_back(req);
        }
        auto status = impl_tent_->submitTransfer(batch_id, requests);
        if (!status.ok())
            return Status::Context(status.ToString());
        else
            return Status::OK();
    } else {
        return impl_->submitTransfer(batch_id, entries);
    }
}

Status TransferEngine::submitTransferWithNotify(
    BatchID batch_id, const std::vector<TransferRequest>& entries,
    TransferMetadata::NotifyDesc notify_msg) {
    if (use_tent_) {
        std::vector<mooncake::tent::Request> requests;
        for (auto& item : entries) {
            mooncake::tent::Request req;
            req.opcode = (mooncake::tent::Request::OpCode)(int)item.opcode;
            req.length = item.length;
            req.source = item.source;
            req.target_id = item.target_id;
            req.target_offset = item.target_offset;
            req.transport_hint =
                mooncake::tent::c_to_transport_hint(item.transport_hint);
            requests.push_back(req);
        }
        mooncake::tent::Notification notifi;
        notifi.name = notify_msg.name;
        notifi.msg = notify_msg.notify_msg;
        auto status = impl_tent_->submitTransfer(batch_id, requests, notifi);
        if (!status.ok())
            return Status::Context(status.ToString());
        else
            return Status::OK();
    } else {
        return impl_->submitTransferWithNotify(batch_id, entries, notify_msg);
    }
}

int TransferEngine::getNotifies(
    std::vector<TransferMetadata::NotifyDesc>& notifies) {
    if (use_tent_) {
        std::vector<mooncake::tent::Notification> notifi_list;
        auto status = impl_tent_->receiveNotification(notifi_list);
        for (auto& entry : notifi_list) {
            TransferMetadata::NotifyDesc desc;
            desc.name = entry.name;
            desc.notify_msg = entry.msg;
            notifies.push_back(desc);
        }
        return (int)status.code();
    } else
        return impl_->getNotifies(notifies);
}

int TransferEngine::sendNotifyByID(SegmentID target_id,
                                   TransferMetadata::NotifyDesc notify_msg) {
    if (use_tent_) {
        mooncake::tent::Notification notifi;
        notifi.name = notify_msg.name;
        notifi.msg = notify_msg.notify_msg;
        auto status = impl_tent_->sendNotification(target_id, notifi);
        return (int)status.code();
    } else
        return impl_->sendNotifyByID(target_id, notify_msg);
}

int TransferEngine::sendNotifyByName(std::string remote_agent,
                                     TransferMetadata::NotifyDesc notify_msg) {
    if (use_tent_) {
        mooncake::tent::Notification notifi;
        notifi.name = notify_msg.name;
        notifi.msg = notify_msg.notify_msg;
        SegmentHandle handle;
        auto status = impl_tent_->openSegment(handle, remote_agent);
        if (!status.ok()) return (int)status.code();
        status = impl_tent_->sendNotification(handle, notifi);
        impl_tent_->closeSegment(handle);
        return (int)status.code();
    } else
        return impl_->sendNotifyByName(std::move(remote_agent), notify_msg);
}

PeerLiveness TransferEngine::probePeerAliveByID(SegmentID target_id) {
    if (use_tent_) {
        auto status = impl_tent_->probePeerAliveByID(target_id);
        return status.ok() ? PeerLiveness::Alive : PeerLiveness::Unreachable;
    }
    return impl_->probePeerAliveByID(target_id) == 0
               ? PeerLiveness::Alive
               : PeerLiveness::Unreachable;
}

Status TransferEngine::getTransferStatus(BatchID batch_id, size_t task_id,
                                         TransferStatus& status) {
    if (use_tent_) {
        mooncake::tent::TransferStatus tent_status;
        auto s = impl_tent_->getTransferStatus(batch_id, task_id, tent_status);
        status.s = (TransferStatusEnum)(int)tent_status.s;
        status.transferred_bytes = tent_status.transferred_bytes;
        if (!s.ok())
            return Status::Context(s.ToString());
        else
            return Status::OK();
    } else {
        return impl_->getTransferStatus(batch_id, task_id, status);
    }
}

Status TransferEngine::getBatchTransferStatus(BatchID batch_id,
                                              TransferStatus& status) {
    if (use_tent_) {
        mooncake::tent::TransferStatus tent_status;
        auto s = impl_tent_->getTransferStatus(batch_id, tent_status);
        status.s = (TransferStatusEnum)(int)tent_status.s;
        status.transferred_bytes = tent_status.transferred_bytes;
        if (!s.ok())
            return Status::Context(s.ToString());
        else
            return Status::OK();
    } else
        return impl_->getBatchTransferStatus(batch_id, status);
}

Status TransferEngine::getNicLoadStats(std::vector<NicLoadStats>& stats) const {
    stats.clear();
    if (use_tent_) {
        std::vector<mooncake::tent::NicLoadStats> tent_stats;
        auto status = impl_tent_->getNicLoadStats(tent_stats);
        if (!status.ok()) return Status::Context(status.ToString());
        stats.reserve(tent_stats.size());
        for (const auto& stat : tent_stats) {
            NicLoadStats load_stats;
            load_stats.device_name = stat.device_name;
            load_stats.inflight_bytes = stat.inflight_bytes;
            load_stats.ewma_bandwidth_bps = stat.ewma_bandwidth_bps;
            stats.push_back(load_stats);
        }
    }
    return Status::OK();
}

Transport* TransferEngine::getTransport(const std::string& proto) {
    if (use_tent_)
        return nullptr;
    else
        return impl_->getTransport(proto);
}

#if (defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_MACA)) && \
    !defined(USE_CXI)
device::P2pTransport* TransferEngine::getOrCreateP2pTransport(int num_ranks) {
    if (use_tent_) return nullptr;
    return impl_->getOrCreateP2pTransport(num_ranks);
}

device::RdmaTransport* TransferEngine::getOrCreateRdmaTransport(
    const std::vector<std::string>& device_filter) {
    if (use_tent_) return nullptr;
    return impl_->getOrCreateRdmaTransport(device_filter);
}
#endif

#ifdef USE_NCCL_DEVICE
device::NcclTransport* TransferEngine::getOrCreateNcclTransport() {
    if (use_tent_) return nullptr;
    return impl_->getOrCreateNcclTransport();
}
#endif

bool TransferEngine::isTcpOnly() const {
    if (use_tent_)
        // TENT already rejects TCP loopback transfers when MC_STORE_MEMCPY
        // is disabled, so auto-enabling memcpy is unnecessary in TENT mode.
        return false;
    else
        return impl_->isTcpOnly();
}

int TransferEngine::syncSegmentCache(const std::string& segment_name) {
    if (use_tent_)
        return 0;
    else
        return impl_->syncSegmentCache(segment_name);
}

std::shared_ptr<TransferMetadata> TransferEngine::getMetadata() {
    if (use_tent_) {
        LOG(WARNING) << "API deprecated in Mooncake TENT";
        return nullptr;
    } else
        return impl_->getMetadata();
}

bool TransferEngine::checkOverlap(void* addr, uint64_t length) {
    if (!use_tent_) return impl_->checkOverlap(addr, length);
    return false;
}

void TransferEngine::setAutoDiscover(bool auto_discover) {
    if (!use_tent_) impl_->setAutoDiscover(auto_discover);
}

void TransferEngine::setAutoDiscover(const AutoDiscoverConfig& config) {
    if (!use_tent_) impl_->setAutoDiscover(config);
}

void TransferEngine::setWhitelistFilters(std::vector<std::string>&& filters) {
    if (!use_tent_) {
        impl_->setWhitelistFilters(std::move(filters));
    } else if (!impl_tent_) {
        tent_device_filter_ = std::move(filters);
    } else {
        LOG(WARNING) << "Cannot change the TENT RDMA device filter after init";
    }
}

int TransferEngine::numContexts() const {
    if (use_tent_)
        return 1;  // placeholder
    else
        return impl_->numContexts();
}

std::shared_ptr<Topology> TransferEngine::getLocalTopology() {
    if (use_tent_) {
        // Classic Topology only supports a 2-tier priority matrix. Prefer
        // getLocalTopologyString() for the full TENT nics/mems (rank0/1/2)
        // representation. This method still projects into the classic format
        // for existing C++ callers.
        auto classic = std::make_shared<Topology>();
        if (!impl_tent_ || !impl_tent_->available()) return classic;
        auto tent_topo = impl_tent_->getLocalTopology();
        if (!tent_topo) return classic;

        Json::Value root(Json::objectValue);
        for (size_t mi = 0; mi < tent_topo->getMemCount(); ++mi) {
            const auto* mem =
                tent_topo->getMemEntry(static_cast<tent::Topology::MemID>(mi));
            if (!mem) continue;
            if (mem->name.empty() || mem->name == tent::kWildcardLocation) {
                continue;
            }
            Json::Value preferred(Json::arrayValue);
            Json::Value avail(Json::arrayValue);
            for (auto id : mem->device_list[0]) {
                preferred.append(tent_topo->getNicName(id));
            }
            for (size_t rank = 1; rank < tent::Topology::DevicePriorityRanks;
                 ++rank) {
                for (auto id : mem->device_list[rank]) {
                    avail.append(tent_topo->getNicName(id));
                }
            }
            Json::Value entry(Json::arrayValue);
            entry.append(preferred);
            entry.append(avail);
            root[mem->name] = entry;
        }

        Json::StreamWriterBuilder builder;
        builder["indentation"] = "";
        const std::string json = Json::writeString(builder, root);
        if (classic->parse(json)) {
            LOG(WARNING) << "Failed to translate TENT topology to classic "
                            "priority matrix";
            classic->clear();
        }
        return classic;
    } else {
        return impl_->getLocalTopology();
    }
}

std::string TransferEngine::getLocalTopologyString() {
    if (use_tent_) {
        if (!impl_tent_ || !impl_tent_->available()) return "{}";
        return impl_tent_->getLocalTopologyString();
    }
    auto topo = impl_ ? impl_->getLocalTopology() : nullptr;
    return topo ? topo->toString() : "{}";
}

void* TransferEngine::getBaseAddr() {
    if (use_tent_) {
        // TENT version does not support CXL base address
        return nullptr;
    } else
        return impl_->getBaseAddr();
}

void TransferEngine::enableGracefulShutdown() {
    if (!shutdown_token_) {
        shutdown_token_ = registerTransferEngineShutdownToken(this);
    }
    installGracefulShutdownHandlers();
}

std::string TransferEngine::showLinks(bool json) const {
    if (use_tent_ || !impl_) {
        return json ? "{}" : "(TENT mode or not initialized)";
    }
    return json ? buildShowLinksJson(impl_.get())
                : buildShowLinksReadable(impl_.get());
}

}  // namespace mooncake
#endif

namespace mooncake {

class TransferEngine::ScatterTransferOperation::Impl {
   public:
    struct Backend {
        std::shared_ptr<TransferEngineImpl> legacy;
#ifdef USE_TENT
        std::shared_ptr<mooncake::tent::TransferEngine> tent;
#endif
    };

    Impl(TransferEngine& engine, Backend backend,
         const std::vector<ScatterTransferRange>& ranges,
         bool synchronous_gather)
        : backend_(std::move(backend)),
          synchronous_gather_(synchronous_gather) {
        callbacks_.reserve(ranges.size());
        batch_callbacks_.reserve(ranges.size());
        for (const auto& range : ranges) {
            callbacks_.push_back(range.on_fragment_complete);
            batch_callbacks_.push_back(range.on_fragment_batch_complete);
        }
        build(engine, ranges);
    }

    ~Impl() { wait(); }

    Status wait() {
        while (!completed_) {
            poll();
            if (!completed_) std::this_thread::sleep_for(kPollInterval);
        }
        return aggregate_status_;
    }

    Status waitFor(std::chrono::nanoseconds timeout) {
        const auto now = std::chrono::steady_clock::now();
        const auto until_max =
            std::chrono::steady_clock::time_point::max() - now;
        const auto max_timeout =
            std::chrono::duration_cast<std::chrono::nanoseconds>(until_max);
        auto deadline = now;
        if (timeout > std::chrono::nanoseconds::zero()) {
            deadline = timeout >= max_timeout
                           ? std::chrono::steady_clock::time_point::max()
                           : now + timeout;
        }
        while (!completed_) {
            poll();
            if (completed_) break;
            if (std::chrono::steady_clock::now() >= deadline)
                return Status::Clock("scatter transfer wait timed out");
            std::this_thread::sleep_for(kPollInterval);
        }
        return aggregate_status_;
    }

   private:
    static constexpr auto kPollInterval = std::chrono::microseconds(10);

    bool useTent() const {
#ifdef USE_TENT
        return static_cast<bool>(backend_.tent);
#else
        return false;
#endif
    }

    int closeSegment(SegmentHandle handle) {
#ifdef USE_TENT
        if (backend_.tent)
            return static_cast<int>(backend_.tent->closeSegment(handle).code());
#endif
        return backend_.legacy->closeSegment(handle);
    }

    Status getStatus(BatchID batch_id, size_t task_id, TransferStatus& status) {
#ifdef USE_TENT
        if (backend_.tent) {
            mooncake::tent::TransferStatus tent_status;
            auto result = backend_.tent->getTransferStatus(batch_id, task_id,
                                                           tent_status);
            if (!result.ok()) return Status::Context(result.ToString());
            status.s = static_cast<TransferStatusEnum>(tent_status.s);
            status.transferred_bytes = tent_status.transferred_bytes;
            return Status::OK();
        }
#endif
        return backend_.legacy->getTransferStatus(batch_id, task_id, status);
    }

    Status freeBatch(BatchID batch_id) {
#ifdef USE_TENT
        if (backend_.tent) {
            auto result = backend_.tent->freeBatch(batch_id);
            return result.ok() ? Status::OK()
                               : Status::Context(result.ToString());
        }
#endif
        return backend_.legacy->freeBatchID(batch_id);
    }

    void remember(const Status& status) {
        if (aggregate_status_.ok() && !status.ok()) aggregate_status_ = status;
    }

    Status closeSegments(Status status) {
        for (const auto& entry : segment_handles_) {
            if (entry.second ==
                static_cast<SegmentHandle>(ERR_INVALID_ARGUMENT))
                continue;
            if (closeSegment(entry.second) != 0 && status.ok())
                status =
                    Status::Context("failed to close scatter transfer segment");
        }
        segment_handles_.clear();
        return status;
    }

    void finish() {
        aggregate_status_ = closeSegments(aggregate_status_);
        completed_ = true;
        callbacks_.clear();
        batch_callbacks_.clear();
        requests_.clear();
        request_fragment_runs_.clear();
        gather_fragment_runs_.clear();
        done_.clear();
        task_sizes_.clear();
        backend_ = {};
    }

    void complete(size_t range_index, size_t fragment_index,
                  const Status& status) {
        remember(status);
        const auto& callback = callbacks_[range_index];
        if (!callback) return;
        try {
            callback(fragment_index, status);
        } catch (...) {
            LOG(ERROR) << "scatter transfer callback failed";
            remember(Status::Context("scatter transfer callback failed"));
        }
    }

    struct FragmentRun {
        size_t range;
        size_t begin;
        size_t end;
    };

    void completeRequest(size_t request_index, const Status& status) {
        done_[request_index] = true;
        for (const auto& run : request_fragment_runs_[request_index])
            completeBatch(run.range, run.begin, run.end, status);
    }

    void completeBatch(size_t range_index, size_t begin, size_t end,
                       const Status& status) {
        const auto& callback = batch_callbacks_[range_index];
        if (!callback) {
            for (size_t fragment = begin; fragment < end; ++fragment)
                complete(range_index, fragment, status);
            return;
        }
        remember(status);
        try {
            callback(begin, end, status);
        } catch (...) {
            LOG(ERROR) << "scatter transfer batch callback failed";
            remember(Status::Context("scatter transfer batch callback failed"));
        }
    }

    void completeRequests(size_t begin, size_t end, const Status& status) {
        FragmentRun pending{};
        bool has_pending = false;
        const auto flush = [&] {
            if (!has_pending) return;
            completeBatch(pending.range, pending.begin, pending.end, status);
            has_pending = false;
        };
        for (size_t request = begin; request < end; ++request) {
            done_[request] = true;
            for (const auto& run : request_fragment_runs_[request]) {
                if (has_pending && pending.range == run.range &&
                    pending.end == run.begin) {
                    pending.end = run.end;
                } else {
                    flush();
                    pending = run;
                    has_pending = true;
                }
            }
        }
        flush();
    }

    void failPending(const Status& status) {
        for (size_t i = 0; i < request_fragment_runs_.size(); ++i) {
            if (!done_[i]) completeRequest(i, status);
        }
        remaining_ = 0;
    }

    void requestAbort(const Status& status) {
        remember(status);
        if (abort_requested_) return;
        abort_requested_ = true;
#ifdef USE_TENT
        if (!backend_.tent) return;
        for (size_t i = 0; i < requests_.size(); ++i) {
            if (done_[i]) continue;
            auto cancel_status = backend_.tent->cancelTransfer(batch_id_, i);
            if (!cancel_status.ok() && !cancel_status.IsNotImplemented()) {
                LOG(WARNING) << "failed to cancel scatter transfer task " << i
                             << ": " << cancel_status.ToString();
            }
        }
#endif
    }

    struct GatherPayload {
        std::vector<TransferEngineImpl::ScatterSpan> spans;
        std::vector<uint32_t> span_fragment_counts;
        std::string fixed_relative_offsets;
        uint32_t fixed_span_length = 0;
        bool fixed_offsets_in_destination = false;
    };

    struct GatherTask {
        std::string peer;
        uint64_t destination = 0;
        void* local_buffer = nullptr;
        size_t local_capacity = 0;
        uint64_t source_region_base = 0;
        uint64_t source_region_size = 0;
        uint64_t source_base = 0;
        uint64_t source_size = 0;
        uint64_t total_bytes = 0;
        size_t encoded_span_bytes = 0;
        size_t command_span_budget = 0;
        size_t chunk_bytes = 0;
        uint8_t pipeline_depth = 1;
        bool compact_plan = true;
        uint32_t fixed_span_length = 0;
        size_t fragment_count = 0;
        size_t max_span_length = 0;
        bool direct_contiguous = true;
        bool mixed_source_regions = false;
        size_t range_index = std::numeric_limits<size_t>::max();
        std::vector<TransferEngineImpl::ScatterSpan> spans;
        std::vector<uint32_t> span_fragment_counts;
        std::shared_ptr<const GatherPayload> payload;
        size_t span_begin = 0;
        size_t span_end = 0;
        std::vector<FragmentRun> fragment_runs;
        std::unique_ptr<PreparedHandshakeCommand> prepared_command;
    };

    struct DirectTask {
        std::string peer;
        uint64_t destination = 0;
        void* local_buffer = nullptr;
        size_t local_capacity = 0;
        uint64_t source_region_base = 0;
        uint64_t source_region_size = 0;
        uint64_t source = 0;
        size_t length = 0;
        std::vector<FragmentRun> fragment_runs;
    };

    struct GatherResult {
        std::vector<FragmentRun> fragment_runs;
        // Only populated when the command path falls back to direct reads.
        // A gathered span may represent more than one original fragment, so
        // keep the span-to-fragment cardinality for per-span completion.
        std::vector<uint32_t> span_fragment_counts;
        std::vector<Status> span_statuses;
        Status status;
    };

    size_t buildScatterPlan(const std::vector<ScatterTransferRange>& ranges,
                            size_t total_fragments,
                            std::vector<std::vector<bool>>& planned_fragments,
                            std::vector<DirectTask>& direct_tasks) {
        const bool trace_planning = VLOG_IS_ON(1);
        const auto plan_started = trace_planning
                                      ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
        const auto varintBytes = [](uint64_t value) {
            size_t bytes = 1;
            while (value >= 0x80) {
                value >>= 7;
                ++bytes;
            }
            return bytes;
        };
        if (useTent() || !backend_.legacy || ranges.empty()) return 0;
        const auto profile = backend_.legacy->scatterTransportProfile();
        const size_t small_limit =
            TransferEngineImpl::scatterSmallFragmentLimit(profile);
        if (small_limit == 0) return 0;
        constexpr size_t kMaxTaskFragments = 131072;

        bool all_gather_work = false;

        std::vector<GatherTask> tasks;
        tasks.reserve(profile.pipeline_width);
        GatherTask candidate;
        std::unique_ptr<PreparedHandshakeCommand> prepared_command;
        std::string prepared_peer;
        uint64_t expected_destination = 0;
        uint64_t expected_source = 0;
        const auto appendRun = [](std::vector<FragmentRun>& runs,
                                  const FragmentRun& run) {
            if (!runs.empty() && runs.back().range == run.range &&
                runs.back().end == run.begin) {
                runs.back().end = run.end;
            } else {
                runs.push_back(run);
            }
        };
        const auto appendDirectTask = [&](DirectTask task) {
            if (!direct_tasks.empty()) {
                auto& previous = direct_tasks.back();
                const bool coalesces =
                    previous.peer == task.peer &&
                    previous.local_buffer == task.local_buffer &&
                    previous.local_capacity == task.local_capacity &&
                    previous.source_region_base == task.source_region_base &&
                    previous.source_region_size == task.source_region_size &&
                    previous.length <=
                        std::numeric_limits<size_t>::max() - task.length &&
                    previous.destination <=
                        std::numeric_limits<uint64_t>::max() -
                            previous.length &&
                    previous.destination + previous.length ==
                        task.destination &&
                    previous.source <= std::numeric_limits<uint64_t>::max() -
                                           previous.length &&
                    previous.source + previous.length == task.source;
                if (coalesces) {
                    previous.length += task.length;
                    for (const auto& run : task.fragment_runs)
                        appendRun(previous.fragment_runs, run);
                    return;
                }
            }
            direct_tasks.push_back(std::move(task));
        };
        const auto appendDirectCandidate = [&](const GatherTask& direct) {
            size_t run_index = 0;
            size_t run_offset = 0;
            uint64_t destination = direct.destination;
            for (size_t span_index = 0; span_index < direct.spans.size();
                 ++span_index) {
                DirectTask task{
                    .peer = direct.peer,
                    .destination = destination,
                    .local_buffer = direct.local_buffer,
                    .local_capacity = direct.local_capacity,
                    .source_region_base = direct.source_region_base,
                    .source_region_size = direct.source_region_size,
                    .source = direct.spans[span_index].source_address,
                    .length = direct.spans[span_index].length,
                    .fragment_runs = {},
                };
                size_t remaining = direct.span_fragment_counts[span_index];
                while (remaining != 0) {
                    DCHECK_LT(run_index, direct.fragment_runs.size());
                    const auto& run = direct.fragment_runs[run_index];
                    const size_t begin = run.begin + run_offset;
                    const size_t available = run.end - begin;
                    const size_t take = std::min(remaining, available);
                    appendRun(task.fragment_runs,
                              {run.range, begin, begin + take});
                    remaining -= take;
                    run_offset += take;
                    if (run_offset == run.end - run.begin) {
                        ++run_index;
                        run_offset = 0;
                    }
                }
                if (direct.mixed_source_regions)
                    direct_tasks.push_back(std::move(task));
                else
                    appendDirectTask(std::move(task));
                destination += direct.spans[span_index].length;
            }
            DCHECK_EQ(run_index, direct.fragment_runs.size());
            DCHECK_EQ(run_offset, 0);
        };
        const auto startCandidate =
            [&](const std::string& peer, uint64_t destination,
                void* local_buffer, size_t local_capacity,
                uint64_t source_region_base, uint64_t source_region_size,
                size_t range_index, uint64_t source, size_t length,
                size_t fragment_capacity) {
                candidate.peer = peer;
                candidate.destination = destination;
                candidate.local_buffer = local_buffer;
                candidate.local_capacity = local_capacity;
                candidate.source_region_base = source_region_base;
                candidate.source_region_size = source_region_size;
                candidate.source_base = source;
                candidate.source_size = length;
                candidate.compact_plan = length <= UINT32_MAX;
                candidate.direct_contiguous = true;
                candidate.range_index = range_index;
                candidate.command_span_budget =
                    backend_.legacy->scatterCommandSpanBudget(peer);
                fragment_capacity =
                    std::min(fragment_capacity, kMaxTaskFragments);
                candidate.spans.reserve(fragment_capacity);
                candidate.span_fragment_counts.reserve(fragment_capacity);
                candidate.fragment_runs.reserve(ranges.size());
            };
        const auto useAbsoluteSpanEncoding = [&] {
            if (!candidate.compact_plan) return;
            candidate.compact_plan = false;
            candidate.fixed_span_length = 0;
            candidate.encoded_span_bytes = 0;
            for (const auto& span : candidate.spans) {
                candidate.encoded_span_bytes +=
                    varintBytes(span.source_address) + varintBytes(span.length);
            }
        };
        auto flush = [&] {
            if (candidate.fragment_count == 0) return;
            const auto decision = TransferEngineImpl::planScatter(
                candidate.fragment_count, candidate.spans.size(),
                candidate.direct_contiguous ? 1 : candidate.spans.size(),
                candidate.total_bytes, profile);
            if (decision.gather &&
                backend_.legacy->supportsTransferCommand(candidate.peer)) {
                const bool single_command_pipeline =
                    all_gather_work && decision.pipeline_depth > 1;
                candidate.chunk_bytes =
                    single_command_pipeline
                        ? std::max<size_t>(
                              decision.chunk_bytes / decision.pipeline_depth,
                              candidate.max_span_length)
                        : decision.chunk_bytes;
                candidate.pipeline_depth =
                    single_command_pipeline ? decision.pipeline_depth : 1;
                if (!candidate.prepared_command && prepared_command &&
                    candidate.peer == prepared_peer) {
                    candidate.prepared_command = std::move(prepared_command);
                }
                const size_t shard_count =
                    single_command_pipeline
                        ? 1
                        : (all_gather_work
                               ? std::min<size_t>(decision.pipeline_depth,
                                                  candidate.spans.size())
                               : 1);
                auto payload = std::make_shared<GatherPayload>();
                payload->spans = std::move(candidate.spans);
                payload->span_fragment_counts =
                    std::move(candidate.span_fragment_counts);
                candidate.payload = payload;
                candidate.span_begin = 0;
                candidate.span_end = payload->spans.size();
                if (shard_count == 1) {
                    tasks.push_back(std::move(candidate));
                } else {
                    size_t span_begin = 0;
                    size_t run_index = 0;
                    size_t run_offset = 0;
                    uint64_t destination_offset = 0;
                    uint64_t assigned_bytes = 0;
                    size_t assigned_fragments = 0;
                    for (size_t shard_index = 0; shard_index < shard_count;
                         ++shard_index) {
                        const size_t shards_left = shard_count - shard_index;
                        const uint64_t target_bytes = candidate.total_bytes *
                                                      (shard_index + 1) /
                                                      shard_count;
                        size_t span_end = span_begin;
                        uint64_t shard_bytes = 0;
                        size_t shard_fragments = 0;
                        while (span_end < payload->spans.size()) {
                            if (span_end > span_begin &&
                                assigned_bytes + shard_bytes >= target_bytes &&
                                payload->spans.size() - span_end >=
                                    shards_left - 1) {
                                break;
                            }
                            shard_bytes += payload->spans[span_end].length;
                            shard_fragments +=
                                payload->span_fragment_counts[span_end];
                            ++span_end;
                            if (payload->spans.size() - span_end <
                                shards_left - 1) {
                                break;
                            }
                        }

                        GatherTask shard;
                        shard.peer = candidate.peer;
                        shard.destination =
                            candidate.destination + destination_offset;
                        shard.local_buffer = candidate.local_buffer;
                        shard.local_capacity = candidate.local_capacity;
                        shard.source_region_base = candidate.source_region_base;
                        shard.source_region_size = candidate.source_region_size;
                        shard.total_bytes = shard_bytes;
                        shard.command_span_budget =
                            candidate.command_span_budget;
                        shard.chunk_bytes = candidate.chunk_bytes;
                        shard.pipeline_depth = 1;
                        shard.compact_plan = candidate.compact_plan;
                        shard.fragment_count = shard_fragments;
                        shard.source_base = candidate.source_base;
                        shard.source_size = candidate.source_size;
                        if (shard_index == 0)
                            shard.prepared_command =
                                std::move(candidate.prepared_command);
                        shard.payload = payload;
                        shard.span_begin = span_begin;
                        shard.span_end = span_end;

                        size_t remaining = shard_fragments;
                        while (remaining != 0) {
                            const auto& run =
                                candidate.fragment_runs[run_index];
                            const size_t begin = run.begin + run_offset;
                            const size_t available = run.end - begin;
                            const size_t take = std::min(remaining, available);
                            shard.fragment_runs.push_back(
                                {run.range, begin, begin + take});
                            remaining -= take;
                            run_offset += take;
                            if (run_offset == run.end - run.begin) {
                                ++run_index;
                                run_offset = 0;
                            }
                        }
                        tasks.push_back(std::move(shard));
                        span_begin = span_end;
                        destination_offset += shard_bytes;
                        assigned_bytes += shard_bytes;
                        assigned_fragments += shard_fragments;
                    }
                    DCHECK_EQ(assigned_bytes, candidate.total_bytes);
                    DCHECK_EQ(assigned_fragments, candidate.fragment_count);
                }
            } else {
                appendDirectCandidate(candidate);
            }
            candidate = {};
            expected_destination = 0;
            expected_source = 0;
        };

        const auto tryBuildFixedPlan = [&] {
            if (total_fragments == 0 || total_fragments > kMaxTaskFragments)
                return false;

            GatherTask fixed;
            auto payload = std::make_shared<GatherPayload>();
            const size_t relative_offsets_bytes =
                total_fragments * sizeof(uint32_t);
            char* relative_offset_output = nullptr;
            const auto allocateOwnedOffsets = [&] {
                payload->fixed_relative_offsets.resize(relative_offsets_bytes);
                relative_offset_output = payload->fixed_relative_offsets.data();
            };
            fixed.fragment_runs.reserve(ranges.size());

            uint64_t fixed_expected_destination = 0;
            uint64_t fixed_expected_source = 0;
            uint64_t first_source = 0;
            size_t direct_request_count = 0;
            bool first_fragment = true;
            bool prepare_attempted = false;
            bool mixed_source_regions = false;
            const auto initializeFixed = [&](const auto& range) {
                fixed.peer = range.remote_segment;
                fixed.local_buffer = range.local_buffer;
                fixed.local_capacity = range.local_capacity;
                fixed.source_region_base = range.remote_base_offset;
                fixed.source_region_size = range.remote_size;
                fixed.source_base = range.remote_base_offset;
                fixed.source_size = range.remote_size;
                fixed.command_span_budget =
                    backend_.legacy->scatterCommandSpanBudget(fixed.peer);
                return fixed.command_span_budget >= sizeof(uint32_t) &&
                       fixed.command_span_budget - sizeof(uint32_t) >=
                           relative_offsets_bytes;
            };
            const auto prepareCommand = [&] {
                // Start the non-blocking command connection early enough for
                // the TCP handshake to overlap the remaining plan scan. The
                // old 4096-fragment cutoff left the common Engram lookup
                // (hundreds of 264-byte fragments) on the cold
                // getaddrinfo/connect path, even though the peer advertises
                // the reusable command protocol.
                constexpr size_t kPrepareCommandMinFragments = 128;
                if (prepare_attempted ||
                    total_fragments < kPrepareCommandMinFragments ||
                    fixed.fragment_count < total_fragments / 2 ||
                    direct_request_count <= 1 || fixed.command_span_budget == 0)
                    return;
                prepare_attempted = true;
                prepared_peer = fixed.peer;
                prepared_command =
                    backend_.legacy->prepareScatterCommand(fixed.peer);
            };

            if (ranges.size() == 1) {
                const auto& range = ranges.front();
                const size_t count = range.local_offsets.size();
                if (range.opcode != TransferRequest::READ ||
                    !range.local_buffer || range.remote_segment.empty() ||
                    range.remote_offsets.size() != count ||
                    range.lengths.size() != count || range.remote_size == 0 ||
                    range.remote_size > UINT32_MAX ||
                    range.remote_base_offset > UINT64_MAX - range.remote_size) {
                    return false;
                }
                if (!initializeFixed(range)) return false;

                const size_t length = range.lengths.front();
                if (length == 0 || length >= small_limit ||
                    length > UINT32_MAX || count > (512ULL << 20) / length) {
                    return false;
                }
                payload->fixed_span_length = static_cast<uint32_t>(length);
                fixed.total_bytes = count * length;

                const size_t first_local_offset = range.local_offsets.front();
                if (first_local_offset > range.local_capacity ||
                    fixed.total_bytes >
                        range.local_capacity - first_local_offset) {
                    return false;
                }
                fixed.destination = reinterpret_cast<uint64_t>(
                    static_cast<char*>(range.local_buffer) +
                    first_local_offset);
                if (fixed.destination > UINT64_MAX - fixed.total_bytes) {
                    return false;
                }
                const auto plan_overlaps = [&](std::span<const size_t> values) {
                    const auto plan_begin = fixed.destination;
                    const auto plan_end = plan_begin + relative_offsets_bytes;
                    const auto values_begin =
                        reinterpret_cast<uintptr_t>(values.data());
                    const auto values_bytes = values.size() * sizeof(size_t);
                    if (values_begin >
                        std::numeric_limits<uintptr_t>::max() - values_bytes) {
                        return true;
                    }
                    const auto values_end = values_begin + values_bytes;
                    return plan_begin < values_end && values_begin < plan_end;
                };
                // Inline the tiny offset table in the command.  The remote
                // fixed-plan encoding adds one RDMA READ for the table, which
                // is not amortized by a sub-16 KiB plan.
                constexpr size_t kRemotePlanMinBytes = 16ULL << 10;
                const bool can_stage_offsets_in_destination =
                    relative_offsets_bytes >= kRemotePlanMinBytes &&
                    relative_offsets_bytes <= fixed.total_bytes &&
                    !plan_overlaps(range.local_offsets) &&
                    !plan_overlaps(range.remote_offsets) &&
                    !plan_overlaps(range.lengths) &&
                    backend_.legacy->canUseRemoteScatterPlan(
                        fixed.peer, fixed.destination, relative_offsets_bytes);
                std::string* scratch_offsets = nullptr;
                if (can_stage_offsets_in_destination) {
                    thread_local std::string reusable_scratch_offsets;
                    reusable_scratch_offsets.resize(relative_offsets_bytes);
                    scratch_offsets = &reusable_scratch_offsets;
                    relative_offset_output = scratch_offsets->data();
                } else {
                    allocateOwnedOffsets();
                }

                const auto validate_local_layout = [&] {
                    size_t expected = first_local_offset;
                    for (size_t fragment_index = 0; fragment_index < count;
                         ++fragment_index) {
                        if (range.lengths[fragment_index] != length ||
                            range.local_offsets[fragment_index] != expected) {
                            return false;
                        }
                        expected += length;
                    }
                    return true;
                };
                auto& validation_worker = plannerValidationWorker();
                const bool validation_started =
                    validation_worker.start(validate_local_layout);
                const auto* remote_offsets = range.remote_offsets.data();
                const bool fixed_length_fits = length <= range.remote_size;
                const size_t max_remote_offset =
                    fixed_length_fits ? range.remote_size - length : 0;
                uint32_t invalid_remote_layout = 0;
                for (size_t fragment_index = 0; fragment_index < count;
                     ++fragment_index) {
                    const size_t remote_offset = remote_offsets[fragment_index];
                    invalid_remote_layout |=
                        !fixed_length_fits || remote_offset > max_remote_offset;
                    if (fragment_index == 0 ||
                        remote_offset !=
                            remote_offsets[fragment_index - 1] + length) {
                        ++direct_request_count;
                    }
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                    const uint32_t encoded_offset =
                        static_cast<uint32_t>(remote_offset);
                    std::memcpy(relative_offset_output +
                                    fragment_index * sizeof(uint32_t),
                                &encoded_offset, sizeof(encoded_offset));
#else
                    char* output = relative_offset_output +
                                   fragment_index * sizeof(uint32_t);
                    writeLittleEndian32(output,
                                        static_cast<uint32_t>(remote_offset));
#endif
                }
                relative_offset_output += relative_offsets_bytes;
                const bool local_layout_valid = validation_started
                                                    ? validation_worker.wait()
                                                    : validate_local_layout();
                if (!local_layout_valid || invalid_remote_layout != 0)
                    return false;

                // Only publish the fixed plan into the caller's destination
                // after both layouts are known to be valid. A non-contiguous
                // local layout must be able to fall back without modifying
                // gap bytes that are outside the requested fragments.
                if (can_stage_offsets_in_destination) {
                    std::memcpy(reinterpret_cast<void*>(fixed.destination),
                                scratch_offsets->data(),
                                relative_offsets_bytes);
                    payload->fixed_offsets_in_destination = true;
                    relative_offset_output =
                        reinterpret_cast<char*>(fixed.destination) +
                        relative_offsets_bytes;
                }
                fixed.fragment_count = count;
                prepareCommand();
                first_source = range.remote_base_offset + remote_offsets[0];
                fixed.fragment_runs.push_back({0, 0, count});
            } else {
                allocateOwnedOffsets();
                uint64_t source_window_base = UINT64_MAX;
                uint64_t source_window_end = 0;
                for (const auto& range : ranges) {
                    if (range.local_offsets.empty()) continue;
                    if (range.remote_size == 0 ||
                        range.remote_base_offset >
                            UINT64_MAX - range.remote_size) {
                        return false;
                    }
                    source_window_base =
                        std::min(source_window_base, range.remote_base_offset);
                    source_window_end =
                        std::max(source_window_end,
                                 range.remote_base_offset + range.remote_size);
                }
                if (source_window_base == UINT64_MAX ||
                    source_window_end - source_window_base > UINT32_MAX) {
                    return false;
                }
                for (size_t range_index = 0; range_index < ranges.size();
                     ++range_index) {
                    const auto& range = ranges[range_index];
                    const size_t count = range.local_offsets.size();
                    if (range.opcode != TransferRequest::READ ||
                        !range.local_buffer || range.remote_segment.empty() ||
                        range.remote_offsets.size() != count ||
                        range.lengths.size() != count ||
                        range.remote_size == 0 ||
                        range.remote_size > UINT32_MAX ||
                        range.remote_base_offset >
                            UINT64_MAX - range.remote_size) {
                        return false;
                    }
                    if (fixed.fragment_count == 0) {
                        if (!initializeFixed(range)) return false;
                        fixed.source_base = source_window_base;
                        fixed.source_size =
                            source_window_end - source_window_base;
                    } else if (fixed.peer != range.remote_segment ||
                               fixed.local_buffer != range.local_buffer ||
                               fixed.local_capacity != range.local_capacity) {
                        return false;
                    }
                    if (fixed.source_region_base != range.remote_base_offset ||
                        fixed.source_region_size != range.remote_size) {
                        mixed_source_regions = true;
                    }
                    for (size_t fragment_index = 0; fragment_index < count;
                         ++fragment_index) {
                        const size_t length = range.lengths[fragment_index];
                        const size_t local_offset =
                            range.local_offsets[fragment_index];
                        const size_t remote_offset =
                            range.remote_offsets[fragment_index];
                        if (length == 0 || length >= small_limit ||
                            length > UINT32_MAX ||
                            (payload->fixed_span_length != 0 &&
                             payload->fixed_span_length != length) ||
                            local_offset > range.local_capacity ||
                            length > range.local_capacity - local_offset ||
                            remote_offset > range.remote_size ||
                            length > range.remote_size - remote_offset ||
                            fixed.total_bytes > (512ULL << 20) - length) {
                            return false;
                        }
                        const uint64_t destination = reinterpret_cast<uint64_t>(
                            static_cast<char*>(range.local_buffer) +
                            local_offset);
                        const uint64_t source =
                            range.remote_base_offset + remote_offset;
                        if (destination > UINT64_MAX - length ||
                            source > UINT64_MAX - length ||
                            (!first_fragment &&
                             destination != fixed_expected_destination)) {
                            return false;
                        }
                        if (first_fragment) {
                            fixed.destination = destination;
                            first_source = source;
                            payload->fixed_span_length =
                                static_cast<uint32_t>(length);
                        }
                        if (first_fragment || source != fixed_expected_source)
                            ++direct_request_count;
                        writeLittleEndian32(
                            relative_offset_output,
                            static_cast<uint32_t>(source - fixed.source_base));
                        ++fixed.fragment_count;
                        prepareCommand();
                        fixed.total_bytes += length;
                        fixed_expected_destination = destination + length;
                        fixed_expected_source = source + length;
                        first_fragment = false;
                    }
                    if (count != 0)
                        fixed.fragment_runs.push_back({range_index, 0, count});
                }
            }
            if (fixed.fragment_count != total_fragments ||
                payload->fixed_span_length == 0 ||
                relative_offset_output !=
                    (payload->fixed_offsets_in_destination
                         ? reinterpret_cast<char*>(fixed.destination) +
                               relative_offsets_bytes
                         : payload->fixed_relative_offsets.data() +
                               payload->fixed_relative_offsets.size()))
                return false;

            const auto decision = TransferEngineImpl::planScatter(
                fixed.fragment_count, fixed.fragment_count,
                direct_request_count, fixed.total_bytes, profile);
            if (decision.gather &&
                backend_.legacy->supportsTransferCommand(fixed.peer)) {
                // Fixed plans have no owner-side span materialization cost.
                // Keep short Engram lookups in one bulk transfer, but use the
                // available lanes for medium requests where a single owner-side
                // pack otherwise serializes a few MiB before RDMA can overlap
                // it.
                constexpr size_t kMultiLaneMinBytes = 1ULL << 20;
                constexpr size_t kMultiLaneMaxBytes = 8ULL << 20;
                size_t chunk_bytes = decision.chunk_bytes;
                if (fixed.total_bytes >= kMultiLaneMinBytes &&
                    fixed.total_bytes <= kMultiLaneMaxBytes &&
                    decision.pipeline_depth >= 2) {
                    const size_t lanes = decision.pipeline_depth;
                    const size_t lane_bytes =
                        (fixed.total_bytes + lanes - 1) / lanes;
                    constexpr size_t kChunkAlignment = 64ULL << 10;
                    chunk_bytes = (lane_bytes + kChunkAlignment - 1) &
                                  ~(kChunkAlignment - 1);
                }
                uint8_t pipeline_depth = decision.pipeline_depth;
                // Large fixed plans otherwise keep the planner's multi-MiB
                // chunk, so the first CPU pack cannot overlap RDMA. On the
                // 26 MiB Engram scatter, 512 KiB and depth 2 was the best
                // measured point; smaller lookups keep the planner decision.
                constexpr size_t kLargeFixedChunkBytes = 512ULL << 10;
                if (fixed.total_bytes > kMultiLaneMaxBytes) {
                    chunk_bytes = kLargeFixedChunkBytes;
                    pipeline_depth = 2;
                }
                fixed.chunk_bytes =
                    std::max<size_t>(chunk_bytes, payload->fixed_span_length);
                fixed.pipeline_depth = pipeline_depth;
                fixed.compact_plan = true;
                fixed.fixed_span_length = payload->fixed_span_length;
                fixed.prepared_command = std::move(prepared_command);
                fixed.payload = std::move(payload);
                fixed.span_begin = 0;
                fixed.span_end = fixed.fragment_count;
                tasks.push_back(std::move(fixed));
                return true;
            }
            if (mixed_source_regions) return false;
            if (direct_request_count != 1) return false;

            DirectTask direct{
                .peer = fixed.peer,
                .destination = fixed.destination,
                .local_buffer = fixed.local_buffer,
                .local_capacity = fixed.local_capacity,
                .source_region_base = fixed.source_region_base,
                .source_region_size = fixed.source_region_size,
                .source = first_source,
                .length = static_cast<size_t>(fixed.total_bytes),
                .fragment_runs = std::move(fixed.fragment_runs),
            };
            appendDirectTask(std::move(direct));
            return true;
        };

        const bool fixed_plan_built = tryBuildFixedPlan();
        if (!fixed_plan_built) {
            bool has_gather_work = false;
            all_gather_work = true;
            for (const auto& range : ranges) {
                const size_t count = range.local_offsets.size();
                if (range.opcode != TransferRequest::READ ||
                    !range.local_buffer || range.remote_segment.empty() ||
                    range.remote_offsets.size() != count ||
                    range.lengths.size() != count) {
                    all_gather_work = false;
                    continue;
                }
                for (const size_t length : range.lengths) {
                    if (length != 0 && length < small_limit) {
                        has_gather_work = true;
                    } else {
                        all_gather_work = false;
                    }
                }
            }
            if (!has_gather_work) return 0;
            for (size_t range_index = 0; range_index < ranges.size();
                 ++range_index) {
                const auto& range = ranges[range_index];
                const size_t count = range.local_offsets.size();
                if (range.opcode != TransferRequest::READ ||
                    !range.local_buffer || range.remote_segment.empty() ||
                    range.remote_offsets.size() != count ||
                    range.lengths.size() != count) {
                    flush();
                    continue;
                }
                for (size_t fragment_index = 0; fragment_index < count;
                     ++fragment_index) {
                    const size_t length = range.lengths[fragment_index];
                    const size_t local_offset =
                        range.local_offsets[fragment_index];
                    const size_t remote_offset =
                        range.remote_offsets[fragment_index];
                    if (length == 0 || length >= small_limit ||
                        local_offset > range.local_capacity ||
                        length > range.local_capacity - local_offset ||
                        remote_offset > range.remote_size ||
                        length > range.remote_size - remote_offset ||
                        range.remote_base_offset > UINT64_MAX - remote_offset) {
                        flush();
                        continue;
                    }
                    const uint64_t destination = reinterpret_cast<uint64_t>(
                        static_cast<char*>(range.local_buffer) + local_offset);
                    const uint64_t source =
                        range.remote_base_offset + remote_offset;
                    if (destination > UINT64_MAX - length ||
                        source > UINT64_MAX - length) {
                        flush();
                        continue;
                    }
                    if (candidate.fragment_count != 0 &&
                        (candidate.peer != range.remote_segment ||
                         candidate.local_buffer != range.local_buffer ||
                         candidate.local_capacity != range.local_capacity ||
                         destination != expected_destination)) {
                        flush();
                    }
                    if (candidate.fragment_count == 0) {
                        startCandidate(range.remote_segment, destination,
                                       range.local_buffer, range.local_capacity,
                                       range.remote_base_offset,
                                       range.remote_size, range_index, source,
                                       length, count - fragment_index);
                        if (candidate.command_span_budget == 0) {
                            candidate = {};
                            continue;
                        }
                    }
                    if (candidate.fragment_count >= kMaxTaskFragments ||
                        candidate.total_bytes > (512ULL << 20) - length) {
                        flush();
                        startCandidate(range.remote_segment, destination,
                                       range.local_buffer, range.local_capacity,
                                       range.remote_base_offset,
                                       range.remote_size, range_index, source,
                                       length, count - fragment_index);
                    }

                    bool same_source_region =
                        candidate.source_region_base ==
                            range.remote_base_offset &&
                        candidate.source_region_size == range.remote_size;
                    if (candidate.fragment_count != 0 && !same_source_region) {
                        // Store ranged reads describe each object as a
                        // separate source window even when all objects live on
                        // the same peer. Keep one command across those object
                        // boundaries, but use absolute spans so the owner
                        // validates each registered region independently.
                        candidate.mixed_source_regions = true;
                        candidate.direct_contiguous = false;
                        useAbsoluteSpanEncoding();
                    }

                    if (candidate.fragment_count != 0 &&
                        source != expected_source)
                        candidate.direct_contiguous = false;

                    if (candidate.compact_plan) {
                        const uint64_t source_begin =
                            std::min(candidate.source_base, source);
                        const uint64_t source_end = std::max(
                            candidate.source_base + candidate.source_size,
                            source + length);
                        if (source_end - source_begin <= UINT32_MAX) {
                            candidate.source_base = source_begin;
                            candidate.source_size = source_end - source_begin;
                        } else {
                            useAbsoluteSpanEncoding();
                        }
                    }

                    bool coalesces =
                        same_source_region && !candidate.spans.empty() &&
                        candidate.spans.back().source_address <=
                            UINT64_MAX - candidate.spans.back().length &&
                        candidate.spans.back().source_address +
                                candidate.spans.back().length ==
                            source &&
                        candidate.spans.back().length <= (1ULL << 20) - length;
                    size_t encoded_span_bytes = candidate.encoded_span_bytes;
                    if (candidate.compact_plan) {
                        uint32_t fixed_span_length =
                            candidate.fixed_span_length;
                        if (coalesces) {
                            if (candidate.spans.size() == 1) {
                                fixed_span_length = static_cast<uint32_t>(
                                    candidate.spans.back().length + length);
                            } else {
                                fixed_span_length = 0;
                            }
                        } else if (candidate.spans.empty()) {
                            fixed_span_length = static_cast<uint32_t>(length);
                        } else if (fixed_span_length != length) {
                            fixed_span_length = 0;
                        }
                        encoded_span_bytes =
                            (candidate.spans.size() + (coalesces ? 0 : 1)) *
                            (fixed_span_length ? sizeof(uint32_t)
                                               : 2 * sizeof(uint32_t));
                    } else if (coalesces) {
                        encoded_span_bytes -=
                            varintBytes(candidate.spans.back().length);
                        encoded_span_bytes +=
                            varintBytes(candidate.spans.back().length + length);
                    } else {
                        encoded_span_bytes +=
                            varintBytes(source) + varintBytes(length);
                    }
                    if (encoded_span_bytes > candidate.command_span_budget &&
                        candidate.fragment_count != 0) {
                        flush();
                        startCandidate(range.remote_segment, destination,
                                       range.local_buffer, range.local_capacity,
                                       range.remote_base_offset,
                                       range.remote_size, range_index, source,
                                       length, count - fragment_index);
                        same_source_region = true;
                        coalesces = false;
                        encoded_span_bytes =
                            candidate.compact_plan
                                ? sizeof(uint32_t)
                                : varintBytes(source) + varintBytes(length);
                    }
                    if (coalesces) {
                        candidate.spans.back().length +=
                            static_cast<uint32_t>(length);
                        ++candidate.span_fragment_counts.back();
                    } else {
                        candidate.spans.push_back(
                            {source, static_cast<uint32_t>(length)});
                        candidate.span_fragment_counts.push_back(1);
                    }
                    candidate.max_span_length =
                        std::max<size_t>(candidate.max_span_length,
                                         candidate.spans.back().length);
                    if (candidate.compact_plan) {
                        if (candidate.spans.size() == 1) {
                            candidate.fixed_span_length =
                                candidate.spans.front().length;
                        } else if (coalesces ||
                                   candidate.fixed_span_length != length) {
                            candidate.fixed_span_length = 0;
                        }
                    }
                    candidate.encoded_span_bytes = encoded_span_bytes;
                    if (!candidate.fragment_runs.empty() &&
                        candidate.fragment_runs.back().range == range_index &&
                        candidate.fragment_runs.back().end == fragment_index) {
                        ++candidate.fragment_runs.back().end;
                    } else {
                        candidate.fragment_runs.push_back(
                            {range_index, fragment_index, fragment_index + 1});
                    }
                    ++candidate.fragment_count;
                    candidate.total_bytes += length;
                    candidate.range_index = range_index;
                    expected_destination = destination + length;
                    expected_source = source + length;
                    constexpr size_t kPrepareCommandMinFragments = 128;
                    if (prepared_peer.empty() &&
                        total_fragments >= kPrepareCommandMinFragments &&
                        candidate.fragment_count >= total_fragments / 2 &&
                        candidate.command_span_budget != 0) {
                        prepared_peer = candidate.peer;
                        prepared_command =
                            backend_.legacy->prepareScatterCommand(
                                candidate.peer);
                    }
                }
            }
            flush();
        }

        size_t gather_planned_fragments = 0;
        size_t planned_spans = 0;
        uint64_t planned_bytes = 0;
        for (const auto& task : tasks) {
            gather_planned_fragments += task.fragment_count;
            planned_spans += task.span_end - task.span_begin;
            planned_bytes += task.total_bytes;
        }
        size_t direct_planned_fragments = 0;
        uint64_t direct_planned_bytes = 0;
        for (const auto& task : direct_tasks) {
            direct_planned_bytes += task.length;
            for (const auto& run : task.fragment_runs)
                direct_planned_fragments += run.end - run.begin;
        }
        size_t planned_fragment_count =
            gather_planned_fragments + direct_planned_fragments;
        if (planned_fragment_count == 0) {
            if (trace_planning) {
                VLOG(1) << "scatter requester skipped plan_ms="
                        << std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - plan_started)
                               .count();
            }
            return 0;
        }

        if (trace_planning) {
            VLOG(1) << "scatter requester planning gather_fragments="
                    << gather_planned_fragments
                    << " gather_spans=" << planned_spans
                    << " gather_bytes=" << planned_bytes
                    << " gather_tasks=" << tasks.size()
                    << " direct_fragments=" << direct_planned_fragments
                    << " direct_requests=" << direct_tasks.size()
                    << " direct_bytes=" << direct_planned_bytes << " plan_ms="
                    << std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - plan_started)
                           .count();
        }

        if (planned_fragment_count != total_fragments) {
            planned_fragments.reserve(ranges.size());
            for (const auto& range : ranges)
                planned_fragments.emplace_back(range.lengths.size(), false);
        }
        for (const auto& task : direct_tasks) {
            if (planned_fragments.empty()) break;
            for (const auto& run : task.fragment_runs) {
                for (size_t fragment = run.begin; fragment < run.end;
                     ++fragment) {
                    planned_fragments[run.range][fragment] = true;
                }
            }
        }
        gather_fragment_runs_.reserve(tasks.size());
        for (const auto& task : tasks) {
            for (const auto& run : task.fragment_runs) {
                if (!planned_fragments.empty()) {
                    for (size_t fragment = run.begin; fragment < run.end;
                         ++fragment)
                        planned_fragments[run.range][fragment] = true;
                }
                if (!gather_fragment_runs_.empty() &&
                    gather_fragment_runs_.back().range == run.range &&
                    gather_fragment_runs_.back().end == run.begin) {
                    gather_fragment_runs_.back().end = run.end;
                } else {
                    gather_fragment_runs_.push_back(run);
                }
            }
        }

        if (tasks.empty()) return direct_planned_fragments;

        try {
            auto impl = backend_.legacy;
            const size_t max_parallel =
                std::max<size_t>(1, profile.pipeline_width);
            auto execute_tasks = [impl, max_parallel,
                                  tasks = std::move(tasks)]() mutable {
                const auto run_task = [impl](GatherTask task) {
                    GatherResult result;
                    result.fragment_runs = std::move(task.fragment_runs);
                    DCHECK(task.payload);
                    DCHECK_LE(task.span_begin, task.span_end);
                    const bool fixed_relative =
                        task.payload->fixed_span_length != 0 &&
                        (task.payload->fixed_offsets_in_destination ||
                         !task.payload->fixed_relative_offsets.empty());
                    const size_t payload_span_count =
                        fixed_relative
                            ? (task.payload->fixed_offsets_in_destination
                                   ? task.fragment_count
                                   : task.payload->fixed_relative_offsets
                                             .size() /
                                         sizeof(uint32_t))
                            : task.payload->spans.size();
                    DCHECK_LE(task.span_end, payload_span_count);
                    const size_t span_count = task.span_end - task.span_begin;
                    const auto* spans =
                        fixed_relative
                            ? nullptr
                            : task.payload->spans.data() + task.span_begin;
                    const std::string_view fixed_relative_offsets =
                        fixed_relative
                            ? (task.payload->fixed_offsets_in_destination
                                   ? std::string_view(
                                         reinterpret_cast<const char*>(
                                             task.destination),
                                         task.fragment_count * sizeof(uint32_t))
                                   : std::string_view(
                                         task.payload->fixed_relative_offsets))
                                  .substr(task.span_begin * sizeof(uint32_t),
                                          span_count * sizeof(uint32_t))
                            : std::string_view{};
                    std::unique_ptr<PreparedHandshakeCommand> prepared_command =
                        std::move(task.prepared_command);
                    Status status = impl->requestScatterGather(
                        task.peer, task.destination, spans, span_count,
                        fixed_relative_offsets, task.payload->fixed_span_length,
                        task.source_base, task.source_size, task.total_bytes,
                        task.chunk_bytes, task.pipeline_depth,
                        task.compact_plan, std::move(prepared_command));
                    if (!status.ok()) {
                        const auto segment = impl->openSegment(task.peer);
                        if (segment ==
                            static_cast<SegmentHandle>(ERR_INVALID_ARGUMENT)) {
                            status = Status::Endpoint(
                                "failed to open direct fallback segment");
                        } else {
                            status = Status::OK();
                            if (fixed_relative) {
                                result.span_fragment_counts.assign(span_count,
                                                                   1);
                            } else {
                                result.span_fragment_counts.assign(
                                    task.payload->span_fragment_counts.begin() +
                                        task.span_begin,
                                    task.payload->span_fragment_counts.begin() +
                                        task.span_end);
                            }
                            std::vector<TransferRequest> requests;
                            requests.reserve(span_count);
                            uint64_t destination = task.destination;
                            for (size_t i = 0; i < span_count; ++i) {
                                const uint64_t source =
                                    fixed_relative
                                        ? task.source_base +
                                              loadLittleEndian32(
                                                  fixed_relative_offsets
                                                      .data() +
                                                  i * sizeof(uint32_t))
                                        : spans[i].source_address;
                                const size_t length =
                                    fixed_relative
                                        ? task.payload->fixed_span_length
                                        : spans[i].length;
                                requests.push_back(TransferRequest{
                                    .opcode = TransferRequest::READ,
                                    .source =
                                        reinterpret_cast<void*>(destination),
                                    .target_id = segment,
                                    .target_offset = source,
                                    .length = length,
                                    // Keep fallback reads independent. A
                                    // mixed-success transport task must
                                    // not turn one failed read into a
                                    // failure for every original
                                    // fragment.
                                    .task_group_id =
                                        TransferRequest::kNoTaskGroup,
                                });
                                destination += length;
                            }
                            status = impl->transferDirect(
                                requests, &result.span_statuses);
                            if (impl->closeSegment(segment) != 0 && status.ok())
                                status = Status::Endpoint(
                                    "failed to close direct fallback "
                                    "segment");
                        }
                    }
                    result.status = std::move(status);
                    return result;
                };
                std::vector<GatherResult> results(tasks.size());
                for (size_t wave = 0; wave < tasks.size();
                     wave += max_parallel) {
                    const size_t wave_end =
                        std::min(tasks.size(), wave + max_parallel);
                    std::vector<std::future<GatherResult>> futures;
                    futures.reserve(wave_end - wave - 1);
                    for (size_t i = wave + 1; i < wave_end; ++i) {
                        futures.push_back(std::async(
                            std::launch::async, run_task, std::move(tasks[i])));
                    }
                    results[wave] = run_task(std::move(tasks[wave]));
                    for (size_t i = wave + 1; i < wave_end; ++i)
                        results[i] = futures[i - wave - 1].get();
                }
                return results;
            };
            const bool run_inline = synchronous_gather_ &&
                                    direct_planned_fragments == 0 &&
                                    planned_fragment_count == total_fragments;
            if (run_inline) {
                std::promise<std::vector<GatherResult>> promise;
                gather_future_ = promise.get_future();
                try {
                    promise.set_value(execute_tasks());
                } catch (...) {
                    promise.set_exception(std::current_exception());
                }
            } else {
                gather_future_ =
                    std::async(std::launch::async, std::move(execute_tasks));
            }
        } catch (...) {
            gather_future_ = {};
            if (planned_fragments.empty() &&
                direct_planned_fragments != total_fragments) {
                planned_fragments.reserve(ranges.size());
                for (const auto& range : ranges)
                    planned_fragments.emplace_back(range.lengths.size(), false);
                for (const auto& task : direct_tasks) {
                    for (const auto& run : task.fragment_runs) {
                        for (size_t fragment = run.begin; fragment < run.end;
                             ++fragment) {
                            planned_fragments[run.range][fragment] = true;
                        }
                    }
                }
            } else if (!planned_fragments.empty()) {
                for (const auto& run : gather_fragment_runs_)
                    for (size_t fragment = run.begin; fragment < run.end;
                         ++fragment)
                        planned_fragments[run.range][fragment] = false;
            }
            gather_fragment_runs_.clear();
            return direct_planned_fragments;
        }
        return planned_fragment_count;
    }

    bool pollGather() {
        if (!gather_future_.valid()) return false;
        if (gather_future_.wait_for(std::chrono::nanoseconds::zero()) !=
            std::future_status::ready)
            return true;
        const bool trace_callbacks = VLOG_IS_ON(1);
        const auto callback_started =
            trace_callbacks ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
        size_t completed_fragments = 0;
        const auto complete_fragments = [&](const auto& runs,
                                            const Status& status) {
            for (const auto& run : runs) {
                completeBatch(run.range, run.begin, run.end, status);
                completed_fragments += run.end - run.begin;
            }
        };
        try {
            auto results = gather_future_.get();
            for (const auto& result : results) {
                remember(result.status);
                if (result.span_statuses.empty()) {
                    complete_fragments(result.fragment_runs, result.status);
                    continue;
                }

                const auto mapping_status =
                    Status::Context("invalid scatter gather fallback mapping");
                if (result.span_statuses.size() !=
                    result.span_fragment_counts.size()) {
                    remember(mapping_status);
                    complete_fragments(result.fragment_runs, mapping_status);
                    continue;
                }

                size_t span_index = 0;
                size_t span_remaining = 0;
                const auto advance_span = [&] {
                    while (span_remaining == 0 &&
                           span_index < result.span_fragment_counts.size()) {
                        span_remaining =
                            result.span_fragment_counts[span_index];
                        if (span_remaining == 0) ++span_index;
                    }
                };
                for (const auto& run : result.fragment_runs) {
                    size_t fragment = run.begin;
                    while (fragment < run.end) {
                        advance_span();
                        if (span_index >= result.span_statuses.size()) {
                            remember(mapping_status);
                            completeBatch(run.range, fragment, run.end,
                                          mapping_status);
                            completed_fragments += run.end - fragment;
                            break;
                        }
                        const size_t count =
                            std::min(run.end - fragment, span_remaining);
                        completeBatch(run.range, fragment, fragment + count,
                                      result.span_statuses[span_index]);
                        completed_fragments += count;
                        fragment += count;
                        span_remaining -= count;
                    }
                }
                advance_span();
                if (span_index != result.span_statuses.size() ||
                    span_remaining != 0) {
                    remember(mapping_status);
                }
            }
        } catch (...) {
            const auto status = Status::Context("scatter gather worker failed");
            for (const auto& run : gather_fragment_runs_) {
                completeBatch(run.range, run.begin, run.end, status);
                completed_fragments += run.end - run.begin;
            }
        }
        if (trace_callbacks) {
            VLOG(1) << "scatter gather requester callbacks fragments="
                    << completed_fragments << " callback_ms="
                    << std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - callback_started)
                           .count();
        }
        gather_fragment_runs_.clear();
        return false;
    }

    void build(TransferEngine& engine,
               const std::vector<ScatterTransferRange>& ranges) {
        const bool trace_build = VLOG_IS_ON(1);
        const auto build_started =
            trace_build ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
        size_t total_fragments = 0;
        for (const auto& range : ranges)
            total_fragments += range.local_offsets.size();
        std::vector<std::vector<bool>> planned_fragments;
        std::vector<DirectTask> direct_tasks;
        const size_t optimized_fragments = buildScatterPlan(
            ranges, total_fragments, planned_fragments, direct_tasks);
        const size_t direct_request_upper_bound =
            direct_tasks.size() + (total_fragments - optimized_fragments);
        requests_.reserve(direct_request_upper_bound);
        request_fragment_runs_.reserve(direct_request_upper_bound);
        for (auto& task : direct_tasks) {
            auto [segment, inserted] = segment_handles_.emplace(
                task.peer, static_cast<SegmentHandle>(ERR_INVALID_ARGUMENT));
            if (inserted) segment->second = engine.openSegment(task.peer);
            if (segment->second ==
                static_cast<SegmentHandle>(ERR_INVALID_ARGUMENT)) {
                const auto status = Status::InvalidArgument(
                    "failed to open scatter transfer segment");
                for (const auto& run : task.fragment_runs)
                    completeBatch(run.range, run.begin, run.end, status);
                continue;
            }
            requests_.push_back(TransferRequest{
                .opcode = TransferRequest::READ,
                .source = reinterpret_cast<void*>(task.destination),
                .target_id = segment->second,
                .target_offset = task.source,
                .length = task.length,
                .task_group_id = 1,
            });
            request_fragment_runs_.push_back(std::move(task.fragment_runs));
        }
        for (size_t range_index = 0; optimized_fragments != total_fragments &&
                                     range_index < ranges.size();
             ++range_index) {
            const auto& range = ranges[range_index];
            const size_t fragment_count = range.local_offsets.size();
            SegmentHandle* segment_handle = nullptr;
            if (range.remote_offsets.size() != fragment_count ||
                range.lengths.size() != fragment_count ||
                range.local_buffer == nullptr || range.remote_segment.empty()) {
                const auto status =
                    Status::InvalidArgument("invalid scatter transfer range");
                remember(status);
                for (size_t i = 0; i < fragment_count; ++i)
                    complete(range_index, i, status);
                continue;
            }

            for (size_t fragment_index = 0; fragment_index < fragment_count;
                 ++fragment_index) {
                if (!planned_fragments.empty() &&
                    fragment_index < planned_fragments[range_index].size() &&
                    planned_fragments[range_index][fragment_index])
                    continue;
                const size_t length = range.lengths[fragment_index];
                const size_t local_offset = range.local_offsets[fragment_index];
                const size_t remote_offset =
                    range.remote_offsets[fragment_index];
                if (local_offset > range.local_capacity ||
                    length > range.local_capacity - local_offset ||
                    remote_offset > range.remote_size ||
                    length > range.remote_size - remote_offset ||
                    range.remote_base_offset >
                        std::numeric_limits<uint64_t>::max() - remote_offset ||
                    length > std::numeric_limits<uint64_t>::max() -
                                 (range.remote_base_offset + remote_offset)) {
                    complete(range_index, fragment_index,
                             Status::InvalidArgument(
                                 "invalid scatter transfer fragment"));
                    continue;
                }

                if (length == 0) {
                    complete(range_index, fragment_index, Status::OK());
                    continue;
                }

                if (!segment_handle) {
                    auto [segment, inserted] = segment_handles_.emplace(
                        range.remote_segment,
                        static_cast<SegmentHandle>(ERR_INVALID_ARGUMENT));
                    if (inserted)
                        segment->second =
                            engine.openSegment(range.remote_segment);
                    segment_handle = &segment->second;
                }
                if (*segment_handle ==
                    static_cast<SegmentHandle>(ERR_INVALID_ARGUMENT)) {
                    complete(range_index, fragment_index,
                             Status::InvalidArgument(
                                 "failed to open scatter transfer segment"));
                    continue;
                }

                auto* local_address =
                    static_cast<char*>(range.local_buffer) + local_offset;
                const uint64_t target_offset =
                    range.remote_base_offset + remote_offset;
                const auto local_address_value =
                    reinterpret_cast<uintptr_t>(local_address);
                const bool has_previous_run =
                    !request_fragment_runs_.empty() &&
                    !request_fragment_runs_.back().empty();
                const auto previous_range_index =
                    has_previous_run
                        ? request_fragment_runs_.back().back().range
                        : size_t{0};
                const auto& previous_range =
                    ranges[has_previous_run ? previous_range_index
                                            : range_index];
                const auto previous_local_address =
                    requests_.empty()
                        ? uintptr_t{0}
                        : reinterpret_cast<uintptr_t>(requests_.back().source);
                const bool coalesces =
                    !requests_.empty() && has_previous_run &&
                    requests_.back().opcode == range.opcode &&
                    requests_.back().target_id == *segment_handle &&
                    previous_range.local_buffer == range.local_buffer &&
                    previous_range.local_capacity == range.local_capacity &&
                    previous_range.remote_base_offset ==
                        range.remote_base_offset &&
                    previous_range.remote_size == range.remote_size &&
                    requests_.back().length <=
                        std::numeric_limits<size_t>::max() - length &&
                    previous_local_address <=
                        std::numeric_limits<uintptr_t>::max() -
                            requests_.back().length &&
                    previous_local_address + requests_.back().length ==
                        local_address_value &&
                    requests_.back().target_offset <=
                        std::numeric_limits<uint64_t>::max() -
                            requests_.back().length &&
                    requests_.back().target_offset + requests_.back().length ==
                        target_offset;
                if (coalesces) {
                    requests_.back().length += length;
                    auto& runs = request_fragment_runs_.back();
                    if (!runs.empty() && runs.back().range == range_index &&
                        runs.back().end == fragment_index) {
                        ++runs.back().end;
                    } else {
                        runs.push_back(
                            {range_index, fragment_index, fragment_index + 1});
                    }
                } else {
                    requests_.push_back(TransferRequest{
                        .opcode = range.opcode,
                        .source = local_address,
                        .target_id = *segment_handle,
                        .target_offset = target_offset,
                        .length = length,
                        .task_group_id = 1,
                    });
                    request_fragment_runs_.push_back(
                        {{range_index, fragment_index, fragment_index + 1}});
                }
            }
        }
        const auto requests_built =
            trace_build ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};

        if (requests_.empty()) {
            if (trace_build) {
                VLOG(1) << "scatter direct requester build_ms="
                        << std::chrono::duration<double, std::milli>(
                               requests_built - build_started)
                               .count()
                        << " requests=0";
            }
            if (!gather_future_.valid()) finish();
            return;
        }

        done_.assign(requests_.size(), false);
        Status submit_status = Status::OK();
        if (useTent()) {
            task_sizes_.assign(requests_.size(), 1);
            batch_id_ = engine.allocateBatchID(requests_.size());
            if (batch_id_ != INVALID_BATCH_ID)
                submit_status = engine.submitTransfer(batch_id_, requests_);
        } else {
            MultiTransport::ScatterSubmission submission;
            submit_status =
                backend_.legacy->submitScatter(requests_, submission);
            batch_id_ = submission.batch_id;
            task_sizes_ = std::move(submission.task_sizes);
        }
        if (trace_build) {
            VLOG(1) << "scatter direct requester requests=" << requests_.size()
                    << " build_ms="
                    << std::chrono::duration<double, std::milli>(
                           requests_built - build_started)
                           .count()
                    << " submit_ms="
                    << std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - requests_built)
                           .count();
        }
        remaining_ = task_sizes_.size();
        if (batch_id_ == INVALID_BATCH_ID) {
            failPending(submit_status.ok()
                            ? Status::InvalidArgument(
                                  "failed to allocate scatter transfer batch")
                            : submit_status);
            if (!gather_future_.valid()) finish();
            return;
        }

        if (submit_status.ok()) return;

#ifdef USE_TENT
        if (backend_.tent) {
            // TENT prepare failures publish no task slots, while committed
            // submissions publish all slots and report transport failures in
            // task status. On a live batch, task 0 being out of range thus
            // identifies the unpublished case.
            mooncake::tent::TransferStatus status;
            auto probe = backend_.tent->getTransferStatus(batch_id_, 0, status);
            if (probe.ok() || !probe.IsInvalidArgument()) {
                requestAbort(submit_status);
                return;
            }
            remember(submit_status);
            remember(freeBatch(batch_id_));
            batch_id_ = INVALID_BATCH_ID;
            failPending(submit_status);
            finish();
            return;
        }
#endif

        requestAbort(submit_status);
        // Published tasks own the exact fragment results, even when submit
        // itself reports an error. Poll them to physical completion.
        if (!task_sizes_.empty()) return;
        remember(freeBatch(batch_id_));
        batch_id_ = INVALID_BATCH_ID;
        failPending(submit_status);
        if (!gather_future_.valid()) finish();
    }

    void poll() {
        const bool gather_pending = pollGather();
        if (batch_id_ == INVALID_BATCH_ID) {
            if (!gather_pending) finish();
            return;
        }
        size_t request_index = 0;
        for (size_t task_id = 0; task_id < task_sizes_.size(); ++task_id) {
            const size_t request_start = request_index;
            request_index += task_sizes_[task_id];
            if (done_[request_start]) continue;
            TransferStatus status;
            auto result = getStatus(batch_id_, task_id, status);
            if (!result.ok()) {
                requestAbort(result);
                completeRequests(request_start, request_index, result);
                --remaining_;
                continue;
            }

            if (!useTent() && status.s == TransferStatusEnum::FAILED &&
                task_sizes_[task_id] > 1) {
                std::vector<TransferStatusEnum> request_statuses;
                auto detail_status = backend_.legacy->getScatterRequestStatuses(
                    batch_id_, task_id, request_statuses);
                if (detail_status.ok() &&
                    request_statuses.size() == task_sizes_[task_id]) {
                    for (size_t i = request_start; i < request_index; ++i) {
                        const auto fragment_status =
                            request_statuses[i - request_start] ==
                                    TransferStatusEnum::COMPLETED
                                ? Status::OK()
                                : Status::Socket(
                                      "scatter transfer fragment failed");
                        if (!fragment_status.ok())
                            requestAbort(fragment_status);
                        completeRequest(i, fragment_status);
                    }
                    --remaining_;
                    continue;
                }
                remember(detail_status.ok()
                             ? Status::Context(
                                   "invalid grouped scatter status count")
                             : detail_status);
            }

            Status fragment_status;
            if (status.s == TransferStatusEnum::COMPLETED) {
                fragment_status = Status::OK();
            } else if (status.s == TransferStatusEnum::WAITING ||
                       status.s == TransferStatusEnum::PENDING) {
                continue;
            } else if (status.s == TransferStatusEnum::TIMEOUT) {
                fragment_status =
                    Status::Socket("scatter transfer fragment timed out");
            } else {
                fragment_status =
                    Status::Socket("scatter transfer fragment failed");
            }
            if (!fragment_status.ok()) requestAbort(fragment_status);
            completeRequests(request_start, request_index, fragment_status);
            --remaining_;
        }
        assert(request_index == requests_.size());

        if (remaining_ != 0) return;
        auto free_status = freeBatch(batch_id_);
        if (free_status.IsBatchBusy()) return;
        remember(free_status);
        batch_id_ = INVALID_BATCH_ID;
        if (!gather_pending) finish();
    }

    Backend backend_;
    std::vector<TransferRequest> requests_;
    std::vector<std::vector<FragmentRun>> request_fragment_runs_;
    std::vector<std::function<void(size_t, const Status&)>> callbacks_;
    std::vector<std::function<void(size_t, size_t, const Status&)>>
        batch_callbacks_;
    std::unordered_map<std::string, SegmentHandle> segment_handles_;
    std::vector<uint8_t> done_;
    std::vector<size_t> task_sizes_;
    std::future<std::vector<GatherResult>> gather_future_;
    std::vector<FragmentRun> gather_fragment_runs_;
    BatchID batch_id_ = INVALID_BATCH_ID;
    size_t remaining_ = 0;
    Status aggregate_status_;
    bool abort_requested_ = false;
    bool completed_ = false;
    bool synchronous_gather_ = false;
};

TransferEngine::ScatterTransferOperation::ScatterTransferOperation(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TransferEngine::ScatterTransferOperation::ScatterTransferOperation(
    ScatterTransferOperation&&) noexcept = default;

TransferEngine::ScatterTransferOperation&
TransferEngine::ScatterTransferOperation::operator=(
    ScatterTransferOperation&&) noexcept = default;

TransferEngine::ScatterTransferOperation::~ScatterTransferOperation() = default;

Status TransferEngine::ScatterTransferOperation::wait() {
    return impl_
               ? impl_->wait()
               : Status::InvalidArgument("invalid scatter transfer operation");
}

Status TransferEngine::ScatterTransferOperation::waitFor(
    std::chrono::nanoseconds timeout) {
    return impl_
               ? impl_->waitFor(timeout)
               : Status::InvalidArgument("invalid scatter transfer operation");
}

void TransferEngine::setScatterStagingAllocator(
    ScatterStagingAllocator allocator) {
    if (!use_tent_ && impl_)
        impl_->setScatterStagingAllocator(std::move(allocator));
}

TransferEngine::ScatterTransferOperation TransferEngine::submitScatter(
    const std::vector<ScatterTransferRange>& ranges) {
    ScatterTransferOperation::Impl::Backend backend;
    backend.legacy = impl_;
#ifdef USE_TENT
    backend.tent = impl_tent_;
#endif

    return ScatterTransferOperation(
        std::make_unique<ScatterTransferOperation::Impl>(
            *this, std::move(backend), ranges, false));
}

Status TransferEngine::transferScatter(
    const std::vector<ScatterTransferRange>& ranges) {
    ScatterTransferOperation::Impl::Backend backend;
    backend.legacy = impl_;
#ifdef USE_TENT
    backend.tent = impl_tent_;
#endif
    ScatterTransferOperation operation(
        std::make_unique<ScatterTransferOperation::Impl>(
            *this, std::move(backend), ranges, true));
    return operation.wait();
}

}  // namespace mooncake

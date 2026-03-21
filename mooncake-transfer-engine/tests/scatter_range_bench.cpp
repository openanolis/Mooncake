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
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "transfer_engine.h"

using namespace mooncake;

namespace {

std::atomic<bool> g_stop{false};
constexpr SegmentHandle kInvalidSegment = static_cast<SegmentHandle>(-1);

void handle_signal(int) { g_stop.store(true); }

std::pair<std::string, uint16_t> parse_host_port(const std::string& value) {
    auto pos = value.rfind(':');
    if (pos == std::string::npos) {
        throw std::runtime_error("expected host:port, got: " + value);
    }
    return {value.substr(0, pos),
            static_cast<uint16_t>(std::stoul(value.substr(pos + 1)))};
}

std::vector<size_t> parse_size_list(const std::string& value) {
    std::vector<size_t> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(static_cast<size_t>(std::stoull(item)));
        }
    }
    return out;
}

std::string json_array(const std::vector<size_t>& values) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            os << ',';
        }
        os << values[i];
    }
    os << ']';
    return os.str();
}

std::string build_nic_priority_matrix(const std::string& device_name) {
    return std::string("{\"cpu:0\":[[\"") + device_name +
           "\"],[]],\"cpu:1\":[[\"" + device_name + "\"],[]],\"cuda:0\":[[\"" +
           device_name + "\"],[]]}";
}

void install_rdma_transport(TransferEngine& engine, const std::string& device) {
    std::string matrix = build_nic_priority_matrix(device);
    void* args[2];
    args[0] = const_cast<char*>(matrix.c_str());
    args[1] = nullptr;
    auto* transport = engine.installTransport("rdma", args);
    if (transport == nullptr) {
        throw std::runtime_error("failed to install rdma transport");
    }
}

struct CaseResult {
    size_t range_count;
    size_t range_size;
    size_t total_bytes;
    double mean_us;
    double p50_us;
    double p99_us;
};

double percentile_us(std::vector<double> values_us, double pct) {
    if (values_us.empty()) {
        return 0.0;
    }
    std::sort(values_us.begin(), values_us.end());
    double idx = (pct / 100.0) * static_cast<double>(values_us.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, values_us.size() - 1);
    double frac = idx - static_cast<double>(lo);
    return values_us[lo] * (1.0 - frac) + values_us[hi] * frac;
}

CaseResult run_case(TransferEngine& engine, SegmentHandle segment,
                    uint64_t remote_base, uint8_t* destination, size_t stride,
                    size_t range_count, size_t range_size, int warmup,
                    int iters) {
    const size_t total_bytes = range_count * range_size;
    std::vector<TransferRequest> entries;
    entries.reserve(range_count);
    for (size_t i = 0; i < range_count; ++i) {
        TransferRequest req;
        req.opcode = TransferRequest::READ;
        req.source = destination + i * range_size;
        req.target_id = segment;
        req.target_offset = remote_base + i * stride;
        req.length = range_size;
        req.task_group_id = 1;
        entries.push_back(req);
    }

    auto submit_once = [&](bool validate) {
        std::memset(destination, 0, total_bytes);
        BatchID batch = engine.allocateBatchID(1);
        if (batch == INVALID_BATCH_ID) {
            throw std::runtime_error("allocateBatchID failed");
        }
        auto begin = std::chrono::steady_clock::now();
        auto status = engine.submitTransfer(batch, entries);
        if (!status.ok()) {
            engine.freeBatchID(batch);
            throw std::runtime_error(std::string("submitTransfer failed: ") +
                                     status.ToString());
        }
        while (true) {
            TransferStatus transfer_status;
            auto rc = engine.getBatchTransferStatus(batch, transfer_status);
            if (!rc.ok()) {
                engine.freeBatchID(batch);
                throw std::runtime_error(
                    std::string("getBatchTransferStatus failed: ") +
                    rc.ToString());
            }
            if (transfer_status.s == TransferStatusEnum::COMPLETED) {
                break;
            }
            if (transfer_status.s == TransferStatusEnum::FAILED ||
                transfer_status.s == TransferStatusEnum::INVALID ||
                transfer_status.s == TransferStatusEnum::TIMEOUT) {
                engine.freeBatchID(batch);
                throw std::runtime_error("scatter batch transfer failed");
            }
        }
        auto end = std::chrono::steady_clock::now();
        auto free_rc = engine.freeBatchID(batch);
        if (!free_rc.ok()) {
            throw std::runtime_error(std::string("freeBatchID failed: ") +
                                     free_rc.ToString());
        }

        if (validate) {
            for (size_t i = 0; i < range_count; ++i) {
                for (size_t j = 0; j < range_size; ++j) {
                    uint8_t expected =
                        static_cast<uint8_t>(((i * stride) + j) & 0xFF);
                    if (destination[i * range_size + j] != expected) {
                        std::ostringstream os;
                        os << "data mismatch at range=" << i << " offset=" << j
                           << " got="
                           << static_cast<int>(destination[i * range_size + j])
                           << " expected=" << static_cast<int>(expected);
                        throw std::runtime_error(os.str());
                    }
                }
            }
        }

        return std::chrono::duration<double, std::micro>(end - begin).count();
    };

    submit_once(true);
    for (int i = 0; i < warmup; ++i) {
        submit_once(false);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        latencies_us.push_back(submit_once(false));
    }

    double mean =
        std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) /
        static_cast<double>(latencies_us.size());
    return CaseResult{range_count,
                      range_size,
                      total_bytes,
                      mean,
                      percentile_us(latencies_us, 50.0),
                      percentile_us(latencies_us, 99.0)};
}

std::string get_arg(int argc, char** argv, const std::string& key,
                    const std::string& default_value = "") {
    const std::string prefix = "--" + key + "=";
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind(prefix, 0) == 0) {
            return arg.substr(prefix.size());
        }
    }
    return default_value;
}

int get_arg_int(int argc, char** argv, const std::string& key, int def) {
    std::string value = get_arg(argc, argv, key);
    return value.empty() ? def : std::stoi(value);
}

size_t get_arg_size(int argc, char** argv, const std::string& key, size_t def) {
    std::string value = get_arg(argc, argv, key);
    return value.empty() ? def : static_cast<size_t>(std::stoull(value));
}

}  // namespace

int main(int argc, char** argv) {
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    const std::string mode = get_arg(argc, argv, "mode");
    const std::string metadata = get_arg(argc, argv, "metadata");
    const std::string local = get_arg(argc, argv, "local");
    const std::string device = get_arg(argc, argv, "device", "erdma_0");

    if (mode.empty() || metadata.empty() || local.empty()) {
        std::cerr << "required: --mode=producer|consumer --metadata=... "
                  << "--local=host:port\n";
        return 2;
    }

    auto [local_host, local_port] = parse_host_port(local);

    TransferEngine engine(false);
    if (engine.init(metadata, local, local_host, local_port) != 0) {
        std::cerr << "engine.init failed\n";
        return 1;
    }
    install_rdma_transport(engine, device);

    const size_t max_range_size =
        get_arg_size(argc, argv, "max-range-size", 16384);
    const size_t max_ranges = get_arg_size(argc, argv, "max-ranges", 256);
    const size_t stride = max_range_size;
    const size_t buffer_size = max_ranges * stride;

    void* buffer = nullptr;
    if (posix_memalign(&buffer, 4096, buffer_size) != 0) {
        std::cerr << "posix_memalign failed\n";
        return 1;
    }
    auto* bytes = static_cast<uint8_t*>(buffer);
    for (size_t i = 0; i < buffer_size; ++i) {
        bytes[i] = static_cast<uint8_t>(i & 0xFF);
    }

    if (engine.registerLocalMemory(buffer, buffer_size, "cpu:0", true, true) !=
        0) {
        std::cerr << "registerLocalMemory failed\n";
        return 1;
    }

    if (mode == "producer") {
        std::cout << "{\n"
                  << "  \"status\": \"ready\",\n"
                  << "  \"local_server_name\": \"" << local << "\",\n"
                  << "  \"buffer_addr\": " << reinterpret_cast<uint64_t>(buffer)
                  << ",\n"
                  << "  \"buffer_size\": " << buffer_size << ",\n"
                  << "  \"stride\": " << stride << "\n"
                  << "}" << std::endl;
        while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } else if (mode == "consumer") {
        const std::string peer = get_arg(argc, argv, "peer");
        const std::string remote_base_arg = get_arg(argc, argv, "remote-base");
        const int warmup = get_arg_int(argc, argv, "warmup", 20);
        const int iters = get_arg_int(argc, argv, "iters", 200);
        const auto range_counts = parse_size_list(
            get_arg(argc, argv, "range-counts", "1,2,4,8,16,32,64,128,256"));
        const auto range_sizes = parse_size_list(
            get_arg(argc, argv, "range-sizes", "64,256,1024,4096,16384"));

        if (peer.empty() || remote_base_arg.empty()) {
            std::cerr << "consumer requires --peer and --remote-base\n";
            return 2;
        }
        const uint64_t remote_base =
            static_cast<uint64_t>(std::stoull(remote_base_arg));

        SegmentHandle segment = kInvalidSegment;
        for (int attempt = 0; attempt < 100; ++attempt) {
            segment = engine.openSegment(peer);
            if (segment != kInvalidSegment) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (segment == kInvalidSegment) {
            std::cerr << "openSegment failed for peer " << peer << "\n";
            return 1;
        }

        std::vector<CaseResult> results;
        for (size_t range_size : range_sizes) {
            for (size_t range_count : range_counts) {
                results.push_back(run_case(engine, segment, remote_base, bytes,
                                           stride, range_count, range_size,
                                           warmup, iters));
            }
        }

        std::cout << "{\n"
                  << "  \"protocol\": \"rdma\",\n"
                  << "  \"local_server_name\": \"" << local << "\",\n"
                  << "  \"peer_server_name\": \"" << peer << "\",\n"
                  << "  \"warmup\": " << warmup << ",\n"
                  << "  \"iters\": " << iters << ",\n"
                  << "  \"range_counts\": " << json_array(range_counts) << ",\n"
                  << "  \"range_sizes\": " << json_array(range_sizes) << ",\n"
                  << "  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            std::cout << "    {\"range_count\": " << result.range_count
                      << ", \"range_size\": " << result.range_size
                      << ", \"total_bytes\": " << result.total_bytes
                      << ", \"latency\": {\"mean_us\": " << std::fixed
                      << std::setprecision(3) << result.mean_us
                      << ", \"p50_us\": " << result.p50_us
                      << ", \"p99_us\": " << result.p99_us << "}}";
            if (i + 1 != results.size()) {
                std::cout << ',';
            }
            std::cout << '\n';
        }
        std::cout << "  ]\n}" << std::endl;
    } else {
        std::cerr << "unknown mode: " << mode << "\n";
        return 2;
    }

    engine.unregisterLocalMemory(buffer, true);
    engine.freeEngine();
    free(buffer);
    return 0;
}

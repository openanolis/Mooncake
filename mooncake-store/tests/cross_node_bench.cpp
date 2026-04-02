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

// Cross-node benchmark: measures regular Get, ProgressiveGet wait_all,
// sequential wait_chunk, and first-chunk latency across two nodes.
//
// Usage (requires a running mooncake_master on Node 0):
//
//   # Node 0 (provider):
//   # Node 0 (provider):
//   ./cross_node_bench --role=provider --local_hostname=192.168.22.70
//       --metadata_server=http://192.168.22.70:8080/metadata
//       --master_server=192.168.22.70:50051 --protocol=rdma
//
//   # Node 1 (consumer):
//   ./cross_node_bench --role=consumer --local_hostname=192.168.22.72
//       --metadata_server=http://192.168.22.70:8080/metadata
//       --master_server=192.168.22.70:50051 --protocol=rdma

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "client_service.h"
#include "types.h"

DEFINE_string(role, "", "provider or consumer");
DEFINE_string(protocol, "rdma", "Transfer protocol: rdma|tcp");
DEFINE_string(device_name, "", "RDMA device name (empty=auto)");
DEFINE_string(local_hostname, "", "Local host IP");
DEFINE_string(metadata_server, "http://192.168.22.70:8080/metadata",
              "Metadata server URL");
DEFINE_string(master_server, "192.168.22.70:50051", "Master server address");
DEFINE_uint64(segment_size_mb, 4096, "Provider segment size in MB");
DEFINE_uint64(buffer_size_mb, 2048, "Local buffer size in MB");

namespace mooncake {

// Fill buffer with a reproducible byte pattern.
static void FillPattern(void* buf, size_t size, const std::string& seed) {
    auto* p = static_cast<uint8_t*>(buf);
    uint32_t h = 0;
    for (char c : seed) h = h * 31 + static_cast<uint8_t>(c);
    for (size_t i = 0; i < size; ++i) {
        h ^= (h << 13);
        h ^= (h >> 17);
        h ^= (h << 5);
        p[i] = static_cast<uint8_t>(h);
    }
}

static bool VerifyPattern(const void* buf, size_t size,
                          const std::string& seed) {
    std::vector<uint8_t> expected(size);
    FillPattern(expected.data(), size, seed);
    return std::memcmp(buf, expected.data(), size) == 0;
}

static std::string FmtBytes(size_t n) {
    if (n >= 1024ULL * 1024 * 1024)
        return std::to_string(n / (1024ULL * 1024 * 1024)) + "GB";
    if (n >= 1024ULL * 1024) return std::to_string(n / (1024ULL * 1024)) + "MB";
    return std::to_string(n / 1024) + "KB";
}

static double Median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return 0.0;
    return n % 2 == 0 ? (v[n / 2 - 1] + v[n / 2]) / 2.0 : v[n / 2];
}

// ==========================================================================
// Object tiers
// ==========================================================================

struct ObjTier {
    const char* label;
    size_t obj_size;
    std::vector<size_t> chunk_sizes;
    int warmup;
    int iters;
};

static const std::vector<ObjTier>& GetTiers() {
    static const std::vector<ObjTier> tiers = {
        {"SMALL(4MB)",
         4ULL * 1024 * 1024,
         {256 * 1024, 1024 * 1024, 4 * 1024 * 1024},
         2,
         10},
        {"MEDIUM(64MB)",
         64ULL * 1024 * 1024,
         {1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024},
         2,
         5},
        {"LARGE(1GB)",
         1ULL * 1024 * 1024 * 1024,
         {4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024},
         1,
         3},
    };
    return tiers;
}

// ==========================================================================
// Provider: Put objects and wait for consumer
// ==========================================================================

static int RunProvider() {
    const size_t seg_size = FLAGS_segment_size_mb * 1024ULL * 1024;

    std::optional<std::string> dev = FLAGS_device_name.empty()
                                         ? std::nullopt
                                         : std::optional(FLAGS_device_name);
    auto client_opt =
        Client::Create(FLAGS_local_hostname, FLAGS_metadata_server,
                       FLAGS_protocol, dev, FLAGS_master_server);
    if (!client_opt.has_value()) {
        LOG(ERROR) << "Failed to create provider client";
        return 1;
    }
    auto client = client_opt.value();

    // Mount segment for hosting objects
    void* seg_ptr = std::malloc(seg_size);
    CHECK(seg_ptr) << "malloc segment failed";
    auto mount = client->MountSegment(seg_ptr, seg_size, FLAGS_protocol);
    CHECK(mount.has_value())
        << "MountSegment failed: " << toString(mount.error());

    // Register a large working buffer
    const auto& tiers = GetTiers();
    size_t max_obj = std::max_element(tiers.begin(), tiers.end(),
                                      [](const ObjTier& a, const ObjTier& b) {
                                          return a.obj_size < b.obj_size;
                                      })
                         ->obj_size;

    void* work_buf = std::malloc(max_obj);
    CHECK(work_buf) << "malloc work_buf failed";
    auto reg =
        client->RegisterLocalMemory(work_buf, max_obj, "cpu:0", false, false);
    CHECK(reg.has_value()) << "RegisterLocalMemory failed: "
                           << toString(reg.error());

    LOG(INFO) << "Provider ready, putting objects ...";

    for (const auto& tier : tiers) {
        const std::string key = "bench_" + std::to_string(tier.obj_size);
        FillPattern(work_buf, tier.obj_size, key);
        std::vector<Slice> slices{{work_buf, tier.obj_size}};
        ReplicateConfig cfg;
        cfg.replica_num = 1;
        auto r = client->Put(key, slices, cfg);
        CHECK(r.has_value())
            << "Put " << key << " failed: " << toString(r.error());
        LOG(INFO) << "  put " << key << " (" << FmtBytes(tier.obj_size)
                  << ") ok";
    }

    // Write sentinel
    const std::string sentinel_key = "__bench_ready__";
    uint8_t sentinel_byte = 0x42;
    std::memset(work_buf, sentinel_byte, 8);
    {
        std::vector<Slice> slices{{work_buf, 8}};
        ReplicateConfig cfg;
        cfg.replica_num = 1;
        client->Put(sentinel_key, slices, cfg);
    }

    LOG(INFO) << "All objects stored. Waiting for consumer ...";

    // Poll for done sentinel
    for (int i = 0; i < 600; ++i) {
        auto q = client->Query("__bench_done__");
        if (q.has_value()) {
            LOG(INFO) << "Consumer signalled done.";
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Cleanup
    for (const auto& tier : tiers) {
        client->Remove("bench_" + std::to_string(tier.obj_size));
    }
    client->Remove("__bench_ready__");
    client->Remove("__bench_done__");

    std::free(work_buf);
    client->UnmountSegment(seg_ptr, seg_size);
    std::free(seg_ptr);
    LOG(INFO) << "Provider done.";
    return 0;
}

// ==========================================================================
// Consumer: Benchmark Get + ProgressiveGet
// ==========================================================================

static int RunConsumer() {
    std::optional<std::string> dev = FLAGS_device_name.empty()
                                         ? std::nullopt
                                         : std::optional(FLAGS_device_name);
    auto client_opt =
        Client::Create(FLAGS_local_hostname, FLAGS_metadata_server,
                       FLAGS_protocol, dev, FLAGS_master_server);
    if (!client_opt.has_value()) {
        LOG(ERROR) << "Failed to create consumer client";
        return 1;
    }
    auto client = client_opt.value();

    const auto& tiers = GetTiers();
    size_t max_obj = std::max_element(tiers.begin(), tiers.end(),
                                      [](const ObjTier& a, const ObjTier& b) {
                                          return a.obj_size < b.obj_size;
                                      })
                         ->obj_size;

    void* buf = std::malloc(max_obj);
    CHECK(buf) << "malloc buf failed";
    auto reg = client->RegisterLocalMemory(buf, max_obj, "cpu:0", false, false);
    CHECK(reg.has_value()) << "RegisterLocalMemory failed: "
                           << toString(reg.error());

    // Wait for provider
    LOG(INFO) << "Consumer waiting for provider ...";
    for (int i = 0; i < 120; ++i) {
        auto q = client->Query("__bench_ready__");
        if (q.has_value()) break;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (i == 119) {
            LOG(ERROR) << "Timed out waiting for provider";
            return 1;
        }
    }
    LOG(INFO) << "Provider ready. Starting benchmark.";

    // Print header
    std::cout << "\n"
              << std::string(100, '=') << "\n"
              << "  Cross-Node Benchmark  |  protocol=" << FLAGS_protocol
              << "\n"
              << std::string(100, '=') << "\n";

    for (const auto& tier : tiers) {
        const std::string key = "bench_" + std::to_string(tier.obj_size);

        std::cout << "\n--- " << tier.label
                  << " (obj_size=" << FmtBytes(tier.obj_size) << ") ---\n";
        std::cout << std::setw(10) << "chunk" << std::setw(12) << "baseline"
                  << std::setw(12) << "wait_all" << std::setw(8) << "ratio"
                  << std::setw(12) << "seq_chunk" << std::setw(8) << "ratio"
                  << std::setw(12) << "1st_chunk" << std::setw(10) << "bw_MB/s"
                  << "\n";

        // --- Baseline: Regular Get ---
        std::vector<double> get_us;
        for (int iter = 0; iter < tier.warmup + tier.iters; ++iter) {
            std::vector<Slice> slices{{buf, tier.obj_size}};
            auto t0 = std::chrono::steady_clock::now();
            auto r = client->Get(key, slices);
            auto t1 = std::chrono::steady_clock::now();
            if (!r.has_value()) {
                LOG(ERROR) << "Get " << key
                           << " failed: " << toString(r.error());
                continue;
            }
            if (iter >= tier.warmup) {
                get_us.push_back(
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
        }
        double get_med = Median(get_us);
        double bw_baseline =
            get_med > 0 ? tier.obj_size / (get_med / 1e6) / (1024.0 * 1024.0)
                        : 0;

        std::cout << std::setw(10) << "baseline" << std::setw(12) << std::fixed
                  << std::setprecision(0) << get_med << std::setw(12) << "-"
                  << std::setw(8) << "-" << std::setw(12) << "-" << std::setw(8)
                  << "-" << std::setw(12) << "-" << std::setw(10)
                  << std::setprecision(1) << bw_baseline << "\n";

        // --- ProgressiveGet for each chunk size ---
        for (size_t chunk_size : tier.chunk_sizes) {
            std::vector<double> wa_us, sc_us, fc_us;

            for (int iter = 0; iter < tier.warmup + tier.iters; ++iter) {
                bool measure = (iter >= tier.warmup);

                // wait_all
                {
                    auto t0 = std::chrono::steady_clock::now();
                    auto h = client->ProgressiveGet(key, buf, tier.obj_size,
                                                    chunk_size);
                    CHECK(h.has_value()) << "ProgressiveGet failed";
                    auto ec = h->wait_all();
                    auto t1 = std::chrono::steady_clock::now();
                    CHECK(ec == ErrorCode::OK)
                        << "wait_all failed: " << static_cast<int>(ec);
                    if (measure)
                        wa_us.push_back(
                            std::chrono::duration<double, std::micro>(t1 - t0)
                                .count());
                }
                // sequential wait_chunk
                {
                    auto t0 = std::chrono::steady_clock::now();
                    auto h = client->ProgressiveGet(key, buf, tier.obj_size,
                                                    chunk_size);
                    CHECK(h.has_value()) << "ProgressiveGet failed";
                    for (size_t c = 0; c < h->num_chunks(); ++c) {
                        auto ec = h->wait_chunk(c);
                        CHECK(ec == ErrorCode::OK)
                            << "wait_chunk " << c
                            << " failed: " << static_cast<int>(ec);
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    if (measure)
                        sc_us.push_back(
                            std::chrono::duration<double, std::micro>(t1 - t0)
                                .count());
                }
                // first-chunk latency
                {
                    auto t0 = std::chrono::steady_clock::now();
                    auto h = client->ProgressiveGet(key, buf, tier.obj_size,
                                                    chunk_size);
                    CHECK(h.has_value()) << "ProgressiveGet failed";
                    auto ec = h->wait_chunk(0);
                    auto t1 = std::chrono::steady_clock::now();
                    CHECK(ec == ErrorCode::OK)
                        << "wait_chunk(0) failed: " << static_cast<int>(ec);
                    h->wait_all();  // drain remaining
                    if (measure)
                        fc_us.push_back(
                            std::chrono::duration<double, std::micro>(t1 - t0)
                                .count());
                }
            }

            double wa_med = Median(wa_us);
            double sc_med = Median(sc_us);
            double fc_med = Median(fc_us);

            double wa_ratio = get_med > 0 ? wa_med / get_med : 0;
            double sc_ratio = get_med > 0 ? sc_med / get_med : 0;
            double bw_wa =
                wa_med > 0 ? tier.obj_size / (wa_med / 1e6) / (1024.0 * 1024.0)
                           : 0;

            std::cout << std::setw(10) << FmtBytes(chunk_size) << std::setw(12)
                      << std::fixed << std::setprecision(0) << get_med
                      << std::setw(12) << wa_med << std::setw(8)
                      << std::setprecision(2) << wa_ratio << std::setw(12)
                      << std::setprecision(0) << sc_med << std::setw(8)
                      << std::setprecision(2) << sc_ratio << std::setw(12)
                      << std::setprecision(0) << fc_med << std::setw(10)
                      << std::setprecision(1) << bw_wa << "\n";
        }

        // Verify data correctness (last ProgressiveGet result)
        if (VerifyPattern(buf, tier.obj_size, key)) {
            std::cout << "  [data verified OK]\n";
        } else {
            std::cout << "  [DATA VERIFICATION FAILED]\n";
        }
    }

    std::cout << "\n" << std::string(100, '=') << "\n\n";

    // Signal provider we are done
    uint8_t done_byte = 0x99;
    std::memset(buf, done_byte, 8);
    {
        std::vector<Slice> slices{{buf, 8}};
        ReplicateConfig cfg;
        cfg.replica_num = 1;
        client->Put("__bench_done__", slices, cfg);
    }

    std::free(buf);
    LOG(INFO) << "Consumer done.";
    return 0;
}

}  // namespace mooncake

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_role == "provider") {
        return mooncake::RunProvider();
    } else if (FLAGS_role == "consumer") {
        return mooncake::RunConsumer();
    } else {
        LOG(ERROR) << "Must specify --role=provider or --role=consumer";
        return 1;
    }
}

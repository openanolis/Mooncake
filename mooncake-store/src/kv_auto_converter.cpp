// ============================================================================
// KV Auto Converter - C2C cross-model projection
// ============================================================================
// Full C2CProjector architecture (thu-nics/C2C):
//   h = Linear(src_kv)                           # input proj, NO activation
//   h = h + w2(GELU(w1(RMSNorm(h))))             # mlp1 blocks
//   proj = proj_mlp2(h)                           # N blocks RMSNorm+GELU+residual
//   out = Linear(proj)                            # proj_out -> [tgt_dim]
//   scalar = sigmoid(scalar_head(scalar_mlp2(h))) # per-head weights
//   gate = (gate_logit > 0) ? 1.0 : 0.0          # binary gate
//   output = gate * scalar * out                  # per-head scaled projection
// Optimized with OpenBLAS + AVX2 SIMD
// ============================================================================
#include "kv_auto_converter.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <json/reader.h>
#include <json/value.h>

#ifdef USE_CURL
#include <curl/curl.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

// SIMD support detection
#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#endif

namespace mooncake {

// v2 binary format magic number
static constexpr uint32_t kMagicV2 = 0xC2C20002;
static constexpr float kRmsNormEps = 1e-6f;

// ============================================================================
// Matrix transpose: [rows, cols] -> [cols, rows]
// Done once at load time so runtime matmul uses row-contiguous access
// ============================================================================
static void transpose_matrix(std::vector<float>& mat, int rows, int cols) {
    std::vector<float> tmp(mat.size());
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            tmp[c * rows + r] = mat[r * cols + c];
        }
    }
    mat = std::move(tmp);
}

KVAutoConverter& KVAutoConverter::instance() {
    static KVAutoConverter inst;
    return inst;
}

// ============================================================================
// Cache directory + URL hashing utilities
// ============================================================================
#ifdef USE_CURL
std::string KVAutoConverter::get_cache_dir() {
    std::string home;
    const char* h = std::getenv("HOME");
    if (h)
        home = h;
    else
        home = "/tmp";
    std::string dir = home + "/.cache/mooncake/c2c";
    mkdir((home + "/.cache").c_str(), 0755);
    mkdir((home + "/.cache/mooncake").c_str(), 0755);
    if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        LOG(WARNING) << "[C2C] Failed to create cache dir: " << dir << " ("
                     << strerror(errno) << ")";
    }
    return dir;
}

// Simple hash: djb2 -> hex prefix (no OpenSSL dependency)
std::string KVAutoConverter::url_to_cache_path(const std::string& url) {
    uint64_t hash = 5381;
    for (char c : url) hash = ((hash << 5) + hash) + static_cast<uint8_t>(c);
    std::string fname = "projector.bin";
    size_t slash = url.rfind('/');
    if (slash != std::string::npos && slash + 1 < url.size()) {
        fname = url.substr(slash + 1);
        size_t q = fname.find('?');
        if (q != std::string::npos) fname = fname.substr(0, q);
    }
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return get_cache_dir() + "/" + oss.str() + "_" + fname;
}

// libcurl write-to-string callback
static size_t curl_string_cb(void* ptr, size_t size, size_t nmemb,
                             void* userdata) {
    auto* s = static_cast<std::string*>(userdata);
    s->append(static_cast<const char*>(ptr), size * nmemb);
    return size * nmemb;
}

// ============================================================================
// Multi-threaded parallel download with progress bar
// ============================================================================
struct MultiDownloadProgress {
    std::atomic<int64_t> downloaded{0};
    int64_t total{0};
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point last_update;
    int64_t last_downloaded{0};
    double current_speed{0.0};
    std::mutex mtx;
};

static void print_progress_bar(MultiDownloadProgress& prog,
                               bool final = false) {
    auto now = std::chrono::steady_clock::now();
    int64_t dl = prog.downloaded.load(std::memory_order_relaxed);
    int64_t total = prog.total;

    double dt = std::chrono::duration<double>(now - prog.last_update).count();
    if (dt > 0.1) {
        double instant = (dl - prog.last_downloaded) / 1048576.0 / dt;
        prog.current_speed = prog.current_speed * 0.7 + instant * 0.3;
        prog.last_downloaded = dl;
        prog.last_update = now;
    }

    if (total > 0) {
        int pct = static_cast<int>(dl * 100 / total);
        int filled = pct / 2;
        fprintf(stderr, "\r[C2C] [");
        for (int i = 0; i < 50; ++i) fputc(i < filled ? '#' : '.', stderr);
        fprintf(stderr, "] %3d%% %lld/%lld MB  %.1f MB/s", pct,
                (long long)(dl >> 20), (long long)(total >> 20),
                prog.current_speed);
    } else {
        static const char spin[] = "|/-\\";
        static int idx = 0;
        fprintf(stderr, "\r[C2C] %c  %lld MB  %.1f MB/s", spin[(idx++) & 3],
                (long long)(dl >> 20), prog.current_speed);
    }

    if (final) fputc('\n', stderr);
    fflush(stderr);
}

struct ChunkWriter {
    int fd;
    int64_t offset;
    MultiDownloadProgress* prog;
};

static size_t chunk_write_cb(void* ptr, size_t size, size_t nmemb,
                             void* userdata) {
    auto* cw = static_cast<ChunkWriter*>(userdata);
    size_t bytes = size * nmemb;
    ssize_t written = pwrite(cw->fd, ptr, bytes, cw->offset);
    if (written <= 0) return 0;
    cw->offset += written;
    cw->prog->downloaded.fetch_add(written, std::memory_order_relaxed);
    return static_cast<size_t>(written);
}

struct SingleWriter {
    std::ofstream* out;
    MultiDownloadProgress* prog;
};

static size_t single_write_cb(void* ptr, size_t size, size_t nmemb,
                              void* userdata) {
    auto* sw = static_cast<SingleWriter*>(userdata);
    size_t bytes = size * nmemb;
    sw->out->write(static_cast<const char*>(ptr), bytes);
    if (!sw->out->good()) return 0;
    sw->prog->downloaded.fetch_add(bytes, std::memory_order_relaxed);
    return bytes;
}

static int multi_progress_cb(void* clientp, curl_off_t, curl_off_t,
                             curl_off_t, curl_off_t) {
    auto* prog = static_cast<MultiDownloadProgress*>(clientp);
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(prog->mtx);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now - prog->last_update)
                  .count();
    if (ms >= 500) {
        print_progress_bar(*prog);
        prog->last_update = now;
    }
    return 0;
}

static size_t range_header_cb(char* buf, size_t sz, size_t n, void* ud) {
    size_t len = sz * n;
    auto* total = static_cast<int64_t*>(ud);
    if (len > 22 && strncasecmp(buf, "Content-Range:", 14) == 0) {
        const char* slash = static_cast<const char*>(memchr(buf, '/', len));
        if (slash) *total = strtoll(slash + 1, nullptr, 10);
    }
    return len;
}

static bool probe_file_size(const std::string& url, int64_t& file_size) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string dummy;
    file_size = -1;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "mooncake-c2c/1.0");
    curl_easy_setopt(curl, CURLOPT_RANGE, "0-0");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_string_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &dummy);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, range_header_cb);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &file_size);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (res == CURLE_OK && http_code == 206 && file_size > 0) {
        LOG(INFO) << "[C2C] Probed file size: " << (file_size >> 20) << " MB";
        return true;
    }
    return false;
}
#else
std::string KVAutoConverter::get_cache_dir() { return "/tmp"; }
std::string KVAutoConverter::url_to_cache_path(const std::string&) {
    return "";
}
#endif

// ============================================================================
// Download file from URL (multi-threaded parallel, skip if cached)
// ============================================================================
static constexpr int kDownloadThreads = 4;
static constexpr int64_t kMinChunkSize = 4 * 1024 * 1024;

bool KVAutoConverter::download_file(const std::string& url,
                                    const std::string& local_path) {
#ifdef USE_CURL
    struct stat st;
    if (stat(local_path.c_str(), &st) == 0 && st.st_size > 0) {
        LOG(INFO) << "[C2C] Cache hit: " << local_path;
        return true;
    }

    LOG(INFO) << "[C2C] Downloading: " << url;
    std::string tmp_path = local_path + ".tmp";

    int64_t file_size = 0;
    bool can_parallel = probe_file_size(url, file_size);

    int num_threads = 1;
    if (can_parallel && file_size >= kMinChunkSize * 2) {
        num_threads = kDownloadThreads;
    }

    MultiDownloadProgress prog;
    prog.total = file_size > 0 ? file_size : 0;
    prog.start = std::chrono::steady_clock::now();
    prog.last_update = prog.start;

    if (num_threads <= 1) {
        std::ofstream out(tmp_path, std::ios::binary);
        if (!out.is_open()) {
            LOG(ERROR) << "[C2C] Cannot create temp file: " << tmp_path;
            return false;
        }

        CURL* curl = curl_easy_init();
        if (!curl) return false;

        SingleWriter sw{&out, &prog};
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "mooncake-c2c/1.0");
        curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, single_write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &sw);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 30L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, multi_progress_cb);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &prog);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        out.close();
        print_progress_bar(prog, true);

        if (res != CURLE_OK) {
            LOG(ERROR) << "[C2C] Download failed: " << url << " ("
                       << curl_easy_strerror(res) << ")";
            std::remove(tmp_path.c_str());
            return false;
        }
    } else {
        LOG(INFO) << "[C2C] Parallel download: " << num_threads << " threads, "
                  << (file_size >> 20) << " MB";

        int fd = open(tmp_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) {
            LOG(ERROR) << "[C2C] Cannot create temp file: " << tmp_path;
            return false;
        }
        if (ftruncate(fd, file_size) != 0) {
            LOG(ERROR) << "[C2C] ftruncate failed: " << tmp_path;
            close(fd);
            return false;
        }

        int64_t chunk_size = file_size / num_threads;
        std::atomic<bool> any_failed{false};
        std::vector<std::thread> threads;
        std::vector<ChunkWriter> writers(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            int64_t start = i * chunk_size;
            int64_t end =
                (i == num_threads - 1) ? file_size - 1 : start + chunk_size - 1;
            writers[i] = {fd, start, &prog};

            threads.emplace_back([&, i, start, end]() {
                static constexpr int kMaxRetries = 3;
                int64_t cur_offset = start;

                for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
                    if (attempt > 0) {
                        LOG(WARNING) << "[C2C] Chunk " << i << " retry "
                                     << attempt << "/" << kMaxRetries
                                     << " from offset " << cur_offset;
                    }

                    CURL* curl = curl_easy_init();
                    if (!curl) {
                        any_failed = true;
                        return;
                    }

                    char range_buf[64];
                    snprintf(range_buf, sizeof(range_buf), "%lld-%lld",
                             (long long)cur_offset, (long long)end);
                    writers[i].offset = cur_offset;

                    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                    curl_easy_setopt(curl, CURLOPT_USERAGENT,
                                     "mooncake-c2c/1.0");
                    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION,
                                     CURL_HTTP_VERSION_1_1);
                    curl_easy_setopt(curl, CURLOPT_RANGE, range_buf);
                    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                                     chunk_write_cb);
                    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &writers[i]);
                    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
                    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
                    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 30L);
                    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
                    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION,
                                     multi_progress_cb);
                    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &prog);
                    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

                    CURLcode res = curl_easy_perform(curl);
                    cur_offset = writers[i].offset;
                    curl_easy_cleanup(curl);

                    if (res == CURLE_OK) return;

                    LOG(WARNING) << "[C2C] Chunk " << i
                                 << " error: " << curl_easy_strerror(res);
                }

                LOG(ERROR) << "[C2C] Chunk " << i << " failed after "
                           << kMaxRetries << " retries";
                any_failed = true;
            });
        }

        for (auto& t : threads) t.join();
        close(fd);
        print_progress_bar(prog, true);

        if (any_failed) {
            LOG(ERROR) << "[C2C] Parallel download failed: " << url;
            std::remove(tmp_path.c_str());
            return false;
        }
    }

    if (std::rename(tmp_path.c_str(), local_path.c_str()) != 0) {
        LOG(ERROR) << "[C2C] Rename failed: " << tmp_path << " -> "
                   << local_path;
        std::remove(tmp_path.c_str());
        return false;
    }

    LOG(INFO) << "[C2C] Downloaded: " << local_path;
    return true;
#else
    LOG(ERROR) << "[C2C] URL download requires libcurl (USE_CURL not enabled)";
    return false;
#endif
}

// ============================================================================
// Fetch model params from HuggingFace config.json
// ============================================================================
bool KVAutoConverter::fetch_hf_model_info(const std::string& hf_model,
                                          KVModelInfo& info) {
#ifdef USE_CURL
    std::string url =
        "https://huggingface.co/" + hf_model + "/resolve/main/config.json";
    LOG(INFO) << "[C2C] Fetching HF config: " << url;

    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "mooncake-c2c/1.0");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_string_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        LOG(ERROR) << "[C2C] HF config fetch failed: " << hf_model << " ("
                   << curl_easy_strerror(res) << ")";
        return false;
    }

    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    Json::Value root;
    std::string errs;
    if (!reader->parse(body.data(), body.data() + body.size(), &root, &errs)) {
        LOG(ERROR) << "[C2C] HF config parse error: " << errs;
        return false;
    }

    if (root.isMember("num_hidden_layers"))
        info.num_layers = root["num_hidden_layers"].asInt();
    if (root.isMember("num_key_value_heads"))
        info.num_kv_heads = root["num_key_value_heads"].asInt();

    if (root.isMember("head_dim")) {
        info.head_dim = root["head_dim"].asInt();
    } else if (root.isMember("hidden_size") &&
               root.isMember("num_attention_heads")) {
        info.head_dim =
            root["hidden_size"].asInt() / root["num_attention_heads"].asInt();
    }

    LOG(INFO) << "[C2C] HF config: " << hf_model
              << " (layers=" << info.num_layers
              << ", kv_heads=" << info.num_kv_heads
              << ", head_dim=" << info.head_dim << ")";
    return true;
#else
    LOG(ERROR)
        << "[C2C] HF config fetch requires libcurl (USE_CURL not enabled)";
    return false;
#endif
}

void KVAutoConverter::init(int workers) {
    if (running_.exchange(true)) return;
    last_log_time_ = std::chrono::steady_clock::now();
    for (int i = 0; i < workers; ++i) {
        workers_.emplace_back(&KVAutoConverter::worker, this);
    }
    LOG(INFO) << "[C2C] Auto converter started, workers=" << workers;
}

void KVAutoConverter::shutdown() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();
    for (auto& w : workers_)
        if (w.joinable()) w.join();
    workers_.clear();
    put_fn_ = nullptr;
    batch_put_fn_ = nullptr;
}

// ============================================================================
// Load MLP block array from binary file
// ============================================================================
static bool read_mlp_blocks(std::ifstream& f,
                            std::vector<KVConversionRule::MLPBlock>& blocks,
                            int count, int hidden_dim, int inter_dim) {
    blocks.resize(count);
    for (int i = 0; i < count; ++i) {
        auto& blk = blocks[i];
        blk.norm_weight.resize(hidden_dim);
        blk.w1_weight.resize(static_cast<size_t>(hidden_dim) * inter_dim);
        blk.w2_weight.resize(static_cast<size_t>(inter_dim) * hidden_dim);

        f.read(reinterpret_cast<char*>(blk.norm_weight.data()),
               hidden_dim * sizeof(float));
        f.read(reinterpret_cast<char*>(blk.w1_weight.data()),
               static_cast<size_t>(hidden_dim) * inter_dim * sizeof(float));
        f.read(reinterpret_cast<char*>(blk.w2_weight.data()),
               static_cast<size_t>(inter_dim) * hidden_dim * sizeof(float));
        if (!f) return false;
    }
    return true;
}

// ============================================================================
// Load C2C projector weight file (v2 format with magic 0xC2C20002)
// ============================================================================
bool KVAutoConverter::load_projector_file(const std::string& path,
                                          KVConversionRule& rule) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        LOG(ERROR) << "[C2C] Failed to open projector file: " << path;
        return false;
    }

    // Detect format: read first 4 bytes
    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

    if (magic != kMagicV2) {
        // v1 format has num_layers as first field (typically 1-200)
        int32_t maybe_layers = static_cast<int32_t>(magic);
        if (maybe_layers > 0 && maybe_layers <= 1024) {
            LOG(ERROR) << "[C2C] Detected v1 projector format (layers="
                       << maybe_layers << "). Please re-convert weights with "
                       << "convert_c2c_weights.py to get v2 format with full "
                       << "C2CProjector architecture.";
        } else {
            LOG(ERROR) << "[C2C] Unknown projector format (magic=0x"
                       << std::hex << magic << std::dec
                       << "), expected v2 (0xC2C20002)";
        }
        return false;
    }

    // v2 header
    f.read(reinterpret_cast<char*>(&rule.num_layers), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&rule.src_dim), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&rule.tgt_dim), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&rule.hidden_dim), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&rule.inter_dim), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&rule.scalar_inter_dim), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&rule.num_kv_heads), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&rule.head_dim), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&rule.num_mlp1_blocks), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&rule.num_proj_mlp2_blocks), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&rule.num_scalar_mlp2_blocks), sizeof(int32_t));

    // Validate header
    if (rule.num_layers <= 0 || rule.num_layers > 1024 ||
        rule.src_dim <= 0 || rule.src_dim > 65536 ||
        rule.tgt_dim <= 0 || rule.tgt_dim > 65536 ||
        rule.hidden_dim <= 0 || rule.hidden_dim > 65536 ||
        rule.inter_dim <= 0 || rule.inter_dim > 65536 ||
        rule.num_kv_heads <= 0 || rule.num_kv_heads > 1024) {
        LOG(ERROR) << "[C2C] Invalid v2 header: layers=" << rule.num_layers
                   << " src=" << rule.src_dim << " tgt=" << rule.tgt_dim
                   << " hidden=" << rule.hidden_dim
                   << " inter=" << rule.inter_dim
                   << " kv_heads=" << rule.num_kv_heads;
        return false;
    }

    size_t in_sz = static_cast<size_t>(rule.src_dim) * rule.hidden_dim;
    size_t proj_out_sz = static_cast<size_t>(rule.hidden_dim) * rule.tgt_dim;
    size_t scalar_head_sz = static_cast<size_t>(rule.hidden_dim) * rule.num_kv_heads;

    rule.layers.resize(rule.num_layers);

    for (int32_t L = 0; L < rule.num_layers; ++L) {
        auto& lw = rule.layers[L];

        // Gate logits
        f.read(reinterpret_cast<char*>(&lw.key_gate_logit), sizeof(float));
        f.read(reinterpret_cast<char*>(&lw.value_gate_logit), sizeof(float));

        for (int kv = 0; kv < 2; ++kv) {
            bool is_key = (kv == 0);

            // Input projection
            auto& in_w = is_key ? lw.key_in_weight : lw.value_in_weight;
            auto& in_b = is_key ? lw.key_in_bias : lw.value_in_bias;
            in_w.resize(in_sz);
            in_b.resize(rule.hidden_dim);
            f.read(reinterpret_cast<char*>(in_w.data()), in_sz * sizeof(float));
            f.read(reinterpret_cast<char*>(in_b.data()),
                   rule.hidden_dim * sizeof(float));

            // MLP1 blocks
            auto& mlp1 = is_key ? lw.key_mlp1 : lw.value_mlp1;
            if (!read_mlp_blocks(f, mlp1, rule.num_mlp1_blocks,
                                 rule.hidden_dim, rule.inter_dim))
                goto read_error;

            // Proj MLP2 blocks
            auto& proj_mlp2 = is_key ? lw.key_proj_mlp2 : lw.value_proj_mlp2;
            if (!read_mlp_blocks(f, proj_mlp2, rule.num_proj_mlp2_blocks,
                                 rule.hidden_dim, rule.inter_dim))
                goto read_error;

            // Proj output
            auto& proj_out_w = is_key ? lw.key_proj_out_weight : lw.value_proj_out_weight;
            auto& proj_out_b = is_key ? lw.key_proj_out_bias : lw.value_proj_out_bias;
            proj_out_w.resize(proj_out_sz);
            proj_out_b.resize(rule.tgt_dim);
            f.read(reinterpret_cast<char*>(proj_out_w.data()),
                   proj_out_sz * sizeof(float));
            f.read(reinterpret_cast<char*>(proj_out_b.data()),
                   rule.tgt_dim * sizeof(float));

            // Scalar MLP2 blocks
            auto& scalar_mlp2 = is_key ? lw.key_scalar_mlp2 : lw.value_scalar_mlp2;
            if (!read_mlp_blocks(f, scalar_mlp2, rule.num_scalar_mlp2_blocks,
                                 rule.hidden_dim, rule.scalar_inter_dim))
                goto read_error;

            // Scalar head
            auto& sh_w = is_key ? lw.key_scalar_head_weight : lw.value_scalar_head_weight;
            auto& sh_b = is_key ? lw.key_scalar_head_bias : lw.value_scalar_head_bias;
            sh_w.resize(scalar_head_sz);
            sh_b.resize(rule.num_kv_heads);
            f.read(reinterpret_cast<char*>(sh_w.data()),
                   scalar_head_sz * sizeof(float));
            f.read(reinterpret_cast<char*>(sh_b.data()),
                   rule.num_kv_heads * sizeof(float));
        }
    }

    if (!f) {
    read_error:
        LOG(ERROR) << "[C2C] Failed to read projector file: " << path;
        return false;
    }

    // =========================================================================
    // Pre-transpose all weight matrices for cache-friendly row-major access
    // Python converter already pre-transposes, but we transpose again here
    // to match the expected layout: B_T[N,K] for sgemv(NoTrans, N, K)
    // Note: Python writes [K,N] (pre-transposed from PyTorch [N,K]),
    // so C++ reads [K,N] and we need [N,K] for CblasNoTrans.
    // Actually, Python .T gives us [K,N], and we want B_T[N,K] for
    // sgemv(NoTrans, N, K, ..., B_T, K, A, ...) -> C[N] = B_T @ A
    // So we transpose [K,N] -> [N,K] here.
    // =========================================================================
    for (int32_t L = 0; L < rule.num_layers; ++L) {
        auto& lw = rule.layers[L];

        for (int kv = 0; kv < 2; ++kv) {
            bool is_key = (kv == 0);

            // in_weight: stored [src, hidden] -> transpose to [hidden, src]
            auto& in_w = is_key ? lw.key_in_weight : lw.value_in_weight;
            transpose_matrix(in_w, rule.src_dim, rule.hidden_dim);

            // MLP blocks: w1 stored [hidden, inter] -> [inter, hidden]
            //             w2 stored [inter, hidden] -> [hidden, inter]
            auto transpose_blocks = [](std::vector<KVConversionRule::MLPBlock>& blocks,
                                       int hidden, int inter) {
                for (auto& blk : blocks) {
                    transpose_matrix(blk.w1_weight, hidden, inter);
                    transpose_matrix(blk.w2_weight, inter, hidden);
                }
            };

            auto& mlp1 = is_key ? lw.key_mlp1 : lw.value_mlp1;
            transpose_blocks(mlp1, rule.hidden_dim, rule.inter_dim);

            auto& proj_mlp2 = is_key ? lw.key_proj_mlp2 : lw.value_proj_mlp2;
            transpose_blocks(proj_mlp2, rule.hidden_dim, rule.inter_dim);

            auto& scalar_mlp2 = is_key ? lw.key_scalar_mlp2 : lw.value_scalar_mlp2;
            transpose_blocks(scalar_mlp2, rule.hidden_dim, rule.scalar_inter_dim);

            // proj_out: stored [hidden, tgt] -> [tgt, hidden]
            auto& proj_out_w = is_key ? lw.key_proj_out_weight : lw.value_proj_out_weight;
            transpose_matrix(proj_out_w, rule.hidden_dim, rule.tgt_dim);

            // scalar_head: stored [hidden, num_heads] -> [num_heads, hidden]
            auto& sh_w = is_key ? lw.key_scalar_head_weight : lw.value_scalar_head_weight;
            transpose_matrix(sh_w, rule.hidden_dim, rule.num_kv_heads);
        }
    }

    LOG(INFO) << "[C2C] Loaded v2 projector: " << path
              << " (layers=" << rule.num_layers
              << ", src=" << rule.src_dim << ", tgt=" << rule.tgt_dim
              << ", hidden=" << rule.hidden_dim << ", inter=" << rule.inter_dim
              << ", kv_heads=" << rule.num_kv_heads
              << ", mlp1=" << rule.num_mlp1_blocks
              << ", proj_mlp2=" << rule.num_proj_mlp2_blocks
              << ", scalar_mlp2=" << rule.num_scalar_mlp2_blocks << ")";
    return true;
}

// ============================================================================
// Parse buffer_size from C2C config JSON
// ============================================================================
size_t KVAutoConverter::parse_buffer_size(const std::string& config_json) {
    if (config_json.empty()) return kC2cDefaultBufferSize;

    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    Json::Value root;
    std::string errs;

    if (!reader->parse(config_json.data(),
                       config_json.data() + config_json.size(), &root, &errs))
        return kC2cDefaultBufferSize;

    if (root.isMember("buffer_size") && root["buffer_size"].isUInt64())
        return static_cast<size_t>(root["buffer_size"].asUInt64());

    return kC2cDefaultBufferSize;
}

bool KVAutoConverter::load_config(const std::string& config_json) {
    if (config_json.empty()) return false;

    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    Json::Value root;
    std::string errs;

    if (!reader->parse(config_json.data(),
                       config_json.data() + config_json.size(), &root, &errs)) {
        LOG(ERROR) << "[C2C] Config parse error: " << errs;
        return false;
    }

    if (root.isMember("models") && root["models"].isArray()) {
        for (const auto& m : root["models"]) {
            KVModelInfo info;
            info.model_id = m["model_id"].asString();
            info.hf_model = m.get("hf_model", "").asString();
            info.num_layers = m.get("num_layers", 0).asInt();
            info.num_kv_heads = m.get("num_kv_heads", 0).asInt();
            info.head_dim = m.get("head_dim", 0).asInt();

            if (!info.hf_model.empty() &&
                (info.num_layers == 0 || info.num_kv_heads == 0 ||
                 info.head_dim == 0)) {
                KVModelInfo hf_info = info;
                if (fetch_hf_model_info(info.hf_model, hf_info)) {
                    if (info.num_layers == 0)
                        info.num_layers = hf_info.num_layers;
                    if (info.num_kv_heads == 0)
                        info.num_kv_heads = hf_info.num_kv_heads;
                    if (info.head_dim == 0) info.head_dim = hf_info.head_dim;
                } else {
                    LOG(WARNING)
                        << "[C2C] HF fetch failed for: " << info.hf_model
                        << ", using manual config";
                }
            }

            add_model(info);
        }
    }

    if (root.isMember("rules") && root["rules"].isArray()) {
        for (const auto& r : root["rules"]) {
            KVConversionRule rule;
            rule.source_model = r["source_model"].asString();
            rule.target_model = r["target_model"].asString();
            rule.source_model_name = r.get("source_model_name", "").asString();
            rule.target_model_name = r.get("target_model_name", "").asString();
            rule.projector_url = r.get("projector_url", "").asString();
            rule.projector_file = r.get("projector_file", "").asString();

            std::string projector_path;
            if (!rule.projector_url.empty()) {
                projector_path = url_to_cache_path(rule.projector_url);
                if (!download_file(rule.projector_url, projector_path)) {
                    LOG(ERROR)
                        << "[C2C] Skip rule (download failed): "
                        << rule.source_model << " -> " << rule.target_model;
                    continue;
                }
            } else if (!rule.projector_file.empty()) {
                projector_path = rule.projector_file;
            }

            if (!projector_path.empty()) {
                if (!load_projector_file(projector_path, rule)) {
                    LOG(ERROR) << "[C2C] Skip rule: " << rule.source_model
                               << " -> " << rule.target_model;
                    continue;
                }
            }

            if (!rule.is_valid()) {
                LOG(ERROR) << "[C2C] Rule missing projector weights: "
                           << rule.source_model << " -> " << rule.target_model;
                continue;
            }

            add_rule(rule);
        }
    }

    LOG(INFO) << "[C2C] Config loaded, " << rules_.size() << " rules";
    return !rules_.empty();
}

void KVAutoConverter::add_model(const KVModelInfo& info) {
    models_[info.model_id] = info;
    LOG(INFO) << "[C2C] Model: " << info.model_id
              << " (layers=" << info.num_layers
              << ", kv_heads=" << info.num_kv_heads
              << ", head_dim=" << info.head_dim << ")";
}

void KVAutoConverter::add_rule(const KVConversionRule& rule) {
    rules_[rule.source_model] = rule;
    LOG(INFO) << "[C2C] Rule: " << rule.source_model << " -> "
              << rule.target_model;
}

const KVConversionRule* KVAutoConverter::match_rule(const std::string& key) {
    for (const auto& [id, rule] : rules_) {
        if (!rule.source_model_name.empty() &&
            key.find(rule.source_model_name) != std::string::npos) {
            return &rule;
        }
    }
    return nullptr;
}

void KVAutoConverter::on_put(const std::string& key, const void* data,
                             size_t size) {
    if (!running_) return;
    if (key.find("::") != std::string::npos) return;

    const KVConversionRule* rule = match_rule(key);
    if (!rule) return;

    auto src_it = models_.find(rule->source_model);
    auto tgt_it = models_.find(rule->target_model);
    if (src_it == models_.end() || tgt_it == models_.end()) return;

    LOG(INFO) << "[C2C] on_put: matched " << rule->source_model
              << " for key: " << key;

    Task task;
    task.src_key = key;
    task.tgt_key = key;
    size_t pos = task.tgt_key.find(rule->source_model_name);
    if (pos != std::string::npos) {
        task.tgt_key.replace(pos, rule->source_model_name.length(),
                             rule->target_model_name);
    }
    task.data.resize(size);
    std::memcpy(task.data.data(), data, size);
    task.src_info = src_it->second;
    task.tgt_info = tgt_it->second;
    task.rule = rule;

    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.push(std::vector<Task>{std::move(task)});
    }
    cv_.notify_one();
}

void KVAutoConverter::on_batch_put(const std::vector<std::string>& keys,
                                   const std::vector<void*>& buffers,
                                   const std::vector<size_t>& sizes) {
    if (!running_ || keys.empty()) return;

    struct GroupKey {
        const KVConversionRule* rule;
        bool is_key;
        bool operator==(const GroupKey& o) const {
            return rule == o.rule && is_key == o.is_key;
        }
    };
    struct GroupKeyHash {
        size_t operator()(const GroupKey& k) const {
            return std::hash<const void*>()(k.rule) ^
                   (std::hash<bool>()(k.is_key) << 1);
        }
    };
    std::unordered_map<GroupKey, std::vector<Task>, GroupKeyHash> groups;

    for (size_t i = 0; i < keys.size(); ++i) {
        const auto& key = keys[i];
        if (key.find("::") != std::string::npos) continue;

        const KVConversionRule* rule = match_rule(key);
        if (!rule) continue;

        auto src_it = models_.find(rule->source_model);
        auto tgt_it = models_.find(rule->target_model);
        if (src_it == models_.end() || tgt_it == models_.end()) continue;

        size_t upos = key.rfind('_');
        if (upos == std::string::npos || upos + 1 >= key.size()) continue;
        bool is_key = (key[upos + 1] == 'k');

        Task task;
        task.src_key = key;
        task.tgt_key = key;
        size_t pos = task.tgt_key.find(rule->source_model_name);
        if (pos != std::string::npos) {
            task.tgt_key.replace(pos, rule->source_model_name.length(),
                                 rule->target_model_name);
        }
        task.data.resize(sizes[i]);
        std::memcpy(task.data.data(), buffers[i], sizes[i]);
        task.src_info = src_it->second;
        task.tgt_info = tgt_it->second;
        task.rule = rule;

        groups[{rule, is_key}].push_back(std::move(task));
    }

    if (!groups.empty()) {
        std::lock_guard<std::mutex> lk(mtx_);
        for (auto& [gk, batch] : groups) {
            queue_.push(std::move(batch));
        }
    }
    cv_.notify_all();
}

void KVAutoConverter::worker() {
    while (running_) {
        std::vector<Task> batch;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return !queue_.empty() || !running_; });
            if (!running_ && queue_.empty()) break;
            if (queue_.empty()) continue;
            batch = std::move(queue_.front());
            queue_.pop();
        }
        if (batch.size() == 1) {
            convert(batch[0]);
        } else {
            convert_batch(batch);
        }

        std::lock_guard<std::mutex> log_lk(log_mtx_);
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           now - last_log_time_)
                           .count();
        size_t pending;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            pending = queue_.size();
        }
        if (pending == 0 || elapsed >= 5) {
            auto keys = completed_keys_.exchange(0, std::memory_order_relaxed);
            auto pages =
                completed_pages_.exchange(0, std::memory_order_relaxed);
            auto ms = total_ms_.exchange(0, std::memory_order_relaxed);
            if (keys > 0) {
                if (pending == 0) {
                    LOG(INFO) << "[C2C] Done: " << keys << " keys, " << pages
                              << " pages, " << ms << "ms";
                } else {
                    LOG(INFO)
                        << "[C2C] Progress: " << keys << " keys, " << pages
                        << " pages, " << ms << "ms, " << pending << " pending";
                }
            }
            last_log_time_ = now;
        }
    }
}

// ============================================================================
// Scalar GELU: 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3)))
// Uses std::tanh for precision (Pade approximation has 0.022 max error
// in GELU's critical range, unacceptable over 28+ layers)
// ============================================================================
static inline float gelu(float v) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    float v3 = v * v * v;
    float inner = sqrt_2_over_pi * (v + coeff * v3);
    return 0.5f * v * (1.0f + std::tanh(inner));
}

// ============================================================================
// AVX2 vectorized GELU — process 8 floats, scalar tanh fallback
// AVX2 has no native tanh; scalar std::tanh is more accurate than any
// polynomial that fits in reasonable instruction count
// ============================================================================
#ifdef USE_AVX2
static inline __m256 gelu_avx2(__m256 v) {
    // Extract to scalar, compute exact GELU, reload
    // This is ~4x slower than pure SIMD but the bottleneck is sgemm not GELU
    alignas(32) float vf[8], rf[8];
    _mm256_store_ps(vf, v);
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    for (int i = 0; i < 8; ++i) {
        float x = vf[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        rf[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
    return _mm256_load_ps(rf);
}
#endif

// ============================================================================
// RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
// ============================================================================
static void rmsnorm(const float* x, const float* weight, float* out,
                    int M, int dim) {
    for (int m = 0; m < M; ++m) {
        const float* xr = x + m * dim;
        float* or_ = out + m * dim;

        float sum_sq = 0.0f;
#ifdef USE_AVX2
        __m256 vsum = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 v = _mm256_loadu_ps(xr + i);
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(v, v));
        }
        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        sum_sq = _mm_cvtss_f32(s);
        for (; i < dim; ++i) sum_sq += xr[i] * xr[i];
#else
        for (int i = 0; i < dim; ++i) sum_sq += xr[i] * xr[i];
#endif
        float scale = 1.0f / std::sqrt(sum_sq / dim + kRmsNormEps);

#ifdef USE_AVX2
        __m256 vscale = _mm256_set1_ps(scale);
        i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 v = _mm256_loadu_ps(xr + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            _mm256_storeu_ps(or_ + i, _mm256_mul_ps(_mm256_mul_ps(v, vscale), w));
        }
        for (; i < dim; ++i) or_[i] = xr[i] * scale * weight[i];
#else
        for (int i = 0; i < dim; ++i) or_[i] = xr[i] * scale * weight[i];
#endif
    }
}

// ============================================================================
// matmul_bias: C = A @ B_T + bias (no activation)
// A: [M, K], B_T: [N, K] (pre-transposed), C: [M, N], bias: [N]
// ============================================================================
static void matmul_bias(const float* A, const float* B_T, const float* bias,
                        float* C, int M, int K, int N) {
#ifdef USE_OPENBLAS
    if (M == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0f, B_T, K, A, 1,
                    0.0f, C, 1);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A,
                    K, B_T, K, 0.0f, C, N);
    }
    // Add bias per row
    for (int m = 0; m < M; ++m) {
        float* row = C + m * N;
#ifdef USE_AVX2
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256 c = _mm256_loadu_ps(row + n);
            __m256 b = _mm256_loadu_ps(bias + n);
            _mm256_storeu_ps(row + n, _mm256_add_ps(c, b));
        }
        for (; n < N; ++n) row[n] += bias[n];
#else
        for (int n = 0; n < N; ++n) row[n] += bias[n];
#endif
    }
#else
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = bias[n];
            const float* b_row = B_T + n * K;
            for (int k = 0; k < K; ++k) sum += A[m * K + k] * b_row[k];
            C[m * N + n] = sum;
        }
    }
#endif
}

// ============================================================================
// matmul_gelu: C = GELU(A @ B_T) — no bias, for MLP w1
// ============================================================================
static void matmul_gelu(const float* A, const float* B_T, float* C,
                        int M, int K, int N) {
#ifdef USE_OPENBLAS
    if (M == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0f, B_T, K, A, 1,
                    0.0f, C, 1);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A,
                    K, B_T, K, 0.0f, C, N);
    }
    int total = M * N;
#ifdef USE_AVX2
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 v = _mm256_loadu_ps(C + i);
        _mm256_storeu_ps(C + i, gelu_avx2(v));
    }
    for (; i < total; ++i) C[i] = gelu(C[i]);
#else
    for (int i = 0; i < total; ++i) C[i] = gelu(C[i]);
#endif
#else
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const float* b_row = B_T + n * K;
            for (int k = 0; k < K; ++k) sum += A[m * K + k] * b_row[k];
            C[m * N + n] = gelu(sum);
        }
    }
#endif
}

// ============================================================================
// matmul_add: dst += A @ B_T — residual connection for w2
// ============================================================================
static void matmul_add(const float* A, const float* B_T, float* dst,
                       int M, int K, int N) {
#ifdef USE_OPENBLAS
    if (M == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0f, B_T, K, A, 1,
                    1.0f, dst, 1);  // beta=1 for accumulate
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A,
                    K, B_T, K, 1.0f, dst, N);  // beta=1 for accumulate
    }
#else
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const float* b_row = B_T + n * K;
            for (int k = 0; k < K; ++k) sum += A[m * K + k] * b_row[k];
            dst[m * N + n] += sum;
        }
    }
#endif
}

// ============================================================================
// MLP block: x += w2(GELU(w1(RMSNorm(x))))
// norm_buf: [M, hidden], w1_buf: [M, inter] — caller-provided scratch
// ============================================================================
static void mlp_block(float* x, const KVConversionRule::MLPBlock& blk,
                      int M, int hidden, int inter,
                      float* norm_buf, float* w1_buf) {
    rmsnorm(x, blk.norm_weight.data(), norm_buf, M, hidden);
    matmul_gelu(norm_buf, blk.w1_weight.data(), w1_buf, M, hidden, inter);
    matmul_add(w1_buf, blk.w2_weight.data(), x, M, inter, hidden);
}

// ============================================================================
// Apply sigmoid to array in-place
// ============================================================================
static void sigmoid_inplace(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

// ============================================================================
// Apply per-head scalar gate: out[m, h*head_dim + d] *= gate * scalar[m, h]
// out: [M, num_heads * head_dim], scalar: [M, num_heads]
// ============================================================================
static void apply_scalar_gate(float* out, const float* scalar, float gate,
                               int M, int num_heads, int head_dim) {
    for (int m = 0; m < M; ++m) {
        float* out_row = out + m * num_heads * head_dim;
        const float* scalar_row = scalar + m * num_heads;
        for (int h = 0; h < num_heads; ++h) {
            float s = gate * scalar_row[h];
#ifdef USE_AVX2
            __m256 vs = _mm256_set1_ps(s);
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                float* p = out_row + h * head_dim + d;
                __m256 v = _mm256_loadu_ps(p);
                _mm256_storeu_ps(p, _mm256_mul_ps(v, vs));
            }
            for (; d < head_dim; ++d) {
                out_row[h * head_dim + d] *= s;
            }
#else
            for (int d = 0; d < head_dim; ++d) {
                out_row[h * head_dim + d] *= s;
            }
#endif
        }
    }
}

// ============================================================================
// Thread-local scratch buffers for c2c_project_layer
// ============================================================================
struct ProjectionBuffers {
    std::vector<float> h;          // [M, hidden_dim] — shared embedding
    std::vector<float> proj_h;     // [M, hidden_dim] — projection path
    std::vector<float> scalar_h;   // [M, hidden_dim] — scalar path
    std::vector<float> norm_buf;   // [M, hidden_dim] — RMSNorm scratch
    std::vector<float> w1_buf;     // [M, max(inter_dim, scalar_inter_dim)]
    std::vector<float> scalar_out; // [M, num_heads] — scalar output

    void ensure(int M, int hidden, int inter, int scalar_inter, int num_heads) {
        size_t h_sz = static_cast<size_t>(M) * hidden;
        size_t w1_sz = static_cast<size_t>(M) * std::max(inter, scalar_inter);
        size_t sc_sz = static_cast<size_t>(M) * num_heads;
        if (h.size() < h_sz) h.resize(h_sz);
        if (proj_h.size() < h_sz) proj_h.resize(h_sz);
        if (scalar_h.size() < h_sz) scalar_h.resize(h_sz);
        if (norm_buf.size() < h_sz) norm_buf.resize(h_sz);
        if (w1_buf.size() < w1_sz) w1_buf.resize(w1_sz);
        if (scalar_out.size() < sc_sz) scalar_out.resize(sc_sz);
    }
};

// ============================================================================
// C2C projection: full C2CProjector architecture
// ============================================================================
void KVAutoConverter::c2c_project_layer(
    const float* src, float* out, const KVConversionRule& rule,
    const KVConversionRule::LayerWeights& lw, bool is_key, int M) {

    int src_dim = rule.src_dim;
    int hidden = rule.hidden_dim;
    int inter = rule.inter_dim;
    int scalar_inter = rule.scalar_inter_dim;
    int num_heads = rule.num_kv_heads;
    int hdim = rule.head_dim;

    thread_local ProjectionBuffers buf;
    buf.ensure(M, hidden, inter, scalar_inter, num_heads);

    const auto& in_w = is_key ? lw.key_in_weight : lw.value_in_weight;
    const auto& in_b = is_key ? lw.key_in_bias : lw.value_in_bias;
    const auto& mlp1 = is_key ? lw.key_mlp1 : lw.value_mlp1;
    const auto& proj_mlp2 = is_key ? lw.key_proj_mlp2 : lw.value_proj_mlp2;
    const auto& proj_out_w = is_key ? lw.key_proj_out_weight : lw.value_proj_out_weight;
    const auto& proj_out_b = is_key ? lw.key_proj_out_bias : lw.value_proj_out_bias;
    const auto& scalar_mlp2 = is_key ? lw.key_scalar_mlp2 : lw.value_scalar_mlp2;
    const auto& sh_w = is_key ? lw.key_scalar_head_weight : lw.value_scalar_head_weight;
    const auto& sh_b = is_key ? lw.key_scalar_head_bias : lw.value_scalar_head_bias;
    float gate_logit = is_key ? lw.key_gate_logit : lw.value_gate_logit;

    // Binary gate: paper uses (logit > 0) ? 1.0 : 0.0
    float gate = (gate_logit > 0.0f) ? 1.0f : 0.0f;

    // If gate is zero, output is all zeros — skip computation
    if (gate == 0.0f) {
        std::memset(out, 0, static_cast<size_t>(M) * rule.tgt_dim * sizeof(float));
        return;
    }

    // 1. Input projection: h = Linear(src) — NO activation
    matmul_bias(src, in_w.data(), in_b.data(), buf.h.data(),
                M, src_dim, hidden);

    // 2. MLP1 blocks: h += w2(GELU(w1(RMSNorm(h))))
    for (const auto& blk : mlp1) {
        mlp_block(buf.h.data(), blk, M, hidden, inter,
                  buf.norm_buf.data(), buf.w1_buf.data());
    }

    // 3. Projection path: copy h, run proj_mlp2, then proj_out
    size_t h_bytes = static_cast<size_t>(M) * hidden * sizeof(float);
    std::memcpy(buf.proj_h.data(), buf.h.data(), h_bytes);
    for (const auto& blk : proj_mlp2) {
        mlp_block(buf.proj_h.data(), blk, M, hidden, inter,
                  buf.norm_buf.data(), buf.w1_buf.data());
    }
    matmul_bias(buf.proj_h.data(), proj_out_w.data(), proj_out_b.data(),
                out, M, hidden, rule.tgt_dim);

    // 4. Scalar path: copy h, run scalar_mlp2, then scalar_head + sigmoid
    std::memcpy(buf.scalar_h.data(), buf.h.data(), h_bytes);
    for (const auto& blk : scalar_mlp2) {
        mlp_block(buf.scalar_h.data(), blk, M, hidden, scalar_inter,
                  buf.norm_buf.data(), buf.w1_buf.data());
    }
    matmul_bias(buf.scalar_h.data(), sh_w.data(), sh_b.data(),
                buf.scalar_out.data(), M, hidden, num_heads);
    sigmoid_inplace(buf.scalar_out.data(), M * num_heads);

    // 5. Apply: out *= gate * scalar (per-head broadcast)
    apply_scalar_gate(out, buf.scalar_out.data(), gate, M, num_heads, hdim);
}

// ============================================================================
// SIMD optimized bf16 <-> float batch conversion
// ============================================================================
#ifdef USE_AVX2
static void bf16_to_float_avx2(const uint16_t* src, float* dst, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i bf16_vals =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256i lo = _mm256_cvtepu16_epi32(bf16_vals);
        __m256i shifted = _mm256_slli_epi32(lo, 16);
        __m256 floats = _mm256_castsi256_ps(shifted);
        _mm256_storeu_ps(dst + i, floats);
    }
    for (; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &bits, sizeof(float));
    }
}

static void float_to_bf16_avx2(const float* src, uint16_t* dst, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 floats = _mm256_loadu_ps(src + i);
        __m256i ints = _mm256_castps_si256(floats);
        __m256i shifted = _mm256_srli_epi32(ints, 16);
        __m128i lo = _mm256_castsi256_si128(shifted);
        __m128i hi = _mm256_extracti128_si256(shifted, 1);
        __m128i packed = _mm_packus_epi32(lo, hi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), packed);
    }
    for (; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &src[i], sizeof(float));
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
}
#endif

static void bf16_to_float_batch(const uint16_t* src, float* dst, int n) {
#ifdef USE_AVX2
    bf16_to_float_avx2(src, dst, n);
#else
    for (int i = 0; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &bits, sizeof(float));
    }
#endif
}

static void float_to_bf16_batch(const float* src, uint16_t* dst, int n) {
#ifdef USE_AVX2
    float_to_bf16_avx2(src, dst, n);
#else
    for (int i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &src[i], sizeof(float));
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
#endif
}

// ============================================================================
// Parse SGLang key format: model_hash_tprank_type
// ============================================================================
static bool parse_sglang_key(const std::string& key, bool& is_key) {
    size_t pos = key.rfind('_');
    if (pos == std::string::npos || pos == 0) return false;
    char type = key[pos + 1];
    if (type != 'k' && type != 'v') return false;
    is_key = (type == 'k');
    return true;
}

// ============================================================================
// Conversion core (SGLang packed all-layers format)
// ============================================================================
void KVAutoConverter::convert(Task& t) {
    if (!t.rule || !t.rule->is_valid()) {
        LOG(ERROR) << "[C2C] Invalid rule for: " << t.src_key;
        return;
    }

    bool is_key;
    if (!parse_sglang_key(t.src_key, is_key)) {
        LOG(ERROR) << "[C2C] Failed to parse key: " << t.src_key;
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    int32_t src_dim = t.rule->src_dim;
    int32_t tgt_dim = t.rule->tgt_dim;
    int32_t num_proj_layers = t.rule->num_layers;
    int32_t num_src_layers = t.src_info.num_layers;

    size_t elem_sz = 2;  // bf16
    size_t layer_size = src_dim * elem_sz;
    size_t page_size = num_src_layers * layer_size;

    if (page_size == 0 || t.data.size() % page_size != 0) {
        LOG(ERROR) << "[C2C] Invalid data for: " << t.src_key
                   << " (size=" << t.data.size() << ", page_size=" << page_size
                   << ")";
        return;
    }

    int32_t num_pages = static_cast<int32_t>(t.data.size() / page_size);
    if (num_pages <= 0) return;

    int32_t layer_offset = num_src_layers - num_proj_layers;
    if (layer_offset < 0) layer_offset = 0;

    size_t tgt_layer_size = tgt_dim * elem_sz;
    size_t tgt_page_size = num_proj_layers * tgt_layer_size;
    std::vector<uint8_t> tgt_data(num_pages * tgt_page_size);

    const uint8_t* src_base = t.data.data();
    uint8_t* tgt_base = tgt_data.data();

    // thread_local buffers for bf16 <-> float conversion
    thread_local std::vector<float> src_batch, tgt_batch;
    size_t src_need = num_pages * src_dim;
    size_t tgt_need = num_pages * tgt_dim;
    if (src_batch.size() < src_need) src_batch.resize(src_need);
    if (tgt_batch.size() < tgt_need) tgt_batch.resize(tgt_need);

    for (int32_t proj_layer = 0; proj_layer < num_proj_layers; ++proj_layer) {
        int32_t src_layer = layer_offset + proj_layer;

        for (int32_t p = 0; p < num_pages; ++p) {
            const uint8_t* src_ptr =
                src_base + p * page_size + src_layer * layer_size;
            bf16_to_float_batch(reinterpret_cast<const uint16_t*>(src_ptr),
                                src_batch.data() + p * src_dim, src_dim);
        }

        c2c_project_layer(src_batch.data(), tgt_batch.data(),
                          *t.rule, t.rule->layers[proj_layer],
                          is_key, num_pages);

        for (int32_t p = 0; p < num_pages; ++p) {
            uint8_t* tgt_ptr =
                tgt_base + p * tgt_page_size + proj_layer * tgt_layer_size;
            float_to_bf16_batch(tgt_batch.data() + p * tgt_dim,
                                reinterpret_cast<uint16_t*>(tgt_ptr), tgt_dim);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    if (put_fn_) {
        int ret = put_fn_(t.tgt_key, tgt_data.data(), tgt_data.size());
        if (ret != 0) {
            LOG(ERROR) << "[C2C] put failed: " << t.tgt_key << " ret=" << ret;
        }
        completed_keys_.fetch_add(1, std::memory_order_relaxed);
        completed_pages_.fetch_add(num_pages, std::memory_order_relaxed);
        total_ms_.fetch_add(ms, std::memory_order_relaxed);
    }
}

// ============================================================================
// Batch conversion: merge same-rule same-type tasks into large matrix
// ============================================================================
void KVAutoConverter::convert_batch(std::vector<Task>& tasks) {
    if (tasks.empty()) return;
    if (tasks.size() == 1) {
        convert(tasks[0]);
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    const KVConversionRule* rule = tasks[0].rule;
    bool is_key;
    if (!parse_sglang_key(tasks[0].src_key, is_key)) {
        for (auto& t : tasks) convert(t);
        return;
    }

    int32_t src_dim = rule->src_dim;
    int32_t tgt_dim = rule->tgt_dim;
    int32_t num_proj_layers = rule->num_layers;
    int32_t num_src_layers = tasks[0].src_info.num_layers;

    size_t elem_sz = 2;
    size_t layer_size = src_dim * elem_sz;
    size_t page_size = num_src_layers * layer_size;

    if (page_size == 0) {
        LOG(ERROR) << "[C2C] Zero page_size in batch convert";
        return;
    }

    int32_t total_pages = 0;
    std::vector<int32_t> page_offsets(tasks.size() + 1);
    std::vector<int32_t> task_pages(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i) {
        page_offsets[i] = total_pages;
        task_pages[i] = static_cast<int32_t>(tasks[i].data.size() / page_size);
        total_pages += task_pages[i];
    }
    page_offsets[tasks.size()] = total_pages;

    int32_t layer_offset = num_src_layers - num_proj_layers;
    if (layer_offset < 0) layer_offset = 0;

    size_t tgt_layer_size = tgt_dim * elem_sz;
    size_t tgt_page_size = num_proj_layers * tgt_layer_size;
    std::vector<std::vector<uint8_t>> tgt_data(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i) {
        tgt_data[i].resize(task_pages[i] * tgt_page_size);
    }

    thread_local std::vector<float> src_batch, tgt_batch;
    size_t src_need = total_pages * src_dim;
    size_t tgt_need = total_pages * tgt_dim;
    if (src_batch.size() < src_need) src_batch.resize(src_need);
    if (tgt_batch.size() < tgt_need) tgt_batch.resize(tgt_need);

    for (int32_t proj_layer = 0; proj_layer < num_proj_layers; ++proj_layer) {
        int32_t src_layer = layer_offset + proj_layer;

        for (size_t ti = 0; ti < tasks.size(); ++ti) {
            const uint8_t* base = tasks[ti].data.data();
            int32_t np = task_pages[ti];
            int32_t off = page_offsets[ti];
            for (int32_t p = 0; p < np; ++p) {
                auto* bf16 = reinterpret_cast<const uint16_t*>(
                    base + p * page_size + src_layer * layer_size);
                bf16_to_float_batch(
                    bf16, src_batch.data() + (off + p) * src_dim, src_dim);
            }
        }

        c2c_project_layer(src_batch.data(), tgt_batch.data(),
                          *rule, rule->layers[proj_layer],
                          is_key, total_pages);

        for (size_t ti = 0; ti < tasks.size(); ++ti) {
            uint8_t* base = tgt_data[ti].data();
            int32_t np = task_pages[ti];
            int32_t off = page_offsets[ti];
            for (int32_t p = 0; p < np; ++p) {
                auto* bf16 = reinterpret_cast<uint16_t*>(
                    base + p * tgt_page_size + proj_layer * tgt_layer_size);
                float_to_bf16_batch(tgt_batch.data() + (off + p) * tgt_dim,
                                    bf16, tgt_dim);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    if (batch_put_fn_) {
        std::vector<std::string> keys(tasks.size());
        std::vector<void*> bufs(tasks.size());
        std::vector<size_t> szs(tasks.size());
        for (size_t i = 0; i < tasks.size(); ++i) {
            keys[i] = tasks[i].tgt_key;
            bufs[i] = tgt_data[i].data();
            szs[i] = tgt_data[i].size();
        }
        int ret = batch_put_fn_(keys, bufs, szs);
        if (ret != 0) {
            LOG(ERROR) << "[C2C] batch put failed, ret=" << ret;
        }
    } else if (put_fn_) {
        for (size_t i = 0; i < tasks.size(); ++i) {
            put_fn_(tasks[i].tgt_key, tgt_data[i].data(), tgt_data[i].size());
        }
    }
    completed_keys_.fetch_add(tasks.size(), std::memory_order_relaxed);
    completed_pages_.fetch_add(total_pages, std::memory_order_relaxed);
    total_ms_.fetch_add(ms, std::memory_order_relaxed);
}

}  // namespace mooncake

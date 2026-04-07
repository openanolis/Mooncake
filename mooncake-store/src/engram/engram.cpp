#include "engram/engram.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mooncake {
namespace engram {

namespace {

bool TablesMatchLayout(
    const std::vector<std::shared_ptr<BufferHandle>>& tables,
    const std::vector<int64_t>& table_vocab_sizes, size_t row_bytes) {
    if (tables.size() != table_vocab_sizes.size()) {
        return false;
    }

    for (size_t head_idx = 0; head_idx < tables.size(); ++head_idx) {
        const size_t expected_table_bytes =
            static_cast<size_t>(table_vocab_sizes[head_idx]) * row_bytes;
        if (!tables[head_idx] ||
            tables[head_idx]->size() < expected_table_bytes) {
            return false;
        }
    }
    return true;
}

std::vector<QueryResult> UnwrapQueryResults(
    const std::vector<tl::expected<QueryResult, ErrorCode>>& query_results) {
    std::vector<QueryResult> resolved_query_results;
    resolved_query_results.reserve(query_results.size());
    for (const auto& result : query_results) {
        if (!result) {
            return {};
        }
        resolved_query_results.push_back(result.value());
    }
    return resolved_query_results;
}

}  // namespace

Engram::Engram(int layer_id, const EngramConfig& config,
               std::shared_ptr<PyClient> store)
    : store_(std::move(store)),
      table_vocab_sizes_(config.table_vocab_sizes),
      embedding_dim_(config.embedding_dim) {
    if (table_vocab_sizes_.empty()) {
        throw std::invalid_argument(
            "EngramConfig.table_vocab_sizes must not be empty");
    }
    if (embedding_dim_ <= 0) {
        throw std::invalid_argument(
            "EngramConfig.embedding_dim must be positive");
    }
    for (int64_t vocab_size : table_vocab_sizes_) {
        if (vocab_size <= 0) {
            throw std::invalid_argument(
                "EngramConfig.table_vocab_sizes must contain only positive "
                "values");
        }
    }

    embed_keys_.reserve(table_vocab_sizes_.size());
    for (size_t h = 0; h < table_vocab_sizes_.size(); ++h) {
        std::ostringstream oss;
        oss << "engram:l" << layer_id << ":h" << h;
        embed_keys_.push_back(oss.str());
    }
}

bool Engram::load_tables(std::vector<std::shared_ptr<BufferHandle>>& tables,
                         size_t row_bytes) const {
    if (store_ == nullptr) {
        return false;
    }

    auto fetch_with_query_results = [&](bool allow_cached) {
        bool used_cache = false;
        auto query_results = get_query_results(allow_cached, &used_cache);
        auto resolved_query_results = UnwrapQueryResults(query_results);
        if (resolved_query_results.empty()) {
            return false;
        }

        tables = store_->batch_get_buffer_with_query_results(
            embed_keys_, resolved_query_results);
        if (TablesMatchLayout(tables, table_vocab_sizes_, row_bytes)) {
            return true;
        }

        if (used_cache) {
            invalidate_query_result_cache();
        }
        return false;
    };

    if (fetch_with_query_results(true)) {
        return true;
    }

    if (fetch_with_query_results(false)) {
        return true;
    }

    tables = store_->batch_get_buffer(embed_keys_);
    return TablesMatchLayout(tables, table_vocab_sizes_, row_bytes);
}

std::vector<tl::expected<QueryResult, ErrorCode>> Engram::get_query_results(
    bool allow_cached, bool* used_cache) const {
    if (used_cache != nullptr) {
        *used_cache = false;
    }
    if (store_ == nullptr) {
        return {};
    }

    auto now = std::chrono::steady_clock::now();
    if (allow_cached) {
        std::lock_guard<std::mutex> lock(query_result_cache_mu_);
        if (query_result_cache_valid_ &&
            cached_query_results_.size() == embed_keys_.size()) {
            bool cache_still_valid = true;
            for (const auto& result : cached_query_results_) {
                if (result.IsLeaseExpired(now)) {
                    cache_still_valid = false;
                    break;
                }
            }
            if (cache_still_valid) {
                if (used_cache != nullptr) {
                    *used_cache = true;
                }
                std::vector<tl::expected<QueryResult, ErrorCode>>
                    query_results;
                query_results.reserve(cached_query_results_.size());
                for (const auto& result : cached_query_results_) {
                    query_results.emplace_back(result);
                }
                return query_results;
            }
        }
    }

    auto query_results = store_->batch_query(embed_keys_);
    {
        std::lock_guard<std::mutex> lock(query_result_cache_mu_);
        cached_query_results_.clear();
        query_result_cache_valid_ = false;

        bool cacheable = query_results.size() == embed_keys_.size();
        if (cacheable) {
            cached_query_results_.reserve(query_results.size());
            for (const auto& result : query_results) {
                if (!result) {
                    cacheable = false;
                    break;
                }
                cached_query_results_.push_back(result.value());
            }
        }

        if (cacheable) {
            query_result_cache_valid_ = true;
        } else {
            cached_query_results_.clear();
        }
    }
    return query_results;
}

void Engram::invalidate_query_result_cache() const {
    std::lock_guard<std::mutex> lock(query_result_cache_mu_);
    cached_query_results_.clear();
    query_result_cache_valid_ = false;
}

int Engram::lookup_rows_contiguous(const int64_t* row_ids, int B, int L,
                                   void* output_buffer,
                                   size_t output_size) const {
    if (store_ == nullptr || row_ids == nullptr || output_buffer == nullptr ||
        B <= 0 || L <= 0) {
        return -1;
    }

    const int num_heads = static_cast<int>(table_vocab_sizes_.size());
    const size_t row_bytes =
        static_cast<size_t>(embedding_dim_) * sizeof(float);
    const size_t expected_size =
        static_cast<size_t>(B) * L * num_heads * embedding_dim_ * sizeof(float);
    if (output_size < expected_size) {
        return -1;
    }

    auto fail_lookup = [&]() {
        std::memset(output_buffer, 0, expected_size);
        return -1;
    };

    std::vector<std::shared_ptr<BufferHandle>> tables;
    if (!load_tables(tables, row_bytes)) {
        return fail_lookup();
    }

    float* output = static_cast<float*>(output_buffer);
    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            const size_t token_index = static_cast<size_t>(b) * L + l;
            const size_t row_offset =
                token_index * static_cast<size_t>(num_heads);
            for (int h = 0; h < num_heads; ++h) {
                const int64_t idx =
                    row_ids[row_offset + static_cast<size_t>(h)];
                if (idx < 0 || idx >= table_vocab_sizes_[h]) {
                    return fail_lookup();
                }

                const float* table =
                    static_cast<const float*>(tables[h]->ptr());
                const float* src =
                    table + static_cast<size_t>(idx) * embedding_dim_;
                float* dst = output + (row_offset + static_cast<size_t>(h)) *
                                          embedding_dim_;
                std::memcpy(dst, src, row_bytes);
            }
        }
    }

    return 0;
}

int Engram::lookup_rows(
    const std::vector<std::vector<std::vector<int64_t>>>& row_ids,
    void* output_buffer, size_t output_size) const {
    if (store_ == nullptr || output_buffer == nullptr || row_ids.empty() ||
        row_ids[0].empty()) {
        return -1;
    }

    const int B = static_cast<int>(row_ids.size());
    const int L = static_cast<int>(row_ids[0].size());
    const int num_heads = static_cast<int>(table_vocab_sizes_.size());
    const size_t row_bytes =
        static_cast<size_t>(embedding_dim_) * sizeof(float);
    const size_t expected_size =
        static_cast<size_t>(B) * L * num_heads * embedding_dim_ * sizeof(float);
    if (output_size < expected_size) {
        return -1;
    }

    auto fail_lookup = [&]() {
        std::memset(output_buffer, 0, expected_size);
        return -1;
    };

    std::vector<std::shared_ptr<BufferHandle>> tables;
    if (!load_tables(tables, row_bytes)) {
        return fail_lookup();
    }

    float* output = static_cast<float*>(output_buffer);
    for (int b = 0; b < B; ++b) {
        if (static_cast<int>(row_ids[b].size()) != L) {
            return fail_lookup();
        }
        for (int l = 0; l < L; ++l) {
            if (static_cast<int>(row_ids[b][l].size()) != num_heads) {
                return fail_lookup();
            }

            const size_t token_index = static_cast<size_t>(b) * L + l;
            const size_t row_offset =
                token_index * static_cast<size_t>(num_heads);
            for (int h = 0; h < num_heads; ++h) {
                const int64_t idx = row_ids[b][l][h];
                if (idx < 0 || idx >= table_vocab_sizes_[h]) {
                    return fail_lookup();
                }

                const float* table =
                    static_cast<const float*>(tables[h]->ptr());
                const float* src =
                    table + static_cast<size_t>(idx) * embedding_dim_;
                float* dst = output + (row_offset + static_cast<size_t>(h)) *
                                          embedding_dim_;
                std::memcpy(dst, src, row_bytes);
            }
        }
    }

    return 0;
}

std::vector<int64_t> Engram::get_table_vocab_sizes() const {
    return table_vocab_sizes_;
}

std::vector<std::string> Engram::get_store_keys() const { return embed_keys_; }

int Engram::get_num_heads() const {
    return static_cast<int>(table_vocab_sizes_.size());
}

int Engram::get_embedding_dim() const { return embedding_dim_; }

int Engram::remove_from_store(bool force) {
    if (store_ == nullptr) {
        return static_cast<int>(ErrorCode::INVALID_PARAMS);
    }

    constexpr int kObjectNotFound =
        static_cast<int>(ErrorCode::OBJECT_NOT_FOUND);
    int removed = 0;
    int first_error = 0;

    for (const auto& key : embed_keys_) {
        int rc = store_->remove(key, force);
        if (rc == 0) {
            ++removed;
            continue;
        }
        if (rc == kObjectNotFound) {
            continue;
        }
        if (first_error == 0) {
            first_error = rc;
        }
    }

    invalidate_query_result_cache();
    return first_error != 0 ? first_error : removed;
}

int Engram::populate(const std::vector<void*>& embedding_buffers,
                     const std::vector<size_t>& buffer_sizes) {
    if (store_ == nullptr) {
        return -1;
    }
    if (embedding_buffers.size() != embed_keys_.size() ||
        buffer_sizes.size() != embed_keys_.size()) {
        return -1;
    }

    for (size_t i = 0; i < buffer_sizes.size(); ++i) {
        const size_t expected = static_cast<size_t>(table_vocab_sizes_[i]) *
                                embedding_dim_ * sizeof(float);
        if (embedding_buffers[i] == nullptr || buffer_sizes[i] != expected) {
            return -1;
        }
    }

    std::vector<int> exists_results = store_->batchIsExist(embed_keys_);
    if (exists_results.size() != embed_keys_.size()) {
        LOG(ERROR) << "Failed to preflight Engram populate key existence";
        return -1;
    }
    for (size_t i = 0; i < exists_results.size(); ++i) {
        const int exists = exists_results[i];
        if (exists < 0) {
            LOG(ERROR) << "Failed to query Engram key '" << embed_keys_[i]
                       << "' before populate, rc=" << exists;
            return -1;
        }
        if (exists != 0) {
            LOG(ERROR) << "Engram populate requires empty destination key '"
                       << embed_keys_[i]
                       << "'. Remove the existing layer first.";
            return -1;
        }
    }

    auto cleanup_registered_buffers = [&](size_t count) {
        bool cleanup_failed = false;
        for (size_t i = 0; i < count; ++i) {
            int rc = store_->unregister_buffer(embedding_buffers[i]);
            if (rc != 0) {
                cleanup_failed = true;
                LOG(ERROR) << "Failed to unregister embedding buffer at index "
                           << i << ", rc=" << rc;
            }
        }
        return cleanup_failed;
    };

    for (size_t i = 0; i < embedding_buffers.size(); ++i) {
        int ret =
            store_->register_buffer(embedding_buffers[i], buffer_sizes[i]);
        if (ret != 0) {
            if (cleanup_registered_buffers(i)) {
                LOG(ERROR) << "Failed to clean up registered embedding buffers "
                              "after register_buffer error";
            }
            return -1;
        }
    }

    std::vector<int> put_results =
        store_->batch_put_from(embed_keys_, embedding_buffers, buffer_sizes);
    const bool put_succeeded =
        put_results.size() == embed_keys_.size() &&
        std::all_of(put_results.begin(), put_results.end(),
                    [](int result) { return result == 0; });

    const bool unregister_failed =
        cleanup_registered_buffers(embedding_buffers.size());

    if (!put_succeeded || unregister_failed) {
        const bool put_results_complete =
            put_results.size() == embed_keys_.size();
        for (size_t i = 0; i < embed_keys_.size(); ++i) {
            if (put_results_complete && put_results[i] != 0) {
                continue;
            }
            int rc = store_->remove(embed_keys_[i], true);
            if (rc != 0 &&
                rc != static_cast<int>(ErrorCode::OBJECT_NOT_FOUND)) {
                LOG(ERROR) << "Failed to roll back partially populated Engram "
                           << "key '" << embed_keys_[i] << "', rc=" << rc;
            }
        }
        if (unregister_failed) {
            LOG(ERROR) << "Rolling back Engram populate because buffer cleanup "
                          "failed after publish";
        }
        return -1;
    }

    invalidate_query_result_cache();
    return 0;
}

}  // namespace engram
}  // namespace mooncake

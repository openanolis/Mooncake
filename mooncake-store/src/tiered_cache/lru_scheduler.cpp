#include "tiered_cache/lru_scheduler.h"
#include <algorithm>
#include <limits>

namespace mooncake {

#define MOONCAKE_DEBUG
#ifdef MOONCAKE_DEBUG
void LruCacheScheduler::PrintLruList() {
    LOG(INFO) << "LRU Scheduler topology updated. Current tier LRU lists (ordered by placement priority):";

    uint64_t current_tier = placement_tier_id_;
    int tier_index = 0;

    while (current_tier != 0) {
        auto lru_it = tier_lru_lists_.find(current_tier);
        if (lru_it != tier_lru_lists_.end()) {
            const auto& lru_list = lru_it->second;
            std::string list_info = "Tier[" + std::to_string(tier_index) + "] ID=" +
                                   std::to_string(current_tier) +
                                   " (size=" + std::to_string(lru_list.size()) + "): ";

            if (!lru_list.empty()) {
                list_info += "LRU head='" + lru_list.front() + "'";

                if (lru_list.size() > 1) {
                    list_info += ", ... , MRU tail='" + lru_list.back() + "'";
                }
            } else {
                list_info += "empty";
            }

            LOG(INFO) << list_info;
        } else {
            LOG(INFO) << "Tier[" << tier_index << "] ID=" << current_tier << " (not found in LRU lists)";
        }

        uint64_t next_tier = GetNextTierId(current_tier);

        if (next_tier == current_tier) {
            LOG(ERROR) << "Cycle detected in tier chain at tier " << current_tier;
            break;
        }

        current_tier = next_tier;
        tier_index++;
    }

    if (tier_index == 0) {
        LOG(INFO) << "No valid tiers configured";
    }
}
#else
void LruCacheScheduler::PrintLruList() {}
#endif

uint64_t LruCacheScheduler::GetNextTierId(uint64_t tier_id) {
    auto it = next_tier_map_.find(tier_id);
    if (it != next_tier_map_.end()) {
        return it->second;
    }
    return 0;
}

void LruCacheScheduler::InitTopology(const std::vector<TierView>& tier_views) {
    tier_lru_lists_.clear();
    tier_views_lists_.clear();
    next_tier_map_.clear();

    if (tier_views.empty())
        return;

    auto sorted_views = tier_views;

    std::sort(sorted_views.begin(), sorted_views.end(), [](const TierView& a, const TierView& b) {
        return a.priority < b.priority;
    });

    for (size_t i = 0; i < sorted_views.size(); ++i) {
        tier_lru_lists_[sorted_views[i].id];
        tier_views_lists_[sorted_views[i].id] = sorted_views[i];
        if (i < sorted_views.size() - 1)
            next_tier_map_[sorted_views[i].id] = sorted_views[i+1].id;
    }
    placement_tier_id_ = sorted_views[0].id;
    PrintLruList();
}

void LruCacheScheduler::UpdateTopology(const std::vector<TierView>& tier_views) {
    if (tier_views.empty()) {
        return;
    }

    tier_views_lists_.clear();
    next_tier_map_.clear();

    auto sorted_views = tier_views;

    std::sort(sorted_views.begin(), sorted_views.end(), [](const TierView& a, const TierView& b) {
        return a.priority < b.priority;
    });

    for (size_t i = 0; i < sorted_views.size(); ++i) {
        tier_views_lists_[sorted_views[i].id] = sorted_views[i];
        if (i < sorted_views.size() - 1)
            next_tier_map_[sorted_views[i].id] = sorted_views[i+1].id;
    }
    placement_tier_id_ = sorted_views[0].id;
    PrintLruList();
}

uint64_t LruCacheScheduler::GetPlacement(const std::string& key, const SchedulingContext& ctx) {
    // Always place in the first tier.
    return placement_tier_id_;
}

MovementPlan LruCacheScheduler::EvictKey(uint64_t tier_id_to_free_up) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tier_lru_lists_.find(tier_id_to_free_up);
    if (it == tier_lru_lists_.end() || it->second.empty()) {
        return MovementPlan{"", 0, 0};
    }

    // The eviction is at the front of the list (least recently used).
    const std::string& coldest_key = it->second.front();

    // Just evict to the next tier (TODO)
    auto next_tier_id = GetNextTierId(tier_id_to_free_up);

    if (next_tier_id == 0)
        return MovementPlan{coldest_key, tier_id_to_free_up, MovementPlan::DISCARD};

    // No space in the next tier
    auto tier_view = tier_views_lists_[next_tier_id];
    if ((tier_view.capacity - tier_view.usage) < key_sizes_[coldest_key]) {
        return MovementPlan{coldest_key, tier_id_to_free_up, MovementPlan::DISCARD};
    }

    return MovementPlan{coldest_key, tier_id_to_free_up, next_tier_id};
}

MovementPlan LruCacheScheduler::PromoteKey(uint64_t tier_id_to_promote_in) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tier_lru_lists_.find(tier_id_to_promote_in);
    if (it == tier_lru_lists_.end()) {
        return MovementPlan{"", 0, 0};
    }

    // Just find the hotest key in the next tier now. (TODO)
    auto next_tier_id = GetNextTierId(tier_id_to_promote_in);

    it = tier_lru_lists_.find(next_tier_id);
    if (it == tier_lru_lists_.end() || it->second.empty()) {
        return MovementPlan{"", 0, 0};
    }

    const std::string& hotest_key = it->second.back();
    return MovementPlan{hotest_key, next_tier_id, tier_id_to_promote_in};
}

std::vector<MovementPlan> LruCacheScheduler::EvictBytes(uint64_t tier_id_to_free_up, size_t bytes_to_free) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MovementPlan> plans;
    size_t freed_bytes = 0;

    auto lru_it = tier_lru_lists_.find(tier_id_to_free_up);
    if (lru_it == tier_lru_lists_.end()) {
        return plans;
    }

    auto& lru_list = lru_it->second;
    auto it = lru_list.begin();

    while (freed_bytes < bytes_to_free && it != lru_list.end()) {
        const std::string& key_to_evict = *it;

        auto key_size_it = key_sizes_.find(key_to_evict);
        if (key_size_it == key_sizes_.end()) {
            // Should not happen if data is consistent
            ++it;
            continue;
        }

        uint64_t next_tier_id = GetNextTierId(tier_id_to_free_up);
        uint64_t destination_tier = MovementPlan::DISCARD;

        if (next_tier_id != 0) {
            auto tier_view_it = tier_views_lists_.find(next_tier_id);
            if (tier_view_it != tier_views_lists_.end()) {
                if ((tier_view_it->second.capacity - tier_view_it->second.usage) >= key_size_it->second) {
                    destination_tier = next_tier_id;
                }
            }
        }

        plans.push_back({key_to_evict, tier_id_to_free_up, destination_tier});
        freed_bytes += key_size_it->second;

        ++it;
    }

    return plans;
}

void LruCacheScheduler::OnAccess(const std::string& key, uint64_t tier_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    Touch(key, tier_id);
    PrintLruList();
}

void LruCacheScheduler::OnPut(const std::string& key, uint64_t tier_id, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto lru_it = tier_lru_lists_.find(tier_id);
    if (lru_it == tier_lru_lists_.end()) {
        return;
    }

    // If the key already exists (e.g., an overwrite), remove its old position first.
    if (key_iterators_.count(key)) {
        lru_it->second.erase(key_iterators_[key]);
    }

    // Add the new key to the back of the list (most recently used).
    lru_it->second.push_back(key);
    key_iterators_[key] = std::prev(lru_it->second.end());
    key_sizes_[key] = size;
    PrintLruList();
}

void LruCacheScheduler::OnDelete(const std::string& key, uint64_t tier_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto lru_it = tier_lru_lists_.find(tier_id);
    if (lru_it == tier_lru_lists_.end()) {
        return;
    }

    auto key_it = key_iterators_.find(key);
    if (key_it != key_iterators_.end()) {
        lru_it->second.erase(key_it->second);
        key_iterators_.erase(key_it);
        key_sizes_.erase(key);
    }
    PrintLruList();
}

void LruCacheScheduler::Touch(const std::string& key, uint64_t tier_id) {
    auto lru_it = tier_lru_lists_.find(tier_id);
    if (lru_it == tier_lru_lists_.end()) {
        return;
    }

    auto key_it = key_iterators_.find(key);
    if (key_it == key_iterators_.end()) {
        return;
    }

    // Splice the node from its current position to the end of the list.
    lru_it->second.splice(lru_it->second.end(), lru_it->second, key_it->second);
}

} // namespace mooncake
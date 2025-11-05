#pragma once

#include "tiered_cache/cache_scheduler.h"
#include <list>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace mooncake {

/**
 * @class LruCacheScheduler
 * @brief A simple LRU scheduler
 *
 * This scheduler implements a basic Least Recently Used (LRU) policy for eviction
 * and a simple placement strategy.
 * - Placement: Always places new keys in the tier with the highest priority.
 * - Eviction: When a tier is full, it plans to evict the least recently used key.
 * - Eviction: (TODO) Add more eviction policies (e.g. Batch Eviction).
 * - Promotion: When a tier is free, it can promote a hotest key from the next tier.
 * - Promotion: (TODO) Add more promotion policies (e.g. Batch Promotion).
 */
class LruCacheScheduler : public CacheScheduler {
public:
    LruCacheScheduler() = default;
    ~LruCacheScheduler() override = default;

    void InitTopology(const std::vector<TierView>& tier_views) override;

    void UpdateTopology(const std::vector<TierView>& tier_views) override;

    uint64_t GetPlacement(const std::string& key, const SchedulingContext& ctx) override;

    MovementPlan EvictKey(uint64_t tier_id_to_free_up) override;

    std::vector<MovementPlan> EvictBytes(uint64_t tier_id_to_free_up, size_t bytes_to_free) override;

    MovementPlan PromoteKey(uint64_t tier_id_to_promote_in) override;

    void OnAccess(const std::string& key, uint64_t tier_id) override;

    void OnPut(const std::string& key, uint64_t tier_id, size_t size) override;

    void OnDelete(const std::string& key, uint64_t tier_id) override;

private:
    // A helper function to move a key to the back of the LRU list.
    void Touch(const std::string& key, uint64_t tier_id);

    // Print the LRU lists for debugging purposes.
    void PrintLruList();

    // Get the next tier ID after the given tier ID.
    uint64_t GetNextTierId(uint64_t tier_id);

    // A map from a tier ID to its LRU list of keys.
    // The front of the list is the least recently used (victim).
    // The back of the list is the most recently used.
    std::unordered_map<uint64_t, std::list<std::string>> tier_lru_lists_;

    // Get the TierView for a given tier ID.
    std::unordered_map<uint64_t, TierView> tier_views_lists_;

    // tier_id -> next_tier_id
    std::unordered_map<uint64_t, uint64_t> next_tier_map_;

    // To quickly find a key's iterator in the list for efficient updates.
    std::unordered_map<std::string, std::list<std::string>::iterator> key_iterators_;

    // Get the cache size of a key
    std::unordered_map<std::string, size_t> key_sizes_;

    // The tier with the highest priority, used for placement.
    uint64_t placement_tier_id_ = -1;

    // Mutex to protect internal data structures from concurrent access.
    std::mutex mutex_;
};

} // namespace mooncake
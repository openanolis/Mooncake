#pragma once

#include "tiered_cache/tiered_backend.h"
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace mooncake {

/**
 * @struct SchedulingContext
 * @brief Contains contextual information for a scheduling decision.
 *
 * This can be extended to include details like the NUMA node of the request,
 * the priority of the request, or other metadata that can help the scheduler
 * make a more informed decision.
 */
struct SchedulingContext {};

/**
 * @struct MovementPlan
 * @brief Represents a plan for data movement, as decided by the scheduler.
 *
 * This struct is the output of the eviction planning process and serves as a
 * concrete instruction for the Worker to execute using the TieredBackend.
 */
struct MovementPlan {
    // The key of the cache entry that should be moved or discarded.
    std::string key_to_move;

    // The ID of the source tier from which the key should be moved.
    uint64_t from_tier_id;

    // The ID of the destination tier.
    uint64_t to_tier_id;

    // A special value for to_tier_id, indicating that the key should be
    // discarded entirely, not moved to another tier.
    static constexpr uint64_t DISCARD = -1;
};

/**
 * @class CacheScheduler
 * @brief Abstract base class for all cache scheduling policies.
 *
 * This class defines the interface for making all policy decisions related to
 * data placement, eviction, and promotion. It is designed to be a stateful
 * component that tracks the usage of cache entries and makes decisions based
 * on a specific algorithm (e.g., LRU, LFU, etc.).
 */
class CacheScheduler {
public:
    virtual ~CacheScheduler() = default;

    /**
     * @brief Informs the scheduler about the system's tier topology and status.
     * This method is typically called by the Worker during initialization or
     * whenever the topology needs to be updated.
     * @param tier_views A vector of TierView structs describing all available tiers.
     */
    virtual void InitTopology(const std::vector<TierView>& tier_views) = 0;

    /**
     * @brief Updates the scheduler's internal state when the topology changes.
     * This method is called by the Worker whenever the topology needs to be updated.
     * @param tier_views A vector of TierView structs describing all available tiers.
     */
    virtual void UpdateTopology(const std::vector<TierView>& tier_views) = 0;

    /**
     * @brief Decides the best tier to initially place a new key in.
     * @param key The key being placed.
     * @param ctx The context of the request (e.g., NUMA node, priority).
     * @return The ID of the target tier for the initial placement.
     */
    virtual uint64_t GetPlacement(const std::string& key, const SchedulingContext& ctx) = 0;

    /**
     * @brief Creates a plan to evict a key out of a specific tier.
     * @param tier_id_to_free_up The ID of the tier that requires space.
     * @return An MovementPlan describing which key to move and where to move it
     * (or if it should be discarded). Returns an empty plan if no
     * candidate can be found.
     */
    virtual MovementPlan EvictKey(uint64_t tier_id_to_free_up) = 0;

    /**
     * @brief Creates a plan to evict a specific number of bytes out of a specific tier.
     * @param tier_id_to_free_up The ID of the tier that requires space.
     * @param bytes_to_free The number of bytes to be freed.
     * @return A vector of MovementPlan structs describing which keys to move and where to move them
     * (or if they should be discarded). Returns an empty vector if no
     * candidates can be found.
     */
    virtual std::vector<MovementPlan> EvictBytes(uint64_t tier_id_to_free_up, size_t bytes_to_free) = 0;

    /**
     * @brief Creates a plan to promote a key into a specific tier.
     * @param tier_id_to_promote_in The ID of the tier that is needed to add key.
     * @return An MovementPlan describing which key to move and where to move it
     * (or if it should be discarded). Returns an empty plan if no
     * candidate can be found.
     */
    virtual MovementPlan PromoteKey(uint64_t tier_id_to_promote_in) = 0;

    /**
     * @brief Notifies the scheduler that a key has been accessed (a cache hit).
     * This is crucial for usage-based policies like LRU or LFU to update their
     * internal state.
     * @param key The key that was accessed.
     * @param tier_id The ID of the tier where the key resides.
     */
    virtual void OnAccess(const std::string& key, uint64_t tier_id) = 0;

    /**
     * @brief Notifies the scheduler that a new key has been successfully placed in a tier.
     * @param key The key that was placed.
     * @param tier_id The ID of the tier where the key was placed.
     */
    virtual void OnPut(const std::string& key, uint64_t tier_id, size_t size) = 0;

    /**
     * @brief Notifies the scheduler that a key has been deleted from a tier.
     * This allows the scheduler to clean up its internal tracking data for the key.
     * @param key The key that was deleted.
     * @param tier_id The ID of the tier from which the key was deleted.
     */
    virtual void OnDelete(const std::string& key, uint64_t tier_id) = 0;
};

} // namespace mooncake
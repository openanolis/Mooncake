#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <msgpack.hpp>
#include <ylt/util/tl/expected.hpp>

#include "replica.h"
#include "types.h"

namespace mooncake {

/**
 * @brief Metadata structure for Standby to store and restore object information
 *
 * This structure contains all essential metadata information needed by Standby
 * to immediately serve as Primary when promoted.
 */
struct StandbyObjectMetadata {
    UUID client_id{0, 0};
    uint64_t size{0};
    std::string tenant_id{"default"};
    std::string domain_id{"default"};
    std::string object_set{"default"};
    std::string sharing_scope{};
    std::string qos_tier{"default"};
    std::string logical_key{};
    std::string canonical_key{};
    std::string legacy_raw_key{};
    std::vector<Replica::Descriptor> replicas;
    // NOTE: Lease information is NOT stored because:
    // 1. Standby does not perform eviction, so lease info is not used
    // 2. After promotion, new Primary should grant fresh leases, not restore
    // old ones
    uint64_t last_sequence_id{
        0};  // Last OpLog sequence ID that modified this key

    StandbyObjectMetadata() = default;

    // Check if this metadata has valid replicas
    bool HasReplicas() const { return !replicas.empty(); }
};

/**
 * @brief Payload structure for struct_pack serialization (msgpack binary
 * format)
 *
 * Now uses UUID directly since struct_pack natively supports std::pair.
 */
struct MetadataPayload {
    UUID client_id{0, 0};
    uint64_t size{0};
    std::string tenant_id{"default"};
    std::string domain_id{"default"};
    std::string object_set{"default"};
    std::string sharing_scope{};
    std::string qos_tier{"default"};
    std::string logical_key{};
    std::string canonical_key{};
    std::vector<Replica::Descriptor> replicas;
    // NOTE: Lease information removed - not needed by Standby

    YLT_REFL(MetadataPayload, client_id, size, tenant_id, domain_id, object_set,
             sharing_scope, qos_tier, logical_key, canonical_key, replicas);

    // Convert to StandbyObjectMetadata
    StandbyObjectMetadata ToStandbyMetadata(
        uint64_t sequence_id,
        const std::string& legacy_raw_key = std::string()) const {
        StandbyObjectMetadata meta;
        meta.client_id = client_id;
        meta.size = size;
        meta.tenant_id = tenant_id;
        meta.domain_id = domain_id;
        meta.object_set = object_set;
        meta.sharing_scope = sharing_scope;
        meta.qos_tier = qos_tier;
        meta.logical_key = logical_key;
        meta.canonical_key = canonical_key;
        meta.legacy_raw_key = legacy_raw_key;
        meta.replicas = replicas;
        meta.last_sequence_id = sequence_id;
        return meta;
    }
};

// Shared parser for snapshot metadata entries used by both master restore and
// standby snapshot bootstrap.
struct SnapshotMetadataFields {
    UUID client_id{0, 0};
    uint64_t put_start_time_ms{0};
    uint64_t size{0};
    uint64_t lease_timeout_ms{0};
    bool has_soft_pin_timeout{false};
    uint64_t soft_pin_timeout_ms{0};
    std::string tenant_id{"default"};
    std::string domain_id{"default"};
    std::string object_set{"default"};
    std::string sharing_scope{};
    std::string qos_tier{"default"};
    std::string logical_key{};
    std::string canonical_key{};
    uint32_t replica_count{0};
    uint32_t replica_index{0};
    bool hard_pinned{false};
};

tl::expected<SnapshotMetadataFields, SerializationError>
ParseSnapshotMetadataFields(const msgpack::object& object);

/**
 * @brief Abstract interface for metadata storage on Standby
 *
 * This interface provides basic operations for storing and managing object
 * metadata. In a full implementation, this would mirror MasterService's
 * metadata_shards_ structure.
 */
class MetadataStore {
   public:
    virtual ~MetadataStore() = default;

    /**
     * @brief Put or update metadata for a key with structured metadata
     * @param key Object key
     * @param metadata Structured metadata object
     * @return true on success, false on failure
     */
    virtual bool PutMetadata(const std::string& key,
                             const StandbyObjectMetadata& metadata) = 0;

    virtual bool PutMetadata(const LogicalObjectId& object_id,
                             const StandbyObjectMetadata& metadata) = 0;

    /**
     * @brief Put or update metadata for a key (legacy interface for backward
     * compatibility)
     * @param key Object key
     * @param payload Optional payload data (JSON serialized metadata)
     * @return true on success, false on failure
     */
    virtual bool Put(const std::string& key,
                     const std::string& payload = std::string()) = 0;

    /**
     * @brief Get metadata for a key
     * @param key Object key
     * @return Copy of metadata if found, std::nullopt otherwise
     */
    virtual std::optional<StandbyObjectMetadata> GetMetadata(
        const std::string& key) const = 0;

    /**
     * @brief Remove metadata for a key
     * @param key Object key
     * @return true if key was found and removed, false otherwise
     */
    virtual bool Remove(const std::string& key) = 0;

    /**
     * @brief Check if a key exists
     * @param key Object key
     * @return true if key exists, false otherwise
     */
    virtual bool Exists(const std::string& key) const = 0;

    /**
     * @brief Get the count of keys in the store
     * @return Number of keys
     */
    virtual size_t GetKeyCount() const = 0;
};

}  // namespace mooncake

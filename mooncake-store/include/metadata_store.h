#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

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
    uint32_t schema_version{1};
    UUID client_id{0, 0};
    uint64_t size{0};
    std::string logical_key;
    std::string canonical_key;
    std::string tenant_id{"default"};
    std::string domain_id{"default"};
    std::optional<std::string> version;
    std::optional<std::string> sharing_scope;
    ObjectDataType data_type{ObjectDataType::UNKNOWN};
    QoSTier qos_tier{QoSTier::SHARED};
    AccessMode access_mode{AccessMode::NORMAL};
    std::unordered_map<std::string, std::string> type_hints;
    RetentionSpec retention{};
    PinSpec pin{};
    std::string group_path;
    uint64_t generation{0};
    int64_t created_at_ms{0};
    int64_t last_accessed_at_ms{0};
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
    uint32_t schema_version{1};
    UUID client_id{0, 0};
    uint64_t size{0};
    std::string logical_key;
    std::string canonical_key;
    std::string tenant_id{"default"};
    std::string domain_id{"default"};
    std::optional<std::string> version;
    std::optional<std::string> sharing_scope;
    ObjectDataType data_type{ObjectDataType::UNKNOWN};
    QoSTier qos_tier{QoSTier::SHARED};
    AccessMode access_mode{AccessMode::NORMAL};
    std::unordered_map<std::string, std::string> type_hints;
    RetentionSpec retention{};
    PinSpec pin{};
    std::string group_path;
    uint64_t generation{0};
    int64_t created_at_ms{0};
    int64_t last_accessed_at_ms{0};
    std::vector<Replica::Descriptor> replicas;
    // NOTE: Lease information removed - not needed by Standby

    YLT_REFL(MetadataPayload, schema_version, client_id, size, logical_key,
             canonical_key, tenant_id, domain_id, version, sharing_scope,
             data_type, qos_tier, access_mode, type_hints, retention, pin,
             group_path, generation, created_at_ms, last_accessed_at_ms,
             replicas);

    // Convert to StandbyObjectMetadata
    StandbyObjectMetadata ToStandbyMetadata(uint64_t sequence_id) const {
        StandbyObjectMetadata meta;
        meta.schema_version = schema_version;
        meta.client_id = client_id;
        meta.size = size;
        meta.logical_key = logical_key;
        meta.canonical_key = canonical_key;
        meta.tenant_id = tenant_id;
        meta.domain_id = domain_id;
        meta.version = version;
        meta.sharing_scope = sharing_scope;
        meta.data_type = data_type;
        meta.qos_tier = qos_tier;
        meta.access_mode = access_mode;
        meta.type_hints = type_hints;
        meta.retention = retention;
        meta.pin = pin;
        meta.group_path = group_path;
        meta.generation = generation;
        meta.created_at_ms = created_at_ms;
        meta.last_accessed_at_ms = last_accessed_at_ms;
        meta.replicas = replicas;
        meta.last_sequence_id = sequence_id;
        return meta;
    }
};

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

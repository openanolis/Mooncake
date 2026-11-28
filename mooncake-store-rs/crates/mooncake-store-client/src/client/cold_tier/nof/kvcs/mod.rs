//! KVCS Low-Level C API binding for the existing Cold Tier/NoF data path.
//!
//! Mooncake remains authoritative for object metadata. KVCS is one opaque
//! target that stores physical records and reports target capacity.

use mooncake_store_core::{Result, StoreError};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::client::cold_tier::nof::{NofBackend, NofTargetConfig, OpaquePhysicalKey};

include!("capi.rs");

mod executor;
pub use executor::{KvcsCapiExecutor, KvcsLowLevelClient, KvcsMode};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KvcsBackendConfig {
    pub storage_domain_id: String,
    #[serde(default = "default_mode")]
    pub mode: KvcsMode,
    #[serde(default = "default_efc_socket")]
    pub efc_socket: String,
    #[serde(default)]
    pub mountpoint_index: u32,
    #[serde(default = "default_max_value_size")]
    pub max_value_size: u64,
}

impl KvcsBackendConfig {
    pub fn validate(&self) -> Result<()> {
        if self.storage_domain_id.trim().is_empty()
            || self.storage_domain_id.trim() != self.storage_domain_id
        {
            return Err(StoreError::InvalidState(
                "KVCS storage_domain_id must be non-empty without surrounding whitespace"
                    .to_string(),
            ));
        }
        if self.efc_socket.trim().is_empty() {
            return Err(StoreError::InvalidState(
                "KVCS efc_socket must not be empty".to_string(),
            ));
        }
        executor::validate_max_value_size(self.max_value_size)
    }
}

fn default_mode() -> KvcsMode {
    KvcsMode::LowLevel
}

fn default_efc_socket() -> String {
    "/var/run/kvcs/efc-grpc.sock".to_string()
}

fn default_max_value_size() -> u64 {
    SDK_DEFAULT_MAX_VALUE_SIZE
}

const KVCS_MODE_ENV: &str = "MOONCAKE_KVCS_MODE";
const KVCS_EFC_SOCKET_ENV: &str = "MOONCAKE_KVCS_EFC_SOCKET";
const KVCS_MAX_VALUE_SIZE_ENV: &str = "MOONCAKE_KVCS_MAX_VALUE_SIZE";
const KVCS_MOUNTPOINT_INDEX_ENV: &str = "MOONCAKE_KVCS_MOUNTPOINT_INDEX";
const KVCS_NAMESPACE_ENV: &str = "MOONCAKE_KVCS_NAMESPACE";

// Keep the existing Low-Level tuning names and defaults. The provider-query
// timeout is intentionally gone because Mooncake no longer queries metadata.
const KVCS_MAX_KEY_SIZE_ENV: &str = "MOONCAKE_KVCS_MAX_KEY_SIZE";
const KVCS_MAX_KEYS_PER_BATCH_ENV: &str = "MOONCAKE_KVCS_MAX_KEYS_PER_BATCH";
const KVCS_GET_WORKERS_ENV: &str = "MOONCAKE_KVCS_GET_WORKERS";
const KVCS_SET_WORKERS_ENV: &str = "MOONCAKE_KVCS_SET_WORKERS";
const KVCS_SIMPLE_WORKERS_ENV: &str = "MOONCAKE_KVCS_SIMPLE_WORKERS";
const KVCS_OPERATION_TIMEOUT_MS_ENV: &str = "MOONCAKE_KVCS_OPERATION_TIMEOUT_MS";

type EnvLookup = dyn Fn(&str) -> Result<Option<String>>;

/// Reads the KVCS target from the deployment environment.
///
/// An unset `MOONCAKE_KVCS_MODE` returns an empty list, so a client that does not enable KVCS
/// resolves exactly the Cold Tier configuration it had before.
pub fn configurations_from_env() -> Result<Vec<KvcsBackendConfig>> {
    let Some(mode) = env_value(KVCS_MODE_ENV)? else {
        return Ok(Vec::new());
    };
    let mode = KvcsMode::parse(mode.trim())?;
    let config = KvcsBackendConfig {
        storage_domain_id: env_value(KVCS_NAMESPACE_ENV)?
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "mooncake".to_string()),
        mode,
        efc_socket: env_value(KVCS_EFC_SOCKET_ENV)?
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(default_efc_socket),
        mountpoint_index: parse_u32(
            KVCS_MOUNTPOINT_INDEX_ENV,
            env_value(KVCS_MOUNTPOINT_INDEX_ENV)?,
            0,
        )?,
        max_value_size: parse_max_value_size(env_value(KVCS_MAX_VALUE_SIZE_ENV)?)?,
    };
    config.validate()?;
    Ok(vec![config])
}

/// Builds the single KVCS physical target described by `configs`.
///
/// The adapter accepts one storage domain: Mooncake keeps logical placement in `ObjectRoute`,
/// and the configured provider target is the smallest placement unit.
pub fn build_targets(configs: &[KvcsBackendConfig]) -> Result<Vec<NofTargetConfig>> {
    if configs.len() > 1 {
        return Err(StoreError::InvalidState(
            "KVCS integration accepts exactly one target".to_string(),
        ));
    }
    configs
        .iter()
        .map(|config| {
            config.validate()?;
            let client = KvcsLowLevelClient::new(config.efc_socket.clone(), config.max_value_size)?;
            let backend = NofBackend::new(Arc::new(client.executor(config.mountpoint_index)))?;
            NofTargetConfig::new(format!("kvcs:{}", config.storage_domain_id), backend)
        })
        .collect()
}

fn parse_max_value_size(value: Option<String>) -> Result<u64> {
    let value = match value {
        Some(value) if !value.trim().is_empty() => value.trim().parse::<u64>().map_err(|_| {
            StoreError::InvalidState(format!(
                "{KVCS_MAX_VALUE_SIZE_ENV} must be an integer byte count"
            ))
        })?,
        Some(_) => {
            return Err(StoreError::InvalidState(format!(
                "{KVCS_MAX_VALUE_SIZE_ENV} must not be empty"
            )))
        }
        None => default_max_value_size(),
    };
    executor::validate_max_value_size(value)?;
    Ok(value)
}

fn parse_u32(name: &str, value: Option<String>, default: u32) -> Result<u32> {
    match value {
        Some(value) if !value.trim().is_empty() => value.trim().parse().map_err(|_| {
            StoreError::InvalidState(format!("{name} must be a non-negative integer"))
        }),
        Some(_) | None => Ok(default),
    }
}

fn env_value(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(StoreError::InvalidState(format!(
            "{name} is not valid UTF-8"
        ))),
    }
}

#[cfg(test)]
mod capi_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_low_level_target_is_supported() {
        assert_eq!(KvcsMode::parse("low-level").unwrap(), KvcsMode::LowLevel);
        assert!(KvcsMode::parse("standard").is_err());
    }

    #[test]
    fn defaults_preserve_the_existing_low_level_configuration() {
        let config: KvcsBackendConfig = serde_json::from_value(serde_json::json!({
            "storage_domain_id": "mooncake"
        }))
        .unwrap();
        assert_eq!(config.efc_socket, "/var/run/kvcs/efc-grpc.sock");
        assert_eq!(config.mountpoint_index, 0);
        assert_eq!(config.max_value_size, SDK_DEFAULT_MAX_VALUE_SIZE);
        config.validate().unwrap();
    }

    #[test]
    fn max_value_size_validation_is_shared_with_the_executor() {
        assert_eq!(
            parse_max_value_size(Some("8388608".into())).unwrap(),
            8 * 1024 * 1024
        );
        assert!(parse_max_value_size(Some(String::new())).is_err());
    }
}

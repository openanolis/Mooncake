use std::ffi::{c_int, CString};
use std::sync::Arc;

use mooncake_store_core::{Result, StoreError};

use super::*;
use crate::client::cold_tier::nof::{
    NofBacking, NofHealth, NofPhysicalDelete, NofPhysicalDeleteRequest, NofPhysicalRead,
    NofPhysicalReadRequest, NofPhysicalWrite, NofPhysicalWriteRequest, NofStorageHealth,
    OpaquePhysicalKey,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvcsMode {
    LowLevel,
}

impl KvcsMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "low-level" => Ok(Self::LowLevel),
            _ => Err(StoreError::InvalidState(format!(
                "{KVCS_MODE_ENV} must be 'low-level', got {value:?}"
            ))),
        }
    }

    fn from_env() -> Result<Self> {
        // An unset or empty mode means KVCS is not enabled, so there is no implicit data-plane
        // mode to fall back to here.
        let value = env_value(KVCS_MODE_ENV)?
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                StoreError::InvalidState(format!(
                    "{KVCS_MODE_ENV} is not set, so no KVCS backend is enabled"
                ))
            })?;
        Self::parse(value.trim())
    }
}

/// Default Low-Level batch size.
const DEFAULT_MAX_KEYS_PER_BATCH: u32 = 8;
/// Explicit worker-count defaults. These deliberately differ from the SDK header defaults
/// (4/4/2), preserving the deployed adapter behavior.
const DEFAULT_IO_WORKERS: u32 = 1;
/// Default Low-Level operation timeout.
const DEFAULT_OPERATION_TIMEOUT_MS: u64 = 30_000;
/// SDK tuning resolved from the deployment environment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KvcsSdkTuning {
    max_key_size: u32,
    max_keys_per_batch: u32,
    get_workers: u32,
    set_workers: u32,
    simple_workers: u32,
    operation_timeout_ms: u64,
}

impl KvcsSdkTuning {
    fn from_env() -> Result<Self> {
        Self::parse(&env_value)
    }

    /// Parses the tuning surface through an injectable lookup so defaults and rejection rules
    /// stay testable without mutating process-wide environment state.
    fn parse(get: &EnvLookup) -> Result<Self> {
        Ok(Self {
            max_key_size: bounded_env(
                get,
                KVCS_MAX_KEY_SIZE_ENV,
                DEFAULT_MAX_KEY_SIZE as u32,
                MAX_KEY_SIZE_LIMIT as u32,
            )?,
            max_keys_per_batch: nonneg_env(
                get,
                KVCS_MAX_KEYS_PER_BATCH_ENV,
                DEFAULT_MAX_KEYS_PER_BATCH,
            )?,
            get_workers: nonneg_env(get, KVCS_GET_WORKERS_ENV, DEFAULT_IO_WORKERS)?,
            set_workers: nonneg_env(get, KVCS_SET_WORKERS_ENV, DEFAULT_IO_WORKERS)?,
            simple_workers: nonneg_env(get, KVCS_SIMPLE_WORKERS_ENV, DEFAULT_IO_WORKERS)?,
            operation_timeout_ms: timeout_env(
                get,
                KVCS_OPERATION_TIMEOUT_MS_ENV,
                DEFAULT_OPERATION_TIMEOUT_MS,
            )?,
        })
    }

    /// Inclusive raw key ceiling, used for both validation and the SDK config.
    fn key_size(&self) -> usize {
        self.max_key_size as usize
    }

    /// Deadline for data-plane batch calls.
    fn deadline_ns(&self) -> i64 {
        operation_deadline_ns(self.operation_timeout_ms)
    }
}

/// Executors are built lazily from several entry points, so resolve the process environment once
/// and share the resulting tuning.
fn sdk_tuning() -> Result<&'static KvcsSdkTuning> {
    static TUNING: std::sync::OnceLock<KvcsSdkTuning> = std::sync::OnceLock::new();
    if let Some(tuning) = TUNING.get() {
        return Ok(tuning);
    }
    let parsed = KvcsSdkTuning::from_env()?;
    // A racing initializer can only produce an equal value: both read the same process env.
    Ok(TUNING.get_or_init(|| parsed))
}

/// Empty or blank values are treated as not set by the parser.
fn configured(get: &EnvLookup, name: &str) -> Result<Option<String>> {
    Ok(get(name)?.filter(|value| !value.trim().is_empty()))
}

fn tuning_error(name: &str, value: &str, expected: &str) -> StoreError {
    StoreError::InvalidState(format!("{name}={value:?} is invalid: expected {expected}"))
}

/// Zero is accepted and means "use the SDK default".
fn nonneg_env(get: &EnvLookup, name: &str, default: u32) -> Result<u32> {
    const EXPECTED: &str = "an integer in 0..=2147483647";
    let Some(value) = configured(get, name)? else {
        return Ok(default);
    };
    let parsed = value
        .trim()
        .parse::<u64>()
        .map_err(|_| tuning_error(name, &value, EXPECTED))?;
    if parsed > i32::MAX as u64 {
        return Err(tuning_error(name, &value, EXPECTED));
    }
    Ok(parsed as u32)
}

/// A zero or unrepresentable timeout is rejected rather than disabling the deadline.
fn timeout_env(get: &EnvLookup, name: &str, default: u64) -> Result<u64> {
    const EXPECTED: &str = "a positive number of milliseconds";
    let Some(value) = configured(get, name)? else {
        return Ok(default);
    };
    let parsed = value
        .trim()
        .parse::<u64>()
        .map_err(|_| tuning_error(name, &value, EXPECTED))?;
    if parsed == 0 || parsed > (i64::MAX / 1_000_000) as u64 {
        return Err(tuning_error(name, &value, EXPECTED));
    }
    Ok(parsed)
}

/// Key size is nonzero and bounded by the public SDK limit.
fn bounded_env(get: &EnvLookup, name: &str, default: u32, limit: u32) -> Result<u32> {
    let Some(value) = configured(get, name)? else {
        return Ok(default);
    };
    let expected = format!("an integer in 1..={limit}");
    let parsed = value
        .trim()
        .parse::<u64>()
        .map_err(|_| tuning_error(name, &value, &expected))?;
    if parsed == 0 || parsed > u64::from(limit) {
        return Err(tuning_error(name, &value, &expected));
    }
    Ok(parsed as u32)
}

struct KvcsConfig {
    client: KvcsLowLevelClient,
    mountpoint_index: u32,
}

/// One shareable KVCS Low-Level SDK client for the local EFC process.
///
/// `mountpoint_index` selects a backend inside this EFC. It is not a Mooncake node identity.
#[derive(Clone)]
pub struct KvcsLowLevelClient {
    inner: Arc<KvcsLowLevelClientInner>,
}

struct KvcsLowLevelClientInner {
    efc_socket: CString,
    client: ClientSlot,
    max_value_size: u64,
    tuning: KvcsSdkTuning,
}

impl KvcsLowLevelClient {
    pub fn new(efc_socket: String, max_value_size: u64) -> Result<Self> {
        validate_max_value_size(max_value_size)?;
        let efc_socket = CString::new(efc_socket)
            .map_err(|_| StoreError::InvalidState("KVCS EFC socket contains NUL".to_string()))?;
        Ok(Self {
            inner: Arc::new(KvcsLowLevelClientInner {
                efc_socket,
                client: ClientSlot::default(),
                max_value_size,
                tuning: *sdk_tuning()?,
            }),
        })
    }

    /// Binds one storage domain inside the local EFC to an executor.
    pub fn executor(&self, mountpoint_index: u32) -> KvcsCapiExecutor {
        KvcsCapiExecutor {
            config: KvcsConfig {
                client: self.clone(),
                mountpoint_index,
            },
            max_value_size: self.inner.max_value_size,
            tuning: self.inner.tuning,
        }
    }

    fn ensure_client(&self) -> Result<*mut KvcsClient> {
        self.inner.client.get_or_create(
            "low-level",
            || {
                let tuning = self.inner.tuning;
                let raw = KvcsLlConfig {
                    // The Low-Level client requires a non-empty local EFC socket.
                    efc_socket: self.inner.efc_socket.as_ptr(),
                    max_value_size: self.inner.max_value_size as i64,
                    max_key_size: tuning.max_key_size as c_int,
                    max_keys_per_batch: tuning.max_keys_per_batch as c_int,
                    get_workers: tuning.get_workers as c_int,
                    set_workers: tuning.set_workers as c_int,
                    simple_workers: tuning.simple_workers as c_int,
                    ..KvcsLlConfig::default()
                };
                unsafe { kvcs_ll_create(&raw) }
            },
            kvcs_ll_client_destroy,
        )
    }
}

/// KVCS Low-Level C API executor.
pub struct KvcsCapiExecutor {
    config: KvcsConfig,
    max_value_size: u64,
    tuning: KvcsSdkTuning,
}

impl KvcsCapiExecutor {
    pub fn new() -> Result<Self> {
        Self::with_mode(KvcsMode::from_env()?)
    }

    pub fn with_mode(mode: KvcsMode) -> Result<Self> {
        match mode {
            KvcsMode::LowLevel => {
                let efc_socket = env_value(KVCS_EFC_SOCKET_ENV)?
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(default_efc_socket);
                Self::with_low_level_config(
                    efc_socket,
                    env_u32(KVCS_MOUNTPOINT_INDEX_ENV, 0)?,
                    parse_max_value_size(env_value(KVCS_MAX_VALUE_SIZE_ENV)?)?,
                )
            }
        }
    }

    pub fn with_low_level_config(
        efc_socket: String,
        mountpoint_index: u32,
        max_value_size: u64,
    ) -> Result<Self> {
        Ok(KvcsLowLevelClient::new(efc_socket, max_value_size)?.executor(mountpoint_index))
    }

    fn ensure_client(&self) -> Result<*mut KvcsClient> {
        self.config.client.ensure_client()
    }

    fn prepare_batch(&self, len: usize) -> Result<(*mut KvcsClient, c_int)> {
        Ok((self.ensure_client()?, checked_batch_len(len)?))
    }

    fn low_level_options(&self) -> KvcsLlBatchOptions {
        KvcsLlBatchOptions {
            mountpoint_index: self.config.mountpoint_index,
        }
    }
}

pub(super) fn validate_max_value_size(max_value_size: u64) -> Result<()> {
    if !(1..=SDK_MAX_VALUE_SIZE).contains(&max_value_size) {
        return Err(StoreError::InvalidState(format!(
            "KVCS max value size must be in 1..={SDK_MAX_VALUE_SIZE}, got {max_value_size}"
        )));
    }
    Ok(())
}

impl NofBacking for KvcsCapiExecutor {
    fn physical_write(&self) -> Option<&dyn NofPhysicalWrite> {
        Some(self)
    }

    fn physical_read(&self) -> Option<&dyn NofPhysicalRead> {
        Some(self)
    }

    fn physical_delete(&self) -> Option<&dyn NofPhysicalDelete> {
        Some(self)
    }

    fn health_capability(&self) -> Option<&dyn NofHealth> {
        Some(self)
    }
}

mod low_level {
    //! KVCS Low-Level SDK executor for the provider-addressed physical KV contract.

    use std::collections::BTreeSet;
    use std::sync::Mutex;

    use mooncake_store_core::{Result, StoreError};

    use super::super::*;
    use super::{
        NofHealth, NofPhysicalDelete, NofPhysicalDeleteRequest, NofPhysicalRead,
        NofPhysicalReadRequest, NofPhysicalWrite, NofPhysicalWriteRequest, NofStorageHealth,
        OpaquePhysicalKey,
    };

    fn pipeline_batch_items(configured: u32) -> usize {
        usize::try_from(if configured == 0 {
            super::DEFAULT_MAX_KEYS_PER_BATCH
        } else {
            configured
        })
        .unwrap_or(super::DEFAULT_MAX_KEYS_PER_BATCH as usize)
    }

    impl KvcsCapiExecutor {
        fn put_records(&self, requests: &[(&OpaquePhysicalKey, &[u8])]) -> Vec<Result<()>> {
            let max_key_size = self.tuning.key_size();
            let (results, valid) = prepare_positional(requests, |(key, value)| {
                Ok((encode_physical_key(key, max_key_size)?, *value))
            });
            if valid.is_empty() {
                return finish_positional_results(results);
            }
            let (client, count) = match self.prepare_batch(valid.len()) {
                Ok(prepared) => prepared,
                Err(error) => return fail_positional_results(results, error),
            };
            let (pointers, lengths) = input_segments(valid.iter().map(|(_, (_, value))| *value));
            let items = valid
                .iter()
                .enumerate()
                .map(|(offset, (_, (key, _)))| KvcsLlPutItem {
                    key: key.as_ptr(),
                    value_segs: &pointers[offset],
                    seg_lens: &lengths[offset],
                    seg_count: 1,
                })
                .collect::<Vec<_>>();
            let mut native_results = put_results(valid.len());
            let options = self.low_level_options();
            let status = unsafe {
                kvcs_ll_batch_put(
                    client,
                    items.as_ptr(),
                    count,
                    native_results.as_mut_ptr(),
                    count,
                    self.tuning.deadline_ns(),
                    &options,
                )
            };
            if status != 0 {
                return fail_positional_results(results, map_call_status(status, "low-level put"));
            }
            finish_unit_statuses(
                results,
                &valid,
                native_results.into_iter().map(|result| result.status),
                "low-level put item",
                true,
            )
        }

        fn get_records_dynamic(&self, keys: &[&OpaquePhysicalKey]) -> Vec<Result<Option<Vec<u8>>>> {
            struct DynamicGets {
                values: Vec<Option<Vec<u8>>>,
                statuses: Vec<c_int>,
                seen: Vec<bool>,
                malformed: bool,
                max_value_size: usize,
            }

            unsafe extern "C" fn capture(
                count: c_int,
                data_ptrs: *const *const std::ffi::c_void,
                data_lens: *const libc::size_t,
                statuses: *const c_int,
                context: *mut std::ffi::c_void,
            ) {
                if count < 0 || context.is_null() || statuses.is_null() {
                    return;
                }
                let output = unsafe { &*context.cast::<Mutex<DynamicGets>>() };
                let Ok(mut output) = output.lock() else {
                    return;
                };
                let count = count as usize;
                if count != output.values.len() {
                    output.malformed = true;
                }
                for index in 0..count.min(output.values.len()) {
                    output.seen[index] = true;
                    let status = unsafe { *statuses.add(index) };
                    if status <= 0 {
                        // The SDK may invoke the callback more than once while falling back from
                        // local to remote storage. A later miss must not erase an earlier hit.
                        if output.statuses[index] <= 0 {
                            output.statuses[index] = status;
                        }
                        continue;
                    }
                    output.statuses[index] = status;
                    if data_lens.is_null() || data_ptrs.is_null() {
                        output.statuses[index] = -libc::EIO;
                        continue;
                    }
                    let length = unsafe { *data_lens.add(index) };
                    if length > output.max_value_size {
                        output.statuses[index] = -libc::E2BIG;
                        continue;
                    }
                    let data = unsafe { *data_ptrs.add(index) };
                    if data.is_null() {
                        output.statuses[index] = -libc::EIO;
                        continue;
                    }
                    output.values[index] = Some(
                        unsafe { std::slice::from_raw_parts(data.cast::<u8>(), length) }.to_vec(),
                    );
                }
            }

            let max_key_size = self.tuning.key_size();
            let (mut results, valid) =
                prepare_positional(keys, |key| encode_physical_key(key, max_key_size));
            if valid.is_empty() {
                return finish_positional_results(results);
            }
            let (client, count) = match self.prepare_batch(valid.len()) {
                Ok(prepared) => prepared,
                Err(error) => return fail_positional_results(results, error),
            };
            let items = valid
                .iter()
                .map(|(_, key)| KvcsLlGetItem { key: key.as_ptr() })
                .collect::<Vec<_>>();
            let output = Mutex::new(DynamicGets {
                values: vec![None; valid.len()],
                statuses: vec![0; valid.len()],
                seen: vec![false; valid.len()],
                malformed: false,
                max_value_size: usize::try_from(self.max_value_size).unwrap_or(usize::MAX),
            });
            let options = self.low_level_options();
            let status = unsafe {
                kvcs_ll_batch_get(
                    client,
                    items.as_ptr(),
                    count,
                    Some(capture),
                    std::ptr::from_ref(&output).cast_mut().cast(),
                    self.tuning.deadline_ns(),
                    &options,
                )
            };
            if status != 0 {
                return fail_positional_results(results, map_call_status(status, "low-level get"));
            }
            let output = output
                .into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if output.malformed || output.seen.iter().any(|seen| !seen) {
                return fail_positional_results(
                    results,
                    StoreError::InvalidState(
                        "KVCS low-level get callback omitted positional results".to_string(),
                    ),
                );
            }
            for ((request_index, _), (status, value)) in valid
                .iter()
                .zip(output.statuses.into_iter().zip(output.values))
            {
                results[*request_index] = Some(if status < 0 {
                    let err = map_item_status(status, "low-level get item", false);
                    if matches!(err, StoreError::NotFound(_)) {
                        Ok(None)
                    } else {
                        Err(err)
                    }
                } else if status == 0 {
                    Ok(None)
                } else {
                    value.map(Some).ok_or_else(|| {
                        StoreError::InvalidState(
                            "KVCS low-level get callback omitted a value".to_string(),
                        )
                    })
                });
            }
            finish_positional_results(results)
        }

        fn validate_value(&self, value: &[u8]) -> Result<()> {
            if value.is_empty() {
                return Err(StoreError::InvalidState(
                    "KVCS physical value must not be empty".to_string(),
                ));
            }
            if value.len() as u64 > self.max_value_size {
                return Err(StoreError::InvalidState(format!(
                    "KVCS value is {} bytes, exceeding configured limit {}",
                    value.len(),
                    self.max_value_size
                )));
            }
            Ok(())
        }

        /// Returns `true` only for a record confirmed inserted by this call. An existing
        /// identical record is reusable, but must never be included in rollback.
        fn put_retryable_records(
            &self,
            requests: &[(&OpaquePhysicalKey, &[u8])],
        ) -> Vec<Result<bool>> {
            let mut results = (0..requests.len()).map(|_| None).collect::<Vec<_>>();
            let mut seen = BTreeSet::new();
            let mut duplicates = BTreeSet::new();
            for (key, _) in requests {
                let encoded = key.to_hex();
                if !seen.insert(encoded.clone()) {
                    duplicates.insert(encoded);
                }
            }
            let keys = requests.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            let existing = self.get_records_dynamic(&keys);
            let mut absent = Vec::new();
            for (index, ((key, value), result)) in requests.iter().zip(existing).enumerate() {
                if duplicates.contains(&key.to_hex()) {
                    results[index] = Some(Err(StoreError::Conflict(
                        "KVCS batch contains duplicate physical keys".into(),
                    )));
                    continue;
                }
                results[index] = Some(match result {
                    Ok(Some(bytes)) if bytes == *value => Ok(false),
                    Ok(Some(_)) => Err(StoreError::Conflict(
                        "KVCS low-level key already contains a different value".to_string(),
                    )),
                    Ok(None) => {
                        absent.push((index, *key, *value));
                        continue;
                    }
                    Err(error) => Err(error),
                });
            }
            let writes = absent
                .iter()
                .map(|(_, key, value)| (*key, *value))
                .collect::<Vec<_>>();
            for ((index, key, value), status) in absent.into_iter().zip(self.put_records(&writes)) {
                results[index] = Some(match status {
                    Ok(()) => Ok(true),
                    Err(StoreError::Conflict(_)) => {
                        match self.get_records_dynamic(&[key]).remove(0) {
                            Ok(Some(existing)) if existing == value => Ok(false),
                            Ok(_) => Err(StoreError::Conflict(
                                "KVCS low-level retry found a different value".to_string(),
                            )),
                            Err(error) => Err(error),
                        }
                    }
                    Err(error) => Err(error),
                });
            }
            finish_positional_results(results)
        }

        fn delete_records(&self, keys: &[&OpaquePhysicalKey]) -> Vec<Result<()>> {
            let max_key_size = self.tuning.key_size();
            let (results, valid) =
                prepare_positional(keys, |key| encode_physical_key(key, max_key_size));
            if valid.is_empty() {
                return finish_positional_results(results);
            }
            let (client, count) = match self.prepare_batch(valid.len()) {
                Ok(prepared) => prepared,
                Err(error) => return fail_positional_results(results, error),
            };
            let pointers = valid
                .iter()
                .map(|(_, key)| key.as_ptr())
                .collect::<Vec<_>>();
            let mut statuses = vec![0; valid.len()];
            let options = self.low_level_options();
            let status = unsafe {
                kvcs_ll_batch_delete(
                    client,
                    pointers.as_ptr(),
                    count,
                    statuses.as_mut_ptr(),
                    count,
                    self.tuning.deadline_ns(),
                    &options,
                )
            };
            if status != 0 {
                return fail_positional_results(
                    results,
                    map_call_status(status, "low-level delete"),
                );
            }
            finish_unit_statuses(results, &valid, statuses, "low-level delete item", true)
        }

        fn mountpoint_health(&self) -> Result<NofStorageHealth> {
            let client = self.ensure_client()?;
            let mut stats = (0..KVCS_MOUNTPOINT_MAX)
                .map(|_| unsafe { std::mem::zeroed::<KvcsMountpointStat>() })
                .collect::<Vec<_>>();
            let capacity = c_int::try_from(stats.len()).expect("KVCS mountpoint limit fits c_int");
            let mut count = 0;
            let status = unsafe {
                kvcs_ll_mp_stat(
                    client,
                    stats.as_mut_ptr(),
                    capacity,
                    &mut count,
                    self.tuning.deadline_ns(),
                )
            };
            if status != 0 {
                return Err(map_call_status(status, "low-level mountpoint capacity"));
            }
            let count = usize::try_from(count).map_err(|_| {
                StoreError::InvalidState("KVCS returned a negative mountpoint count".to_string())
            })?;
            if count > stats.len() {
                return Err(StoreError::InvalidState(format!(
                    "KVCS returned {count} mountpoints above the SDK limit {}",
                    stats.len()
                )));
            }
            stats.truncate(count);
            let stat = if self.config.mountpoint_index == 0 && stats.len() == 1 {
                stats.first()
            } else {
                stats
                    .iter()
                    .find(|stat| stat.index == self.config.mountpoint_index)
            }
            .ok_or_else(|| {
                StoreError::NotFound(format!(
                    "KVCS mountpoint index {} is absent from the capacity snapshot",
                    self.config.mountpoint_index
                ))
            })?;
            if stat.total_bytes == 0 {
                return Ok(NofStorageHealth::default());
            }
            if stat.used_bytes > stat.total_bytes {
                return Err(StoreError::InvalidState(format!(
                    "KVCS mountpoint {} reports used bytes {} above total bytes {}",
                    stat.index, stat.used_bytes, stat.total_bytes
                )));
            }
            Ok(NofStorageHealth {
                capacity_bytes: Some(stat.total_bytes),
                available_bytes: Some(stat.total_bytes - stat.used_bytes),
            })
        }
    }

    impl NofPhysicalWrite for KvcsCapiExecutor {
        fn put_batch(&self, requests: &[NofPhysicalWriteRequest<'_>]) -> Vec<Result<()>> {
            let (mut results, valid) = prepare_positional(requests, |request| {
                self.validate_value(request.value)?;
                Ok((&request.key, request.value))
            });
            let batch_items = pipeline_batch_items(self.tuning.max_keys_per_batch);
            for chunk in valid.chunks(batch_items) {
                let writes = chunk.iter().map(|(_, write)| *write).collect::<Vec<_>>();
                for ((index, _), status) in chunk.iter().zip(self.put_retryable_records(&writes)) {
                    results[*index] = Some(status.map(|_| ()));
                }
            }
            finish_positional_results(results)
        }
    }

    impl NofPhysicalRead for KvcsCapiExecutor {
        fn get_batch(&self, requests: &[NofPhysicalReadRequest]) -> Vec<Result<Option<Vec<u8>>>> {
            let keys = requests
                .iter()
                .map(|request| &request.key)
                .collect::<Vec<_>>();
            let batch_items = pipeline_batch_items(self.tuning.max_keys_per_batch);
            keys.chunks(batch_items)
                .flat_map(|chunk| self.get_records_dynamic(chunk))
                .collect()
        }
    }

    impl NofPhysicalDelete for KvcsCapiExecutor {
        fn delete_batch(&self, requests: &[NofPhysicalDeleteRequest]) -> Vec<Result<()>> {
            let keys = requests
                .iter()
                .map(|request| &request.key)
                .collect::<Vec<_>>();
            let batch_items = pipeline_batch_items(self.tuning.max_keys_per_batch);
            keys.chunks(batch_items)
                .flat_map(|chunk| self.delete_records(chunk))
                .collect()
        }
    }

    impl NofHealth for KvcsCapiExecutor {
        fn health(&self) -> Result<NofStorageHealth> {
            self.mountpoint_health()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::client::cold_tier::nof::NofBacking;

        #[test]
        fn pipeline_window_honors_explicit_batch_size_and_defaults_zero() {
            assert_eq!(pipeline_batch_items(1), 1);
            assert_eq!(pipeline_batch_items(32), 32);
            assert_eq!(
                pipeline_batch_items(0),
                super::super::DEFAULT_MAX_KEYS_PER_BATCH as usize
            );
        }

        fn round_trip(value: &[u8]) {
            let executor = super::KvcsCapiExecutor::with_mode(super::KvcsMode::LowLevel).unwrap();
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let key = OpaquePhysicalKey::new(format!("mooncake-smoke-{nonce}").into_bytes());
            executor
                .physical_write()
                .unwrap()
                .put_batch(&[NofPhysicalWriteRequest {
                    key: key.clone(),
                    value,
                }])
                .remove(0)
                .unwrap();
            assert_eq!(
                executor
                    .physical_read()
                    .unwrap()
                    .get_batch(&[NofPhysicalReadRequest { key: key.clone() }])
                    .remove(0)
                    .unwrap(),
                Some(value.to_vec())
            );
            executor
                .physical_write()
                .unwrap()
                .put_batch(&[NofPhysicalWriteRequest {
                    key: key.clone(),
                    value,
                }])
                .remove(0)
                .expect("identical retry");
            let different = vec![0xff; value.len()];
            assert!(matches!(
                executor
                    .physical_write()
                    .unwrap()
                    .put_batch(&[NofPhysicalWriteRequest {
                        key: key.clone(),
                        value: &different,
                    }])
                    .remove(0),
                Err(StoreError::Conflict(_))
            ));
            assert_eq!(
                executor
                    .physical_read()
                    .unwrap()
                    .get_batch(&[NofPhysicalReadRequest { key: key.clone() }])
                    .remove(0)
                    .unwrap(),
                Some(value.to_vec())
            );
            executor
                .physical_delete()
                .unwrap()
                .delete_batch(&[NofPhysicalDeleteRequest { key: key.clone() }])
                .remove(0)
                .unwrap();
            assert_eq!(
                executor
                    .physical_read()
                    .unwrap()
                    .get_batch(&[NofPhysicalReadRequest { key }])
                    .remove(0)
                    .unwrap(),
                None
            );
            assert!(executor.physical_read().is_some());
            assert!(executor.physical_write().is_some());
            assert!(executor.health_capability().is_some());
            let health = executor.health_capability().unwrap().health().unwrap();
            assert!(health.capacity_bytes.is_some());
            assert!(health.available_bytes.is_some());
        }

        #[test]
        #[ignore = "requires a live KVCS EFC and configured mountpoint, or the official SDK mock"]
        fn live_low_level_round_trip_smoke() {
            round_trip(b"kvcs-capi-smoke");
        }

        #[test]
        fn oversized_value_is_rejected_before_client_creation() {
            let executor =
                KvcsCapiExecutor::with_low_level_config("/missing/kvcs.sock".into(), 0, 64)
                    .unwrap();
            let key = OpaquePhysicalKey::new(b"oversized".to_vec());
            let result = executor
                .physical_write()
                .unwrap()
                .put_batch(&[NofPhysicalWriteRequest {
                    key,
                    value: &[7; 65],
                }])
                .remove(0);
            assert!(matches!(result, Err(StoreError::InvalidState(_))));
        }
    }
}

#[cfg(test)]
mod mode_tests {
    use super::*;

    #[test]
    fn mode_parser_accepts_only_documented_values() {
        assert_eq!(KvcsMode::parse("low-level").unwrap(), KvcsMode::LowLevel);
        assert!(KvcsMode::parse("standard").is_err());
        assert!(KvcsMode::parse("lowlevel").is_err());
    }

    #[test]
    fn selected_mode_exposes_only_the_physical_data_plane() {
        let low_level = KvcsCapiExecutor::with_mode(KvcsMode::LowLevel).unwrap();
        assert!(low_level.object_read().is_none());
        assert!(low_level.physical_read().is_some());
        assert!(low_level.physical_write().is_some());
        assert!(low_level.physical_query().is_none());
        assert!(low_level.health_capability().is_some());
    }

    #[test]
    fn low_level_client_binds_static_targets_to_one_sdk_client() {
        let client =
            KvcsLowLevelClient::new("/run/kvcs.sock".to_string(), SDK_DEFAULT_MAX_VALUE_SIZE)
                .unwrap();
        let first = client.executor(3);
        let second = client.executor(7);
        assert_eq!(
            (
                first.config.mountpoint_index,
                second.config.mountpoint_index
            ),
            (3, 7)
        );
        assert!(Arc::ptr_eq(
            &first.config.client.inner,
            &second.config.client.inner
        ));
    }
}

#[cfg(test)]
mod tuning_tests {
    use super::*;
    use std::collections::BTreeMap;

    fn parse(values: &[(&str, &str)]) -> Result<KvcsSdkTuning> {
        let map: BTreeMap<String, String> = values
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect();
        KvcsSdkTuning::parse(&move |name: &str| Ok(map.get(name).cloned()))
    }

    #[test]
    fn defaults_preserve_the_adapter_contract() {
        let tuning = parse(&[]).unwrap();
        assert_eq!(tuning.max_key_size, DEFAULT_MAX_KEY_SIZE as u32);
        assert_eq!(tuning.max_keys_per_batch, 8);
        assert_eq!(
            (
                tuning.get_workers,
                tuning.set_workers,
                tuning.simple_workers
            ),
            (1, 1, 1)
        );
        assert_eq!(tuning.operation_timeout_ms, 30_000);
    }

    #[test]
    fn an_empty_value_is_treated_as_unset() {
        let tuning = parse(&[(KVCS_MAX_KEYS_PER_BATCH_ENV, "   ")]).unwrap();
        assert_eq!(tuning.max_keys_per_batch, DEFAULT_MAX_KEYS_PER_BATCH);
    }

    #[test]
    fn worker_and_batch_overrides_accept_zero_for_the_sdk_default() {
        let tuning = parse(&[
            (KVCS_MAX_KEYS_PER_BATCH_ENV, "0"),
            (KVCS_GET_WORKERS_ENV, "0"),
            (KVCS_SIMPLE_WORKERS_ENV, "0"),
        ])
        .unwrap();
        assert_eq!(tuning.max_keys_per_batch, 0);
        assert_eq!(tuning.get_workers, 0);
        assert_eq!(tuning.simple_workers, 0);
    }

    #[test]
    fn max_key_size_is_bounded_by_the_sdk_limit() {
        assert_eq!(
            parse(&[(KVCS_MAX_KEY_SIZE_ENV, "1024")])
                .unwrap()
                .max_key_size,
            MAX_KEY_SIZE_LIMIT as u32
        );
        assert!(parse(&[(KVCS_MAX_KEY_SIZE_ENV, "1025")]).is_err());
        assert!(parse(&[(KVCS_MAX_KEY_SIZE_ENV, "0")]).is_err());
    }

    #[test]
    fn timeouts_reject_zero_and_unparsable_values() {
        assert_eq!(
            parse(&[(KVCS_OPERATION_TIMEOUT_MS_ENV, "1500")])
                .unwrap()
                .operation_timeout_ms,
            1_500
        );
        assert!(parse(&[(KVCS_OPERATION_TIMEOUT_MS_ENV, "0")]).is_err());
    }

    #[test]
    fn zero_deadline_keeps_the_sdk_default() {
        let tuning = parse(&[]).unwrap();
        assert!(tuning.deadline_ns() > 0);
        // The SDK reads zero as "no deadline", so a disabled timeout must not become "expired".
        assert_eq!(operation_deadline_ns(0), 0);
    }

    #[test]
    fn low_level_value_ceiling_matches_the_public_sdk_limit() {
        assert!(validate_max_value_size(1).is_ok());
        assert!(validate_max_value_size(SDK_MAX_VALUE_SIZE).is_ok());
        assert!(validate_max_value_size(SDK_MAX_VALUE_SIZE + 1).is_err());
    }

    #[test]
    fn an_unset_mode_disables_kvcs_instead_of_defaulting_to_low_level() {
        let _guard = crate::observability::test_process_lock().lock();
        let previous = std::env::var_os(KVCS_MODE_ENV);
        std::env::remove_var(KVCS_MODE_ENV);
        let unset = KvcsMode::from_env();
        std::env::set_var(KVCS_MODE_ENV, "   ");
        let blank = KvcsMode::from_env();
        match previous {
            Some(value) => std::env::set_var(KVCS_MODE_ENV, value),
            None => std::env::remove_var(KVCS_MODE_ENV),
        }
        assert!(unset.is_err());
        assert!(blank.is_err());
    }
}

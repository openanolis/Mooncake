use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::Mutex;

use libc::size_t;
use mooncake_store_core::error::QuotaKind;

pub(super) const SDK_DEFAULT_MAX_VALUE_SIZE: u64 = 4 * 1024 * 1024;
const SDK_MAX_VALUE_SIZE: u64 = 4 * 1024 * 1024 * 1024;
/// Default raw key size passed to the SDK.
const DEFAULT_MAX_KEY_SIZE: usize = 256;
/// The SDK rejects a larger inclusive raw key.
const MAX_KEY_SIZE_LIMIT: usize = 1024;
const KVCS_NAMESPACE_MAX_BYTES: usize = 127;

#[allow(dead_code, non_camel_case_types)]
mod sdk_ffi {
    include!(env!("KVCS_SDK_RUST_FFI"));
}

use sdk_ffi::{
    kvcs_client_t as KvcsClient, kvcs_ll_batch_delete, kvcs_ll_batch_get,
    kvcs_ll_batch_opts_t as KvcsLlBatchOptions, kvcs_ll_batch_put,
    kvcs_ll_config_t as KvcsLlConfig, kvcs_ll_create, kvcs_ll_get_item_t as KvcsLlGetItem,
    kvcs_ll_mp_stat, kvcs_ll_put_item_t as KvcsLlPutItem,
    kvcs_mountpoint_stat_t as KvcsMountpointStat, kvcs_put_result_t as KvcsPutResult,
    KVCS_MOUNTPOINT_MAX,
};

impl Default for sdk_ffi::kvcs_ll_config_t {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

unsafe extern "C" {
    fn kvcs_ll_client_destroy(client: *mut KvcsClient);
}

struct ClientHandle {
    client: *mut KvcsClient,
    destroy: unsafe extern "C" fn(*mut KvcsClient),
}

// KVCS documents one client as safe for concurrent synchronous batch operations. Mutexes below
// protect only lazy construction and destruction.
unsafe impl Send for ClientHandle {}

impl Drop for ClientHandle {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.client) };
    }
}

#[derive(Default)]
struct ClientSlot(Mutex<Option<ClientHandle>>);

impl ClientSlot {
    fn get_or_create(
        &self,
        mode: &str,
        create: impl FnOnce() -> *mut KvcsClient,
        destroy: unsafe extern "C" fn(*mut KvcsClient),
    ) -> Result<*mut KvcsClient> {
        let mut slot = self
            .0
            .lock()
            .map_err(|_| StoreError::InvalidState(format!("KVCS {mode} client lock poisoned")))?;
        if let Some(handle) = slot.as_ref() {
            return Ok(handle.client);
        }
        let client = create();
        if client.is_null() {
            return Err(StoreError::Transport(format!(
                "KVCS {mode} client creation failed"
            )));
        }
        slot.replace(ClientHandle { client, destroy });
        Ok(client)
    }
}

fn env_u32(name: &str, default: u32) -> Result<u32> {
    match env_value(name)? {
        Some(value) => value.parse().map_err(|_| {
            StoreError::InvalidState(format!("{name} must be a non-negative integer"))
        }),
        None => Ok(default),
    }
}

fn checked_batch_len(len: usize) -> Result<c_int> {
    c_int::try_from(len).map_err(|_| {
        StoreError::InvalidState(format!("KVCS batch size {len} exceeds the C ABI limit"))
    })
}

/// The SDK takes an absolute `CLOCK_MONOTONIC`
/// timestamp and reads `0` as "no deadline", so a zero timeout disables the deadline instead
/// of expiring the call immediately.
fn operation_deadline_ns(timeout_ms: u64) -> i64 {
    if timeout_ms == 0 {
        return 0;
    }
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid writable `timespec` and CLOCK_MONOTONIC is always present.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) } != 0 {
        return 0;
    }
    let elapsed_ns = i128::from(now.tv_sec) * 1_000_000_000 + i128::from(now.tv_nsec);
    let timeout_ns = i128::from(timeout_ms) * 1_000_000;
    i64::try_from(elapsed_ns + timeout_ns).unwrap_or(i64::MAX)
}

fn input_segments<'a>(
    values: impl IntoIterator<Item = &'a [u8]>,
) -> (Vec<*const c_void>, Vec<size_t>) {
    values
        .into_iter()
        .map(|value| {
            let pointer = if value.is_empty() {
                std::ptr::null()
            } else {
                value.as_ptr().cast()
            };
            (pointer, value.len())
        })
        .unzip()
}

fn put_results(len: usize) -> Vec<KvcsPutResult> {
    (0..len)
        .map(|_| KvcsPutResult {
            location: [0; 256],
            status: 0,
        })
        .collect()
}

fn map_call_status(status: c_int, operation: &str) -> StoreError {
    if status >= 0 {
        return StoreError::Transport(format!(
            "KVCS {operation} returned unexpected call status {status}"
        ));
    }
    match status.unsigned_abs() {
        1 => StoreError::NotFound(format!("KVCS {operation}")),
        2 => StoreError::Backpressure(format!("KVCS {operation} is incomplete")),
        3 => StoreError::Transport(format!("KVCS {operation} provider is unavailable")),
        4 => StoreError::Backpressure(format!("KVCS {operation} resource exhausted")),
        5 => StoreError::InvalidState(format!("KVCS {operation} invalid argument")),
        6 => StoreError::NotFound(format!("KVCS {operation} namespace")),
        7 => StoreError::Backpressure(format!("KVCS {operation} buffer too small")),
        8 => StoreError::Conflict(format!("KVCS {operation} already exists")),
        // Treat an unconfigured mountpoint as unavailable so the caller retries instead of
        // giving up.
        code if code == libc::ENODEV as u32 => {
            StoreError::Transport(format!("KVCS {operation} mountpoint unavailable"))
        }
        10009 => StoreError::Transport(format!("KVCS {operation} invalid mount point")),
        10010 => StoreError::InvalidState(format!("KVCS {operation} invalid shard ID")),
        10011 => StoreError::Backpressure(format!("KVCS {operation} namespace is deleting")),
        _ => StoreError::Transport(format!("KVCS {operation} failed: status={status}")),
    }
}

fn map_item_status(status: c_int, operation: &str, write: bool) -> StoreError {
    if status >= 0 {
        return StoreError::Transport(format!(
            "KVCS {operation} returned unexpected item status {status}"
        ));
    }
    let code = status.unsigned_abs();
    if code == libc::ENOENT as u32 {
        StoreError::NotFound(format!("KVCS {operation}"))
    } else if code == libc::EEXIST as u32 {
        StoreError::Conflict(format!("KVCS {operation} already exists"))
    } else if code == libc::ENOSPC as u32 {
        if write {
            StoreError::QuotaExceeded {
                kind: QuotaKind::Bytes,
                message: format!("KVCS {operation} space exhausted"),
            }
        } else {
            StoreError::Backpressure(format!("KVCS {operation} destination buffer too small"))
        }
    } else if code == libc::EINVAL as u32
        || code == libc::E2BIG as u32
        || code == libc::ENAMETOOLONG as u32
    {
        StoreError::InvalidState(format!("KVCS {operation} invalid argument"))
    } else if code == libc::ENODEV as u32 {
        StoreError::Transport(format!("KVCS {operation} mountpoint unavailable"))
    } else if code == libc::ENOMEM as u32 {
        StoreError::Backpressure(format!("KVCS {operation} memory exhausted"))
    } else {
        StoreError::Transport(format!("KVCS {operation} failed: errno={status}"))
    }
}

type PreparedBatch<U, V> = (Vec<Option<Result<V>>>, Vec<(usize, U)>);

fn prepare_positional<'a, T, U, V>(
    requests: &'a [T],
    validate: impl Fn(&'a T) -> Result<U>,
) -> PreparedBatch<U, V> {
    let mut results = (0..requests.len()).map(|_| None).collect::<Vec<_>>();
    let valid = requests
        .iter()
        .enumerate()
        .filter_map(|(index, request)| match validate(request) {
            Ok(value) => Some((index, value)),
            Err(error) => {
                results[index] = Some(Err(error));
                None
            }
        })
        .collect();
    (results, valid)
}

fn fill_missing<T>(results: &mut [Option<Result<T>>], error: &StoreError) {
    for result in results {
        if result.is_none() {
            *result = Some(Err(error.clone()));
        }
    }
}

fn fail_positional_results<T>(
    mut results: Vec<Option<Result<T>>>,
    error: StoreError,
) -> Vec<Result<T>> {
    fill_missing(&mut results, &error);
    finish_positional_results(results)
}

fn finish_unit_statuses<U>(
    mut results: Vec<Option<Result<()>>>,
    valid: &[(usize, U)],
    statuses: impl IntoIterator<Item = c_int>,
    operation: &str,
    write: bool,
) -> Vec<Result<()>> {
    for ((index, _), status) in valid.iter().zip(statuses) {
        results[*index] = Some(if status == 0 {
            Ok(())
        } else {
            Err(map_item_status(status, operation, write))
        });
    }
    finish_positional_results(results)
}

fn validate_string(value: &str, limit: usize, label: &str) -> Result<CString> {
    if value.len() > limit {
        return Err(StoreError::InvalidState(format!(
            "KVCS {label} is {} bytes, exceeding provider limit {limit}",
            value.len()
        )));
    }
    CString::new(value).map_err(|_| StoreError::InvalidState(format!("KVCS {label} contains NUL")))
}

fn encode_physical_key(key: &OpaquePhysicalKey, key_limit: usize) -> Result<CString> {
    let encoded = key.to_hex();
    validate_string(&encoded, key_limit, "encoded low-level key")
}

fn char_array_str(value: &[c_char]) -> std::result::Result<&str, std::str::Utf8Error> {
    let len = value
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(value.len());
    let bytes = unsafe { std::slice::from_raw_parts(value.as_ptr().cast::<u8>(), len) };
    std::str::from_utf8(bytes)
}

fn finish_positional_results<T>(results: Vec<Option<Result<T>>>) -> Vec<Result<T>> {
    results
        .into_iter()
        .map(|result| {
            result.unwrap_or_else(|| {
                Err(StoreError::Transport(
                    "KVCS executor omitted a positional batch result".to_string(),
                ))
            })
        })
        .collect()
}

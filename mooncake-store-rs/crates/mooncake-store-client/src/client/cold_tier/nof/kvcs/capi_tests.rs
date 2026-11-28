use super::*;

#[test]
fn physical_key_limit_is_checked_after_hex_encoding() {
    let key = OpaquePhysicalKey::new(vec![0xab; 128]);
    assert!(encode_physical_key(&key, 256).is_ok());
    let oversized = OpaquePhysicalKey::new(vec![0xab; 129]);
    assert!(encode_physical_key(&oversized, 256).is_err());
}

#[test]
fn incomplete_is_not_mapped_to_not_found() {
    assert!(matches!(
        map_call_status(-2, "query"),
        StoreError::Backpressure(_)
    ));
}

#[test]
fn deleting_namespace_is_retryable() {
    assert!(matches!(
        map_call_status(-10011, "create namespace"),
        StoreError::Backpressure(_)
    ));
}

#[test]
fn low_level_capi_layout_matches_the_sdk_abi() {
    assert_eq!(std::mem::size_of::<KvcsLlConfig>(), 80);
    assert_eq!(std::mem::size_of::<KvcsMountpointStat>(), 288);
}

from __future__ import annotations

import ctypes
import threading
import time

import numpy as np
import pytest

from mooncake.bundle_transfer import MooncakeBundleTransfer


class InMemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.lock = threading.Lock()
        self.registered: set[int] = set()
        self.max_active_puts = 0
        self.max_active_gets = 0
        self.active_puts = 0
        self.active_gets = 0
        self.batch_get_into_calls = 0
        self.batch_remove_calls = 0

    def put(self, key: str, value) -> int:
        self.active_puts += 1
        self.max_active_puts = max(self.max_active_puts, self.active_puts)
        try:
            time.sleep(0.01)
            with self.lock:
                self.objects[key] = bytes(value)
            return 0
        finally:
            self.active_puts -= 1

    def get(self, key: str) -> bytes:
        self.active_gets += 1
        self.max_active_gets = max(self.max_active_gets, self.active_gets)
        try:
            time.sleep(0.01)
            with self.lock:
                return self.objects[key]
        finally:
            self.active_gets -= 1

    def remove(self, key: str, force: bool = False) -> int:
        with self.lock:
            self.objects.pop(key, None)
        return 0

    def batch_remove(self, keys: list[str]) -> list[int]:
        self.batch_remove_calls += 1
        for key in keys:
            self.remove(key, True)
        return [0 for _key in keys]

    def register_buffer(self, buffer_ptr: int, size: int) -> int:
        self.registered.add(buffer_ptr)
        return 0

    def unregister_buffer(self, buffer_ptr: int) -> int:
        self.registered.remove(buffer_ptr)
        return 0

    def batch_get_into(
        self, keys: list[str], ptrs: list[int], sizes: list[int]
    ) -> list[int]:
        self.batch_get_into_calls += 1
        self.active_gets += len(keys)
        self.max_active_gets = max(self.max_active_gets, self.active_gets)
        try:
            time.sleep(0.01)
            results = []
            for key, ptr, size in zip(keys, ptrs, sizes):
                with self.lock:
                    data = self.objects[key]
                if len(data) > size:
                    results.append(-1)
                    continue
                ctypes.memmove(ptr, data, len(data))
                results.append(len(data))
            return results
        finally:
            self.active_gets -= len(keys)


class GetOnlyStore(InMemoryStore):
    batch_get_into = None
    register_buffer = None
    unregister_buffer = None


class FailingPutStore(InMemoryStore):
    def __init__(self, fail_on_put: int) -> None:
        super().__init__()
        self.put_count = 0
        self.fail_on_put = fail_on_put

    def put(self, key: str, value) -> int:
        self.put_count += 1
        if self.put_count == self.fail_on_put:
            raise RuntimeError("injected put failure")
        return super().put(key, value)


class FailingBatchGetStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_key = ""

    def batch_get_into(
        self, keys: list[str], ptrs: list[int], sizes: list[int]
    ) -> list[int]:
        if self.fail_key in keys:
            raise RuntimeError("injected get failure")
        return super().batch_get_into(keys, ptrs, sizes)


class FailingUnregisterStore(InMemoryStore):
    def unregister_buffer(self, buffer_ptr: int) -> int:
        self.registered.remove(buffer_ptr)
        return -1


class FailingRemoveStore(FailingPutStore):
    def remove(self, key: str, force: bool = False) -> int:
        raise RuntimeError("injected remove failure")

    batch_remove = None


def test_bundle_roundtrip_with_named_buffers() -> None:
    store = InMemoryStore()
    transfer = MooncakeBundleTransfer(store, key_prefix="test")
    array = np.arange(16, dtype=np.int32).reshape(4, 4)

    ref = transfer.put_bundle(b"metadata", {"array": array, "raw": b"abc"})
    meta, buffers = transfer.get_bundle(ref)

    assert meta == b"metadata"
    assert np.array_equal(
        np.frombuffer(buffers["array"], dtype=np.int32).reshape(4, 4), array
    )
    assert buffers["raw"] == b"abc"
    assert store.batch_get_into_calls > 0
    assert store.registered == set()


def test_bundle_selected_get() -> None:
    store = InMemoryStore()
    transfer = MooncakeBundleTransfer(store, key_prefix="test")

    ref = transfer.put_bundle(b"metadata", {"a": b"1", "b": b"2"})
    meta, buffers = transfer.get_bundle(ref, buffer_names=["b"])

    assert meta == b"metadata"
    assert buffers == {"b": b"2"}
    with pytest.raises(KeyError, match="unknown bundle buffers"):
        transfer.get_bundle(ref, buffer_names=["missing"])


def test_bundle_chunked_roundtrip() -> None:
    store = InMemoryStore()
    transfer = MooncakeBundleTransfer(store, key_prefix="test")
    payload = bytes(range(128))

    ref = transfer.put_bundle(b"meta", {"payload": payload}, chunk_bytes=17)
    _meta, buffers = transfer.get_bundle(ref)

    assert buffers["payload"] == payload
    assert len(ref.manifest["buffers"]["payload"]["chunks"]) > 1


def test_bundle_remove_deletes_payload_and_manifest() -> None:
    store = InMemoryStore()
    transfer = MooncakeBundleTransfer(store, key_prefix="test")

    ref = transfer.put_bundle(b"meta", {"payload": b"data"})
    assert store.objects
    transfer.remove_bundle(ref)

    assert store.objects == {}
    assert store.batch_remove_calls == 1


def test_bundle_partial_put_failure_cleans_payloads() -> None:
    store = FailingPutStore(fail_on_put=2)
    transfer = MooncakeBundleTransfer(store, key_prefix="test")

    with pytest.raises(RuntimeError, match="injected put failure"):
        transfer.put_bundle(b"meta", {"payload": b"data"}, chunk_bytes=2)

    assert store.objects == {}


def test_bundle_cleanup_failure_preserves_put_error() -> None:
    store = FailingRemoveStore(fail_on_put=2)
    transfer = MooncakeBundleTransfer(store, key_prefix="test")

    with pytest.raises(RuntimeError, match="injected put failure"):
        transfer.put_bundle(b"meta", {"payload": b"data"}, chunk_bytes=2)


def test_bundle_concurrent_put_and_get() -> None:
    store = GetOnlyStore()
    transfer = MooncakeBundleTransfer(store, key_prefix="test")
    payload = bytes(range(128))

    ref = transfer.put_bundle(
        b"meta", {"payload": payload}, chunk_bytes=8, max_inflight_put=4
    )
    _meta, buffers = transfer.get_bundle(ref, max_inflight_get=4)

    assert buffers["payload"] == payload
    assert store.max_active_puts > 1
    assert store.max_active_gets > 1
    assert store.registered == set()


def test_bundle_batch_get_failure_unregisters_buffer() -> None:
    store = FailingBatchGetStore()
    transfer = MooncakeBundleTransfer(store, key_prefix="test")
    ref = transfer.put_bundle(b"meta", {"payload": bytes(range(64))}, chunk_bytes=8)
    store.fail_key = ref.manifest["buffers"]["payload"]["chunks"][2]["key"]

    with pytest.raises(RuntimeError, match="injected get failure"):
        transfer.get_bundle(ref, max_inflight_get=4)

    assert store.registered == set()


def test_bundle_falls_back_to_get_without_batch_get_into() -> None:
    store = GetOnlyStore()
    transfer = MooncakeBundleTransfer(store, key_prefix="test")

    ref = transfer.put_bundle(b"meta", {"payload": b"abcdef"}, chunk_bytes=2)
    _meta, buffers = transfer.get_bundle(ref, max_inflight_get=2)

    assert buffers["payload"] == b"abcdef"
    assert store.max_active_gets > 1


def test_bundle_invalid_policy_and_chunk_size_raise() -> None:
    store = InMemoryStore()
    transfer = MooncakeBundleTransfer(store, key_prefix="test")

    with pytest.raises(ValueError, match="max_inflight_put"):
        transfer.put_bundle(b"meta", {"payload": b"data"}, max_inflight_put=0)
    with pytest.raises(ValueError, match="max_inflight_get"):
        transfer.get_bundle(
            {"manifest": {"version": 1, "layout": "bundle"}}, max_inflight_get=0
        )
    with pytest.raises(ValueError, match="chunk_bytes"):
        transfer.put_bundle(b"meta", {"payload": b"data"}, chunk_bytes=0)


def test_bundle_invalid_name_and_prefix_raise() -> None:
    store = InMemoryStore()
    transfer = MooncakeBundleTransfer(store, key_prefix="test")

    with pytest.raises(ValueError, match="buffer name"):
        transfer.put_bundle(b"meta", {"bad/name": b"data"})
    with pytest.raises(ValueError, match="partition"):
        transfer.put_bundle(b"meta", {"payload": b"data"}, partition="bad/name")
    with pytest.raises(ValueError, match="key_prefix"):
        MooncakeBundleTransfer(store, key_prefix="")


def test_bundle_rejects_tampered_manifest() -> None:
    store = InMemoryStore()
    transfer = MooncakeBundleTransfer(store, key_prefix="test")
    ref = transfer.put_bundle(b"meta", {"payload": b"abcdef"}, chunk_bytes=2)
    tampered = dict(ref.manifest)
    tampered["buffers"] = dict(ref.manifest["buffers"])
    tampered["buffers"]["payload"] = dict(ref.manifest["buffers"]["payload"])
    tampered["buffers"]["payload"]["chunks"] = [
        {"key": "other/object", "bytes": 6},
    ]

    with pytest.raises(ValueError, match="namespace"):
        transfer.get_bundle({"manifest": tampered})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

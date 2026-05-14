from __future__ import annotations

import ctypes
import json
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

DEFAULT_BUNDLE_CHUNK_BYTES = 512 * 1024**2
MISSING_OBJECT_ERROR = -704


class BundleStore(Protocol):
    def put(self, key: str, value: Any) -> int: ...

    def get(self, key: str) -> bytes: ...

    def remove(self, key: str, force: bool = False) -> int: ...


@dataclass(frozen=True)
class BundleTransferPolicy:
    """Controls generic bundle transfer parallelism."""

    max_inflight_put: int = 1
    max_inflight_get: int = 1


@dataclass
class RemoteBundleRef:
    """Reference to a generic named-buffer bundle stored in Mooncake."""

    manifest_key: str
    manifest: dict[str, Any]
    total_bytes: int


class MooncakeBundleTransfer:
    """Transfer generic metadata bytes and named buffers through Mooncake."""

    def __init__(self, store: BundleStore, key_prefix: str = "bundle") -> None:
        self.store = store
        self.key_prefix = _normalize_key_prefix(key_prefix)

    def put_bundle(
        self,
        meta: bytes | bytearray | memoryview,
        buffers: Mapping[str, Any],
        partition: str = "default",
        chunk_bytes: Optional[int] = None,
        policy: Optional[BundleTransferPolicy] = None,
        max_inflight_put: Optional[int] = None,
    ) -> RemoteBundleRef:
        """Store metadata bytes plus named buffers and return a remote reference."""
        _validate_key_segment(partition, "partition")
        meta_view = _bytes_view(meta, "meta")
        target_chunk_bytes = _resolve_chunk_bytes(chunk_bytes)
        transfer_policy = self._resolve_policy(
            policy, max_inflight_put=max_inflight_put
        )

        object_id = f"{partition}/{uuid.uuid4().hex}"
        base_key = f"{self.key_prefix}/{object_id}"
        manifest_key = f"{base_key}/manifest"
        written_keys: list[str] = []
        buffer_specs: dict[str, Any] = {}
        total_bytes = 0
        try:
            meta_spec, meta_bytes, meta_keys = self._put_payload(
                f"{base_key}/meta",
                meta_view,
                target_chunk_bytes,
                transfer_policy.max_inflight_put,
            )
            written_keys.extend(meta_keys)
            total_bytes += meta_bytes

            for name, value in buffers.items():
                _validate_key_segment(name, "buffer name")
                spec, buffer_bytes, buffer_keys = self._put_payload(
                    f"{base_key}/buffer/{name}",
                    _payload_to_view(value, name),
                    target_chunk_bytes,
                    transfer_policy.max_inflight_put,
                )
                buffer_specs[name] = spec
                total_bytes += buffer_bytes
                written_keys.extend(buffer_keys)

            manifest = {
                "version": 1,
                "layout": "bundle",
                "object_id": object_id,
                "meta": meta_spec,
                "buffers": buffer_specs,
            }
            _check_status(
                self.store.put(manifest_key, _encode_manifest(manifest)),
                "put",
                manifest_key,
            )
            written_keys.append(manifest_key)
        except Exception:
            _cleanup_keys(self.store, written_keys, strict=False)
            raise

        return RemoteBundleRef(
            manifest_key=manifest_key, manifest=manifest, total_bytes=total_bytes
        )

    def get_bundle(
        self,
        ref: RemoteBundleRef | Mapping[str, Any],
        buffer_names: Optional[Sequence[str]] = None,
        policy: Optional[BundleTransferPolicy] = None,
        max_inflight_get: Optional[int] = None,
    ) -> Tuple[bytes, dict[str, bytes]]:
        """Materialize metadata bytes and selected named buffers."""
        transfer_policy = self._resolve_policy(
            policy, max_inflight_get=max_inflight_get
        )
        manifest = self._resolve_manifest(ref)
        selected = self._resolve_buffer_names(manifest, buffer_names)
        meta = self._read_payload(manifest["meta"], transfer_policy.max_inflight_get)
        buffers: dict[str, bytes] = {}
        for name in selected:
            buffers[name] = self._read_payload(
                manifest["buffers"][name], transfer_policy.max_inflight_get
            )
        return meta, buffers

    def remove_bundle(self, ref: RemoteBundleRef | Mapping[str, Any]) -> None:
        """Remove all Mooncake objects that belong to a stored bundle."""
        manifest = self._resolve_manifest(ref)
        keys = self._payload_keys(manifest)
        manifest_key = self._manifest_key(ref, manifest)
        if manifest_key is not None:
            keys.append(manifest_key)
        _cleanup_keys(self.store, keys, strict=True)

    def _put_payload(
        self,
        key: str,
        value: memoryview,
        chunk_bytes: int,
        max_inflight_put: int,
    ) -> tuple[dict[str, Any], int, list[str]]:
        chunks = _split_view(value, chunk_bytes)
        chunk_keys = [
            key if len(chunks) == 1 else f"{key}/chunk/{index}"
            for index in range(len(chunks))
        ]
        written_keys = self._put_chunks(chunk_keys, chunks, max_inflight_put)
        spec = {
            "key": key,
            "bytes": len(value),
            "chunks": [
                {"key": chunk_key, "bytes": len(chunk)}
                for chunk_key, chunk in zip(chunk_keys, chunks)
            ],
        }
        return spec, len(value), written_keys

    def _put_chunks(
        self,
        chunk_keys: list[str],
        chunks: list[memoryview],
        max_inflight_put: int,
    ) -> list[str]:
        if max_inflight_put == 1 or len(chunks) <= 1:
            return self._put_chunks_serial(chunk_keys, chunks)

        futures: dict[Future[None], str] = {}
        try:
            with ThreadPoolExecutor(
                max_workers=min(max_inflight_put, len(chunks))
            ) as executor:
                for chunk_key, chunk in zip(chunk_keys, chunks):
                    futures[executor.submit(self._put_chunk, chunk_key, chunk)] = (
                        chunk_key
                    )
                for future in as_completed(futures):
                    future.result()
        except Exception:
            for future in futures:
                future.cancel()
            for future in futures:
                if future.done() and future.cancelled():
                    continue
                try:
                    future.result()
                except Exception:
                    pass
            _cleanup_keys(self.store, chunk_keys, strict=False)
            raise
        return chunk_keys

    def _put_chunks_serial(
        self, chunk_keys: list[str], chunks: list[memoryview]
    ) -> list[str]:
        written_keys: list[str] = []
        try:
            for chunk_key, chunk in zip(chunk_keys, chunks):
                self._put_chunk(chunk_key, chunk)
                written_keys.append(chunk_key)
        except Exception:
            _cleanup_keys(self.store, written_keys, strict=False)
            raise
        return written_keys

    def _put_chunk(self, chunk_key: str, chunk: memoryview) -> None:
        _check_status(self.store.put(chunk_key, chunk), "put", chunk_key)

    def _read_payload(self, spec: Mapping[str, Any], max_inflight_get: int) -> bytes:
        expected_bytes = int(spec["bytes"])
        if expected_bytes == 0:
            return b""
        data = bytearray(expected_bytes)
        self._read_payload_into(spec, data, max_inflight_get)
        return bytes(data)

    def _read_payload_into(
        self, spec: Mapping[str, Any], destination: bytearray, max_inflight_get: int
    ) -> None:
        chunks = spec["chunks"]
        offsets = _chunk_offsets(chunks)
        if self._read_chunks_with_batch_get_into(chunks, offsets, destination):
            return
        if max_inflight_get == 1 or len(chunks) <= 1:
            for offset, chunk in zip(offsets, chunks):
                self._read_chunk_with_get(chunk, destination, offset)
            return

        with ThreadPoolExecutor(
            max_workers=min(max_inflight_get, len(chunks))
        ) as executor:
            futures = [
                executor.submit(self._read_chunk_with_get, chunk, destination, offset)
                for offset, chunk in zip(offsets, chunks)
            ]
            for future in as_completed(futures):
                future.result()

    def _read_chunks_with_batch_get_into(
        self,
        chunks: Sequence[Mapping[str, Any]],
        offsets: Sequence[int],
        destination: bytearray,
    ) -> bool:
        batch_get_into = getattr(self.store, "batch_get_into", None)
        register_buffer = getattr(self.store, "register_buffer", None)
        unregister_buffer = getattr(self.store, "unregister_buffer", None)
        if not (
            callable(batch_get_into)
            and callable(register_buffer)
            and callable(unregister_buffer)
        ):
            return False
        base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(destination))
        _check_status(
            register_buffer(base_ptr, len(destination)),
            "register_buffer",
            "bundle payload",
        )
        error: Exception | None = None
        try:
            keys = [chunk["key"] for chunk in chunks]
            ptrs = [base_ptr + offset for offset in offsets]
            sizes = [int(chunk["bytes"]) for chunk in chunks]
            read_sizes = batch_get_into(keys, ptrs, sizes)
            if len(read_sizes) != len(sizes):
                raise RuntimeError(
                    f"batch_get_into returned {len(read_sizes)} results for {len(sizes)} chunks"
                )
            for key, expected, actual in zip(keys, sizes, read_sizes):
                if actual != expected:
                    raise RuntimeError(
                        f"batch_get_into failed for {key}: expected {expected}, got {actual}"
                    )
        except Exception as read_error:
            error = read_error
            raise
        finally:
            try:
                _check_status(
                    unregister_buffer(base_ptr), "unregister_buffer", "bundle payload"
                )
            except Exception:
                if error is None:
                    raise
        return True

    def _read_chunk_with_get(
        self, chunk: Mapping[str, Any], destination: bytearray, offset: int
    ) -> None:
        chunk_bytes = int(chunk["bytes"])
        if chunk_bytes == 0:
            return
        data = self.store.get(chunk["key"])
        if len(data) != chunk_bytes:
            raise RuntimeError(
                f"get failed for {chunk['key']}: expected {chunk_bytes} bytes, got {len(data)}"
            )
        destination[offset : offset + chunk_bytes] = data

    def _resolve_policy(
        self,
        policy: Optional[BundleTransferPolicy],
        max_inflight_put: Optional[int] = None,
        max_inflight_get: Optional[int] = None,
    ) -> BundleTransferPolicy:
        result = policy or BundleTransferPolicy()
        if max_inflight_put is not None or max_inflight_get is not None:
            result = BundleTransferPolicy(
                max_inflight_put=(
                    max_inflight_put
                    if max_inflight_put is not None
                    else result.max_inflight_put
                ),
                max_inflight_get=(
                    max_inflight_get
                    if max_inflight_get is not None
                    else result.max_inflight_get
                ),
            )
        if result.max_inflight_put < 1:
            raise ValueError("max_inflight_put must be positive")
        if result.max_inflight_get < 1:
            raise ValueError("max_inflight_get must be positive")
        return result

    def _resolve_buffer_names(
        self, manifest: Mapping[str, Any], buffer_names: Optional[Sequence[str]]
    ) -> list[str]:
        buffers = manifest["buffers"]
        if buffer_names is None:
            return list(buffers)
        missing = [name for name in buffer_names if name not in buffers]
        if missing:
            raise KeyError(f"unknown bundle buffers: {missing}")
        return list(buffer_names)

    def _resolve_manifest(
        self, ref: RemoteBundleRef | Mapping[str, Any]
    ) -> dict[str, Any]:
        if isinstance(ref, RemoteBundleRef):
            manifest = ref.manifest
        else:
            manifest = ref.get("manifest")
            if not isinstance(manifest, dict):
                manifest_key = ref.get("manifest_key")
                if not isinstance(manifest_key, str):
                    raise ValueError("bundle ref must include manifest or manifest_key")
                manifest = _decode_manifest(self.store.get(manifest_key))
        self._validate_manifest(manifest)
        return manifest

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("version") != 1 or manifest.get("layout") != "bundle":
            raise ValueError("invalid bundle manifest")
        object_id = manifest.get("object_id")
        if not isinstance(object_id, str):
            raise ValueError("bundle manifest object_id must be a string")
        base_key = f"{self.key_prefix}/{object_id}"
        self._validate_payload_spec(manifest.get("meta"), base_key)
        buffers = manifest.get("buffers")
        if not isinstance(buffers, dict):
            raise ValueError("bundle manifest buffers must be a dict")
        for name, spec in buffers.items():
            _validate_key_segment(name, "buffer name")
            self._validate_payload_spec(spec, base_key)

    def _validate_payload_spec(self, spec: Any, base_key: str) -> None:
        if not isinstance(spec, dict):
            raise ValueError("bundle payload spec must be a dict")
        expected_bytes = int(spec.get("bytes", -1))
        if expected_bytes < 0:
            raise ValueError("bundle payload bytes must be non-negative")
        chunks = spec.get("chunks")
        if not isinstance(chunks, list) or (expected_bytes and not chunks):
            raise ValueError("bundle payload chunks are invalid")
        seen_keys = set()
        total_bytes = 0
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("bundle chunk must be a dict")
            key = chunk.get("key")
            chunk_bytes = int(chunk.get("bytes", -1))
            if not isinstance(key, str) or not key.startswith(f"{base_key}/"):
                raise ValueError("bundle chunk key is outside the bundle namespace")
            if key in seen_keys:
                raise ValueError("bundle chunk keys must be unique")
            if chunk_bytes < 0:
                raise ValueError("bundle chunk bytes must be non-negative")
            seen_keys.add(key)
            total_bytes += chunk_bytes
        if total_bytes != expected_bytes:
            raise ValueError(
                f"bundle payload chunks total {total_bytes} bytes, expected {expected_bytes}"
            )

    def _manifest_key(
        self, ref: RemoteBundleRef | Mapping[str, Any], manifest: Mapping[str, Any]
    ) -> Optional[str]:
        if isinstance(ref, RemoteBundleRef):
            return ref.manifest_key
        manifest_key = ref.get("manifest_key")
        if isinstance(manifest_key, str):
            return manifest_key
        return f"{self.key_prefix}/{manifest['object_id']}/manifest"

    def _payload_keys(self, manifest: Mapping[str, Any]) -> list[str]:
        keys = self._spec_keys(manifest["meta"])
        for spec in manifest["buffers"].values():
            keys.extend(self._spec_keys(spec))
        return keys

    def _spec_keys(self, spec: Mapping[str, Any]) -> list[str]:
        return [chunk["key"] for chunk in spec["chunks"]]


def _payload_to_view(value: Any, name: str) -> memoryview:
    if isinstance(value, np.ndarray):
        return _bytes_view(np.ascontiguousarray(value).view(np.uint8).reshape(-1), name)
    return _bytes_view(value, name)


def _bytes_view(value: Any, name: str) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as error:
        raise TypeError(
            f"{name} must be bytes-like, got {type(value).__name__}"
        ) from error
    if not view.contiguous:
        view = memoryview(bytes(view))
    return view.cast("B")


def _split_view(view: memoryview, chunk_bytes: int) -> list[memoryview]:
    if len(view) == 0:
        return [view]
    return [
        view[start : start + chunk_bytes] for start in range(0, len(view), chunk_bytes)
    ]


def _chunk_offsets(chunks: Sequence[Mapping[str, Any]]) -> list[int]:
    offsets = [0]
    for chunk in chunks[:-1]:
        offsets.append(offsets[-1] + int(chunk["bytes"]))
    return offsets


def _resolve_chunk_bytes(chunk_bytes: Optional[int]) -> int:
    result = DEFAULT_BUNDLE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes
    if result <= 0:
        raise ValueError("chunk_bytes must be positive")
    return result


def _normalize_key_prefix(key_prefix: str) -> str:
    prefix = key_prefix.strip("/")
    if not prefix or any(ord(char) < 32 for char in prefix):
        raise ValueError("key_prefix must be a non-empty key prefix")
    return prefix


def _validate_key_segment(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or "/" in value
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"invalid bundle {name}: {value!r}")


def _encode_manifest(manifest: Mapping[str, Any]) -> bytes:
    return json.dumps(manifest, separators=(",", ":")).encode("utf-8")


def _decode_manifest(payload: bytes) -> dict[str, Any]:
    manifest = json.loads(payload.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("bundle manifest must be a dict")
    return manifest


def _check_status(status: Any, operation: str, key: str) -> None:
    if status not in (None, 0):
        raise RuntimeError(f"{operation} failed for {key}: {status}")


def _cleanup_keys(store: BundleStore, keys: Sequence[str], strict: bool) -> None:
    errors = []
    unique_keys = list(dict.fromkeys(keys))
    batch_remove = getattr(store, "batch_remove", None)
    if callable(batch_remove) and unique_keys:
        try:
            results = batch_remove(unique_keys)
        except Exception:
            if strict:
                raise
            return
        errors = [
            (key, status)
            for key, status in zip(unique_keys, results)
            if status not in (None, 0, MISSING_OBJECT_ERROR)
        ]
    else:
        for key in unique_keys:
            try:
                status = store.remove(key, True)
            except KeyError:
                continue
            except Exception:
                if strict:
                    raise
                continue
            if status not in (None, 0, MISSING_OBJECT_ERROR):
                errors.append((key, status))
    if errors and strict:
        raise RuntimeError(
            f"failed to remove {len(errors)} Mooncake keys: {errors[:3]}"
        )

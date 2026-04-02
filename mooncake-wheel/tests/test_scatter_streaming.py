"""Smoke / integration tests for the streaming scatter read Python API.

Covers:
  1. Basic happy-path: put objects, streaming_batch_get_buffer_ranges, wait_all
  2. Per-chunk wait path: wait_chunk / is_chunk_ready / completed_count
  3. Invalid input (length mismatch) -> None
  4. DummyClient unsupported -> None
"""

import ctypes
import os
import time
import unittest

from mooncake.store import MooncakeDistributedStore

# The lease time of the kv object, should be set equal to
# the master's value.
DEFAULT_DEFAULT_KV_LEASE_TTL = 5000  # 5000 milliseconds
# Use environment variable if set, otherwise use default
default_kv_lease_ttl = int(
    os.getenv("DEFAULT_KV_LEASE_TTL", DEFAULT_DEFAULT_KV_LEASE_TTL)
)


def get_client(store, local_buffer_size_param=None):
    """Initialize and setup the distributed store client."""
    protocol = os.getenv("PROTOCOL", "tcp")
    device_name = os.getenv("DEVICE_NAME", "ibp6s0")
    local_hostname = os.getenv("LOCAL_HOSTNAME", "localhost")
    metadata_server = os.getenv(
        "MC_METADATA_SERVER", "http://127.0.0.1:8080/metadata"
    )
    global_segment_size = 3200 * 1024 * 1024  # 3200 MB
    local_buffer_size = (
        local_buffer_size_param
        if local_buffer_size_param is not None
        else 512 * 1024 * 1024  # 512 MB
    )
    master_server_address = os.getenv("MASTER_SERVER", "127.0.0.1:50051")

    retcode = store.setup(
        local_hostname,
        metadata_server,
        global_segment_size,
        local_buffer_size,
        protocol,
        device_name,
        master_server_address,
    )

    if retcode:
        raise RuntimeError(
            f"Failed to setup store client. Return code: {retcode}"
        )


class TestScatterStreamingBasic(unittest.TestCase):
    """Tests for streaming_batch_get_buffer_ranges with a real client."""

    @classmethod
    def setUpClass(cls):
        cls.store = MooncakeDistributedStore()
        get_client(cls.store)

    # ------------------------------------------------------------------
    # 1. Happy-path: put two objects, scatter-read ranges, wait_all
    # ------------------------------------------------------------------
    def test_basic_scatter_streaming(self):
        """Put two objects, scatter-read specific ranges, verify data."""
        obj_a = b"A" * 4096
        obj_b = b"B" * 4096
        key_a = "scatter_stream_a"
        key_b = "scatter_stream_b"

        self.assertEqual(self.store.put(key_a, obj_a), 0)
        self.assertEqual(self.store.put(key_b, obj_b), 0)

        # Allocate a destination buffer large enough for two 1024-byte ranges
        buf_size = 2048
        buf = (ctypes.c_ubyte * buf_size)()
        buf_ptr = ctypes.addressof(buf)
        self.assertEqual(self.store.register_buffer(buf_ptr, buf_size), 0)

        # Read 1024 bytes from each object:
        #   key_a[0..1024) -> buf[0..1024)
        #   key_b[0..1024) -> buf[1024..2048)
        handle = self.store.streaming_batch_get_buffer_ranges(
            keys=[key_a, key_b],
            buffer=buf_ptr,
            dest_offsets=[0, 1024],
            src_offsets=[0, 0],
            sizes=[1024, 1024],
        )
        self.assertIsNotNone(handle, "Handle should not be None")

        # Wait for all chunks to finish
        rc = handle.wait_all()
        self.assertEqual(rc, 0, "wait_all should return OK (0)")

        # Verify data
        self.assertEqual(bytes(buf[0:1024]), b"A" * 1024)
        self.assertEqual(bytes(buf[1024:2048]), b"B" * 1024)

        # Cleanup
        self.assertEqual(self.store.unregister_buffer(buf_ptr), 0)
        time.sleep(default_kv_lease_ttl / 1000)
        self.assertEqual(self.store.remove(key_a), 0)
        self.assertEqual(self.store.remove(key_b), 0)

    # ------------------------------------------------------------------
    # 2. Per-chunk wait path
    # ------------------------------------------------------------------
    def test_per_chunk_wait(self):
        """Exercise wait_chunk, is_chunk_ready, completed_count."""
        obj = b"C" * 8192
        key = "scatter_stream_chunk"

        self.assertEqual(self.store.put(key, obj), 0)

        buf_size = 4096
        buf = (ctypes.c_ubyte * buf_size)()
        buf_ptr = ctypes.addressof(buf)
        self.assertEqual(self.store.register_buffer(buf_ptr, buf_size), 0)

        handle = self.store.streaming_batch_get_buffer_ranges(
            keys=[key, key],
            buffer=buf_ptr,
            dest_offsets=[0, 2048],
            src_offsets=[0, 4096],
            sizes=[2048, 2048],
        )
        self.assertIsNotNone(handle)

        num = handle.num_chunks
        self.assertGreater(num, 0, "num_chunks should be > 0")

        # Wait on every chunk individually
        for i in range(num):
            rc = handle.wait_chunk(i)
            self.assertEqual(rc, 0, f"wait_chunk({i}) should succeed")
            self.assertTrue(handle.is_chunk_ready(i))

        self.assertEqual(handle.completed_count(), num)

        # Verify data
        self.assertEqual(bytes(buf[0:2048]), b"C" * 2048)
        self.assertEqual(bytes(buf[2048:4096]), b"C" * 2048)

        self.assertEqual(self.store.unregister_buffer(buf_ptr), 0)
        time.sleep(default_kv_lease_ttl / 1000)
        self.assertEqual(self.store.remove(key), 0)

    # ------------------------------------------------------------------
    # 3. Invalid input: length mismatch -> None
    # ------------------------------------------------------------------
    def test_length_mismatch_returns_none(self):
        """Lists of different lengths should cause None return."""
        buf_size = 1024
        buf = (ctypes.c_ubyte * buf_size)()
        buf_ptr = ctypes.addressof(buf)
        self.assertEqual(self.store.register_buffer(buf_ptr, buf_size), 0)

        handle = self.store.streaming_batch_get_buffer_ranges(
            keys=["k1", "k2"],
            buffer=buf_ptr,
            dest_offsets=[0],  # length 1, mismatch
            src_offsets=[0, 0],
            sizes=[512, 512],
        )
        self.assertIsNone(handle, "Mismatched input lengths should return None")

        self.assertEqual(self.store.unregister_buffer(buf_ptr), 0)


class TestScatterStreamingDummy(unittest.TestCase):
    """DummyClient should return None for streaming scatter."""

    @classmethod
    def setUpClass(cls):
        cls.store = MooncakeDistributedStore()
        retcode = cls.store.setup_dummy(
            64 * 1024 * 1024,  # mem_pool_size
            16 * 1024 * 1024,  # local_buffer_size
            "127.0.0.1:50052",
        )
        if retcode:
            raise RuntimeError(
                f"Failed to setup dummy client. Return code: {retcode}"
            )

    def test_dummy_returns_none(self):
        """streaming_batch_get_buffer_ranges is unsupported on DummyClient."""
        buf_size = 1024
        buf = (ctypes.c_ubyte * buf_size)()
        buf_ptr = ctypes.addressof(buf)

        handle = self.store.streaming_batch_get_buffer_ranges(
            keys=["k"],
            buffer=buf_ptr,
            dest_offsets=[0],
            src_offsets=[0],
            sizes=[512],
        )
        self.assertIsNone(handle, "DummyClient should return None")


if __name__ == "__main__":
    unittest.main()

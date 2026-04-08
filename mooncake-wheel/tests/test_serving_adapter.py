import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mooncake-wheel"))
sys.path.insert(1, str(ROOT / "build-phase-a-check" / "mooncake-integration"))

store_so = ROOT / "build-phase-a-check" / "mooncake-integration" / "store.cpython-312-x86_64-linux-gnu.so"
package_store_so = ROOT / "mooncake-wheel" / "mooncake" / "store.so"
if store_so.exists() and package_store_so.exists():
    package_store_so.unlink()
    package_store_so.symlink_to(store_so)

from mooncake.serving_adapter import (
    MooncakeServingDefaults,
    MooncakeServingStore,
    build_canonical_object_key,
    resolve_replicate_config,
)
from mooncake.mooncake_store_service import MooncakeStoreService


class TestServingAdapter(unittest.TestCase):
    def test_build_canonical_object_key(self):
        self.assertEqual(
            build_canonical_object_key("tenant-a", "domain-a", "set-a", "logical-a"),
            "tenant-a/domain-a/set-a/logical-a",
        )

    def test_resolve_replicate_config_with_defaults(self):
        defaults = MooncakeServingDefaults(
            tenant_id="tenant-a",
            domain_id="domain-a",
            object_set="set-a",
            qos_tier="gold",
            sharing_scope="scope-a",
        )

        config = resolve_replicate_config(logical_key="logical-a", defaults=defaults)

        self.assertEqual(config.tenant_id, "tenant-a")
        self.assertEqual(config.domain_id, "domain-a")
        self.assertEqual(config.object_set, "set-a")
        self.assertEqual(config.qos_tier, "gold")
        self.assertEqual(config.sharing_scope, "scope-a")
        self.assertEqual(config.logical_key, "logical-a")
        self.assertEqual(config.canonical_key, "tenant-a/domain-a/set-a/logical-a")

    def test_request_values_override_defaults(self):
        defaults = MooncakeServingDefaults(
            tenant_id="tenant-a",
            domain_id="domain-a",
            object_set="set-a",
            qos_tier="gold",
            sharing_scope="scope-a",
        )

        config = resolve_replicate_config(
            logical_key="logical-a",
            defaults=defaults,
            tenant_id="tenant-b",
            object_set="set-b",
            sharing_scope="scope-b",
            qos_tier="silver",
        )

        self.assertEqual(config.tenant_id, "tenant-b")
        self.assertEqual(config.domain_id, "domain-a")
        self.assertEqual(config.object_set, "set-b")
        self.assertEqual(config.qos_tier, "silver")
        self.assertEqual(config.sharing_scope, "scope-b")
        self.assertEqual(config.canonical_key, "tenant-b/domain-a/set-b/logical-a")

    def test_cache_salt_fills_sharing_scope(self):
        config = resolve_replicate_config(
            logical_key="logical-a",
            cache_salt="salt-a",
        )

        self.assertEqual(config.sharing_scope, "salt-a")
        self.assertEqual(config.tenant_id, "default")
        self.assertEqual(config.domain_id, "default")
        self.assertEqual(config.object_set, "default")
        self.assertEqual(config.canonical_key, "default/default/default/logical-a")

    def test_explicit_canonical_key_wins(self):
        config = resolve_replicate_config(
            logical_key="logical-a",
            canonical_key="manual/key",
        )

        self.assertEqual(config.canonical_key, "manual/key")

    def test_serving_store_put_with_context(self):
        raw_store = Mock()
        raw_store.put.return_value = 0
        serving_store = MooncakeServingStore(
            raw_store,
            MooncakeServingDefaults(tenant_id="tenant-a", domain_id="domain-a", object_set="set-a"),
        )

        ret = serving_store.put_with_context("raw-key", b"value", logical_key="logical-a")

        self.assertEqual(ret, 0)
        raw_store.put.assert_called_once()
        _, _, config = raw_store.put.call_args[0]
        self.assertEqual(config.logical_key, "logical-a")
        self.assertEqual(config.canonical_key, "tenant-a/domain-a/set-a/logical-a")


class FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class TestMooncakeStoreServicePut(unittest.IsolatedAsyncioTestCase):
    async def test_handle_put_without_metadata_uses_raw_store(self):
        service = MooncakeStoreService.__new__(MooncakeStoreService)
        service.store = Mock()
        service.store.put.return_value = 0
        service.serving_store = Mock()

        response = await service.handle_put(FakeRequest({"key": "k", "value": "v"}))

        self.assertEqual(response.status, 200)
        service.store.put.assert_called_once_with("k", b"v")
        service.serving_store.put_with_context.assert_not_called()

    async def test_handle_put_with_metadata_uses_serving_store(self):
        service = MooncakeStoreService.__new__(MooncakeStoreService)
        service.store = Mock()
        service.serving_store = Mock()
        service.serving_store.put_with_context.return_value = 0

        response = await service.handle_put(
            FakeRequest(
                {
                    "key": "k",
                    "value": "v",
                    "metadata": {
                        "logical_key": "logical-k",
                        "tenant_id": "tenant-a",
                        "cache_salt": "salt-a",
                    },
                }
            )
        )

        self.assertEqual(response.status, 200)
        service.serving_store.put_with_context.assert_called_once_with(
            "k",
            b"v",
            logical_key="logical-k",
            tenant_id="tenant-a",
            domain_id=None,
            object_set=None,
            qos_tier=None,
            sharing_scope=None,
            cache_salt="salt-a",
            canonical_key=None,
        )
        service.store.put.assert_not_called()

    async def test_handle_put_rejects_invalid_metadata_shape(self):
        service = MooncakeStoreService.__new__(MooncakeStoreService)
        service.store = Mock()
        service.serving_store = Mock()

        response = await service.handle_put(
            FakeRequest({"key": "k", "value": "v", "metadata": "bad"})
        )

        self.assertEqual(response.status, 400)
        body = json.loads(response.text)
        self.assertEqual(body["error"], "metadata must be an object")


if __name__ == "__main__":
    unittest.main()

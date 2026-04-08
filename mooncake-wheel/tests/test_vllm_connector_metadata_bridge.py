import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mooncake-wheel"))

from mooncake.serving_adapter import normalize_vllm_serving_metadata


class TestVllmConnectorMetadataBridge(unittest.TestCase):
    def test_promotes_top_level_metadata_fields(self):
        params = {
            "do_remote_decode": True,
            "cache_salt": "salt-a",
            "tenant_id": "tenant-a",
        }

        normalized = normalize_vllm_serving_metadata(params, "req-1")

        self.assertEqual(normalized["mooncake_metadata"]["cache_salt"], "salt-a")
        self.assertEqual(normalized["mooncake_metadata"]["tenant_id"], "tenant-a")
        self.assertEqual(normalized["mooncake_metadata"]["logical_key"], "req-1")

    def test_preserves_explicit_metadata_values(self):
        params = {
            "cache_salt": "salt-a",
            "mooncake_metadata": {
                "cache_salt": "salt-b",
                "logical_key": "logical-a",
            },
        }

        normalized = normalize_vllm_serving_metadata(params, "req-1")

        self.assertEqual(normalized["mooncake_metadata"]["cache_salt"], "salt-b")
        self.assertEqual(normalized["mooncake_metadata"]["logical_key"], "logical-a")

    def test_ignores_non_dict_metadata(self):
        params = {"mooncake_metadata": "bad"}

        normalized = normalize_vllm_serving_metadata(params, "req-1")

        self.assertEqual(normalized["mooncake_metadata"]["logical_key"], "req-1")


if __name__ == "__main__":
    unittest.main()

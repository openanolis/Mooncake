from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from mooncake.store import MooncakeDistributedStore, ReplicateConfig


DEFAULT_TENANT_ID = "default"
DEFAULT_DOMAIN_ID = "default"
DEFAULT_OBJECT_SET = "default"
DEFAULT_QOS_TIER = "default"


@dataclass(frozen=True)
class MooncakeServingDefaults:
    tenant_id: Optional[str] = None
    domain_id: Optional[str] = None
    object_set: Optional[str] = None
    qos_tier: Optional[str] = None
    sharing_scope: Optional[str] = None

    @classmethod
    def from_mapping(cls, values: Optional[dict[str, Any]]) -> "MooncakeServingDefaults":
        if not values:
            return cls()
        return cls(
            tenant_id=_normalize_optional_string(values.get("tenant_id")),
            domain_id=_normalize_optional_string(values.get("domain_id")),
            object_set=_normalize_optional_string(values.get("object_set")),
            qos_tier=_normalize_optional_string(values.get("qos_tier")),
            sharing_scope=_normalize_optional_string(values.get("sharing_scope")),
        )


def _normalize_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _get_config_attr(config: ReplicateConfig, name: str, fallback: str) -> str:
    value = getattr(config, name, fallback)
    return value if value is not None else fallback


def _set_config_attr(config: ReplicateConfig, name: str, value: str) -> None:
    if hasattr(config, name):
        setattr(config, name, value)


def build_canonical_object_key(
    tenant_id: str,
    domain_id: str,
    object_set: str,
    logical_key: str,
) -> str:
    return f"{tenant_id}/{domain_id}/{object_set}/{logical_key}"


VLLM_METADATA_FIELDS = (
    "tenant_id",
    "domain_id",
    "object_set",
    "qos_tier",
    "sharing_scope",
    "cache_salt",
    "logical_key",
    "canonical_key",
)


def normalize_vllm_serving_metadata(
    params: Optional[dict[str, Any]], request_id: str
) -> dict[str, Any]:
    normalized = deepcopy(params) if params else {}
    metadata = normalized.get("mooncake_metadata")
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    for field in VLLM_METADATA_FIELDS:
        value = normalized.get(field)
        if value is not None and metadata.get(field) is None:
            metadata[field] = value

    if metadata.get("logical_key") is None:
        metadata["logical_key"] = request_id

    if metadata:
        normalized["mooncake_metadata"] = metadata
    return normalized


def resolve_replicate_config(
    *,
    logical_key: str,
    defaults: Optional[MooncakeServingDefaults] = None,
    config: Optional[ReplicateConfig] = None,
    tenant_id: Optional[str] = None,
    domain_id: Optional[str] = None,
    object_set: Optional[str] = None,
    qos_tier: Optional[str] = None,
    sharing_scope: Optional[str] = None,
    cache_salt: Optional[str] = None,
    canonical_key: Optional[str] = None,
) -> ReplicateConfig:
    resolved = config if config is not None else ReplicateConfig()
    defaults = defaults or MooncakeServingDefaults()

    resolved_tenant_id = _resolve_string(
        primary=tenant_id,
        secondary=defaults.tenant_id,
        fallback=_get_config_attr(resolved, "tenant_id", DEFAULT_TENANT_ID),
    )
    resolved_domain_id = _resolve_string(
        primary=domain_id,
        secondary=defaults.domain_id,
        fallback=_get_config_attr(resolved, "domain_id", DEFAULT_DOMAIN_ID),
    )
    resolved_object_set = _resolve_string(
        primary=object_set,
        secondary=defaults.object_set,
        fallback=_get_config_attr(resolved, "object_set", DEFAULT_OBJECT_SET),
    )
    resolved_qos_tier = _resolve_string(
        primary=qos_tier,
        secondary=defaults.qos_tier,
        fallback=_get_config_attr(resolved, "qos_tier", DEFAULT_QOS_TIER),
    )
    resolved_sharing_scope = _resolve_string(
        primary=sharing_scope,
        secondary=cache_salt,
        fallback=(
            defaults.sharing_scope
            if defaults.sharing_scope is not None
            else _get_config_attr(resolved, "sharing_scope", "")
        ),
    )

    _set_config_attr(resolved, "tenant_id", resolved_tenant_id)
    _set_config_attr(resolved, "domain_id", resolved_domain_id)
    _set_config_attr(resolved, "object_set", resolved_object_set)
    _set_config_attr(resolved, "qos_tier", resolved_qos_tier)
    _set_config_attr(resolved, "logical_key", logical_key)
    _set_config_attr(resolved, "sharing_scope", resolved_sharing_scope)

    resolved_canonical_key = canonical_key or build_canonical_object_key(
        resolved_tenant_id,
        resolved_domain_id,
        resolved_object_set,
        logical_key,
    )
    _set_config_attr(resolved, "canonical_key", resolved_canonical_key)
    return resolved


def _resolve_string(*, primary: Optional[str], secondary: Optional[str], fallback: str) -> str:
    if primary is not None:
        return primary
    if secondary is not None:
        return secondary
    return fallback


class MooncakeServingStore:
    def __init__(
        self,
        store: MooncakeDistributedStore,
        defaults: Optional[MooncakeServingDefaults] = None,
    ):
        self.store = store
        self.defaults = defaults or MooncakeServingDefaults()

    def _resolve_config(
        self,
        *,
        key: str,
        logical_key: Optional[str] = None,
        config: Optional[ReplicateConfig] = None,
        tenant_id: Optional[str] = None,
        domain_id: Optional[str] = None,
        object_set: Optional[str] = None,
        qos_tier: Optional[str] = None,
        sharing_scope: Optional[str] = None,
        cache_salt: Optional[str] = None,
        canonical_key: Optional[str] = None,
    ) -> ReplicateConfig:
        return resolve_replicate_config(
            logical_key=logical_key or key,
            defaults=self.defaults,
            config=config,
            tenant_id=tenant_id,
            domain_id=domain_id,
            object_set=object_set,
            qos_tier=qos_tier,
            sharing_scope=sharing_scope,
            cache_salt=cache_salt,
            canonical_key=canonical_key,
        )

    def put_with_context(self, key: str, value: bytes, **kwargs) -> int:
        config = self._resolve_config(key=key, **kwargs)
        return self.store.put(key, value, config)

    def upsert_with_context(self, key: str, value: bytes, **kwargs) -> int:
        config = self._resolve_config(key=key, **kwargs)
        return self.store.upsert(key, value, config)

    def put_from_with_context(self, key: str, buffer_ptr: int, size: int, **kwargs) -> int:
        config = self._resolve_config(key=key, **kwargs)
        return self.store.put_from(key, buffer_ptr, size, config)

    def pub_tensor_with_context(self, key: str, tensor, **kwargs) -> int:
        config = self._resolve_config(key=key, **kwargs)
        return self.store.pub_tensor(key, tensor, config)

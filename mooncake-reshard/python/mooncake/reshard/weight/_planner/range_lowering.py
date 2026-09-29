"""Lower logical store-load segments into efficient ranged-read requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ...contracts import RuntimeFragmentId
from ..._typing import TypeAlias
from ..manifest import RuntimeBindingFragment

if TYPE_CHECKING:
    from .contracts import StoredLoadOperation


StoreReadRange: TypeAlias = tuple[RuntimeBindingFragment, str, int, int, int]


def coalesce_adjacent_store_ranges(
    ranges: Sequence[StoreReadRange],
    *,
    max_range_bytes: int,
) -> tuple[StoreReadRange, ...]:
    """Merge exactly adjacent ranges without changing read semantics.

    The merge is deliberately done while lowering a logical weight plan. It
    only combines requests that address the same runtime fragment and Store
    object, with both source and destination offsets contiguous. No gap,
    object, fragment, or max-range boundary may be crossed.
    """

    if type(max_range_bytes) is not int or max_range_bytes <= 0:
        raise ValueError("max_range_bytes must be a positive integer")

    lowered: list[StoreReadRange] = []
    for target, key, target_offset, source_offset, nbytes in ranges:
        if not isinstance(target, RuntimeBindingFragment):
            raise ValueError("store read range target is invalid")
        if type(key) is not str or not key:
            raise ValueError("store read range object key is invalid")
        if (
            type(target_offset) is not int
            or type(source_offset) is not int
            or type(nbytes) is not int
            or target_offset < 0
            or source_offset < 0
            or nbytes <= 0
        ):
            raise ValueError("store read range geometry is invalid")
        if nbytes > max_range_bytes:
            raise ValueError(
                "store read range exceeds max_range_bytes: "
                f"{nbytes} > {max_range_bytes}"
            )

        if lowered:
            (
                previous_target,
                previous_key,
                previous_target_offset,
                previous_source_offset,
                previous_size,
            ) = lowered[-1]
            if (
                previous_target == target
                and previous_key == key
                and previous_target_offset + previous_size == target_offset
                and previous_source_offset + previous_size == source_offset
                and previous_size + nbytes <= max_range_bytes
            ):
                lowered[-1] = (
                    previous_target,
                    previous_key,
                    previous_target_offset,
                    previous_source_offset,
                    previous_size + nbytes,
                )
                continue
        lowered.append((target, key, target_offset, source_offset, nbytes))

    return tuple(lowered)


def lower_stored_load_ranges(
    operations_by_target: Mapping[
        RuntimeFragmentId, Sequence["StoredLoadOperation"]
    ],
    targets: Mapping[RuntimeFragmentId, RuntimeBindingFragment],
    *,
    max_segments: int,
    max_range_bytes: int,
) -> tuple[StoreReadRange, ...]:
    """Lower stored-load operations before request-size batching.

    Request-size limits belong to the backend, but the decision about which
    logical segments can become one physical read belongs to the planner.  In
    particular, coalescing must happen *before* applying the per-request range
    limit; otherwise a long run of adjacent segments can be split into several
    Store calls based on the pre-coalesced count.
    """

    if type(max_segments) is not int or max_segments <= 0:
        raise ValueError("max_segments must be a positive integer")
    if type(max_range_bytes) is not int or max_range_bytes <= 0:
        raise ValueError("max_range_bytes must be a positive integer")

    lowered: list[StoreReadRange] = []
    for fragment_id in sorted(operations_by_target):
        target = targets.get(fragment_id)
        if target is None:
            raise ValueError(f"missing runtime target fragment: {fragment_id}")

        operations = sorted(
            operations_by_target[fragment_id],
            key=lambda operation: (
                operation.target_offset,
                operation.source.object_key,
                operation.source_offset,
            ),
        )
        for operation in operations:
            if operation.target.fragment_id != fragment_id:
                raise ValueError(
                    f"stored-load target mismatch: {operation.target.fragment_id}"
                )
            for source_offset, target_offset, nbytes in operation.iter_segments(
                max_segments=max_segments
            ):
                for chunk_offset in range(0, nbytes, max_range_bytes):
                    chunk_size = min(max_range_bytes, nbytes - chunk_offset)
                    lowered.append(
                        (
                            target,
                            operation.source.object_key,
                            target_offset + chunk_offset,
                            operation.source.object_offset
                            + source_offset
                            + chunk_offset,
                            chunk_size,
                        )
                    )

    # Operations are sorted independently per target, and an N-D lowering can
    # emit physical ranges from interleaved source fragments.  Normalize the
    # complete physical stream before coalescing so adjacency is determined by
    # the final target/object/source geometry rather than operation traversal
    # order.
    lowered.sort(
        key=lambda item: (
            item[0].fragment_id,
            item[2],
            item[1],
            item[3],
        )
    )
    return coalesce_adjacent_store_ranges(
        lowered,
        max_range_bytes=max_range_bytes,
    )


__all__ = [
    "StoreReadRange",
    "coalesce_adjacent_store_ranges",
    "lower_stored_load_ranges",
]

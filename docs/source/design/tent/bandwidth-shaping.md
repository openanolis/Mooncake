# TENT Runtime Bandwidth Shaping Design

## Status

This design is now implemented in TENT runtime as the phase-1/1.5 baseline and extended with transport-aware pacing hints, hierarchical shaping keys, and adaptive closed-loop controls in the runtime shaper.

## Background

TENT already supports QoS normalization, weighted-fair request ordering, and per-tenant inflight gating. Those mechanisms decide **who runs first** and **who must wait**, but they do not yet decide **how many bytes per unit time** each tenant is allowed to put onto the datapath.

The runtime shaper now adds that byte-rate control without turning fairness into a throughput tax.

This design targets four requirements:

1. **Single active transfer must not be rate-limited by default.**
2. **Fairness must not reduce aggregate throughput or leave bandwidth idle.**
3. **The implementation must work across multiple transports, not only TCP.**
4. **The design should remain robust even when physical link bandwidth is unknown.**

## Goals

- Add real per-tenant bandwidth shaping to TENT.
- Keep the mechanism transport-agnostic in the first phase.
- Preserve the existing QoS layers:
  - weighted-fair ordering
  - per-tenant inflight gating
  - transport/runtime failure handling
- Ensure the system is **work-conserving**: if useful work can be sent, the shaper must not block total throughput.
- Allow future refinement for RDMA posting-level pacing without making it a phase-1 requirement.

## Non-Goals

- Do not rely on Linux kernel traffic control as the main implementation.
- Do not introduce a transport-specific scheduler per backend in phase 1.
- Do not build a multi-level shaping tree (`tenant -> domain -> object_set`) in phase 1.
- Do not require prior knowledge of physical link bandwidth.

## Why runtime shaping instead of transport-local shaping

The strongest tenant semantics exist in TENT runtime:

- `Request::qos_context` already carries `tenant_id`, `tenant_shares`, and `qos_tier`.
- `TransferEngineImpl` already owns:
  - request normalization
  - weighted-fair ordering
  - per-tenant pending queues
  - inflight gating
  - transport handoff

By contrast, once requests enter transport-specific internals, especially RDMA worker and endpoint code, tenant identity becomes weaker and transport-specific control becomes much more invasive.

For that reason, phase 1 places shaping in `TransferEngineImpl`, immediately before transport submission.

## Linux-inspired control model

This design borrows the **control model** from Linux scheduling and bandwidth control, but not the Linux implementation boundary.

The three key ideas are:

1. **Relative fairness by shares**
   - Similar to Linux scheduler weights / cgroup shares.
   - `tenant_shares` affect how active tenants divide bandwidth under contention.

2. **Absolute control by token bucket**
   - Similar to token bucket / HTB style sustained rate + burst semantics.
   - Used to enforce byte-rate control when fairness must be materialized into real bandwidth behavior.

3. **Work-conserving scheduling**
   - Similar to DRR-style byte-fair scheduling.
   - If one tenant cannot use its ideal share, the remaining capacity is immediately available to others.

The combination is:

- **shares** decide relative entitlement under contention
- **token budget** decides current byte admission
- **work-conserving drain** ensures throughput is not sacrificed for fairness aesthetics

## Hard behavioral constraints

### 1. Lone active transfer must not be rate-limited

When there is only one active tenant with backlog, the system must not slow it down by default.

This means:

- shaping engages only when there is real contention among multiple active tenants, or
- an explicit hard-cap policy is configured for a tenant/tier and that policy is intended to apply even without contention

Default behavior is therefore:

- **single active tenant -> bypass fairness shaping**
- **multiple active tenants -> apply fairness shaping**

### 2. Fairness must not reduce aggregate throughput

Fairness is a redistribution policy, not a throughput limiter.

The design must therefore be **work-conserving**:

- idle capacity must never be reserved for inactive tenants
- if one tenant lacks backlog or cannot consume its ideal share, other tenants must be allowed to use that capacity immediately
- the scheduler must not wait for lagging tenants just to preserve a clean share ratio snapshot

### 3. Active-set fairness only

Fairness is computed only across the **currently active tenants**:

- tenants with backlog participate
- idle tenants do not reserve share
- active-set membership can change dynamically as workloads change

This avoids static reservation and improves robustness when the actual contention set is unknown ahead of time.

## Black-box robustness requirement

The system should remain effective even when it does not know the true physical bandwidth of the path or machine.

This is important because:

- transport backends differ widely (TCP, RDMA, SHM, io_uring, GDS, NVLink, etc.)
- path bandwidth can change due to topology, congestion, retries, or failures
- deployment environments may not provide a stable or trustworthy physical bandwidth value

The design therefore treats total capacity as a **runtime-observed quantity**, not a fixed input.

### Phase 1 approach

Phase 1 does **not** require an explicit physical-bandwidth estimator to function.

Instead it uses:

- work-conserving scheduling
- active-set fairness
- lone-tenant bypass
- dynamic chunk sizing derived from configured effective rate policy

This already provides strong robustness and avoids underutilization.

### Phase 2 direction

A later enhancement can add a black-box bandwidth estimator:

- maintain an observed effective bandwidth estimate `B_est`
- increase offered load while throughput continues rising cleanly
- reduce estimate when retries, backlog, or completion latency indicate congestion
- use `B_est` as the base quantity for share-based division among active tenants

This would produce:

- no need for preconfigured physical bandwidth
- continued ability to approach full link utilization
- better adaptation across different hardware and path conditions

## Control granularity evaluation

Three candidate control granularities were considered.

### Task-level shaping

- **Precision**: low
- **Runtime overhead**: low
- **Complexity**: low
- **Risk**: low
- **Problem**: large requests create large bursts; fairness becomes too coarse

Conclusion: not suitable as the main design.

### Chunk / fragment-level shaping

- **Precision**: medium to high
- **Runtime overhead**: medium
- **Complexity**: medium
- **Risk**: medium
- **Benefit**: controls burst size while staying transport-agnostic

Conclusion: best phase-1 trade-off.

### Transport post / WQE-level shaping

- **Precision**: high
- **Runtime overhead**: high
- **Complexity**: high
- **Risk**: high
- **Problem**: requires deeper transport-specific control and explicit tenant propagation into RDMA internals

Conclusion: phase-2 option, not phase 1.

## Chosen phase-1 granularity

Phase 1 uses **chunk-level shaping**.

Default policy:

- shaping unit is a runtime-generated fragment of a logical transfer request
- transport backends continue to receive ordinary requests
- runtime decides how much of a logical request is allowed to advance at each shaping opportunity

## Dynamic chunk sizing

A fixed chunk size is not ideal because machine and path bandwidth can vary widely.

The design therefore uses **dynamic chunk sizing**, guided by a target scheduling interval.

### Formula

```text
chunk_bytes = clamp(effective_rate_bytes_per_sec * target_interval_sec,
                    min_chunk_bytes,
                    max_chunk_bytes)
```

Where:

- `effective_rate_bytes_per_sec` comes from configured shaping policy in phase 1
- `target_interval_sec` is a scheduling timeslice target, for example 1–2 ms
- `min_chunk_bytes` prevents excessive scheduling overhead
- `max_chunk_bytes` prevents visible burst inflation

### Recommended defaults

Initial design defaults:

- `target_interval_us = 2000`
- `min_chunk_bytes = 256 KiB`
- `max_chunk_bytes = 1 MiB`
- `bandwidth_burst_bytes = 2x to 8x chunk_bytes`

This yields:

- lower-rate tenants get smaller fragments
- higher-rate tenants get larger fragments
- the scheduler adapts better across different machines and transports

## High-level architecture

The phase-1 control stack becomes:

1. **QoS normalization**
2. **Weighted-fair ordering**
3. **Inflight admission gating**
4. **Bandwidth shaping admission**
5. **Transport submission**

The new shaping stage sits inside `TransferEngineImpl` before requests are handed to `transport->submitTransferTasks(...)`.

## Runtime shaping semantics

### Per-tenant state

For each active tenant, runtime maintains shaping state such as:

- current token count
- refill timestamp
- effective rate limit
- burst cap
- share value
- active / pending state

### Task state

Each logical transfer task must track fragment progress, for example:

- logical total length
- remaining bytes
- next offset
- completed bytes
- active fragment count
- waiting-for-bandwidth flag

This is required because transport submissions now represent fragments of a logical task rather than the entire task.

### Admission flow

For each logical task:

1. Check inflight admission.
2. Check tenant bandwidth admission.
3. If bandwidth budget is sufficient:
   - create a fragment request
   - submit that fragment to the chosen transport
4. If budget is insufficient:
   - keep the task in the existing tenant pending queue

### Completion flow

Fragment completion does **not** imply logical task completion.

A logical task becomes terminal only when:

- all bytes have been submitted and completed, and
- there are no remaining active fragments

Only then:

- task status becomes terminal
- inflight credit is released
- pending drain is triggered normally

## Work-conserving drain behavior

The drain logic must preserve total throughput.

This implies:

- if multiple tenants are active, the scheduler tries to honor shares
- if one tenant lacks backlog or cannot consume its share, the scheduler immediately gives useful work to other tenants
- no tenant receives a reserved idle slice of time or bandwidth just because it is entitled in theory

This is the core safeguard that prevents fairness from reducing aggregate throughput.

## Interaction with existing TENT features

### Weighted ordering

Current `QosScheduler::OrderRequests(...)` remains intact.

Its job continues to be:

- decide service order across requests and tenants
- reflect `tenant_shares` in request scheduling order

The new bandwidth shaper materializes byte-rate control after ordering.

### Inflight gating

Current `max_inflight_per_tenant` remains intact.

Its job continues to be:

- prevent one tenant from flooding runtime concurrency slots

The new shaper controls bytes/sec after inflight admission.

### Failure handling and retries

Retries and resubmission paths must preserve shaping correctness:

- budget must not be double-charged on retry
- inflight credit must not be released early
- logical task accounting must remain tied to the full task, not a single fragment attempt

## Why Linux kernel bandwidth control is not the main solution

Linux kernel traffic control can help on TCP/socket paths, but it is not sufficient as the main mechanism here.

Reasons:

- TENT supports many non-TCP transports
- RDMA, SHM, GDS, io_uring, and local copy paths do not share one kernel network control surface
- tenant policy is already expressed in runtime request metadata, not in one universal kernel-visible flow identity

Therefore the design borrows Linux’s **policy model** but keeps enforcement in TENT runtime.

Kernel mechanisms can remain an optional TCP optimization later.

## Config model

Phase 1 adds shaping-specific knobs under `qos/`.

Proposed configuration:

- `qos/scheduler/bandwidth_shaping_enabled`
- `qos/default_tenant_rate_limit_bytes_per_sec`
- `qos/tier_rate_limits/<tier>`
- `qos/scheduler/bandwidth_burst_bytes`
- `qos/scheduler/target_interval_us`
- `qos/scheduler/min_chunk_bytes`
- `qos/scheduler/max_chunk_bytes`

Rules:

- shaping disabled by default
- invalid values are rejected during config validation
- current behavior remains unchanged when shaping is off

## Phase-1 file scope

Primary files:

- `mooncake-transfer-engine/tent/include/tent/runtime/qos_scheduler.h`
- `mooncake-transfer-engine/tent/src/runtime/qos_scheduler.cpp`
- `mooncake-transfer-engine/tent/include/tent/runtime/transfer_engine_impl.h`
- `mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp`
- `mooncake-transfer-engine/tent/include/tent/common/types.h`

Reference-only in phase 1 unless necessary:

- `mooncake-transfer-engine/tent/src/transport/rdma/rdma_transport.cpp`
- `mooncake-transfer-engine/tent/src/transport/rdma/workers.cpp`
- `mooncake-transfer-engine/tent/src/transport/rdma/endpoint.cpp`
- `mooncake-transfer-engine/tent/src/transport/tcp/tcp_transport.cpp`

## Verification plan

### 1. Config tests

Extend config override and validation tests to cover:

- shaping enable/disable
- default rate limit
- tier rate limits
- burst, interval, and chunk bounds
- invalid values

### 2. Runtime shaping tests

Add or extend runtime tests to verify:

- lone active tenant bypasses shaping by default
- multiple active tenants are shaped fairly under contention
- fairness remains work-conserving when one tenant cannot use its ideal share
- large logical requests progress via multiple fragments
- logical task completion occurs only after all fragments complete

### 3. No-regression tests

Verify that when shaping is disabled:

- weighted-fair ordering still behaves the same
- inflight gating still behaves the same
- pending queue drain semantics do not regress

### 4. Transport smoke tests

At minimum, verify fragment requests complete correctly through:

- TCP path
- RDMA path

### 5. End-to-end throughput checks

Use existing throughput surfaces such as `tebench` and runtime metrics to confirm:

- a single active tenant can still drive full throughput
- fairness under contention does not leave obvious capacity idle
- capped or lower-share tenants are suppressed relative to higher-share tenants only when contention exists

## Implemented extensions beyond phase 1

### Adaptive runtime shaping

The runtime shaper now keeps per-hierarchy bandwidth state:

- configured rate limit
- adaptive rate limit
- estimated capacity
- throughput EMA
- completed bytes per control interval
- burst/tokens/refill timestamps

At each control interval the runtime updates the estimate from observed completions, clamps it into the configured min/max range, and derives an adaptive rate that still respects explicit tenant or tier caps.

### Transport-aware pacing

Runtime shaping remains transport-agnostic at admission time, but request fragments now also carry a transport pacing quantum hint. Today this is enforced in two layers:

- runtime fragment sizing clamps each fragment by the transport pacing quantum
- RDMA posting further limits each `submitSlices()` batch so one post wave does not exceed the pacing quantum carried by the fragment request

This keeps transport pacing compatible with the runtime-owned shaper without requiring a separate per-transport scheduler.

### Hierarchical shaping

Hierarchical shaping is implemented by deriving shaping state from `tenant/domain/object_set` scope instead of tenant alone when enabled. This means:

- inflight accounting is isolated per hierarchy key
- pending queues drain per hierarchy key
- bandwidth tokens and adaptive estimates are maintained per hierarchy key
- weighted ordering still uses normalized request metadata, while runtime admission enforces the hierarchical scope boundary

### Closed-loop QoS

Closed-loop control builds on adaptive shaping by:

- shrinking estimates when observed throughput materially falls below the current estimate
- recovering estimates when useful work is available but capacity appears idle
- preserving work-conserving behavior so idle tenants do not reserve bandwidth

## Remaining future work

The main remaining refinement is deeper transport-native pacing if runtime fragment pacing proves too coarse for a specific backend. In particular, RDMA could still be extended to pace at finer WQE or lane-scheduling granularity, but that is now an optimization pass on top of the existing runtime + endpoint pacing path rather than a prerequisite for correctness.

## Summary

The implemented design adds Linux-inspired bandwidth shaping to TENT runtime without depending on Linux kernel traffic control.

The current system is:

- runtime-owned
- chunk-based
- work-conserving
- adaptive
- hierarchy-aware
- transport-aware via pacing hints and RDMA endpoint post limiting

Most importantly, it preserves two invariants:

- **a lone active transfer is not rate-limited by default**
- **fairness must not reduce aggregate throughput**

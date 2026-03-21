# Transfer Engine Benchmarking & Tuning Guide
## Background

Mooncake’s data transfer backbone is the **Transfer Engine (Mooncake TE)**. Through the adoption of **SGLang**, Mooncake TE is being widely applied across different scenarios and vendors, and user feedback has been increasing.

To support validation and quick prototyping, we provide a **benchmark program** under `mooncake-transfer-engine/example`. This benchmark is intended to:

1. Demonstrate basic usage of the Mooncake TE API.
2. Verify whether Mooncake TE can run in a given environment.
3. Evaluate the performance impact of new patches.

⚠️ **Note:** This benchmark is closer to a *prototype micro-benchmark* than a production-grade tool. Compared with standardized benchmarking frameworks, it has limited functionality and rigor. Results should be interpreted accordingly.

## Usage

The benchmark requires **two processes**:  
- **Target** (must start first)  
- **Initiator** (must start after target)

### Example

- Target:
```bash
./transfer_engine_bench --mode=target --auto_discovery --metadata_server=P2PHANDSHAKE
````

* Initiator (`segment_id` is `target_ip:RPC_port`; RPC port is printed by target, range 15000–20000):
```
Transfer Engine RPC using XXX, listening on YYY:ZZZ
                                                ~~~
```

```bash
./transfer_engine_bench --mode=initiator --auto_discovery \
    --metadata_server=P2PHANDSHAKE --segment_id=node050:15428
```

The initiator will report bandwidth in GB/s.
By default, all NICs are used. To test a single NIC, replace `--auto_discovery` with `--device_name=mlx_XXXX`.

---

## Workflow

1. **Target** allocates 1GB DRAM per NUMA node (configurable via `--buffer_size`) and registers buffers. Then it waits indefinitely.
2. **Initiator** allocates and registers buffers similarly. Data transfers are performed via `submitTransfer`.
3. **Threads**: The initiator spawns multiple threads (default 12, configurable with `--threads`). Each thread:

   * Allocates a batch (`--batch_size`, default 128).
   * Issues read/write requests (`--block_size=65536`).
   * Submits requests with `submitTransfer`.
   * Polls `getTransferStatus` until all complete.
   * Releases batch and repeats.
4. After the specified duration (`--duration=10`), results are collected and printed.

---

## Advanced Parameters

Some advanced tuning parameters are passed via environment variables:

* **`MC_SLICE_SIZE`**: Request slicing granularity (default 65536).
  Larger values reduce CPU overhead but may limit multi-NIC aggregation. Smaller values increase parallelism but add software overhead.

Refer to the Mooncake TE API documentation for additional parameters.

---

## Performance Considerations

* Multiple frontend threads are required to fully utilize bandwidth due to CPU overhead in request preparation.
* `--batch_size`: Large values may cause higher latency if batch completion is gated by slow requests.
* `MC_SLICE_SIZE`: Requires workload-specific tuning for optimal performance.

The next-generation Mooncake TE reduces CPU overhead and achieves near full utilization with fewer threads.

## Grouped Scatter Range Benchmark

The scatter patch also ships a focused validation tool at
`mooncake-transfer-engine/tests/scatter_range_bench.cpp`. It exercises grouped
RDMA `READ` requests that share one `task_group_id`, validates returned bytes,
and emits JSON with `mean`, `p50`, and `p99` latency for each range-count /
range-size pair.

Build it from the Transfer Engine test directory:

```bash
cmake -S . -B build -DWITH_STORE=OFF -DWITH_TE=ON
cmake --build build -j --target scatter_range_bench
```

Run one node as the producer (it registers a remote buffer and prints the base
address), then run the consumer against that server name and remote base:

```bash
# producer
./build/mooncake-transfer-engine/tests/scatter_range_bench \
    --mode=producer \
    --metadata=http://<metadata-host>:<port>/metadata \
    --local=<producer-ip>:24001 \
    --device=<rdma-device>

# consumer
./build/mooncake-transfer-engine/tests/scatter_range_bench \
    --mode=consumer \
    --metadata=http://<metadata-host>:<port>/metadata \
    --local=<consumer-ip>:24002 \
    --peer=<producer-ip>:24001 \
    --remote-base=<buffer_addr_from_producer> \
    --device=<rdma-device> \
    --range-counts=1,2,4,8,16,32,64,128,256 \
    --range-sizes=64,256,1024,4096,16384
```

### Empirical Model

Across grouped RDMA scatter reads, the measured latency can be approximated by
two terms:

```text
control_path_latency(total_kib, range_count)
    = fixed_control_latency
    + per_range_latency * range_count
    + per_kib_latency * total_kib

bandwidth_limited_latency(total_bytes)
    = total_bytes / effective_bandwidth

scatter_read_latency(total_bytes, range_count)
    ~= max(
           control_path_latency(total_bytes / 1024, range_count),
           bandwidth_limited_latency(total_bytes)
       )
       + tail_latency_penalty(range_count)
```

Where:

* `fixed_control_latency` is the fixed submission, polling, and completion
  overhead of one grouped scatter operation.
* `per_range_latency` captures the additional latency introduced by each extra
  range in the grouped request.
* `per_kib_latency` captures the byte-dependent cost in the small/medium
  transfer regime.
* `effective_bandwidth` is the sustained cross-node RDMA bandwidth once the
  request becomes bandwidth-bound.
* `tail_latency_penalty(range_count)` captures extra completion variance when a
  grouped scatter request has to retire many sub-operations. It is workload- and
  run-dependent rather than a fixed cost.

For the cross-node RDMA measurements in this document, the fitted coefficients
are approximately:

* `fixed_control_latency = 33.9` when latency is measured in microseconds
* `per_range_latency = 1.53` microseconds per range
* `per_kib_latency = 0.088` microseconds per KiB
* `effective_bandwidth = 10.4 ~ 11.0 GiB/s`

For the small/medium transfer matrix below, the control-path model fits the
measured mean latency with `R^2 = 0.997`. This is a Transfer Engine level model;
it is not specific to one application such as Engram.

### Cross-Node RDMA Measurements

The following results were collected with grouped RDMA reads over `erdma_0`.

Small and medium transfers (`warmup=20`, `iters=200`):

| range_size \\ range_count | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 B | 38.1 us | 41.4 us | 44.0 us | 49.3 us | 58.8 us | 75.2 us | 131.1 us | 237.8 us | 443.3 us |
| 256 B | 37.6 us | 40.5 us | 43.3 us | 48.2 us | 56.9 us | 72.1 us | 127.6 us | 235.9 us | 443.0 us |
| 1024 B | 38.1 us | 40.8 us | 44.0 us | 49.3 us | 57.6 us | 72.5 us | 129.5 us | 241.1 us | 450.4 us |
| 4096 B | 39.0 us | 41.8 us | 46.0 us | 51.6 us | 61.1 us | 77.2 us | 138.7 us | 257.7 us | 486.7 us |
| 16384 B | 42.8 us | 45.8 us | 52.2 us | 61.6 us | 78.0 us | 127.7 us | 225.6 us | 408.4 us | 795.0 us |

Representative fixed-total-byte comparisons show that range count dominates
latency for small and medium transfers:

| total_bytes | split | mean latency |
| --- | --- | ---: |
| 1024 B | `1 x 1024 B` | 38.1 us |
| 1024 B | `4 x 256 B` | 43.3 us |
| 1024 B | `16 x 64 B` | 58.8 us |
| 16384 B | `1 x 16384 B` | 42.8 us |
| 16384 B | `4 x 4096 B` | 46.0 us |
| 16384 B | `16 x 1024 B` | 57.6 us |
| 16384 B | `64 x 256 B` | 127.6 us |
| 16384 B | `256 x 64 B` | 443.3 us |

Large transfers (`1 GiB` total, `warmup=3`, `iters=30`) move into a
bandwidth-bound regime. Because individual reruns sometimes show one transient
outlier split, the table below reports the median of `10` repeated reruns for
each split:

| split | per-range size | mean latency | p50 latency | p99 latency | effective throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` range | `1024 MiB` | 96.45 ms | 96.44 ms | 96.57 ms | 10.37 GiB/s |
| `4` ranges | `256 MiB` | 96.68 ms | 96.43 ms | 101.70 ms | 10.34 GiB/s |
| `16` ranges | `64 MiB` | 96.68 ms | 96.43 ms | 99.84 ms | 10.34 GiB/s |
| `64` ranges | `16 MiB` | 96.48 ms | 96.45 ms | 96.58 ms | 10.36 GiB/s |
| `256` ranges | `4 MiB` | 96.53 ms | 96.51 ms | 96.64 ms | 10.36 GiB/s |

### Takeaways

* For small and medium transfers, grouped scatter latency is dominated by a
  fixed base cost plus roughly `1.53 us` per range.
* At fixed total bytes, fewer ranges are consistently faster because they avoid
  per-range control and RDMA request overhead.
* For very large transfers, the median latency is mostly set by sustained RDMA
  bandwidth, not by the exact range split.
* Across repeated `1 GiB` reruns, the mean and p50 latency for `1`, `4`, `16`,
  `64`, and `256` ranges all stay in the same `~96.4 - 96.7 ms` band after
  filtering transient one-off outliers.
* The remaining p99 spread is much smaller than the small/medium-transfer
  range-count effect and is consistent with occasional tail jitter rather than
  a sustained bandwidth difference between splits.

---

## Alternative Benchmarking

Mooncake TE is also supported in **NIXL** as a backend plugin. **NIXL Bench** is recommended for standardized performance testing and comparisons.

Steps:

1. Build `libtransfer_engine.so` with `-DBUILD_SHARED_LIBS=ON` and install.
2. Build and install NIXL & NIXL Bench.
3. Verify `Mooncake.so` is available under `plugins/`.
4. Run benchmark:

```bash
numactl -m0 -N0 ./nixlbench --etcd-endpoints http://node050:2838 --backend UCX \
     [--start_batch_size=XXX] [--num_threads=YYY]
```

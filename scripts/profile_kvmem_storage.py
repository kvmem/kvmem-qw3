#!/usr/bin/env python3
"""Profile the storage path used by KVMem without device-specific assumptions.

The benchmark creates and removes one temporary file on the requested
filesystem.  It prefers O_DIRECT so the result describes the block device
rather than the Linux page cache.  No third-party Python packages are needed.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import mmap
import os
import random
import shutil
import statistics
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def read_int(path: Path) -> int | None:
    value = read_text(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def read_uevent(path: Path) -> dict[str, str]:
    value = read_text(path)
    if value is None:
        return {}
    fields: dict[str, str] = {}
    for line in value.splitlines():
        key, separator, field_value = line.partition("=")
        if separator:
            fields[key] = field_value
    return fields


def parse_cpu_list(value: str | None) -> list[int]:
    if not value:
        return []
    cpus: list[int] = []
    for field in value.split(","):
        bounds = field.split("-", 1)
        start = int(bounds[0])
        end = int(bounds[-1])
        cpus.extend(range(start, end + 1))
    return cpus


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)
    return ordered[max(0, index)]


def block_sysfs_for_path(path: Path) -> tuple[Path | None, Path | None]:
    """Return (partition/block node, whole device) sysfs paths."""
    stat = os.stat(path)
    dev = Path("/sys/dev/block") / f"{os.major(stat.st_dev)}:{os.minor(stat.st_dev)}"
    try:
        node = dev.resolve(strict=True)
    except FileNotFoundError:
        return None, None
    whole = node.parent if (node / "partition").exists() else node
    return node, whole


def pci_device_for_block(whole: Path | None) -> Path | None:
    if whole is None:
        return None
    device = whole / "device"
    try:
        resolved = device.resolve(strict=True)
    except FileNotFoundError:
        return None
    for candidate in [resolved, *resolved.parents]:
        if (candidate / "vendor").exists() and (candidate / "device").exists():
            return candidate
    return None


@dataclass
class StorageCapabilities:
    path: str
    filesystem_device: str | None
    block_device: str | None
    sysfs_block_device: str | None
    model: str | None
    serial: str | None
    firmware_rev: str | None
    rotational: bool | None
    logical_block_bytes: int | None
    physical_block_bytes: int | None
    minimum_io_bytes: int | None
    optimal_io_bytes: int | None
    max_request_kib: int | None
    queue_depth_limit: int | None
    scheduler: str | None
    numa_node: int | None
    local_cpus: str | None
    pci_address: str | None
    current_link_speed: str | None
    maximum_link_speed: str | None
    current_link_width: int | None
    maximum_link_width: int | None
    free_gib: float
    direct_io_available: bool


def probe(path: Path) -> StorageCapabilities:
    node, whole = block_sysfs_for_path(path)
    pci = pci_device_for_block(whole)
    queue = whole / "queue" if whole is not None else None
    device = whole / "device" if whole is not None else None
    node_uevent = read_uevent(node / "uevent") if node is not None else {}
    whole_uevent = read_uevent(whole / "uevent") if whole is not None else {}
    usage = shutil.disk_usage(path)
    direct_io_available = hasattr(os, "O_DIRECT")

    return StorageCapabilities(
        path=str(path),
        filesystem_device=(
            f"/dev/{node_uevent['DEVNAME']}" if "DEVNAME" in node_uevent else None
        ),
        block_device=(
            f"/dev/{whole_uevent['DEVNAME']}"
            if "DEVNAME" in whole_uevent
            else None
        ),
        sysfs_block_device=str(whole) if whole is not None else None,
        model=read_text(device / "model") if device else None,
        serial=read_text(device / "serial") if device else None,
        firmware_rev=read_text(device / "firmware_rev") if device else None,
        rotational=(
            bool(read_int(queue / "rotational"))
            if queue and read_int(queue / "rotational") is not None
            else None
        ),
        logical_block_bytes=read_int(queue / "logical_block_size") if queue else None,
        physical_block_bytes=read_int(queue / "physical_block_size") if queue else None,
        minimum_io_bytes=read_int(queue / "minimum_io_size") if queue else None,
        optimal_io_bytes=read_int(queue / "optimal_io_size") if queue else None,
        max_request_kib=read_int(queue / "max_sectors_kb") if queue else None,
        queue_depth_limit=read_int(queue / "nr_requests") if queue else None,
        scheduler=read_text(queue / "scheduler") if queue else None,
        numa_node=read_int(pci / "numa_node") if pci else None,
        local_cpus=read_text(pci / "local_cpulist") if pci else None,
        pci_address=pci.name if pci else None,
        current_link_speed=read_text(pci / "current_link_speed") if pci else None,
        maximum_link_speed=read_text(pci / "max_link_speed") if pci else None,
        current_link_width=read_int(pci / "current_link_width") if pci else None,
        maximum_link_width=read_int(pci / "max_link_width") if pci else None,
        free_gib=usage.free / GIB,
        direct_io_available=direct_io_available,
    )


class AlignedBuffer:
    def __init__(self, size: int, fill: bool = False) -> None:
        self.mapping = mmap.mmap(-1, size)
        if fill:
            page = bytes((i * 131 + 17) & 0xFF for i in range(4096))
            for offset in range(0, size, len(page)):
                self.mapping[offset : min(size, offset + len(page))] = page[
                    : min(len(page), size - offset)
                ]

    def close(self) -> None:
        self.mapping.close()


def open_benchmark_file(path: Path, direct: bool) -> tuple[int, bool]:
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC
    if direct:
        flags |= os.O_DIRECT
    try:
        return os.open(path, flags, 0o600), direct
    except OSError as exc:
        if direct and exc.errno in (errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTTY):
            return os.open(
                path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC, 0o600
            ), False
        raise


def initialize_file(fd: int, size: int, chunk_bytes: int) -> dict[str, float]:
    os.posix_fallocate(fd, 0, size)
    buf = AlignedBuffer(chunk_bytes, fill=True)
    started = time.monotonic()
    try:
        for offset in range(0, size, chunk_bytes):
            count = min(chunk_bytes, size - offset)
            if count != chunk_bytes:
                raise ValueError("benchmark size must be a multiple of init chunk")
            written = os.pwritev(fd, [buf.mapping], offset)
            if written != count:
                raise OSError(f"short initialization write: {written} != {count}")
        os.fsync(fd)
    finally:
        buf.close()
    elapsed = time.monotonic() - started
    return {"seconds": elapsed, "gib_per_second": size / GIB / elapsed}


def sequential_read(fd: int, size: int, chunk_bytes: int) -> dict[str, float]:
    buf = AlignedBuffer(chunk_bytes)
    started = time.monotonic()
    try:
        for offset in range(0, size, chunk_bytes):
            count = min(chunk_bytes, size - offset)
            if count != chunk_bytes:
                raise ValueError("benchmark size must be a multiple of read chunk")
            read = os.preadv(fd, [buf.mapping], offset)
            if read != count:
                raise OSError(f"short sequential read: {read} != {count}")
    finally:
        buf.close()
    elapsed = time.monotonic() - started
    return {"seconds": elapsed, "gib_per_second": size / GIB / elapsed}


@dataclass
class WorkerResult:
    bytes_read: int
    latencies_ms: list[float]


def random_read(
    fd: int,
    file_bytes: int,
    request_bytes: int,
    queue_depth: int,
    duration_seconds: float,
    local_cpus: list[int],
) -> dict[str, Any]:
    aligned_limit = file_bytes // request_bytes
    if aligned_limit < queue_depth:
        raise ValueError("benchmark file is too small for the requested queue depth")

    barrier = threading.Barrier(queue_depth + 1)
    stop_at = [0.0]
    results: list[WorkerResult | None] = [None] * queue_depth
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        buf = AlignedBuffer(request_bytes)
        latencies: list[float] = []
        total = 0
        rng = random.Random(0x4B564D45 + index)
        try:
            if local_cpus:
                try:
                    os.sched_setaffinity(0, {local_cpus[index % len(local_cpus)]})
                except OSError:
                    pass
            barrier.wait()
            while time.monotonic() < stop_at[0]:
                offset = rng.randrange(aligned_limit) * request_bytes
                started_ns = time.monotonic_ns()
                count = os.preadv(fd, [buf.mapping], offset)
                elapsed_ms = (time.monotonic_ns() - started_ns) / 1e6
                if count != request_bytes:
                    raise OSError(f"short random read: {count} != {request_bytes}")
                total += count
                latencies.append(elapsed_ms)
            results[index] = WorkerResult(total, latencies)
        except BaseException as exc:  # propagate worker failures to main thread
            errors.append(exc)
        finally:
            buf.close()

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(queue_depth)
    ]
    for thread in threads:
        thread.start()
    stop_at[0] = time.monotonic() + duration_seconds
    started = time.monotonic()
    barrier.wait()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - started
    if errors:
        raise errors[0]

    completed = [result for result in results if result is not None]
    total_bytes = sum(result.bytes_read for result in completed)
    latencies = [
        latency for result in completed for latency in result.latencies_ms
    ]
    return {
        "queue_depth": queue_depth,
        "request_kib": request_bytes / KIB,
        "seconds": elapsed,
        "operations": len(latencies),
        "gib_per_second": total_bytes / GIB / elapsed,
        "latency_ms_mean": statistics.fmean(latencies),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_p99": percentile(latencies, 0.99),
    }


def print_human(result: dict[str, Any]) -> None:
    capabilities = result["capabilities"]
    print("Storage capabilities")
    for key, value in capabilities.items():
        print(f"  {key}: {value}")
    if "initialization" not in result:
        return
    init = result["initialization"]
    seq = result["sequential_read"]
    print(
        f"Initialization write: {init['gib_per_second']:.2f} GiB/s "
        f"({init['seconds']:.2f}s)"
    )
    print(
        f"Sequential read:     {seq['gib_per_second']:.2f} GiB/s "
        f"({seq['seconds']:.2f}s)"
    )
    print("KVMem-sized random reads")
    print("  QD  GiB/s  mean-ms  p50-ms  p95-ms  p99-ms")
    for row in result["random_reads"]:
        print(
            f"  {row['queue_depth']:>2}  {row['gib_per_second']:>5.2f}"
            f"  {row['latency_ms_mean']:>7.3f}"
            f"  {row['latency_ms_p50']:>6.3f}"
            f"  {row['latency_ms_p95']:>6.3f}"
            f"  {row['latency_ms_p99']:>6.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="directory on the filesystem to probe and benchmark",
    )
    parser.add_argument(
        "--probe-only", action="store_true", help="report capabilities without I/O"
    )
    parser.add_argument("--size-gib", type=float, default=8.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument(
        "--queue-depths",
        default="1,4,8,16,32",
        help="comma-separated random-read concurrency sweep",
    )
    parser.add_argument(
        "--request-kib",
        type=int,
        default=2176,
        help="request size; 2176 KiB is one Qwen3.6-27B fp16 raw-K+V block",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    path = args.path.resolve()
    if not path.is_dir():
        parser.error(f"--path is not a directory: {path}")
    capabilities = probe(path)
    result: dict[str, Any] = {"capabilities": asdict(capabilities)}
    if args.probe_only:
        print(json.dumps(result, indent=2) if args.json else "")
        if not args.json:
            print_human(result)
        return 0

    alignment = max(
        4096,
        capabilities.logical_block_bytes or 0,
        capabilities.physical_block_bytes or 0,
    )
    request_bytes = args.request_kib * KIB
    if request_bytes % alignment:
        parser.error(
            f"--request-kib must produce a multiple of {alignment} bytes"
        )
    init_chunk = math.lcm(8 * MIB, request_bytes, alignment)
    file_bytes = int(args.size_gib * GIB)
    file_bytes = max(init_chunk, (file_bytes // init_chunk) * init_chunk)
    if file_bytes + GIB > shutil.disk_usage(path).free:
        parser.error("insufficient free space for benchmark plus 1 GiB reserve")
    queue_depths = [int(value) for value in args.queue_depths.split(",")]
    if any(value <= 0 for value in queue_depths):
        parser.error("queue depths must be positive")

    filename = path / f".kvmem-storage-profile-{uuid.uuid4().hex}.tmp"
    fd = -1
    try:
        fd, direct = open_benchmark_file(
            filename, capabilities.direct_io_available
        )
        result["direct_io"] = direct
        result["file_bytes"] = file_bytes
        result["request_bytes"] = request_bytes
        result["initialization"] = initialize_file(fd, file_bytes, init_chunk)
        if not direct and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, file_bytes, os.POSIX_FADV_DONTNEED)
        result["sequential_read"] = sequential_read(fd, file_bytes, init_chunk)
        local_cpus = parse_cpu_list(capabilities.local_cpus)
        result["random_reads"] = [
            random_read(
                fd,
                file_bytes,
                request_bytes,
                queue_depth,
                args.duration,
                local_cpus,
            )
            for queue_depth in queue_depths
        ]
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            filename.unlink()
        except FileNotFoundError:
            pass

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

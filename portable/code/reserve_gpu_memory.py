#!/usr/bin/env python3
"""Reserve CUDA memory on each visible GPU until the process is stopped.

Use this only on GPUs assigned to you or on a machine you administer. On a
scheduled/shared machine, set CUDA_VISIBLE_DEVICES to the GPUs allocated to
your job before launching this script.
"""

import argparse
import os
import signal
import sys
import threading
from typing import Dict, List, Optional


GIB = 1024 ** 3
MIB = 1024 ** 2
STOP_EVENT = threading.Event()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / GIB:.2f} GiB"


def _parse_devices(raw_devices: Optional[str], visible_count: int) -> List[int]:
    if raw_devices is None:
        return list(range(visible_count))

    try:
        devices = [int(item.strip()) for item in raw_devices.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--devices must be comma-separated CUDA indexes, e.g. 0,1") from exc

    if not devices:
        raise ValueError("--devices did not contain any CUDA indexes")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices contains duplicate CUDA indexes")

    invalid = [device for device in devices if device < 0 or device >= visible_count]
    if invalid:
        raise ValueError(
            f"--devices contains indexes outside the visible range 0..{visible_count - 1}: {invalid}"
        )
    return devices


def _handle_stop(signum: int, _frame: object) -> None:
    STOP_EVENT.set()
    print(f"\nReceived signal {signum}; releasing reserved GPU memory.", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reserve a fixed amount of CUDA memory on every visible GPU until stopped."
    )
    parser.add_argument(
        "--gib-per-gpu",
        type=_positive_float,
        default=40.0,
        help="GiB of tensor storage to hold on each selected GPU (default: 40).",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated local CUDA indexes to reserve after CUDA_VISIBLE_DEVICES filtering "
            "(default: all visible devices)."
        ),
    )
    parser.add_argument(
        "--chunk-mib",
        type=_positive_int,
        default=512,
        help="Allocation chunk size in MiB (default: 512).",
    )
    parser.add_argument(
        "--headroom-mib",
        type=_nonnegative_int,
        default=256,
        help="Require this much free memory beyond the requested reservation (default: 256).",
    )
    parser.add_argument(
        "--status-seconds",
        type=_positive_int,
        default=300,
        help="Seconds between keep-alive status messages while holding memory (default: 300).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected GPUs and free-memory checks without holding reservation tensors.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch is required. Run this script inside the project's CUDA environment.", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("CUDA is not available in the current PyTorch environment.", file=sys.stderr)
        return 2

    visible_count = torch.cuda.device_count()
    if visible_count == 0:
        print("No CUDA devices are visible to this process.", file=sys.stderr)
        return 2

    try:
        devices = _parse_devices(args.devices, visible_count)
    except ValueError as exc:
        print(f"Invalid device selection: {exc}", file=sys.stderr)
        return 2

    requested_bytes = int(args.gib_per_gpu * GIB)
    chunk_bytes = args.chunk_mib * MIB
    headroom_bytes = args.headroom_mib * MIB
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset; all CUDA-visible devices>")

    print(f"PID: {os.getpid()}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {visible_env}", flush=True)
    print(
        f"Target: {_format_gib(requested_bytes)} per GPU on local CUDA indexes {devices}",
        flush=True,
    )
    print("CUDA context overhead may make process usage slightly higher than this target.", flush=True)

    checks = []
    for device in devices:
        torch.cuda.set_device(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        name = torch.cuda.get_device_name(device)
        checks.append((device, name, free_bytes, total_bytes))
        print(
            f"cuda:{device} {name}: free {_format_gib(free_bytes)} / total {_format_gib(total_bytes)}",
            flush=True,
        )

    insufficient = [
        (device, free_bytes)
        for device, _name, free_bytes, _total_bytes in checks
        if free_bytes < requested_bytes + headroom_bytes
    ]
    if insufficient:
        for device, free_bytes in insufficient:
            print(
                f"cuda:{device} has only {_format_gib(free_bytes)} free; "
                f"need {_format_gib(requested_bytes + headroom_bytes)} including headroom.",
                file=sys.stderr,
            )
        return 1

    if args.dry_run:
        print("Dry run complete; no reservation tensors were held.", flush=True)
        return 0

    allocations: Dict[int, List[object]] = {}
    try:
        for device in devices:
            torch.cuda.set_device(device)
            remaining_bytes = requested_bytes
            allocations[device] = []
            while remaining_bytes > 0:
                allocation_bytes = min(chunk_bytes, remaining_bytes)
                allocations[device].append(
                    torch.empty(allocation_bytes, dtype=torch.uint8, device=f"cuda:{device}")
                )
                remaining_bytes -= allocation_bytes
            torch.cuda.synchronize(device)
            print(
                f"cuda:{device}: holding {_format_gib(requested_bytes)} "
                f"({len(allocations[device])} allocations).",
                flush=True,
            )
    except RuntimeError as exc:
        print(f"CUDA allocation failed: {exc}", file=sys.stderr)
        allocations.clear()
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        return 1

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    print("Reservation active. Press Ctrl-C or send SIGTERM to release it.", flush=True)

    try:
        while not STOP_EVENT.wait(args.status_seconds):
            if not STOP_EVENT.is_set():
                print(
                    f"Holding {_format_gib(requested_bytes)} on each of {len(devices)} GPU(s).",
                    flush=True,
                )
    finally:
        allocations.clear()
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        print("GPU memory released.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

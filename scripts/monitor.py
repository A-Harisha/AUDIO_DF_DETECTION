"""Monitor a running training job: GPU state, elapsed time, log tail, checkpoint arrival.

    uv run python scripts/monitor.py --checkpoint experiments/phase2_condaware/model.pt
    uv run python scripts/monitor.py --checkpoint <ckpt> --log <task-output-file> --interval 15

Stops when the checkpoint file appears (training saved). Note: a job started with block-buffered
output only flushes its log at exit; run training with `python -u` (or PYTHONUNBUFFERED=1) to
stream loss lines live. GPU state and checkpoint arrival update regardless of buffering.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


def gpu_line() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        util, used, total = (x.strip() for x in out.split(","))
        return f"GPU {util:>3}%  mem {int(used)/1000:.1f}/{int(total)/1000:.0f} GB"
    except (OSError, subprocess.SubprocessError, ValueError):
        return "GPU n/a"


def tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:]) if lines else "(log empty / buffered)"
    except FileNotFoundError:
        return "(no log file yet)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path that appears when training finishes")
    ap.add_argument("--log", default=None, help="optional log file to tail")
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--tail-lines", type=int, default=6)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    log_path = Path(args.log) if args.log else None
    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        done = ckpt.exists()
        size = f"{ckpt.stat().st_size/1e6:.0f} MB" if done else "pending"
        print(f"\n[{stamp}] elapsed {elapsed//60}m{elapsed%60:02d}s | {gpu_line()} | checkpoint: {size}")
        if log_path:
            print(tail(log_path, args.tail_lines))
        if done:
            print(f"\n== checkpoint saved: {ckpt} ==")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

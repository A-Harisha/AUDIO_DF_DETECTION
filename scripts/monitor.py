"""Live terminal view of the running jobs. Read-only: it never touches a job.

    uv run python scripts/monitor.py            # refresh each 5 s; Ctrl+C stops the view only

Claim served: none (observability). It shows, for the newest logs in experiments/logs_partial:
the last progress lines, the per-epoch metrics of the newest run, and the GPU load.
Standard library only, so it starts fast and cannot hang on the pandas import.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent / "experiments"
KEEP = ("clips done", "epoch", "scored", "wrote", "Error", "shape |")


def tail(path: Path, n: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ln[:160] for ln in lines if any(k in ln for k in KEEP)][-n:]


def main(period: float = 5.0) -> None:
    while True:
        out = [time.strftime("%H:%M:%S") + "  CAESD monitor (Ctrl+C stops the view, not the jobs)", ""]
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
        out += [f"GPU (load, memory, temperature): {gpu}", ""]
        logs = sorted((ROOT / "logs_partial").glob("*.log"), key=os.path.getmtime, reverse=True)[:3]
        for log in logs:
            age = (time.time() - log.stat().st_mtime) / 60
            out += [f"--- {log.name}  (last write {age:.0f} min ago)", *tail(log, 4), ""]
        runs = sorted((ROOT / "runs_partial").glob("*/metrics.jsonl"), key=os.path.getmtime, reverse=True)[:1]
        for run in runs:
            out.append(f"--- {run.parent.name}: epoch | train BCE | select EER (lower is better)")
            for line in run.read_text(encoding="utf-8").splitlines()[-8:]:
                m = json.loads(line)
                out.append(f"    {m['epoch']:>3} | {m['train/bce']:.4f} | {100 * m['select']:.2f}%")
        os.system("cls")
        sys.stdout.write("\n".join(out) + "\n")
        time.sleep(period)


if __name__ == "__main__":
    main()
